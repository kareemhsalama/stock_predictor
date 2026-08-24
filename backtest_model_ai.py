"""
MODEL_AI walk-forward backtest
==============================
Validates the strategy out-of-sample before it is trusted with a ledger. The
spec's own caveat applies and is worth repeating: in-sample Sharpe is nearly
meaningless, so this runs strictly walk-forward — at every step the decision
uses only data the strategy could actually have seen.

LOOK-AHEAD DISCIPLINE (the bug that has bitten this repo before):
  * weights are decided on bar t from data <= t (enforced by model_ai._slice),
  * they earn the return from t -> t+1, never t-1 -> t,
  * stops are evaluated on bar t's close and act on t+1.

Models the parts of Layer 4 that move the metrics: hard stop, ATR trailing
stop (bar-level approximation off the daily high/low, not tick-level -
see run_backtest()), portfolio drawdown breaker, min-hold, weekly cadence,
and per-name transaction costs. Realized live results will still differ
from a genuinely intraday-honest trail, but backtest max-drawdown is no
longer an unknown-size upper bound on what live risk controls produce.

Usage:
    python backtest_model_ai.py            # both markets
    python backtest_model_ai.py US 8y
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import model_ai
from paper_trader import MARKETS

REBALANCE_EVERY = 5   # trading days ~ weekly

# Append-only research trials ledger. Every backtest run that informs a real
# decision in the Model AI redesign - kept or discarded - gets logged here,
# not just the ones that "worked". This is what makes a real deflated Sharpe
# / PBO calculation possible at the end instead of an unfalsifiable trust-me
# about how many variants were actually tried. Unit tests on synthetic data
# must not call log_trial() - it's for real research trials only.
TRIALS_LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "research", "trials.jsonl")

# Nominal book size used only to scale dollar-denominated cost terms (the
# per-order commission floor, and participation-vs-ADV impact) realistically.
# Matches the actual paper account sizes (data/model_ai_ledger.json,
# egx_model_ai.py's INITIAL_BUDGET) rather than being an arbitrary constant.
NOMINAL_BUDGET = {"US": 100_000.0, "EGX": 1_000_000.0}

# Phase 1 out-of-sample reserve (inclusive). Nothing in research/redesign work
# may compute metrics using dates on or after this cutoff - run_backtest()
# enforces it by truncating the data before this date unless the caller
# explicitly passes allow_holdout=True, which is reserved for the one-time
# final verification pass comparing baseline vs. redesign.
HOLDOUT_START = "2025-01-01"


def _metrics(returns: pd.Series, exposure: pd.Series,
             turnover: float, n_rebalances: int) -> dict:
    """Sharpe / Sortino / Calmar and friends from a daily return series."""
    r = returns.dropna()
    if len(r) < 20:
        return {}

    equity = (1 + r).cumprod()
    years = len(r) / model_ai.TRADING_DAYS
    total_return = float(equity.iloc[-1] - 1)
    cagr = float(equity.iloc[-1] ** (1 / years) - 1) if years > 0 else 0.0

    ann_vol = float(r.std() * np.sqrt(model_ai.TRADING_DAYS))
    sharpe = float(r.mean() / r.std() * np.sqrt(model_ai.TRADING_DAYS)) if r.std() > 0 else 0.0

    downside = r[r < 0]
    sortino = (float(r.mean() / downside.std() * np.sqrt(model_ai.TRADING_DAYS))
               if len(downside) > 1 and downside.std() > 0 else 0.0)

    dd = equity / equity.cummax() - 1
    max_dd = float(dd.min())
    calmar = float(cagr / abs(max_dd)) if max_dd < 0 else 0.0

    return {
        "total_return": total_return,
        "cagr": cagr,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_dd": max_dd,
        "calmar": calmar,
        "hit_rate": float((r > 0).mean()),
        "avg_exposure": float(exposure.mean()),
        "turnover_annual": float(turnover / years) if years > 0 else 0.0,
        "n_rebalances": n_rebalances,
        "days": len(r),
    }


def run_backtest(market: str, period: str = "8y", config: dict | None = None,
                 verbose: bool = True, allow_holdout: bool = False,
                 data: model_ai.PriceData | None = None,
                 weight_fn=None, disable_portfolio_risk: bool = False) -> dict:
    """
    ``weight_fn(as_of, market, cfg, data, verbose=False) -> {ticker: weight}``
    defaults to ``model_ai.generate_target_weights``. Baselines/ablations
    (Tier 0 item 2 of the Model AI redesign) pass an alternative here to reuse
    this exact loop — same cadence, same per-name costs, same holdout guard —
    so a baseline's number is comparable to the real strategy's, not an
    apples-to-oranges reimplementation.

    ``disable_portfolio_risk=True`` skips the drawdown breaker and hard-stop
    blocks entirely (pure periodic-reweight-and-hold, subject only to costs).
    Use this for "no signal" / "always-on" baselines, where the point is to
    isolate what the SIGNAL contributes — folding Model AI's own risk overlay
    into a null baseline would conflate "the breaker helps" with "the
    selection helps."
    """
    weight_fn = weight_fn or model_ai.generate_target_weights
    cfg = config or model_ai.market_config(market)
    if data is None:
        data = model_ai.load_price_data(market, cfg, period=period)

    if not allow_holdout:
        cutoff = pd.Timestamp(HOLDOUT_START) - pd.Timedelta(days=1)
        data = model_ai._slice(data, cutoff)

    close, high, low = data.close, data.high, data.low
    dates = close.index
    rets = close.pct_change()
    # Trailing 20-session dollar volume, for the impact/participation terms in
    # CostModel.slippage_bps() below. Rolling, so bar i only ever sees <= i.
    adv = (close * data.volume).rolling(20).mean()
    # Daily-high/low approximation of the live ATR trailing stop
    # (live_trader_ai.check_stops). Not tick-level - it can't be, from daily
    # bars - but bar-level low-vs-trail is directionally honest and was
    # simply absent before, making backtest max-drawdown an unknown-size
    # UPPER bound on what live risk controls actually produce.
    atr_panel = pd.DataFrame({t: model_ai.atr(high[t], low[t], close[t])
                              for t in close.columns})
    costs = MARKETS[market].costs
    budget = NOMINAL_BUDGET[market]

    warmup = max(cfg["mom_long_lookback"] + cfg["mom_long_skip"],
                 cfg["regime_sma_slow"]) + 5
    if len(dates) <= warmup + 40:
        raise RuntimeError(f"{market}: only {len(dates)} bars, need > {warmup + 40}")

    weights = pd.Series(dtype=float)     # current book
    entry_px: dict[str, float] = {}      # for the hard stop
    held_since: dict[str, int] = {}      # for min-hold
    peak_price: dict[str, float] = {}    # for the ATR trailing stop
    peak_equity, equity, breaker_on = 1.0, 1.0, False

    daily_returns, exposures, turnovers = [], [], []
    n_rebalances = 0
    total_turnover = 0.0

    for i in range(warmup, len(dates) - 1):
        today, tomorrow = dates[i], dates[i + 1]

        # ---- decisions made on bar i, from data <= bar i ----
        is_rebalance = (i - warmup) % REBALANCE_EVERY == 0
        new_weights = weights.copy()

        # Portfolio drawdown breaker (Layer 4): flatten and stay out until the
        # regime is RISK_ON again.
        if not disable_portfolio_risk:
            dd = equity / peak_equity - 1
            if dd < -cfg["dd_breaker"]:
                breaker_on = True
            if breaker_on:
                regime = model_ai.detect_regime(model_ai._slice(data, today), cfg)
                if regime["regime"] == "RISK_ON":
                    breaker_on = False
                else:
                    new_weights = pd.Series(dtype=float)

        if not breaker_on:
            if not disable_portfolio_risk:
                # Hard stop, checked daily against the entry price.
                for t in list(new_weights.index):
                    px = close[t].iloc[i]
                    ep = entry_px.get(t)
                    if ep and np.isfinite(px) and (px / ep - 1) <= cfg["hard_stop"]:
                        new_weights = new_weights.drop(t)
                        entry_px.pop(t, None)
                        held_since.pop(t, None)
                        peak_price.pop(t, None)
                        continue

                    # ATR trailing stop, off the high-water mark since entry.
                    if t in high.columns:
                        h = high[t].iloc[i]
                        if np.isfinite(h):
                            peak_price[t] = max(peak_price.get(t, h), h)
                        a = atr_panel[t].iloc[i] if t in atr_panel.columns else float("nan")
                        lo = low[t].iloc[i] if t in low.columns else float("nan")
                        if t in peak_price and np.isfinite(a) and np.isfinite(lo):
                            trail = peak_price[t] - cfg["atr_stop_mult"] * a
                            if lo < trail:
                                new_weights = new_weights.drop(t)
                                entry_px.pop(t, None)
                                held_since.pop(t, None)
                                peak_price.pop(t, None)

            if is_rebalance:
                targets = weight_fn(today, market, cfg, data, verbose=False)
                n_rebalances += 1
                target_s = pd.Series(targets, dtype=float)

                # Min-hold: don't drop a name that hasn't aged enough yet.
                for t in list(new_weights.index):
                    age = i - held_since.get(t, i)
                    if t not in target_s.index and age < cfg["min_hold_days"]:
                        target_s[t] = new_weights[t]

                for t in target_s.index:
                    if t not in weights.index:
                        entry_px[t] = float(close[t].iloc[i])
                        held_since[t] = i
                        peak_price[t] = float(close[t].iloc[i])
                for t in list(entry_px):
                    if t not in target_s.index:
                        entry_px.pop(t, None)
                        held_since.pop(t, None)
                        peak_price.pop(t, None)
                new_weights = target_s

        # ---- cost of moving from `weights` to `new_weights` ----
        # Per-name, not a flat portfolio-level bps rate: commission (with its
        # floor), spread, and ADV-scaled impact all depend on how big a slice
        # of THIS name's own liquidity the trade is, and that varies a lot
        # across the book.
        all_names = weights.index.union(new_weights.index)
        turnover = float((new_weights.reindex(all_names).fillna(0)
                          - weights.reindex(all_names).fillna(0)).abs().sum())
        book_value = equity * budget
        cost_dollars = 0.0
        for t in all_names:
            delta_w = float(new_weights.get(t, 0.0) - weights.get(t, 0.0))
            if delta_w == 0.0:
                continue
            notional = abs(delta_w) * book_value
            adv_t = float(adv[t].iloc[i]) if t in adv.columns else None
            cost_dollars += costs.fees(notional)
            cost_dollars += notional * costs.slippage_bps(notional, adv_t) / 1e4
        cost = cost_dollars / book_value if book_value > 0 else 0.0
        total_turnover += turnover
        turnovers.append(turnover)
        weights = new_weights[new_weights > 0] if len(new_weights) else new_weights

        # ---- earn bar i -> i+1. THIS is the no-look-ahead step. ----
        if len(weights):
            step = rets.loc[tomorrow, list(weights.index)].fillna(0)
            gross_ret = float((step * weights).sum())
        else:
            step, gross_ret = None, 0.0
        port_ret = gross_ret - cost

        equity *= (1 + port_ret)
        peak_equity = max(peak_equity, equity)
        daily_returns.append(port_ret)
        exposures.append(float(weights.sum()) if len(weights) else 0.0)

        # Drift weights with returns. Weights are fractions OF EQUITY, so the
        # book has to be renormalized by portfolio growth — otherwise exposure
        # ratchets up on winning days and the turnover figure double-counts
        # drift that never actually required a trade.
        if len(weights):
            weights = (weights * (1 + step)) / (1 + gross_ret)

    idx = dates[warmup + 1: warmup + 1 + len(daily_returns)]
    returns = pd.Series(daily_returns, index=idx)
    exposure = pd.Series(exposures, index=idx)

    stats = _metrics(returns, exposure, total_turnover, n_rebalances)

    # Benchmark: buy and hold, same window.
    bench = data.benchmark.reindex(dates).ffill()
    bench_ret = bench.pct_change().loc[idx]
    bench_stats = _metrics(bench_ret, pd.Series(1.0, index=idx), 0.0, 0)

    if verbose:
        _report(market, data.benchmark_name, idx, stats, bench_stats)

    return {"market": market, "stats": stats, "benchmark": bench_stats,
            "returns": returns, "exposure": exposure}


# ============================================================
# TIER 0 BASELINES — is the signal actually adding anything, or is this just
# "being invested in a curated, liquid, momentum-tilted list"? Each of these
# runs through run_backtest()'s exact same loop (cadence, per-name costs,
# holdout guard) via weight_fn/config overrides, so every number below is
# directly comparable to the real strategy's baseline metrics.
# ============================================================
def equal_weight_universe(as_of, market, cfg, data, verbose=False):
    """
    No signal at all: equal-weight the entire liquidity-filtered candidate
    set (the same pool select_names() ranks from — not the raw ticker list,
    which would unfairly include names the real strategy could never hold).
    Isolates curation/survivorship effects from any actual selection skill.
    """
    sliced = model_ai._slice(data, as_of)
    feat = model_ai.compute_features(sliced, cfg)
    pool = model_ai.liquid_universe(feat, cfg)
    if pool.empty:
        return {}
    n = len(pool)
    return {t: 1.0 / n for t in pool.index}


def equal_weight_selected(as_of, market, cfg, data, verbose=False):
    """
    Ablation: real selection (momentum ranking + liquidity + trend gate),
    but equal-weighted instead of inverse-vol/regime-scaled, and combined
    with disable_portfolio_risk=True this is "always-on" exposure — prices
    the sizing + regime layers' combined marginal contribution over the raw
    ranking.
    """
    sliced = model_ai._slice(data, as_of)
    feat = model_ai.compute_features(sliced, cfg)
    selected = model_ai.select_names(feat, cfg)
    if selected.empty:
        return {}
    n = len(selected)
    return {t: 1.0 / n for t in selected.index}


def mean_reversion_config(cfg: dict) -> dict:
    """Sign-flip the momentum terms, keep everything else (sizing, regime,
    costs) identical. Falsifiability check: if reversal 'beats' buy-and-hold
    by a similar margin to the real strategy, the edge is likely a generic
    curated-list artifact of the sample period, not momentum-specific."""
    weights = {k: (-v if k.startswith("mom_") else v)
              for k, v in cfg["score_weights"].items()}
    return {**cfg, "score_weights": weights}


def no_regime_gate_config(cfg: dict) -> dict:
    """Force the regime multiplier to 1.0 in every state, isolating the
    regime gate's own marginal contribution to Sharpe/Sortino/Calmar."""
    return {**cfg, "regime_factors": {k: 1.0 for k in cfg["regime_factors"]}}


def random_portfolio_bootstrap(market: str, period: str = "8y",
                               config: dict | None = None,
                               data: model_ai.PriceData | None = None,
                               allow_holdout: bool = False,
                               n_draws: int = 100, seed: int = 0) -> dict:
    """
    Empirical null distribution for the "is the ranking adding anything"
    question: n_draws random equal-weight portfolios of n_positions names,
    drawn from the same liquidity-filtered pool select_names() ranks from,
    rebalanced on the same weekly cadence — answers what Sharpe a portfolio
    gets in this universe with ZERO selection skill.

    Deliberately lighter-weight than run_backtest(): gross of transaction
    costs and with no stop/breaker overlay (a random draw has nothing for a
    hard-stop to protect against that isn't already reflected in the draw
    itself), computed at rebalance-period granularity rather than a
    day-by-day loop, so hundreds of draws run in seconds rather than minutes.
    Returns the actual strategy's percentile within this null distribution
    when `reference_sharpe` is passed via the caller comparing results.
    """
    cfg = config or model_ai.market_config(market)
    if data is None:
        data = model_ai.load_price_data(market, cfg, period=period)
    if not allow_holdout:
        cutoff = pd.Timestamp(HOLDOUT_START) - pd.Timedelta(days=1)
        data = model_ai._slice(data, cutoff)

    close = data.close
    dates = close.index
    warmup = max(cfg["mom_long_lookback"] + cfg["mom_long_skip"],
                 cfg["regime_sma_slow"]) + 5
    rebalance_idx = list(range(warmup, len(dates) - 1, REBALANCE_EVERY))
    n_pos = cfg["n_positions"]

    # Eligible pool and forward period return per rebalance date, computed
    # once (not per draw) — this is the only part that walks the panel.
    pools, period_rets = [], []
    for k, i in enumerate(rebalance_idx):
        today = dates[i]
        end_i = rebalance_idx[k + 1] if k + 1 < len(rebalance_idx) else len(dates) - 1
        sliced = model_ai._slice(data, today)
        feat = model_ai.compute_features(sliced, cfg)
        pool = model_ai.liquid_universe(feat, cfg)
        pools.append(list(pool.index))
        fwd = (close.iloc[end_i] / close.iloc[i] - 1).reindex(pool.index)
        period_rets.append(fwd)

    periods_per_year = model_ai.TRADING_DAYS / REBALANCE_EVERY
    rng = np.random.default_rng(seed)
    sharpes, sortinos = [], []
    for _ in range(n_draws):
        draw_rets = []
        for pool, fwd in zip(pools, period_rets):
            if len(pool) < n_pos:
                draw_rets.append(0.0)
                continue
            chosen = rng.choice(pool, size=n_pos, replace=False)
            draw_rets.append(float(fwd.loc[chosen].fillna(0).mean()))
        r = pd.Series(draw_rets)
        if r.std() > 0:
            sharpes.append(float(r.mean() / r.std() * np.sqrt(periods_per_year)))
        downside = r[r < 0]
        if len(downside) > 1 and downside.std() > 0:
            sortinos.append(float(r.mean() / downside.std() * np.sqrt(periods_per_year)))

    return {"market": market, "n_draws": n_draws,
            "sharpe_mean": float(np.mean(sharpes)) if sharpes else None,
            "sharpe_std": float(np.std(sharpes)) if sharpes else None,
            "sharpes": sharpes, "sortinos": sortinos}


def percentile_of(value: float, distribution: list[float]) -> float:
    """What fraction of `distribution` falls at or below `value`."""
    if not distribution:
        return float("nan")
    return float(np.mean(np.array(distribution) <= value))


def run_baselines(market: str, period: str = "8y", config: dict | None = None,
                  data: model_ai.PriceData | None = None,
                  n_random_draws: int = 100, seed: int = 0,
                  verbose: bool = True) -> dict:
    """Run the real strategy alongside all Tier 0 baselines/ablations on the
    identical (holdout-excluded) panel, and report a comparison table."""
    cfg = config or model_ai.market_config(market)
    if data is None:
        data = model_ai.load_price_data(market, cfg, period=period)

    real = run_backtest(market, config=cfg, data=data, verbose=False)
    no_signal = run_backtest(market, config=cfg, data=data, verbose=False,
                             weight_fn=equal_weight_universe,
                             disable_portfolio_risk=True)
    selection_only = run_backtest(market, config=cfg, data=data, verbose=False,
                                  weight_fn=equal_weight_selected,
                                  disable_portfolio_risk=True)
    no_regime = run_backtest(market, config=no_regime_gate_config(cfg),
                             data=data, verbose=False)
    reversal = run_backtest(market, config=mean_reversion_config(cfg),
                            data=data, verbose=False)
    null_dist = random_portfolio_bootstrap(market, config=cfg, data=data,
                                           n_draws=n_random_draws, seed=seed)

    rows = [
        ("Model AI (real)", real["stats"]),
        ("Buy & hold", real["benchmark"]),
        ("No signal (equal-wt universe)", no_signal["stats"]),
        ("Selection-only (equal-wt, always-on)", selection_only["stats"]),
        ("No regime gate", no_regime["stats"]),
        ("Sign-flipped (mean-reversion)", reversal["stats"]),
    ]
    if verbose:
        print(f"\n{'=' * 80}\n  BASELINES - {market}\n{'=' * 80}")
        print(f"  {'variant':<38} {'sharpe':>8} {'sortino':>8} {'calmar':>8} {'max_dd':>8}")
        for label, s in rows:
            if not s:
                print(f"  {label:<38} (no data)")
                continue
            print(f"  {label:<38} {s.get('sharpe', 0):>8.2f} {s.get('sortino', 0):>8.2f} "
                  f"{s.get('calmar', 0):>8.2f} {s.get('max_dd', 0):>8.1%}")
        if null_dist["sharpe_mean"] is not None:
            pct = percentile_of(real["stats"]["sharpe"], null_dist["sharpes"])
            print(f"\n  Random {n_random_draws}-portfolio null: Sharpe "
                  f"{null_dist['sharpe_mean']:.2f} +/- {null_dist['sharpe_std']:.2f} "
                  f"(gross of costs) - real strategy at the {pct:.0%} percentile")

    return {"real": real, "no_signal": no_signal, "selection_only": selection_only,
            "no_regime": no_regime, "reversal": reversal, "null_distribution": null_dist}


def _git_commit() -> str:
    """Short SHA, with a +dirty suffix if the tree has uncommitted changes.
    'unknown' rather than raising - a missing git binary must never break a
    research run."""
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=here,
                             capture_output=True, text=True, timeout=5).stdout.strip()
        if not sha:
            return "unknown"
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=here,
                               capture_output=True, text=True, timeout=5).stdout.strip()
        return f"{sha}+dirty" if dirty else sha
    except Exception:
        return "unknown"


def log_trial(result: dict, config_diff: dict, layer_touched: str, rationale: str) -> dict:
    """
    Append one row to the trials ledger (research/trials.jsonl).

    `config_diff` should be just the keys overridden vs. the CONFIG baseline
    (not the full config), so the ledger stays readable. `layer_touched` is a
    short label (e.g. "regime", "sizing", "cost-model") and `rationale` is a
    one-line reason this trial was run, for the eventual DSR/PBO pass to
    cluster correlated variants rather than counting each as an independent
    trial.
    """
    stats = result["stats"]
    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "market": result["market"],
        "config_diff": config_diff,
        "git_commit": _git_commit(),
        "sharpe": stats.get("sharpe"),
        "sortino": stats.get("sortino"),
        "calmar": stats.get("calmar"),
        "cagr": stats.get("cagr"),
        "max_dd": stats.get("max_dd"),
        "n_rebalances": stats.get("n_rebalances"),
        "layer_touched": layer_touched,
        "rationale": rationale,
    }
    os.makedirs(os.path.dirname(TRIALS_LOG), exist_ok=True)
    with open(TRIALS_LOG, "a") as f:
        f.write(json.dumps(row) + "\n")
    return row


def _report(market, bench_name, idx, s, b):
    print(f"\n{'=' * 66}")
    print(f"  MODEL_AI walk-forward - {market}   "
          f"{idx[0].date()} -> {idx[-1].date()}  ({s['days']} bars)")
    print(f"{'=' * 66}")
    rows = [
        ("Total return", "total_return", "{:+.1%}"),
        ("CAGR", "cagr", "{:+.1%}"),
        ("Annualized vol", "ann_vol", "{:.1%}"),
        ("Sharpe", "sharpe", "{:.2f}"),
        ("Sortino", "sortino", "{:.2f}"),
        ("Max drawdown", "max_dd", "{:.1%}"),
        ("Calmar", "calmar", "{:.2f}"),
        ("Hit rate (daily)", "hit_rate", "{:.1%}"),
        ("Avg exposure", "avg_exposure", "{:.1%}"),
        ("Turnover / yr", "turnover_annual", "{:.1f}x"),
    ]
    print(f"  {'metric':<20} {'MODEL_AI':>14} {'buy & hold ' + bench_name:>22}")
    print(f"  {'-' * 20} {'-' * 14} {'-' * 22}")
    for label, key, fmt in rows:
        mine = fmt.format(s[key]) if key in s else "-"
        theirs = fmt.format(b[key]) if key in b else "-"
        print(f"  {label:<20} {mine:>14} {theirs:>22}")
    print(f"\n  Rebalances: {s['n_rebalances']}   "
          f"(real fees/spread/impact costs included, per-name)")


def main():
    markets = [sys.argv[1]] if len(sys.argv) > 1 else ["US", "EGX"]
    period = sys.argv[2] if len(sys.argv) > 2 else "8y"
    out = {}
    for m in markets:
        try:
            out[m] = run_backtest(m, period=period)
        except Exception as e:
            print(f"\n[{m}] backtest failed: {type(e).__name__}: {e}")
    return out


if __name__ == "__main__":
    main()
