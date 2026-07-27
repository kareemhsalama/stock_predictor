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

Models the parts of Layer 4 that move the metrics: hard stop, portfolio
drawdown breaker, min-hold, weekly cadence, and per-side transaction costs.
The ATR trailing stop is NOT modeled here (it needs intraday highs to be
honest); the live runners implement it, so realized results will differ.

Usage:
    python backtest_model_ai.py            # both markets
    python backtest_model_ai.py US 8y
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

import model_ai

COST_BPS = 5.0        # per side, in basis points of traded notional
REBALANCE_EVERY = 5   # trading days ~ weekly


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
                 verbose: bool = True) -> dict:
    cfg = config or model_ai.market_config(market)
    data = model_ai.load_price_data(market, cfg, period=period)

    close = data.close
    dates = close.index
    rets = close.pct_change()

    warmup = max(cfg["mom_long_lookback"] + cfg["mom_long_skip"],
                 cfg["regime_sma_slow"]) + 5
    if len(dates) <= warmup + 40:
        raise RuntimeError(f"{market}: only {len(dates)} bars, need > {warmup + 40}")

    weights = pd.Series(dtype=float)     # current book
    entry_px: dict[str, float] = {}      # for the hard stop
    held_since: dict[str, int] = {}      # for min-hold
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
            # Hard stop, checked daily against the entry price.
            for t in list(new_weights.index):
                px = close[t].iloc[i]
                ep = entry_px.get(t)
                if ep and np.isfinite(px) and (px / ep - 1) <= cfg["hard_stop"]:
                    new_weights = new_weights.drop(t)
                    entry_px.pop(t, None)
                    held_since.pop(t, None)

            if is_rebalance:
                targets = model_ai.generate_target_weights(
                    today, market, cfg, data, verbose=False)
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
                for t in list(entry_px):
                    if t not in target_s.index:
                        entry_px.pop(t, None)
                        held_since.pop(t, None)
                new_weights = target_s

        # ---- cost of moving from `weights` to `new_weights` ----
        all_names = weights.index.union(new_weights.index)
        turnover = float((new_weights.reindex(all_names).fillna(0)
                          - weights.reindex(all_names).fillna(0)).abs().sum())
        cost = turnover * COST_BPS / 10_000
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
          f"(costs {COST_BPS:.0f}bps/side included)")


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
