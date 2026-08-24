"""
MODEL_AI - Regime-aware cross-sectional momentum with volatility targeting
==========================================================================
Design goal: win on RISK-ADJUSTED return (Sharpe / Sortino / Calmar), not raw
return. The edge is allocation, not prediction:

  Layer 1  regime      trade only when the benchmark says an edge persists
  Layer 2  signal      cross-sectional momentum composite, trend-gated
  Layer 3  sizing      inverse-vol weights scaled to a portfolio vol target
  Layer 5  meta-label  (optional) RF filter on P(trade is profitable)

This module is PURE STRATEGY. It computes target weights and nothing else;
execution and ledger-keeping live in ``live_trader_ai.py`` (US / Alpaca) and
``egx_model_ai.py`` (EGX / PaperTrader). The single entry point is::

    generate_target_weights(as_of_date, market, config) -> {ticker: weight}

Weights sum to <= 1.0; the remainder is cash.

NO LOOK-AHEAD: every feature is computed from data through ``as_of_date``
inclusive. The caller applies the resulting weights to returns from the NEXT
bar onward. ``_slice()`` is the single choke point that enforces this — all
feature code reads from its output, never from the raw frame.
"""

from __future__ import annotations

import os
import pickle
from dataclasses import dataclass

import numpy as np
import pandas as pd
import yfinance as yf

# ============================================================
# LAYER 0 - UNIVERSE & CONFIG
# ============================================================

# US: liquid large caps across all 11 GICS sectors. Hardcoded rather than
# scraped from Wikipedia because a CI job that silently loses its universe to
# a layout change is worse than one that trades a fixed, well-understood list.
# The 20-day dollar-volume filter below prunes this at runtime.
US_UNIVERSE = [
    # tech / comms
    "AAPL", "MSFT", "NVDA", "GOOGL", "META", "AVGO", "AMD", "CRM", "ADBE",
    "ORCL", "CSCO", "ACN", "TXN", "QCOM", "INTU", "NFLX", "DIS", "CMCSA",
    # consumer
    "AMZN", "TSLA", "HD", "MCD", "NKE", "SBUX", "LOW", "TJX", "BKNG",
    "WMT", "COST", "PG", "KO", "PEP", "PM", "MDLZ", "CL",
    # financials
    "BRK-B", "JPM", "BAC", "WFC", "GS", "MS", "SPGI", "BLK", "AXP", "C",
    # healthcare
    "UNH", "JNJ", "LLY", "ABBV", "MRK", "PFE", "TMO", "ABT", "DHR", "AMGN",
    # industrials / energy / utilities / materials / real estate
    "CAT", "HON", "UNP", "GE", "BA", "RTX", "LMT", "DE",
    "XOM", "CVX", "COP", "SLB", "NEE", "DUK", "SO", "LIN", "SHW", "AMT", "PLD",
]

# EGX: EGX30 names that actually carry usable yfinance history (from egy.py).
# ORAS.CA is deliberately absent-in-practice: it reports zero volume on every
# bar, so the liquidity filter drops it. Kept in the list so the filter's
# decision is visible in logs rather than hidden in a hand-edited universe.
EGX_UNIVERSE = [
    "COMI.CA", "TMGH.CA", "HRHO.CA", "EAST.CA", "FWRY.CA",
    "ORAS.CA", "PHDC.CA", "EFID.CA", "ABUK.CA", "SWDY.CA",
]

# Redesign project, Tier 1 sector-concentration guard: with only 5 of 10
# names held at once, an uncapped momentum book can end up 3+ names deep in
# one sector (a real risk with no correlation/covariance data feed available
# to catch it directly - see the redesign plan's Tier 2 item 8). Standard
# EGX30 sector groupings, static and hand-classified (there is no sector
# data feed in this repo to source them from automatically).
EGX_SECTOR = {
    "COMI.CA": "Financials", "HRHO.CA": "Financials", "FWRY.CA": "Financials",
    "TMGH.CA": "RealEstate", "PHDC.CA": "RealEstate",
    "EAST.CA": "Consumer", "EFID.CA": "Consumer",
    "SWDY.CA": "Industrials", "ORAS.CA": "Industrials",
    "ABUK.CA": "Materials",
}

CONFIG = {
    "universe_size":       50,     # cap after the liquidity ranking
    "n_positions":         8,      # 5 for EGX (see MARKET_DEFAULTS)
    "mom_long_lookback":   252,
    "mom_long_skip":       21,
    "mom_short_lookback":  63,
    "trend_sma":           100,
    "regime_sma_fast":     50,
    "regime_sma_slow":     200,
    "vol_window":          20,
    "high_vol_pct":        0.80,
    "score_weights":       {"mom_12_1": 0.5, "mom_3": 0.3, "vol_20": -0.2},
    "rsi_entry_max":       40,
    "rsi_tilt":            0.15,   # see _composite_score()
    "sigma_target":        0.12,   # 0.10 for EGX
    "max_weight":          0.25,
    "atr_stop_mult":       3.0,
    "hard_stop":           -0.08,
    "dd_breaker":          0.15,
    "rebalance":           "weekly",
    "min_hold_days":       3,
    "use_meta_label":      False,
    "meta_prob_threshold": 0.55,
    "min_dollar_volume":   50_000_000,   # 20d average, market currency
    "regime_factors":      {"RISK_ON": 1.0, "NEUTRAL": 0.5, "RISK_OFF": 0.1},
    "regime_confirm_weeks": 1,   # 1 = off (raw regime, unchanged behavior)
    "sector_map":          None,   # {ticker: sector}; None = no sector cap
    "max_sector_weight":   1.0,    # 1.0 = off (no sector concentration cap)
}

MARKET_DEFAULTS = {
    "US": {
        "universe": US_UNIVERSE,
        "benchmark": "SPY",
        "n_positions": 8,
        "sigma_target": 0.12,
        "min_dollar_volume": 50_000_000,
        "allow_short": False,
        # Redesign project, Tier 1 finding (research/trials.jsonl): the base
        # 3.0x multiplier was tight enough to whipsaw daily volatility and
        # stop out of names before their momentum played out. A sweep from
        # 3.0->8.0 (in-sample, both markets excluded from the holdout) shows
        # a smooth, non-isolated-spike improvement through 5.0x, which
        # strictly dominates the 3.0x baseline: Sharpe 0.74->0.95, Sortino
        # 0.89->1.19, Calmar 0.42->0.83, AND max drawdown improves
        # -19.2%->-13.9% (not a risk-for-return tradeoff). Scoped to US only
        # - EGX was not swept at this value and keeps the base 3.0x default.
        "atr_stop_mult": 5.0,
    },
    "EGX": {
        "universe": EGX_UNIVERSE,
        # ^CASE30 returns a single row from yfinance (verified 2026-07-27), so
        # the EGX regime benchmark falls back to an equal-weight composite of
        # the universe itself. See _benchmark_series().
        "benchmark": "^CASE30",
        "n_positions": 5,
        "sigma_target": 0.10,
        "min_dollar_volume": 1_000_000,   # EGP
        "allow_short": False,             # thin liquidity, no reliable borrow
    },
}

TRADING_DAYS = 252
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache")


def market_config(market: str, overrides: dict | None = None) -> dict:
    """Merge global CONFIG <- per-market defaults <- explicit overrides."""
    if market not in MARKET_DEFAULTS:
        raise ValueError(f"Unknown market {market!r}. Known: {list(MARKET_DEFAULTS)}")
    cfg = {**CONFIG, **MARKET_DEFAULTS[market], "market": market}
    if overrides:
        cfg.update(overrides)
    return cfg


# ============================================================
# DATA
# ============================================================
@dataclass
class PriceData:
    """OHLCV panel + benchmark, indexed by date, columns = tickers."""

    close: pd.DataFrame
    high: pd.DataFrame
    low: pd.DataFrame
    volume: pd.DataFrame
    benchmark: pd.Series
    benchmark_name: str

    def tickers(self) -> list[str]:
        return list(self.close.columns)


def _download(tickers: list[str], period: str) -> pd.DataFrame:
    return yf.download(
        tickers, period=period, auto_adjust=True, progress=False,
        group_by="column", threads=True,
    )


def load_price_data(market: str, config: dict | None = None, period: str = "5y",
                    use_cache: bool = True) -> PriceData:
    """
    Fetch the OHLCV panel + benchmark for a market.

    Cached to ``.cache/`` keyed by (market, period, today) so a backtest sweep
    doesn't re-pull 250+ days on every iteration. CI runs start cold, which is
    fine — one pull per job.
    """
    cfg = config or market_config(market)
    universe = list(cfg["universe"])
    stamp = pd.Timestamp.now("UTC").strftime("%Y-%m-%d")
    cache_path = os.path.join(CACHE_DIR, f"{market}_{period}_{stamp}.pkl")

    if use_cache and os.path.exists(cache_path):
        try:
            with open(cache_path, "rb") as f:
                blob = pickle.load(f)
            return PriceData(**blob)
        except Exception as e:
            # A stale or unreadable cache must never be fatal — just refetch.
            print(f"[cache] ignoring {os.path.basename(cache_path)} "
                  f"({type(e).__name__}) - refetching")

    raw = _download(universe, period)
    if raw is None or raw.empty:
        raise RuntimeError(f"No price data returned for {market} universe")

    def field(name: str) -> pd.DataFrame:
        f = raw[name]
        if isinstance(f, pd.Series):          # single-ticker edge case
            f = f.to_frame(universe[0])
        return f.reindex(columns=universe).dropna(axis=1, how="all")

    close, high, low, volume = (field(f) for f in ("Close", "High", "Low", "Volume"))

    bench_name = cfg["benchmark"]
    bench = _benchmark_series(bench_name, close, period)

    data = PriceData(close=close, high=high, low=low, volume=volume,
                     benchmark=bench[0], benchmark_name=bench[1])

    if use_cache:
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(cache_path, "wb") as f:
            # Pickle plain frames, not the dataclass instance: a PriceData
            # pickled from `python model_ai.py` carries a __main__ qualname and
            # fails to load when the module is later imported as `model_ai`.
            pickle.dump(data.__dict__, f)
    return data


def _benchmark_series(name: str, close: pd.DataFrame,
                      period: str) -> tuple[pd.Series, str]:
    """
    Benchmark close series for regime detection.

    Falls back to an equal-weight composite of the universe when the named
    index is unusable. This is not cosmetic: ^CASE30 returns a single row from
    yfinance, and Layer 1 is the whole thesis of this model — silently running
    EGX at regime_factor=1.0 forever would delete the risk management. An
    equal-weight composite of EGX30 names tracks the index closely enough to
    drive a 200-SMA trend gate.
    """
    try:
        raw = _download([name], period)
        series = raw["Close"]
        if isinstance(series, pd.DataFrame):
            series = series.iloc[:, 0]
        series = series.dropna()
        # A real index needs enough history for the 200-SMA + a 252d vol
        # percentile. One row (the ^CASE30 case) is not a benchmark.
        if len(series) >= 260:
            return series, name
        print(f"[regime] benchmark {name!r} returned {len(series)} usable rows "
              f"- falling back to equal-weight universe composite")
    except Exception as e:
        print(f"[regime] benchmark {name!r} failed ({type(e).__name__}: {e}) "
              f"- falling back to equal-weight universe composite")

    # Equal-weight composite: mean of per-name cumulative returns, rebased to 100.
    rets = close.pct_change()
    composite = (1 + rets.mean(axis=1).fillna(0)).cumprod() * 100
    return composite, "EW-COMPOSITE"


def _slice(data: PriceData, as_of: pd.Timestamp) -> PriceData:
    """
    Truncate every series to <= as_of. THE look-ahead guardrail: all feature
    code below reads from this, so a future bar cannot leak into a decision.
    """
    as_of = pd.Timestamp(as_of)
    if data.close.index.tz is not None and as_of.tz is None:
        as_of = as_of.tz_localize(data.close.index.tz)
    return PriceData(
        close=data.close.loc[:as_of],
        high=data.high.loc[:as_of],
        low=data.low.loc[:as_of],
        volume=data.volume.loc[:as_of],
        benchmark=data.benchmark.loc[:as_of],
        benchmark_name=data.benchmark_name,
    )


# ============================================================
# INDICATORS
# ============================================================
def rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Rolling-mean RSI — matches the definition already used repo-wide."""
    delta = prices.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - (100 / (1 + rs))).fillna(50.0)


def atr(high: pd.Series, low: pd.Series, close: pd.Series,
        period: int = 14) -> pd.Series:
    """Average True Range — drives the trailing stop in the runners."""
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def realized_vol(returns: pd.Series | pd.DataFrame, window: int) -> pd.Series:
    """Annualized realized volatility over a trailing window."""
    return returns.rolling(window).std() * np.sqrt(TRADING_DAYS)


def _zscore(s: pd.Series) -> pd.Series:
    """Cross-sectional z-score. Zero-variance -> all zeros (no spurious tilt)."""
    sd = s.std()
    if not np.isfinite(sd) or sd == 0:
        return pd.Series(0.0, index=s.index)
    return (s - s.mean()) / sd


# ============================================================
# LAYER 1 - REGIME DETECTION
# ============================================================
def detect_regime(data: PriceData, cfg: dict) -> dict:
    """
    Classify the benchmark's regime. Returns the factor that scales total
    gross exposure, plus the diagnostics that produced it (logged by the
    runners so a flat week is explainable after the fact).
    """
    bench = data.benchmark.dropna()
    slow, fast = cfg["regime_sma_slow"], cfg["regime_sma_fast"]

    if len(bench) < slow + 1:
        return {"regime": "RISK_OFF", "factor": cfg["regime_factors"]["RISK_OFF"],
                "reason": f"insufficient benchmark history ({len(bench)} < {slow + 1})",
                "trend_up": False, "vol_pct": float("nan"),
                "benchmark": data.benchmark_name}

    sma_slow = bench.rolling(slow).mean().iloc[-1]
    sma_fast = bench.rolling(fast).mean().iloc[-1]
    price = bench.iloc[-1]
    trend_up = bool(price > sma_slow and sma_fast > sma_slow)

    rets = bench.pct_change()
    vol_20 = realized_vol(rets, cfg["vol_window"])
    current_vol = vol_20.iloc[-1]
    trailing = vol_20.dropna().tail(TRADING_DAYS)
    vol_pct = float((trailing <= current_vol).mean()) if len(trailing) > 20 else 0.5
    high_vol = vol_pct > cfg["high_vol_pct"]

    if not trend_up:
        regime = "RISK_OFF"
        reason = f"benchmark below {slow}-SMA (or fast<slow)"
    elif high_vol:
        regime = "NEUTRAL"
        reason = f"uptrend but vol in {vol_pct:.0%} percentile"
    else:
        regime = "RISK_ON"
        reason = f"uptrend, vol in {vol_pct:.0%} percentile"

    return {"regime": regime, "factor": cfg["regime_factors"][regime],
            "reason": reason, "trend_up": trend_up, "vol_pct": vol_pct,
            "annualized_vol": float(current_vol) if np.isfinite(current_vol) else None,
            "benchmark": data.benchmark_name}


# ~1 trading week; matches backtest_model_ai.REBALANCE_EVERY. Used only to
# space the trailing samples detect_regime_confirmed() walks back through -
# it does not assume the caller's actual rebalance calendar.
REGIME_CONFIRM_STEP_DAYS = 5


def detect_regime_confirmed(data: PriceData, cfg: dict) -> dict:
    """
    Hysteresis-filtered regime, to stop the exposure factor flipping (and
    paying transaction costs) on a single noisy week.

    A regime change only takes effect once the RAW weekly classification
    (plain detect_regime, unchanged above) has held for
    ``cfg["regime_confirm_weeks"]`` consecutive ~weekly observations, sampled
    every REGIME_CONFIRM_STEP_DAYS trading days walking backward from the
    last bar in ``data``. Still a pure function of ``data`` alone - every
    sample point is <= data's last bar, so this adds zero new look-ahead
    surface beyond what detect_regime() already has via _slice().

    ``regime_confirm_weeks <= 1`` (the default) returns detect_regime(data,
    cfg) unchanged, so this is fully backward compatible.
    """
    n = int(cfg.get("regime_confirm_weeks", 1))
    raw_now = detect_regime(data, cfg)
    if n <= 1:
        return raw_now

    dates = data.close.index
    if len(dates) == 0:
        return raw_now
    pos = len(dates) - 1

    buffer = n + 8   # a few extra samples so a real prior regime can be found
    positions = [pos - k * REGIME_CONFIRM_STEP_DAYS for k in range(buffer)]
    positions = [p for p in positions if p >= 0]
    raw_series = [detect_regime(_slice(data, dates[p]), cfg)["regime"] for p in positions]

    run_len = 1
    for j in range(1, len(raw_series)):
        if raw_series[j] == raw_series[0]:
            run_len += 1
        else:
            break

    if run_len >= n:
        confirmed = raw_series[0]
    else:
        # Not enough confirmation for the new raw regime yet - hold whatever
        # was confirmed just before this run started (or the raw regime
        # itself if history runs out, a safe fallback since it's still no
        # more forward-looking than plain detect_regime already is).
        confirmed = raw_series[run_len] if run_len < len(raw_series) else raw_series[0]

    return {**raw_now, "regime": confirmed, "factor": cfg["regime_factors"][confirmed],
           "raw_regime": raw_series[0], "confirm_run_len": run_len}


# ============================================================
# LAYER 2 - SIGNAL GENERATION
# ============================================================
def compute_features(data: PriceData, cfg: dict) -> pd.DataFrame:
    """Per-ticker feature table as of the last bar in ``data``."""
    close, volume = data.close, data.volume
    long_lb, skip = cfg["mom_long_lookback"], cfg["mom_long_skip"]
    short_lb, trend_sma = cfg["mom_short_lookback"], cfg["trend_sma"]
    vol_w = cfg["vol_window"]

    min_rows = max(long_lb + 1, trend_sma + 1)
    rows = []
    for t in close.columns:
        px = close[t].dropna()
        if len(px) < min_rows:
            continue

        # 12-1 momentum: t-252 -> t-21. Skipping the last month sidesteps the
        # well-documented short-term reversal that contaminates raw 12m momentum.
        mom_long = px.iloc[-1 - skip] / px.iloc[-1 - long_lb] - 1
        mom_short = px.iloc[-1] / px.iloc[-1 - short_lb] - 1
        sma = px.rolling(trend_sma).mean().iloc[-1]
        vol = realized_vol(px.pct_change(), vol_w).iloc[-1]

        vol_series = volume[t].reindex(px.index).fillna(0)
        dollar_vol = float((px * vol_series).tail(vol_w).mean())

        if not np.isfinite(vol) or vol <= 0:
            continue

        rows.append({
            "ticker": t,
            "close": float(px.iloc[-1]),
            "mom_12_1": float(mom_long),
            "mom_3": float(mom_short),
            "trend_ok": bool(px.iloc[-1] > sma),
            "vol_20": float(vol),
            "rsi_14": float(rsi(px).iloc[-1]),
            "dollar_volume": dollar_vol,
        })

    return pd.DataFrame(rows).set_index("ticker") if rows else pd.DataFrame()


def _composite_score(feat: pd.DataFrame, cfg: dict) -> pd.Series:
    """
    Cross-sectional composite: 0.5*z(mom_12_1) + 0.3*z(mom_3) - 0.2*z(vol_20).

    The RSI overlay is a SOFT tilt, not a gate. A hard `rsi_14 < 40` filter on
    a weekly rebalance frequently leaves the book empty or near-empty — strong
    momentum names rarely sit oversold — which would hand the vol-target layer
    nothing to size. A bonus of `rsi_tilt` z-units preserves the intent (prefer
    buying pullbacks within uptrends) without starving the sleeve.
    """
    w = cfg["score_weights"]
    score = (w["mom_12_1"] * _zscore(feat["mom_12_1"])
             + w["mom_3"] * _zscore(feat["mom_3"])
             + w["vol_20"] * _zscore(feat["vol_20"]))
    if cfg.get("rsi_tilt"):
        score = score + cfg["rsi_tilt"] * (feat["rsi_14"] < cfg["rsi_entry_max"])
    return score


def liquid_universe(feat: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Liquidity filter -> universe_size cap. The candidate set select_names()
    ranks from, exposed separately (not just inlined there) so a "no signal"
    baseline can compare against exactly this same eligible set rather than
    the raw ticker list — otherwise a no-skill baseline would be unfairly
    handicapped by illiquid names the real strategy would never hold either.
    """
    if feat.empty:
        return feat
    liquid = feat[feat["dollar_volume"] >= cfg["min_dollar_volume"]]
    if liquid.empty:
        return liquid
    return liquid.nlargest(min(cfg["universe_size"], len(liquid)), "dollar_volume")


def select_names(feat: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Liquidity filter -> trend gate -> composite score -> top N."""
    liquid = liquid_universe(feat, cfg)
    if liquid.empty:
        return liquid

    eligible = liquid[liquid["trend_ok"]].copy()
    if eligible.empty:
        return eligible

    eligible["score"] = _composite_score(eligible, cfg)
    return eligible.nlargest(min(cfg["n_positions"], len(eligible)), "score")


# ============================================================
# LAYER 3 - POSITION SIZING
# ============================================================
def _sector_capped(w: pd.Series, sector_map: dict[str, str], cap: float,
                   max_iter: int = 20) -> pd.Series:
    """
    Cap every sector's total weight at `cap`, water-filling the freed weight
    into sectors that still have headroom.

    A naive "redistribute freed weight into everyone else" pass can push a
    previously-fine sector over cap in turn, which then needs its own
    capping pass, whose redistribution can re-breach the FIRST sector -
    an oscillation that never converges. This avoids that by giving each
    under-cap sector strictly no more than its own headroom (cap minus its
    current total) per pass, so no pass can ever create a new breach.

    If every selected name shares one over-cap sector, there's no headroom
    anywhere and the freed weight is left as cash - same degenerate-case
    philosophy as the per-name cap (test_cap_below_full_allocation_leaves_cash).
    """
    w = w.copy()
    sectors = pd.Series({t: sector_map.get(t, "UNKNOWN") for t in w.index})

    for _ in range(max_iter):
        totals = w.groupby(sectors).sum()
        over = totals[totals > cap + 1e-12]
        if over.empty:
            break

        freed = 0.0
        for sec, total in over.items():
            names = sectors[sectors == sec].index
            freed += total - cap
            w[names] = w[names] * (cap / total)

        totals_after = w.groupby(sectors).sum()
        headroom = (cap - totals_after).clip(lower=0)
        headroom = headroom[~headroom.index.isin(over.index)]
        total_headroom = float(headroom.sum())
        if total_headroom <= 0 or freed <= 0:
            break   # nowhere left to put it - leaves the rest as cash

        give = min(freed, total_headroom)
        for sec, room in headroom.items():
            if room <= 0:
                continue
            share = give * (room / total_headroom)
            names = sectors[sectors == sec].index
            sec_total = w[names].sum()
            if sec_total > 0:
                w[names] += share * w[names] / sec_total
            else:
                w[names] += share / len(names)

    return w


def inverse_vol_weights(selected: pd.DataFrame, cfg: dict) -> pd.Series:
    """Risk parity within the sleeve, capped (per-name and, optionally,
    per-sector) and renormalized."""
    raw = 1.0 / selected["vol_20"]
    w = raw / raw.sum()

    cap = cfg["max_weight"]
    sector_map = cfg.get("sector_map")
    sector_cap = cfg.get("max_sector_weight", 1.0)

    # Iterate: capping frees weight that renormalization pushes back onto other
    # names, which can push THEM (or their sector) over a cap. Converges in a
    # few passes - both caps are applied every pass so neither one's fix can
    # silently break the other.
    for _ in range(10):
        before = w.copy()

        over = w > cap
        if over.any():
            excess = (w[over] - cap).sum()
            w[over] = cap
            under = ~over
            if under.any() and w[under].sum() > 0:
                w[under] += excess * w[under] / w[under].sum()

        if sector_map and sector_cap < 1.0:
            w = _sector_capped(w, sector_map, sector_cap)

        if (w - before).abs().max() < 1e-9:
            break

    return w.clip(upper=cap)


def volatility_scalar(weights: pd.Series, data: PriceData, cfg: dict) -> tuple[float, float]:
    """
    Scale gross exposure so trailing portfolio vol hits sigma_target.
    Never levers above 1.0 (paper accounts, no margin).
    """
    rets = data.close[list(weights.index)].pct_change().tail(cfg["vol_window"] * 2)
    port_rets = (rets * weights).sum(axis=1).dropna()
    if len(port_rets) < cfg["vol_window"]:
        return 1.0, float("nan")

    sigma = float(port_rets.tail(cfg["vol_window"]).std() * np.sqrt(TRADING_DAYS))
    if not np.isfinite(sigma) or sigma <= 0:
        return 1.0, sigma
    return float(min(1.0, cfg["sigma_target"] / sigma)), sigma


# ============================================================
# ENTRY POINT
# ============================================================
def generate_target_weights(as_of_date, market: str, config: dict | None = None,
                            data: PriceData | None = None,
                            verbose: bool = True) -> dict[str, float]:
    """
    Target portfolio weights as of ``as_of_date``.

    Returns {ticker: weight} summing to <= 1.0; the remainder is cash. An empty
    dict means "go to cash" — a valid, and in RISK_OFF the correct, answer.

    Pure function of data through ``as_of_date``. Pass ``data`` to reuse one
    download across a backtest sweep; omit it and the panel is fetched fresh.
    """
    cfg = config or market_config(market)
    if data is None:
        data = load_price_data(market, cfg)
    d = _slice(data, as_of_date)

    if len(d.close) == 0:
        return {}

    regime = detect_regime_confirmed(d, cfg)
    if verbose:
        print(f"[regime] {regime['regime']} (factor {regime['factor']:.2f}) "
              f"via {regime['benchmark']} — {regime['reason']}")

    if regime["factor"] <= 0:
        return {}

    feat = compute_features(d, cfg)
    selected = select_names(feat, cfg)
    if selected.empty:
        if verbose:
            print("[select] no eligible names (trend gate / liquidity) — cash")
        return {}

    weights = inverse_vol_weights(selected, cfg)

    conviction = 1.0
    if cfg.get("use_meta_label"):
        from model_ai_meta import conviction_for   # local import: sklearn optional
        weights, conviction = conviction_for(weights, selected, d, cfg)
        if weights.empty:
            if verbose:
                print("[meta] all candidates below P(win) threshold - cash")
            return {}

    vol_scalar, sigma = volatility_scalar(weights, d, cfg)
    exposure = vol_scalar * regime["factor"] * conviction
    targets = (weights * exposure).round(4)
    targets = targets[targets > 0.001]

    if verbose:
        print(f"[size] realized vol {sigma:.1%} -> vol_scalar {vol_scalar:.2f} "
              f"x regime {regime['factor']:.2f} x conviction {conviction:.2f} "
              f"= exposure {exposure:.1%} across {len(targets)} names")
        for t, w in targets.sort_values(ascending=False).items():
            f = selected.loc[t]
            print(f"       {t:9} w={w:6.2%} score={f['score']:+.2f} "
                  f"vol={f['vol_20']:.1%} rsi={f['rsi_14']:.0f}")

    return targets.to_dict()


def describe(as_of_date, market: str, config: dict | None = None,
             data: PriceData | None = None) -> dict:
    """Diagnostics bundle (regime + selection + weights) for logging/debugging."""
    cfg = config or market_config(market)
    if data is None:
        data = load_price_data(market, cfg)
    d = _slice(data, as_of_date)
    return {
        "as_of": str(pd.Timestamp(as_of_date).date()),
        "market": market,
        "regime": detect_regime(d, cfg),
        "weights": generate_target_weights(as_of_date, market, cfg, data, verbose=False),
    }


if __name__ == "__main__":
    import sys
    mkt = sys.argv[1] if len(sys.argv) > 1 else "US"
    cfg = market_config(mkt)
    pdata = load_price_data(mkt, cfg)
    latest = pdata.close.index[-1]
    print(f"=== MODEL_AI {mkt} target weights as of {latest.date()} ===")
    tw = generate_target_weights(latest, mkt, cfg, pdata)
    print(f"\ntotal exposure {sum(tw.values()):.1%} | cash {1 - sum(tw.values()):.1%}")
