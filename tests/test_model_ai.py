"""
MODEL_AI correctness tests
==========================
The critical one is test_no_lookahead: weights computed at date t must be
byte-identical whether the panel handed to the strategy ends at t or extends
years into the future. If that ever fails, every backtest number in this repo
is fiction.

Run:  python -m pytest tests/test_model_ai.py -v
      python tests/test_model_ai.py          (no pytest needed)
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import model_ai  # noqa: E402


def _synthetic_panel(n_days: int = 900, n_tickers: int = 12, seed: int = 7):
    """Deterministic random-walk panel — no network, no yfinance flakiness."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n_days)
    tickers = [f"T{i:02d}" for i in range(n_tickers)]

    # Drifty geometric random walks so the trend gate has something to pass.
    drift = rng.normal(0.0004, 0.0003, n_tickers)
    vol = rng.uniform(0.01, 0.03, n_tickers)
    steps = rng.normal(drift, vol, size=(n_days, n_tickers))
    close = pd.DataFrame(100 * np.exp(np.cumsum(steps, axis=0)),
                         index=dates, columns=tickers)

    high = close * (1 + rng.uniform(0, 0.01, size=close.shape))
    low = close * (1 - rng.uniform(0, 0.01, size=close.shape))
    volume = pd.DataFrame(rng.uniform(2e6, 9e6, size=close.shape),
                          index=dates, columns=tickers)
    bench = close.mean(axis=1)

    return model_ai.PriceData(close=close, high=high, low=low, volume=volume,
                              benchmark=bench, benchmark_name="TEST")


def _test_config():
    cfg = model_ai.market_config("US")
    cfg.update({
        "universe": [f"T{i:02d}" for i in range(12)],
        "min_dollar_volume": 0,   # synthetic volumes shouldn't gate the test
        "n_positions": 5,
    })
    return cfg


def test_no_lookahead():
    """Weights at t must not depend on data after t."""
    full = _synthetic_panel()
    cfg = _test_config()

    for offset in (0, 30, 90, 200):
        as_of = full.close.index[-1 - offset]
        truncated = model_ai._slice(full, as_of)

        from_full = model_ai.generate_target_weights(
            as_of, "US", cfg, full, verbose=False)
        from_trunc = model_ai.generate_target_weights(
            as_of, "US", cfg, truncated, verbose=False)

        assert from_full == from_trunc, (
            f"LOOK-AHEAD at {as_of.date()}: full-panel weights {from_full} "
            f"!= truncated-panel weights {from_trunc}")
    print("PASS test_no_lookahead (4 dates)")


def test_weights_are_valid():
    """Sum <= 1.0,每 name within cap, no negatives (long-only)."""
    data = _synthetic_panel()
    cfg = _test_config()
    w = model_ai.generate_target_weights(
        data.close.index[-1], "US", cfg, data, verbose=False)

    total = sum(w.values())
    assert total <= 1.0 + 1e-9, f"gross exposure {total} > 1.0 (implies leverage)"
    assert all(v > 0 for v in w.values()), f"non-positive weight in {w}"
    assert all(v <= cfg["max_weight"] + 1e-9 for v in w.values()), \
        f"weight exceeds max_weight={cfg['max_weight']}: {w}"
    assert len(w) <= cfg["n_positions"], f"{len(w)} positions > n_positions"
    print(f"PASS test_weights_are_valid (exposure {total:.1%}, {len(w)} names)")


def test_risk_off_goes_to_cash():
    """A benchmark in a downtrend must flatten the book."""
    data = _synthetic_panel()
    cfg = _test_config()

    # Force RISK_OFF: a monotonically decaying benchmark is below its 200-SMA.
    n = len(data.close)
    falling = pd.Series(np.linspace(200, 80, n), index=data.close.index)
    bear = model_ai.PriceData(close=data.close, high=data.high, low=data.low,
                              volume=data.volume, benchmark=falling,
                              benchmark_name="BEAR")

    regime = model_ai.detect_regime(bear, cfg)
    assert regime["regime"] == "RISK_OFF", f"expected RISK_OFF, got {regime}"

    cfg_off = {**cfg, "regime_factors": {**cfg["regime_factors"], "RISK_OFF": 0.0}}
    w = model_ai.generate_target_weights(
        bear.close.index[-1], "US", cfg_off, bear, verbose=False)
    assert w == {}, f"RISK_OFF with factor 0 must go to cash, got {w}"
    print("PASS test_risk_off_goes_to_cash")


def test_inverse_vol_favours_calm_names():
    """
    The lower-vol name must carry the larger weight.

    Uses a non-binding cap: with max_weight=0.25 a two-name sleeve is capped at
    0.25 each and the ordering information is destroyed by construction (see
    test_cap_below_full_allocation_leaves_cash), which would test the cap
    rather than the sizing.
    """
    cfg = {**_test_config(), "max_weight": 1.0}
    selected = pd.DataFrame(
        {"vol_20": [0.10, 0.40], "score": [1.0, 2.0]}, index=["CALM", "WILD"])
    w = model_ai.inverse_vol_weights(selected, cfg)

    assert w["CALM"] > w["WILD"], f"inverse-vol sizing inverted: {w.to_dict()}"
    # 1/0.10 : 1/0.40 = 4:1 -> 80% / 20%
    assert abs(w["CALM"] - 0.8) < 1e-9, f"expected 80% in CALM, got {w['CALM']}"
    assert abs(w.sum() - 1.0) < 1e-9, f"weights must normalize to 1, got {w.sum()}"
    print("PASS test_inverse_vol_favours_calm_names")


def test_cap_below_full_allocation_leaves_cash():
    """
    Documented degenerate case: when n_names * max_weight < 1 the sleeve CANNOT
    be fully invested, and that is the intended outcome — the per-name cap is a
    concentration limit, so a two-name book holds 25% + 25% and 50% cash rather
    than forcing half the portfolio into one ticker.
    """
    cfg = {**_test_config(), "max_weight": 0.25}
    selected = pd.DataFrame({"vol_20": [0.10, 0.40]}, index=["CALM", "WILD"])
    w = model_ai.inverse_vol_weights(selected, cfg)

    assert w.max() <= 0.25 + 1e-9, f"cap breached: {w.to_dict()}"
    assert abs(w.sum() - 0.5) < 1e-9, \
        f"two names capped at 25% should total 50%, got {w.sum()}"
    print("PASS test_cap_below_full_allocation_leaves_cash (50% cash by design)")


def test_max_weight_cap_renormalizes():
    """Capping must not leave the sleeve under-allocated."""
    cfg = {**_test_config(), "max_weight": 0.25}
    # One ultra-calm name would otherwise take ~85% of the book.
    selected = pd.DataFrame(
        {"vol_20": [0.01, 0.30, 0.30, 0.30, 0.30]},
        index=list("ABCDE"))
    w = model_ai.inverse_vol_weights(selected, cfg)
    assert w.max() <= 0.25 + 1e-9, f"cap breached: {w.to_dict()}"
    assert abs(w.sum() - 1.0) < 1e-6, f"weights must still sum to 1, got {w.sum()}"
    print("PASS test_max_weight_cap_renormalizes")


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
            except AssertionError as e:
                failures += 1
                print(f"FAIL {name}: {e}")
            except Exception as e:
                failures += 1
                print(f"ERROR {name}: {type(e).__name__}: {e}")
    print("\n" + ("all tests passed" if not failures else f"{failures} FAILURE(S)"))
    sys.exit(1 if failures else 0)
