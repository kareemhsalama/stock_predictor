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


def test_meta_label_no_lookahead():
    """
    model_ai_meta.build_training_set()'s own no-lookahead claim (its docstring:
    "a label can never peek past the decision date"), turned into an automated
    check rather than a one-time manual trace. Two different as_of truncations
    of the SAME panel must produce byte-identical training rows for the sample
    dates they share - if a later cutoff ever changed an EARLIER sample's
    features or label, that sample would have been peeking at data that, at
    its own decision date, hadn't happened yet.

    test_no_lookahead above never exercises this module at all (use_meta_label
    defaults to False in _test_config), so without this test the meta-label
    path has zero automated lookahead coverage.
    """
    import model_ai_meta

    full = _synthetic_panel(n_days=700, n_tickers=12, seed=7)
    cfg = _test_config()

    earlier = model_ai._slice(full, full.close.index[-300])
    later = model_ai._slice(full, full.close.index[-250])   # 50 bars further

    from_earlier = model_ai_meta.build_training_set(earlier, cfg)
    from_later = model_ai_meta.build_training_set(later, cfg)

    assert len(from_earlier) > 0, "synthetic panel produced no training rows at all"
    assert len(from_later) >= len(from_earlier), (
        "a strictly longer panel produced FEWER training rows - "
        f"{len(from_later)} < {len(from_earlier)}"
    )

    shared = from_later.iloc[:len(from_earlier)].reset_index(drop=True)
    mismatch = shared.compare(from_earlier.reset_index(drop=True))
    assert mismatch.empty, (
        f"LOOK-AHEAD in meta-label training set: {len(mismatch)} row(s) "
        f"changed when the panel was extended 50 bars further:\n{mismatch}"
    )
    print(f"PASS test_meta_label_no_lookahead ({len(from_earlier)} shared rows)")


def test_regime_confirm_weeks_default_is_a_no_op():
    """regime_confirm_weeks=1 (the default) must return byte-identical output
    to plain detect_regime - no behavior change for anyone who hasn't opted
    in to hysteresis."""
    data = _synthetic_panel()
    cfg = _test_config()
    assert cfg.get("regime_confirm_weeks", 1) == 1

    plain = model_ai.detect_regime(data, cfg)
    confirmed = model_ai.detect_regime_confirmed(data, cfg)
    assert plain == confirmed, f"default (n=1) changed output: {plain} vs {confirmed}"
    print("PASS test_regime_confirm_weeks_default_is_a_no_op")


def test_regime_confirm_weeks_delays_a_flip():
    """
    With hysteresis on, a regime flip that only just happened (raw regime
    differs from a few weeks ago) must NOT be reported yet - the prior
    confirmed regime should still be returned until the new one has held
    for regime_confirm_weeks consecutive ~weekly samples.
    """
    n = len(_synthetic_panel().close)
    dates = pd.bdate_range("2020-01-01", periods=n)
    tickers = [f"T{i:02d}" for i in range(12)]

    # A benchmark that trends up for a long stretch, then drops sharply RIGHT
    # at the end - just 1 weekly sample's worth of "new" regime, not enough
    # to confirm under a 3-week hysteresis window.
    bench_vals = np.linspace(100, 200, n)
    bench_vals[-3:] = bench_vals[-4] * 0.7   # sudden break at the very end
    bench = pd.Series(bench_vals, index=dates)

    rng = np.random.default_rng(3)
    steps = rng.normal(0.0004, 0.01, size=(n, 12))
    close = pd.DataFrame(100 * np.exp(np.cumsum(steps, axis=0)), index=dates, columns=tickers)
    high = close * 1.01
    low = close * 0.99
    volume = pd.DataFrame(rng.uniform(2e6, 9e6, size=close.shape), index=dates, columns=tickers)
    data = model_ai.PriceData(close=close, high=high, low=low, volume=volume,
                              benchmark=bench, benchmark_name="TEST")

    cfg_off = _test_config()
    cfg_on = {**cfg_off, "regime_confirm_weeks": 3}

    raw = model_ai.detect_regime(data, cfg_off)
    hysteresis = model_ai.detect_regime_confirmed(data, cfg_on)

    assert raw["regime"] == "RISK_OFF", f"test setup didn't produce a fresh RISK_OFF break: {raw}"
    assert hysteresis["regime"] != "RISK_OFF", (
        f"a single-sample break was NOT held back by 3-week hysteresis: {hysteresis}"
    )
    assert hysteresis["raw_regime"] == "RISK_OFF"
    print(f"PASS test_regime_confirm_weeks_delays_a_flip (raw={raw['regime']}, "
         f"confirmed={hysteresis['regime']})")


def test_regime_confirm_weeks_preserves_no_lookahead():
    """Same invariant as test_no_lookahead, but through the hysteresis path -
    a longer trailing history must not change what's reported as of `as_of`."""
    full = _synthetic_panel()
    cfg = {**_test_config(), "regime_confirm_weeks": 3}

    for offset in (0, 30, 90):
        as_of = full.close.index[-1 - offset]
        truncated = model_ai._slice(full, as_of)

        from_full = model_ai.detect_regime_confirmed(model_ai._slice(full, as_of), cfg)
        from_trunc = model_ai.detect_regime_confirmed(truncated, cfg)

        assert from_full["regime"] == from_trunc["regime"], (
            f"LOOK-AHEAD in detect_regime_confirmed at {as_of.date()}: "
            f"{from_full['regime']} != {from_trunc['regime']}")
    print("PASS test_regime_confirm_weeks_preserves_no_lookahead")


def test_us_atr_stop_override_does_not_leak_into_egx():
    """
    The Tier 1 ATR-multiplier fix (research/trials.jsonl) was evidence-tested
    for US only - EGX must still get the global CONFIG default (3.0), not
    US's tuned override, until EGX is separately tested at this value.
    """
    us_cfg = model_ai.market_config("US")
    egx_cfg = model_ai.market_config("EGX")
    assert us_cfg["atr_stop_mult"] == 5.0, us_cfg["atr_stop_mult"]
    assert egx_cfg["atr_stop_mult"] == model_ai.CONFIG["atr_stop_mult"] == 3.0, \
        egx_cfg["atr_stop_mult"]
    print("PASS test_us_atr_stop_override_does_not_leak_into_egx")


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
