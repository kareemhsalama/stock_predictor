"""
Backtest harness trust tests (Phase 1 of the Model AI research project)
=========================================================================
Two properties the harness must have before any performance number from it
is trustworthy:

  holdout enforced   nothing computed by run_backtest() may see a date on or
                     after HOLDOUT_START unless the caller explicitly opts in
                     with allow_holdout=True. Phases 2-3 of the redesign are
                     never supposed to pass that flag - this test is what
                     makes that a code-enforced guarantee instead of an
                     honor-system comment.

  costs are real     the per-name commission/spread/impact cost (from
                     paper_trader.CostModel) must actually respond to book
                     size and each name's own liquidity - a flat bps-of-
                     turnover rate (the old behaviour) cannot do that, and a
                     flat rate is exactly the kind of unrealism that
                     overstates backtest Sharpe.

Run:  python -m pytest tests/test_backtest_harness.py -v
      python tests/test_backtest_harness.py          (no pytest needed)
"""

from __future__ import annotations

import json
import os
import sys
import tempfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import backtest_model_ai as bt  # noqa: E402
import model_ai  # noqa: E402


def _synthetic_panel(n_days: int = 1400, n_tickers: int = 15, seed: int = 11,
                     start: str = "2022-06-01") -> model_ai.PriceData:
    """Deterministic random-walk panel spanning past HOLDOUT_START - no network."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=n_days)
    tickers = [f"T{i:02d}" for i in range(n_tickers)]

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


def _config():
    cfg = model_ai.market_config("US")
    # Smaller universe/warmup-friendly knobs so the synthetic 15-ticker panel
    # produces a tradable book without needing the real ~60-name universe.
    return {**cfg, "min_dollar_volume": 0}


def test_holdout_is_enforced_by_default():
    panel = _synthetic_panel()
    cfg = _config()

    in_sample = bt.run_backtest("US", config=cfg, data=panel, verbose=False)
    assert in_sample["returns"].index.max() < pd.Timestamp(bt.HOLDOUT_START), (
        f"default run leaked into the holdout: last date "
        f"{in_sample['returns'].index.max().date()}"
    )

    full = bt.run_backtest("US", config=cfg, data=panel, verbose=False,
                           allow_holdout=True)
    assert full["returns"].index.max() >= pd.Timestamp(bt.HOLDOUT_START), (
        "allow_holdout=True did not actually reach the holdout period - "
        "the guard isn't proven to be doing anything"
    )
    print("PASS test_holdout_is_enforced_by_default")


def test_costs_respond_to_book_size_and_liquidity():
    """
    A flat bps-of-turnover rate produces IDENTICAL cost fractions regardless
    of book size, because it's a pure percentage. Real costs must not: a
    bigger book pushes more dollars through the same ADV, so the participation
    /impact term - and the commission floor's bite - both change with size.
    """
    panel = _synthetic_panel()
    cfg = _config()

    small = bt.run_backtest("US", config=cfg, data=panel, verbose=False)

    original_budget = bt.NOMINAL_BUDGET["US"]
    bt.NOMINAL_BUDGET["US"] = original_budget * 50   # same %, much bigger book
    try:
        large = bt.run_backtest("US", config=cfg, data=panel, verbose=False)
    finally:
        bt.NOMINAL_BUDGET["US"] = original_budget

    assert not small["returns"].equals(large["returns"]), (
        "cost fraction was identical across a 50x book-size change - "
        "this is still effectively a flat bps rate, not a real cost model"
    )
    # A larger book pushing the same % through the same liquidity should cost
    # MORE (impact scales with participation), not less.
    assert large["stats"]["cagr"] <= small["stats"]["cagr"], (
        f"larger book somehow cost less: CAGR {small['stats']['cagr']:.2%} "
        f"(small) vs {large['stats']['cagr']:.2%} (large)"
    )
    print("PASS test_costs_respond_to_book_size_and_liquidity")


def test_log_trial_appends_jsonl_rows():
    """
    The trials ledger is the whole basis for an eventual deflated-Sharpe/PBO
    calculation - it must actually accumulate rows (not overwrite), and each
    row must carry the fields that calculation needs.
    """
    panel = _synthetic_panel(n_days=500)
    cfg = _config()
    result = bt.run_backtest("US", config=cfg, data=panel, verbose=False)

    handle, path = tempfile.mkstemp(suffix=".jsonl")
    os.close(handle)
    os.unlink(path)   # log_trial() should create it fresh
    original_log = bt.TRIALS_LOG
    bt.TRIALS_LOG = path
    try:
        bt.log_trial(result, config_diff={"n_positions": 8},
                    layer_touched="sizing", rationale="unit test row 1")
        bt.log_trial(result, config_diff={"n_positions": 6},
                    layer_touched="sizing", rationale="unit test row 2")

        with open(path) as f:
            rows = [json.loads(line) for line in f]
    finally:
        bt.TRIALS_LOG = original_log
        if os.path.exists(path):
            os.unlink(path)

    assert len(rows) == 2, f"expected 2 appended rows, got {len(rows)}"
    for row, expected_diff in zip(rows, [{"n_positions": 8}, {"n_positions": 6}]):
        assert row["config_diff"] == expected_diff
        assert row["market"] == "US"
        assert row["layer_touched"] == "sizing"
        for key in ("sharpe", "sortino", "calmar", "cagr", "max_dd",
                   "n_rebalances", "timestamp", "git_commit"):
            assert key in row, f"missing required field '{key}' in logged row"
    print("PASS test_log_trial_appends_jsonl_rows")


def test_mean_reversion_config_only_flips_momentum_terms():
    cfg = _config()
    flipped = bt.mean_reversion_config(cfg)
    for k, v in cfg["score_weights"].items():
        expected = -v if k.startswith("mom_") else v
        assert flipped["score_weights"][k] == expected, k
    # everything else untouched
    assert flipped["max_weight"] == cfg["max_weight"]
    assert flipped["regime_factors"] == cfg["regime_factors"]
    print("PASS test_mean_reversion_config_only_flips_momentum_terms")


def test_no_regime_gate_config_forces_full_exposure():
    cfg = _config()
    forced = bt.no_regime_gate_config(cfg)
    assert set(forced["regime_factors"].values()) == {1.0}
    assert set(forced["regime_factors"].keys()) == set(cfg["regime_factors"].keys())
    print("PASS test_no_regime_gate_config_forces_full_exposure")


def test_equal_weight_baselines_produce_valid_weights():
    panel = _synthetic_panel()
    cfg = _config()
    as_of = panel.close.index[-1]

    universe_w = bt.equal_weight_universe(as_of, "US", cfg, panel)
    selected_w = bt.equal_weight_selected(as_of, "US", cfg, panel)

    for label, w in [("universe", universe_w), ("selected", selected_w)]:
        assert w, f"{label} weights empty on a healthy synthetic panel"
        assert all(v > 0 for v in w.values()), f"{label}: non-positive weight"
        total = sum(w.values())
        assert abs(total - 1.0) < 1e-9, f"{label}: weights sum to {total}, not 1.0"
    # selection is real (score/trend/liquidity-gated), so it must be a subset
    # of - never larger than - the raw liquidity-filtered universe.
    assert set(selected_w) <= set(universe_w)
    print("PASS test_equal_weight_baselines_produce_valid_weights")


def test_random_portfolio_bootstrap_respects_holdout_and_shape():
    panel = _synthetic_panel()
    cfg = _config()

    null = bt.random_portfolio_bootstrap("US", config=cfg, data=panel, n_draws=20, seed=1)
    assert null["n_draws"] == 20
    assert len(null["sharpes"]) > 0, "no draws produced a usable Sharpe at all"
    assert null["sharpe_mean"] is not None

    # The actual holdout check: default (allow_holdout=False) must never use
    # a rebalance date on or after HOLDOUT_START, and allow_holdout=True must
    # actually reach past it - otherwise this "guard" is unverified by
    # anything in this test (a prior version of this test asserted shape/
    # determinism only, and would have passed even with the holdout slice
    # in random_portfolio_bootstrap deleted entirely).
    assert null["last_date"] is not None
    assert null["last_date"] < pd.Timestamp(bt.HOLDOUT_START), (
        f"default run's last rebalance date leaked into the holdout: "
        f"{null['last_date'].date()}"
    )
    full = bt.random_portfolio_bootstrap("US", config=cfg, data=panel, n_draws=20,
                                         seed=1, allow_holdout=True)
    assert full["last_date"] >= pd.Timestamp(bt.HOLDOUT_START), (
        "allow_holdout=True did not actually reach the holdout period - "
        "the guard isn't proven to be doing anything"
    )

    # Determinism: same seed -> identical distribution.
    again = bt.random_portfolio_bootstrap("US", config=cfg, data=panel, n_draws=20, seed=1)
    assert null["sharpes"] == again["sharpes"], "same seed produced a different draw"

    # Different seed -> (almost certainly) a different distribution.
    other = bt.random_portfolio_bootstrap("US", config=cfg, data=panel, n_draws=20, seed=2)
    assert null["sharpes"] != other["sharpes"]
    print("PASS test_random_portfolio_bootstrap_respects_holdout_and_shape")


def test_run_baselines_end_to_end_on_synthetic_panel():
    panel = _synthetic_panel()
    cfg = _config()
    result = bt.run_baselines("US", config=cfg, data=panel, n_random_draws=10,
                              verbose=False)
    for key in ("real", "no_signal", "selection_only", "no_regime", "reversal",
               "null_distribution"):
        assert key in result
    assert result["real"]["returns"].index.max() < pd.Timestamp(bt.HOLDOUT_START)
    print("PASS test_run_baselines_end_to_end_on_synthetic_panel")


def test_atr_trailing_stop_reduces_drawdown_when_tightened():
    """
    A much tighter ATR multiplier must produce a shallower (or equal) max
    drawdown than a much looser one - if it doesn't, the ATR stop added to
    run_backtest() isn't actually constraining the loop, it's decorative.

    Checked across several independent synthetic panels (not just one seed)
    per a Phase 6 review note: a single-seed pass here couldn't rule out the
    result being a seed-specific artifact rather than the stop genuinely
    constraining drawdown in general.
    """
    cfg = _config()
    for seed in (23, 1, 2, 3, 4, 5, 42, 99):
        panel = _synthetic_panel(n_days=1400, n_tickers=15, seed=seed)

        tight = bt.run_backtest("US", config={**cfg, "atr_stop_mult": 0.5},
                                data=panel, verbose=False)
        loose = bt.run_backtest("US", config={**cfg, "atr_stop_mult": 100.0},
                                data=panel, verbose=False)

        assert tight["stats"]["max_dd"] >= loose["stats"]["max_dd"], (
            f"seed={seed}: tight ATR stop drawdown ({tight['stats']['max_dd']:.2%}) "
            f"is WORSE than a loose one's ({loose['stats']['max_dd']:.2%}) - the "
            f"stop isn't actually constraining drawdown"
        )
        assert tight["returns"].to_numpy().tolist() != loose["returns"].to_numpy().tolist(), (
            f"seed={seed}: tight vs. loose ATR multiplier produced identical "
            f"return series - the stop appears to be a no-op"
        )
    print("PASS test_atr_trailing_stop_reduces_drawdown_when_tightened (8 seeds)")


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
            except AssertionError as e:
                failures += 1
                print(f"FAIL {name}: {e}")
    sys.exit(1 if failures else 0)
