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

import os
import sys

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
