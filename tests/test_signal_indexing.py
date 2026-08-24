"""
Signal-indexing regression tests (H1)
======================================
live_trader.py and egx_random_forest.py both build a `Tomorrow_Return` /
`Target` column via `Close.shift(-1)`, which is NaN on the most recent row (no
"tomorrow" exists yet, since it hasn't traded). A plain `data.dropna()`
therefore drops that row before it can be used as "today" - so
`data[features].iloc[[-1]]` (and, in live_trader.py, the Close price used to
size the order) was silently picking up yesterday's row and calling it
"today", on every single run.

These tests reproduce that exact index shift with a synthetic price series
and pin the fix: the row handed to the model for inference, and the price
used to size any resulting order, must come from the true last row of the
input data - not whatever survives the training-set dropna.

Run:  python -m pytest tests/test_signal_indexing.py -v
      python tests/test_signal_indexing.py          (no pytest needed)
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import egx_random_forest  # noqa: E402
import live_trader  # noqa: E402


def _synthetic_ohlcv(n_days: int = 320, seed: int = 3) -> pd.DataFrame:
    """A deterministic random-walk OHLCV panel - no network, no yfinance flakiness."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-01", periods=n_days)
    steps = rng.normal(0.0006, 0.015, n_days)
    close = pd.Series(100 * np.exp(np.cumsum(steps)), index=dates)
    high = close * (1 + rng.uniform(0, 0.01, n_days))
    low = close * (1 - rng.uniform(0, 0.01, n_days))
    open_ = close.shift(1).fillna(close.iloc[0])
    volume = pd.Series(rng.uniform(2e6, 9e6, n_days), index=dates)
    return pd.DataFrame({"Open": open_, "High": high, "Low": low,
                         "Close": close, "Volume": volume})


class _RecordingModel:
    """Stands in for RandomForestClassifier: records what it was trained and
    asked to predict on, without actually fitting anything."""

    def __init__(self, *a, **k):
        pass

    def fit(self, x, y):
        self.recorded["train_rows"] = len(x)
        return self

    def predict_proba(self, X):
        self.recorded["today_index"] = X.index[0]
        return np.array([[0.1, 0.9]])   # confidence 0.9 - above every threshold used here


# ============================================================
# egx_random_forest.py
# ============================================================
def test_egx_predict_signal_uses_true_last_row_as_today():
    raw = _synthetic_ohlcv()
    real_download = egx_random_forest.yf.download
    real_model = egx_random_forest.RandomForestClassifier

    recorded = {}
    Model = type("Model", (_RecordingModel,), {"recorded": recorded})
    egx_random_forest.yf.download = lambda *a, **k: raw.copy()
    egx_random_forest.RandomForestClassifier = Model
    try:
        confidence, last_close, adv = egx_random_forest.predict_signal("TEST")
    finally:
        egx_random_forest.yf.download = real_download
        egx_random_forest.RandomForestClassifier = real_model

    assert confidence is not None, "predict_signal skipped when it should have run"
    assert recorded["today_index"] == raw.index[-1], (
        f"predicted off {recorded['today_index'].date()}, "
        f"expected the true last session {raw.index[-1].date()}"
    )
    assert last_close == raw["Close"].iloc[-1]
    print("PASS test_egx_predict_signal_uses_true_last_row_as_today")


# ============================================================
# live_trader.py
# ============================================================
class _FakeClock:
    is_open = True


class _FakeAccount:
    cash = "100000"
    portfolio_value = "100000"


class _FakeClient:
    def __init__(self):
        self.orders = []

    def get_clock(self):
        return _FakeClock()

    def get_all_positions(self):
        return []

    def get_account(self):
        return _FakeAccount()

    def submit_order(self, order):
        self.orders.append(order)


def test_live_trader_uses_true_last_row_for_signal_and_sizing():
    raw = _synthetic_ohlcv()
    # A single-ticker yfinance download carries a MultiIndex on columns;
    # live_trade() immediately does data.columns.droplevel(1) to flatten it.
    multi = raw.copy()
    multi.columns = pd.MultiIndex.from_product([multi.columns, ["TEST"]])

    real_download = live_trader.yf.download
    real_client = live_trader.client
    real_model = live_trader.RandomForestClassifier

    recorded = {}
    Model = type("Model", (_RecordingModel,), {"recorded": recorded})
    fake_client = _FakeClient()
    live_trader.yf.download = lambda *a, **k: multi.copy()
    live_trader.client = fake_client
    live_trader.RandomForestClassifier = Model
    try:
        live_trader.live_trade("TEST")
    finally:
        live_trader.yf.download = real_download
        live_trader.client = real_client
        live_trader.RandomForestClassifier = real_model

    true_last_close = float(raw["Close"].iloc[-1])

    assert recorded["today_index"] == raw.index[-1], (
        f"predicted off {recorded['today_index'].date()}, "
        f"expected the true last session {raw.index[-1].date()}"
    )
    assert len(fake_client.orders) == 1, "expected exactly one BUY at confidence 0.9"
    expected_qty = int((float(_FakeAccount.cash) * 0.10) / true_last_close)
    assert fake_client.orders[0].qty == expected_qty, (
        f"order sized {fake_client.orders[0].qty} shares using a stale close; "
        f"expected {expected_qty} shares off the true last close {true_last_close:.2f}"
    )
    print("PASS test_live_trader_uses_true_last_row_for_signal_and_sizing")


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
