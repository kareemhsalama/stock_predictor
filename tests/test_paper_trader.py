"""
PaperTrader execution-cost tests
================================
The engine used to fill any size, instantly, at the marked close, for free.
Every EGX result in this repo was produced under those assumptions, so these
tests pin down the ones that replaced them:

  * cash is conserved - what leaves the balance equals notional plus fees,
  * fills are worse than the mark, in the direction that hurts,
  * an order bigger than the book fills partially rather than pretending,
  * sizing through affordable_qty never produces a rejected order,
  * a session-dated snapshot survives the UTC-midnight crossing that cost this
    repo 2026-08-06.

Run:  python -m pytest tests/test_paper_trader.py -v
      python tests/test_paper_trader.py          (no pytest needed)
"""

from __future__ import annotations

import os
import sys
import tempfile
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from paper_trader import CostModel, Market, PaperTrader, MARKETS  # noqa: E402


def _trader(costs: CostModel | None = None, budget: float = 1_000_000) -> PaperTrader:
    """A throwaway EGX trader on a temp ledger, optionally with custom costs."""
    handle, path = tempfile.mkstemp(suffix=".json")
    os.close(handle)
    os.unlink(path)                      # PaperTrader creates it on first save

    if costs is not None:
        base = MARKETS["EGX"]
        market = Market(code="TEST", currency=base.currency,
                        currency_symbol=base.currency_symbol,
                        ticker_suffix="", timezone=base.timezone,
                        open_time=base.open_time, close_time=base.close_time,
                        trading_days=base.trading_days, costs=costs)
        MARKETS["TEST"] = market
        return PaperTrader(model="T", market="TEST", ledger_path=path,
                           initial_budget=budget)
    return PaperTrader(model="T", market="EGX", ledger_path=path,
                       initial_budget=budget)


def test_free_cost_model_is_the_old_behaviour():
    """A zero cost model must fill exactly at the mark, as the engine used to."""
    t = _trader(CostModel())
    before = t.cash
    result = t.buy("X", 100, 50.0)

    assert result["status"] == "FILLED"
    assert result["price"] == 50.0, result["price"]
    assert result["fees"] == 0.0
    assert abs(before - t.cash - 5000.0) < 1e-9
    print("PASS test_free_cost_model_is_the_old_behaviour")


def test_buy_pays_spread_and_fees():
    costs = CostModel(commission_bps=10, stamp_duty_bps=5, half_spread_bps=20)
    t = _trader(costs)
    before = t.cash
    r = t.buy("X", 100, 100.0)

    # 20 bps of spread means buying at 100.20, never at the mark.
    assert abs(r["price"] - 100.20) < 1e-9, r["price"]
    notional = 100 * 100.20
    expected_fees = notional * 15 / 1e4
    assert abs(r["fees"] - expected_fees) < 1e-6, (r["fees"], expected_fees)
    # Cash conservation: the balance falls by exactly notional + fees.
    assert abs((before - t.cash) - (notional + expected_fees)) < 1e-6
    print("PASS test_buy_pays_spread_and_fees")


def test_sell_receives_less_than_the_mark():
    costs = CostModel(commission_bps=10, stamp_duty_bps=5, half_spread_bps=20)
    t = _trader(costs)
    t.buy("X", 100, 100.0)
    before = t.cash
    r = t.sell("X", 100, 100.0)

    assert abs(r["price"] - 99.80) < 1e-9, r["price"]
    notional = 100 * 99.80
    assert abs((t.cash - before) - (notional - r["fees"])) < 1e-6
    # A flat round trip must lose money once costs exist.
    assert r["pnl"] < 0, r["pnl"]
    print(f"PASS test_sell_receives_less_than_the_mark (round trip {r['pnl']:.2f})")


def test_minimum_commission_floor_bites_small_orders():
    costs = CostModel(commission_bps=1, min_commission=10.0)
    t = _trader(costs)
    r = t.buy("X", 1, 100.0)          # 1 bp of 100 = 0.01, floor is 10
    assert abs(r["fees"] - 10.0) < 1e-9, r["fees"]
    print("PASS test_minimum_commission_floor_bites_small_orders")


def test_impact_scales_with_participation():
    """Bigger orders must fill worse, and proportionally so."""
    costs = CostModel(impact_coef_bps=100, max_participation=1.0)
    t = _trader(costs)
    adv = 1_000_000.0

    small = t.buy("X", 100, 100.0, adv_value=adv)      # 1% of ADV -> 1 bp
    large = t.buy("Y", 1000, 100.0, adv_value=adv)     # 10% of ADV -> 10 bps

    assert abs(small["price"] - 100.01) < 1e-6, small["price"]
    assert abs(large["price"] - 100.10) < 1e-6, large["price"]
    print("PASS test_impact_scales_with_participation")


def test_order_larger_than_the_book_fills_partially():
    costs = CostModel(max_participation=0.20)
    t = _trader(costs)
    adv = 100_000.0                    # 20% cap at price 10 -> 2000 shares

    r = t.buy("THIN", 10_000, 10.0, adv_value=adv)
    assert r["status"] == "FILLED"
    assert r["qty"] == 2000, r["qty"]
    assert r["requested_qty"] == 10_000
    assert r["partial"] is True
    assert t.position_qty("THIN") == 2000
    print("PASS test_order_larger_than_the_book_fills_partially")


def test_partial_exit_keeps_the_remainder():
    costs = CostModel(max_participation=0.20)
    t = _trader(costs)
    t.buy("THIN", 2000, 10.0)                        # no ADV: fills in full
    # 20% of a 50,000 book at price 10 is 1,000 shares - half the position.
    r = t.sell("THIN", 2000, 10.0, adv_value=50_000.0)

    assert r["partial"] is True, r
    assert r["qty"] == 1000, r["qty"]
    assert t.position_qty("THIN") == 1000, t.position_qty("THIN")
    print(f"PASS test_partial_exit_keeps_the_remainder "
          f"(sold {r['qty']}, held {t.position_qty('THIN')})")


def test_affordable_qty_never_gets_rejected():
    """
    The regression that would have bitten in production: sizing on bare
    cash/price puts the order just over the balance once fees exist, and the
    model silently stops trading.
    """
    t = _trader(budget=10_000)         # real EGX costs, small book
    # A price that divides the balance evenly is the worst case: naive sizing
    # spends the entire book on shares and leaves nothing for spread or fees.
    price = 100.0

    naive = int(t.cash / price)
    assert t.buy("X", naive, price)["status"] == "REJECTED"

    qty = t.affordable_qty(price)
    assert qty > 0
    assert t.buy("X", qty, price)["status"] == "FILLED"
    assert t.cash >= 0, t.cash
    print(f"PASS test_affordable_qty_never_gets_rejected "
          f"(naive {naive} rejected, {qty} filled, {t.cash:.2f} left)")


def test_cost_basis_includes_entry_fees():
    costs = CostModel(commission_bps=50, half_spread_bps=0)
    t = _trader(costs)
    t.buy("X", 100, 100.0)
    # 100 shares at 100 plus 50 bps = 10,050 -> basis 100.50 per share.
    assert abs(t.positions["X"]["avg_price"] - 100.50) < 1e-6, t.positions["X"]
    print("PASS test_cost_basis_includes_entry_fees")


def test_fees_and_slippage_are_accumulated():
    t = _trader()
    t.buy("COMI", 100, 100.0, adv_value=10_000_000.0)
    t.sell("COMI", 50, 101.0, adv_value=10_000_000.0)
    assert t.fees_paid > 0
    assert t.slippage_paid > 0
    assert t.cost_model_from is not None
    print(f"PASS test_fees_and_slippage_are_accumulated "
          f"({t.fees_paid:.2f} fees, {t.slippage_paid:.2f} slippage)")


def test_session_date_survives_the_utc_midnight_crossing():
    """
    The 2026-08-06 bug: a US run delayed to 01:03 UTC belongs to the previous
    session, not to the UTC date the clock happens to show.
    """
    us = MARKETS["US"]
    delayed = datetime(2026, 8, 7, 1, 3, tzinfo=timezone.utc)   # 21:03 ET on 08-06
    assert us.session_date(delayed) == "2026-08-06", us.session_date(delayed)

    on_time = datetime(2026, 8, 6, 21, 30, tzinfo=timezone.utc)  # 17:30 ET on 08-06
    assert us.session_date(on_time) == "2026-08-06"

    saturday = datetime(2026, 8, 8, 22, 0, tzinfo=timezone.utc)
    assert us.session_date(saturday) == "2026-08-07"

    # EGX trades Sun-Thu: a Friday run belongs to Thursday.
    egx = MARKETS["EGX"]
    friday = datetime(2026, 8, 7, 18, 0, tzinfo=timezone.utc)
    assert egx.session_date(friday) == "2026-08-06", egx.session_date(friday)
    print("PASS test_session_date_survives_the_utc_midnight_crossing")


def test_snapshot_replaces_and_stays_sorted():
    t = _trader()
    t.snapshot({})
    first = t.snapshots[0]["date"]
    t.snapshot({})                      # same session, must replace
    assert len(t.snapshots) == 1, t.snapshots

    # An out-of-order session must be inserted in place, never appended.
    t.snapshots.insert(0, {"date": "2020-01-01", "equity": 1.0,
                           "cash": 1.0, "pnl": 0.0, "positions": []})
    t.snapshot({})
    dates = [s["date"] for s in t.snapshots]
    assert dates == sorted(dates), dates
    assert dates[-1] == first
    print("PASS test_snapshot_replaces_and_stays_sorted")


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
