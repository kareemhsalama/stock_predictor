"""
MODEL_AI - EGX paper trading
============================
The EGX counterpart of ``live_trader_ai.py``. Same strategy module, different
execution: EGX has no broker API, so orders go through the ``PaperTrader``
simulator and the ledger at ``data/model_ai_egx_ledger.json``.

Runs end-of-day after the 14:15 EET close: compute target weights from the
latest closes, move the book toward them, apply stops, then snapshot.

Differences from the US runner, all forced by the market:
  * long-only (no reliable borrow on EGX),
  * one combined pass instead of separate rebalance/stops jobs — EGX gets a
    single post-close run per day, so stops are checked in the same job and
    the weight rebalance is gated to one weekday,
  * regime benchmark falls back to an equal-weight composite because
    ^CASE30 is unusable on yfinance (see model_ai._benchmark_series).
"""

from __future__ import annotations

import os
from datetime import datetime, timezone

import model_ai
from paper_trader import PaperTrader

MARKET = "EGX"
LEDGER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "data", "model_ai_egx_ledger.json")

# ~1M EGP, matching EGX Model A so the two are directly comparable.
INITIAL_BUDGET = 1_000_000

# EGX trades Sun-Thu; rebalance weekly on Sunday (weekday 6), check stops daily.
REBALANCE_WEEKDAY = 6


def _sync_position_state(trader: PaperTrader, prices: dict[str, float]):
    """
    Track per-position high-water marks for the ATR trailing stop.

    Lives in ``trader.extra`` because that is the only part of the engine that
    is persisted to the ledger — a plain attribute would be silently dropped on
    save and the trailing stop would never fire on a fresh CI checkout.
    """
    state = trader.extra.setdefault("peak_price", {})
    for sym in list(state):
        if sym not in trader.positions:
            state.pop(sym)
    for sym in trader.positions:
        px = prices.get(sym)
        if px:
            state[sym] = max(state.get(sym, px), px)
    return state


def apply_stops(trader: PaperTrader, data: model_ai.PriceData,
                prices: dict[str, float], cfg: dict) -> None:
    """Hard stop + ATR trailing stop, checked every run."""
    peaks = _sync_position_state(trader, prices)

    for sym in list(trader.positions):
        pos = trader.positions[sym]
        px = prices.get(sym)
        if not px:
            continue

        pnl_pct = px / pos["avg_price"] - 1
        reason = None

        if pnl_pct <= cfg["hard_stop"]:
            reason = f"hard stop {pnl_pct:.2%}"
        elif sym in data.close.columns:
            a = model_ai.atr(data.high[sym], data.low[sym], data.close[sym]).iloc[-1]
            if a and a == a:
                trail = peaks.get(sym, px) - cfg["atr_stop_mult"] * float(a)
                if px < trail:
                    reason = f"ATR trail {px:.2f} < {trail:.2f}"

        if reason:
            print(f"  STOP {sym}: {reason}")
            trader.sell(sym, int(pos["qty"]), px)
            peaks.pop(sym, None)


def rebalance(trader: PaperTrader, data: model_ai.PriceData,
              prices: dict[str, float], cfg: dict) -> None:
    """Move the book toward MODEL_AI's target weights."""
    as_of = data.close.index[-1]
    targets = model_ai.generate_target_weights(as_of, MARKET, cfg, data)

    equity = trader.get_portfolio_value(prices)
    print(f"\nRebalancing {equity:,.2f} EGP -> {len(targets)} targets "
          f"({sum(targets.values()):.1%} gross)")

    # Exits first so their proceeds fund the entries.
    for sym in list(trader.positions):
        if sym not in targets:
            px = prices.get(sym)
            if px:
                print(f"  exit {sym}: dropped from targets")
                trader.sell(sym, int(trader.positions[sym]["qty"]), px)

    for sym, weight in sorted(targets.items(), key=lambda kv: -kv[1]):
        px = prices.get(sym)
        if not px or px <= 0:
            continue
        target_qty = int((equity * weight) / px)
        held = trader.position_qty(sym)
        delta = target_qty - held

        if abs(delta) * px < equity * 0.01:      # ignore sub-1% drift
            continue
        if delta > 0:
            trader.buy(sym, delta, px)
        elif delta < 0:
            trader.sell(sym, -delta, px)


def run():
    cfg = model_ai.market_config(MARKET)
    trader = PaperTrader(
        model="AI", market=MARKET, ledger_path=LEDGER_PATH,
        initial_budget=INITIAL_BUDGET,
        name="Regime-Aware Momentum + Vol Target (EGX)",
    )

    data = model_ai.load_price_data(MARKET, cfg)
    latest = data.close.index[-1]
    prices = {t: float(data.close[t].iloc[-1])
              for t in data.close.columns
              if data.close[t].iloc[-1] == data.close[t].iloc[-1]}

    print(f"=== MODEL_AI EGX {latest.date()} ===")

    apply_stops(trader, data, prices, cfg)

    # Bootstrap on the first run ever: waiting for the weekly Sunday slot would
    # leave a freshly deployed model in cash for up to six days. One-shot flag
    # rather than "is the book empty?" — empty is the correct state in RISK_OFF,
    # and re-bootstrapping daily would override the regime gate.
    weekday = datetime.now(timezone.utc).weekday()
    first_run = not trader.extra.get("bootstrapped")

    if first_run:
        print(f"\nFirst run: bootstrapping the book rather than waiting for "
              f"weekday {REBALANCE_WEEKDAY}.")
    if first_run or weekday == REBALANCE_WEEKDAY:
        rebalance(trader, data, prices, cfg)
        trader.extra["bootstrapped"] = True
    else:
        print(f"\nNot rebalance day (weekday {weekday}, "
              f"rebalance on {REBALANCE_WEEKDAY}) - stops only")

    snap = trader.snapshot(prices)
    print(f"\n--- MODEL_AI EGX snapshot {snap['date']} ---")
    print(f"  Equity: {snap['equity']:,.2f} EGP | Cash: {snap['cash']:,.2f} EGP "
          f"| PnL: {snap['pnl']:+,.2f} EGP")
    print(f"  Open positions: {len(snap['positions'])}")
    print(f"  Ledger: {trader.ledger_path}")


if __name__ == "__main__":
    run()
