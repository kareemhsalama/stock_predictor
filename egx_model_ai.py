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
from telegram_notifier import TelegramNotifier

try:
    notifier = TelegramNotifier()
except ValueError as e:
    print(f"Telegram alerts disabled: {e}")
    notifier = None

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


def average_daily_value(data: model_ai.PriceData, window: int = 20) -> dict[str, float]:
    """
    Average daily traded value per name, in EGP.

    Feeds PaperTrader's impact term and participation cap. Without it every
    fill is priced as if the book were infinitely deep, which on EGX is the
    single most flattering assumption the simulator can make.
    """
    adv = {}
    for sym in data.close.columns:
        value = (data.close[sym] * data.volume[sym]).tail(window).mean()
        if value == value and value > 0:      # NaN-safe
            adv[sym] = float(value)
    return adv


def _notify_fill(result: dict, reason: str, trades_today: list | None = None):
    """Build a Telegram-shaped trade record from a PaperTrader buy()/sell()
    result and send it, if the order actually filled."""
    if result.get("status") != "FILLED":
        return
    trade_record = {
        "symbol": result["symbol"], "side": result["side"], "qty": result["qty"],
        "price": result["price"], "timestamp": result["timestamp"], "reason": reason,
    }
    if trades_today is not None:
        trades_today.append(trade_record)
    if notifier:
        try:
            notifier.send_trade_alert(trade_record)
        except Exception as e:
            print(f"Telegram alert failed: {e}")


def apply_stops(trader: PaperTrader, data: model_ai.PriceData,
                prices: dict[str, float], cfg: dict,
                adv: dict[str, float] | None = None,
                trades_today: list | None = None) -> None:
    """Hard stop + ATR trailing stop, checked every run."""
    adv = adv or {}
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
            result = trader.sell(sym, int(pos["qty"]), px, adv.get(sym))
            _notify_fill(result, reason, trades_today)
            # A thin name may only part-exit today; keep its high-water mark
            # so the trail still governs the remainder tomorrow.
            if sym not in trader.positions:
                peaks.pop(sym, None)
            elif result.get("partial"):
                print(f"    {sym}: {trader.position_qty(sym)} shares still held "
                      f"— stop continues next session")


def rebalance(trader: PaperTrader, data: model_ai.PriceData,
              prices: dict[str, float], cfg: dict,
              adv: dict[str, float] | None = None,
              trades_today: list | None = None) -> None:
    """Move the book toward MODEL_AI's target weights."""
    adv = adv or {}
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
                result = trader.sell(sym, int(trader.positions[sym]["qty"]), px, adv.get(sym))
                _notify_fill(result, "dropped from rebalance targets", trades_today)

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
            # Cap the buy at what cash covers once fees are counted; the
            # target weight is a goal, not a promise the book can fund.
            qty = min(delta, trader.affordable_qty(px, adv.get(sym)))
            if qty > 0:
                result = trader.buy(sym, qty, px, adv.get(sym))
                _notify_fill(result, f"rebalance entry, target weight {weight:.1%}",
                            trades_today)
            else:
                print(f"  {sym}: skipped — cash cannot fund 1 share with fees")
        elif delta < 0:
            result = trader.sell(sym, -delta, px, adv.get(sym))
            _notify_fill(result, f"rebalance trim to target weight {weight:.1%}",
                        trades_today)


def run():
    cfg = model_ai.market_config(MARKET)
    trader = PaperTrader(
        model="AI", market=MARKET, ledger_path=LEDGER_PATH,
        initial_budget=INITIAL_BUDGET,
        name="Regime-Aware Momentum + Vol Target (EGX)",
    )

    trades_today = []
    snap = None
    prev_equity = None
    try:
        data = model_ai.load_price_data(MARKET, cfg)
        latest = data.close.index[-1]
        prices = {t: float(data.close[t].iloc[-1])
                  for t in data.close.columns
                  if data.close[t].iloc[-1] == data.close[t].iloc[-1]}

        adv = average_daily_value(data)

        print(f"=== MODEL_AI EGX {latest.date()} ===")
        print(f"ADV known for {len(adv)}/{len(data.close.columns)} names")

        apply_stops(trader, data, prices, cfg, adv, trades_today=trades_today)

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
            rebalance(trader, data, prices, cfg, adv, trades_today=trades_today)
            trader.extra["bootstrapped"] = True
        else:
            print(f"\nNot rebalance day (weekday {weekday}, "
                  f"rebalance on {REBALANCE_WEEKDAY}) - stops only")

        prev_equity = trader.snapshots[-1]["equity"] if trader.snapshots else None
        snap = trader.snapshot(prices)
        print(f"\n--- MODEL_AI EGX snapshot {snap['date']} ---")
        print(f"  Equity: {snap['equity']:,.2f} EGP | Cash: {snap['cash']:,.2f} EGP "
              f"| PnL: {snap['pnl']:+,.2f} EGP")
        print(f"  Costs to date: {trader.fees_paid:,.2f} EGP fees + "
              f"{trader.slippage_paid:,.2f} EGP slippage "
              f"(applied from {trader.cost_model_from})")
        print(f"  Open positions: {len(snap['positions'])}")
        print(f"  Ledger: {trader.ledger_path}")
    finally:
        if notifier:
            try:
                pnl = ({"daily_pnl": snap["equity"] - prev_equity}
                      if snap is not None and prev_equity is not None else None)
                equity = snap["equity"] if snap is not None else None
                notifier.send_daily_summary(trades_today, pnl=pnl, equity=equity)
            except Exception as e:
                print(f"Telegram daily summary failed: {e}")


if __name__ == "__main__":
    run()
