"""
MODEL_AI - US live paper trading (Alpaca)
=========================================
Execution layer for ``model_ai.generate_target_weights``. The strategy decides
WHAT to hold; this file decides HOW to get there and enforces Layer 4 risk
controls that need to run more often than the weekly rebalance.

Two modes, both driven by GitHub Actions:

    python live_trader_ai.py rebalance   # weekly (Monday): move to target weights
    python live_trader_ai.py stops       # daily: ATR trailing + hard stop + DD breaker

Account isolation: set ALPACA_API_KEY_AI / ALPACA_SECRET_KEY_AI to a THIRD
paper account, separate from Model A and Model D. Without them the script
exits non-zero rather than silently trading the wrong account.

Trailing-stop state (per-position high-water mark, entry date for min-hold,
portfolio equity peak for the drawdown breaker) lives in
``data/model_ai_state.json`` — Alpaca does not remember any of it, and a fresh
CI checkout has no memory, so the workflow commits it back.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest
from dotenv import load_dotenv

import model_ai

load_dotenv()

MARKET = "US"
STATE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "data", "model_ai_state.json")

api_key = os.getenv("ALPACA_API_KEY_AI")
secret_key = os.getenv("ALPACA_SECRET_KEY_AI")
if not api_key or not secret_key:
    sys.exit(
        "Missing ALPACA_API_KEY_AI / ALPACA_SECRET_KEY_AI.\n"
        "MODEL_AI trades a THIRD Alpaca paper account, isolated from Model A "
        "and Model D. Create one at https://app.alpaca.markets (Paper), then "
        "add both keys under Settings > Secrets and variables > Actions."
    )

client = TradingClient(api_key, secret_key, paper=True)


# ============================================================
# STATE
# ============================================================
def load_state() -> dict:
    if os.path.exists(STATE_PATH):
        with open(STATE_PATH) as f:
            return json.load(f)
    return {"peak_equity": None, "breaker_on": False, "positions": {}}


def save_state(state: dict):
    os.makedirs(os.path.dirname(STATE_PATH), exist_ok=True)
    with open(STATE_PATH, "w") as f:
        json.dump(state, f, indent=2)


def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _days_held(entry_date: str | None) -> int:
    if not entry_date:
        return 999
    try:
        d = datetime.strptime(entry_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except ValueError:
        return 999
    return (datetime.now(timezone.utc) - d).days


# ============================================================
# BROKER HELPERS
# ============================================================
def current_positions() -> dict[str, dict]:
    out = {}
    for p in client.get_all_positions():
        out[p.symbol] = {
            "qty": int(float(p.qty)),
            "avg_entry": float(p.avg_entry_price),
            "price": float(p.current_price),
            "market_value": float(p.market_value),
        }
    return out


def submit(symbol: str, qty: int, side: OrderSide) -> bool:
    if qty <= 0:
        return False
    try:
        client.submit_order(MarketOrderRequest(
            symbol=symbol, qty=qty, side=side, time_in_force=TimeInForce.DAY))
        print(f"  {side.value.upper():4} {qty:>5}x {symbol}")
        return True
    except Exception as e:
        print(f"  ORDER FAILED {side.value} {qty}x {symbol}: {e}")
        return False


def market_is_open() -> bool:
    if not client.get_clock().is_open:
        print("Market is closed - no action.")
        return False
    return True


# ============================================================
# LAYER 4 - DAILY RISK CHECKS
# ============================================================
def check_stops(state: dict, cfg: dict) -> dict:
    """
    Daily risk pass, independent of the rebalance cadence:
      * hard stop     -8% from entry
      * ATR trailing  3x ATR(14) below the high-water mark since entry
      * DD breaker    flatten everything if equity is >15% off its peak

    The min-hold rule does NOT protect a stopped-out position; risk limits
    always win over churn control.
    """
    account = client.get_account()
    equity = float(account.equity)

    peak = state.get("peak_equity") or equity
    peak = max(peak, equity)
    state["peak_equity"] = peak
    drawdown = equity / peak - 1
    print(f"Equity ${equity:,.2f} | peak ${peak:,.2f} | drawdown {drawdown:+.2%}")

    positions = current_positions()
    if not positions:
        print("No open positions.")
        state["positions"] = {}
        return state

    # ---- portfolio drawdown breaker ----
    if drawdown <= -cfg["dd_breaker"]:
        print(f"!! DD BREAKER: drawdown {drawdown:.2%} beyond "
              f"-{cfg['dd_breaker']:.0%} - flattening to cash")
        state["breaker_on"] = True
        for sym, pos in positions.items():
            submit(sym, pos["qty"], OrderSide.SELL)
        state["positions"] = {}
        return state

    # ---- per-position stops ----
    data = model_ai.load_price_data(MARKET, cfg)
    pos_state = state.setdefault("positions", {})

    for sym, pos in positions.items():
        px, entry = pos["price"], pos["avg_entry"]
        ps = pos_state.setdefault(sym, {"entry_date": _today(), "peak_price": px})
        ps["peak_price"] = max(ps.get("peak_price", px), px)

        pnl_pct = px / entry - 1
        reason = None

        if pnl_pct <= cfg["hard_stop"]:
            reason = f"hard stop {pnl_pct:.2%} <= {cfg['hard_stop']:.0%}"
        elif sym in data.close.columns:
            a = model_ai.atr(data.high[sym], data.low[sym], data.close[sym]).iloc[-1]
            if a and a == a:  # not NaN
                trail = ps["peak_price"] - cfg["atr_stop_mult"] * float(a)
                if px < trail:
                    reason = (f"ATR trail: {px:.2f} < {trail:.2f} "
                              f"(peak {ps['peak_price']:.2f} - "
                              f"{cfg['atr_stop_mult']}x ATR {float(a):.2f})")

        if reason:
            print(f"  STOP {sym}: {reason}")
            if submit(sym, pos["qty"], OrderSide.SELL):
                pos_state.pop(sym, None)
        else:
            print(f"  hold {sym}: {pnl_pct:+.2%} from entry")

    return state


# ============================================================
# WEEKLY REBALANCE
# ============================================================
def rebalance(state: dict, cfg: dict) -> dict:
    """Move the book to MODEL_AI's target weights."""
    account = client.get_account()
    equity = float(account.equity)

    peak = max(state.get("peak_equity") or equity, equity)
    state["peak_equity"] = peak
    drawdown = equity / peak - 1

    data = model_ai.load_price_data(MARKET, cfg)
    as_of = data.close.index[-1]
    regime = model_ai.detect_regime(model_ai._slice(data, as_of), cfg)

    # The breaker latches: once tripped, stay in cash until RISK_ON returns.
    if state.get("breaker_on"):
        if regime["regime"] == "RISK_ON":
            print("DD breaker released - regime back to RISK_ON")
            state["breaker_on"] = False
        else:
            print(f"DD breaker still engaged (regime {regime['regime']}, "
                  f"drawdown {drawdown:+.2%}) - staying in cash")
            return state

    targets = model_ai.generate_target_weights(as_of, MARKET, cfg, data)
    positions = current_positions()
    pos_state = state.setdefault("positions", {})

    print(f"\nRebalancing ${equity:,.2f} equity -> "
          f"{len(targets)} targets ({sum(targets.values()):.1%} gross)")

    # ---- exits: held but no longer targeted ----
    for sym, pos in positions.items():
        if sym in targets:
            continue
        held = _days_held(pos_state.get(sym, {}).get("entry_date"))
        if held < cfg["min_hold_days"]:
            print(f"  keep {sym}: min-hold ({held}d < {cfg['min_hold_days']}d)")
            continue
        print(f"  exit {sym}: dropped from targets")
        if submit(sym, pos["qty"], OrderSide.SELL):
            pos_state.pop(sym, None)

    # ---- entries / resizes ----
    for sym, weight in sorted(targets.items(), key=lambda kv: -kv[1]):
        if sym not in data.close.columns:
            continue
        price = float(data.close[sym].iloc[-1])
        if price <= 0:
            continue

        target_qty = int((equity * weight) / price)
        held_qty = positions.get(sym, {}).get("qty", 0)
        delta = target_qty - held_qty

        if target_qty <= 0:
            continue
        # Ignore sub-1% drift so the book isn't churned for rounding.
        if abs(delta) * price < equity * 0.01:
            print(f"  hold {sym}: {held_qty} ~= target {target_qty}")
            continue

        if delta > 0:
            if submit(sym, delta, OrderSide.BUY):
                pos_state.setdefault(sym, {"entry_date": _today(),
                                           "peak_price": price})
        else:
            held = _days_held(pos_state.get(sym, {}).get("entry_date"))
            if held < cfg["min_hold_days"]:
                print(f"  keep {sym}: min-hold blocks trim ({held}d)")
                continue
            submit(sym, -delta, OrderSide.SELL)

    return state


# ============================================================
def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "rebalance"
    if mode not in ("rebalance", "stops"):
        sys.exit(f"Unknown mode {mode!r}. Use 'rebalance' or 'stops'.")

    cfg = model_ai.market_config(MARKET)
    print(f"=== MODEL_AI US [{mode}] {_today()} ===")

    if not market_is_open():
        return

    state = load_state()
    state = check_stops(state, cfg) if mode == "stops" else rebalance(state, cfg)
    save_state(state)

    account = client.get_account()
    print(f"\nPortfolio value: ${float(account.portfolio_value):,.2f} "
          f"| cash ${float(account.cash):,.2f}")


if __name__ == "__main__":
    main()
