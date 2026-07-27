"""
Universal paper-trading engine
===============================
A market-agnostic simulator for markets where we have no broker API (e.g. EGX).
Generalized from the EGX-only ``EGXPaperTrader`` prototype in ``egy.py``:
currency, ticker suffix and trading-hours are now driven by a ``MARKETS``
registry instead of being hard-coded.

Each model-in-a-market gets its own isolated JSON ledger, e.g.::

    model_a_egx_ledger.json   # Model A (Random Forest) on EGX

The ledger's ``snapshots`` array is written in the exact same shape the US
Alpaca ledger uses (``scripts/snapshot_ledger.py``) so the Chart.js dashboard
renders both without any per-market transform code:

    {"date", "equity", "cash", "pnl", "positions": [
        {"symbol", "qty", "avg_entry", "current_price",
         "market_value", "unrealized_pnl", "side"}]}

Internal bookkeeping (running cash, avg entry price per position, full trade
log) is persisted alongside the snapshots so state survives between runs on a
fresh CI checkout.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, time, timezone
from zoneinfo import ZoneInfo


# ============================================================
# MARKET REGISTRY
# ============================================================
@dataclass(frozen=True)
class Market:
    """Static configuration for a tradable market."""

    code: str                     # short id, e.g. "EGX", "US"
    currency: str                 # ISO-ish currency code, e.g. "EGP", "USD"
    currency_symbol: str          # display symbol, e.g. "EGP", "$"
    ticker_suffix: str            # Yahoo Finance suffix, e.g. ".CA" (US = "")
    timezone: str                 # IANA tz for trading-hours checks
    open_time: time               # local market open
    close_time: time              # local market close
    trading_days: tuple[int, ...] = field(default=(0, 1, 2, 3, 4))  # Mon=0..Sun=6

    def localize(self, symbol: str) -> str:
        """Ensure a bare ticker carries this market's Yahoo suffix."""
        if self.ticker_suffix and not symbol.endswith(self.ticker_suffix):
            return f"{symbol}{self.ticker_suffix}"
        return symbol


MARKETS: dict[str, Market] = {
    # Egyptian Exchange: Sun-Thu, 10:00-14:15 EET (Africa/Cairo).
    "EGX": Market(
        code="EGX",
        currency="EGP",
        currency_symbol="EGP",
        ticker_suffix=".CA",
        timezone="Africa/Cairo",
        open_time=time(10, 0),
        close_time=time(14, 15),
        trading_days=(6, 0, 1, 2, 3),  # Sun, Mon, Tue, Wed, Thu
    ),
    # US markets: Mon-Fri, 09:30-16:00 ET. Included for completeness / testing;
    # live US trading actually goes through Alpaca, not this simulator.
    "US": Market(
        code="US",
        currency="USD",
        currency_symbol="$",
        ticker_suffix="",
        timezone="America/New_York",
        open_time=time(9, 30),
        close_time=time(16, 0),
        trading_days=(0, 1, 2, 3, 4),
    ),
}


# ============================================================
# ENGINE
# ============================================================
class PaperTrader:
    """
    Market-agnostic paper-trading engine.

    Args:
        model:          short model id, e.g. "A" (used for labelling only).
        market:         market code present in ``MARKETS`` (e.g. "EGX").
        ledger_path:    JSON ledger file. Defaults to
                        ``model_<model>_<market>_ledger.json`` (lower-cased).
        initial_budget: starting cash. Only used when the ledger is first
                        created; afterwards the persisted value wins.
        name:           human-readable model name stored in the ledger header.
    """

    def __init__(
        self,
        model: str,
        market: str,
        ledger_path: str | None = None,
        initial_budget: float = 1_000_000,
        name: str | None = None,
    ):
        if market not in MARKETS:
            raise ValueError(f"Unknown market {market!r}. Known: {list(MARKETS)}")
        self.model = model
        self.market = MARKETS[market]
        self.name = name or f"Model {model} - {market}"
        self.ledger_path = ledger_path or f"model_{model}_{market}_ledger.json".lower()

        self.initial_budget = float(initial_budget)
        self.cash = float(initial_budget)
        self.positions: dict[str, dict] = {}   # symbol -> {"qty", "avg_price"}
        self.trade_log: list[dict] = []
        self.snapshots: list[dict] = []
        # Free-form per-strategy state that must survive a fresh CI checkout
        # (e.g. MODEL_AI's ATR high-water marks). Persisted with the ledger.
        self.extra: dict = {}

        if os.path.exists(self.ledger_path):
            self._load_state()

    # ---------------------------------------------------------- orders
    def buy(self, symbol: str, qty: int, price: float) -> dict:
        symbol = self.market.localize(symbol)
        cost = qty * price
        if qty <= 0:
            return {"status": "REJECTED", "reason": "qty must be positive"}
        if cost > self.cash:
            return {"status": "REJECTED",
                    "reason": f"Insufficient cash: {self.cash:.2f} < {cost:.2f}"}

        self.cash -= cost
        if symbol in self.positions:
            old = self.positions[symbol]
            total_qty = old["qty"] + qty
            avg = ((old["qty"] * old["avg_price"]) + cost) / total_qty
            self.positions[symbol] = {"qty": total_qty, "avg_price": avg}
        else:
            self.positions[symbol] = {"qty": qty, "avg_price": price}

        trade = {"timestamp": datetime.now(timezone.utc).isoformat(),
                 "side": "BUY", "symbol": symbol, "qty": qty,
                 "price": price, "cost": cost}
        self.trade_log.append(trade)
        print(f"  BUY  {qty}x {symbol} @ {price:.2f} "
              f"= {cost:,.2f} {self.market.currency}")
        return {"status": "FILLED", **trade}

    def sell(self, symbol: str, qty: int, price: float) -> dict:
        symbol = self.market.localize(symbol)
        pos = self.positions.get(symbol)
        if pos is None or pos["qty"] < qty:
            return {"status": "REJECTED",
                    "reason": f"Insufficient shares of {symbol}"}

        proceeds = qty * price
        self.cash += proceeds
        pnl = (price - pos["avg_price"]) * qty
        pos["qty"] -= qty
        if pos["qty"] == 0:
            del self.positions[symbol]

        trade = {"timestamp": datetime.now(timezone.utc).isoformat(),
                 "side": "SELL", "symbol": symbol, "qty": qty,
                 "price": price, "proceeds": proceeds, "pnl": pnl}
        self.trade_log.append(trade)
        print(f"  SELL {qty}x {symbol} @ {price:.2f} "
              f"= {proceeds:,.2f} {self.market.currency} (PnL: {pnl:+,.2f})")
        return {"status": "FILLED", **trade}

    def position_qty(self, symbol: str) -> int:
        """Current held quantity for a symbol (0 if flat)."""
        symbol = self.market.localize(symbol)
        pos = self.positions.get(symbol)
        return int(pos["qty"]) if pos else 0

    # ---------------------------------------------------------- valuation
    def get_portfolio_value(self, current_prices: dict[str, float]) -> float:
        prices = {self.market.localize(k): v for k, v in current_prices.items()}
        positions_value = sum(
            pos["qty"] * prices.get(sym, pos["avg_price"])
            for sym, pos in self.positions.items()
        )
        return self.cash + positions_value

    def snapshot(self, current_prices: dict[str, float]) -> dict:
        """
        Append a dashboard-compatible daily snapshot and persist.

        ``current_prices`` is keyed by symbol (with or without the market
        suffix). Skips duplication if today's snapshot already exists
        (idempotent, matching ``scripts/snapshot_ledger.py``).
        """
        prices = {self.market.localize(k): v for k, v in current_prices.items()}
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        pos_list = []
        for sym, pos in self.positions.items():
            cur = prices.get(sym, pos["avg_price"])
            qty = pos["qty"]
            pos_list.append({
                "symbol": sym,
                "qty": float(qty),
                "avg_entry": round(pos["avg_price"], 4),
                "current_price": round(cur, 4),
                "market_value": round(qty * cur, 2),
                "unrealized_pnl": round((cur - pos["avg_price"]) * qty, 2),
                "side": "long",
            })

        equity = self.cash + sum(p["market_value"] for p in pos_list)
        snap = {
            "date": today,
            "equity": round(equity, 2),
            "cash": round(self.cash, 2),
            "pnl": round(equity - self.initial_budget, 2),
            "positions": pos_list,
        }

        if self.snapshots and self.snapshots[-1]["date"] == today:
            # Replace today's snapshot rather than duplicating it.
            self.snapshots[-1] = snap
        else:
            self.snapshots.append(snap)
        self._save_state()
        return snap

    # ---------------------------------------------------------- clock
    def is_market_open(self, now: datetime | None = None) -> bool:
        """True if ``now`` (default: current time) is within trading hours."""
        tz = ZoneInfo(self.market.timezone)
        now = now.astimezone(tz) if now else datetime.now(tz)
        if now.weekday() not in self.market.trading_days:
            return False
        return self.market.open_time <= now.time() <= self.market.close_time

    # ---------------------------------------------------------- persistence
    def _save_state(self):
        state = {
            "model": self.model,
            "name": self.name,
            "market": self.market.code,
            "currency": self.market.currency,
            "initial_budget": self.initial_budget,
            "cash": self.cash,
            "positions": self.positions,
            "trade_log": self.trade_log,
            "snapshots": self.snapshots,
            "extra": self.extra,
        }
        directory = os.path.dirname(self.ledger_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(self.ledger_path, "w") as f:
            json.dump(state, f, indent=2)

    def _load_state(self):
        with open(self.ledger_path) as f:
            state = json.load(f)
        # initial_budget is authoritative once persisted.
        self.initial_budget = float(state.get("initial_budget", self.initial_budget))
        self.cash = float(state.get("cash", self.initial_budget))
        self.positions = state.get("positions", {})
        self.trade_log = state.get("trade_log", [])
        self.snapshots = state.get("snapshots", [])
        self.extra = state.get("extra", {})
        print(f"Loaded {self.ledger_path}: {self.cash:,.2f} "
              f"{self.market.currency} cash, {len(self.positions)} positions")
