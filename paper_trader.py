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
from datetime import datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo


# ============================================================
# EXECUTION COSTS
# ============================================================
@dataclass(frozen=True)
class CostModel:
    """
    What it costs to actually get the trade done.

    Filling an entire order at the marked close for free flatters every result
    the simulator produces, and it flatters EGX most: thin books, a real
    commission floor, and stamp duty on both sides. Three separate effects,
    kept separate so each can be argued with:

      fees        commission (with a per-order floor) + stamp duty, per side.
      spread      you buy at the offer and sell at the bid, never at the mark.
      impact      pushing size through a thin book moves it against you,
                  scaled by the order's participation in average daily value.

    ``max_participation`` is the fourth, and the one that changes behaviour
    rather than just the arithmetic: an order for three days of a name's volume
    does not fill, so it is capped and reported as a partial.
    """

    commission_bps: float = 0.0     # per side, on notional
    min_commission: float = 0.0     # per-order floor, market currency
    stamp_duty_bps: float = 0.0     # per side, on notional
    half_spread_bps: float = 0.0    # cost of crossing the touch, per side
    impact_coef_bps: float = 0.0    # bps of impact at 100% participation
    max_participation: float = 1.0  # cap on order value / ADV (1.0 = no cap)

    def is_free(self) -> bool:
        """True when this model imposes no cost at all (the legacy behaviour)."""
        return not any((self.commission_bps, self.min_commission, self.stamp_duty_bps,
                        self.half_spread_bps, self.impact_coef_bps))

    def fees(self, notional: float) -> float:
        """Commission (floored) + stamp duty on one side of a trade."""
        commission = max(notional * self.commission_bps / 1e4, self.min_commission)
        return commission + notional * self.stamp_duty_bps / 1e4

    def slippage_bps(self, notional: float, adv_value: float | None) -> float:
        """
        Distance from the marked price to the fill, in bps.

        Without an ADV the impact term is unknowable, so it is dropped rather
        than guessed — the fill then carries spread only and the trade log
        records ``adv_known: false`` so the optimism is visible later.
        """
        bps = self.half_spread_bps
        if adv_value and adv_value > 0:
            bps += self.impact_coef_bps * (notional / adv_value)
        return bps

    def max_qty(self, price: float, adv_value: float | None) -> int | None:
        """Largest fillable quantity given the participation cap; None = uncapped."""
        if not adv_value or adv_value <= 0 or self.max_participation >= 1:
            return None
        if price <= 0:
            return 0
        return int((self.max_participation * adv_value) // price)


# EGX: 7.5 bps per side in fees -> 0.15% round trip, the middle of the
# 0.1-0.2% range Egyptian retail brokerage charges. The commission/stamp split
# is nominal; only the total matters here, and it is deliberately at the
# quoted range rather than at the all-in figure a resident actually pays
# (stamp duty alone can run 0.05% a side). Raise both if you want the
# pessimistic case.
#
# The 10 EGP floor is what bites small rebalancing trades. 10 bps of
# half-spread is a fair touch for an EGX 30 constituent, and the impact
# coefficient charges another 10 bps to an order taking 10% of a day's volume.
EGX_COSTS = CostModel(
    commission_bps=5.0,
    min_commission=10.0,
    stamp_duty_bps=2.5,
    half_spread_bps=10.0,
    impact_coef_bps=100.0,
    max_participation=0.20,
)

# US: commission-free at the broker, deep books. Only the simulator uses this —
# live US trading goes through Alpaca, which reports its own fills.
US_COSTS = CostModel(
    half_spread_bps=1.0,
    impact_coef_bps=25.0,
    max_participation=0.10,
)


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
    costs: CostModel = field(default_factory=CostModel)

    def localize(self, symbol: str) -> str:
        """Ensure a bare ticker carries this market's Yahoo suffix."""
        if self.ticker_suffix and not symbol.endswith(self.ticker_suffix):
            return f"{symbol}{self.ticker_suffix}"
        return symbol

    def session_date(self, now: datetime | None = None) -> str:
        """
        The trading session ``now`` belongs to, as YYYY-MM-DD in market-local
        terms.

        Not the UTC date, and that distinction has already cost this repo a
        day of history: on 2026-08-06 a delayed Actions run fired at 01:03 UTC,
        ``datetime.now(timezone.utc)`` stamped it 2026-08-07, and the next
        evening's on-time run skipped as a duplicate. A run before the open
        belongs to the previous trading day, so the delayed run lands on 08-06
        where it belongs.

        Weekday-based, so it does not know about holidays; callers with access
        to a real calendar (``scripts/snapshot_ledger.py`` via Alpaca) should
        prefer that, and ``scripts/check_freshness.py`` flags off-calendar rows
        for those that cannot.
        """
        tz = ZoneInfo(self.timezone)
        now = now.astimezone(tz) if now else datetime.now(tz)
        day = now.date()
        # Before the bell, the session in progress is yesterday's.
        if now.time() < self.open_time:
            day -= timedelta(days=1)
        while day.weekday() not in self.trading_days:
            day -= timedelta(days=1)
        return day.strftime("%Y-%m-%d")


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
        costs=EGX_COSTS,
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
        costs=US_COSTS,
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
        self.fees_paid = 0.0
        self.slippage_paid = 0.0
        # Session date from which fills carry costs. Stamped once, never
        # backdated: trades booked before it were filled free, and a chart
        # that blends the two regimes without saying so is a lie of omission.
        self.cost_model_from: str | None = None

        if os.path.exists(self.ledger_path):
            self._load_state()

        if not self.market.costs.is_free() and not self.cost_model_from:
            self.cost_model_from = self.market.session_date()

    # ---------------------------------------------------------- fills
    def _quote(self, side: str, qty: int, price: float,
               adv_value: float | None) -> dict:
        """
        Price an order without executing it: participation cap, fill price,
        fees. Shared by ``buy``/``sell`` and by ``affordable_qty``, so sizing
        and execution can never disagree about what a trade costs.
        """
        costs = self.market.costs
        requested = qty
        cap = costs.max_qty(price, adv_value)
        if cap is not None:
            qty = min(qty, cap)

        gross = qty * price
        slip_bps = costs.slippage_bps(gross, adv_value)
        sign = 1 if side == "BUY" else -1
        fill_price = price * (1 + sign * slip_bps / 1e4)

        notional = qty * fill_price
        fees = costs.fees(notional) if qty > 0 else 0.0

        return {
            "qty": qty,
            "requested_qty": requested,
            "partial": qty < requested,
            "ref_price": price,
            "fill_price": fill_price,
            "notional": notional,
            "fees": fees,
            "slippage": abs(notional - gross),
            "slippage_bps": slip_bps,
            # Cash leaves on a buy (notional + fees), arrives on a sell.
            "cash_delta": -(notional + fees) if side == "BUY" else (notional - fees),
            "adv_known": bool(adv_value and adv_value > 0),
        }

    def affordable_qty(self, price: float, adv_value: float | None = None,
                       budget: float | None = None) -> int:
        """
        Largest quantity this book can actually buy, costs included.

        Sizing on bare ``cash / price`` was fine when fills were free; with
        fees it lands just over the line and the order is rejected, which would
        show up as a model that mysteriously stops trading. Callers should size
        through this instead.
        """
        budget = self.cash if budget is None else min(budget, self.cash)
        if price <= 0 or budget <= 0:
            return 0

        qty = int(budget // price)
        while qty > 0:
            quote = self._quote("BUY", qty, price, adv_value)
            need = -quote["cash_delta"]
            if need <= budget:
                return quote["qty"]
            # Scale to what the budget covers, always making progress.
            qty = min(qty - 1, int(qty * budget / need))
        return 0

    # ---------------------------------------------------------- orders
    def buy(self, symbol: str, qty: int, price: float,
            adv_value: float | None = None) -> dict:
        """
        Buy at ``price``, adjusted for spread, impact and fees.

        ``adv_value`` is the name's average daily traded *value* in market
        currency. Supplying it enables the impact term and the participation
        cap; omitting it prices the fill on spread alone, which is optimistic.
        """
        symbol = self.market.localize(symbol)
        if qty <= 0:
            return {"status": "REJECTED", "reason": "qty must be positive"}

        quote = self._quote("BUY", qty, price, adv_value)
        if quote["qty"] <= 0:
            return {"status": "REJECTED",
                    "reason": f"{symbol}: too illiquid to fill at "
                              f"{self.market.costs.max_participation:.0%} of ADV"}

        total = -quote["cash_delta"]
        if total > self.cash:
            return {"status": "REJECTED",
                    "reason": f"Insufficient cash: {self.cash:.2f} < {total:.2f} "
                              f"(incl. {quote['fees']:.2f} fees) — size with "
                              f"affordable_qty()"}

        filled = quote["qty"]
        self.cash -= total
        self.fees_paid += quote["fees"]
        self.slippage_paid += quote["slippage"]

        # Cost basis carries the fees, so unrealized P&L is not quietly
        # overstated by the price of getting in.
        basis = quote["notional"] + quote["fees"]
        if symbol in self.positions:
            old = self.positions[symbol]
            total_qty = old["qty"] + filled
            avg = ((old["qty"] * old["avg_price"]) + basis) / total_qty
            self.positions[symbol] = {"qty": total_qty, "avg_price": avg}
        else:
            self.positions[symbol] = {"qty": filled, "avg_price": basis / filled}

        trade = {"timestamp": datetime.now(timezone.utc).isoformat(),
                 "side": "BUY", "symbol": symbol, "qty": filled,
                 "price": round(quote["fill_price"], 4),
                 "ref_price": round(price, 4),
                 "cost": round(total, 4),
                 "fees": round(quote["fees"], 4),
                 "slippage": round(quote["slippage"], 4),
                 "requested_qty": quote["requested_qty"],
                 "adv_known": quote["adv_known"]}
        self.trade_log.append(trade)

        note = f" [partial: wanted {quote['requested_qty']}]" if quote["partial"] else ""
        print(f"  BUY  {filled}x {symbol} @ {quote['fill_price']:.2f} "
              f"(mark {price:.2f}) = {total:,.2f} {self.market.currency} "
              f"incl. {quote['fees']:,.2f} fees{note}")
        return {"status": "FILLED", **trade, "partial": quote["partial"]}

    def sell(self, symbol: str, qty: int, price: float,
             adv_value: float | None = None) -> dict:
        """
        Sell at ``price``, adjusted for spread, impact and fees.

        A position too large for the book exits partially and keeps the
        remainder — the honest outcome, and the one that lets a stop-loss
        finish the job over the following sessions.
        """
        symbol = self.market.localize(symbol)
        pos = self.positions.get(symbol)
        if pos is None or pos["qty"] < qty:
            return {"status": "REJECTED",
                    "reason": f"Insufficient shares of {symbol}"}

        quote = self._quote("SELL", qty, price, adv_value)
        if quote["qty"] <= 0:
            return {"status": "REJECTED",
                    "reason": f"{symbol}: too illiquid to fill at "
                              f"{self.market.costs.max_participation:.0%} of ADV"}

        filled = quote["qty"]
        proceeds = quote["cash_delta"]
        self.cash += proceeds
        self.fees_paid += quote["fees"]
        self.slippage_paid += quote["slippage"]

        pnl = proceeds - pos["avg_price"] * filled
        pos["qty"] -= filled
        if pos["qty"] == 0:
            del self.positions[symbol]

        trade = {"timestamp": datetime.now(timezone.utc).isoformat(),
                 "side": "SELL", "symbol": symbol, "qty": filled,
                 "price": round(quote["fill_price"], 4),
                 "ref_price": round(price, 4),
                 "proceeds": round(proceeds, 4),
                 "fees": round(quote["fees"], 4),
                 "slippage": round(quote["slippage"], 4),
                 "pnl": round(pnl, 4),
                 "requested_qty": quote["requested_qty"],
                 "adv_known": quote["adv_known"]}
        self.trade_log.append(trade)

        note = f" [partial: wanted {quote['requested_qty']}]" if quote["partial"] else ""
        print(f"  SELL {filled}x {symbol} @ {quote['fill_price']:.2f} "
              f"(mark {price:.2f}) = {proceeds:,.2f} {self.market.currency} "
              f"net of {quote['fees']:,.2f} fees (PnL: {pnl:+,.2f}){note}")
        return {"status": "FILLED", **trade, "partial": quote["partial"]}

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
        suffix). Filed under the market session it belongs to, not the UTC
        date, and a re-run replaces that session rather than duplicating it
        (idempotent, matching ``scripts/snapshot_ledger.py``).
        """
        prices = {self.market.localize(k): v for k, v in current_prices.items()}
        today = self.market.session_date()

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

        # Replace this session rather than duplicating it, and keep the series
        # sorted even if a delayed run reports an older session than the last
        # row — an out-of-order date would silently corrupt every return
        # computed off consecutive pairs.
        existing = next((i for i, s in enumerate(self.snapshots)
                         if s["date"] == today), None)
        if existing is not None:
            self.snapshots[existing] = snap
        else:
            insert_at = len(self.snapshots)
            while insert_at > 0 and self.snapshots[insert_at - 1]["date"] > today:
                insert_at -= 1
            self.snapshots.insert(insert_at, snap)

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
        costs = self.market.costs
        state = {
            "model": self.model,
            "name": self.name,
            "market": self.market.code,
            "currency": self.market.currency,
            "initial_budget": self.initial_budget,
            "cash": self.cash,
            "positions": self.positions,
            "fees_paid": round(self.fees_paid, 4),
            "slippage_paid": round(self.slippage_paid, 4),
            "cost_model_from": self.cost_model_from,
            # Persisted so a past result can be reproduced against the exact
            # cost assumptions it was produced under.
            "cost_model": None if costs.is_free() else {
                "commission_bps": costs.commission_bps,
                "min_commission": costs.min_commission,
                "stamp_duty_bps": costs.stamp_duty_bps,
                "half_spread_bps": costs.half_spread_bps,
                "impact_coef_bps": costs.impact_coef_bps,
                "max_participation": costs.max_participation,
            },
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
        self.fees_paid = float(state.get("fees_paid", 0.0))
        self.slippage_paid = float(state.get("slippage_paid", 0.0))
        self.cost_model_from = state.get("cost_model_from")
        print(f"Loaded {self.ledger_path}: {self.cash:,.2f} "
              f"{self.market.currency} cash, {len(self.positions)} positions")
