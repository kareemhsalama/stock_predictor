"""
EGX (Egyptian Exchange) Data & Paper Trading Example
=====================================================
Pull Egyptian stock data with yfinance, run a mean reversion backtest,
and simulate paper trading — all without Alpaca.

Ticker format: Yahoo Finance uses ISIN-style codes with .CA suffix
  e.g. COMI.CA (CIB Egypt), TMGH.CA (Talaat Moustafa), HRHO.CA (Hermes)

The EGX 30 index: ^CASE30
Trading hours: Sun–Thu, 10:00–14:15 EET (Cairo time)
Currency: EGP (Egyptian Pound)
"""

import yfinance as yf
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timedelta


# ============================================================
# 1. EGX TICKER MAPPING
# ============================================================
# Yahoo uses ISIN.CA — here's a handy mapping for the top EGX 30 stocks.
# You can scrape the full list from stockanalysis.com/list/egyptian-stock-exchange/

EGX_TICKERS = {
    "CIB":              "COMI.CA",
    "Talaat Moustafa":  "TMGH.CA",
    "EFG Hermes":       "HRHO.CA",
    "Eastern Company":  "EAST.CA",
    "Fawry":            "FWRY.CA",
    "Orascom Const":    "ORAS.CA",
    "Palm Hills":       "PHDC.CA",
    "Edita Food":       "EFID.CA",
    "Abu Qir Fert":     "ABUK.CA",
    "Elsewedy Elec":    "SWDY.CA",
    "EGX 30 Index":     "^CASE30",
}


# ============================================================
# 2. PULL HISTORICAL DATA
# ============================================================
def fetch_egx_data(ticker: str, period: str = "1y") -> pd.DataFrame:
    """
    Fetch OHLCV data for an EGX stock.
    Works exactly like your US stock pulls — same yfinance API.

    Args:
        ticker: Yahoo Finance ticker (e.g. "COMI.CA" or "^CASE30")
        period: "1mo", "3mo", "6mo", "1y", "2y", "5y", "max"
    """
    stock = yf.Ticker(ticker)
    df = stock.history(period=period)

    if df.empty:
        print(f"WARNING: No data returned for {ticker}")
        return df

    # Clean up — drop the multi-index if present (same fix as your US project)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Keep only what we need
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.index = df.index.tz_localize(None)  # strip timezone for consistency

    print(f"[{ticker}] Fetched {len(df)} rows | "
          f"{df.index[0].date()} → {df.index[-1].date()} | "
          f"Last close: {df['Close'].iloc[-1]:.2f} EGP")
    return df


def fetch_multiple(tickers: dict, period: str = "1y") -> dict[str, pd.DataFrame]:
    """Fetch data for multiple EGX stocks at once."""
    data = {}
    for name, ticker in tickers.items():
        try:
            df = fetch_egx_data(ticker, period)
            if not df.empty:
                data[name] = df
        except Exception as e:
            print(f"ERROR fetching {name} ({ticker}): {e}")
    return data


# ============================================================
# 3. MEAN REVERSION STRATEGY (same approach as your Model D)
# ============================================================
def mean_reversion_signals(
    df: pd.DataFrame,
    lookback: int = 20,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
    stop_z: float = 3.0,
) -> pd.DataFrame:
    """
    Generate mean reversion signals using z-scores.
    Same logic as your Model D — Bollinger-style with z-score thresholds.
    """
    df = df.copy()
    df["SMA"] = df["Close"].rolling(lookback).mean()
    df["STD"] = df["Close"].rolling(lookback).std()
    df["Z_Score"] = (df["Close"] - df["SMA"]) / df["STD"]

    # Signal generation: same 3-state loop as Model D
    df["Signal"] = 0  # 0 = flat, 1 = long, -1 = short
    position = 0

    for i in range(lookback, len(df)):
        z = df["Z_Score"].iloc[i]

        if position == 0:  # flat
            if z < -entry_z:
                position = 1   # oversold → go long
            elif z > entry_z:
                position = -1  # overbought → go short
        elif position == 1:  # long
            if z > -exit_z or z < -stop_z:
                position = 0   # exit
        elif position == -1:  # short
            if z < exit_z or z > stop_z:
                position = 0   # exit

        df.iloc[i, df.columns.get_loc("Signal")] = position

    # CRITICAL: shift forward 1 day to avoid look-ahead bias
    df["Position"] = df["Signal"].shift(1)
    df["Returns"] = df["Close"].pct_change()
    df["Strategy_Returns"] = df["Position"] * df["Returns"]

    return df.dropna()


# ============================================================
# 4. PAPER TRADING SIMULATOR (replaces Alpaca for EGX)
# ============================================================
class EGXPaperTrader:
    """
    A simple paper trading engine for EGX stocks.
    Since Alpaca doesn't support EGX, this simulates order fills
    using delayed/EOD prices from yfinance.

    Tracks: cash, positions, trade history, daily snapshots.
    Stores state in a JSON ledger (same pattern as your dashboard).
    """

    def __init__(self, initial_cash: float = 1_000_000, ledger_path: str = "egx_ledger.json"):
        self.initial_cash = initial_cash
        self.ledger_path = ledger_path
        self.cash = initial_cash
        self.positions: dict[str, dict] = {}  # ticker -> {qty, avg_price}
        self.trade_log: list[dict] = []
        self.snapshots: list[dict] = []

        # Load existing state if ledger exists
        if os.path.exists(ledger_path):
            self._load_state()

    def buy(self, ticker: str, qty: int, price: float) -> dict:
        """Execute a paper buy order."""
        cost = qty * price
        if cost > self.cash:
            return {"status": "REJECTED", "reason": f"Insufficient cash: {self.cash:.2f} < {cost:.2f}"}

        self.cash -= cost

        if ticker in self.positions:
            old = self.positions[ticker]
            total_qty = old["qty"] + qty
            avg = ((old["qty"] * old["avg_price"]) + cost) / total_qty
            self.positions[ticker] = {"qty": total_qty, "avg_price": avg}
        else:
            self.positions[ticker] = {"qty": qty, "avg_price": price}

        trade = {
            "timestamp": datetime.now().isoformat(),
            "side": "BUY",
            "ticker": ticker,
            "qty": qty,
            "price": price,
            "cost": cost,
        }
        self.trade_log.append(trade)
        print(f"  BUY  {qty}x {ticker} @ {price:.2f} EGP = {cost:,.2f} EGP")
        return {"status": "FILLED", **trade}

    def sell(self, ticker: str, qty: int, price: float) -> dict:
        """Execute a paper sell order."""
        if ticker not in self.positions or self.positions[ticker]["qty"] < qty:
            return {"status": "REJECTED", "reason": f"Insufficient shares of {ticker}"}

        proceeds = qty * price
        self.cash += proceeds

        pos = self.positions[ticker]
        pnl = (price - pos["avg_price"]) * qty
        pos["qty"] -= qty
        if pos["qty"] == 0:
            del self.positions[ticker]

        trade = {
            "timestamp": datetime.now().isoformat(),
            "side": "SELL",
            "ticker": ticker,
            "qty": qty,
            "price": price,
            "proceeds": proceeds,
            "pnl": pnl,
        }
        self.trade_log.append(trade)
        print(f"  SELL {qty}x {ticker} @ {price:.2f} EGP = {proceeds:,.2f} EGP (PnL: {pnl:+,.2f})")
        return {"status": "FILLED", **trade}

    def get_portfolio_value(self, current_prices: dict[str, float]) -> float:
        """Calculate total portfolio value (cash + positions at market price)."""
        positions_value = sum(
            pos["qty"] * current_prices.get(ticker, pos["avg_price"])
            for ticker, pos in self.positions.items()
        )
        return self.cash + positions_value

    def snapshot(self, current_prices: dict[str, float]) -> dict:
        """Take a daily snapshot — same idea as your snapshot_ledger.py script."""
        total = self.get_portfolio_value(current_prices)
        snap = {
            "timestamp": datetime.now().isoformat(),
            "cash": round(self.cash, 2),
            "positions": {
                t: {"qty": p["qty"], "avg_price": round(p["avg_price"], 2),
                    "market_price": current_prices.get(t, p["avg_price"]),
                    "value": round(p["qty"] * current_prices.get(t, p["avg_price"]), 2)}
                for t, p in self.positions.items()
            },
            "total_value": round(total, 2),
            "total_return_pct": round((total / self.initial_cash - 1) * 100, 2),
        }
        self.snapshots.append(snap)
        self._save_state()
        return snap

    def _save_state(self):
        state = {
            "initial_cash": self.initial_cash,
            "cash": self.cash,
            "positions": self.positions,
            "trade_log": self.trade_log,
            "snapshots": self.snapshots,
        }
        with open(self.ledger_path, "w") as f:
            json.dump(state, f, indent=2)

    def _load_state(self):
        with open(self.ledger_path) as f:
            state = json.load(f)
        self.cash = state["cash"]
        self.positions = state["positions"]
        self.trade_log = state["trade_log"]
        self.snapshots = state["snapshots"]
        print(f"Loaded state: {self.cash:,.2f} EGP cash, {len(self.positions)} positions")


# ============================================================
# 5. PERFORMANCE METRICS (same as your dashboard)
# ============================================================
def calc_performance(returns: pd.Series) -> dict:
    """Calculate Sharpe, max drawdown, win rate — mirrors your dashboard logic."""
    total_return = (1 + returns).prod() - 1
    sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
    cumulative = (1 + returns).cumprod()
    drawdown = (cumulative / cumulative.cummax()) - 1
    max_drawdown = drawdown.min()
    win_rate = (returns > 0).sum() / (returns != 0).sum() if (returns != 0).sum() > 0 else 0

    return {
        "total_return": f"{total_return:.2%}",
        "sharpe_ratio": f"{sharpe:.2f}",
        "max_drawdown": f"{max_drawdown:.2%}",
        "win_rate": f"{win_rate:.2%}",
        "trading_days": len(returns),
    }


# ============================================================
# 6. PUTTING IT ALL TOGETHER
# ============================================================
def run_backtest_example():
    """Full example: pull CIB data, run mean reversion, show results."""
    print("=" * 60)
    print("EGX MEAN REVERSION BACKTEST — CIB Egypt (COMI.CA)")
    print("=" * 60)

    # Pull data
    df = fetch_egx_data("COMI.CA", period="2y")
    if df.empty:
        print("No data — check your internet connection and ticker")
        return

    # Run strategy
    results = mean_reversion_signals(df, lookback=20, entry_z=2.0, exit_z=0.5, stop_z=3.0)

    # Performance
    strat_perf = calc_performance(results["Strategy_Returns"])
    bnh_perf = calc_performance(results["Returns"])

    print(f"\n{'Metric':<20} {'Strategy':>15} {'Buy & Hold':>15}")
    print("-" * 50)
    for key in strat_perf:
        print(f"{key:<20} {strat_perf[key]:>15} {bnh_perf[key]:>15}")

    # Count trades
    position_changes = results["Position"].diff().abs()
    n_trades = int(position_changes.sum() / 2)
    print(f"\nTotal round-trip trades: {n_trades}")

    return results


def run_paper_trading_example():
    """Example: simulate live paper trading with the EGX paper trader."""
    print("\n" + "=" * 60)
    print("EGX PAPER TRADING SIMULATION")
    print("=" * 60)

    # Init with 1M EGP (roughly $20k USD)
    trader = EGXPaperTrader(initial_cash=1_000_000, ledger_path="egx_ledger.json")

    # Fetch latest prices
    tickers_to_trade = {"COMI.CA": "CIB", "TMGH.CA": "Talaat Moustafa", "SWDY.CA": "Elsewedy"}
    prices = {}
    for ticker, name in tickers_to_trade.items():
        df = fetch_egx_data(ticker, period="5d")
        if not df.empty:
            prices[ticker] = df["Close"].iloc[-1]
            print(f"  {name}: {prices[ticker]:.2f} EGP")

    if not prices:
        print("No prices available — can't simulate")
        return

    # Simulate some trades
    print("\n--- Executing trades ---")

    # Buy 100 shares of CIB
    if "COMI.CA" in prices:
        trader.buy("COMI.CA", qty=100, price=prices["COMI.CA"])

    # Buy 200 shares of Talaat Moustafa
    if "TMGH.CA" in prices:
        trader.buy("TMGH.CA", qty=200, price=prices["TMGH.CA"])

    # Take a snapshot
    snap = trader.snapshot(prices)
    print(f"\n--- Portfolio Snapshot ---")
    print(f"  Cash:        {snap['cash']:>15,.2f} EGP")
    print(f"  Total Value: {snap['total_value']:>15,.2f} EGP")
    print(f"  Return:      {snap['total_return_pct']:>14.2f}%")

    # Simulate selling CIB later at a higher price
    print("\n--- Next day: CIB rallies 3% ---")
    new_price = prices.get("COMI.CA", 100) * 1.03
    trader.sell("COMI.CA", qty=100, price=new_price)

    snap2 = trader.snapshot({"TMGH.CA": prices.get("TMGH.CA", 50), "COMI.CA": new_price})
    print(f"\n--- Updated Portfolio ---")
    print(f"  Cash:        {snap2['cash']:>15,.2f} EGP")
    print(f"  Total Value: {snap2['total_value']:>15,.2f} EGP")
    print(f"  Return:      {snap2['total_return_pct']:>14.2f}%")
    print(f"\n  Ledger saved to: {trader.ledger_path}")


# ============================================================
# 7. GITHUB ACTIONS SNAPSHOT (for your daily pipeline)
# ============================================================
SNAPSHOT_WORKFLOW = """
# .github/workflows/egx_snapshot.yml
# Same pattern as your US snapshot — runs at market close Cairo time
# EGX closes at 14:15 EET, so schedule ~14:30 EET = 12:30 UTC

name: EGX Daily Snapshot
on:
  schedule:
    - cron: '30 12 * * 0-4'  # Sun-Thu at 12:30 UTC (14:30 Cairo)
  workflow_dispatch:

permissions:
  contents: write

jobs:
  snapshot:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - run: pip install yfinance pandas numpy
      - run: python scripts/egx_snapshot.py
      - run: |
          git config user.name "github-actions[bot]"
          git config user.email "github-actions[bot]@users.noreply.github.com"
          git add data/egx_ledger.json
          git diff --staged --quiet || git commit -m "EGX snapshot $(date -u +%Y-%m-%d)"
          git push
"""


if __name__ == "__main__":
    # Run the backtest
    results = run_backtest_example()

    # Run the paper trading sim
    run_paper_trading_example()

    # Print the GitHub Actions workflow for reference
    print("\n" + "=" * 60)
    print("GITHUB ACTIONS WORKFLOW (copy to .github/workflows/)")
    print("=" * 60)
    print(SNAPSHOT_WORKFLOW)