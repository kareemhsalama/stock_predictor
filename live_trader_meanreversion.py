"""
Model D - US Mean Reversion (live paper trading)
================================================
Live counterpart of the backtest in WIP/mean_reversion.py. Mirrors the
structure of live_trader.py (Model A) but swaps the Random Forest signal for
the z-score mean-reversion rule:

    market clock check -> fetch data -> compute z-score
    -> check current Alpaca position -> buy / sell / hold

Design choices for Model D's "capital preservation" role:
  * long-only (no shorting) — oversold dips only,
  * conservative sizing (RISK_PCT below Model A's 10%),
  * isolated from Model A via its own Alpaca paper account.

Account isolation: set ALPACA_API_KEY_D / ALPACA_SECRET_KEY_D to a SECOND
paper account's keys. If those are unset, it falls back to the primary
ALPACA_API_KEY / ALPACA_SECRET_KEY (fine for a local dry run, but for real
isolation create a separate paper account and wire the _D secrets).
"""

import os

import yfinance as yf
from dotenv import load_dotenv

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

load_dotenv()

# Prefer Model D's dedicated account; fall back to the primary keys.
api_key = os.getenv("ALPACA_API_KEY_D") or os.getenv("ALPACA_API_KEY")
secret_key = os.getenv("ALPACA_SECRET_KEY_D") or os.getenv("ALPACA_SECRET_KEY")

client = TradingClient(api_key, secret_key, paper=True)

# Mean-reversion universe (from WIP/mean_reversion.py).
TICKERS = ["AAPL", "JPM", "KO", "NVDA", "XOM"]

# Signal params — same thresholds as the backtest.
WINDOW = 20
ENTRY_Z = 2.0   # z below -ENTRY_Z => oversold => enter long
EXIT_Z = 0.5    # z above -EXIT_Z  => reverted => take profit
STOP_Z = 3.0    # z below -STOP_Z  => kept falling => stop out

# Conservative sizing for a capital-preservation model (half of Model A's 10%).
RISK_PCT = 0.05


def zscore(ticker):
    """Return (latest z-score, latest close) or (None, None) if data is missing."""
    data = yf.download(ticker, period="1y", multi_level_index=False, progress=False)
    if data is None or data.empty or len(data) < WINDOW + 1:
        print(f"{ticker}: insufficient data")
        return None, None

    close = data["Close"]
    sma = close.rolling(WINDOW).mean()
    std = close.rolling(WINDOW).std()
    z = (close - sma) / std

    latest_z = z.iloc[-1]
    if latest_z != latest_z:  # NaN guard
        return None, None
    return float(latest_z), float(close.iloc[-1])


def current_qty(ticker, positions):
    for p in positions:
        if p.symbol == ticker:
            return int(float(p.qty))
    return 0


def trade(ticker, positions):
    z, price = zscore(ticker)
    if z is None:
        return

    held = current_qty(ticker, positions)

    # Long-only mean reversion (mirrors the pos==1 branch of the backtest).
    if held == 0:
        action = "BUY" if z < -ENTRY_Z else "HOLD"
    else:
        action = "SELL" if (z > -EXIT_Z or z < -STOP_Z) else "HOLD"

    print(f"{ticker}: z={z:+.2f} | price=${price:.2f} | held={held} | {action}")

    if action == "BUY":
        av_cash = float(client.get_account().cash)
        spend = av_cash * RISK_PCT
        qty = int(spend / price)
        if qty <= 0:
            print(f"  {ticker}: BUY skipped — cash slice too small for 1 share")
            return
        order = MarketOrderRequest(symbol=ticker, qty=qty,
                                   side=OrderSide.BUY, time_in_force=TimeInForce.DAY)
        try:
            client.submit_order(order)
            print(f"  BUY {qty}x {ticker} placed")
        except Exception as e:
            print(f"  Order failed: {e}")
    elif action == "SELL":
        order = MarketOrderRequest(symbol=ticker, qty=held,
                                   side=OrderSide.SELL, time_in_force=TimeInForce.DAY)
        try:
            client.submit_order(order)
            print(f"  SELL {held}x {ticker} placed")
        except Exception as e:
            print(f"  Order failed: {e}")


def main():
    clock = client.get_clock()
    if not clock.is_open:
        print("Market is closed.")
        return

    positions = client.get_all_positions()
    for ticker in TICKERS:
        try:
            trade(ticker, positions)
        except Exception as e:
            print(f"Error with {ticker}: {e}")

    print(f"\nPortfolio: ${float(client.get_account().portfolio_value):,.2f}")


if __name__ == "__main__":
    main()
