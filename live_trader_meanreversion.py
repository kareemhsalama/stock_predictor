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
from datetime import datetime, timezone

import yfinance as yf
from dotenv import load_dotenv

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

from telegram_notifier import TelegramNotifier

load_dotenv()

# Prefer Model D's dedicated account; fall back to the primary keys.
api_key = os.getenv("ALPACA_API_KEY_D") or os.getenv("ALPACA_API_KEY")
secret_key = os.getenv("ALPACA_SECRET_KEY_D") or os.getenv("ALPACA_SECRET_KEY")

client = TradingClient(api_key, secret_key, paper=True)

try:
    notifier = TelegramNotifier()
except ValueError as e:
    print(f"Telegram alerts disabled: {e}")
    notifier = None

# Mean-reversion universe. Widened from the backtest's original 5 names
# (AAPL/JPM/KO/NVDA/XOM) because that universe only produced ~4 entries per
# ticker per year at ENTRY_Z=2.0 — over a 3y sweep, 63 entries total, which
# left the model sitting in cash for weeks at a stretch and made the dashboard
# panel look dead. Twelve liquid large caps across sectors give ~160 entries
# over the same 3y window at the SAME threshold; loosening ENTRY_Z to 1.5
# instead would have bought activity at the cost of in-trade Sharpe
# (2.50 -> 1.87), so the universe is what changed, not the signal.
TICKERS = [
    "AAPL", "MSFT", "NVDA",   # tech
    "JPM", "BAC",             # financials
    "KO", "PG", "JNJ", "WMT", # staples / healthcare
    "XOM", "CVX",             # energy
    "HD",                     # discretionary
]

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


def trade(ticker, positions, trades_today=None):
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
            trade_record = {
                "symbol": ticker, "side": "BUY", "qty": qty, "price": price,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "reason": f"z-score {z:+.2f} < entry threshold -{ENTRY_Z}",
            }
            if trades_today is not None:
                trades_today.append(trade_record)
            if notifier:
                try:
                    notifier.send_trade_alert(trade_record)
                except Exception as e:
                    print(f"  Telegram alert failed: {e}")
        except Exception as e:
            print(f"  Order failed: {e}")
    elif action == "SELL":
        order = MarketOrderRequest(symbol=ticker, qty=held,
                                   side=OrderSide.SELL, time_in_force=TimeInForce.DAY)
        try:
            client.submit_order(order)
            print(f"  SELL {held}x {ticker} placed")
            reason = (f"z-score {z:+.2f} reverted above -{EXIT_Z}" if z > -EXIT_Z
                     else f"z-score {z:+.2f} breached stop -{STOP_Z}")
            trade_record = {
                "symbol": ticker, "side": "SELL", "qty": held, "price": price,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "reason": reason,
            }
            if trades_today is not None:
                trades_today.append(trade_record)
            if notifier:
                try:
                    notifier.send_trade_alert(trade_record)
                except Exception as e:
                    print(f"  Telegram alert failed: {e}")
        except Exception as e:
            print(f"  Order failed: {e}")


def main():
    clock = client.get_clock()
    if not clock.is_open:
        print("Market is closed.")
        return

    trades_today = []
    try:
        positions = client.get_all_positions()
        for ticker in TICKERS:
            try:
                trade(ticker, positions, trades_today=trades_today)
            except Exception as e:
                print(f"Error with {ticker}: {e}")

        print(f"\nPortfolio: ${float(client.get_account().portfolio_value):,.2f}")
    finally:
        if notifier:
            try:
                account = client.get_account()
                equity = float(account.portfolio_value)
                last_equity = getattr(account, "last_equity", None)
                pnl = ({"daily_pnl": equity - float(last_equity)}
                      if last_equity is not None else None)
                notifier.send_daily_summary(trades_today, pnl=pnl, equity=equity)
            except Exception as e:
                print(f"Telegram daily summary failed: {e}")


if __name__ == "__main__":
    main()
