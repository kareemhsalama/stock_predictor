import yfinance as yf
import pandas as pd

from dotenv import load_dotenv
import os
from datetime import datetime, timezone

load_dotenv()

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

from telegram_notifier import TelegramNotifier

api_key = os.getenv("ALPACA_API_KEY")
secret_key = os.getenv("ALPACA_SECRET_KEY")


from alpaca.trading.client import TradingClient

client = TradingClient(api_key, secret_key, paper=True)

try:
    notifier = TelegramNotifier()
except ValueError as e:
    print(f"Telegram alerts disabled: {e}")
    notifier = None

def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def live_trade(ticker, threshold=0.55, risk_pct=0.10, trades_today=None):
    clock = client.get_clock()
    if not clock.is_open:
        print("Market is closed.")
        return
    
    data = yf.download(ticker, period="2y")
    data.columns = data.columns.droplevel(1)
    data["Return"] = data["Close"].pct_change()
    data["Momentum_5"] = data["Close"].pct_change(5)
    data["Momentum_10"] = data["Close"].pct_change(10)
    data["Momentum_20"] = data["Close"].pct_change(20)
    data["Volatility"] = data["Close"].rolling(20).std()
    data["RSI"] = calculate_rsi(data["Close"])
    data["20_day_avg"] = data["Close"].rolling(20).mean()
    data["50_day_avg"] = data["Close"].rolling(50).mean()
    data["200_day_avg"] = data["Close"].rolling(200).mean()
    data["Price_vs_MA"] = (data["Close"] - data["50_day_avg"])/data["50_day_avg"]
    data["Volume_Change"] = data["Volume"].pct_change()
    data["MA_cross"] = (data["50_day_avg"] - data["200_day_avg"])/data["200_day_avg"]
    data["High"] = data["Close"].rolling(252).max()
    data["Distance_from_high"] = (data["Close"]-data["High"])/data["High"]
    data["Volatility_change"] = data["Volatility"].pct_change()
    data["Bollinger_position"] = (data["Close"]-data["20_day_avg"])/(2*data["Volatility"])
    data["Volume_vs_avg"] = data["Volume"]/data["Volume"].rolling(20).mean()
    data["Day_of_week"] = data.index.dayofweek

    features = ["Return", "Momentum_5", "Momentum_10", "Momentum_20", "Volatility", "RSI", "Price_vs_MA", "Volume_Change", "MA_cross", "Distance_from_high", "Volatility_change", "Bollinger_position", "Volume_vs_avg", "Day_of_week"]

    # Capture today's row and price before the Target-driven dropna below.
    # Tomorrow_Return (shift(-1)) is NaN on the most recent row by
    # construction - tomorrow hasn't happened - so a plain dropna() always
    # drops it, and today/curr_price were silently coming from yesterday's
    # row instead on every run.
    today = data[features].iloc[[-1]].replace([float("inf"), float("-inf")], float("nan"))
    if today.isna().any(axis=None):
        print(f"{ticker}: today's features contain NaN/inf - skipping")
        return
    curr_price = float(data["Close"].iloc[-1])

    data["Tomorrow_Return"] = data["Close"].shift(-1).div(data["Close"]).sub(1)
    data["Target"] = data["Tomorrow_Return"] > 0
    data["Target"] = data["Target"].astype(int)
    data = data.dropna()

    # dropna() above already removes the one row with an unknown Target (the
    # row captured as `today`), so every remaining row is fully labelled -
    # no further trimming needed.
    x = data[features]
    y = data["Target"]

    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(x, y)

    confidence = model.predict_proba(today)[0,1]
    print(f"{ticker}: Model confidence = {confidence:.1%}")

    positions = client.get_all_positions()
    owned_symbols = [p.symbol for p in positions]

    already_own = ticker in owned_symbols

    current_qty = 0
    for p in positions:
        if p.symbol == ticker:
            current_qty = int(p.qty)

    if confidence > threshold and not already_own:
        av_cash = float(client.get_account().cash)
        spend = av_cash * risk_pct
        quantity = int(spend / curr_price)
        if quantity > 0:
            order = MarketOrderRequest(
                symbol=ticker,
                qty=quantity,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY
            )  
            try:
                client.submit_order(order)
                print(f"Order placed!")
                trade_record = {
                    "symbol": ticker, "side": "BUY", "qty": quantity,
                    "price": curr_price,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "reason": f"RF confidence {confidence:.1%} > threshold {threshold:.1%}",
                }
                if trades_today is not None:
                    trades_today.append(trade_record)
                if notifier:
                    try:
                        notifier.send_trade_alert(trade_record)
                    except Exception as e:
                        print(f"Telegram alert failed: {e}")
            except Exception as e:
                print(f"Order failed: {e}")
    elif confidence <= threshold and already_own:
        order = MarketOrderRequest(
            symbol=ticker,
            qty=current_qty,
            side=OrderSide.SELL,
            time_in_force=TimeInForce.DAY
            )
        try:
            client.submit_order(order)
            print(f"Order placed!")
            trade_record = {
                "symbol": ticker, "side": "SELL", "qty": current_qty,
                "price": curr_price,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "reason": f"RF confidence {confidence:.1%} <= threshold {threshold:.1%}",
            }
            if trades_today is not None:
                trades_today.append(trade_record)
            if notifier:
                try:
                    notifier.send_trade_alert(trade_record)
                except Exception as e:
                    print(f"Telegram alert failed: {e}")
        except Exception as e:
            print(f"Order failed: {e}")
    else:
        print(f"No action for {ticker}")
    print(f"\n--- Summary ---")
    print(f"Ticker:     {ticker}")
    print(f"Confidence: {confidence:.1%}")
    print(f"Threshold:  {threshold:.1%}")
    action = "BUY" if confidence > threshold and not already_own else "SELL" if confidence <= threshold and already_own else "HOLD"
    print(f"Action:     {action}")
    print(f"Portfolio:  ${float(client.get_account().portfolio_value):,.2f}")


if __name__ == "__main__":
    tickers = ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA"]
    trades_today = []
    try:
        for t in tickers:
            try:
                live_trade(t, trades_today=trades_today)
            except Exception as e:
                print(f"Error with {t}: {e}")
    finally:
        if notifier:
            try:
                account = client.get_account()
                equity = float(account.portfolio_value)
                # last_equity is the prior session's closing equity (a
                # standard Alpaca account field, not a new calculation) -
                # lets the summary show a real daily P&L rather than
                # mislabeling cumulative equity as "today's" change.
                last_equity = getattr(account, "last_equity", None)
                pnl = ({"daily_pnl": equity - float(last_equity)}
                      if last_equity is not None else None)
                notifier.send_daily_summary(trades_today, pnl=pnl, equity=equity)
            except Exception as e:
                print(f"Telegram daily summary failed: {e}")
