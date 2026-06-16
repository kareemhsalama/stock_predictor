<<<<<<< HEAD
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

from dotenv import load_dotenv
import os

load_dotenv()

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

api_key = os.getenv("ALPACA_API_KEY")
secret_key = os.getenv("ALPACA_SECRET_KEY")


from alpaca.trading.client import TradingClient

client = TradingClient(api_key, secret_key, paper=True)

def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def live_trade(ticker, threshold=0.55, risk_pct=0.10):
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

    data["Tomorrow_Return"] = data["Close"].shift(-1).div(data["Close"]).sub(1)
    data["Target"] = data["Tomorrow_Return"] > 0 
    data["Target"] = data["Target"].astype(int)
    data = data.dropna()

    features = ["Return", "Momentum_5", "Momentum_10", "Momentum_20", "Volatility", "RSI", "Price_vs_MA", "Volume_Change", "MA_cross", "Distance_from_high", "Volatility_change", "Bollinger_position", "Volume_vs_avg", "Day_of_week"]
    x = data[features]
    y = data["Target"]

    x = x.iloc[:-1]
    y = y.iloc[:-1]

    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)    
    model.fit(x, y)

    today = data[features].iloc[[-1]]

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
        curr_price = float(data["Close"].iloc[-1])
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


live_trade("AAPL")
=======
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

from dotenv import load_dotenv
import os

load_dotenv()

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

api_key = os.getenv("ALPACA_API_KEY")
secret_key = os.getenv("ALPACA_SECRET_KEY")


from alpaca.trading.client import TradingClient

client = TradingClient(api_key, secret_key, paper=True)

def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def live_trade(ticker, threshold=0.55, risk_pct=0.10):
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

    data["Tomorrow_Return"] = data["Close"].shift(-1).div(data["Close"]).sub(1)
    data["Target"] = data["Tomorrow_Return"] > 0 
    data["Target"] = data["Target"].astype(int)
    data = data.dropna()

    features = ["Return", "Momentum_5", "Momentum_10", "Momentum_20", "Volatility", "RSI", "Price_vs_MA", "Volume_Change", "MA_cross", "Distance_from_high", "Volatility_change", "Bollinger_position", "Volume_vs_avg", "Day_of_week"]
    x = data[features]
    y = data["Target"]

    x = x.iloc[:-1]
    y = y.iloc[:-1]

    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)    
    model.fit(x, y)

    today = data[features].iloc[[-1]]

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
        curr_price = float(data["Close"].iloc[-1])
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


live_trade("AAPL")
>>>>>>> de7ab959d2f3f2b553fad595ca68218d23d31d41
