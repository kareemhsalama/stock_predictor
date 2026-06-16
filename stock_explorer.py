import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

from dotenv import load_dotenv
import os

load_dotenv(r"C:\Users\DR-HUSSAM\Desktop\Broker\.env")

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

api_key = os.getenv("ALPACA_API_KEY")
secret_key = os.getenv("ALPACA_SECRET_KEY")


from alpaca.trading.client import TradingClient

client = TradingClient(api_key, secret_key, paper=True)

# Download aapl data for a year
def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def dashboard(ticker, period="1y"):
    # Download and fix columns
    data = yf.download(ticker, period=period)
    data.columns = data.columns.droplevel(1)
    
    # Calculate indicators
    data["MA_50"] = data["Close"].rolling(50).mean()
    data["MA_200"] = data["Close"].rolling(200).mean()
    data["RSI"] = calculate_rsi(data["Close"])
    
    # Build 3-panel chart
    fig, axes = plt.subplots(3, 1, figsize=(12, 10),
                              gridspec_kw={"height_ratios": [3, 1, 1]})
    fig.suptitle(f"{ticker} Dashboard", fontsize=16, fontweight="bold")
    
    # Panel 1: Price + MAs
    axes[0].plot(data["Close"], label="Price", alpha=0.7)
    axes[0].plot(data["MA_50"], label="50-day MA", linewidth=2)
    axes[0].plot(data["MA_200"], label="200-day MA", linewidth=2)
    axes[0].set_ylabel("Price ($)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Panel 2: Volume
    colors = ["green" if c >= o else "red" 
              for c, o in zip(data["Close"], data["Open"])]
    axes[1].bar(data.index, data["Volume"], color=colors, alpha=0.5)
    axes[1].set_ylabel("Volume")
    axes[1].grid(True, alpha=0.3)
    
    # Panel 3: RSI
    axes[2].plot(data["RSI"], color="purple")
    axes[2].axhline(70, color="red", linestyle="--", alpha=0.5)
    axes[2].axhline(30, color="green", linestyle="--", alpha=0.5)
    axes[2].fill_between(data.index, 70, 100, alpha=0.05, color="red")
    axes[2].fill_between(data.index, 0, 30, alpha=0.05, color="green")
    axes[2].set_ylabel("RSI")
    axes[2].set_ylim(0, 100)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def predict(ticker, period="5y", n_estimators=100, max_depth=5):
    data = yf.download(ticker, period=period)
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

    split = int(0.8 * len(data))

    x_train = x[:split]
    x_test = x[split:]

    y_train = y[:split]
    y_test = y[split:]

    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)    
    model.fit(x_train, y_train)

    predictions = model.predict_proba(x_test)

    accuracy = accuracy_score(y_test, predictions)
    return accuracy
    # print(f"Accuracy: {accuracy:.1%}")

    # print(classification_report(y_test, predictions, target_names=["Down", "Up"]))
    # chart = pd.Series(model.feature_importances_, index=features)
    # chart = chart.sort_values(ascending=True)
    # chart.plot(kind="barh", title=ticker)
    # plt.show()
tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "JPM", "BAC", "JNJ", "PFE", "WMT", "KO", "PG", "XOM", "CVX", "DIS", "NFLX", "NVDA", "AMD", "META", "V"]

def backtest(ticker, period="5y", n_estimators=100, max_depth=5, threshold=0.6, initial_cash=10000):
    data = yf.download(ticker, period=period)
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

    split = int(0.8 * len(data))

    x_train = x[:split]
    x_test = x[split:]

    y_train = y[:split]
    y_test = y[split:]

    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)    
    model.fit(x_train, y_train)

    probabilities = model.predict_proba(x_test)
    up_proba = probabilities[:, 1]
    predictions = (up_proba > threshold).astype(int)
    actual_returns = data["Return"].iloc[split:].values
    strategy_returns = predictions * actual_returns

    strategy_curve = pd.Series((1+strategy_returns).cumprod())
    buyhold_curve = pd.Series((1+actual_returns).cumprod())

    strategy_curve = initial_cash * strategy_curve
    buyhold_curve = initial_cash * buyhold_curve

    total_return = strategy_curve.iloc[-1] / initial_cash - 1
    bh_return = buyhold_curve.iloc[-1] / initial_cash - 1
    num_trades = predictions.sum()
    trade_returns = strategy_returns[predictions == 1]
    win_rate = (trade_returns > 0).sum() / len(trade_returns)
    running_max = strategy_curve.cummax()
    max_drawdown = (strategy_curve / running_max - 1).min()

    print(f"\n{ticker} Backtest Results")
    print(f"Strategy Return:   {total_return:.1%}")
    print(f"Buy & Hold Return: {bh_return:.1%}")
    print(f"Trades:            {num_trades} / {len(predictions)} days")
    print(f"Win Rate:          {win_rate:.1%}")
    print(f"Max Drawdown:      {max_drawdown:.1%}")

    plt.figure(figsize=(12, 6))
    plt.plot(strategy_curve.values, label="Model Strategy")
    plt.plot(buyhold_curve.values, label="Buy & Hold")
    plt.title(f"{ticker} Backtest: Model vs Buy & Hold")
    plt.ylabel("Portfolio Value ($)")
    plt.xlabel("Trading Days")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

