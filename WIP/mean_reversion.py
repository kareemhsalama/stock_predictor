import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def calculate_signals(ticker, window=20, entry_z=2.0, exit_z=0.5,stop_z=3.0):
    data = yf.download(ticker, period="1y", multi_level_index=False)
    data["SMA"] = data["Close"].rolling(window).mean()
    data["STD"] = data["Close"].rolling(window).std()
    data["z_score"] = (data["Close"] - data["SMA"])/data["STD"]
    
    data["daily_return"] = data["Close"].pct_change()
    positions = []
    pos = 0

    for i in range(len(data)):
        z = data["z_score"].iloc[i]

        if pos == 0:
            if z < -entry_z:
                pos = 1
            elif z > entry_z:
                pos = -1

        elif pos == 1:
            if z > -exit_z:
                pos = 0
            elif z < -stop_z:
                pos = 0

        elif pos == -1:
            if z < exit_z:
                pos = 0
            elif z > stop_z:
                pos = 0

        positions.append(pos)
    data["position"] = positions
    data["strategy_return"] = data["position"].shift(1) * data["daily_return"]
    data.dropna(inplace=True)

    total_return = (1 + data["strategy_return"]).cumprod().iloc[-1] - 1
    win_rate = (data.loc[data["position"] != 0, "strategy_return"] > 0).mean()

    cumulative = (1 + data["strategy_return"]).cumprod()
    max_drawdown = (cumulative / cumulative.cummax() - 1).min()
    
    fig, ax = plt.subplots(figsize=(14, 7))

    ax.plot(data.index, (1 + data["daily_return"]).cumprod(), label="Buy & Hold", linewidth=1.5)
    ax.plot(data.index, (1 + data["strategy_return"]).cumprod(), label="Mean Reversion", linewidth=1.5)

    ax.axhline(y=1, color="gray", linestyle="--", linewidth=0.5)
    ax.set_title(f"Mean Reversion vs Buy & Hold  |  Return: {total_return:.2%}  |  Win Rate: {win_rate:.2%}  |  Max DD: {max_drawdown:.2%}")
    ax.set_ylabel("Growth of $1")
    ax.legend()
    plt.tight_layout()
    plt.show()

tickers = ["AAPL","JPM","KO","NVDA","XOM"]
for ticker in tickers:
    calculate_signals(ticker)