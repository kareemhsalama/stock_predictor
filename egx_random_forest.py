"""
EGX Model A - Technical ML (Random Forest)
==========================================
The EGX counterpart of the US ``live_trader.py``. Same Random Forest, same
14-feature technical set, but:

  * data comes from yfinance ``.CA`` tickers (Egyptian Exchange), and
  * execution goes through the ``PaperTrader`` simulator (EGX has no Alpaca),
    writing an isolated ledger at ``data/model_a_egx_ledger.json``.

Runs end-of-day (after the 14:15 EET close): fetch history, predict tomorrow's
up-probability, buy/sell/hold at the latest close, then snapshot the ledger.
A day with no data (holiday / illiquid) simply no-ops for that ticker.
"""

import os

import numpy as np
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier

from paper_trader import PaperTrader

# EGX universe to trade (subset of EGX_TICKERS in egy.py).
EGX_TICKERS = ["COMI.CA", "TMGH.CA", "SWDY.CA"]

LEDGER_PATH = os.path.join(os.path.dirname(__file__), "data", "model_a_egx_ledger.json")

# ~1M EGP starting budget (roughly $20k USD), mirroring egy.py's simulation.
INITIAL_BUDGET = 1_000_000

THRESHOLD = 0.55   # up-probability required to hold/enter
RISK_PCT = 0.10    # fraction of available cash per new position

FEATURES = [
    "Return", "Momentum_5", "Momentum_10", "Momentum_20", "Volatility", "RSI",
    "Price_vs_MA", "Volume_Change", "MA_cross", "Distance_from_high",
    "Volatility_change", "Bollinger_position", "Volume_vs_avg", "Day_of_week",
]


def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def engineer_features(data):
    """Add the shared 14-feature technical set + Target (matches live_trader.py)."""
    data["Return"] = data["Close"].pct_change()
    data["Momentum_5"] = data["Close"].pct_change(5)
    data["Momentum_10"] = data["Close"].pct_change(10)
    data["Momentum_20"] = data["Close"].pct_change(20)
    data["Volatility"] = data["Close"].rolling(20).std()
    data["RSI"] = calculate_rsi(data["Close"])
    data["20_day_avg"] = data["Close"].rolling(20).mean()
    data["50_day_avg"] = data["Close"].rolling(50).mean()
    data["200_day_avg"] = data["Close"].rolling(200).mean()
    data["Price_vs_MA"] = (data["Close"] - data["50_day_avg"]) / data["50_day_avg"]
    data["Volume_Change"] = data["Volume"].pct_change()
    data["MA_cross"] = (data["50_day_avg"] - data["200_day_avg"]) / data["200_day_avg"]
    data["High"] = data["Close"].rolling(252).max()
    data["Distance_from_high"] = (data["Close"] - data["High"]) / data["High"]
    data["Volatility_change"] = data["Volatility"].pct_change()
    data["Bollinger_position"] = (data["Close"] - data["20_day_avg"]) / (2 * data["Volatility"])
    data["Volume_vs_avg"] = data["Volume"] / data["Volume"].rolling(20).mean()
    data["Day_of_week"] = data.index.dayofweek

    data["Tomorrow_Return"] = data["Close"].shift(-1).div(data["Close"]).sub(1)
    data["Target"] = (data["Tomorrow_Return"] > 0).astype(int)
    return data


# Window for average daily traded value, which sets the fill's impact term and
# the participation cap in PaperTrader. 20 sessions is a month of EGX trading —
# long enough to survive one dead day, short enough to track a drying-up name.
ADV_WINDOW = 20


def predict_signal(ticker):
    """
    Train an RF on `ticker`'s history and predict tomorrow's up-probability.

    Returns (confidence, last_close, adv_value), or (None, None, None) if there
    isn't enough usable data (holiday / illiquid / delisted). ``adv_value`` is
    average daily traded value in EGP.
    """
    data = yf.download(ticker, period="2y", multi_level_index=False,
                       auto_adjust=True, progress=False)
    if data is None or data.empty:
        print(f"[{ticker}] no data — skipping")
        return None, None, None

    # Defensive: collapse a MultiIndex if yfinance still returns one.
    if hasattr(data.columns, "nlevels") and data.columns.nlevels > 1:
        data.columns = data.columns.get_level_values(0)

    last_close = float(data["Close"].iloc[-1])
    adv_value = float((data["Close"] * data["Volume"])
                      .tail(ADV_WINDOW).mean())
    if not np.isfinite(adv_value) or adv_value <= 0:
        adv_value = None

    data = engineer_features(data)
    # EGX has zero-volume days, so volume-based ratios can yield ±inf; drop
    # those alongside the warm-up NaNs before training.
    data = data.replace([np.inf, -np.inf], np.nan).dropna()
    if len(data) < 50:
        print(f"[{ticker}] only {len(data)} usable rows — skipping")
        return None, last_close, adv_value

    # Train on everything except the final (still-open) row, exactly like
    # live_trader.py: the last row's Target is unknown until tomorrow.
    x = data[FEATURES].iloc[:-1]
    y = data["Target"].iloc[:-1]

    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(x, y)

    today = data[FEATURES].iloc[[-1]]
    confidence = float(model.predict_proba(today)[0, 1])
    adv_note = f"{adv_value:,.0f} EGP" if adv_value else "unknown"
    print(f"[{ticker}] confidence = {confidence:.1%} | last close = "
          f"{last_close:.2f} EGP | ADV = {adv_note}")
    return confidence, last_close, adv_value


def run():
    trader = PaperTrader(
        model="A",
        market="EGX",
        ledger_path=LEDGER_PATH,
        initial_budget=INITIAL_BUDGET,
        name="Technical ML - Random Forest (EGX)",
    )

    prices = {}
    for ticker in EGX_TICKERS:
        try:
            confidence, price, adv_value = predict_signal(ticker)
        except Exception as e:
            print(f"[{ticker}] error: {e}")
            continue
        if price is not None:
            prices[ticker] = price
        if confidence is None or price is None:
            continue

        held_qty = trader.position_qty(ticker)

        if confidence > THRESHOLD and held_qty == 0:
            # Size through affordable_qty so fees come out of the slice rather
            # than pushing the order past the cash balance and getting it
            # rejected at the boundary.
            qty = trader.affordable_qty(price, adv_value,
                                        budget=trader.cash * RISK_PCT)
            if qty > 0:
                trader.buy(ticker, qty, price, adv_value)
            else:
                print(f"[{ticker}] BUY skipped — cash slice too small for 1 share")
        elif confidence <= THRESHOLD and held_qty > 0:
            trader.sell(ticker, held_qty, price, adv_value)
        else:
            action = "HOLD" if held_qty > 0 else "STAY FLAT"
            print(f"[{ticker}] {action}")

    snap = trader.snapshot(prices)
    print(f"\n--- EGX Model A snapshot {snap['date']} ---")
    print(f"  Equity: {snap['equity']:,.2f} EGP | "
          f"Cash: {snap['cash']:,.2f} EGP | PnL: {snap['pnl']:+,.2f} EGP")
    print(f"  Costs to date: {trader.fees_paid:,.2f} EGP fees + "
          f"{trader.slippage_paid:,.2f} EGP slippage "
          f"(applied from {trader.cost_model_from})")
    print(f"  Open positions: {len(snap['positions'])}")
    print(f"  Ledger: {trader.ledger_path}")


if __name__ == "__main__":
    run()
