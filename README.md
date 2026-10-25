# stock_predictor

A project where I test trading strategies with paper money. Several models trade US stocks and Egyptian (EGX) stocks every day, and the results show up on a public dashboard next to the S&P 500.

Live dashboard: https://kareemhsalama.github.io/stock_predictor/

This is paper trading only. No real money is involved.

## The models

- **Model A**: a random forest that looks at 14 technical indicators (momentum, RSI, moving averages and so on) and predicts whether a stock goes up the next day.
- **Model D**: a mean reversion strategy for US stocks.
- **Model AI**: a momentum strategy that ranks stocks against each other and uses stop losses and limits on position size.
- **EGX versions** of Model A and Model AI for the Egyptian Exchange.

The US models trade on an Alpaca paper account. The Egyptian ones run on a small paper-trading engine in this repo (`paper_trader.py`).

## How it runs

GitHub Actions starts the trading scripts on a schedule. Every day the results are saved to JSON files in `data/`, and the dashboard (in `docs/`, hosted on GitHub Pages) reads them. A health check opens a GitHub issue if data goes missing or looks stale, and closes it again once it's fixed. If you set up Telegram, every trade is also sent there.

## Folders

- `data/`: the results (ledgers) for each model
- `docs/`: the dashboard
- `research/`: log of the experiments I ran
- `scripts/`: helper scripts for snapshots and checks
- `tests/`: tests
- the Python files in the root are the models and traders

## Run it yourself

```bash
git clone https://github.com/kareemhsalama/stock_predictor.git
cd stock_predictor
pip install -r requirements.txt pytest
pytest
```

To run a trader, create a `.env` file with your Alpaca paper trading keys:

```
ALPACA_API_KEY=your_key
ALPACA_SECRET_KEY=your_secret
```

Telegram alerts are optional (`TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID`). Then run, for example:

```bash
python live_trader.py
```

It only places paper orders, and only while the market is open. The files in `.github/workflows/` show how each script is run.

## Notes

- Market data comes from yfinance.
- The results are from a short time window, so don't read too much into them.
- Not financial advice.
- Built with help from Claude Code.
