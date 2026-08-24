"""
telegram_notifier.py

Drop-in module for sending trade alerts and a daily summary to Telegram.

Usage:
    from telegram_notifier import TelegramNotifier

    notifier = TelegramNotifier()  # reads TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID from env

    # Fire immediately whenever your model executes a trade:
    notifier.send_trade_alert({
        "symbol": "AAPL",
        "side": "BUY",
        "qty": 10,
        "price": 231.42,
        "reason": "momentum signal crossed threshold",
    })

    # Fire once at the end of the day's run:
    notifier.send_daily_summary(trades_today, pnl={"daily_pnl": 128.40, "daily_pnl_pct": 0.64})

Design notes:
- Never raises on network failure. A Telegram outage should not be able to crash
  your trading loop. Failures are logged; send() returns None on failure.
- Reads credentials from environment variables by default so the token never
  needs to live in code or in version control.
"""

import os
import logging
from datetime import datetime

import requests

logger = logging.getLogger(__name__)

TELEGRAM_API_URL = "https://api.telegram.org/bot{token}/sendMessage"


class TelegramNotifier:
    def __init__(self, token=None, chat_id=None, timeout=10):
        self.token = token or os.environ.get("TELEGRAM_BOT_TOKEN")
        self.chat_id = chat_id or os.environ.get("TELEGRAM_CHAT_ID")
        self.timeout = timeout

        if not self.token or not self.chat_id:
            raise ValueError(
                "Missing Telegram credentials. Set TELEGRAM_BOT_TOKEN and "
                "TELEGRAM_CHAT_ID as environment variables (see .env.example)."
            )

        self.url = TELEGRAM_API_URL.format(token=self.token)

    def send(self, text, parse_mode="Markdown", retries=3):
        """Low-level send. Returns the parsed JSON response, or None on failure."""
        payload = {"chat_id": self.chat_id, "text": text, "parse_mode": parse_mode}

        for attempt in range(1, retries + 1):
            try:
                resp = requests.post(self.url, data=payload, timeout=self.timeout)
                resp.raise_for_status()
                return resp.json()
            except requests.RequestException as exc:
                logger.warning(
                    "Telegram send failed (attempt %d/%d): %s", attempt, retries, exc
                )
                if attempt == retries:
                    logger.error("Giving up on sending Telegram message after %d attempts.", retries)
                    return None

    def send_trade_alert(self, trade):
        """
        trade: dict with keys:
            symbol (str), side ("BUY"/"SELL"), qty (number), price (number)
            optional: timestamp (str/datetime), reason (str)
        """
        side = str(trade["side"]).upper()
        emoji = "\U0001F7E2" if side == "BUY" else "\U0001F534"  # green / red circle
        timestamp = trade.get("timestamp") or datetime.utcnow().isoformat(timespec="seconds")

        text = (
            f"{emoji} *{side}* `{trade['symbol']}`\n"
            f"Qty: {trade['qty']}\n"
            f"Price: {float(trade['price']):.4f}\n"
            f"Time: {timestamp}"
        )
        if trade.get("reason"):
            text += f"\nSignal: {trade['reason']}"

        return self.send(text)

    def send_daily_summary(self, trades, pnl=None, equity=None, date_str=None):
        """
        trades: list of trade dicts for the day (same shape as send_trade_alert)
        pnl: optional dict, e.g. {"daily_pnl": 128.40, "daily_pnl_pct": 0.64}
        equity: optional current portfolio equity/balance
        date_str: optional override for the date shown in the header
        """
        date_str = date_str or datetime.utcnow().strftime("%Y-%m-%d")

        if not trades:
            text = f"\U0001F4CB *Daily Summary — {date_str}*\nNo trades executed today."
            return self.send(text)

        buys = [t for t in trades if str(t["side"]).upper() == "BUY"]
        sells = [t for t in trades if str(t["side"]).upper() == "SELL"]

        lines = [
            f"\U0001F4CB *Daily Summary — {date_str}*",
            "",
            f"Trades: {len(trades)}  (Buys: {len(buys)}, Sells: {len(sells)})",
        ]
        for t in trades:
            side = str(t["side"]).upper()
            emoji = "\U0001F7E2" if side == "BUY" else "\U0001F534"
            lines.append(f"{emoji} {side} {t['qty']} `{t['symbol']}` @ {float(t['price']):.4f}")

        if pnl:
            lines.append("")
            daily_pnl = pnl.get("daily_pnl")
            daily_pnl_pct = pnl.get("daily_pnl_pct")
            pnl_line = "P&L today:"
            if daily_pnl is not None:
                pnl_line += f" {daily_pnl:+.2f}"
            if daily_pnl_pct is not None:
                pnl_line += f" ({daily_pnl_pct:+.2f}%)"
            lines.append(pnl_line)

        if equity is not None:
            lines.append(f"Equity: {equity:.2f}")

        return self.send("\n".join(lines))