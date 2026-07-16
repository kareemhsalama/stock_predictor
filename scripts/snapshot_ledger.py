"""
Snapshot an Alpaca paper account's portfolio state and append to a JSON ledger.
Runs daily via GitHub Actions after market close.

Defaults to Model A, but is parameterized by env vars so the same script serves
any US Alpaca model (e.g. Model D). Override any of:

    LEDGER_PATH      ledger file (default: data/model_a_ledger.json)
    MODEL_ID         ledger "model" field for a freshly-created ledger (default: A)
    MODEL_NAME       ledger "name" field for a fresh ledger
    INITIAL_BUDGET   P&L baseline for a fresh ledger (default: 100000)
    ALPACA_API_KEY / ALPACA_SECRET_KEY   the account to snapshot

Point LEDGER_PATH + the Alpaca secrets at a second paper account to snapshot
Model D in isolation from Model A.
"""

import json
import os
import sys
import requests
from datetime import datetime, timezone

ALPACA_BASE = "https://paper-api.alpaca.markets"

_DEFAULT_LEDGER = os.path.join(os.path.dirname(__file__), "..", "data", "model_a_ledger.json")
LEDGER_PATH = os.environ.get("LEDGER_PATH", _DEFAULT_LEDGER)

MODEL_ID = os.environ.get("MODEL_ID", "A")
MODEL_NAME = os.environ.get("MODEL_NAME", "Technical ML - Random Forest")

# Alpaca paper accounts are provisioned with $100,000 by default. This is the
# baseline we measure P&L against; the ledger's own initial_budget overrides it
# once the file exists.
DEFAULT_INITIAL_BUDGET = float(os.environ.get("INITIAL_BUDGET", 100000))

try:
    HEADERS = {
        "APCA-API-KEY-ID": os.environ["ALPACA_API_KEY"],
        "APCA-API-SECRET-KEY": os.environ["ALPACA_SECRET_KEY"],
    }
except KeyError as e:
    sys.exit(
        f"Missing required environment variable {e}. "
        "Wire ALPACA_API_KEY / ALPACA_SECRET_KEY as GitHub Actions repo secrets "
        "(Settings > Secrets and variables > Actions), or set them locally."
    )


def get_account():
    resp = requests.get(f"{ALPACA_BASE}/v2/account", headers=HEADERS)
    resp.raise_for_status()
    return resp.json()


def get_positions():
    resp = requests.get(f"{ALPACA_BASE}/v2/positions", headers=HEADERS)
    resp.raise_for_status()
    return resp.json()


def load_ledger():
    if os.path.exists(LEDGER_PATH):
        with open(LEDGER_PATH, "r") as f:
            return json.load(f)
    return {
        "model": MODEL_ID,
        "name": MODEL_NAME,
        "initial_budget": DEFAULT_INITIAL_BUDGET,
        "snapshots": [],
    }


def save_ledger(ledger):
    os.makedirs(os.path.dirname(LEDGER_PATH), exist_ok=True)
    with open(LEDGER_PATH, "w") as f:
        json.dump(ledger, f, indent=2)


def build_snapshot(account, positions, initial_budget):
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    equity = float(account["equity"])
    cash = float(account["cash"])

    pos_list = []
    for p in positions:
        pos_list.append({
            "symbol": p["symbol"],
            "qty": float(p["qty"]),
            "avg_entry": float(p["avg_entry_price"]),
            "current_price": float(p["current_price"]),
            "market_value": float(p["market_value"]),
            "unrealized_pnl": float(p["unrealized_pl"]),
            "side": p["side"],
        })

    return {
        "date": today,
        "equity": equity,
        "cash": cash,
        "pnl": equity - initial_budget,
        "positions": pos_list,
    }


def main():
    account = get_account()
    positions = get_positions()

    ledger = load_ledger()
    initial_budget = ledger.get("initial_budget", DEFAULT_INITIAL_BUDGET)
    snapshot = build_snapshot(account, positions, initial_budget)

    # skip if we already have today's snapshot
    if ledger["snapshots"] and ledger["snapshots"][-1]["date"] == snapshot["date"]:
        print(f"Snapshot for {snapshot['date']} already exists, skipping.")
        return

    ledger["snapshots"].append(snapshot)
    save_ledger(ledger)
    print(f"Snapshot saved: {snapshot['date']} | equity: ${snapshot['equity']:,.2f} | pnl: ${snapshot['pnl']:+,.2f}")


if __name__ == "__main__":
    main()