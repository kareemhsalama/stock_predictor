"""
Snapshot Model A's portfolio state from Alpaca and append to the JSON ledger.
Runs daily via GitHub Actions after market close.
"""

import json
import os
import requests
from datetime import datetime, timezone

ALPACA_BASE = "https://paper-api.alpaca.markets"
LEDGER_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "model_a_ledger.json")

HEADERS = {
    "APCA-API-KEY-ID": os.environ["ALPACA_API_KEY"],
    "APCA-API-SECRET-KEY": os.environ["ALPACA_SECRET_KEY"],
}


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
        "model": "A",
        "name": "Technical ML - Random Forest",
        "initial_budget": 25000,
        "snapshots": [],
    }


def save_ledger(ledger):
    os.makedirs(os.path.dirname(LEDGER_PATH), exist_ok=True)
    with open(LEDGER_PATH, "w") as f:
        json.dump(ledger, f, indent=2)


def build_snapshot(account, positions):
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
        "pnl": equity - 25000,
        "positions": pos_list,
    }


def main():
    account = get_account()
    positions = get_positions()
    snapshot = build_snapshot(account, positions)

    ledger = load_ledger()

    # skip if we already have today's snapshot
    if ledger["snapshots"] and ledger["snapshots"][-1]["date"] == snapshot["date"]:
        print(f"Snapshot for {snapshot['date']} already exists, skipping.")
        return

    ledger["snapshots"].append(snapshot)
    save_ledger(ledger)
    print(f"Snapshot saved: {snapshot['date']} | equity: ${snapshot['equity']:,.2f} | pnl: ${snapshot['pnl']:+,.2f}")


if __name__ == "__main__":
    main()