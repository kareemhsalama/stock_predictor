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
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

ALPACA_BASE = "https://paper-api.alpaca.markets"
NY = ZoneInfo("America/New_York")

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


def resolve_session(now: datetime | None = None) -> str:
    """
    The trading session this run's equity actually belongs to.

    Reads Alpaca's calendar rather than a clock, because both of the obvious
    shortcuts have already lost data here:

      * ``datetime.now(timezone.utc)`` files an evening run under tomorrow's
        date once GitHub delays the cron past 20:00 ET. On 2026-08-06 the job
        ran at 01:03 UTC, stamped itself 2026-08-07, and the next day's
        on-time run then skipped as a duplicate — 08-06 vanished and 08-07's
        real close was never recorded.
      * a weekday check still snapshots on Juneteenth and July 4th, which is
        how 2026-06-19 and 2026-07-03 ended up in Model A's ledger as flat
        rows against a closed market.

    Returns the most recent session that has actually closed, so a delayed run
    lands on the session it measured.
    """
    now = (now or datetime.now(timezone.utc)).astimezone(NY)
    start = (now - timedelta(days=14)).strftime("%Y-%m-%d")
    resp = requests.get(f"{ALPACA_BASE}/v2/calendar", headers=HEADERS,
                        params={"start": start, "end": now.strftime("%Y-%m-%d")})
    resp.raise_for_status()

    closed = []
    for day in resp.json():
        close_h, close_m = (int(x) for x in day["close"].split(":"))
        close_at = datetime.strptime(day["date"], "%Y-%m-%d").replace(
            hour=close_h, minute=close_m, tzinfo=NY)
        if close_at <= now:
            closed.append(day["date"])

    if not closed:
        raise RuntimeError(
            f"No closed session in the 14 days to {now:%Y-%m-%d %H:%M %Z} — "
            "refusing to guess a date")
    return closed[-1]


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


def build_snapshot(account, positions, initial_budget, session):
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
        "date": session,
        "equity": equity,
        "cash": cash,
        "pnl": equity - initial_budget,
        "positions": pos_list,
    }


def main():
    session = resolve_session()
    account = get_account()
    positions = get_positions()

    ledger = load_ledger()
    initial_budget = ledger.get("initial_budget", DEFAULT_INITIAL_BUDGET)
    snapshot = build_snapshot(account, positions, initial_budget, session)

    snapshots = ledger["snapshots"]
    # Replace rather than skip: a re-run of the same session carries fresher
    # data, and skipping is what silently dropped 2026-08-07.
    existing = next((i for i, s in enumerate(snapshots)
                     if s["date"] == snapshot["date"]), None)
    if existing is not None:
        snapshots[existing] = snapshot
        action = "updated"
    else:
        insert_at = len(snapshots)
        while insert_at > 0 and snapshots[insert_at - 1]["date"] > snapshot["date"]:
            insert_at -= 1
        snapshots.insert(insert_at, snapshot)
        action = "saved"

    save_ledger(ledger)
    print(f"Snapshot {action}: {snapshot['date']} | "
          f"equity: ${snapshot['equity']:,.2f} | pnl: ${snapshot['pnl']:+,.2f}")


if __name__ == "__main__":
    main()