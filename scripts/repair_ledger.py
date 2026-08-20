"""
Repair a US ledger against Alpaca's own portfolio history.

Built for the 2026-08-06 damage, but general: it reconciles a ledger's daily
equity series against the broker's record and fixes three specific defects.

  relabel        A snapshot stamped with the UTC date when a delayed run
                 crossed midnight UTC belongs to the previous session. On
                 2026-08-06 the job fired at 01:03 UTC on the 7th, so the row
                 labelled 2026-08-07 actually holds the 08-06 close - cash,
                 positions and all. Moving the label keeps a real capture in
                 the record instead of throwing it away.

  backfill       Sessions Alpaca has and the ledger does not are inserted with
                 the broker's closing equity. Position-level detail for those
                 days is not recoverable from portfolio history, so the row
                 carries positions: [] and a `backfilled` marker rather than
                 inventing a book.

  off-calendar   Rows dated to days the market never traded (2026-06-19,
                 2026-07-03 in Model A). They contribute a spurious flat day
                 to every daily-return statistic - Sharpe, volatility, win
                 rate - so they are dropped, with --keep-off-calendar to
                 override.

Dry-run by default. Nothing is written without --apply.

    python scripts/repair_ledger.py --ledger data/model_a_ledger.json
    python scripts/repair_ledger.py --ledger data/model_a_ledger.json --apply

Credentials come from ALPACA_API_KEY / ALPACA_SECRET_KEY, so point them at the
account that owns the ledger being repaired.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import requests

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ALPACA_BASE = "https://paper-api.alpaca.markets"
NY = ZoneInfo("America/New_York")


def headers() -> dict:
    try:
        return {
            "APCA-API-KEY-ID": os.environ["ALPACA_API_KEY"],
            "APCA-API-SECRET-KEY": os.environ["ALPACA_SECRET_KEY"],
        }
    except KeyError as e:
        sys.exit(f"Missing {e}. Set the keys for the account owning this ledger.")


def alpaca_equity(period: str = "3M") -> dict[str, float]:
    """
    Closing equity per session, keyed by session date.

    Alpaca stamps each daily point at 00:00 UTC of the *following* day (20:00
    ET of the session itself), so the timestamp is converted to New York before
    the date is taken. Verified against the ledgers' own captures: same-date
    agreement averages a few tens of dollars, while shifting by one session
    averages an order of magnitude worse.
    """
    resp = requests.get(f"{ALPACA_BASE}/v2/account/portfolio/history",
                        headers=headers(),
                        params={"period": period, "timeframe": "1D",
                                "intraday_reporting": "market_hours",
                                "pnl_reset": "no_reset"},
                        timeout=30)
    resp.raise_for_status()
    data = resp.json()
    return {
        datetime.fromtimestamp(ts, timezone.utc).astimezone(NY).strftime("%Y-%m-%d"):
            float(equity)
        for ts, equity in zip(data["timestamp"], data["equity"])
    }


def trading_sessions(start: str, end: str) -> list[str]:
    resp = requests.get(f"{ALPACA_BASE}/v2/calendar", headers=headers(),
                        params={"start": start, "end": end}, timeout=30)
    resp.raise_for_status()
    return [day["date"] for day in resp.json()]


def repair(ledger_path: str, apply: bool, keep_off_calendar: bool) -> int:
    full = os.path.join(REPO_ROOT, ledger_path)
    with open(full) as f:
        ledger = json.load(f)

    snapshots = ledger["snapshots"]
    if not snapshots:
        print(f"{ledger_path}: no snapshots, nothing to repair")
        return 0

    first, last = snapshots[0]["date"], snapshots[-1]["date"]
    sessions = trading_sessions(first, last)
    session_set = set(sessions)
    equity_by_date = alpaca_equity()
    initial_budget = ledger.get("initial_budget", 100_000)

    actions: list[str] = []

    # --- 1. relabel rows misdated by the UTC-crossing bug.
    #
    # The signature is narrow on purpose: a row sitting on session D whose
    # immediately preceding session P is missing, where the row's equity
    # matches the broker's P far better than its own D. Both guards matter.
    #
    # The missing set is frozen before the pass, so relabelling D->P cannot
    # make D look "missing" and drag the next row backwards behind it. An
    # earlier version without that freeze walked Model A's whole tail back a
    # session each, turning one hole into three wrong dates.
    dates = {s["date"] for s in snapshots}
    originally_missing = {s for s in sessions if s not in dates}
    position = {session: i for i, session in enumerate(sessions)}

    # Only a decisively better match moves a row. Near-ties are noise - the
    # ledger captures at 17:30-18:35 ET and Alpaca marks at 20:00, so a few
    # tens of dollars of after-hours drift is normal - and a wrong relabel is
    # worse than an honest gap.
    DECISIVE = 0.5

    for snap in snapshots:
        date = snap["date"]
        idx = position.get(date)
        if not idx:                      # unknown session, or the very first
            continue
        prior = sessions[idx - 1]
        if prior not in originally_missing or prior not in equity_by_date:
            continue

        drift_here = abs(snap["equity"] - equity_by_date.get(date, snap["equity"]))
        drift_prior = abs(snap["equity"] - equity_by_date[prior])
        if drift_prior > drift_here * DECISIVE:
            actions.append(f"leave {date} in place - equity {snap['equity']:,.2f} "
                           f"does not clearly belong to {prior} "
                           f"(drift {drift_prior:,.2f} vs {drift_here:,.2f}); "
                           f"{prior} will be backfilled instead")
            continue

        actions.append(f"relabel {date} -> {prior} "
                       f"(equity {snap['equity']:,.2f}; Alpaca {prior}="
                       f"{equity_by_date[prior]:,.2f}, {date}="
                       f"{equity_by_date.get(date, float('nan')):,.2f})")
        snap["date"] = prior
        dates.discard(date)
        dates.add(prior)

    # --- 2. backfill sessions Alpaca has that the ledger is missing.
    for session in sessions:
        if session in dates or session not in equity_by_date:
            continue
        equity = equity_by_date[session]
        snapshots.append({
            "date": session,
            "equity": round(equity, 2),
            "cash": None,
            "pnl": round(equity - initial_budget, 2),
            "positions": [],
            "backfilled": "alpaca-portfolio-history",
            "note": "equity restored from the broker; position detail for this "
                    "session is not recoverable",
        })
        dates.add(session)
        actions.append(f"backfill {session} equity={equity:,.2f}")

    # --- 3. drop rows dated to days the market never traded.
    if not keep_off_calendar:
        for snap in list(snapshots):
            if snap["date"] not in session_set and snap["date"] <= last:
                snapshots.remove(snap)
                actions.append(f"drop off-calendar row {snap['date']} "
                               f"(equity {snap['equity']:,.2f}, market closed)")

    snapshots.sort(key=lambda s: s["date"])

    print(f"\n=== {ledger_path} ===")
    if not actions:
        print("  nothing to repair")
        return 0
    for action in actions:
        print(f"  {action}")

    if apply:
        with open(full, "w") as f:
            json.dump(ledger, f, indent=2)
        print(f"  -> written ({len(snapshots)} snapshots)")
    else:
        print("  -> dry run, nothing written (pass --apply)")
    return len(actions)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", required=True, help="path to the ledger JSON")
    parser.add_argument("--apply", action="store_true", help="write the changes")
    parser.add_argument("--keep-off-calendar", action="store_true",
                        help="leave rows dated to non-trading days in place")
    args = parser.parse_args()
    repair(args.ledger, args.apply, args.keep_off_calendar)


if __name__ == "__main__":
    main()
