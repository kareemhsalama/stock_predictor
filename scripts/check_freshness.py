"""
Pipeline heartbeat: does every ledger still have the days it should?

Written after 2026-08-06 went missing from all three US ledgers and nobody
noticed for two weeks. The failure had two independent causes on the same day —
the trade jobs were cancelled while queued, and a delayed snapshot run stamped
itself with the UTC date, so the session was filed as 08-07 and the next day's
run skipped it as a duplicate. Both are invisible from the dashboard, which
happily draws a confident line through a hole.

What this checks, per ledger:

  gaps          sessions the market traded but the ledger has no row for. This
                is the one that catches 08-06: the *tail* looked healthy the
                next morning, so staleness alone would have said nothing.
  staleness     how far the newest row lags the newest closed session.
  off-calendar  rows dated to days the market never traded (2026-06-19 and
                2026-07-03 are sitting in Model A's ledger right now).
  ordering      duplicate or out-of-order dates, which corrupt every return
                computed from consecutive pairs.

and for the benchmark store:

  ^GSPC staleness, plus ^CASE30 accrual — that series has no history to
  backfill from, so a missed run is a permanent hole in it.

The market calendar comes from data/benchmarks.json: those series *are* the
days each market actually traded, which beats a hand-maintained holiday list
and stays correct by itself.

Exit code 1 if anything is wrong, so CI fails loudly.

    python scripts/check_freshness.py [--max-lag 1] [--report report.md]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timedelta, timezone

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BENCH_PATH = os.path.join(REPO_ROOT, "data", "benchmarks.json")

# Which benchmark series supplies each market's trading calendar. EGX uses the
# equal-weight composite rather than EGX30: the real index series only started
# accruing on 2026-08-20 and carries no history to check older rows against.
CALENDARS = {"US": "SP500", "EGX": "EGX30_EW"}

LEDGERS = [
    {"label": "Model A",        "path": "data/model_a_ledger.json",      "market": "US"},
    {"label": "Model D",        "path": "data/model_d_ledger.json",      "market": "US"},
    {"label": "Model AI",       "path": "data/model_ai_ledger.json",     "market": "US"},
    # ASCII labels: this report is printed to consoles whose encoding we do
    # not control, and a mangled character in an alert is a distraction.
    {"label": "Model A EGX",    "path": "data/model_a_egx_ledger.json",  "market": "EGX"},
    {"label": "Model AI EGX",   "path": "data/model_ai_egx_ledger.json", "market": "EGX"},
]

# How far the calendar itself may lag before ledger checks become unreliable.
# Four days covers a normal weekend plus a holiday.
CALENDAR_STALE_DAYS = 4


class Problem:
    """One thing that is wrong, with enough detail to act on without digging."""

    def __init__(self, subject: str, kind: str, detail: str):
        self.subject = subject
        self.kind = kind
        self.detail = detail

    def __str__(self):
        return f"[{self.kind}] {self.subject}: {self.detail}"


# ============================================================
# Loading
# ============================================================
def load_calendars() -> dict[str, list[str]]:
    if not os.path.exists(BENCH_PATH):
        return {}
    with open(BENCH_PATH) as f:
        benchmarks = json.load(f).get("benchmarks", {})
    calendars = {}
    for market, key in CALENDARS.items():
        series = benchmarks.get(key, {}).get("series", [])
        calendars[market] = sorted(p["date"] for p in series)
    return calendars


def load_snapshot_dates(path: str) -> list[str]:
    with open(os.path.join(REPO_ROOT, path)) as f:
        ledger = json.load(f)
    return [s["date"] for s in ledger.get("snapshots", [])]


# ============================================================
# Checks
# ============================================================
def check_ledger(spec: dict, calendar: list[str], max_lag: int) -> list[Problem]:
    label, path = spec["label"], spec["path"]
    problems = []

    full_path = os.path.join(REPO_ROOT, path)
    if not os.path.exists(full_path):
        return [Problem(label, "missing", f"{path} does not exist")]

    dates = load_snapshot_dates(path)
    if not dates:
        return [Problem(label, "empty", f"{path} has no snapshots")]

    if dates != sorted(dates):
        problems.append(Problem(label, "ordering", "snapshot dates are not ascending"))
    duplicates = sorted({d for d in dates if dates.count(d) > 1})
    if duplicates:
        problems.append(Problem(label, "ordering", f"duplicate dates: {duplicates}"))

    if not calendar:
        problems.append(Problem(label, "no-calendar",
                                "no market calendar available — cannot verify"))
        return problems

    known = set(dates)
    first, last = dates[0], dates[-1]

    # Interior gaps: sessions inside the ledger's own window with no row.
    expected = [d for d in calendar if first <= d <= last]
    gaps = [d for d in expected if d not in known]
    if gaps:
        shown = ", ".join(gaps[:10]) + (f" (+{len(gaps) - 10} more)" if len(gaps) > 10 else "")
        problems.append(Problem(label, "gap",
                                f"{len(gaps)} session(s) missing inside "
                                f"{first}..{last}: {shown}"))

    # Rows dated to days the market never traded.
    calendar_set = set(calendar)
    off = [d for d in dates if d not in calendar_set and d <= calendar[-1]]
    if off:
        problems.append(Problem(label, "off-calendar",
                                f"{len(off)} row(s) on non-trading days: "
                                + ", ".join(off[:10])))

    # Tail staleness, counted in sessions rather than calendar days so a
    # weekend or a holiday never raises a false alarm.
    behind = [d for d in calendar if d > last]
    if len(behind) > max_lag:
        problems.append(Problem(label, "stale",
                                f"newest row is {last}, {len(behind)} session(s) "
                                f"behind (latest: {calendar[-1]})"))
    return problems


def check_benchmarks(max_lag: int) -> list[Problem]:
    problems = []
    if not os.path.exists(BENCH_PATH):
        return [Problem("benchmarks.json", "missing",
                        "no benchmark store — dashboard comparisons are blank")]

    with open(BENCH_PATH) as f:
        store = json.load(f)
    benchmarks = store.get("benchmarks", {})
    today = datetime.now(timezone.utc).date()

    for key in ("SP500", "EGX30", "EGX30_EW"):
        series = benchmarks.get(key, {}).get("series", [])
        if not series:
            problems.append(Problem(key, "empty", "series has no points"))
            continue
        last = series[-1]["date"]
        age = (today - datetime.strptime(last, "%Y-%m-%d").date()).days
        # ^CASE30 cannot be backfilled: whatever a missed run skips is gone.
        limit = CALENDAR_STALE_DAYS if key != "EGX30" else CALENDAR_STALE_DAYS + 1
        if age > limit:
            problems.append(Problem(key, "stale",
                                    f"newest point is {last} ({age} days old)"
                                    + (" — ^CASE30 has no history to backfill "
                                       "from, so this gap is permanent"
                                       if key == "EGX30" else "")))
    return problems


def check_calendars(calendars: dict[str, list[str]]) -> list[Problem]:
    """A stale calendar makes every ledger look healthy. Check it first."""
    problems = []
    today = datetime.now(timezone.utc).date()
    for market, dates in calendars.items():
        if not dates:
            problems.append(Problem(f"{market} calendar", "missing",
                                    f"benchmarks.json has no {CALENDARS[market]} series"))
            continue
        age = (today - datetime.strptime(dates[-1], "%Y-%m-%d").date()).days
        if age > CALENDAR_STALE_DAYS:
            problems.append(Problem(f"{market} calendar", "stale",
                                    f"newest session is {dates[-1]} ({age} days old) "
                                    "— ledger checks below are unreliable"))
    return problems


# ============================================================
# Reporting
# ============================================================
def build_report(problems: list[Problem], summaries: list[str]) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [f"## Pipeline health - {now}", ""]

    if not problems:
        lines.append("All ledgers current, no gaps, no off-calendar rows.")
    else:
        lines.append(f"**{len(problems)} problem(s) found.**")
        lines.append("")
        lines.append("| Subject | Kind | Detail |")
        lines.append("|---|---|---|")
        for p in problems:
            lines.append(f"| {p.subject} | `{p.kind}` | {p.detail} |")

    lines += ["", "### Ledger state", "", "```"] + summaries + ["```"]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-lag", type=int,
                        default=int(os.environ.get("MAX_LAG_SESSIONS", 1)),
                        help="sessions a ledger may lag before it counts as stale")
    parser.add_argument("--report", help="write a markdown report to this path")
    args = parser.parse_args()

    calendars = load_calendars()
    problems = check_calendars(calendars)

    summaries = []
    for spec in LEDGERS:
        problems += check_ledger(spec, calendars.get(spec["market"], []), args.max_lag)
        try:
            dates = load_snapshot_dates(spec["path"])
            summaries.append(f"{spec['label']:16s} {len(dates):3d} snapshots  "
                             f"{dates[0]} -> {dates[-1]}" if dates else
                             f"{spec['label']:16s} empty")
        except (OSError, json.JSONDecodeError) as e:
            summaries.append(f"{spec['label']:16s} unreadable ({e})")

    problems += check_benchmarks(args.max_lag)

    report = build_report(problems, summaries)
    print(report)

    if args.report:
        with open(args.report, "w") as f:
            f.write(report)

    # Surface each problem in the Actions run's annotation list too.
    if os.environ.get("GITHUB_ACTIONS"):
        for p in problems:
            print(f"::error title={p.kind}::{p.subject}: {p.detail}")
        summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
        if summary_path:
            with open(summary_path, "a") as f:
                f.write(report + "\n")

    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
