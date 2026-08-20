"""
Heartbeat tests
===============
The heartbeat only earns its place if it fails on the shapes of damage that
have actually happened here. Each test below is a real defect found in this
repo on 2026-08-20, reduced to a synthetic ledger:

  * 2026-08-06 missing from all three US ledgers while the tail looked fine,
  * 2026-06-19 and 2026-07-03 snapshotted against a closed market,
  * a session filed under tomorrow's date, producing a duplicate.

Run:  python -m pytest tests/test_freshness.py -v
      python tests/test_freshness.py          (no pytest needed)
"""

from __future__ import annotations

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

import check_freshness as cf  # noqa: E402

# A five-session week; the calendar the ledgers are checked against.
CALENDAR = ["2026-08-03", "2026-08-04", "2026-08-05", "2026-08-06", "2026-08-07"]


def _ledger(dates: list[str]) -> str:
    """Write a throwaway ledger with these snapshot dates; return its path."""
    handle, path = tempfile.mkstemp(suffix=".json")
    with os.fdopen(handle, "w") as f:
        json.dump({"initial_budget": 100_000,
                   "snapshots": [{"date": d, "equity": 100_000.0,
                                  "cash": 100_000.0, "pnl": 0.0,
                                  "positions": []} for d in dates]}, f)
    return path


def _check(dates: list[str], max_lag: int = 1, calendar=CALENDAR):
    path = _ledger(dates)
    try:
        # check_ledger resolves paths against REPO_ROOT; an absolute temp path
        # survives the join unchanged on POSIX and Windows alike.
        spec = {"label": "Test", "path": path, "market": "US"}
        return cf.check_ledger(spec, calendar, max_lag)
    finally:
        os.unlink(path)


def _kinds(problems):
    return sorted(p.kind for p in problems)


def test_clean_ledger_is_silent():
    assert _check(CALENDAR) == [], _check(CALENDAR)
    print("PASS test_clean_ledger_is_silent")


def test_interior_gap_is_caught_even_when_the_tail_is_fresh():
    """The 2026-08-06 case: newest row is current, but a day is missing."""
    problems = _check([d for d in CALENDAR if d != "2026-08-06"])
    assert "gap" in _kinds(problems), _kinds(problems)
    assert "2026-08-06" in str(problems[0]), str(problems[0])
    print("PASS test_interior_gap_is_caught_even_when_the_tail_is_fresh")


def test_off_calendar_rows_are_caught():
    """Snapshots taken on Juneteenth / July 4th against a closed market."""
    # 2026-08-04 is a session; a row dated 2026-08-02 (a Sunday) is not.
    problems = _check(["2026-08-02"] + CALENDAR)
    assert "off-calendar" in _kinds(problems), _kinds(problems)
    print("PASS test_off_calendar_rows_are_caught")


def test_duplicate_dates_are_caught():
    problems = _check(["2026-08-03", "2026-08-04", "2026-08-04",
                       "2026-08-05", "2026-08-06", "2026-08-07"])
    assert "ordering" in _kinds(problems), _kinds(problems)
    print("PASS test_duplicate_dates_are_caught")


def test_out_of_order_dates_are_caught():
    problems = _check(["2026-08-04", "2026-08-03", "2026-08-05",
                       "2026-08-06", "2026-08-07"])
    assert "ordering" in _kinds(problems), _kinds(problems)
    print("PASS test_out_of_order_dates_are_caught")


def test_staleness_respects_the_lag_tolerance():
    """One late session is tolerated; two is a stopped pipeline."""
    one_behind = _check(CALENDAR[:-1])
    assert "stale" not in _kinds(one_behind), _kinds(one_behind)

    two_behind = _check(CALENDAR[:-2])
    assert "stale" in _kinds(two_behind), _kinds(two_behind)
    print("PASS test_staleness_respects_the_lag_tolerance")


def test_missing_calendar_is_reported_not_ignored():
    """Without a calendar the checker must say so, never quietly pass."""
    problems = _check(CALENDAR, calendar=[])
    assert "no-calendar" in _kinds(problems), _kinds(problems)
    print("PASS test_missing_calendar_is_reported_not_ignored")


def test_empty_ledger_is_reported():
    problems = _check([])
    assert "empty" in _kinds(problems), _kinds(problems)
    print("PASS test_empty_ledger_is_reported")


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
            except AssertionError as e:
                failures += 1
                print(f"FAIL {name}: {e}")
    sys.exit(1 if failures else 0)
