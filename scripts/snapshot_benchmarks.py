"""
Maintain ``data/benchmarks.json`` - the index series the dashboard measures the
models against.

Three series, because the two markets are not equally well served by the data
provider:

  SP500      ^GSPC daily closes. Full history, so every run rebuilds the window
             from the earliest ledger snapshot onward (self-healing: a missed
             day backfills on the next run).

  EGX30      ^CASE30. Yahoo carries the *live level* but no history for it
             (``validRanges: ['1d','5d']``, one row whatever range you ask for
             - re-verified 2026-08-20). So this series can only be accrued a
             point per run, starting the day this script first runs. It is the
             real index, and it is what the EGX comparison converges to.

  EGX30_EW   Equal-weight composite of the EGX 30 constituents that do have
             usable history, rebased to 100. A stand-in with real history, so
             the EGX panels can show comparison stats today instead of in two
             months. Reuses model_ai's regime-benchmark fallback, so the number
             on the dashboard is the same series the EGX models trade against.

Run post-close for either market; merging is keyed on date, so extra runs are
harmless. Wired up in .github/workflows/snapshot_benchmarks.yml.
"""

from __future__ import annotations

import glob
import json
import os
import sys
from datetime import datetime, timedelta, timezone

import pandas as pd
import yfinance as yf

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import model_ai  # noqa: E402  (needs REPO_ROOT on the path first)

BENCH_PATH = os.environ.get(
    "BENCHMARKS_PATH", os.path.join(REPO_ROOT, "data", "benchmarks.json"))
LEDGER_GLOB = os.path.join(REPO_ROOT, "data", "*ledger*.json")

# Pull a little before the first snapshot so a model whose ledger starts on a
# holiday still has an index close to rebase against.
BACKFILL_BUFFER_DAYS = 10
FALLBACK_START = "2026-06-01"


# ============================================================
# Ledger window
# ============================================================
def earliest_snapshot_date() -> str:
    """Earliest date any model ledger has a snapshot for, minus a buffer."""
    dates = []
    for path in glob.glob(LEDGER_GLOB):
        try:
            with open(path, "r") as f:
                ledger = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            print(f"[warn] skipping {os.path.basename(path)}: {e}")
            continue
        dates += [s["date"] for s in ledger.get("snapshots", []) if s.get("date")]

    if not dates:
        return FALLBACK_START
    start = datetime.strptime(min(dates), "%Y-%m-%d") - timedelta(days=BACKFILL_BUFFER_DAYS)
    return start.strftime("%Y-%m-%d")


# ============================================================
# Store
# ============================================================
def load_store() -> dict:
    if os.path.exists(BENCH_PATH):
        try:
            with open(BENCH_PATH, "r") as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            # Never let a corrupt file stop the daily accrual - but never
            # silently discard accrued history either. EGX30 in particular
            # cannot be re-fetched: overwriting it loses those days for good.
            sys.exit(f"{BENCH_PATH} is not valid JSON ({e}). Fix or delete it.")
    return {"benchmarks": {}}


def save_store(store: dict) -> None:
    store["updated"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    os.makedirs(os.path.dirname(BENCH_PATH), exist_ok=True)
    with open(BENCH_PATH, "w") as f:
        json.dump(store, f, indent=2)


def merge_series(store: dict, key: str, meta: dict, points: list[dict]) -> int:
    """
    Merge dated points into a benchmark series, the fresh value winning on a
    date collision. Returns how many dates were added.
    """
    entry = store["benchmarks"].setdefault(key, {})
    existing = {p["date"]: p["close"] for p in entry.get("series", [])}
    before = len(existing)
    for p in points:
        existing[p["date"]] = p["close"]
    entry.update(meta)
    entry["series"] = [{"date": d, "close": round(existing[d], 4)}
                       for d in sorted(existing)]
    return len(existing) - before


# ============================================================
# Sources
# ============================================================
def fetch_sp500(start: str) -> list[dict]:
    """^GSPC daily closes from `start` to now."""
    try:
        raw = yf.download("^GSPC", start=start, auto_adjust=True,
                          progress=False, group_by="column")
    except Exception as e:
        print(f"[sp500] download failed ({type(e).__name__}: {e})")
        return []
    if raw is None or raw.empty:
        print("[sp500] no data returned - keeping existing series")
        return []

    close = raw["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close = close.dropna()
    print(f"[sp500] {len(close)} closes, {close.index.min().date()} "
          f"-> {close.index.max().date()}")
    return [{"date": ts.strftime("%Y-%m-%d"), "close": float(v)}
            for ts, v in close.items()]


def fetch_egx30_quote() -> list[dict]:
    """
    The ^CASE30 level(s) Yahoo will part with - in practice the current
    session's, one point.

    Dates come from the quote's own (Cairo) timestamp rather than the clock, so
    the US-close run at 21:30 UTC - already past midnight in Cairo - still
    files the level under the session it belongs to.
    """
    try:
        hist = yf.Ticker("^CASE30").history(period="5d", auto_adjust=True)
    except Exception as e:
        print(f"[egx30] quote fetch failed ({type(e).__name__}: {e})")
        return []
    if hist is None or hist.empty:
        print("[egx30] no quote returned")
        return []

    close = hist["Close"].dropna()
    points = [{"date": ts.strftime("%Y-%m-%d"), "close": float(v)}
              for ts, v in close.items()]
    print("[egx30] " + ", ".join(f"{p['date']}={p['close']:,.0f}" for p in points))
    return points


def fetch_egx30_composite(period: str = "1y") -> list[dict]:
    """
    Equal-weight composite of the EGX universe, built through model_ai's own
    fallback so the dashboard proxy and the models' regime filter cannot drift
    apart.
    """
    try:
        raw = model_ai._download(model_ai.EGX_UNIVERSE, period)
        close = raw["Close"].dropna(axis=1, how="all")
        series, name = model_ai._benchmark_series("^CASE30", close, period)
    except Exception as e:
        print(f"[egx30-ew] composite failed ({type(e).__name__}: {e})")
        return []

    series = series.dropna()
    if series.empty:
        print("[egx30-ew] composite empty")
        return []
    print(f"[egx30-ew] {len(series)} points ({name}), "
          f"{series.index.min().date()} -> {series.index.max().date()}")
    return [{"date": ts.strftime("%Y-%m-%d"), "close": float(v)}
            for ts, v in series.items()]


# ============================================================
# Main
# ============================================================
def main():
    start = earliest_snapshot_date()
    print(f"[window] backfilling benchmarks from {start}")

    store = load_store()
    store.setdefault("benchmarks", {})

    added = merge_series(store, "SP500", {
        "name": "S&P 500",
        "symbol": "^GSPC",
        "market": "US",
        "currency": "USD",
        "kind": "index",
        "source": "yfinance ^GSPC daily close",
    }, fetch_sp500(start))
    print(f"[sp500] +{added} new dates")

    added = merge_series(store, "EGX30", {
        "name": "EGX 30",
        "symbol": "^CASE30",
        "market": "EGX",
        "currency": "EGP",
        "kind": "index",
        "source": "yfinance ^CASE30 level, accrued one point per run",
        "note": ("Yahoo serves no history for ^CASE30, so this series starts "
                 "the day the collector first ran and grows daily."),
    }, fetch_egx30_quote())
    print(f"[egx30] +{added} new dates")

    added = merge_series(store, "EGX30_EW", {
        "name": "EGX 30 proxy",
        "symbol": "EW-COMPOSITE",
        "market": "EGX",
        "currency": "EGP",
        "kind": "proxy",
        "source": "equal-weight composite of EGX 30 constituents (rebased 100)",
        "note": ("Stand-in for ^CASE30 while the real index series accrues. "
                 "Same series model_ai uses for EGX regime detection."),
    }, fetch_egx30_composite())
    print(f"[egx30-ew] +{added} new dates")

    save_store(store)
    counts = {k: len(v.get("series", [])) for k, v in store["benchmarks"].items()}
    print(f"Saved {os.path.relpath(BENCH_PATH, REPO_ROOT)}: {counts}")


if __name__ == "__main__":
    main()
