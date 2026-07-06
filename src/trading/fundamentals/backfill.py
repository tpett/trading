"""Backfill orchestration: facts -> PIT series -> store, from either of two
sources (companyfacts is the PRIMARY path; the quarterly-ZIP path is RETIRED-
PRIMARY -- see below). Pure composition of already-tested pieces
(edgar/companyfacts/metrics/store/cik_map); network + paths live in
scripts/backfill_fundamentals.py.

companyfacts is the primary backfill source (default, `backfill_from_
companyfacts`): a census of the 2018q1-2026q2 quarterly ZIPs found
dei:EntityCommonStockSharesOutstanding on exactly 1 of 5631 filings (FSDS
strips most dei cover-page facts), which left the ZIP-backfilled store's
shares_outstanding coverage at 59% overall and ZERO for JPM/META/BRK-B --
tickers the live companyfacts-refreshed store resolves at 100%. That
backtest/live regime mismatch (a ranker sees real values in production but
mostly NaN over history) is unacceptable, so companyfacts.facts_from_
companyfacts -> compute_pit_series is now the default rebuild path for
EVERY primitive, not just shares.

The quarterly-ZIP path (`backfill_quarters`) stays in the codebase --
tested, and selectable via `--source zips` -- for its original purpose
(bulk revenue/COGS/assets/net-income/equity coverage without a per-CIK
network round trip) but is no longer what `scripts/backfill_fundamentals.py`
runs by default. All quarters are parsed together so TTM windows can span
quarter boundaries; both paths write through the SAME append-only store, so
reruns of either are idempotent (rows already visible are never rewritten) --
which is also why a source-switching rebuild MUST start from an EMPTY store:
append-only semantics would otherwise silently keep whatever (possibly
NaN-shares) rows a prior run already wrote for a given filed date instead of
replacing them (see scripts/backfill_fundamentals.py's empty-store guard).
"""

from __future__ import annotations

import datetime
from collections.abc import Callable
from pathlib import Path

import pandas as pd

from trading.fundamentals.cik_map import interval_slice
from trading.fundamentals.companyfacts import (
    COMPANYFACTS_URL,
    _http_get_json,
    facts_from_companyfacts,
)
from trading.fundamentals.edgar import empty_facts, load_quarter_facts
from trading.fundamentals.metrics import compute_pit_series, empty_series
from trading.fundamentals.store import FundamentalsStore


def quarter_range(start: str, end: str) -> list[str]:
    """Inclusive "2018q1".."2019q2" -> every quarter label between."""
    year, quarter = int(start[:4]), int(start[5])
    end_year, end_quarter = int(end[:4]), int(end[5])
    out: list[str] = []
    while (year, quarter) <= (end_year, end_quarter):
        out.append(f"{year}q{quarter}")
        quarter += 1
        if quarter == 5:
            year, quarter = year + 1, 1
    return out


def last_complete_quarter(today: datetime.date) -> str:
    """SEC publishes a quarter's ZIP after the quarter ends; the in-progress
    quarter is served by the companyfacts top-up instead."""
    completed = (today.month - 1) // 3
    if completed == 0:
        return f"{today.year - 1}q4"
    return f"{today.year}q{completed}"


def _write_series_to_store(
    series_by_cik: dict[int, pd.DataFrame], cik_map: pd.DataFrame, store: FundamentalsStore
) -> dict[str, int]:
    """Shared split step for both backfill sources: each cik's series is
    sliced per cik_map interval and appended to that interval's symbol.
    Filed dates covered by SOME symbol interval, per cik, are tracked so the
    remainder (fell in an interval gap and reached no store) is observable
    as "dropped" in the returned stats -- behavior is unchanged."""
    rows_appended = 0
    symbols_written: set[str] = set()
    covered: dict[int, set] = {}
    for row in cik_map.itertuples():
        frame = series_by_cik.get(row.cik)
        if frame is None:
            continue
        window = interval_slice(frame, row.start, row.end)
        if window.empty:
            continue
        covered.setdefault(row.cik, set()).update(window.index)
        added = store.append(row.symbol, window)
        if added:
            symbols_written.add(row.symbol)
            rows_appended += added
    dropped = sum(len(frame) - len(covered.get(cik, set())) for cik, frame in series_by_cik.items())
    return {
        "filers": len(series_by_cik),
        "symbols": len(symbols_written),
        "rows": rows_appended,
        "dropped": dropped,
    }


def backfill_quarters(
    zip_paths: list[Path], cik_map: pd.DataFrame, store: FundamentalsStore
) -> dict[str, int]:
    """RETIRED-PRIMARY path (see module docstring): quarterly-ZIP facts ->
    the same PIT series + store split as backfill_from_companyfacts. Kept
    for its bulk-download shape and test coverage; `scripts/backfill_
    fundamentals.py --source zips` selects it explicitly."""
    ciks = set(cik_map["cik"])
    # Drop empty per-quarter frames before concat (pandas 2.x warns on
    # empty-frame concatenation and the suite runs warnings-as-errors).
    parts = [f for path in zip_paths if not (f := load_quarter_facts(path, ciks)).empty]
    facts = pd.concat(parts, ignore_index=True) if parts else empty_facts()
    series_by_cik = compute_pit_series(facts)
    return _write_series_to_store(series_by_cik, cik_map, store)


def backfill_from_companyfacts(
    cik_map: pd.DataFrame,
    store: FundamentalsStore,
    fetch_json: Callable[[str], dict] = _http_get_json,
    on_progress: Callable[[int, int], None] | None = None,
) -> dict[str, int]:
    """PRIMARY backfill path (see module docstring): one companyfacts fetch
    per UNIQUE cik in cik_map (not per symbol -- a rename chain or a
    dual-listing like GOOG/GOOGL shares one cik and is fetched once),
    normalized through facts_from_companyfacts -> compute_pit_series -- the
    SAME normalized table and PIT computation the ZIP path and the runner's
    weekly top-up use, so a rebuild and a top-up can never diverge -- then
    split across every symbol interval that cik maps to.

    A per-cik fetch failure is fail-open here too, exactly like the weekly
    top-up: it is counted in the returned "failed" stat rather than raised,
    so one bad cik never aborts a ~1,100-cik rebuild; append-only semantics
    make a rerun that targets just the gaps safe (already-stored filed dates
    are never rewritten -- see FundamentalsStore.append)."""
    ciks = sorted(set(cik_map["cik"]))
    series_by_cik: dict[int, pd.DataFrame] = {}
    failed = 0
    for i, cik in enumerate(ciks, start=1):
        try:
            payload = fetch_json(COMPANYFACTS_URL.format(cik=cik))
            facts = facts_from_companyfacts(payload, cik)
            series_by_cik[cik] = compute_pit_series(facts).get(cik, empty_series())
        except Exception:
            failed += 1
        if on_progress is not None:
            on_progress(i, len(ciks))
    stats = _write_series_to_store(series_by_cik, cik_map, store)
    stats["failed"] = failed
    return stats
