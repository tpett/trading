"""Backfill orchestration: quarterly ZIPs -> facts -> PIT series -> store.

Pure composition of already-tested pieces (edgar/metrics/store/cik_map);
network + paths live in scripts/backfill_fundamentals.py. All quarters are
parsed together so TTM windows can span quarter boundaries; the append-only
store makes reruns idempotent (rows already visible are never rewritten).
"""

from __future__ import annotations

import datetime
from pathlib import Path

import pandas as pd

from trading.fundamentals.cik_map import interval_slice
from trading.fundamentals.edgar import empty_facts, load_quarter_facts
from trading.fundamentals.metrics import compute_pit_series
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


def backfill_quarters(
    zip_paths: list[Path], cik_map: pd.DataFrame, store: FundamentalsStore
) -> dict[str, int]:
    ciks = set(cik_map["cik"])
    # Drop empty per-quarter frames before concat (pandas 2.x warns on
    # empty-frame concatenation and the suite runs warnings-as-errors).
    parts = [f for path in zip_paths if not (f := load_quarter_facts(path, ciks)).empty]
    facts = pd.concat(parts, ignore_index=True) if parts else empty_facts()
    series_by_cik = compute_pit_series(facts)
    rows_appended = 0
    symbols_written: set[str] = set()
    for row in cik_map.itertuples():
        frame = series_by_cik.get(row.cik)
        if frame is None:
            continue
        window = interval_slice(frame, row.start, row.end)
        if window.empty:
            continue
        added = store.append(row.symbol, window)
        if added:
            symbols_written.add(row.symbol)
            rows_appended += added
    return {"filers": len(series_by_cik), "symbols": len(symbols_written), "rows": rows_appended}
