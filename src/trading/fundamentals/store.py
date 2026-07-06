"""Append-only per-symbol parquet store for fundamentals (spec: M4).

Layout: <root>/<SYMBOL>.parquet, tz-aware UTC DatetimeIndex = FILED dates,
metrics.SERIES_COLUMNS. Fundamentals history is IMMUTABLE: append() never
overwrites an existing filed-date row, so a later restated value can never
replace what was visible at the time. OhlcvCache's trailing-refetch model
deliberately does NOT apply here; its atomic tmp+os.replace write and
cache-through reads do.

<root>/.last_refresh (ISO date) records the last companyfacts top-up; the
runner's weekly refresh gate reads it (see trading.runner).
"""

from __future__ import annotations

import datetime
import os
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from trading.fundamentals.metrics import empty_series


class FundamentalsStore:
    def __init__(self, root: Path):
        self._root = root
        self._root.mkdir(parents=True, exist_ok=True)

    def path_for(self, symbol: str) -> Path:
        return self._root / f"{symbol.replace('/', '-')}.parquet"

    def read(self, symbol: str) -> pd.DataFrame:
        path = self.path_for(symbol)
        if not path.exists():
            return empty_series()
        return pd.read_parquet(path)

    def append(self, symbol: str, rows: pd.DataFrame) -> int:
        """Add rows whose FILED date is not already stored; existing rows are
        never touched. Returns the number of rows actually added."""
        if rows.empty:
            return 0
        if rows.index.tz is None:
            raise ValueError(f"{symbol}: fundamentals index must be tz-aware UTC filed dates")
        existing = self.read(symbol)
        fresh = rows[~rows.index.isin(existing.index)]
        if fresh.empty:
            return 0
        if existing.empty:
            # Never concat an empty frame (pandas 2.x FutureWarning, and the
            # suite runs warnings-as-errors).
            merged = fresh.sort_index(kind="mergesort")
        else:
            merged = pd.concat([existing, fresh]).sort_index(kind="mergesort")
        path = self.path_for(symbol)
        tmp = path.with_suffix(".parquet.tmp")
        merged.to_parquet(tmp)
        os.replace(tmp, path)  # atomic: never leave a torn store file
        return len(fresh)

    def load(self, symbols: Iterable[str]) -> dict[str, pd.DataFrame]:
        out: dict[str, pd.DataFrame] = {}
        for symbol in symbols:
            frame = self.read(symbol)
            if not frame.empty:
                out[symbol] = frame
        return out

    def last_refresh(self) -> datetime.date | None:
        path = self._root / ".last_refresh"
        if not path.exists():
            return None
        return datetime.date.fromisoformat(path.read_text().strip())

    def mark_refreshed(self, day: datetime.date) -> None:
        path = self._root / ".last_refresh"
        tmp = path.with_suffix(".tmp")
        tmp.write_text(day.isoformat())
        os.replace(tmp, path)
