"""Parquet cache-through layer for OHLCV bars.

The trailing `refetch_days` window is always re-fetched: adjusted equity
history rewrites on corporate actions, so recent cache contents are treated
as ephemeral (spec). Older rows are served from per-symbol Parquet files.
"""

from __future__ import annotations

import datetime
import os
from collections.abc import Callable
from pathlib import Path

import pandas as pd

FetchFn = Callable[[str, datetime.date, datetime.date], pd.DataFrame]

# First cached bar may legitimately start after the requested date (weekends,
# holidays, listing date); tolerate this gap before declaring history missing.
_START_TOLERANCE = pd.Timedelta(5, unit="D")


class OhlcvCache:
    def __init__(self, cache_dir: Path, refetch_days: int):
        self._dir = cache_dir
        self._refetch_days = refetch_days
        self._dir.mkdir(parents=True, exist_ok=True)

    def path_for(self, symbol: str) -> Path:
        return self._dir / f"{symbol.replace('/', '-')}.parquet"

    def fetch(
        self,
        symbol: str,
        start: datetime.date,
        end: datetime.date,
        fetch_fn: FetchFn,
    ) -> pd.DataFrame:
        path = self.path_for(symbol)
        start_ts = pd.Timestamp(start, tz="UTC")
        end_ts = pd.Timestamp(end, tz="UTC")
        cutoff = end - datetime.timedelta(days=self._refetch_days)

        cached: pd.DataFrame | None = None
        keep: pd.DataFrame | None = None
        fetch_start = start
        if path.exists():
            cached = pd.read_parquet(path)
            if not cached.empty and cached.index.min() <= start_ts + _START_TOLERANCE:
                keep = cached[cached.index < pd.Timestamp(cutoff, tz="UTC")]
                # Always refetch from the cutoff (not max(cutoff, start)):
                # a narrower request must never drop cached rows in
                # [cutoff, start) from the persisted file.
                fetch_start = cutoff

        fresh = fetch_fn(symbol, fetch_start, end)
        if fresh.empty and cached is not None:
            # A data-source gap must never shrink the file: leave the cached
            # frame untouched and serve the requested slice from it.
            return cached.loc[start_ts:end_ts]

        parts: list[pd.DataFrame] = []
        if keep is not None and not keep.empty:
            parts.append(keep)
        if cached is not None:
            # Preserve cached rows after the requested end so a narrower
            # request (warm or full-refetch) never truncates the tail of the
            # cache file. Bounded by end_ts, not fresh.index.max(), which
            # would be NaT for an empty fresh frame.
            tail = cached[cached.index > end_ts]
            if not tail.empty:
                parts.append(tail)
        parts.append(fresh)  # last: fresh values win any dedup overlap
        merged = parts[0] if len(parts) == 1 else pd.concat(parts)
        # Dedup is a safety net for adapters that over-fetch beyond the
        # requested bounds; keep/fresh/tail are disjoint by construction.
        merged = merged[~merged.index.duplicated(keep="last")].sort_index()

        tmp = path.with_suffix(".parquet.tmp")
        merged.to_parquet(tmp)
        os.replace(tmp, path)  # atomic: never leave a torn cache file

        return merged.loc[start_ts:end_ts]
