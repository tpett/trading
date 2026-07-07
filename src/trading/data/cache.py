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

from trading.venues.base import DataFetchError

FetchFn = Callable[[str, datetime.date, datetime.date], pd.DataFrame]

# First cached bar may legitimately start after the requested date (weekends,
# holidays, listing date); tolerate this gap before declaring history missing.
_START_TOLERANCE = pd.Timedelta(5, unit="D")


class CacheSourceError(RuntimeError):
    """The cache dir was built by a different bar source than the one now
    requesting it -- serving its parquets would silently splice two
    adjustment regimes for the same symbol."""


class OhlcvCache:
    def __init__(self, cache_dir: Path, refetch_days: int, source: str = "yfinance"):
        self._dir = cache_dir
        self._refetch_days = refetch_days
        self._source = source
        self._dir.mkdir(parents=True, exist_ok=True)
        self._guard_source()

    def _guard_source(self) -> None:
        """Bind this cache dir to one bar source. Two sources adjust prices
        on the same basis but not bit-identically, so the cache-through
        merge (old cached rows + fresh rows) must never mix them. The marker
        is written on first use (or when a legacy unmarked dir is empty);
        a mismatch fails loudly rather than corrupting history."""
        marker = self._dir / ".source"
        if marker.exists():
            existing = marker.read_text().strip()
            if existing != self._source:
                raise CacheSourceError(
                    f"{self._dir} holds {existing!r} bars but bar_source={self._source!r}; "
                    "point data.cache_dir at a fresh directory when switching sources"
                )
            return
        # Legacy dirs predate the marker: adopt them ONLY as the default
        # yfinance source; a non-default source must start from an empty dir
        # so it can never inherit yfinance parquets.
        has_parquets = any(self._dir.glob("*.parquet"))
        if has_parquets and self._source != "yfinance":
            raise CacheSourceError(
                f"{self._dir} has unmarked (legacy yfinance) parquets but bar_source="
                f"{self._source!r}; point data.cache_dir at a fresh directory"
            )
        tmp = marker.with_suffix(".source.tmp")
        tmp.write_text(self._source)
        os.replace(tmp, marker)

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

        try:
            fresh = fetch_fn(symbol, fetch_start, end)
        except DataFetchError:
            # The adapter signals "no bars in this window" by RAISING, not by
            # returning empty. On a warm cache that is a gap, not a loss: a
            # delisted name's trailing-refetch window (end-refetch_days .. end)
            # is legitimately empty years after delisting, but its earlier
            # cached history is exactly what a survivorship-free backtest
            # needs. Preserve the file and serve the cached slice; only a
            # COLD miss (no cache) is a real fetch failure worth propagating.
            if cached is not None:
                return cached.loc[start_ts:end_ts]
            raise
        if fresh.empty and cached is not None:
            # Same gap semantics for adapters that return empty instead.
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
