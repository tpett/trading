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

from trading.venues.base import EXTENDED_OHLCV_COLUMNS, DataFetchError

FetchFn = Callable[[str, datetime.date, datetime.date], pd.DataFrame]

# Neutral defaults used to widen a legacy narrow cache to the current schema
# (M2 migration): a dividend/split of "none" and a raw close equal to the
# already-cached (adjusted) close for rows that predate the extended fields.
_EXTENDED_CONSTANT_DEFAULTS = {"div_cash": 0.0, "split_factor": 1.0}

# First cached bar may legitimately start after the requested date (weekends,
# holidays, listing date); tolerate this gap before declaring history missing.
_START_TOLERANCE = pd.Timedelta(5, unit="D")

# Largest gap between consecutive cached bars tolerated in offline mode before
# calling it a truncated backfill. Generous: the longest routine market closure
# (holiday + weekend) is ~4 days; a rare multi-week trading halt should not trip
# it, but a months-long hole from a died-mid-window backfill must.
_MAX_INTERIOR_GAP = pd.Timedelta(30, unit="D")


class CacheSourceError(RuntimeError):
    """The cache dir was built by a different bar source than the one now
    requesting it -- serving its parquets would silently splice two
    adjustment regimes for the same symbol."""


class CacheSchemaError(RuntimeError):
    """A cached parquet and a fresh fetch carry different column sets. A plain
    pandas concat would NaN-fill the narrower side, silently corrupting the
    persisted history. The widened backfill starts from a cleared dir so this
    is not expected in practice; the guard exists so a mixed-width dir fails
    loudly instead of writing garbage."""


class OfflineCacheError(RuntimeError):
    """Offline mode was asked for bars the frozen cache does not cover (missing
    file, a range starting before the cached span beyond tolerance, or a gross
    interior hole from a truncated backfill). Serving a silent partial would
    reintroduce the coverage bias offline mode exists to avoid, so it fails
    loudly instead. A trailing gap is NOT a miss -- it is a delisting."""


def _migrate_cached_schema(
    cached: pd.DataFrame, target_columns: list[str], path: Path
) -> pd.DataFrame:
    """Widen a legacy narrow cache to the current (wide) schema in memory, so a
    live dir written before the corporate-action columns existed upgrades on the
    next fetch instead of raising. ONLY the known extended columns may be added
    (with neutral defaults); a missing CANONICAL column, or any column the cache
    has that the fetch does not, is real corruption and raises."""
    cached_cols = list(cached.columns)
    missing = [c for c in target_columns if c not in cached_cols]
    extra = [c for c in cached_cols if c not in target_columns]
    if extra or any(c not in EXTENDED_OHLCV_COLUMNS for c in missing):
        raise CacheSchemaError(
            f"{path} has columns {cached_cols} but the fetch returned {target_columns}; "
            "rebuild this cache dir from empty after a schema change instead of mixing widths"
        )
    widened = cached.copy()
    for col in missing:
        if col in _EXTENDED_CONSTANT_DEFAULTS:
            widened[col] = _EXTENDED_CONSTANT_DEFAULTS[col]
        else:  # close_raw: best available for pre-migration rows is the cached close
            widened[col] = widened["close"]
    return widened[target_columns]


class OhlcvCache:
    def __init__(
        self,
        cache_dir: Path,
        refetch_days: int,
        source: str = "yfinance",
        offline: bool = False,
    ):
        self._dir = cache_dir
        self._refetch_days = refetch_days
        self._source = source
        self._offline = offline
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

        if self._offline:
            return self._serve_offline(symbol, path, start, end, start_ts, end_ts)

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

        if cached is not None and not cached.empty and list(cached.columns) != list(fresh.columns):
            # An OLD narrow parquet meeting a NEW wide fetch: a plain concat
            # would NaN-fill the missing columns and silently corrupt the file.
            # If the ONLY difference is that the cache lacks the known extended
            # corporate-action columns (the live yfinance cache predates them),
            # migrate it in place -- widen the old rows with neutral defaults --
            # so the live nightly run upgrades seamlessly instead of breaking.
            # Any other mismatch (a canonical column gone, an unknown column) is
            # real corruption and still fails loudly.
            cached = _migrate_cached_schema(cached, list(fresh.columns), path)
            # `keep` was sliced from the pre-migration (narrow) frame; re-slice it
            # from the widened one so the concat below doesn't NaN-fill the extras.
            if keep is not None:
                keep = cached[cached.index < pd.Timestamp(cutoff, tz="UTC")]

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

    def _serve_offline(
        self,
        symbol: str,
        path: Path,
        start: datetime.date,
        end: datetime.date,
        start_ts: pd.Timestamp,
        end_ts: pd.Timestamp,
    ) -> pd.DataFrame:
        """Offline mode: serve [start, end] purely from the parquet, NEVER
        touching the network. Coverage is judged SYMMETRICALLY with the online
        path: online serves a symbol whenever its cached history STARTS early
        enough (index.min() <= start + tolerance) and treats a trailing gap as a
        legitimate delisting -- a delisted name's bars simply stop at delisting,
        years before the uniform backtest `end`. Offline must do the same, or it
        would raise on every delisted symbol, drop them, and silently reintroduce
        the exact survivorship bias a frozen cache exists to eliminate. So we
        require only that the START is covered; a trailing gap is fine. A GROSS
        interior hole (a backfill that died mid-window) is still caught, since
        that is real corruption, not a delisting."""
        if not path.exists():
            raise OfflineCacheError(f"offline cache miss for {symbol}: no parquet at {path}")
        cached = pd.read_parquet(path)
        if cached.empty:
            raise OfflineCacheError(f"offline cache for {symbol} is empty at {path}")
        first = cached.index.min()
        if first > start_ts + _START_TOLERANCE:
            raise OfflineCacheError(
                f"offline cache for {symbol} starts {first.date()}, after requested "
                f"start {start} (beyond the {_START_TOLERANCE.days}-day tolerance)"
            )
        # Interior-hole guard (NOT a trailing-delisting check): within the served
        # span, no gap between consecutive bars should exceed _MAX_INTERIOR_GAP.
        # Normal weekend/holiday stretches are a few days; a hole this large means
        # a truncated backfill, which must fail loudly rather than serve partial.
        served = cached.loc[start_ts:end_ts]
        if len(served) >= 2:
            max_gap = served.index.to_series().diff().max()
            if max_gap > _MAX_INTERIOR_GAP:
                raise OfflineCacheError(
                    f"offline cache for {symbol} has a {max_gap.days}-day interior gap "
                    f"within [{start}, {end}] (> {_MAX_INTERIOR_GAP.days}d): the backfill "
                    "for this dir is incomplete; rebuild it before running offline"
                )
        return served
