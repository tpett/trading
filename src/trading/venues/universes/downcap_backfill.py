"""Warm the down-cap bar cache from a survivorship-free ROSTER (not index
membership) and REPORT coverage explicitly (R3 spec section 3). Mirrors
scripts/backfill_bars.py's cold-warm discipline: fetch every candidate up
front so a source gap is a RECORDED coverage hole, never a silent loss. The
cache dir is fresh with its own `.source = tiingo` marker (OhlcvCache guards
mixing sources)."""

from __future__ import annotations

import datetime
from collections.abc import Callable
from dataclasses import dataclass, field

import pandas as pd

from trading.venues.base import DataFetchError, RateLimitError
from trading.venues.universes.downcap_roster import candidates_at


def _sleep(seconds: float) -> None:
    """Sleep touchpoint, isolated for tests."""
    import time

    time.sleep(seconds)


@dataclass
class BackfillReport:
    fetched: int = 0
    missing: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    total: int = 0

    @property
    def coverage(self) -> float:
        return self.fetched / self.total if self.total else 0.0


def roster_symbols(
    roster: pd.DataFrame, start: datetime.date, end: datetime.date
) -> list[str]:
    """Every ticker that was a roster candidate on ANY month-start in
    [start, end] -- the survivorship-free set including mid-window delistings.
    Monthly sampling suffices: listing intervals are far longer than a month."""
    seen: set[str] = set()
    day = start
    while day <= end:
        seen.update(candidates_at(roster, day))
        year, month = (day.year + 1, 1) if day.month == 12 else (day.year, day.month + 1)
        day = datetime.date(year, month, 1)
    return sorted(seen)


def _fetch_waiting_on_rate_limit(cache, adapter, symbol, start, end, wait_s):
    """cache.fetch, but a rate-limit rejection waits and retries the SAME
    symbol (a metered plan slows, never punches coverage holes). A genuine
    miss (404 -> empty -> DataFetchError) propagates immediately."""
    while True:
        try:
            return cache.fetch(symbol, start, end, adapter.fetch_ohlcv)
        except RateLimitError:
            _sleep(wait_s)


def run_backfill(
    symbols: list[str],
    cache,
    adapter,
    start: datetime.date,
    end: datetime.date,
    *,
    throttle_s: float = 0.0,
    rate_limit_wait_s: float = 300.0,
    on_progress: Callable[[int, int, BackfillReport], None] | None = None,
) -> BackfillReport:
    """Fetch every symbol into `cache`, recording gaps. A DataFetchError (no
    such ticker / thin history) is a RECORDED miss, not an abort; any other
    exception is a recorded hard error (investigate before trusting the run)."""
    report = BackfillReport(total=len(symbols))
    for i, symbol in enumerate(symbols, 1):
        if throttle_s and i > 1:
            _sleep(throttle_s)
        try:
            df = _fetch_waiting_on_rate_limit(
                cache, adapter, symbol, start, end, rate_limit_wait_s
            )
            if df.empty:
                report.missing.append(symbol)
            else:
                report.fetched += 1
        except DataFetchError:
            report.missing.append(symbol)
        except Exception as exc:  # noqa: BLE001 - report, don't abort the pass
            report.errors.append(f"{symbol}: {type(exc).__name__}: {exc}")
        if on_progress is not None:
            on_progress(i, len(symbols), report)
    return report
