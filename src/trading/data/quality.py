"""Data-quality rules applied between fetch and signals (spec: Error Handling).

- Coverage: a run proceeds only if >= min_coverage of the universe fetched;
  excluded symbols are reported, never silently dropped.
- Sanity quarantine: prices are adjusted, so a day-over-day close move beyond
  max_daily_move without a corporate action is bad data; the symbol is
  excluded from ranking and surfaced as a warning.
"""

from __future__ import annotations

import datetime
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class CoverageReport:
    requested: int
    fetched: int
    ratio: float
    ok: bool
    missing: tuple[str, ...]


def check_coverage(
    requested: Sequence[str], fetched: Iterable[str], min_coverage: float
) -> CoverageReport:
    fetched_set = set(fetched)
    missing = tuple(sorted(s for s in requested if s not in fetched_set))
    count = len(requested) - len(missing)
    ratio = count / len(requested) if requested else 0.0
    return CoverageReport(
        requested=len(requested),
        fetched=count,
        ratio=ratio,
        ok=bool(requested) and ratio >= min_coverage,
        missing=missing,
    )


def quarantine_outliers(
    bars: Mapping[str, pd.DataFrame], max_daily_move: float, quarantine_window_days: int
) -> tuple[dict[str, pd.DataFrame], tuple[str, ...]]:
    """Flag symbols with an outlier close-over-close move within a recent window.

    Only pct_change moves within the trailing `quarantine_window_days` calendar
    days of each frame's OWN last bar are scanned; older spikes (a legitimate
    historical move, not a current data problem) are ignored entirely. The
    window is anchored to each frame's last bar, not "today", so this stays
    correct regardless of weekends/holidays or which as_of is used upstream.
    """
    clean: dict[str, pd.DataFrame] = {}
    quarantined: list[str] = []
    window = datetime.timedelta(days=quarantine_window_days)
    for symbol, df in bars.items():
        if df.empty:
            clean[symbol] = df
            continue
        cutoff = df.index[-1] - window
        # Compute moves over the FULL history (so the day straddling the
        # window boundary still has its prior close to compare against),
        # then only inspect moves whose own bar date falls in the window.
        moves = df["close"].pct_change().abs()
        recent_moves = moves.loc[cutoff:]
        if (recent_moves > max_daily_move).any():
            quarantined.append(symbol)
        else:
            clean[symbol] = df
    return clean, tuple(sorted(quarantined))
