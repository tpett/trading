"""Data-quality rules applied between fetch and signals (spec: Error Handling).

- Coverage: a run proceeds only if >= min_coverage of the universe fetched;
  excluded symbols are reported, never silently dropped.
- Sanity quarantine: prices are adjusted, so a day-over-day close move beyond
  max_daily_move without a corporate action is bad data; the symbol is
  excluded from ranking and surfaced as a warning.
"""

from __future__ import annotations

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
    bars: Mapping[str, pd.DataFrame], max_daily_move: float
) -> tuple[dict[str, pd.DataFrame], tuple[str, ...]]:
    clean: dict[str, pd.DataFrame] = {}
    quarantined: list[str] = []
    for symbol, df in bars.items():
        moves = df["close"].pct_change().abs()
        if (moves > max_daily_move).any():
            quarantined.append(symbol)
        else:
            clean[symbol] = df
    return clean, tuple(sorted(quarantined))
