"""Rankings pipeline: universe -> cached fetch -> quality gates -> signals -> regime.

Orchestrates I/O around the pure signal engine. Raises PipelineDataError when
the spec says the run must not proceed (< min_coverage universe fetch,
benchmark fetch failure); the CLI turns that into a warning + nonzero exit.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass

import pandas as pd

from trading.config import VenueConfig
from trading.data.cache import OhlcvCache
from trading.data.quality import CoverageReport, check_coverage, quarantine_outliers
from trading.signals.engine import compute_features, rank
from trading.signals.regime import Regime, compute_regime
from trading.venues.base import VenueAdapter


class PipelineDataError(RuntimeError):
    """Fresh data could not be assembled; the run must not proceed (spec)."""


@dataclass(frozen=True)
class RankingsResult:
    venue: str
    as_of: pd.Timestamp
    regime: Regime
    table: pd.DataFrame  # ranked; leading "status" column + engine OUTPUT_COLUMNS
    coverage: CoverageReport
    quarantined: tuple[str, ...]
    fetch_failures: tuple[str, ...]
    insufficient_history: tuple[str, ...]


def build_rankings(
    config: VenueConfig,
    adapter: VenueAdapter,
    cache: OhlcvCache,
    as_of: datetime.date,
) -> RankingsResult:
    start = as_of - datetime.timedelta(days=config.data.history_days)
    infos = adapter.universe(as_of)

    bars: dict[str, pd.DataFrame] = {}
    failures: list[str] = []
    for info in infos:
        try:
            bars[info.symbol] = cache.fetch(info.symbol, start, as_of, adapter.fetch_ohlcv)
        except Exception:
            # Any per-symbol failure (network, missing pair, bad frame) is an
            # exclusion; the coverage gate below is the safety net.
            failures.append(info.symbol)

    coverage = check_coverage([i.symbol for i in infos], bars, config.data.min_coverage)
    if not coverage.ok:
        raise PipelineDataError(
            f"universe coverage {coverage.ratio:.0%} below "
            f"{config.data.min_coverage:.0%}; missing: {', '.join(coverage.missing)}"
        )

    clean, quarantined = quarantine_outliers(bars, config.data.max_daily_move)

    try:
        benchmark = cache.fetch(config.benchmark, start, as_of, adapter.fetch_ohlcv)
    except Exception as exc:
        raise PipelineDataError(f"benchmark {config.benchmark} fetch failed: {exc}") from exc

    as_of_ts = pd.Timestamp(as_of, tz="UTC")
    regime = compute_regime(benchmark, as_of_ts, config.regime)
    features = compute_features(clean, as_of_ts, config.signals)
    table = rank(features).copy()

    statuses = {i.symbol: i.status for i in infos}
    table.insert(0, "status", [statuses[s] for s in table.index])
    insufficient = tuple(sorted(set(clean) - set(table.index)))

    return RankingsResult(
        venue=config.name,
        as_of=as_of_ts,
        regime=regime,
        table=table,
        coverage=coverage,
        quarantined=quarantined,
        fetch_failures=tuple(sorted(failures)),
        insufficient_history=insufficient,
    )
