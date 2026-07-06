"""Rankings pipeline: universe -> cached fetch -> quality gates -> signals -> regime.

Orchestrates I/O around the pure signal engine. Raises PipelineDataError when
the spec says the run must not proceed (< min_coverage universe fetch,
benchmark fetch failure); the CLI turns that into a warning + nonzero exit.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from trading.config import VenueConfig
from trading.data.cache import OhlcvCache
from trading.data.quality import CoverageReport, check_coverage, quarantine_outliers
from trading.fundamentals.store import FundamentalsStore
from trading.signals.engine import rank
from trading.signals.regime import Regime, compute_regime
from trading.signals.registry import get_ranker
from trading.venues.base import SymbolInfo, VenueAdapter


class PipelineDataError(RuntimeError):
    """Fresh data could not be assembled; the run must not proceed (spec)."""


def _drop_incomplete_last_bar(df: pd.DataFrame, as_of: datetime.date) -> pd.DataFrame:
    """Drop bars dated as_of or later: they may still be in progress (Kraken's
    current UTC day, a partial yfinance intraday print). Decisions use data
    through the last COMPLETED bar only (spec). Frames are assumed indexed by
    a tz-aware UTC timestamp normalized to the bar's date.
    """
    cutoff = pd.Timestamp(as_of, tz="UTC")
    return df[df.index < cutoff]


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
    bars: dict[str, pd.DataFrame]  # clean (quarantine-passed) universe bars, for the M2 simulator
    benchmark_bars: pd.DataFrame


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
            frame = cache.fetch(info.symbol, start, as_of, adapter.fetch_ohlcv)
        except Exception:
            # Any per-symbol failure (network, missing pair, bad frame) is an
            # exclusion; the coverage gate below is the safety net.
            failures.append(info.symbol)
            continue
        if config.data.drop_incomplete_last_bar:
            frame = _drop_incomplete_last_bar(frame, as_of)
        bars[info.symbol] = frame

    # The benchmark symbol is often also a universe member (e.g. crypto's BTC
    # benchmark is also ranked); reuse the bars already fetched above instead
    # of issuing a second fetch for the same symbol.
    if config.benchmark in bars:
        benchmark = bars[config.benchmark]
    else:
        try:
            benchmark = cache.fetch(config.benchmark, start, as_of, adapter.fetch_ohlcv)
        except Exception as exc:
            raise PipelineDataError(f"benchmark {config.benchmark} fetch failed: {exc}") from exc
        if config.data.drop_incomplete_last_bar:
            benchmark = _drop_incomplete_last_bar(benchmark, as_of)

    fundamentals: dict[str, pd.DataFrame] | None = None
    if get_ranker(config.signals.ranker).requires_fundamentals:
        # Read-only here: the live REFRESH (weekly companyfacts top-up) is the
        # runner's job, fail-open. An empty/missing store simply yields all-
        # neutral quality -- never an abort.
        store = FundamentalsStore(Path(config.data.fundamentals_dir))
        fundamentals = store.load([i.symbol for i in infos])

    return assemble_rankings(
        config,
        infos,
        bars,
        benchmark,
        as_of,
        fetch_failures=tuple(sorted(failures)),
        fundamentals=fundamentals,
    )


def assemble_rankings(
    config: VenueConfig,
    infos: list[SymbolInfo],
    bars: dict[str, pd.DataFrame],
    benchmark_bars: pd.DataFrame,
    as_of: datetime.date,
    fetch_failures: tuple[str, ...] = (),
    fundamentals: dict[str, pd.DataFrame] | None = None,
) -> RankingsResult:
    """Pure rankings core: coverage -> quarantine -> regime -> features -> rank
    (-> fundamentals overlay when the configured ranker requires it).

    No I/O, no clock. build_rankings (live) and the M3 backtester's prepare()
    both call this, so backtest and live-paper rank identically by construction.
    """
    coverage = check_coverage([i.symbol for i in infos], bars, config.data.min_coverage)
    if not coverage.ok:
        raise PipelineDataError(
            f"universe coverage {coverage.ratio:.0%} below "
            f"{config.data.min_coverage:.0%}; missing: {', '.join(coverage.missing)}"
        )

    clean, quarantined = quarantine_outliers(
        bars, config.data.max_daily_move, config.data.quarantine_window_days
    )

    # A corrupt benchmark print must not silently flip venue-wide exposure/
    # regime: run it through the same recent-window sanity check universe
    # symbols get, but fail loudly instead of quietly excluding it.
    _, benchmark_quarantined = quarantine_outliers(
        {config.benchmark: benchmark_bars},
        config.data.max_daily_move,
        config.data.quarantine_window_days,
    )
    if benchmark_quarantined:
        raise PipelineDataError(
            f"benchmark {config.benchmark} failed data-sanity check: outlier move "
            f"within the trailing {config.data.quarantine_window_days}d window"
        )

    as_of_ts = pd.Timestamp(as_of, tz="UTC")
    regime = compute_regime(benchmark_bars, as_of_ts, config.regime)
    spec = get_ranker(config.signals.ranker)
    features = spec.fn(clean, as_of_ts, config.signals, fundamentals)
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
        fetch_failures=fetch_failures,
        insufficient_history=insufficient,
        bars=clean,
        benchmark_bars=benchmark_bars,
    )
