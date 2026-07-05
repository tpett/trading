"""Walk-forward validation (spec: Backtesting & Validation).

Tune on train_months, test on the following test_months untouched, roll by
test_months. The tunable surface is EXACTLY two hyperparameters, grid-searched
from TOML grids; selection on the train window is highest Sharpe (tiebreak:
higher total return, then lower threshold, then lower stop multiple --
deterministic). Only stitched OOS segments are reported; each test window
replays from a fresh initial state and the stitch chains their daily returns.
The stitched OOS must fully cover at least one configured stress segment.
Holdout (>= config.backtest.holdout_start) is a one-time-only, hand-off
period: walk-forward clamps its span to never touch it, regardless of the
`end` the caller passes in. Pure module: no I/O, no clock -- the CLI journals
every window as an experiment.
"""

from __future__ import annotations

import datetime
import math
from dataclasses import dataclass, replace

import pandas as pd

from trading.backtest.engine import BacktestResult, PreparedBacktest, replay
from trading.backtest.metrics import BacktestMetrics, compute_metrics, metrics_from_curves
from trading.config import VenueConfig


class WalkForwardError(RuntimeError):
    pass


@dataclass(frozen=True)
class GridPoint:
    entry_score_threshold: float
    stop_atr_multiple: float


@dataclass(frozen=True)
class Window:
    train_start: datetime.date
    train_end: datetime.date  # exclusive
    test_start: datetime.date
    test_end: datetime.date  # exclusive


@dataclass(frozen=True)
class WindowResult:
    window: Window
    best: GridPoint
    train_metrics: BacktestMetrics
    test_result: BacktestResult
    test_metrics: BacktestMetrics


@dataclass(frozen=True)
class WalkForwardResult:
    windows: tuple[WindowResult, ...]
    stitched_equity: pd.Series
    stitched_benchmark: pd.Series
    stitched_metrics: BacktestMetrics
    stress_segments_covered: tuple[str, ...]


def add_months(day: datetime.date, months: int) -> datetime.date:
    if day.day > 28:
        raise ValueError("walk-forward dates must use day-of-month <= 28")
    month = day.month - 1 + months
    return datetime.date(day.year + month // 12, month % 12 + 1, day.day)


def generate_windows(
    start: datetime.date, end: datetime.date, train_months: int, test_months: int
) -> list[Window]:
    windows: list[Window] = []
    cursor = start
    while True:
        train_end = add_months(cursor, train_months)
        test_end = add_months(train_end, test_months)
        if test_end > end:
            break  # only FULL test windows count as OOS
        windows.append(Window(cursor, train_end, train_end, test_end))
        cursor = add_months(cursor, test_months)
    return windows


def grid_points(config: VenueConfig) -> list[GridPoint]:
    return [
        GridPoint(threshold, stop)
        for threshold in config.backtest.entry_score_threshold_grid
        for stop in config.backtest.stop_atr_multiple_grid
    ]


def apply_grid_point(config: VenueConfig, point: GridPoint) -> VenueConfig:
    portfolio = replace(
        config.portfolio,
        entry_score_threshold=point.entry_score_threshold,
        stop_atr_multiple=point.stop_atr_multiple,
    )
    return replace(config, portfolio=portfolio)


def _selection_key(point: GridPoint, metrics: BacktestMetrics) -> tuple:
    sharpe = -math.inf if math.isnan(metrics.sharpe) else metrics.sharpe
    total = -math.inf if math.isnan(metrics.total_return) else metrics.total_return
    return (-sharpe, -total, point.entry_score_threshold, point.stop_atr_multiple)


def _stitch(curves: list[pd.Series], starting_balance: float) -> pd.Series:
    returns = pd.concat([c.pct_change().dropna() for c in curves]).sort_index()
    stitched = starting_balance * (1.0 + returns).cumprod()
    anchor = pd.Series([starting_balance], index=[curves[0].index[0]])
    return pd.concat([anchor, stitched]).sort_index()


def _covered_segments(config: VenueConfig, stitched: pd.Series) -> tuple[str, ...]:
    first, last = stitched.index[0].date(), stitched.index[-1].date()
    return tuple(
        f"{seg_start.isoformat()}..{seg_end.isoformat()}"
        for seg_start, seg_end in config.backtest.stress_segments
        if first <= seg_start and seg_end <= last
    )


def run_walk_forward(
    prepared: PreparedBacktest,
    config: VenueConfig,
    *,
    start: datetime.date,
    end: datetime.date,
) -> WalkForwardResult:
    bt = config.backtest
    # Holdout is a one-time-only, hand-off period (spec): walk-forward must
    # never see it, no matter what `end` the caller passes in. Clamp here so
    # the guarantee holds even if the CLI's own holdout guard is bypassed.
    end = min(end, bt.holdout_start)
    windows = generate_windows(start, end, bt.train_months, bt.test_months)
    if not windows:
        raise WalkForwardError(
            f"span {start}..{end} is shorter than one train+test window "
            f"({bt.train_months}+{bt.test_months} months)"
        )
    assert all(w.test_end <= bt.holdout_start for w in windows), (
        "walk-forward window crossed into holdout despite the end-date clamp; generate_windows bug"
    )
    one_day = datetime.timedelta(days=1)
    results: list[WindowResult] = []
    for window in windows:
        scored: dict[GridPoint, BacktestMetrics] = {}
        for point in grid_points(config):
            train = replay(
                prepared,
                apply_grid_point(config, point),
                start=window.train_start,
                end=window.train_end - one_day,
            )
            scored[point] = compute_metrics(train, bt.periods_per_year)
        best = min(scored, key=lambda p: _selection_key(p, scored[p]))
        test = replay(
            prepared,
            apply_grid_point(config, best),
            start=window.test_start,
            end=window.test_end - one_day,
        )
        results.append(
            WindowResult(
                window, best, scored[best], test, compute_metrics(test, bt.periods_per_year)
            )
        )

    stitched_equity = _stitch(
        [r.test_result.equity_curve for r in results], config.portfolio.starting_balance
    )
    stitched_benchmark = _stitch(
        [r.test_result.benchmark_curve for r in results], config.portfolio.starting_balance
    )
    covered = _covered_segments(config, stitched_equity)
    if not covered:
        raise WalkForwardError(
            "stitched OOS does not fully cover any configured stress segment; a bear "
            "market that only ever appears in training data does not count as tested (spec)"
        )
    trades = tuple(t for r in results for t in r.test_result.trades)
    stitched_metrics = metrics_from_curves(
        stitched_equity,
        stitched_benchmark,
        trades,
        sum(r.test_result.buy_notional for r in results),
        sum(r.test_result.fees_paid for r in results),
        bt.periods_per_year,
    )
    return WalkForwardResult(
        windows=tuple(results),
        stitched_equity=stitched_equity,
        stitched_benchmark=stitched_benchmark,
        stitched_metrics=stitched_metrics,
        stress_segments_covered=covered,
    )
