import datetime

import pytest

from backtest_helpers import FakeBacktestAdapter, noisy_frame, small_config
from trading.backtest.engine import prepare
from trading.backtest.walkforward import (
    GridPoint,
    WalkForwardError,
    add_months,
    apply_grid_point,
    generate_windows,
    grid_points,
    run_walk_forward,
)
from trading.data.cache import OhlcvCache


def test_add_months_rolls_years_and_rejects_late_month_days():
    assert add_months(datetime.date(2018, 1, 1), 30) == datetime.date(2020, 7, 1)
    assert add_months(datetime.date(2025, 11, 15), 3) == datetime.date(2026, 2, 15)
    with pytest.raises(ValueError):
        add_months(datetime.date(2018, 1, 31), 1)


def test_generate_windows_rolls_by_test_months_and_only_full_windows():
    windows = generate_windows(
        datetime.date(2018, 1, 1), datetime.date(2021, 1, 1), train_months=24, test_months=3
    )
    # 2019-01-01 would need test_end 2021-04-01 > end: excluded (full windows only).
    assert [w.train_start for w in windows] == [
        datetime.date(2018, 1, 1),
        datetime.date(2018, 4, 1),
        datetime.date(2018, 7, 1),
        datetime.date(2018, 10, 1),
    ]
    for w in windows:
        assert w.train_end == add_months(w.train_start, 24)
        assert w.test_start == w.train_end
        assert w.test_end == add_months(w.test_start, 3)
        assert w.test_end <= datetime.date(2021, 1, 1)


def test_grid_points_are_the_two_toml_grids_only():
    config = small_config()
    points = grid_points(config)
    assert len(points) == len(config.backtest.entry_score_threshold_grid) * len(
        config.backtest.stop_atr_multiple_grid
    )
    tuned = apply_grid_point(config, GridPoint(0.61, 2.5))
    assert tuned.portfolio.entry_score_threshold == 0.61
    assert tuned.portfolio.stop_atr_multiple == 2.5
    # Nothing else moved: the tunable surface is exactly two hyperparameters.
    assert tuned.portfolio.position_size_pct == config.portfolio.position_size_pct
    assert tuned.signals == config.signals


def _prepared(tmp_path):
    # 14 months of daily bars so train=6 + test=2 rolls at least twice.
    frames = {
        "AAA": noisy_frame(seed=1, drift=0.008, periods=430, start="2025-01-01"),
        "BBB": noisy_frame(seed=2, drift=0.001, periods=430, start="2025-01-01"),
        "CCC": noisy_frame(seed=3, drift=-0.002, periods=430, start="2025-01-01"),
        "BENCH": noisy_frame(seed=9, drift=0.002, periods=430, start="2025-01-01"),
    }
    config = small_config(
        train_months=6,
        test_months=2,
        entry_score_threshold_grid=(0.5, 0.7),
        stop_atr_multiple_grid=(1.0, 2.0),
        stress_segments=((datetime.date(2025, 8, 1), datetime.date(2025, 9, 30)),),
    )
    from dataclasses import replace

    config = replace(config, benchmark="BENCH")
    adapter = FakeBacktestAdapter(frames, "BENCH")
    cache = OhlcvCache(tmp_path / "cache", config.data.refetch_days)
    prepared = prepare(
        config, adapter, cache, datetime.date(2025, 2, 1), datetime.date(2026, 2, 28)
    )
    return config, prepared


def test_walk_forward_stitches_oos_segments_only(tmp_path):
    config, prepared = _prepared(tmp_path)
    wf = run_walk_forward(
        prepared, config, start=datetime.date(2025, 2, 1), end=datetime.date(2026, 2, 28)
    )
    assert len(wf.windows) >= 2
    for wr in wf.windows:
        # Stitched curve contains ONLY test-window dates (OOS), never train dates.
        curve = wr.test_result.equity_curve
        assert curve.index[0].date() >= wr.window.test_start
        assert curve.index[-1].date() < wr.window.test_end
        assert wr.best in grid_points(config)
    stitched_dates = [ts.date() for ts in wf.stitched_equity.index]
    first_test = wf.windows[0].window.test_start
    assert min(stitched_dates) >= first_test
    assert wf.stitched_metrics.trade_count == sum(w.test_metrics.trade_count for w in wf.windows)
    assert wf.stress_segments_covered  # 2025-08..09 lies inside the OOS stitch
    assert len(wf.stitched_equity) == len(wf.stitched_benchmark)


def test_walk_forward_refuses_without_stress_coverage(tmp_path):
    config, prepared = _prepared(tmp_path)
    from dataclasses import replace

    no_stress_in_span = replace(
        config,
        backtest=replace(
            config.backtest,
            stress_segments=((datetime.date(2022, 1, 1), datetime.date(2022, 12, 31)),),
        ),
    )
    with pytest.raises(WalkForwardError, match="stress"):
        run_walk_forward(
            prepared,
            no_stress_in_span,
            start=datetime.date(2025, 2, 1),
            end=datetime.date(2026, 2, 28),
        )


def test_walk_forward_span_too_short_raises(tmp_path):
    config, prepared = _prepared(tmp_path)
    with pytest.raises(WalkForwardError, match="shorter"):
        run_walk_forward(
            prepared, config, start=datetime.date(2025, 2, 1), end=datetime.date(2025, 6, 1)
        )


def test_walk_forward_clamps_end_to_holdout_start(tmp_path):
    """Holdout is a one-time-only period (spec): walk-forward must never see
    it, regardless of what `end` the caller passes in."""
    config, prepared = _prepared(tmp_path)
    from dataclasses import replace

    tight_holdout = replace(
        config, backtest=replace(config.backtest, holdout_start=datetime.date(2025, 10, 1))
    )
    wf = run_walk_forward(
        prepared, tight_holdout, start=datetime.date(2025, 2, 1), end=datetime.date(2026, 2, 28)
    )
    for wr in wf.windows:
        assert wr.window.test_end <= tight_holdout.backtest.holdout_start
    assert wf.stitched_equity.index[-1].date() < tight_holdout.backtest.holdout_start
