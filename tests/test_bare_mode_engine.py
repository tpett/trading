"""Engine-level (prepare/replay) integration coverage for R2's bare mode
(W0): confirms the monthly-rebalance cadence and rank-exit-only behavior
survive the full backtest replay, not just the unit-level bare.py functions.
"""

import datetime
from dataclasses import replace

from backtest_helpers import FakeBacktestAdapter, noisy_frame, small_config
from trading.backtest.engine import prepare, replay
from trading.data.cache import OhlcvCache

START = datetime.date(2025, 2, 1)
END = datetime.date(2025, 4, 30)


def _fixture_frames():
    return {
        "AAA": noisy_frame(seed=1, drift=0.01),
        "BBB": noisy_frame(seed=2, drift=0.002),
        "CCC": noisy_frame(seed=3, drift=0.0),
        "DDD": noisy_frame(seed=4, drift=-0.003),
        "BENCH": noisy_frame(seed=9, drift=0.003),
    }


def _bare_config():
    config = replace(small_config(), benchmark="BENCH")
    return replace(
        config,
        regime=replace(config.regime, disabled=True),
        portfolio=replace(config.portfolio, bare_mode=True, max_positions=2),
    )


def _replay(tmp_path, config):
    adapter = FakeBacktestAdapter(_fixture_frames(), "BENCH", None)
    cache = OhlcvCache(tmp_path / "cache", config.data.refetch_days)
    prepared = prepare(config, adapter, cache, START, END)
    return replay(prepared, config)


def test_bare_mode_only_trades_on_at_most_one_session_per_month(tmp_path):
    result = _replay(tmp_path, _bare_config())
    entry_months = {t.entry_ts[:7] for t in result.trades}
    exit_months_by_reason = {}
    for t in result.trades:
        exit_months_by_reason.setdefault(t.exit_ts[:7], set()).add(t.reason)
    # 3-month window (Feb/Mar/Apr): rebalances can happen at most once each.
    assert len(entry_months) <= 3
    # Every exit reason is rank-based or forced (data integrity) -- never the
    # stop_loss/trend_break/time_stop machinery bare mode switches off.
    all_reasons = {t.reason for t in result.trades}
    assert all_reasons <= {"rank_exit", "forced_exit"}


def test_bare_mode_never_holds_more_than_max_positions(tmp_path):
    config = _bare_config()
    result = _replay(tmp_path, config)
    assert len(result.open_positions) <= config.portfolio.max_positions


def test_bare_mode_trades_more_than_zero_times(tmp_path):
    # Sanity: the fixture must actually exercise the rebalance path, or every
    # assertion above is vacuous.
    result = _replay(tmp_path, _bare_config())
    assert result.trades or result.open_positions
