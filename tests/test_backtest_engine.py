import datetime
import math

import pandas as pd
import pytest

from backtest_helpers import (
    FakeBacktestAdapter,
    noisy_frame,
    prepared_from_sessions,
    small_config,
)
from sim_helpers import make_table
from trading.backtest.engine import (
    CRYPTO_SURVIVORSHIP_CAVEAT,
    BacktestError,
    prepare,
    replay,
)
from trading.data.cache import OhlcvCache

START = datetime.date(2025, 2, 1)
END = datetime.date(2025, 4, 30)


def _fixture_frames() -> dict[str, pd.DataFrame]:
    return {
        "AAA": noisy_frame(seed=1, drift=0.01),  # strong: should get entered
        "BBB": noisy_frame(seed=2, drift=0.002),
        "CCC": noisy_frame(seed=3, drift=0.0),
        "DDD": noisy_frame(seed=4, drift=-0.003),
        "BENCH": noisy_frame(seed=9, drift=0.003),
    }


def _prepare(tmp_path, frames=None, members_on=None):
    config = small_config()
    config = _with_benchmark(config)
    adapter = FakeBacktestAdapter(frames or _fixture_frames(), "BENCH", members_on)
    cache = OhlcvCache(tmp_path / "cache", config.data.refetch_days)
    return config, prepare(config, adapter, cache, START, END)


def _with_benchmark(config):
    from dataclasses import replace

    return replace(config, benchmark="BENCH")


def test_prepare_builds_a_session_per_benchmark_bar(tmp_path):
    config, prepared = _prepare(tmp_path)
    assert prepared.venue == config.name
    session_dates = [s.ts.date() for s in prepared.sessions]
    assert session_dates[0] >= START and session_dates[-1] <= END
    assert len(session_dates) == len(set(session_dates))
    assert all(s.rankings is not None or s.skip_reason for s in prepared.sessions)
    assert prepared.missing_symbols == ()


def test_replay_is_deterministic_and_trades(tmp_path):
    config, prepared = _prepare(tmp_path)
    first = replay(prepared, config)
    second = replay(prepared, config)
    assert first.trades == second.trades
    assert first.equity_curve.equals(second.equity_curve)
    assert first.sessions_run > 0
    assert len(first.equity_curve) == first.sessions_run
    # The fixture has a strongly drifting symbol and a risk-on benchmark:
    # the replay must actually trade. If this fails, raise AAA's drift.
    assert first.trades or first.open_positions


def test_no_lookahead_perturbing_future_data_never_changes_past_decisions(tmp_path):
    cutoff = pd.Timestamp("2025-03-15", tz="UTC")
    frames = _fixture_frames()
    perturbed = {}
    for symbol, df in frames.items():
        bumped = df.copy()
        after = bumped.index > cutoff
        bumped.loc[after, ["open", "high", "low", "close"]] *= 1.5
        perturbed[symbol] = bumped

    config, prepared_a = _prepare(tmp_path / "a", frames)
    _, prepared_b = _prepare(tmp_path / "b", perturbed)
    result_a = replay(prepared_a, config)
    result_b = replay(prepared_b, config)

    def fills_through(result):
        return [t for t in result.trades if pd.Timestamp(t.exit_ts) <= cutoff]

    assert fills_through(result_a) == fills_through(result_b)
    curve_a = result_a.equity_curve[result_a.equity_curve.index <= cutoff]
    curve_b = result_b.equity_curve[result_b.equity_curve.index <= cutoff]
    assert curve_a.equals(curve_b)


def test_member_leaving_universe_is_force_exited(tmp_path):
    drop_after = datetime.date(2025, 3, 15)

    def members_on(as_of: datetime.date) -> list[str]:
        names = ["AAA", "BBB", "CCC", "DDD"]
        return [n for n in names if n != "AAA" or as_of <= drop_after]

    config, prepared = _prepare(tmp_path, members_on=members_on)
    result = replay(prepared, config)
    forced = [t for t in result.trades if t.reason == "forced_exit"]
    assert any(t.symbol == "AAA" for t in forced), (
        "AAA (strongest drift) should be held when it leaves the universe and "
        "then force-exited; if it was never entered, raise its drift in the fixture"
    )


def test_thin_session_is_skipped_and_state_carries_over(tmp_path):
    thin_day = datetime.date(2025, 3, 10)

    ghosts = ["GHOST1", "GHOST2", "GHOST3", "GHOST4", "GHOST5"]

    def members_on(as_of: datetime.date) -> list[str]:
        if as_of == thin_day:
            return ["AAA", "BBB", "CCC", "DDD", *ghosts]  # 4 of 9 have data -> 0.44 < 0.5 floor
        return ["AAA", "BBB", "CCC", "DDD"]

    config, prepared = _prepare(tmp_path, members_on=members_on)
    skipped = [s for s in prepared.sessions if s.skip_reason is not None]
    assert [s.ts.date() for s in skipped] == [thin_day]
    assert skipped[0].survivorship_ratio == pytest.approx(4 / 9)
    result = replay(prepared, config)
    assert any(str(thin_day) in entry for entry in result.sessions_skipped)
    # GHOST symbols never had data: counted as survivorship gaps.
    assert set(prepared.missing_symbols) == set(ghosts)


def test_crypto_results_carry_survivorship_caveat(tmp_path):
    config, prepared = _prepare(tmp_path)  # small_config is crypto-based
    result = replay(prepared, config)
    assert CRYPTO_SURVIVORSHIP_CAVEAT in result.warnings
    assert 0.0 < result.survivorship_ratio <= 1.0


def test_replay_window_slicing_runs_fresh_state_per_window(tmp_path):
    config, prepared = _prepare(tmp_path)
    late = replay(prepared, config, start=datetime.date(2025, 4, 1))
    assert late.equity_curve.index[0] >= pd.Timestamp("2025-04-01", tz="UTC")
    # Fresh state: the window's first marked value is the starting balance
    # (no fills can exist on the first session -- there are no pending orders),
    # not the full run's marked value on that date.
    assert late.equity_curve.iloc[0] == pytest.approx(config.portfolio.starting_balance)


def test_replay_with_no_sessions_raises(tmp_path):
    config, prepared = _prepare(tmp_path)
    try:
        replay(prepared, config, start=datetime.date(2030, 1, 1), end=datetime.date(2030, 2, 1))
        raise AssertionError("expected BacktestError")
    except BacktestError:
        pass


def test_entry_exit_round_trip_pairs_into_trade_records():
    # Hand-built sessions: deterministic entry then stop-out, no feature math.
    config = _with_benchmark(small_config())
    bars = {s: noisy_frame(seed=i, drift=0.001) for i, s in enumerate(["AAA", "BBB"], start=1)}
    bench = noisy_frame(seed=9, drift=0.001)
    enter = make_table(
        {
            "AAA": {"status": "tradable", "composite": 0.9, "raw_return_30d": 0.5},
            "BBB": {"status": "tradable", "composite": 0.2, "raw_return_30d": 0.1},
        }
    )
    neutral = make_table(
        {
            "AAA": {"status": "tradable", "composite": 0.9, "raw_return_30d": 0.5},
            "BBB": {"status": "tradable", "composite": 0.2, "raw_return_30d": 0.1},
        }
    )
    force = make_table(
        {
            "AAA": {"status": "untradable", "composite": 0.9, "raw_return_30d": 0.5},
            "BBB": {"status": "tradable", "composite": 0.2, "raw_return_30d": 0.1},
        }
    )
    prepared = prepared_from_sessions(
        config,
        [
            ("2025-03-01", enter),
            ("2025-03-02", neutral),
            ("2025-03-03", force),
            ("2025-03-04", neutral),
        ],
        bars,
        bench,
    )
    result = replay(prepared, config)
    assert [t.symbol for t in result.trades] == ["AAA"]
    trade = result.trades[0]
    assert trade.reason == "forced_exit"
    assert trade.entry_fee > 0.0  # crypto taker fee, frozen at entry (Task 2)
    assert math.isclose(
        trade.realized_pnl,
        trade.qty * trade.exit_price
        - trade.exit_fee
        - trade.qty * trade.entry_price
        - trade.entry_fee,
        rel_tol=1e-9,
    )
    assert result.fees_paid > 0.0 and result.buy_notional > 0.0
