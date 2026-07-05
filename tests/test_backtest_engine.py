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
from sim_helpers import make_rankings, make_table
from trading.backtest.engine import (
    CRYPTO_SURVIVORSHIP_CAVEAT,
    BacktestError,
    PreparedBacktest,
    SessionPlan,
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
    """A session where most LISTED members lack a bar covering it (exchange
    hole, mid-window delisting) is skipped; state carries over flat."""
    thin_day = datetime.date(2025, 3, 10)
    thin_ts = pd.Timestamp(thin_day, tz="UTC")
    frames = _fixture_frames()
    for symbol in ("BBB", "CCC", "DDD"):  # listed (first bar long before), hole on thin_day
        frames[symbol] = frames[symbol].drop(thin_ts)

    config, prepared = _prepare(tmp_path, frames)
    skipped = [s for s in prepared.sessions if s.skip_reason is not None]
    assert [s.ts.date() for s in skipped] == [thin_day]  # 1 of 4 eligible -> 0.25 < 0.5 floor
    assert skipped[0].survivorship_ratio == pytest.approx(1 / 4)
    result = replay(prepared, config)
    assert any(str(thin_day) in entry for entry in result.sessions_skipped)
    # Both curves are defined on the SAME session index: a skipped session's
    # value is carried forward (flat -- nothing traded), never omitted, so
    # downstream index-aligned arithmetic never sees silent NaNs.
    assert result.equity_curve.index.equals(result.benchmark_curve.index)
    position = result.equity_curve.index.get_loc(thin_ts)
    assert position > 0
    assert result.equity_curve.iloc[position] == result.equity_curve.iloc[position - 1]
    # sessions_run counts only sessions the simulator actually stepped.
    assert result.sessions_run == len(result.equity_curve) - 1


def test_never_listed_members_do_not_gate_coverage_but_stay_visible(tmp_path):
    """Members with NO data in the fetched window (post-window listings in
    crypto's today-snapshot universe) are excluded from the skip gate's
    denominator, but stay in the reported survivorship ratio and the missing
    list -- the shrinking historical universe is visible, never a skip."""
    ghosts = ["GHOST1", "GHOST2", "GHOST3", "GHOST4", "GHOST5"]

    def members_on(as_of: datetime.date) -> list[str]:
        return ["AAA", "BBB", "CCC", "DDD", *ghosts]  # 4/9 = 0.44 < 0.5: old gate skipped ALL

    config, prepared = _prepare(tmp_path, members_on=members_on)
    assert all(s.skip_reason is None for s in prepared.sessions)
    assert all(s.survivorship_ratio == pytest.approx(4 / 9) for s in prepared.sessions)
    assert set(prepared.missing_symbols) == set(ghosts)
    result = replay(prepared, config)
    assert result.sessions_run == len(prepared.sessions)
    assert result.survivorship_ratio == pytest.approx(4 / 9)


def test_mid_window_listing_shrinks_coverage_denominator(tmp_path):
    """Regression for the 2023 crypto smoke: members whose first available
    bar postdates a session (listed mid-window) must not count against that
    session's coverage; they join the denominator once their data begins."""
    listing_ts = pd.Timestamp("2025-03-15", tz="UTC")
    frames = {
        "AAA": noisy_frame(seed=1, drift=0.01),
        "BBB": noisy_frame(seed=2, drift=0.002),
        "CCC": noisy_frame(seed=3, drift=0.0),
        "EEE": noisy_frame(seed=5, start="2025-03-15", periods=60),
        "FFF": noisy_frame(seed=6, start="2025-03-15", periods=60),
        "BENCH": noisy_frame(seed=9, drift=0.003),
    }
    config = _with_benchmark(small_config(min_session_coverage=0.9))
    adapter = FakeBacktestAdapter(frames, "BENCH")
    cache = OhlcvCache(tmp_path / "cache", config.data.refetch_days)
    prepared = prepare(config, adapter, cache, START, END)

    early = [s for s in prepared.sessions if s.ts < listing_ts]
    late = [s for s in prepared.sessions if s.ts >= listing_ts]
    assert early and late
    # Old gate: 3/5 = 60% < 90% floor -> every early session skipped.
    assert all(s.skip_reason is None for s in early)
    assert all(s.survivorship_ratio == pytest.approx(3 / 5) for s in early)
    assert all(s.skip_reason is None for s in late)
    assert all(s.survivorship_ratio == pytest.approx(5 / 5) for s in late)
    # The shrinking gate denominator is surfaced per session and on the result.
    assert all(s.eligible_members == 3 for s in early)
    assert all(s.eligible_members == 5 for s in late)
    result = replay(prepared, config)
    assert result.eligible_min == 3
    assert 3 < result.eligible_mean < 5


def test_point_in_time_member_with_late_data_degrades_coverage(tmp_path):
    """Equities-style contrast: with a PIT universe, membership is the
    listing signal. A member whose bars start after the session is a DATA
    outage -- it stays in the eligible denominator and counts as a miss, so
    coverage drops (and skips below the floor) instead of the denominator
    silently shrinking."""
    from dataclasses import replace

    listing_ts = pd.Timestamp("2025-03-15", tz="UTC")
    frames = {
        "AAA": noisy_frame(seed=1, drift=0.01),
        "BBB": noisy_frame(seed=2, drift=0.002),
        "CCC": noisy_frame(seed=3, drift=0.0),
        "EEE": noisy_frame(seed=5, start="2025-03-15", periods=60),
        "FFF": noisy_frame(seed=6, start="2025-03-15", periods=60),
        "BENCH": noisy_frame(seed=9, drift=0.003),
    }
    config = _with_benchmark(small_config(min_session_coverage=0.9))
    config = replace(config, universe=replace(config.universe, point_in_time=True))
    adapter = FakeBacktestAdapter(frames, "BENCH")
    cache = OhlcvCache(tmp_path / "cache", config.data.refetch_days)
    prepared = prepare(config, adapter, cache, START, END)

    early = [s for s in prepared.sessions if s.ts < listing_ts]
    late = [s for s in prepared.sessions if s.ts >= listing_ts]
    assert early and late
    # Same fixture as the crypto test above, opposite verdict: 3/5 = 60% < 90%.
    assert all(s.skip_reason is not None and "coverage 60%" in s.skip_reason for s in early)
    assert all(s.eligible_members == 5 for s in early)  # PIT members always count
    assert all(s.skip_reason is None for s in late)


def test_crypto_results_carry_survivorship_caveat(tmp_path):
    config, prepared = _prepare(tmp_path)  # small_config is crypto-based
    result = replay(prepared, config)
    assert CRYPTO_SURVIVORSHIP_CAVEAT in result.warnings
    assert 0.0 < result.survivorship_ratio <= 1.0


def test_one_prepare_serves_the_grid_without_bleed_between_replays(tmp_path):
    # Grid-reuse proof: the same PreparedBacktest replayed under two different
    # entry_score_threshold configs must diverge where expected, and rerunning
    # the first config must reproduce its original result exactly (no state
    # bleeding through the shared prepared object).
    from dataclasses import replace

    config, prepared = _prepare(tmp_path)
    strict = replace(config, portfolio=replace(config.portfolio, entry_score_threshold=1.01))
    baseline = replay(prepared, config)
    blocked = replay(prepared, strict)
    assert baseline.trades or baseline.open_positions
    assert blocked.trades == () and blocked.open_positions == ()
    assert not baseline.equity_curve.equals(blocked.equity_curve)
    again = replay(prepared, config)
    assert again.trades == baseline.trades
    assert again.open_positions == baseline.open_positions
    assert again.equity_curve.equals(baseline.equity_curve)


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


def test_quarantine_of_held_symbol_defers_fill_and_marks_at_entry_price():
    """Live parity (replay/live seam): live's RankingsResult.bars excludes
    quarantined symbols even when held, so a pending sell can't find a fill
    bar (deferred) and marking falls back to entry price. Replay must match
    -- not fill at the (possibly outlier) print that triggered quarantine."""
    from dataclasses import replace

    config = _with_benchmark(small_config())
    bars = {s: noisy_frame(seed=i, drift=0.001) for i, s in enumerate(["AAA", "BBB"], start=1)}
    bench = noisy_frame(seed=9, drift=0.001)
    enter = make_table(
        {
            "AAA": {"status": "tradable", "composite": 0.9, "raw_return_30d": 0.5},
            "BBB": {"status": "tradable", "composite": 0.2, "raw_return_30d": 0.1},
        }
    )
    forced_sell = make_table(
        {
            "AAA": {"status": "sell_only", "composite": 0.9, "raw_return_30d": 0.5},
            "BBB": {"status": "tradable", "composite": 0.2, "raw_return_30d": 0.1},
        }
    )
    quarantined_session = make_table(
        {"BBB": {"status": "tradable", "composite": 0.2, "raw_return_30d": 0.1}}
    )
    session_specs = [
        # entry: AAA's composite clears threshold -> pending buy written.
        ("2025-03-01", enter, ("AAA", "BBB"), ()),
        # AAA's buy fills here (bar after the decision bar above).
        ("2025-03-02", enter, ("AAA", "BBB"), ()),
        # forced exit: AAA turns sell_only -> pending sell written.
        ("2025-03-03", forced_sell, ("AAA", "BBB"), ()),
        # AAA is quarantined THIS session: its pending sell has no bar to
        # fill against (live parity), so it must defer, and the still-open
        # position must mark at entry price, not a bar that isn't there.
        ("2025-03-04", quarantined_session, ("BBB",), ("AAA",)),
    ]
    sessions = []
    for iso, table, clean_symbols, quarantined in session_specs:
        ts = pd.Timestamp(iso, tz="UTC")
        rankings = make_rankings(
            config, {s: bars[s].loc[:ts] for s in bars}, table, quarantined=quarantined
        )
        slim = replace(rankings, bars={}, benchmark_bars=bench.iloc[0:0])
        sessions.append(
            SessionPlan(
                ts=ts,
                rankings=slim,
                clean_symbols=clean_symbols,
                survivorship_ratio=1.0,
                eligible_members=len(clean_symbols),
                skip_reason=None,
            )
        )
    prepared = PreparedBacktest(
        venue=config.name,
        start=sessions[0].ts.date(),
        end=sessions[-1].ts.date(),
        sessions=tuple(sessions),
        bars=bars,
        benchmark_bars=bench,
        missing_symbols=(),
    )

    result = replay(prepared, config)
    # Deferred, not filled: AAA is still open, no exit trade recorded for it.
    assert "AAA" in result.open_positions
    assert not any(t.symbol == "AAA" for t in result.trades)
    # Marked at entry price the quarantined session -- the live fallback.
    assert any("AAA" in w and "using entry price" in w for w in result.warnings)
