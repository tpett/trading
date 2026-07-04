import copy

import pandas as pd
import pytest

from sim_helpers import EQ, frame, make_rankings, make_state, make_table
from trading.simulator.core import decision_bar, make_run_key, step
from trading.simulator.state import Position

JUL1 = pd.Timestamp("2026-07-01", tz="UTC")
JUL2 = pd.Timestamp("2026-07-02", tz="UTC")


def _row(composite=0.9):
    return {"status": "tradable", "composite": composite, "raw_return_30d": 0.10}


def _rankings(end="2026-07-01", **kwargs):
    bars = {"AAA": frame(end=end), "BBB": frame(end=end, start_price=50.0)}
    rows = {"AAA": _row(0.95), "BBB": _row(0.60)}  # BBB below entry threshold
    return make_rankings(EQ, bars, make_table(rows), **kwargs)


def test_decision_bar_is_max_last_bar_and_run_key_format():
    rankings = _rankings()
    assert decision_bar(rankings) == JUL1
    assert make_run_key("equities", JUL1) == "equities:2026-07-01T00:00:00+00:00"


def test_step_is_pure_and_deterministic():
    state = make_state(EQ)
    rankings = _rankings()
    before = copy.deepcopy(state)
    a = step(state, rankings, EQ)
    b = step(state, rankings, EQ)
    assert state == before  # input state untouched
    assert a.state == b.state
    assert a.fills == b.fills
    assert a.new_orders == b.new_orders
    assert a.skips == b.skips


def test_two_run_lifecycle_decides_then_fills():
    # Run 1 (decision bar Jul 1): no fills, one pending entry for AAA.
    state = make_state(EQ)
    first = step(state, _rankings(), EQ)
    assert first.run_key == "equities:2026-07-01T00:00:00+00:00"
    assert first.fills == ()
    assert [(o.symbol, o.side) for o in first.new_orders] == [("AAA", "buy")]
    assert first.state.positions == {}
    assert first.state.last_run_key == first.run_key

    # Run 2 (decision bar Jul 2): the pending order fills at Jul 2's open.
    second = step(first.state, _rankings(end="2026-07-02"), EQ)
    assert second.run_key == "equities:2026-07-02T00:00:00+00:00"
    assert [f.symbol for f in second.fills] == ["AAA"]
    assert second.fills[0].bar_ts == "2026-07-02T00:00:00+00:00"
    assert "AAA" in second.state.positions
    # AAA is now held: no re-entry (never average down).
    assert all(o.symbol != "AAA" or o.side != "buy" for o in second.state.pending_orders)


def test_snapshot_marks_positions_at_decision_close():
    state = make_state(EQ)
    first = step(state, _rankings(), EQ)
    second = step(first.state, _rankings(end="2026-07-02"), EQ)
    snap = second.snapshot
    position = second.state.positions["AAA"]
    assert snap.cash == pytest.approx(second.state.cash)
    mark = snap.positions[0]
    assert mark.symbol == "AAA"
    assert mark.last_close == pytest.approx(100.0)
    assert snap.value == pytest.approx(snap.cash + snap.unsettled + position.qty * 100.0)
    assert mark.stop_distance_pct == pytest.approx((100.0 - position.stop_price) / 100.0)


def test_breaker_trips_on_drawdown_and_blocks_entries():
    state = make_state(EQ, cash=700.0, high_water_mark=1000.0)  # 30% drawdown > 20%
    result = step(state, _rankings(), EQ)
    assert result.breaker_tripped_now is True
    assert result.state.breaker_tripped is True
    assert result.state.breaker_tripped_at == JUL1.isoformat()
    assert result.new_orders == ()
    assert any(s.reason == "circuit_breaker" for s in result.skips)

    # Already-tripped breaker does not re-fire the notification flag.
    again = step(result.state, _rankings(end="2026-07-02"), EQ)
    assert again.breaker_tripped_now is False
    assert again.state.breaker_tripped is True


def test_high_water_mark_ratchets_up():
    state = make_state(EQ, cash=1200.0, high_water_mark=1000.0)
    result = step(state, _rankings(), EQ)
    assert result.state.high_water_mark == pytest.approx(1200.0)


def test_stale_run_skips_entries_but_still_fills_and_exits():
    state = make_state(EQ)
    first = step(state, _rankings(), EQ)
    second = step(
        first.state,
        _rankings(end="2026-07-02"),
        EQ,
        allow_entries=False,
        stale_reason="stale_run_entries_skipped",
    )
    assert [f.symbol for f in second.fills] == ["AAA"]  # fills still processed
    assert any(s.reason == "stale_run_entries_skipped" for s in second.skips)
    assert all(o.side != "buy" for o in second.state.pending_orders)


def test_step_decisions_track_the_decision_bar():
    """No-lookahead companion: entry sizing must be a function of the mark at
    the decision bar — crash a held name's decision-bar close and the buy
    order sized off portfolio value must shrink. (The plan's original fixture
    crashed only the candidate's close, which with a hardcoded composite left
    the ATR and dollar-volume gates numerically unaffected; redesigned to pin
    the same property through the mark-to-market sizing channel.)"""
    held = Position(
        symbol="XXX",
        qty=2.0,
        entry_price=100.0,
        entry_ts="2026-06-26T00:00:00+00:00",  # recent: time stop can't fire
        entry_atr=4.0,
        stop_price=10.0,  # far below the crashed close: stop can't fire
        flushed=False,
        entry_composite=0.9,
        entry_rank=1,
    )

    def rankings(crash: bool):
        bars = {"AAA": frame(), "XXX": frame()}
        if crash:
            bars["XXX"].loc[bars["XXX"].index[-1], "close"] = 50.0  # -50% decision bar
        return make_rankings(EQ, bars, make_table({"AAA": _row(0.95)}))

    state = make_state(EQ, positions={"XXX": held})
    base = step(state, rankings(crash=False), EQ)
    crashed = step(state, rankings(crash=True), EQ)

    base_buy = next(o for o in base.new_orders if o.side == "buy" and o.symbol == "AAA")
    crash_buy = next(o for o in crashed.new_orders if o.side == "buy" and o.symbol == "AAA")
    pct = EQ.portfolio.position_size_pct
    assert crash_buy.notional < base_buy.notional
    assert base_buy.notional == pytest.approx(pct * (state.cash + 2.0 * 100.0), abs=1e-9)
    assert crash_buy.notional == pytest.approx(pct * (state.cash + 2.0 * 50.0), abs=1e-9)
    assert crashed.snapshot.value == pytest.approx(state.cash + 2.0 * 50.0)
