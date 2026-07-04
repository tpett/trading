import dataclasses

import pandas as pd

from sim_helpers import EQ, frame, make_rankings, make_state, make_table
from trading.simulator.exits import evaluate_exits
from trading.simulator.state import PendingOrder, Position

DECISION = pd.Timestamp("2026-07-01", tz="UTC")


# Default entry_ts is recent (4 sessions held) so the 20-session time stop
# stays out of the way except in the tests that age the position explicitly.
OLD_ENTRY = "2026-06-01T00:00:00+00:00"  # 22 sessions before DECISION


def _position(
    symbol,
    *,
    entry_price=100.0,
    stop=94.0,
    entry_atr=4.0,
    flushed=False,
    entry_ts="2026-06-25T00:00:00+00:00",
):
    return Position(
        symbol=symbol,
        qty=1.0,
        entry_price=entry_price,
        entry_ts=entry_ts,
        entry_atr=entry_atr,
        stop_price=stop,
        flushed=flushed,
        entry_composite=0.8,
        entry_rank=1,
    )


def _row(status="tradable", composite=0.9):
    return {"status": status, "composite": composite, "raw_return_30d": 0.10}


def _rankings(bars, rows, **kwargs):
    return make_rankings(EQ, bars, make_table(rows), **kwargs)


def test_stop_loss_on_close_at_or_below_frozen_stop():
    bars = {"AAA": frame(start_price=93.0)}  # constant close 93 <= stop 94
    state = make_state(EQ, positions={"AAA": _position("AAA")})
    orders, _, _ = evaluate_exits(state, _rankings(bars, {"AAA": _row()}), EQ, DECISION)
    assert [(o.symbol, o.side, o.reason) for o in orders] == [("AAA", "sell", "stop_loss")]
    assert orders[0].decision_ts == DECISION.isoformat()


def test_no_stop_when_close_above_frozen_stop_even_in_high_vol():
    # The stop is FROZEN at entry: current volatility never widens or triggers it.
    bars = {"AAA": frame(start_price=95.0)}
    state = make_state(EQ, positions={"AAA": _position("AAA")})
    orders, _, _ = evaluate_exits(state, _rankings(bars, {"AAA": _row()}), EQ, DECISION)
    assert orders == []
    assert state.positions["AAA"].stop_price == 94.0  # untouched


def test_regime_flush_ratchets_stop_once_and_never_loosens():
    bars = {"AAA": frame(start_price=99.0)}
    state = make_state(EQ, positions={"AAA": _position("AAA")})
    rankings = _rankings(bars, {"AAA": _row()}, regime_state="risk_off")

    orders, _, _ = evaluate_exits(state, rankings, EQ, DECISION)
    # Ratchet: entry 100 - 1.0 * ATR 4 = 96; close 99 > 96 so no exit yet.
    assert orders == []
    assert state.positions["AAA"].stop_price == 96.0
    assert state.positions["AAA"].flushed is True

    # Second risk_off run must not re-apply; recovery must not loosen.
    state.positions["AAA"] = dataclasses.replace(state.positions["AAA"], stop_price=98.0)
    evaluate_exits(state, rankings, EQ, DECISION)
    assert state.positions["AAA"].stop_price == 98.0
    recovered = _rankings(bars, {"AAA": _row()}, regime_state="risk_on")
    evaluate_exits(state, recovered, EQ, DECISION)
    assert state.positions["AAA"].stop_price == 98.0


def test_regime_flush_can_trigger_immediate_stop():
    bars = {"AAA": frame(start_price=95.0)}  # close 95 < ratcheted 96
    state = make_state(EQ, positions={"AAA": _position("AAA")})
    rankings = _rankings(bars, {"AAA": _row()}, regime_state="risk_off")
    orders, _, _ = evaluate_exits(state, rankings, EQ, DECISION)
    assert [(o.symbol, o.reason) for o in orders] == [("AAA", "stop_loss")]


def test_trend_break_requires_bottom_half_rank_and_close_below_mean():
    # 20-day mean of a downtrending series sits above the last close.
    down = frame(drift=-0.01, start_price=200.0)
    bars = {"AAA": down, "BBB": frame(), "CCC": frame(), "DDD": frame()}
    rows = {
        "BBB": _row(composite=0.9),
        "CCC": _row(composite=0.8),
        "DDD": _row(composite=0.7),
        "AAA": _row(composite=0.1),  # rank 4 of 4: bottom half
    }
    state = make_state(EQ, positions={"AAA": _position("AAA", stop=1.0)})  # stop can't fire
    orders, _, _ = evaluate_exits(state, _rankings(bars, rows), EQ, DECISION)
    assert [(o.symbol, o.reason) for o in orders] == [("AAA", "trend_break")]


def test_no_trend_break_when_bottom_half_but_above_mean():
    up = frame(drift=0.01)  # rising: close above its 20-day mean
    bars = {"AAA": up, "BBB": frame(), "CCC": frame(), "DDD": frame()}
    rows = {
        "BBB": _row(composite=0.9),
        "CCC": _row(composite=0.8),
        "DDD": _row(composite=0.7),
        "AAA": _row(composite=0.1),
    }
    state = make_state(EQ, positions={"AAA": _position("AAA", stop=1.0)})
    orders, _, _ = evaluate_exits(state, _rankings(bars, rows), EQ, DECISION)
    assert orders == []


def test_time_stop_fires_only_when_flat_to_down():
    # Held > 20 sessions (entry June 1, bars through July 1 = 22 sessions after).
    flat = frame(start_price=100.0)
    state = make_state(
        EQ, positions={"AAA": _position("AAA", entry_price=100.0, stop=1.0, entry_ts=OLD_ENTRY)}
    )
    rows = {"AAA": _row(composite=0.9)}
    orders, _, _ = evaluate_exits(state, _rankings({"AAA": flat}, rows), EQ, DECISION)
    assert [(o.symbol, o.reason) for o in orders] == [("AAA", "time_stop")]

    # Same age but profitable: dead-money rule does not fire.
    state = make_state(
        EQ, positions={"AAA": _position("AAA", entry_price=90.0, stop=1.0, entry_ts=OLD_ENTRY)}
    )
    orders, _, _ = evaluate_exits(state, _rankings({"AAA": flat}, rows), EQ, DECISION)
    assert orders == []


def test_young_position_has_no_time_stop():
    # Default _position entry_ts is 4 sessions old; flat at entry price would
    # trip the time stop only after 20 sessions.
    state = make_state(EQ, positions={"AAA": _position("AAA", stop=1.0)})
    orders, _, _ = evaluate_exits(state, _rankings({"AAA": frame()}, {"AAA": _row()}), EQ, DECISION)
    assert orders == []


def test_forced_exit_on_sell_only_untradable_and_delisted():
    bars = {"AAA": frame(), "BBB": frame()}
    rows = {"AAA": _row(status="sell_only"), "BBB": _row(status="untradable")}
    state = make_state(
        EQ,
        positions={
            "AAA": _position("AAA", stop=1.0),
            "BBB": _position("BBB", stop=1.0),
            "GONE": _position("GONE", stop=1.0),  # absent everywhere: delisted
        },
    )
    orders, _, _ = evaluate_exits(state, _rankings(bars, rows), EQ, DECISION)
    assert {(o.symbol, o.reason) for o in orders} == {
        ("AAA", "forced_exit"),
        ("BBB", "forced_exit"),
        ("GONE", "forced_exit"),
    }


def test_quarantined_held_symbol_warns_and_holds():
    state = make_state(EQ, positions={"AAA": _position("AAA")})
    rankings = make_rankings(
        EQ, {"BBB": frame()}, make_table({"BBB": _row()}), quarantined=("AAA",)
    )
    orders, skips, warnings = evaluate_exits(state, rankings, EQ, DECISION)
    assert orders == []
    assert any(s.symbol == "AAA" and s.reason == "quarantined_no_trades" for s in skips)
    assert any("AAA" in w for w in warnings)
    assert "AAA" in state.positions


def test_pending_sell_is_not_duplicated():
    state = make_state(EQ, positions={"AAA": _position("AAA")})
    state.pending_orders = [
        PendingOrder(
            symbol="AAA",
            side="sell",
            notional=0.0,
            decision_ts="2026-06-30T00:00:00+00:00",
            reason="stop_loss",
        )
    ]
    bars = {"AAA": frame(start_price=50.0)}  # way below stop
    orders, _, _ = evaluate_exits(state, _rankings(bars, {"AAA": _row()}), EQ, DECISION)
    assert orders == []
