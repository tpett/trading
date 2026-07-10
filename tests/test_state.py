import pytest

from trading.simulator.state import (
    PendingOrder,
    Position,
    Settlement,
    StateError,
    initial_state,
    state_from_dict,
    to_state_dict,
)


def _populated_state():
    state = initial_state("equities", 1000.0, 620.55, "2026-07-01T22:30:00+00:00")
    state.cash = 640.0
    state.settlements = [Settlement(amount=180.0, available_on="2026-07-03")]
    state.positions = {
        "AAPL": Position(
            symbol="AAPL",
            qty=0.85,
            entry_price=211.5,
            entry_ts="2026-06-25T00:00:00+00:00",
            entry_atr=4.2,
            stop_price=205.2,
            flushed=False,
            entry_composite=0.83,
            entry_rank=2,
        )
    }
    state.pending_orders = [
        PendingOrder(
            symbol="NVDA",
            side="buy",
            notional=180.0,
            decision_ts="2026-07-01T00:00:00+00:00",
            reason="entry",
            atr_at_decision=6.1,
            composite=0.91,
            rank=1,
        ),
        PendingOrder(
            symbol="MSFT",
            side="sell",
            notional=0.0,
            decision_ts="2026-07-01T00:00:00+00:00",
            reason="stop_loss",
        ),
    ]
    state.cooldowns = {"TSLA": "2026-07-05"}
    state.high_water_mark = 1050.0
    state.breaker_tripped = True
    state.breaker_tripped_at = "2026-06-30T00:00:00+00:00"
    state.last_run_key = "equities:2026-06-30T00:00:00+00:00"
    return state


def test_initial_state_shape():
    state = initial_state("crypto", 1000.0, 67000.0, "2026-07-01T06:00:00+00:00")
    assert state.venue == "crypto"
    assert state.cash == 1000.0
    assert state.high_water_mark == 1000.0
    assert state.positions == {}
    assert state.pending_orders == []
    assert state.settlements == []
    assert state.cooldowns == {}
    assert state.breaker_tripped is False
    assert state.breaker_tripped_at is None
    assert state.benchmark_start_price == 67000.0
    assert state.last_run_key is None


def test_dict_round_trip_preserves_everything():
    state = _populated_state()
    payload = to_state_dict(state)
    assert payload["version"] == 1
    restored = state_from_dict(payload)
    assert restored == state


def test_round_trip_survives_json():
    import json

    state = _populated_state()
    restored = state_from_dict(json.loads(json.dumps(to_state_dict(state))))
    assert restored == state


def test_missing_key_raises_state_error():
    payload = to_state_dict(_populated_state())
    del payload["cash"]
    with pytest.raises(StateError, match="corrupt"):
        state_from_dict(payload)


def test_malformed_position_raises_state_error():
    payload = to_state_dict(_populated_state())
    payload["positions"]["AAPL"] = {"symbol": "AAPL"}  # missing fields
    with pytest.raises(StateError, match="corrupt"):
        state_from_dict(payload)


def test_unknown_version_raises_state_error():
    payload = to_state_dict(_populated_state())
    payload["version"] = 99
    with pytest.raises(StateError, match="version"):
        state_from_dict(payload)


def test_position_without_entry_fee_loads_as_zero():
    # Old (pre-M3) state files have no entry_fee; they must load with 0.0.
    state = _populated_state()
    payload = to_state_dict(state)
    for position in payload["positions"].values():
        del position["entry_fee"]
    restored = state_from_dict(payload)
    assert all(p.entry_fee == 0.0 for p in restored.positions.values())


def test_position_without_peak_close_loads_as_none():
    # Old (pre-trailing-exit) state files have no peak_close; they must load as None.
    state = _populated_state()
    payload = to_state_dict(state)
    for position in payload["positions"].values():
        del position["peak_close"]
    restored = state_from_dict(payload)
    assert all(p.peak_close is None for p in restored.positions.values())


def test_bare_last_rebalance_month_round_trips():
    state = _populated_state()
    state.bare_last_rebalance_month = "2026-07"
    restored = state_from_dict(to_state_dict(state))
    assert restored == state


def test_state_without_bare_last_rebalance_month_loads_as_none():
    # Old (pre-R2) state files have no bare_last_rebalance_month key.
    payload = to_state_dict(_populated_state())
    del payload["bare_last_rebalance_month"]
    restored = state_from_dict(payload)
    assert restored.bare_last_rebalance_month is None
