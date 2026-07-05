import math
from dataclasses import replace

import pandas as pd
import pytest

from sim_helpers import AS_OF, CR, EQ, frame, make_rankings, make_state, make_table
from trading.simulator.entries import evaluate_entries
from trading.simulator.state import PendingOrder, Position, Skip

DECISION = pd.Timestamp("2026-07-01", tz="UTC")
VALUE = 1000.0


def _row(status="tradable", composite=0.9, raw_return=0.10):
    return {"status": status, "composite": composite, "raw_return_30d": raw_return}


def _position(symbol):
    return Position(
        symbol=symbol,
        qty=1.0,
        entry_price=100.0,
        entry_ts="2026-06-25T00:00:00+00:00",
        entry_atr=4.0,
        stop_price=94.0,
        flushed=False,
        entry_composite=0.8,
        entry_rank=1,
    )


def _entries(config, bars, rows, state=None, regime_state="risk_on", value=VALUE):
    state = state if state is not None else make_state(config)
    rankings = make_rankings(config, bars, make_table(rows), regime_state=regime_state)
    return evaluate_entries(state, rankings, config, DECISION, value), state


def test_top_candidate_becomes_buy_order_with_decision_evidence():
    (orders, skips), _ = _entries(EQ, {"AAA": frame()}, {"AAA": _row(composite=0.95)})
    assert len(orders) == 1
    order = orders[0]
    assert order == PendingOrder(
        symbol="AAA",
        side="buy",
        notional=pytest.approx(0.18 * VALUE),
        decision_ts=DECISION.isoformat(),
        reason="entry",
        atr_at_decision=pytest.approx(4.0),  # frame(): TR = 0.04 * close = 4.0
        composite=pytest.approx(0.95),
        rank=1,
    )


def test_below_threshold_is_not_entered():
    (orders, skips), _ = _entries(EQ, {"AAA": frame()}, {"AAA": _row(composite=0.69)})
    assert orders == []
    assert skips == []  # below-threshold tail is not journal noise


def test_already_held_never_averages_down():
    state = make_state(EQ, positions={"AAA": _position("AAA")})
    (orders, skips), _ = _entries(EQ, {"AAA": frame()}, {"AAA": _row()}, state=state)
    assert orders == []
    assert Skip("AAA", "entry", "already_held") in skips


def test_non_tradable_status_blocked_from_entry():
    (orders, skips), _ = _entries(EQ, {"AAA": frame()}, {"AAA": _row(status="sell_only")})
    assert orders == []
    assert Skip("AAA", "entry", "status_sell_only") in skips


def test_cooldown_blocks_reentry_until_expiry():
    state = make_state(EQ, cooldowns={"AAA": "2026-07-05"})
    (orders, skips), _ = _entries(EQ, {"AAA": frame()}, {"AAA": _row()}, state=state)
    assert orders == []
    assert Skip("AAA", "entry", "cooldown") in skips

    # On the expiry date itself, re-entry is allowed again.
    state = make_state(EQ, cooldowns={"AAA": "2026-07-01"})
    (orders, _), _ = _entries(EQ, {"AAA": frame()}, {"AAA": _row()}, state=state)
    assert len(orders) == 1


def test_equities_dollar_volume_floor():
    thin = frame(volume=1e3)  # 100 * 1e3 = 1e5 << 2e7 floor
    (orders, skips), _ = _entries(EQ, {"AAA": thin}, {"AAA": _row()})
    assert orders == []
    assert Skip("AAA", "entry", "below_dollar_volume_floor") in skips


def test_crypto_fee_gate_requires_raw_return_multiple():
    # Round trip = 2 * (95 + 5) bps = 2%; gate = 3x = 6%.
    (orders, skips), _ = _entries(CR, {"BTC": frame()}, {"BTC": _row(raw_return=0.05)})
    assert orders == []
    assert Skip("BTC", "entry", "fee_gate") in skips

    (orders, _), _ = _entries(CR, {"BTC": frame()}, {"BTC": _row(raw_return=0.10)})
    assert len(orders) == 1


def test_fee_gate_not_applied_to_equities():
    # Equities multiple is 0.0: negative raw momentum must not block entry.
    (orders, _), _ = _entries(EQ, {"AAA": frame()}, {"AAA": _row(raw_return=-0.02)})
    assert len(orders) == 1


def test_neutral_regime_halves_position_slots():
    # floor(5 * 0.5) = 2 slots; 2 already held -> venue full.
    state = make_state(EQ, positions={"XXX": _position("XXX"), "YYY": _position("YYY")})
    bars = {"AAA": frame(), "XXX": frame(), "YYY": frame()}
    rows = {"AAA": _row(), "XXX": _row(composite=0.85), "YYY": _row(composite=0.84)}
    (orders, skips), _ = _entries(EQ, bars, rows, state=state, regime_state="neutral")
    assert orders == []
    assert Skip("AAA", "entry", "no_free_slot") in skips


def test_risk_off_blocks_all_entries():
    (orders, skips), _ = _entries(EQ, {"AAA": frame()}, {"AAA": _row()}, regime_state="risk_off")
    assert orders == []
    assert skips == [Skip("*", "entry", "regime_risk_off")]


def test_tripped_breaker_blocks_all_entries():
    state = make_state(EQ, breaker_tripped=True, breaker_tripped_at="2026-06-30T00:00:00+00:00")
    (orders, skips), _ = _entries(EQ, {"AAA": frame()}, {"AAA": _row()}, state=state)
    assert orders == []
    assert skips == [Skip("*", "entry", "circuit_breaker")]


def test_daily_deployment_cap_allows_one_equities_entry_per_day():
    bars = {"AAA": frame(), "BBB": frame()}
    rows = {"AAA": _row(composite=0.95), "BBB": _row(composite=0.94)}
    (orders, skips), _ = _entries(EQ, bars, rows)
    # 18% sizing: first entry (180) fits the 25% budget (250); second would breach it.
    assert [o.symbol for o in orders] == ["AAA"]
    assert Skip("BBB", "entry", "daily_deployment_cap") in skips


def test_crypto_full_size_entry_fits_deployment_cap():
    # Guards the Step-1 config fix: 30% sizing must clear the (raised) 35% cap
    # for exactly one entry; a second same-day entry is capped.
    bars = {"BTC": frame(), "ETH": frame()}
    rows = {"BTC": _row(composite=0.95), "ETH": _row(composite=0.94)}
    (orders, skips), _ = _entries(CR, bars, rows)
    assert [o.symbol for o in orders] == ["BTC"]
    assert orders[0].notional == pytest.approx(300.0)
    assert Skip("ETH", "entry", "daily_deployment_cap") in skips


def test_insufficient_settled_cash_blocks_entry():
    state = make_state(EQ, cash=100.0)  # value stays 1000 (rest is in settlements)
    state.settlements = []
    (orders, skips), _ = _entries(EQ, {"AAA": frame()}, {"AAA": _row()}, state=state)
    assert orders == []
    assert Skip("AAA", "entry", "insufficient_settled_cash") in skips


def test_atr_unavailable_blocks_entry():
    short = frame(periods=10)  # < atr_window + 1
    (orders, skips), _ = _entries(EQ, {"AAA": short}, {"AAA": _row()})
    assert orders == []
    assert Skip("AAA", "entry", "insufficient_history_for_atr") in skips


def test_nan_composite_stops_iteration():
    (orders, skips), _ = _entries(EQ, {"AAA": frame()}, {"AAA": _row(composite=math.nan)})
    assert orders == []
    assert skips == []


def test_earnings_within_blackout_blocks_entry():
    earnings = {"AAA": ("2026-07-06",)}  # 3 business days after 2026-07-01
    state = make_state(EQ)
    rankings = make_rankings(EQ, {"AAA": frame()}, make_table({"AAA": _row()}))
    orders, skips = evaluate_entries(state, rankings, EQ, DECISION, VALUE, earnings=earnings)
    assert orders == []
    assert Skip("AAA", "entry", "earnings_blackout") in skips


def test_earnings_beyond_blackout_and_none_do_not_block():
    state = make_state(EQ)
    rankings = make_rankings(EQ, {"AAA": frame()}, make_table({"AAA": _row()}))
    # 2026-07-01 + 5 business days = 2026-07-08; the 9th is outside the window.
    far = {"AAA": ("2026-07-09",)}
    orders, _ = evaluate_entries(state, rankings, EQ, DECISION, VALUE, earnings=far)
    assert len(orders) == 1
    # earnings=None (filter disabled/degraded symbol) never blocks.
    orders, _ = evaluate_entries(state, rankings, EQ, DECISION, VALUE, earnings=None)
    assert len(orders) == 1


def test_earnings_day_of_decision_blocks():
    earnings = {"AAA": ("2026-07-01",)}
    state = make_state(EQ)
    rankings = make_rankings(EQ, {"AAA": frame()}, make_table({"AAA": _row()}))
    orders, skips = evaluate_entries(state, rankings, EQ, DECISION, VALUE, earnings=earnings)
    assert orders == []
    assert Skip("AAA", "entry", "earnings_blackout") in skips


def test_fee_gate_boundary_equality_passes():
    # Crypto round-trip cost = 2 * (95 + 5) bps = 2.0%; gate = 3.0x = 6.0%.
    # A raw 30d return of EXACTLY 6.0% passes (strict <).
    round_trip = 2 * (CR.costs.taker_fee_bps + CR.costs.slippage_bps) / 1e4
    exactly_at_gate = CR.portfolio.min_raw_return_cost_multiple * round_trip
    bars = {"BTC": frame(end="2026-07-01")}
    table = make_table(
        {"BTC": {"status": "tradable", "composite": 0.9, "raw_return_30d": exactly_at_gate}}
    )
    rankings = make_rankings(CR, bars, table)
    state = make_state(CR)
    orders, skips = evaluate_entries(state, rankings, CR, AS_OF, 1000.0)
    assert [o.symbol for o in orders] == ["BTC"]
    assert ("BTC", "fee_gate") not in [(s.symbol, s.reason) for s in skips]


def test_deployment_cap_boundary_equality_passes():
    # With max_daily_deployment_pct == position_size_pct the first entry's
    # notional EQUALS the budget: it must pass (strict >); the second
    # candidate then faces a zero budget and is capped.
    config = replace(
        EQ, portfolio=replace(EQ.portfolio, max_daily_deployment_pct=EQ.portfolio.position_size_pct)
    )
    bars = {"AAA": frame(end="2026-07-01"), "BBB": frame(end="2026-07-01")}
    table = make_table(
        {
            "AAA": {"status": "tradable", "composite": 0.9, "raw_return_30d": 0.5},
            "BBB": {"status": "tradable", "composite": 0.8, "raw_return_30d": 0.5},
        }
    )
    rankings = make_rankings(config, bars, table)
    state = make_state(config)
    orders, skips = evaluate_entries(state, rankings, config, AS_OF, 1000.0)
    assert [o.symbol for o in orders] == ["AAA"]
    assert ("BBB", "daily_deployment_cap") in [(s.symbol, s.reason) for s in skips]
