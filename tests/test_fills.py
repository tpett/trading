import datetime
import math
from dataclasses import replace

import pandas as pd
import pytest

from sim_helpers import CR, EQ, frame, make_state
from trading.simulator.fills import apply_fills, atr, release_settlements
from trading.simulator.state import PendingOrder, Position, Settlement

DECISION = "2026-06-30T00:00:00+00:00"  # bars in frame(end="2026-07-01") have one bar after this


def _atr_bars() -> pd.DataFrame:
    idx = pd.date_range("2026-06-01", periods=4, freq="D", tz="UTC")
    return pd.DataFrame(
        {
            "open": [10.0, 11.0, 12.0, 13.0],
            "high": [10.5, 11.5, 12.5, 13.5],
            "low": [9.5, 10.5, 11.5, 12.5],
            "close": [10.0, 11.0, 12.0, 13.0],
            "volume": [1e6] * 4,
        },
        index=idx,
    )


def test_atr_hand_computed():
    # TR per bar (with prev close): max(high-low, |high-pc|, |low-pc|) = 1.5 for bars 2..4.
    assert atr(_atr_bars(), window=3) == pytest.approx(1.5)


def test_atr_insufficient_history_is_nan():
    assert math.isnan(atr(_atr_bars(), window=4))


def _buy_order(symbol="AAA", notional=180.0, entry_atr=4.0) -> PendingOrder:
    return PendingOrder(
        symbol=symbol,
        side="buy",
        notional=notional,
        decision_ts=DECISION,
        reason="entry",
        atr_at_decision=entry_atr,
        composite=0.9,
        rank=1,
    )


def _position(symbol="AAA", qty=2.0, entry_price=90.0, stop=80.0) -> Position:
    return Position(
        symbol=symbol,
        qty=qty,
        entry_price=entry_price,
        entry_ts="2026-06-20T00:00:00+00:00",
        entry_atr=4.0,
        stop_price=stop,
        flushed=False,
        entry_composite=0.8,
        entry_rank=1,
    )


def test_buy_fills_at_next_bar_open_with_slippage_and_creates_frozen_stop():
    bars = {"AAA": frame(end="2026-07-01")}  # constant close/open 100.0
    state = make_state(EQ, pending_orders=[_buy_order()])
    fills, skips = apply_fills(state, bars, EQ)

    assert skips == []
    assert len(fills) == 1
    fill = fills[0]
    price = 100.0 * (1 + 5.0 / 1e4)  # open + 5 bps slippage
    assert fill.price == pytest.approx(price)
    assert fill.qty == pytest.approx(180.0 / price)
    assert fill.fee == pytest.approx(0.0)  # equities: zero commission
    assert fill.realized_pnl is None
    # The fill bar is the first bar strictly after the decision bar (2026-07-01).
    assert fill.bar_ts == "2026-07-01T00:00:00+00:00"
    position = state.positions["AAA"]
    assert position.entry_atr == 4.0
    assert position.stop_price == pytest.approx(price - EQ.portfolio.stop_atr_multiple * 4.0)
    assert position.flushed is False
    assert state.cash == pytest.approx(1000.0 - 180.0)
    assert state.pending_orders == []


def test_crypto_buy_pays_taker_fee():
    bars = {"BTC": frame(end="2026-07-01")}
    state = make_state(CR, pending_orders=[_buy_order(symbol="BTC", notional=300.0)])
    fills, _ = apply_fills(state, bars, CR)
    assert fills[0].fee == pytest.approx(300.0 * 95.0 / 1e4)
    assert state.cash == pytest.approx(1000.0 - 300.0 - 300.0 * 95.0 / 1e4)


def test_equities_sell_settles_t_plus_1_and_realizes_pnl():
    bars = {"AAA": frame(end="2026-07-01")}
    state = make_state(EQ, positions={"AAA": _position()}, cash=0.0)
    state.pending_orders = [
        PendingOrder(
            symbol="AAA", side="sell", notional=0.0, decision_ts=DECISION, reason="trend_break"
        )
    ]
    fills, _ = apply_fills(state, bars, EQ)

    price = 100.0 * (1 - 5.0 / 1e4)
    proceeds = 2.0 * price  # zero commission
    assert fills[0].realized_pnl == pytest.approx(proceeds - 2.0 * 90.0)
    assert state.positions == {}
    assert state.cash == 0.0  # unspendable until settled (T+1)
    assert state.settlements == [Settlement(amount=proceeds, available_on="2026-07-02")]
    # trend_break is not a stop-out: no cooldown.
    assert state.cooldowns == {}


def test_crypto_sell_is_immediately_settled():
    bars = {"BTC": frame(end="2026-07-01")}
    state = make_state(CR, positions={"BTC": _position(symbol="BTC")}, cash=0.0)
    state.pending_orders = [
        PendingOrder(
            symbol="BTC", side="sell", notional=0.0, decision_ts=DECISION, reason="time_stop"
        )
    ]
    apply_fills(state, bars, CR)
    price = 100.0 * (1 - 5.0 / 1e4)
    gross = 2.0 * price
    assert state.settlements == []
    assert state.cash == pytest.approx(gross - gross * 95.0 / 1e4)


def test_stop_loss_fill_sets_reentry_cooldown():
    bars = {"AAA": frame(end="2026-07-01")}
    state = make_state(EQ, positions={"AAA": _position()})
    state.pending_orders = [
        PendingOrder(
            symbol="AAA", side="sell", notional=0.0, decision_ts=DECISION, reason="stop_loss"
        )
    ]
    apply_fills(state, bars, EQ)
    # Fill bar 2026-07-01 + 7-day cooldown -> re-entry allowed 2026-07-08.
    assert state.cooldowns == {"AAA": "2026-07-08"}


def test_buy_without_fill_bar_is_cancelled_and_sell_is_deferred():
    bars = {"AAA": frame(end="2026-06-30"), "BBB": frame(end="2026-06-30")}
    state = make_state(EQ, positions={"BBB": _position(symbol="BBB")})
    state.pending_orders = [
        _buy_order(symbol="AAA"),
        PendingOrder(
            symbol="BBB", side="sell", notional=0.0, decision_ts=DECISION, reason="stop_loss"
        ),
    ]
    fills, skips = apply_fills(state, bars, EQ)
    assert fills == []
    reasons = {(s.symbol, s.reason) for s in skips}
    assert ("AAA", "entry_cancelled_no_fill_bar") in reasons
    assert ("BBB", "exit_deferred_no_fill_bar") in reasons
    assert [o.symbol for o in state.pending_orders] == ["BBB"]  # sell kept, buy dropped
    assert "BBB" in state.positions


def test_orphaned_sell_is_dropped():
    from trading.simulator.state import Skip

    bars = {"AAA": frame(end="2026-07-01")}
    state = make_state(EQ)  # no positions
    state.pending_orders = [
        PendingOrder(
            symbol="AAA", side="sell", notional=0.0, decision_ts=DECISION, reason="stop_loss"
        )
    ]
    fills, skips = apply_fills(state, bars, EQ)
    assert fills == []
    assert skips == [Skip("AAA", "fill", "exit_orphaned_no_position")]
    assert state.pending_orders == []


def test_release_settlements_moves_due_cash():
    import datetime

    state = make_state(EQ, cash=10.0)
    state.settlements = [
        Settlement(amount=100.0, available_on="2026-07-01"),
        Settlement(amount=50.0, available_on="2026-07-05"),
    ]
    release_settlements(state, datetime.date(2026, 7, 1))
    assert state.cash == pytest.approx(110.0)
    assert state.settlements == [Settlement(amount=50.0, available_on="2026-07-05")]


def test_fill_price_within_next_bar_range_plus_slippage():
    """Spec property: fills are anchored to the next bar's open, so they sit
    within [low*(1-slip), high*(1+slip)] of the fill bar."""
    bars = {"AAA": frame(end="2026-07-01")}
    state = make_state(EQ, pending_orders=[_buy_order()])
    fills, _ = apply_fills(state, bars, EQ)
    fill_bar = bars["AAA"].loc[pd.Timestamp("2026-07-01", tz="UTC")]
    slip = 5.0 / 1e4
    assert fill_bar["low"] * (1 - slip) <= fills[0].price <= fill_bar["high"] * (1 + slip)


def test_buy_fill_stores_entry_fee_on_position():
    bars = {"BTC": frame(end="2026-07-01")}
    state = make_state(CR, pending_orders=[_buy_order(symbol="BTC", notional=300.0)])
    apply_fills(state, bars, CR)
    assert state.positions["BTC"].entry_fee == pytest.approx(300.0 * 95.0 / 1e4)


def test_buy_fill_initializes_peak_close_to_fill_price():
    bars = {"BTC": frame(end="2026-07-01")}
    state = make_state(CR, pending_orders=[_buy_order(symbol="BTC", notional=300.0)])
    fills, _ = apply_fills(state, bars, CR)
    assert state.positions["BTC"].peak_close == pytest.approx(fills[0].price)


def test_sell_realized_pnl_includes_entry_fee():
    bars = {"BTC": frame(end="2026-07-01")}
    position = _position(symbol="BTC")
    position = replace(position, entry_fee=2.85)
    state = make_state(CR, positions={"BTC": position}, cash=0.0)
    state.pending_orders = [
        PendingOrder(
            symbol="BTC", side="sell", notional=0.0, decision_ts=DECISION, reason="time_stop"
        )
    ]
    fills, _ = apply_fills(state, bars, CR)
    price = 100.0 * (1 - 5.0 / 1e4)
    gross = 2.0 * price
    proceeds = gross - gross * 95.0 / 1e4
    assert fills[0].realized_pnl == pytest.approx(proceeds - 2.0 * 90.0 - 2.85)


def test_settlement_crosses_weekend_t_plus_1():
    # Sell decided Thu 2026-06-04 fills Fri 2026-06-05 -> available_on Sat
    # 2026-06-06. Cash must stay unspendable through Friday's session and be
    # released by Monday's (2026-06-08) decision date.
    bars = {"AAA": frame(end="2026-06-05", periods=5)}  # Mon..Fri, freq="B"
    state = make_state(EQ, positions={"AAA": _position()}, cash=0.0)
    state.pending_orders = [
        PendingOrder(
            symbol="AAA",
            side="sell",
            notional=0.0,
            decision_ts="2026-06-04T00:00:00+00:00",
            reason="trend_break",
        )
    ]
    fills, _ = apply_fills(state, bars, EQ)
    proceeds = fills[0].qty * fills[0].price  # zero commission on equities
    assert fills[0].bar_ts == "2026-06-05T00:00:00+00:00"
    assert state.settlements[0].available_on == "2026-06-06"  # Saturday

    release_settlements(state, datetime.date(2026, 6, 5))  # still Friday
    assert state.cash == 0.0
    release_settlements(state, datetime.date(2026, 6, 8))  # Monday's decision date
    assert state.cash == pytest.approx(proceeds)
    assert state.settlements == []
