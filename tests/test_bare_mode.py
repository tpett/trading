"""Unit tests for the R2 ablation bare-mode simulator (W0: rank-only,
equal-weight, monthly rebalance -- see trading.simulator.bare).
"""

from dataclasses import replace

import pandas as pd
import pytest

from sim_helpers import EQ, frame, make_rankings, make_state, make_table
from trading.simulator.bare import evaluate_entries_bare, evaluate_exits_bare, rank_only_top_n
from trading.simulator.state import Position

DECISION = pd.Timestamp("2026-07-01", tz="UTC")
VALUE = 1000.0

BARE = replace(EQ, portfolio=replace(EQ.portfolio, bare_mode=True, max_positions=2))


def _row(status="tradable", composite=0.9):
    return {"status": status, "composite": composite, "raw_return_30d": 0.10}


def _position(symbol, entry_price=100.0):
    return Position(
        symbol=symbol,
        qty=1.0,
        entry_price=entry_price,
        entry_ts="2026-06-01T00:00:00+00:00",
        entry_atr=0.0,
        stop_price=entry_price,
        flushed=False,
        entry_composite=0.8,
        entry_rank=1,
    )


class TestRankOnlyTopN:
    def test_returns_top_n_tradable_symbols_in_rank_order(self):
        bars = {s: frame() for s in ("AAA", "BBB", "CCC")}
        rankings = make_rankings(
            BARE,
            bars,
            make_table(
                {
                    "AAA": _row(composite=0.95),
                    "BBB": _row(composite=0.90),
                    "CCC": _row(composite=0.10),
                }
            ),
        )
        assert rank_only_top_n(rankings, BARE, DECISION) == ["AAA", "BBB"]

    def test_ignores_entry_score_threshold_entirely(self):
        # threshold defaults to 0.70 in EQ/BARE; a composite of 0.01 is still
        # rank-eligible -- W0 has no threshold axis at all.
        bars = {"AAA": frame()}
        low_threshold_config = replace(BARE, portfolio=replace(BARE.portfolio, max_positions=5))
        rankings = make_rankings(
            low_threshold_config, bars, make_table({"AAA": _row(composite=0.01)})
        )
        assert rank_only_top_n(rankings, low_threshold_config, DECISION) == ["AAA"]

    def test_excludes_non_tradable_status(self):
        bars = {"AAA": frame(), "BBB": frame()}
        rankings = make_rankings(
            BARE,
            bars,
            make_table(
                {"AAA": _row(status="sell_only", composite=0.95), "BBB": _row(composite=0.5)}
            ),
        )
        assert rank_only_top_n(rankings, BARE, DECISION) == ["BBB"]

    def test_excludes_nan_composite(self):
        bars = {"AAA": frame(), "BBB": frame()}
        table = make_table({"AAA": _row(composite=0.5), "BBB": _row(composite=0.9)})
        table.loc["BBB", "composite"] = float("nan")
        rankings = make_rankings(BARE, bars, table)
        assert rank_only_top_n(rankings, BARE, DECISION) == ["AAA"]

    def test_below_dollar_volume_floor_is_excluded(self):
        bars = {"AAA": frame(volume=1.0)}  # ~$100 * 1 share/day: far below the floor
        rankings = make_rankings(BARE, bars, make_table({"AAA": _row(composite=0.9)}))
        assert rank_only_top_n(rankings, BARE, DECISION) == []


class TestEvaluateExitsBare:
    def test_forced_exit_fires_even_off_rebalance_day(self):
        state = make_state(BARE, positions={"AAA": _position("AAA")})
        bars = {"AAA": frame()}
        rankings = make_rankings(BARE, bars, make_table({"AAA": _row(status="untradable")}))
        orders, skips, warnings = evaluate_exits_bare(
            state, rankings, DECISION, top_symbols=["AAA"], is_rebalance=False
        )
        assert [(o.symbol, o.side, o.reason) for o in orders] == [("AAA", "sell", "forced_exit")]

    def test_no_trades_off_rebalance_day_even_if_out_of_rank(self):
        state = make_state(BARE, positions={"AAA": _position("AAA")})
        bars = {"AAA": frame()}
        rankings = make_rankings(BARE, bars, make_table({"AAA": _row()}))
        orders, skips, warnings = evaluate_exits_bare(
            state, rankings, DECISION, top_symbols=[], is_rebalance=False
        )
        assert orders == []

    def test_rank_exit_sells_a_held_name_that_fell_out_of_top_n_on_rebalance_day(self):
        state = make_state(BARE, positions={"AAA": _position("AAA")})
        bars = {"AAA": frame(), "BBB": frame()}
        rankings = make_rankings(
            BARE, bars, make_table({"AAA": _row(composite=0.1), "BBB": _row(composite=0.9)})
        )
        orders, skips, warnings = evaluate_exits_bare(
            state, rankings, DECISION, top_symbols=["BBB"], is_rebalance=True
        )
        assert [(o.symbol, o.side, o.reason) for o in orders] == [("AAA", "sell", "rank_exit")]

    def test_held_name_still_in_top_n_is_not_sold_on_rebalance_day(self):
        state = make_state(BARE, positions={"AAA": _position("AAA")})
        bars = {"AAA": frame()}
        rankings = make_rankings(BARE, bars, make_table({"AAA": _row()}))
        orders, skips, warnings = evaluate_exits_bare(
            state, rankings, DECISION, top_symbols=["AAA"], is_rebalance=True
        )
        assert orders == []

    def test_quarantined_held_symbol_holds_and_warns_regardless_of_rebalance_day(self):
        state = make_state(BARE, positions={"AAA": _position("AAA")})
        bars = {"AAA": frame()}
        rankings = make_rankings(BARE, bars, make_table({"AAA": _row()}), quarantined=("AAA",))
        orders, skips, warnings = evaluate_exits_bare(
            state, rankings, DECISION, top_symbols=[], is_rebalance=True
        )
        assert orders == []
        assert any("quarantined" in w for w in warnings)


class TestEvaluateEntriesBare:
    def test_buys_top_n_at_equal_weight(self):
        state = make_state(BARE)
        bars = {"AAA": frame(), "BBB": frame()}
        rankings = make_rankings(
            BARE, bars, make_table({"AAA": _row(composite=0.9), "BBB": _row(composite=0.5)})
        )
        orders, skips = evaluate_entries_bare(
            state, rankings, BARE, DECISION, VALUE, top_symbols=["AAA", "BBB"]
        )
        assert {o.symbol for o in orders} == {"AAA", "BBB"}
        for order in orders:
            assert order.notional == pytest.approx(VALUE / BARE.portfolio.max_positions)
            assert order.side == "buy"
            assert order.reason == "entry"

    def test_already_held_symbol_is_not_rebought(self):
        state = make_state(BARE, positions={"AAA": _position("AAA")})
        bars = {"AAA": frame(), "BBB": frame()}
        rankings = make_rankings(
            BARE, bars, make_table({"AAA": _row(), "BBB": _row(composite=0.5)})
        )
        orders, skips = evaluate_entries_bare(
            state, rankings, BARE, DECISION, VALUE, top_symbols=["AAA", "BBB"]
        )
        assert [o.symbol for o in orders] == ["BBB"]

    def test_insufficient_cash_skips_rather_than_crashes(self):
        state = make_state(BARE)
        state.cash = 1.0
        bars = {"AAA": frame()}
        rankings = make_rankings(BARE, bars, make_table({"AAA": _row()}))
        orders, skips = evaluate_entries_bare(
            state, rankings, BARE, DECISION, VALUE, top_symbols=["AAA"]
        )
        assert orders == []
        assert skips == [("AAA", "entry", "insufficient_settled_cash")] or skips[0].reason == (
            "insufficient_settled_cash"
        )

    def test_no_daily_deployment_cap_all_slots_can_fill_the_same_day(self):
        # Bare mode trades once a month, so the 25%-of-portfolio daily cap
        # (a hidden wrapper component) would make same-day top-N impossible
        # if honored; it must be bypassed entirely.
        wide = replace(BARE, portfolio=replace(BARE.portfolio, max_positions=3))
        state = make_state(wide)
        bars = {s: frame() for s in ("AAA", "BBB", "CCC")}
        rankings = make_rankings(
            wide,
            bars,
            make_table(
                {
                    "AAA": _row(composite=0.9),
                    "BBB": _row(composite=0.8),
                    "CCC": _row(composite=0.7),
                }
            ),
        )
        orders, skips = evaluate_entries_bare(
            state, rankings, wide, DECISION, VALUE, top_symbols=["AAA", "BBB", "CCC"]
        )
        assert len(orders) == 3
        assert sum(o.notional for o in orders) > wide.portfolio.max_daily_deployment_pct * VALUE
