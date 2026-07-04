"""The pure per-venue simulator step (spec: Portfolio Simulator).

step() is the single function the M3 backtester replays: no I/O, no clock.
Everything it needs arrives as bars + state + config; staleness (the only
clock-dependent rule) is decided by the runner and passed in as allow_entries.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import pandas as pd

from trading.config import VenueConfig
from trading.pipeline import RankingsResult
from trading.simulator.entries import evaluate_entries
from trading.simulator.exits import evaluate_exits
from trading.simulator.fills import Fill, apply_fills, release_settlements
from trading.simulator.state import PendingOrder, PortfolioState, Skip


@dataclass(frozen=True)
class PositionMark:
    symbol: str
    qty: float
    entry_price: float
    last_close: float
    market_value: float
    unrealized_pnl_pct: float
    stop_price: float
    stop_distance_pct: float
    entry_rank: int
    entry_composite: float
    entry_ts: str


@dataclass(frozen=True)
class PortfolioSnapshot:
    value: float
    cash: float
    unsettled: float
    positions: tuple[PositionMark, ...]


@dataclass(frozen=True)
class StepResult:
    state: PortfolioState  # updated deep copy; caller's state is untouched
    run_key: str
    decision_ts: pd.Timestamp
    fills: tuple[Fill, ...]
    new_orders: tuple[PendingOrder, ...]
    skips: tuple[Skip, ...]
    warnings: tuple[str, ...]
    breaker_tripped_now: bool
    snapshot: PortfolioSnapshot


def decision_bar(rankings: RankingsResult) -> pd.Timestamp:
    """The decision bar = newest completed bar across the clean universe."""
    return max(df.index[-1] for df in rankings.bars.values() if not df.empty)


def make_run_key(venue: str, decision_ts: pd.Timestamp) -> str:
    return f"{venue}:{decision_ts.isoformat()}"


def _mark_portfolio(
    state: PortfolioState, bars: dict[str, pd.DataFrame], decision_ts: pd.Timestamp
) -> tuple[PortfolioSnapshot, list[str]]:
    warnings: list[str] = []
    marks: list[PositionMark] = []
    for symbol, position in sorted(state.positions.items()):
        df = bars.get(symbol)
        window = df[df.index <= decision_ts] if df is not None else None
        if window is None or window.empty:
            last_close = position.entry_price
            warnings.append(f"{symbol}: no bars to mark position; using entry price")
        else:
            last_close = float(window["close"].iloc[-1])
        marks.append(
            PositionMark(
                symbol=symbol,
                qty=position.qty,
                entry_price=position.entry_price,
                last_close=last_close,
                market_value=position.qty * last_close,
                unrealized_pnl_pct=last_close / position.entry_price - 1.0,
                stop_price=position.stop_price,
                stop_distance_pct=(last_close - position.stop_price) / last_close,
                entry_rank=position.entry_rank,
                entry_composite=position.entry_composite,
                entry_ts=position.entry_ts,
            )
        )
    unsettled = sum(s.amount for s in state.settlements)
    value = state.cash + unsettled + sum(m.market_value for m in marks)
    return PortfolioSnapshot(
        value=value, cash=state.cash, unsettled=unsettled, positions=tuple(marks)
    ), warnings


def step(
    state: PortfolioState,
    rankings: RankingsResult,
    config: VenueConfig,
    *,
    allow_entries: bool = True,
    stale_reason: str | None = None,
) -> StepResult:
    state = copy.deepcopy(state)
    decision_ts = decision_bar(rankings)
    run_key = make_run_key(config.name, decision_ts)
    warnings: list[str] = []
    skips: list[Skip] = []

    # Phase 1: settle yesterday's sale proceeds, then fill pending orders.
    release_settlements(state, decision_ts.date())
    fills, fill_skips = apply_fills(state, rankings.bars, config)
    skips.extend(fill_skips)

    # Phase 2: exits (before entries, against the unfiltered ranking).
    exit_orders, exit_skips, exit_warnings = evaluate_exits(state, rankings, config, decision_ts)
    skips.extend(exit_skips)
    warnings.extend(exit_warnings)

    # Phase 3: mark to the decision bar; ratchet the high-water mark; breaker.
    snapshot, mark_warnings = _mark_portfolio(state, rankings.bars, decision_ts)
    warnings.extend(mark_warnings)
    breaker_tripped_now = False
    if snapshot.value > state.high_water_mark:
        state.high_water_mark = snapshot.value
    drawdown = 1.0 - snapshot.value / state.high_water_mark
    if not state.breaker_tripped and drawdown > config.portfolio.drawdown_halt_pct:
        state.breaker_tripped = True
        state.breaker_tripped_at = decision_ts.isoformat()
        breaker_tripped_now = True
        warnings.append(
            f"circuit breaker tripped: drawdown {drawdown:.1%} exceeds "
            f"{config.portfolio.drawdown_halt_pct:.0%}; entries halted until reset-breaker"
        )

    # Phase 4: entries (regime- and breaker-gated; staleness decided upstream).
    entry_orders: list[PendingOrder] = []
    if allow_entries:
        entry_orders, entry_skips = evaluate_entries(
            state, rankings, config, decision_ts, snapshot.value
        )
        skips.extend(entry_skips)
    else:
        skips.append(Skip("*", "entry", stale_reason or "entries_disabled"))

    new_orders = [*exit_orders, *entry_orders]
    state.pending_orders = [*state.pending_orders, *new_orders]
    state.last_run_key = run_key

    return StepResult(
        state=state,
        run_key=run_key,
        decision_ts=decision_ts,
        fills=tuple(fills),
        new_orders=tuple(new_orders),
        skips=tuple(skips),
        warnings=tuple(warnings),
        breaker_tripped_now=breaker_tripped_now,
        snapshot=snapshot,
    )
