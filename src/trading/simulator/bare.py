"""Bare wrapper-ablation mode (R2 spec, cell W0): top-N by rank, equal
weight, rebalanced monthly, sell only on a rank exit (or a forced exit --
delisted/untradable/quarantined/fetch-failed, which is data integrity, not
wrapper, and always applies). No regime gate, no ATR/trailing/time stops, no
entry-score threshold, no hyperparameter grid, no daily-deployment throttle.

Opt-in via config.portfolio.bare_mode (additive, default False); the normal
evaluate_entries/evaluate_exits path in trading.simulator.entries/exits is
completely untouched by this module -- trading.simulator.core.step only
calls into here when the flag is set (see its docstring for the W4
bit-identical guarantee this preserves).
"""

from __future__ import annotations

import pandas as pd

from trading.config import VenueConfig
from trading.pipeline import RankingsResult
from trading.simulator.exits import FORCED_STATUSES
from trading.simulator.state import PendingOrder, PortfolioState, Skip


def rank_only_top_n(
    rankings: RankingsResult, config: VenueConfig, decision_ts: pd.Timestamp
) -> list[str]:
    """The top max_positions symbols by rank: tradable, with a current bar,
    above the (universe-definition, not-wrapper) dollar-volume floor. No
    entry-score threshold -- rank position is the only filter (spec W0)."""
    top: list[str] = []
    for symbol, row in rankings.table.iterrows():
        if len(top) >= config.portfolio.max_positions:
            break
        composite = row.get("composite")
        if composite is None or pd.isna(composite):
            continue
        if str(row.get("status")) != "tradable":
            continue
        df = rankings.bars.get(symbol)
        window = df[df.index <= decision_ts] if df is not None else None
        if window is None or window.empty:
            continue
        if config.universe.min_dollar_volume > 0:
            recent = (
                window["close"].iloc[-config.signals.mean_window :]
                * window["volume"].iloc[-config.signals.mean_window :]
            )
            if float(recent.mean()) < config.universe.min_dollar_volume:
                continue
        top.append(symbol)
    return top


def _sell(symbol: str, decision_ts: pd.Timestamp, reason: str) -> PendingOrder:
    return PendingOrder(
        symbol=symbol, side="sell", notional=0.0, decision_ts=decision_ts.isoformat(), reason=reason
    )


def evaluate_exits_bare(
    state: PortfolioState,
    rankings: RankingsResult,
    decision_ts: pd.Timestamp,
    top_symbols: list[str],
    is_rebalance: bool,
) -> tuple[list[PendingOrder], list[Skip], list[str]]:
    orders: list[PendingOrder] = []
    skips: list[Skip] = []
    warnings: list[str] = []
    pending_sells = {o.symbol for o in state.pending_orders if o.side == "sell"}
    top_set = set(top_symbols)

    for symbol, _position in sorted(state.positions.items()):
        if symbol in pending_sells:
            continue
        if symbol in rankings.quarantined:
            warnings.append(f"{symbol}: held position quarantined; no trades until it clears")
            skips.append(Skip(symbol, "exit", "quarantined_no_trades"))
            continue
        if symbol in rankings.fetch_failures:
            warnings.append(f"{symbol}: held position fetch failed; exit rules not evaluated")
            skips.append(Skip(symbol, "exit", "fetch_failure_no_evaluation"))
            continue
        in_table = symbol in rankings.table.index
        df = rankings.bars.get(symbol)
        if not in_table and df is None:
            # Dropped from the venue universe entirely: delisted.
            orders.append(_sell(symbol, decision_ts, "forced_exit"))
            continue
        if in_table and str(rankings.table.loc[symbol, "status"]) in FORCED_STATUSES:
            orders.append(_sell(symbol, decision_ts, "forced_exit"))
            continue
        if not is_rebalance:
            continue  # bare mode only re-ranks on the monthly rebalance session
        if symbol not in top_set:
            orders.append(_sell(symbol, decision_ts, "rank_exit"))

    return orders, skips, warnings


def evaluate_entries_bare(
    state: PortfolioState,
    rankings: RankingsResult,
    config: VenueConfig,
    decision_ts: pd.Timestamp,
    portfolio_value: float,
    top_symbols: list[str],
) -> tuple[list[PendingOrder], list[Skip]]:
    orders: list[PendingOrder] = []
    skips: list[Skip] = []
    held_or_pending = set(state.positions) | {o.symbol for o in state.pending_orders}
    fee_rate = config.costs.taker_fee_bps / 1e4
    notional = portfolio_value / config.portfolio.max_positions
    cash = state.cash

    for rank_pos, symbol in enumerate(top_symbols, start=1):
        if symbol in held_or_pending:
            continue
        if notional * (1 + fee_rate) > cash:
            skips.append(Skip(symbol, "entry", "insufficient_settled_cash"))
            continue
        composite = float(rankings.table.loc[symbol, "composite"])
        orders.append(
            PendingOrder(
                symbol=symbol,
                side="buy",
                notional=notional,
                decision_ts=decision_ts.isoformat(),
                reason="entry",
                atr_at_decision=0.0,  # bare mode never evaluates a stop_price
                composite=composite,
                rank=rank_pos,
            )
        )
        cash -= notional * (1 + fee_rate)

    return orders, skips
