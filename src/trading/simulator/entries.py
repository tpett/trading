"""Entry rules (spec: Portfolio Simulator — entries after exits, regime-gated).

Walks the ranked table top-down and emits buy orders for the next bar, with a
journaled skip reason for every blocked candidate above the score threshold.
Hard rails: never average down, never exceed the regime-scaled position count,
no margin (settled cash only), max 25% of portfolio deployed per day.
"""

from __future__ import annotations

import datetime
import math

import pandas as pd

from trading.config import VenueConfig
from trading.pipeline import RankingsResult
from trading.simulator.fills import atr
from trading.simulator.state import PendingOrder, PortfolioState, Skip


def evaluate_entries(
    state: PortfolioState,
    rankings: RankingsResult,
    config: VenueConfig,
    decision_ts: pd.Timestamp,
    portfolio_value: float,
) -> tuple[list[PendingOrder], list[Skip]]:
    p = config.portfolio
    if state.breaker_tripped:
        return [], [Skip("*", "entry", "circuit_breaker")]
    if rankings.regime.state == "risk_off":
        return [], [Skip("*", "entry", "regime_risk_off")]

    orders: list[PendingOrder] = []
    skips: list[Skip] = []
    pending_sells = {o.symbol for o in state.pending_orders if o.side == "sell"}
    pending_buys = sum(1 for o in state.pending_orders if o.side == "buy")
    slots = (
        math.floor(p.max_positions * rankings.regime.exposure_multiplier)
        - len(state.positions)
        - pending_buys
    )
    budget = p.max_daily_deployment_pct * portfolio_value
    cash = state.cash
    fee_rate = config.costs.taker_fee_bps / 1e4
    round_trip_cost = 2 * (config.costs.taker_fee_bps + config.costs.slippage_bps) / 1e4
    decision_date = decision_ts.date()

    for rank_pos, (symbol, row) in enumerate(rankings.table.iterrows(), start=1):
        composite = float(row["composite"])
        if math.isnan(composite) or composite < p.entry_score_threshold:
            break  # sorted desc: no candidate below this can qualify
        if symbol in state.positions or symbol in pending_sells:
            skips.append(Skip(symbol, "entry", "already_held"))
            continue
        if str(row["status"]) != "tradable":
            skips.append(Skip(symbol, "entry", f"status_{row['status']}"))
            continue
        cooldown = state.cooldowns.get(symbol)
        if cooldown is not None and decision_date < datetime.date.fromisoformat(cooldown):
            skips.append(Skip(symbol, "entry", "cooldown"))
            continue
        df = rankings.bars.get(symbol)
        window = df[df.index <= decision_ts] if df is not None else None
        if window is None or window.empty:
            skips.append(Skip(symbol, "entry", "no_bars"))
            continue
        if config.universe.min_dollar_volume > 0:
            recent = (
                window["close"].iloc[-config.signals.mean_window :]
                * window["volume"].iloc[-config.signals.mean_window :]
            )
            if float(recent.mean()) < config.universe.min_dollar_volume:
                skips.append(Skip(symbol, "entry", "below_dollar_volume_floor"))
                continue
        if p.min_raw_return_cost_multiple > 0:
            raw = row["raw_return_30d"]
            if pd.isna(raw) or float(raw) < p.min_raw_return_cost_multiple * round_trip_cost:
                skips.append(Skip(symbol, "entry", "fee_gate"))
                continue
        if slots <= 0:
            skips.append(Skip(symbol, "entry", "no_free_slot"))
            break  # every later candidate hits the same wall
        entry_atr = atr(window, p.atr_window)
        if math.isnan(entry_atr):
            skips.append(Skip(symbol, "entry", "insufficient_history_for_atr"))
            continue
        notional = p.position_size_pct * portfolio_value
        if notional > budget:
            skips.append(Skip(symbol, "entry", "daily_deployment_cap"))
            break
        if notional * (1 + fee_rate) > cash:
            skips.append(Skip(symbol, "entry", "insufficient_settled_cash"))
            break
        orders.append(
            PendingOrder(
                symbol=symbol,
                side="buy",
                notional=notional,
                decision_ts=decision_ts.isoformat(),
                reason="entry",
                atr_at_decision=entry_atr,
                composite=composite,
                rank=rank_pos,
            )
        )
        slots -= 1
        budget -= notional
        cash -= notional * (1 + fee_rate)

    return orders, skips
