"""Fill engine (spec: Fill model / Execution Split, phase 1 of each run).

Pending orders written by the PREVIOUS run fill at the open of the first bar
strictly after their decision bar, plus slippage, plus taker fees. Pure: the
caller (core.step) deep-copies state before handing it here.
"""

from __future__ import annotations

import datetime
import math
from dataclasses import dataclass

import pandas as pd

from trading.config import VenueConfig
from trading.simulator.state import PendingOrder, PortfolioState, Position, Settlement, Skip


@dataclass(frozen=True)
class Fill:
    symbol: str
    side: str
    qty: float
    price: float
    fee: float
    bar_ts: str  # ISO-8601 UTC timestamp of the fill bar
    reason: str
    realized_pnl: float | None  # sells only


def atr(bars: pd.DataFrame, window: int) -> float:
    """Simple (Cutler-style) ATR: mean true range of the last `window` bars.

    Needs window + 1 rows (the previous close seeds the first true range).
    """
    if len(bars) < window + 1:
        return math.nan
    high, low, close = bars["high"], bars["low"], bars["close"]
    prev_close = close.shift(1)
    true_range = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return float(true_range.iloc[-window:].mean())


def release_settlements(state: PortfolioState, decision_date: datetime.date) -> None:
    """Move settled sale proceeds into spendable cash (T+1 for equities)."""
    remaining: list[Settlement] = []
    for settlement in state.settlements:
        if datetime.date.fromisoformat(settlement.available_on) <= decision_date:
            state.cash += settlement.amount
        else:
            remaining.append(settlement)
    state.settlements = remaining


def apply_fills(
    state: PortfolioState, bars: dict[str, pd.DataFrame], config: VenueConfig
) -> tuple[list[Fill], list[Skip]]:
    """Fill pending orders against current bars. Mutates state; returns fills + skips.

    Sells process before buys (deterministic order: side, then symbol). A buy
    with no bar after its decision bar is CANCELLED (stale decision -- no
    catch-up entries); a sell is kept pending (exits always process
    eventually); a sell without a matching position is dropped.
    """
    slip = config.costs.slippage_bps / 1e4
    fee_rate = config.costs.taker_fee_bps / 1e4
    fills: list[Fill] = []
    skips: list[Skip] = []
    remaining: list[PendingOrder] = []

    for order in sorted(state.pending_orders, key=lambda o: (o.side != "sell", o.symbol)):
        df = bars.get(order.symbol)
        after = df[df.index > pd.Timestamp(order.decision_ts)] if df is not None else None
        if after is None or after.empty:
            if order.side == "buy":
                skips.append(Skip(order.symbol, "fill", "entry_cancelled_no_fill_bar"))
            else:
                remaining.append(order)
                skips.append(Skip(order.symbol, "fill", "exit_deferred_no_fill_bar"))
            continue

        bar_ts: pd.Timestamp = after.index[0]
        open_price = float(after["open"].iloc[0])

        if order.side == "buy":
            price = open_price * (1 + slip)
            qty = order.notional / price
            fee = order.notional * fee_rate
            state.cash -= order.notional + fee
            state.positions[order.symbol] = Position(
                symbol=order.symbol,
                qty=qty,
                entry_price=price,
                entry_ts=bar_ts.isoformat(),
                entry_atr=order.atr_at_decision,
                stop_price=price - config.portfolio.stop_atr_multiple * order.atr_at_decision,
                flushed=False,
                entry_composite=order.composite,
                entry_rank=order.rank,
                entry_fee=fee,
                peak_close=price,
            )
            fills.append(
                Fill(order.symbol, "buy", qty, price, fee, bar_ts.isoformat(), order.reason, None)
            )
        else:
            position = state.positions.pop(order.symbol, None)
            if position is None:
                skips.append(Skip(order.symbol, "fill", "exit_orphaned_no_position"))
                continue
            price = open_price * (1 - slip)
            gross = position.qty * price
            fee = gross * fee_rate
            proceeds = gross - fee
            if config.costs.settlement_days == 0:
                state.cash += proceeds
            else:
                available = bar_ts.date() + datetime.timedelta(days=config.costs.settlement_days)
                state.settlements.append(
                    Settlement(amount=proceeds, available_on=available.isoformat())
                )
            if order.reason == "stop_loss":
                until = bar_ts.date() + datetime.timedelta(days=config.portfolio.cooldown_days)
                state.cooldowns[order.symbol] = until.isoformat()
            fills.append(
                Fill(
                    order.symbol,
                    "sell",
                    position.qty,
                    price,
                    fee,
                    bar_ts.isoformat(),
                    order.reason,
                    proceeds - position.qty * position.entry_price - position.entry_fee,
                )
            )

    state.pending_orders = remaining
    return fills, skips
