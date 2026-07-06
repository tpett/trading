"""Exit rules (spec: Portfolio Simulator — exits checked before entries).

Evaluated against the UNFILTERED ranking: held names always remain rankable
(names whose composite is NaN due to degenerate signal inputs (e.g. zero
volatility or volume) are held but unranked — distinct from
insufficient_history, which excludes symbols from the table entirely), so
entry-filter mechanics can never manufacture a spurious exit. Quarantined or
fetch-failed held symbols are never traded this run (bad data), only warned
about. Exits emit pending sell orders that fill next run.
"""

from __future__ import annotations

from dataclasses import replace

import pandas as pd

from trading.config import VenueConfig
from trading.pipeline import RankingsResult
from trading.simulator.state import PendingOrder, PortfolioState, Skip

FORCED_STATUSES = {"sell_only", "untradable"}


def _sell(symbol: str, decision_ts: pd.Timestamp, reason: str) -> PendingOrder:
    return PendingOrder(
        symbol=symbol, side="sell", notional=0.0, decision_ts=decision_ts.isoformat(), reason=reason
    )


def evaluate_exits(
    state: PortfolioState,
    rankings: RankingsResult,
    config: VenueConfig,
    decision_ts: pd.Timestamp,
) -> tuple[list[PendingOrder], list[Skip], list[str]]:
    orders: list[PendingOrder] = []
    skips: list[Skip] = []
    warnings: list[str] = []
    pending_sells = {o.symbol for o in state.pending_orders if o.side == "sell"}
    # Trend-break "top half" is judged among genuinely ranked names only:
    # NaN-composite rows (degenerate signal inputs) must not pad the denominator.
    ranked = list(rankings.table.index[rankings.table["composite"].notna()])

    for symbol, position in sorted(state.positions.items()):
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

        window = df[df.index <= decision_ts] if df is not None else None
        if window is None or window.empty:
            warnings.append(f"{symbol}: held but no bars this run; exit rules not evaluated")
            skips.append(Skip(symbol, "exit", "no_bars_no_evaluation"))
            continue
        last_close = float(window["close"].iloc[-1])

        if config.portfolio.exit_style == "trailing":
            # Trailing stop: the sole profit-protection mechanism in this
            # mode (trend-break is skipped entirely below). The peak ratchets
            # with new highs; the width tightens one-way on a regime flush,
            # mirroring the frozen-mode flushed flag's semantics exactly —
            # once flushed, stays flushed, width never re-widens.
            if rankings.regime.state == "risk_off" and not position.flushed:
                position = replace(position, flushed=True)
                state.positions[symbol] = position
            width = (
                config.portfolio.regime_flush_atr_multiple
                if position.flushed
                else config.portfolio.stop_atr_multiple
            )
            prior_peak = (
                position.peak_close if position.peak_close is not None else position.entry_price
            )
            peak = max(prior_peak, last_close)
            candidate = peak - width * position.entry_atr
            position = replace(
                position, peak_close=peak, stop_price=max(position.stop_price, candidate)
            )
            state.positions[symbol] = position

            if last_close <= position.stop_price:
                orders.append(_sell(symbol, decision_ts, "stop_loss"))
                continue
        else:
            # Regime flush: recompute the stop ONCE at 1.0x the frozen entry ATR.
            # One-way ratchet — never loosened until the position closes.
            if rankings.regime.state == "risk_off" and not position.flushed:
                ratchet = (
                    position.entry_price
                    - config.portfolio.regime_flush_atr_multiple * position.entry_atr
                )
                position = replace(
                    position, stop_price=max(position.stop_price, ratchet), flushed=True
                )
                state.positions[symbol] = position

            if last_close <= position.stop_price:
                orders.append(_sell(symbol, decision_ts, "stop_loss"))
                continue

            if symbol in ranked:
                rank_pos = ranked.index(symbol) + 1
                mean = float(window["close"].iloc[-config.signals.mean_window :].mean())
                if rank_pos > len(ranked) / 2 and last_close < mean:
                    orders.append(_sell(symbol, decision_ts, "trend_break"))
                    continue
            else:
                warnings.append(f"{symbol}: held but unranked this run; trend-break not evaluated")

        bars_held = int((window.index > pd.Timestamp(position.entry_ts)).sum())
        if bars_held >= config.portfolio.time_stop_bars and last_close <= position.entry_price:
            orders.append(_sell(symbol, decision_ts, "time_stop"))

    return orders, skips, warnings
