"""Backtest metrics + go/no-go gate (spec: Backtesting & Validation).

Pure math over equity/benchmark curves and the trade list. The gate metric is
defined by the spec: annualized Sharpe of daily returns (cash yielding 0%) vs
the benchmark over the identical period; "beats" = higher Sharpe AND positive
total return.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

from trading.backtest.engine import BacktestResult, TradeRecord


@dataclass(frozen=True)
class BacktestMetrics:
    total_return: float
    annualized_return: float
    max_drawdown: float
    sharpe: float
    win_rate: float
    avg_win: float
    avg_loss: float
    trade_count: int
    turnover: float  # annualized: total buy notional / mean equity / years
    fees_paid: float
    fee_drag: float  # fees / starting equity (spec: fee drag as its own line)
    gross_profit: float  # P&L before fees: end value - start value + fees_paid
    # fees / gross_profit -- serves the spec's crypto go-live criterion
    # "fee drag < 30% of gross returns". NaN when gross_profit <= 0: fee drag
    # as a share of gross is meaningless with no gross gains, and an
    # unevaluable criterion is itself a failing state for go-live.
    fee_drag_vs_gross: float
    benchmark_total_return: float
    benchmark_sharpe: float
    gate_passed: bool  # sharpe > benchmark_sharpe AND total_return > 0 (spec)


def sharpe_ratio(curve: pd.Series, periods_per_year: int) -> float:
    """Annualized Sharpe of daily returns, cash yielding 0% (spec)."""
    returns = curve.pct_change().dropna()
    if len(returns) < 2:
        return math.nan
    std = float(returns.std())
    if std == 0.0:
        return math.nan
    return float(returns.mean()) / std * math.sqrt(periods_per_year)


def max_drawdown(curve: pd.Series) -> float:
    if curve.empty:
        return math.nan
    return float((1.0 - curve / curve.cummax()).max())


def metrics_from_curves(
    equity: pd.Series,
    benchmark: pd.Series,
    trades: tuple[TradeRecord, ...],
    buy_notional: float,
    fees_paid: float,
    periods_per_year: int,
) -> BacktestMetrics:
    if not equity.index.equals(benchmark.index):
        # The engine guarantees identical session indexes; this is a public
        # pure function, so misuse fails loud instead of comparing curves
        # over different periods.
        raise ValueError("equity and benchmark curves must share an identical index")
    total = float(equity.iloc[-1] / equity.iloc[0]) - 1.0
    years = len(equity) / periods_per_year
    annualized = (1.0 + total) ** (1.0 / years) - 1.0 if years > 0 and total > -1.0 else math.nan
    wins = [t.realized_pnl for t in trades if t.realized_pnl > 0]
    losses = [t.realized_pnl for t in trades if t.realized_pnl <= 0]
    sharpe = sharpe_ratio(equity, periods_per_year)
    benchmark_sharpe = sharpe_ratio(benchmark, periods_per_year)
    gate = (
        not math.isnan(sharpe)
        and not math.isnan(benchmark_sharpe)
        and sharpe > benchmark_sharpe
        and total > 0.0
    )
    gross_profit = float(equity.iloc[-1] - equity.iloc[0]) + fees_paid
    return BacktestMetrics(
        total_return=total,
        annualized_return=annualized,
        max_drawdown=max_drawdown(equity),
        sharpe=sharpe,
        win_rate=len(wins) / len(trades) if trades else math.nan,
        avg_win=sum(wins) / len(wins) if wins else 0.0,
        avg_loss=sum(losses) / len(losses) if losses else 0.0,
        trade_count=len(trades),
        turnover=buy_notional / float(equity.mean()) / years if years > 0 else math.nan,
        fees_paid=fees_paid,
        fee_drag=fees_paid / float(equity.iloc[0]),
        gross_profit=gross_profit,
        fee_drag_vs_gross=fees_paid / gross_profit if gross_profit > 0.0 else math.nan,
        benchmark_total_return=float(benchmark.iloc[-1] / benchmark.iloc[0]) - 1.0,
        benchmark_sharpe=benchmark_sharpe,
        gate_passed=gate,
    )


def compute_metrics(result: BacktestResult, periods_per_year: int) -> BacktestMetrics:
    return metrics_from_curves(
        result.equity_curve,
        result.benchmark_curve,
        result.trades,
        result.buy_notional,
        result.fees_paid,
        periods_per_year,
    )
