import datetime
import math

import pandas as pd
import pytest

from trading.backtest.engine import BacktestResult, TradeRecord
from trading.backtest.metrics import (
    compute_metrics,
    max_drawdown,
    metrics_from_curves,
    sharpe_ratio,
)


def _curve(values: list[float], start: str = "2025-01-01") -> pd.Series:
    idx = pd.date_range(start, periods=len(values), freq="D", tz="UTC")
    return pd.Series(values, index=idx, dtype="float64")


def _trade(pnl: float) -> TradeRecord:
    return TradeRecord(
        symbol="AAA",
        qty=1.0,
        entry_ts="2025-01-02T00:00:00+00:00",
        exit_ts="2025-01-05T00:00:00+00:00",
        entry_price=100.0,
        exit_price=100.0 + pnl,
        entry_fee=0.5,
        exit_fee=0.5,
        realized_pnl=pnl,
        reason="stop_loss",
    )


def test_sharpe_hand_computed():
    # Daily returns: +1%, -1%, +1% -> mean 1/300, std known; 0% cash (spec).
    curve = _curve([100.0, 101.0, 99.99, 100.9899])
    returns = curve.pct_change().dropna()
    expected = float(returns.mean() / returns.std()) * math.sqrt(252)
    assert sharpe_ratio(curve, 252) == pytest.approx(expected)


def test_sharpe_degenerate_inputs_are_nan():
    assert math.isnan(sharpe_ratio(_curve([100.0]), 252))  # one point
    assert math.isnan(sharpe_ratio(_curve([100.0, 100.0, 100.0]), 252))  # zero vol


def test_max_drawdown_hand_computed():
    assert max_drawdown(_curve([100.0, 120.0, 90.0, 110.0])) == pytest.approx(0.25)
    assert max_drawdown(_curve([100.0, 110.0, 121.0])) == pytest.approx(0.0)


def test_metrics_from_curves_hand_computed():
    equity = _curve([1000.0, 1010.0, 1005.0, 1030.0])
    benchmark = _curve([1000.0, 1001.0, 1002.0, 1003.0])
    trades = (_trade(+30.0), _trade(+10.0), _trade(-15.0))
    m = metrics_from_curves(
        equity, benchmark, trades, buy_notional=2000.0, fees_paid=12.5, periods_per_year=252
    )
    assert m.total_return == pytest.approx(0.03)
    years = 4 / 252
    assert m.annualized_return == pytest.approx(1.03 ** (1 / years) - 1)
    assert m.max_drawdown == pytest.approx((1010.0 - 1005.0) / 1010.0)
    assert m.win_rate == pytest.approx(2 / 3)
    assert m.avg_win == pytest.approx(20.0)
    assert m.avg_loss == pytest.approx(-15.0)
    assert m.trade_count == 3
    assert m.turnover == pytest.approx(2000.0 / float(equity.mean()) / years)
    assert m.fees_paid == 12.5
    assert m.fee_drag == pytest.approx(12.5 / 1000.0)
    assert m.gross_profit == pytest.approx(30.0 + 12.5)  # P&L before fees
    assert m.fee_drag_vs_gross == pytest.approx(12.5 / 42.5)
    assert m.benchmark_total_return == pytest.approx(0.003)


def test_fee_drag_vs_gross_hand_computed():
    # Spec crypto go-live criterion: fee drag < 30% of GROSS returns.
    # Start 1000, end 1050, fees 12.5 -> gross = 50 + 12.5 = 62.5;
    # fee_drag_vs_gross = 12.5 / 62.5 = 0.20.
    equity = _curve([1000.0, 1020.0, 1050.0])
    benchmark = _curve([1000.0, 1001.0, 1002.0])
    m = metrics_from_curves(equity, benchmark, (), 0.0, fees_paid=12.5, periods_per_year=252)
    assert m.gross_profit == pytest.approx(62.5)
    assert m.fee_drag_vs_gross == pytest.approx(0.20)


def test_fee_drag_vs_gross_nan_when_no_gross_profit():
    # End 985, fees 10 -> gross = -15 + 10 = -5 <= 0: the criterion is
    # unevaluable (itself a failing state for go-live) -> NaN, not a ratio.
    equity = _curve([1000.0, 995.0, 985.0])
    benchmark = _curve([1000.0, 1001.0, 1002.0])
    m = metrics_from_curves(equity, benchmark, (), 0.0, fees_paid=10.0, periods_per_year=252)
    assert m.gross_profit == pytest.approx(-5.0)
    assert math.isnan(m.fee_drag_vs_gross)


def test_gate_requires_higher_sharpe_and_positive_total_return():
    strong = _curve([1000.0, 1012.0, 1008.0, 1035.0])
    weak_bench = _curve([1000.0, 1001.0, 999.0, 1002.0])
    m = metrics_from_curves(strong, weak_bench, (), 0.0, 0.0, 252)
    assert m.gate_passed is True

    losing = _curve([1000.0, 995.0, 998.0, 990.0])
    m2 = metrics_from_curves(losing, weak_bench, (), 0.0, 0.0, 252)
    assert m2.gate_passed is False  # negative total return fails regardless of Sharpe

    flat = _curve([1000.0, 1000.0, 1000.0, 1000.0])  # NaN sharpe
    m3 = metrics_from_curves(flat, weak_bench, (), 0.0, 0.0, 252)
    assert m3.gate_passed is False


def test_gate_is_a_conjunction_not_a_disjunction():
    # Each fixture satisfies exactly ONE gate condition, so both would pass
    # under `or` but must fail under `and` -- kills an and->or regression in
    # the go-live gate.

    # Higher Sharpe, NEGATIVE total return. Hand-derived:
    #   equity [1000, 1005, 995, 999]: returns +0.005, -0.0099502, +0.0040201
    #     -> mean -0.00031005, std(ddof=1) 0.0083630
    #     -> Sharpe = -0.037074 * sqrt(252) ~= -0.5885; total return -0.1%.
    #   bench  [1000, 990, 985, 960]: returns -0.01, -0.0050505, -0.0253807
    #     -> mean -0.0134771, std 0.0106021
    #     -> Sharpe = -1.27117 * sqrt(252) ~= -20.18.
    #   equity Sharpe (-0.59) > bench Sharpe (-20.18), but total return < 0.
    dipping = _curve([1000.0, 1005.0, 995.0, 999.0])
    crashing_bench = _curve([1000.0, 990.0, 985.0, 960.0])
    m = metrics_from_curves(dipping, crashing_bench, (), 0.0, 0.0, 252)
    assert m.sharpe > m.benchmark_sharpe
    assert m.total_return < 0.0
    assert m.gate_passed is False

    # LOWER Sharpe, positive total return. Hand-derived:
    #   equity [1000, 1030, 980, 1010]: returns +0.03, -0.0485437, +0.0306122
    #     -> mean 0.0040229, std 0.0455250
    #     -> Sharpe = 0.088367 * sqrt(252) ~= 1.40; total return +1.0%.
    #   bench  [1000, 1012, 1022, 1035]: returns +0.012, +0.0098814, +0.0127202
    #     -> mean 0.0115339, std 0.0014757
    #     -> Sharpe = 7.8155 * sqrt(252) ~= 124.06.
    #   equity total return > 0, but equity Sharpe (1.40) < bench (124.06).
    choppy = _curve([1000.0, 1030.0, 980.0, 1010.0])
    smooth_bench = _curve([1000.0, 1012.0, 1022.0, 1035.0])
    m2 = metrics_from_curves(choppy, smooth_bench, (), 0.0, 0.0, 252)
    assert m2.total_return > 0.0
    assert m2.sharpe < m2.benchmark_sharpe
    assert m2.gate_passed is False


def test_metrics_from_curves_rejects_mismatched_indexes():
    equity = _curve([1000.0, 1010.0, 1030.0])
    benchmark = _curve([1000.0, 1001.0, 1002.0], start="2025-02-01")
    with pytest.raises(ValueError, match="identical index"):
        metrics_from_curves(equity, benchmark, (), 0.0, 0.0, 252)


def test_compute_metrics_delegates_from_result():
    equity = _curve([1000.0, 1010.0, 1005.0, 1030.0])
    benchmark = _curve([1000.0, 1001.0, 1002.0, 1003.0])
    result = BacktestResult(
        venue="crypto",
        start=datetime.date(2025, 1, 1),
        end=datetime.date(2025, 1, 4),
        equity_curve=equity,
        benchmark_curve=benchmark,
        trades=(_trade(5.0),),
        open_positions=(),
        fees_paid=3.0,
        buy_notional=500.0,
        sessions_run=4,
        sessions_skipped=(),
        survivorship_ratio=1.0,
        eligible_min=4,
        eligible_mean=4.0,
        warnings=(),
    )
    m = compute_metrics(result, 365)
    assert m.trade_count == 1
    assert m.fee_drag == pytest.approx(3.0 / 1000.0)
