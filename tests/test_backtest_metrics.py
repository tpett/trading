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
    assert m.benchmark_total_return == pytest.approx(0.003)


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
        warnings=(),
    )
    m = compute_metrics(result, 365)
    assert m.trade_count == 1
    assert m.fee_drag == pytest.approx(3.0 / 1000.0)
