import math
import statistics

import pandas as pd
import pytest

from trading.signals.features import (
    breakout_proximity,
    overextension,
    raw_return,
    rsi,
    vol_adjusted_return,
    volume_surge,
)


def _series(values: list[float], freq: str = "B") -> pd.Series:
    idx = pd.date_range("2026-01-05", periods=len(values), freq=freq, tz="UTC")
    return pd.Series(values, index=idx, dtype="float64")


def test_vol_adjusted_return_trading_days():
    close = _series([100, 102, 101, 103, 106, 108])
    changes = [102 / 100 - 1, 101 / 102 - 1, 103 / 101 - 1, 106 / 103 - 1, 108 / 106 - 1]
    expected = (108 / 100 - 1) / statistics.stdev(changes)
    got = vol_adjusted_return(close, lookback=5, vol_window=5, calendar_days=False)
    assert got == pytest.approx(expected)


def test_vol_adjusted_return_calendar_days():
    close = _series([100.0] * 9 + [130.0], freq="D")
    # 7 calendar days before the last bar lands exactly on index[2] (close=100).
    changes = [0.0, 0.0, 0.0, 0.0, 0.3]
    expected = 0.3 / statistics.stdev(changes)
    got = vol_adjusted_return(close, lookback=7, vol_window=5, calendar_days=True)
    assert got == pytest.approx(expected)


def test_vol_adjusted_return_insufficient_history_is_nan():
    close = _series([100.0, 101.0, 102.0])
    assert math.isnan(vol_adjusted_return(close, lookback=5, vol_window=5, calendar_days=False))


def test_vol_adjusted_return_zero_vol_is_nan():
    close = _series([100.0] * 10)
    assert math.isnan(vol_adjusted_return(close, lookback=5, vol_window=5, calendar_days=False))


def test_volume_surge():
    close = _series([10.0] * 12)
    volume = _series([100.0] * 10 + [300.0, 300.0])
    # recent 2-day dollar volume = 3000; trailing 10-day = (8*1000 + 2*3000)/10 = 1400
    assert volume_surge(close, volume, week=2, baseline=10) == pytest.approx(3000 / 1400)


def test_volume_surge_insufficient_history_is_nan():
    close = _series([10.0] * 5)
    volume = _series([100.0] * 5)
    assert math.isnan(volume_surge(close, volume, week=2, baseline=10))


def test_breakout_proximity():
    high = _series([120.0] * 40 + [110.0] * 20)
    close = _series([100.0] * 59 + [99.0])
    # 20-day high = 110, 60-day high = 120
    expected = (99 / 110 + 99 / 120) / 2
    assert breakout_proximity(close, high, windows=(20, 60)) == pytest.approx(expected)


def test_rsi_simple_average():
    close = _series([10.0, 11.0, 10.0, 12.0, 14.0])
    # deltas +1, -1, +2, +2: mean gain 1.25, mean loss 0.25
    assert rsi(close, window=4) == pytest.approx(100 * 1.25 / 1.5)


def test_rsi_all_gains_is_100():
    close = _series([10.0, 11.0, 12.0, 13.0, 14.0])
    assert rsi(close, window=4) == pytest.approx(100.0)


def test_rsi_flat_is_50():
    close = _series([10.0] * 5)
    assert rsi(close, window=4) == pytest.approx(50.0)


def test_overextension_above_mean():
    close = _series([10.0, 11.0, 10.0, 12.0, 14.0])
    sma = (10 + 11 + 10 + 12 + 14) / 5  # 11.4
    expected = (100 * 1.25 / 1.5) / 100 + (14 / sma - 1)
    assert overextension(close, rsi_window=4, mean_window=5) == pytest.approx(expected)


def test_overextension_below_mean_has_no_stretch_term():
    close = _series([14.0, 12.0, 13.0, 12.0, 10.0])
    # deltas -2, +1, -1, -2: mean gain 0.25, mean loss 1.25; close < mean so stretch = 0
    expected = (100 * 0.25 / 1.5) / 100
    assert overextension(close, rsi_window=4, mean_window=5) == pytest.approx(expected)


def test_raw_return_calendar_days():
    close = _series([100.0] * 34 + [130.0], freq="D")
    assert raw_return(close, days=30) == pytest.approx(0.3)


def test_raw_return_insufficient_history_is_nan():
    close = _series([100.0] * 5, freq="D")
    assert math.isnan(raw_return(close, days=30))
