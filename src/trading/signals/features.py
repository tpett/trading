"""Pure per-symbol feature functions (spec: Signal Engine).

Every function takes series already truncated to <= as_of by the caller,
performs no I/O and reads no clock, and returns math.nan when history is
insufficient (NaN symbols drop out of percentile ranking naturally).
"""

from __future__ import annotations

import math

import pandas as pd


def _lookback_price(close: pd.Series, lookback: int, calendar_days: bool) -> float:
    if calendar_days:
        target = close.index[-1] - pd.Timedelta(lookback, unit="D")
        if target < close.index[0]:
            return math.nan
        return float(close.asof(target))
    if len(close) <= lookback:
        return math.nan
    return float(close.iloc[-1 - lookback])


def vol_adjusted_return(
    close: pd.Series, lookback: int, vol_window: int, calendar_days: bool
) -> float:
    past = _lookback_price(close, lookback, calendar_days)
    if math.isnan(past) or past <= 0:
        return math.nan
    changes = close.pct_change().iloc[-vol_window:]
    if changes.isna().any() or len(changes) < vol_window:
        return math.nan
    vol = float(changes.std())
    if not vol > 0:
        return math.nan
    return (float(close.iloc[-1]) / past - 1.0) / vol


def volume_surge(close: pd.Series, volume: pd.Series, week: int, baseline: int) -> float:
    dollar_volume = close * volume
    if len(dollar_volume) < baseline:
        return math.nan
    base = float(dollar_volume.iloc[-baseline:].mean())
    if not base > 0:
        return math.nan
    return float(dollar_volume.iloc[-week:].mean()) / base


def breakout_proximity(close: pd.Series, high: pd.Series, windows: tuple[int, int]) -> float:
    if len(high) < max(windows):
        return math.nan
    last = float(close.iloc[-1])
    proximities = [last / float(high.iloc[-w:].max()) for w in windows]
    return sum(proximities) / len(proximities)


def rsi(close: pd.Series, window: int) -> float:
    """Cutler's RSI: simple means, hand-computable in fixtures."""
    deltas = close.diff().iloc[-window:]
    if deltas.isna().any() or len(deltas) < window:
        return math.nan
    gains = float(deltas.clip(lower=0.0).mean())
    losses = float((-deltas.clip(upper=0.0)).mean())
    if gains + losses == 0:
        return 50.0
    return 100.0 * gains / (gains + losses)


def overextension(close: pd.Series, rsi_window: int, mean_window: int) -> float:
    """Higher = more stretched. The composite counts this negatively."""
    stretch_rsi = rsi(close, rsi_window)
    if math.isnan(stretch_rsi) or len(close) < mean_window:
        return math.nan
    sma = float(close.iloc[-mean_window:].mean())
    stretch = max(float(close.iloc[-1]) / sma - 1.0, 0.0)
    return stretch_rsi / 100.0 + stretch


def raw_return(close: pd.Series, days: int) -> float:
    """Un-normalized calendar-day return; feeds the M2 crypto fee-adjusted entry gate."""
    past = _lookback_price(close, days, calendar_days=True)
    if math.isnan(past) or not past > 0:
        return math.nan
    return float(close.iloc[-1]) / past - 1.0
