"""Pure per-symbol feature functions (spec: Signal Engine).

Two parallel forms live here:

- The scalar functions (``vol_adjusted_return`` etc.) each take a series
  already truncated to <= as_of by the caller and return a single float for
  that as_of. They are the readable reference definition of every feature.
- The ``*_series`` functions compute the SAME feature as a full rolling
  time-series in one vectorized pass: ``fn_series(close, ...).iloc[i]`` equals
  ``fn(close.iloc[: i + 1], ...)`` for every position ``i`` (bit-identical for
  selection/shift/division features; within floating-point noise for the ones
  built on rolling mean/std -- see trading.signals.engine for the precompute
  that relies on this equivalence).

Both forms perform no I/O and read no clock, and yield NaN when history is
insufficient (NaN symbols drop out of percentile ranking naturally). Because a
rolling window / shift / asof looks only BACKWARD, the series form reads no
data after position ``i`` -- so precomputing a feature over a symbol's whole
history and later gathering the value at some as_of can never leak the future.
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


# --- Vectorized (whole-history) forms -------------------------------------
#
# Each returns a Series aligned to `close.index`; entry i is the scalar
# feature evaluated on the window ending at i. NaN before enough history --
# the same skip the scalar form encodes as math.nan.


def _lookback_price_series(close: pd.Series, lookback: int, calendar_days: bool) -> pd.Series:
    """Vectorized _lookback_price: the reference price `lookback` back for every
    bar. Calendar mode uses asof (last close at or before the target date, NaN
    before the first bar); trading mode is a positional shift."""
    if calendar_days:
        targets = close.index - pd.Timedelta(lookback, unit="D")
        return pd.Series(close.asof(targets).to_numpy(), index=close.index)
    return close.shift(lookback)


def vol_adjusted_return_series(
    close: pd.Series, lookback: int, vol_window: int, calendar_days: bool
) -> pd.Series:
    past = _lookback_price_series(close, lookback, calendar_days)
    ret = close / past - 1.0
    vol = close.pct_change().rolling(vol_window).std()
    out = ret / vol
    # `where(cond)` keeps values where cond is True, NaN elsewhere; a NaN past
    # or vol makes cond False, matching the scalar guards (past<=0, vol<=0).
    return out.where(past > 0).where(vol > 0)


def volume_surge_series(close: pd.Series, volume: pd.Series, week: int, baseline: int) -> pd.Series:
    dollar_volume = close * volume
    base = dollar_volume.rolling(baseline).mean()
    recent = dollar_volume.rolling(week).mean()
    return (recent / base).where(base > 0)


def breakout_proximity_series(
    close: pd.Series, high: pd.Series, windows: tuple[int, int]
) -> pd.Series:
    proximities = sum(close / high.rolling(w).max() for w in windows)
    return proximities / len(windows)


def rsi_series(close: pd.Series, window: int) -> pd.Series:
    """Cutler's RSI as a rolling series (simple means, matching rsi())."""
    deltas = close.diff()
    gains = deltas.clip(lower=0.0).rolling(window).mean()
    losses = (-deltas.clip(upper=0.0)).rolling(window).mean()
    total = gains + losses
    out = 100.0 * gains / total
    # Flat window (no gains and no losses) is exactly 50, as in the scalar form.
    return out.mask(total == 0.0, 50.0)


def overextension_series(close: pd.Series, rsi_window: int, mean_window: int) -> pd.Series:
    stretch_rsi = rsi_series(close, rsi_window)
    sma = close.rolling(mean_window).mean()
    stretch = (close / sma - 1.0).clip(lower=0.0)
    return stretch_rsi / 100.0 + stretch


def raw_return_series(close: pd.Series, days: int) -> pd.Series:
    past = _lookback_price_series(close, days, calendar_days=True)
    return (close / past - 1.0).where(past > 0)
