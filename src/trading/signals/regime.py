"""Market-regime gate (spec: Signal Engine / regime gate).

Benchmark trend (price vs 50/200-day SMAs) plus realized-vol percentile map
to an exposure multiplier: risk-on = full, neutral = half, risk-off = no new
entries (exits still honored — enforced by the M2 simulator). Pure: no I/O,
no clock; as_of is a parameter.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

from trading.config import RegimeConfig

RegimeState = Literal["risk_on", "neutral", "risk_off"]


@dataclass(frozen=True)
class Regime:
    state: RegimeState
    exposure_multiplier: float


def compute_regime(
    benchmark_bars: pd.DataFrame, as_of: pd.Timestamp, config: RegimeConfig
) -> Regime:
    if as_of.tzinfo is None:
        raise ValueError("as_of must be tz-aware UTC")
    close = benchmark_bars.loc[:as_of, "close"]  # structural no-lookahead cut
    if len(close) < config.sma_slow:
        return Regime(state="neutral", exposure_multiplier=config.exposure_neutral)

    last = float(close.iloc[-1])
    sma_fast = float(close.iloc[-config.sma_fast :].mean())
    sma_slow = float(close.iloc[-config.sma_slow :].mean())

    vols = close.pct_change().rolling(config.vol_window).std().dropna()
    if vols.empty:
        return Regime(state="neutral", exposure_multiplier=config.exposure_neutral)
    trailing = vols.iloc[-config.vol_lookback :]
    current_vol = float(trailing.iloc[-1])
    vol_pct = float((trailing < current_vol).mean())
    high_vol = vol_pct >= config.vol_high_percentile

    if last < sma_slow or (last < sma_fast and high_vol):
        return Regime(state="risk_off", exposure_multiplier=config.exposure_risk_off)
    if last > sma_fast and last > sma_slow and not high_vol:
        return Regime(state="risk_on", exposure_multiplier=config.exposure_risk_on)
    return Regime(state="neutral", exposure_multiplier=config.exposure_neutral)
