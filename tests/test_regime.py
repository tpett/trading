import dataclasses

import numpy as np
import pandas as pd
import pytest

from trading.config import RegimeConfig
from trading.signals.regime import Regime, compute_regime

CONFIG = RegimeConfig(
    sma_fast=50,
    sma_slow=200,
    vol_window=20,
    vol_lookback=252,
    vol_high_percentile=0.80,
    exposure_risk_on=1.0,
    exposure_neutral=0.5,
    exposure_risk_off=0.0,
)


def _bars_from_rets(rets: np.ndarray) -> pd.DataFrame:
    idx = pd.date_range("2025-01-01", periods=len(rets) + 1, freq="B", tz="UTC")
    close = 100 * np.concatenate([[1.0], np.cumprod(1 + rets)])
    return pd.DataFrame(
        {
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "volume": np.full(len(close), 1e6),
        },
        index=idx,
    )


def _decaying_jitter(n: int, drift: float) -> np.ndarray:
    """Trend with noise that shrinks over time, so current vol is the lowest."""
    signs = np.where(np.arange(n) % 2 == 0, 1.0, -1.0)
    return drift + signs * np.linspace(0.004, 0.001, n)


def test_uptrend_with_falling_vol_is_risk_on():
    bars = _bars_from_rets(_decaying_jitter(300, drift=0.004))
    regime = compute_regime(bars, bars.index[-1], CONFIG)
    assert regime == Regime(state="risk_on", exposure_multiplier=1.0)


def test_downtrend_is_risk_off():
    bars = _bars_from_rets(_decaying_jitter(300, drift=-0.004))
    regime = compute_regime(bars, bars.index[-1], CONFIG)
    assert regime == Regime(state="risk_off", exposure_multiplier=0.0)


def test_uptrend_with_vol_spike_is_neutral():
    # Long calm uptrend, then 11 alternating +/-8% bars ending on +8%: the close
    # stays above both SMAs but current vol is the highest on record.
    calm = _decaying_jitter(290, drift=0.004)
    spike = np.array([0.08 if i % 2 == 0 else -0.08 for i in range(11)])
    bars = _bars_from_rets(np.concatenate([calm, spike]))
    regime = compute_regime(bars, bars.index[-1], CONFIG)
    assert regime == Regime(state="neutral", exposure_multiplier=0.5)


def test_short_history_is_neutral():
    bars = _bars_from_rets(_decaying_jitter(100, drift=0.004))
    regime = compute_regime(bars, bars.index[-1], CONFIG)
    assert regime.state == "neutral"


def test_regime_no_lookahead():
    bars = _bars_from_rets(_decaying_jitter(300, drift=0.004))
    as_of = bars.index[250]
    base = compute_regime(bars, as_of, CONFIG)
    perturbed = bars.copy()
    perturbed.loc[perturbed.index > as_of, "close"] *= 0.1
    assert compute_regime(perturbed, as_of, CONFIG) == base


def test_regime_is_frozen():
    regime = Regime(state="risk_on", exposure_multiplier=1.0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        regime.state = "risk_off"
