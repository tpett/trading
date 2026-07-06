import numpy as np
import pandas as pd
import pytest

from trading.config import SignalConfig
from trading.signals.engine import compute_features
from trading.signals.registry import RANKERS, get_ranker

CONFIG = SignalConfig(
    momentum_windows=(5, 10, 20),
    calendar_days=False,
    vol_window=5,
    volume_week=5,
    volume_baseline=20,
    breakout_windows=(10, 20),
    rsi_window=14,
    mean_window=20,
    raw_return_days=30,
    ranker="momentum_v1",
)


def _trending_bars(drift: float, periods: int = 80) -> pd.DataFrame:
    idx = pd.date_range("2025-11-03", periods=periods, freq="B", tz="UTC")
    jitter = np.where(np.arange(periods) % 2 == 0, 0.002, -0.002)
    close = 100 * np.cumprod(1 + drift + jitter)
    return pd.DataFrame(
        {
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": np.full(periods, 1e6),
        },
        index=idx,
    )


def test_get_ranker_unknown_name_raises_listing_known_names():
    with pytest.raises(ValueError, match="momentum_v1"):
        get_ranker("bogus")


def test_get_ranker_returns_registered_callable():
    assert get_ranker("momentum_v1") is RANKERS["momentum_v1"]


def test_momentum_v1_output_matches_compute_features_exactly():
    bars = {
        "UP": _trending_bars(0.01),
        "FLAT": _trending_bars(0.0),
        "DOWN": _trending_bars(-0.01),
    }
    as_of = bars["UP"].index[-1]
    ranker = get_ranker("momentum_v1")
    via_registry = ranker(bars, as_of, CONFIG)
    direct = compute_features(bars, as_of, CONFIG)
    pd.testing.assert_frame_equal(via_registry, direct)
