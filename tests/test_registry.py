import numpy as np
import pandas as pd
import pytest

from trading.config import SignalConfig
from trading.signals.engine import compute_features
from trading.signals.registry import RANKERS, RankerSpec, get_ranker

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


def test_get_ranker_returns_registered_spec():
    spec = get_ranker("momentum_v1")
    assert isinstance(spec, RankerSpec)
    assert spec is RANKERS["momentum_v1"]
    assert spec.requires_fundamentals is False


def test_momentum_v1_ignores_fundamentals_and_matches_compute_features():
    bars = {
        "UP": _trending_bars(0.01),
        "FLAT": _trending_bars(0.0),
        "DOWN": _trending_bars(-0.01),
    }
    as_of = bars["UP"].index[-1]
    spec = get_ranker("momentum_v1")
    with_none = spec.fn(bars, as_of, CONFIG, None)
    with_junk = spec.fn(bars, as_of, CONFIG, {"UP": pd.DataFrame()})
    direct = compute_features(bars, as_of, CONFIG)
    pd.testing.assert_frame_equal(with_none, direct)
    pd.testing.assert_frame_equal(with_junk, direct)
