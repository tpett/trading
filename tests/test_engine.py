import numpy as np
import pandas as pd
import pytest

from trading.config import SignalConfig
from trading.signals.engine import FEATURE_COLUMNS, OUTPUT_COLUMNS, compute_features, rank

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


def _random_walk_bars(seed: int, periods: int = 150) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2025-11-03", periods=periods, freq="B", tz="UTC")
    rets = rng.normal(0.001, 0.02, periods)
    close = 100 * np.cumprod(1 + rets)
    return pd.DataFrame(
        {
            "open": np.concatenate([[100.0], close[:-1]]),
            "high": close * (1 + rng.uniform(0.0, 0.02, periods)),
            "low": close * (1 - rng.uniform(0.0, 0.02, periods)),
            "close": close,
            "volume": rng.uniform(1e5, 1e6, periods),
        },
        index=idx,
    )


def test_columns_are_locked():
    assert OUTPUT_COLUMNS == [
        "mom_short",
        "mom_med",
        "mom_long",
        "volume_surge",
        "breakout",
        "overextension",
        "composite",
        "raw_return_30d",
    ]
    assert FEATURE_COLUMNS == OUTPUT_COLUMNS[:6]


def test_compute_features_ranks_momentum_cross_sectionally():
    bars = {
        "UP": _trending_bars(0.01),
        "FLAT": _trending_bars(0.0),
        "DOWN": _trending_bars(-0.01),
    }
    as_of = bars["UP"].index[-1]
    out = compute_features(bars, as_of, CONFIG)
    assert list(out.columns) == OUTPUT_COLUMNS
    assert set(out.index) == {"UP", "FLAT", "DOWN"}
    assert out.loc["UP", "mom_med"] > out.loc["FLAT", "mom_med"] > out.loc["DOWN", "mom_med"]
    assert out.loc["UP", "composite"] > out.loc["FLAT", "composite"]
    assert out.loc["FLAT", "composite"] > out.loc["DOWN", "composite"]
    assert out.loc["UP", "raw_return_30d"] > 0 > out.loc["DOWN", "raw_return_30d"]
    feats = out[FEATURE_COLUMNS]
    assert ((feats >= 0) & (feats <= 1)).all().all()
    # Pin the equal-weight blend and overextension inversion by exact value.
    # UP leads on every feature: percentiles all 1.0, so
    # composite = (1 + 1 + 1 + 1 + 1 + (1 - 1)) / 6 = 5/6.
    assert (out.loc["UP", FEATURE_COLUMNS] == 1.0).all()
    assert out.loc["UP", "composite"] == pytest.approx(5 / 6)
    # DOWN trails on every feature: percentiles all 1/3, so
    # composite = (5 * (1/3) + (1 - 1/3)) / 6 = 7/18.
    assert (out.loc["DOWN", FEATURE_COLUMNS] == 1 / 3).all()
    assert out.loc["DOWN", "composite"] == pytest.approx(7 / 18)


def test_symbol_with_short_history_is_dropped():
    bars = {"UP": _trending_bars(0.01), "NEW": _trending_bars(0.01, periods=10)}
    out = compute_features(bars, bars["UP"].index[-1], CONFIG)
    assert "NEW" not in out.index
    assert "UP" in out.index


def test_empty_universe_returns_empty_frame():
    as_of = pd.Timestamp("2026-07-01", tz="UTC")
    out = compute_features({}, as_of, CONFIG)
    assert list(out.columns) == OUTPUT_COLUMNS
    assert out.empty


def test_no_lookahead_property():
    """Spec property test: perturbing data after as_of must not change features at as_of."""
    bars = {f"S{i}": _random_walk_bars(i) for i in range(6)}
    as_of = bars["S0"].index[100]
    base = compute_features(bars, as_of, CONFIG)

    perturbed = {}
    for symbol, df in bars.items():
        p = df.copy()
        future = p.index > as_of
        p.loc[future, ["open", "high", "low", "close"]] *= 7.5
        p.loc[future, "volume"] *= 100.0
        perturbed[symbol] = p

    after = compute_features(perturbed, as_of, CONFIG)
    pd.testing.assert_frame_equal(base, after)


def test_rank_sorts_by_composite_desc_nans_last():
    features = pd.DataFrame(
        {"composite": [0.2, 0.9, float("nan"), 0.5]}, index=["A", "B", "C", "D"]
    )
    assert list(rank(features).index) == ["B", "D", "A", "C"]


def test_rank_breaks_composite_ties_by_symbol():
    """Ties resolve symbol-alphabetically so ranking is deterministic across runs."""
    features = pd.DataFrame(
        {"composite": [0.5, 0.9, 0.5, float("nan")]}, index=["Z", "B", "A", "C"]
    )
    expected = ["B", "A", "Z", "C"]
    assert list(rank(features).index) == expected
    assert list(rank(features).index) == expected  # repeatable
