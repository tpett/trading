import numpy as np
import pandas as pd
import pytest

from trading.config import SignalConfig
from trading.signals.engine import FEATURE_COLUMNS, compute_features
from trading.signals.quality import OUTPUT_COLUMNS, latest_filed_row, quality_momentum_v1
from trading.signals.registry import get_ranker

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
    ranker="quality_momentum_v1",
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


def _fund(dated: dict[str, float]) -> pd.DataFrame:
    idx = pd.DatetimeIndex([pd.Timestamp(d, tz="UTC") for d in dated], name="filed")
    return pd.DataFrame({"gross_profitability": list(dated.values())}, index=idx)


BARS = {
    "UP": _trending_bars(0.01),
    "FLAT": _trending_bars(0.0),
    "DOWN": _trending_bars(-0.01),
}
AS_OF = BARS["UP"].index[-1]


def test_registered_and_requires_fundamentals():
    spec = get_ranker("quality_momentum_v1")
    assert spec.fn is quality_momentum_v1
    assert spec.requires_fundamentals is True


def test_composite_is_equal_weight_over_seven():
    fundamentals = {
        "UP": _fund({"2025-11-10": 0.6}),
        "FLAT": _fund({"2025-11-10": 0.2}),
        "DOWN": _fund({"2025-11-10": 0.4}),
    }
    out = quality_momentum_v1(BARS, AS_OF, CONFIG, fundamentals)
    base = compute_features(BARS, AS_OF, CONFIG)
    assert list(out.columns) == OUTPUT_COLUMNS
    # quality = cross-sectional percentile of the latest filed value.
    assert out.loc["UP", "quality"] == pytest.approx(1.0)
    assert out.loc["DOWN", "quality"] == pytest.approx(2 / 3)
    assert out.loc["FLAT", "quality"] == pytest.approx(1 / 3)
    for symbol in BARS:
        expected = (
            base.loc[symbol, "composite"] * len(FEATURE_COLUMNS) + out.loc[symbol, "quality"]
        ) / (len(FEATURE_COLUMNS) + 1)
        assert out.loc[symbol, "composite"] == pytest.approx(expected)
        # The six price features and raw_return_30d pass through unchanged.
        for col in [*FEATURE_COLUMNS, "raw_return_30d"]:
            assert out.loc[symbol, col] == base.loc[symbol, col]


def test_missing_or_nan_fundamentals_are_neutral_half():
    fundamentals = {
        "UP": _fund({"2025-11-10": 0.6}),
        "FLAT": _fund({"2025-11-10": 0.2}),
        # DOWN absent entirely; and a NaN latest value must also be neutral.
    }
    out = quality_momentum_v1(BARS, AS_OF, CONFIG, fundamentals)
    assert out.loc["DOWN", "quality"] == 0.5

    fundamentals["DOWN"] = _fund({"2025-11-10": float("nan")})
    out = quality_momentum_v1(BARS, AS_OF, CONFIG, fundamentals)
    assert out.loc["DOWN", "quality"] == 0.5


def test_no_fundamentals_at_all_is_all_neutral_and_momentum_ordering():
    out = quality_momentum_v1(BARS, AS_OF, CONFIG, None)
    assert set(out["quality"]) == {0.5}
    base = compute_features(BARS, AS_OF, CONFIG)
    assert list(out.sort_values("composite").index) == list(base.sort_values("composite").index)


def test_step_function_uses_latest_value_filed_at_or_before_as_of():
    fundamentals = {
        "UP": _fund({"2025-11-10": 0.1, "2026-06-01": 0.9}),  # 2nd filing AFTER as_of
        "FLAT": _fund({"2025-11-10": 0.6}),
        "DOWN": _fund({"2025-11-10": 0.4}),
    }
    out = quality_momentum_v1(BARS, AS_OF, CONFIG, fundamentals)
    # The post-as_of filing is invisible: UP's visible value is 0.1 -> lowest percentile.
    assert out.loc["UP", "quality"] == pytest.approx(1 / 3)
    # A frame with only pre-listing/no rows as-of is neutral, not a crash.
    fundamentals["UP"] = _fund({"2026-06-01": 0.9})
    out = quality_momentum_v1(BARS, AS_OF, CONFIG, fundamentals)
    assert out.loc["UP", "quality"] == 0.5


def test_latest_filed_row_boundary_is_inclusive_of_as_of():
    frame = _fund({"2025-11-10": 0.7})
    # Filed exactly ON as_of -> visible (pins <=, not <).
    row = latest_filed_row(frame, pd.Timestamp("2025-11-10", tz="UTC"))
    assert row is not None
    assert row["gross_profitability"] == 0.7
    # Same row, as_of one day earlier -> invisible.
    assert latest_filed_row(frame, pd.Timestamp("2025-11-09", tz="UTC")) is None


def test_quality_percentile_unchanged_by_adding_nan_peer():
    fundamentals = {
        "UP": _fund({"2025-11-10": 0.6}),
        "FLAT": _fund({"2025-11-10": 0.2}),
        "DOWN": _fund({"2025-11-10": 0.4}),
    }
    before = quality_momentum_v1(BARS, AS_OF, CONFIG, fundamentals)
    # A 4th symbol with no visible fundamentals joins the universe: the
    # defined symbols' percentiles must not dilute (rank skips NaN; NaN
    # slots land on the neutral 0.5 via fillna, outside the rank pool).
    bars = {**BARS, "EXTRA": _trending_bars(0.005)}
    after = quality_momentum_v1(bars, AS_OF, CONFIG, fundamentals)
    for symbol in BARS:
        assert after.loc[symbol, "quality"] == before.loc[symbol, "quality"]
    assert after.loc["EXTRA", "quality"] == 0.5


def test_nan_latest_never_reaches_back_to_an_older_value():
    fundamentals = {
        "UP": _fund({"2025-11-10": 0.9, "2025-12-01": float("nan")}),
        "FLAT": _fund({"2025-11-10": 0.6}),
        "DOWN": _fund({"2025-11-10": 0.4}),
    }
    out = quality_momentum_v1(BARS, AS_OF, CONFIG, fundamentals)
    # UP's LATEST filing as-of is NaN -> neutral; the 0.9 is history, not current.
    assert out.loc["UP", "quality"] == 0.5


def test_empty_universe_returns_empty_frame_with_columns():
    out = quality_momentum_v1({}, AS_OF, CONFIG, None)
    assert out.empty
    assert list(out.columns) == OUTPUT_COLUMNS


def test_naive_as_of_rejected():
    with pytest.raises(ValueError, match="tz-aware"):
        quality_momentum_v1(BARS, pd.Timestamp("2026-02-27"), CONFIG, None)
