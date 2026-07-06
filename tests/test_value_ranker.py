import numpy as np
import pandas as pd
import pytest

from trading.config import SignalConfig
from trading.signals.engine import FEATURE_COLUMNS, compute_features
from trading.signals.registry import get_ranker
from trading.signals.value import OUTPUT_COLUMNS, value_momentum_v1

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
    ranker="value_momentum_v1",
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


def _fund(dated: dict[str, tuple[float, float, float]]) -> pd.DataFrame:
    """date -> (ttm_net_income, book_equity, shares_outstanding)."""
    idx = pd.DatetimeIndex([pd.Timestamp(d, tz="UTC") for d in dated], name="filed")
    ni, be, sh = zip(*dated.values(), strict=True)
    return pd.DataFrame(
        {"ttm_net_income": ni, "book_equity": be, "shares_outstanding": sh}, index=idx
    )


BARS = {
    "UP": _trending_bars(0.01),
    "FLAT": _trending_bars(0.0),
    "DOWN": _trending_bars(-0.01),
}
AS_OF = BARS["UP"].index[-1]


def test_registered_and_requires_fundamentals():
    spec = get_ranker("value_momentum_v1")
    assert spec.fn is value_momentum_v1
    assert spec.requires_fundamentals is True


def test_ratios_computed_at_ranking_time_and_composite_over_eight():
    fundamentals = {
        "UP": _fund({"2025-11-10": (10.0, 50.0, 100.0)}),
        "FLAT": _fund({"2025-11-10": (30.0, 40.0, 100.0)}),
        "DOWN": _fund({"2025-11-10": (20.0, 90.0, 100.0)}),
    }
    out = value_momentum_v1(BARS, AS_OF, CONFIG, fundamentals)
    base = compute_features(BARS, AS_OF, CONFIG)
    assert list(out.columns) == OUTPUT_COLUMNS
    # Recompute the expected ratios by hand: market cap uses the LAST close
    # <= as_of from bars (ratios are NOT stored -- the store is price-free).
    closes = {s: float(BARS[s].loc[:AS_OF, "close"].iloc[-1]) for s in BARS}
    ey = {s: fundamentals[s]["ttm_net_income"].iloc[-1] / (100.0 * closes[s]) for s in BARS}
    bm = {s: fundamentals[s]["book_equity"].iloc[-1] / (100.0 * closes[s]) for s in BARS}
    ey_rank = pd.Series(ey).rank(pct=True)
    bm_rank = pd.Series(bm).rank(pct=True)
    for s in BARS:
        assert out.loc[s, "earnings_yield"] == pytest.approx(ey_rank[s])
        assert out.loc[s, "book_to_market"] == pytest.approx(bm_rank[s])
        expected = (base.loc[s, "composite"] * len(FEATURE_COLUMNS) + ey_rank[s] + bm_rank[s]) / (
            len(FEATURE_COLUMNS) + 2
        )
        assert out.loc[s, "composite"] == pytest.approx(expected)
        for col in [*FEATURE_COLUMNS, "raw_return_30d"]:
            assert out.loc[s, col] == base.loc[s, col]


def test_missing_components_and_bad_market_cap_are_neutral():
    fundamentals = {
        "UP": _fund({"2025-11-10": (10.0, 50.0, 100.0)}),
        "FLAT": _fund({"2025-11-10": (30.0, 40.0, 100.0)}),
        "DOWN": _fund({"2025-11-10": (20.0, 90.0, 0.0)}),  # zero shares -> no market cap
    }
    out = value_momentum_v1(BARS, AS_OF, CONFIG, fundamentals)
    assert out.loc["DOWN", "earnings_yield"] == 0.5
    assert out.loc["DOWN", "book_to_market"] == 0.5

    # NaN NET INCOME only: earnings yield neutral, book-to-market still real
    # (DOWN's cheap price makes it the highest B/M of the three).
    fundamentals["DOWN"] = _fund({"2025-11-10": (float("nan"), 90.0, 100.0)})
    out = value_momentum_v1(BARS, AS_OF, CONFIG, fundamentals)
    assert out.loc["DOWN", "earnings_yield"] == 0.5
    assert out.loc["DOWN", "book_to_market"] == pytest.approx(1.0)


def test_negative_earnings_rank_low_not_neutral():
    # A loss-maker has a REAL (negative) earnings yield -- that is signal,
    # not missing data, and must rank at the bottom rather than 0.5.
    fundamentals = {
        "UP": _fund({"2025-11-10": (-10.0, 50.0, 100.0)}),
        "FLAT": _fund({"2025-11-10": (30.0, 40.0, 100.0)}),
        "DOWN": _fund({"2025-11-10": (20.0, 90.0, 100.0)}),
    }
    out = value_momentum_v1(BARS, AS_OF, CONFIG, fundamentals)
    assert out.loc["UP", "earnings_yield"] == pytest.approx(1 / 3)


def test_no_fundamentals_all_neutral_and_step_function_as_of():
    out = value_momentum_v1(BARS, AS_OF, CONFIG, None)
    assert set(out["earnings_yield"]) == {0.5}
    assert set(out["book_to_market"]) == {0.5}
    base = compute_features(BARS, AS_OF, CONFIG)
    assert list(out.sort_values("composite").index) == list(base.sort_values("composite").index)

    fundamentals = {
        "UP": _fund({"2026-06-01": (10.0, 50.0, 100.0)}),  # filed AFTER as_of: invisible
        "FLAT": _fund({"2025-11-10": (30.0, 40.0, 100.0)}),
        "DOWN": _fund({"2025-11-10": (20.0, 90.0, 100.0)}),
    }
    out = value_momentum_v1(BARS, AS_OF, CONFIG, fundamentals)
    assert out.loc["UP", "earnings_yield"] == 0.5
    assert out.loc["UP", "book_to_market"] == 0.5


def test_empty_universe_returns_empty_frame_with_columns():
    out = value_momentum_v1({}, AS_OF, CONFIG, None)
    assert out.empty
    assert list(out.columns) == OUTPUT_COLUMNS
