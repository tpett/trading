"""Identity + no-lookahead proofs for the vectorized feature precompute.

The backtester precomputes each symbol's rolling feature time-series once and
gathers the value at each session's as_of, instead of recomputing 7 rolling
features per symbol per session. These tests pin that the precompute is a
faithful, lookahead-free stand-in for the original per-as_of scalar path:

1. ``test_vectorized_raw_matches_scalar_reference`` -- the vectorized raw
   feature values equal the scalar reference. breakout and raw_return_30d are
   bit-identical (selection / shift / asof / division only); the rolling
   mean/std features (mom_short/med/long, volume_surge, overextension) can
   differ by floating-point noise because pandas' rolling mean/std use a
   different summation order than a slice ``.mean()`` / ``.std()`` -- bounded
   here at 1e-9, and far below any level that could flip a rank.
2. ``test_vectorized_output_and_ranks_match_scalar_reference`` -- the full
   cross-sectional table (percentiles + composite) is EXACTLY equal and the
   rank ordering is identical: ``rank(pct=True)`` discretizes the raw values,
   so sub-ULP raw noise is absorbed and never flips a decision.
3. ``test_precompute_has_no_lookahead`` -- a value gathered from a full-span
   panel equals one gathered from a panel built only from bars <= as_of, and
   perturbing bars after as_of changes nothing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trading.config import SignalConfig
from trading.signals.engine import (
    FEATURE_COLUMNS,
    OUTPUT_COLUMNS,
    RAW_COLUMNS,
    FeaturePanel,
    compute_features,
    min_history_rows,
    rank,
)
from trading.signals.features import (
    breakout_proximity,
    overextension,
    raw_return,
    vol_adjusted_return,
    volume_surge,
)

# Two configs: trading-day momentum (equities) and calendar-day momentum
# (crypto / the golden fixture). Both momentum paths are exercised.
TRADING = SignalConfig(
    momentum_windows=(3, 5, 8),
    calendar_days=False,
    vol_window=5,
    volume_week=3,
    volume_baseline=10,
    breakout_windows=(5, 8),
    rsi_window=5,
    mean_window=5,
    raw_return_days=5,
    ranker="momentum_v1",
)
CALENDAR = SignalConfig(**{**TRADING.__dict__, "calendar_days": True})

# Columns bit-identical to the scalar path vs. those allowed rolling FP noise.
EXACT_RAW = ["breakout", "raw_return_30d"]
TOLERANCE_RAW = ["mom_short", "mom_med", "mom_long", "volume_surge", "overextension"]
RAW_TOLERANCE = 1e-9


def _bars(seed: int, periods: int, *, freq: str, gaps: bool = False) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2025-01-01", periods=periods, freq=freq, tz="UTC")
    if gaps:  # drop a few interior bars so the calendar-day asof lookback is exercised
        keep = np.ones(periods, dtype=bool)
        keep[[periods // 3, periods // 3 + 1, 2 * periods // 3]] = False
        idx = idx[keep]
    n = len(idx)
    rets = rng.normal(0.001, 0.02, n)
    close = 100 * np.cumprod(1 + rets)
    open_ = np.concatenate([[100.0], close[:-1]])
    return pd.DataFrame(
        {
            "open": open_,
            "high": np.maximum(open_, close) * (1 + rng.uniform(0, 0.02, n)),
            "low": np.minimum(open_, close) * (1 - rng.uniform(0, 0.02, n)),
            "close": close,
            "volume": rng.uniform(1e5, 1e6, n),
        },
        index=idx,
    )


def _universe(freq: str, gaps: bool) -> dict[str, pd.DataFrame]:
    bars = {f"S{i}": _bars(i, 90, freq=freq, gaps=gaps) for i in range(6)}
    # A degenerate zero-volume symbol (volume_surge -> NaN but enough history:
    # stays in the cross-section with a NaN composite) and a too-short symbol
    # (dropped by the history gate) exercise both skip semantics.
    bars["ZEROVOL"] = _bars(99, 90, freq=freq, gaps=gaps).assign(volume=0.0)
    bars["SHORT"] = _bars(7, 8, freq=freq)
    return bars


def _reference_raw(
    bars: dict[str, pd.DataFrame], as_of: pd.Timestamp, config: SignalConfig
) -> pd.DataFrame:
    """The original per-as_of scalar path (pre-percentile), verbatim."""
    required = min_history_rows(config)
    short, med, long_ = config.momentum_windows
    rows: dict[str, dict[str, float]] = {}
    for symbol, df in bars.items():
        window = df.loc[:as_of]
        if len(window) < required:
            continue
        close, high, volume = window["close"], window["high"], window["volume"]
        rows[symbol] = {
            "mom_short": vol_adjusted_return(close, short, config.vol_window, config.calendar_days),
            "mom_med": vol_adjusted_return(close, med, config.vol_window, config.calendar_days),
            "mom_long": vol_adjusted_return(close, long_, config.vol_window, config.calendar_days),
            "volume_surge": volume_surge(close, volume, config.volume_week, config.volume_baseline),
            "breakout": breakout_proximity(close, high, config.breakout_windows),
            "overextension": overextension(close, config.rsi_window, config.mean_window),
            "raw_return_30d": raw_return(close, config.raw_return_days),
        }
    if not rows:
        return pd.DataFrame(columns=RAW_COLUMNS, dtype="float64")
    return pd.DataFrame.from_dict(rows, orient="index")[RAW_COLUMNS]


def _reference_output(
    bars: dict[str, pd.DataFrame], as_of: pd.Timestamp, config: SignalConfig
) -> pd.DataFrame:
    raw = _reference_raw(bars, as_of, config)
    if raw.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS, dtype="float64")
    pct = raw[FEATURE_COLUMNS].rank(pct=True)
    pct["composite"] = (
        pct["mom_short"]
        + pct["mom_med"]
        + pct["mom_long"]
        + pct["volume_surge"]
        + pct["breakout"]
        + (1.0 - pct["overextension"])
    ) / 6.0
    pct["raw_return_30d"] = raw["raw_return_30d"]
    return pct[OUTPUT_COLUMNS]


def _as_of_dates(bars: dict[str, pd.DataFrame]) -> list[pd.Timestamp]:
    idx = bars["S0"].index
    return [idx[20], idx[45], idx[70], idx[-1]]


def _configs_and_universes():
    return [
        (TRADING, _universe("B", gaps=False)),
        (CALENDAR, _universe("D", gaps=True)),
    ]


@pytest.mark.parametrize("config,bars", _configs_and_universes())
def test_vectorized_raw_matches_scalar_reference(config, bars):
    panel = FeaturePanel.from_bars(bars, config)
    for as_of in _as_of_dates(bars):
        ref = _reference_raw(bars, as_of, config)
        got = panel.gather(bars.keys(), as_of).reindex(ref.index)
        # Same symbols survive the history gate in both paths.
        assert set(got.index) == set(ref.index)
        # Bit-identical for selection / shift / asof / division features.
        for col in EXACT_RAW:
            pd.testing.assert_series_equal(got[col], ref[col], check_exact=True)
        # Rolling mean/std features: identical up to floating-point noise only.
        for col in TOLERANCE_RAW:
            diff = (got[col] - ref[col]).abs()
            assert (diff.dropna() <= RAW_TOLERANCE).all(), (col, diff.max())
            # NaN masks must line up exactly (a NaN is a real skip, not noise).
            assert (got[col].isna() == ref[col].isna()).all()


@pytest.mark.parametrize("config,bars", _configs_and_universes())
def test_vectorized_output_and_ranks_match_scalar_reference(config, bars):
    panel = FeaturePanel.from_bars(bars, config)
    for as_of in _as_of_dates(bars):
        ref = _reference_output(bars, as_of, config)
        got = compute_features(bars, as_of, config, panel=panel)
        # Percentiles + composite are EXACT: rank(pct) discretizes away any
        # sub-ULP raw noise. raw_return_30d is bit-identical on its own.
        pd.testing.assert_frame_equal(got, ref, check_exact=True)
        # And the decision-carrying rank ordering is identical.
        assert list(rank(got).index) == list(rank(ref).index)
    # A rank flip anywhere would have failed assert_frame_equal above; make the
    # invariant explicit so a future regression names it.


def test_panel_gather_matches_from_scratch_compute():
    """compute_features with a shared panel == compute_features building its own."""
    config, bars = TRADING, _universe("B", gaps=False)
    panel = FeaturePanel.from_bars(bars, config)
    for as_of in _as_of_dates(bars):
        with_panel = compute_features(bars, as_of, config, panel=panel)
        without = compute_features(bars, as_of, config)
        pd.testing.assert_frame_equal(with_panel, without, check_exact=True)


@pytest.mark.parametrize("config,bars", _configs_and_universes())
def test_precompute_has_no_lookahead(config, bars):
    """A value gathered from a full-span panel must equal one gathered from a
    panel built only from bars <= as_of, and perturbing the future changes
    nothing. This is exactly where a bad vectorization would leak the future."""
    full_panel = FeaturePanel.from_bars(bars, config)

    perturbed = {}
    for symbol, df in bars.items():
        p = df.copy()
        future = p.index > _as_of_dates(bars)[1]
        p.loc[future, ["open", "high", "low", "close"]] *= 9.0
        p.loc[future, "volume"] *= 50.0
        perturbed[symbol] = p
    perturbed_panel = FeaturePanel.from_bars(perturbed, config)

    for as_of in _as_of_dates(bars)[:2]:
        full = full_panel.gather(bars.keys(), as_of)
        truncated = {s: df.loc[:as_of] for s, df in bars.items()}
        trunc = FeaturePanel.from_bars(truncated, config).gather(truncated.keys(), as_of)
        pd.testing.assert_frame_equal(full.sort_index(), trunc.sort_index(), check_exact=True)
        # Future perturbation (beyond as_of=dates[1]) leaves as_of=dates[0] and
        # dates[1] untouched.
        pert = perturbed_panel.gather(bars.keys(), as_of)
        pd.testing.assert_frame_equal(full.sort_index(), pert.sort_index(), check_exact=True)


def test_history_skip_boundary_inclusive_at_exactly_required():
    """A symbol with EXACTLY min_history_rows bars at as_of is INCLUDED; one
    bar fewer is EXCLUDED -- pins the `< required` gate against a `<=` mutation
    (the one surviving mutation the identity tests didn't cover)."""
    config = TRADING
    required = min_history_rows(config)
    df = _bars(123, required, freq="B")  # exactly `required` bars
    panel = FeaturePanel.from_bars({"EXACT": df}, config)
    # as_of = last bar -> exactly `required` bars <= as_of -> INCLUDED
    assert "EXACT" in panel.gather(["EXACT"], df.index[-1]).index
    # as_of = penultimate bar -> only required-1 bars <= as_of -> EXCLUDED
    assert "EXACT" not in panel.gather(["EXACT"], df.index[-2]).index
