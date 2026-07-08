"""Offline tests for the skew_v1 (OPT-1) and skew_change_v1 (OPT-2) rankers.

Synthetic bars + a hand-made skew store; every asserted composite is a
hand-computable cross-sectional percentile so the sign and the neutral-on-
missing policy are pinned exactly.

Fixtures are sized RELATIVE to SKEW_MIN_CROSS_SECTION so the percentile/sign
tests exercise a real (guard-satisfying) cross-section and won't silently rot
if the constant moves. The thin-cross-section guard -- which ranks a whole
session neutral rather than let a handful of names fabricate strong buys off a
degenerate percentile -- has its own dedicated tests below.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trading.config import SignalConfig
from trading.signals.registry import get_ranker
from trading.signals.skew import (
    SKEW_CHANGE_MIN_OBS,
    SKEW_CHANGE_V1_COLUMNS,
    SKEW_MIN_CROSS_SECTION,
    SKEW_NEUTRAL,
    SKEW_V1_COLUMNS,
    skew_change_v1,
    skew_v1,
)

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
    ranker="skew_v1",
)

# Universe size for the guard-satisfying tests: exactly the minimum dense
# cross-section, so N real skews clears the guard while N-1 would not.
N = SKEW_MIN_CROSS_SECTION


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


CANON_BARS = _trending_bars(0.004)
AS_OF = CANON_BARS.index[-1]


def _name(i: int) -> str:
    return f"S{i:02d}"


def _bars_for(names: list[str]) -> dict[str, pd.DataFrame]:
    # Skew rankers do not blend momentum into the composite, so identical
    # (read-only) price frames per name are fine: every name clears the history
    # gate and the composite is driven purely by skew.
    return {name: CANON_BARS for name in names}


def _skew_hist(values: list[float]) -> pd.DataFrame:
    """A per-symbol skew frame already truncated to <= as_of (what the panel
    hands a ranker). The engine does the as-of cut, so only the ORDER matters
    to the ranker (it reads the last row and the trailing-window mean); a plain
    monthly index suffices."""
    idx = pd.date_range(end="2025-10-01", periods=len(values), freq="MS", tz="UTC")
    data = {"skew_put_atm": values, "skew_put_call": [np.nan] * len(values)}
    return pd.DataFrame(data, index=idx)


def _linear_skews(n: int) -> dict[str, pd.DataFrame]:
    """n names whose as-of skew increases 0.02, 0.03, ... so S00 is flattest
    (most attractive) and S{n-1} steepest (least attractive)."""
    return {_name(i): _skew_hist([0.02 + 0.01 * i]) for i in range(n)}


def _change_hist(delta: float) -> pd.DataFrame:
    """A 3-obs history [0.10, 0.10, 0.10+delta] -> de-meaned change =
    (2/3)*delta (below own mean for delta<0, above for delta>0)."""
    return _skew_hist([0.10, 0.10, 0.10 + delta])


def _linear_changes(n: int) -> dict[str, pd.DataFrame]:
    """n names with distinct de-meaned changes spanning negative -> positive, so
    S00 sits most below its own mean (attractive) and S{n-1} most above."""
    return {_name(i): _change_hist(-0.05 + 0.01 * i) for i in range(n)}


def test_registered_requires_skew_not_fundamentals():
    for name, fn in (("skew_v1", skew_v1), ("skew_change_v1", skew_change_v1)):
        spec = get_ranker(name)
        assert spec.fn is fn
        assert spec.requires_skew is True
        assert spec.requires_fundamentals is False


# --- OPT-1: level -----------------------------------------------------------


def test_skew_v1_composite_is_percentile_of_negated_skew():
    names = [_name(i) for i in range(N)]
    out = skew_v1(_bars_for(names), AS_OF, CONFIG, skew=_linear_skews(N))
    assert list(out.columns) == SKEW_V1_COLUMNS
    # composite = cross-sectional percentile of -skew over N names: the flattest
    # (S00) tops out at 1.0, the steepest (S{N-1}) at 1/N.
    assert out.loc[_name(0), "composite"] == pytest.approx(1.0)
    assert out.loc[_name(N - 1), "composite"] == pytest.approx(1 / N)
    # An interior name is a genuine percentile: S04 is the 5th-lowest skew of N,
    # so -skew ranks it (N-4)/N.
    assert out.loc[_name(4), "composite"] == pytest.approx((N - 4) / N)
    assert out.loc[_name(4), "skew"] == pytest.approx(0.06)
    # Strictly monotone: composite decreases as skew increases (the sign of the
    # hypothesis -- flat skew buys, steep skew does not).
    composites = [out.loc[_name(i), "composite"] for i in range(N)]
    assert composites == sorted(composites, reverse=True)
    assert out.loc[_name(0), "composite"] > out.loc[_name(N - 1), "composite"]
    assert "raw_return_30d" in out.columns


def test_skew_v1_missing_skew_is_neutral_not_dropped():
    # N names carry skew (guard satisfied); one extra name has NO skew and must
    # be neutral while the rest still rank -- not the whole session going neutral.
    names = [_name(i) for i in range(N)]
    out = skew_v1(_bars_for([*names, "MISSING"]), AS_OF, CONFIG, skew=_linear_skews(N))
    assert "MISSING" in out.index  # never dropped for a skew reason (it has price history)
    assert out.loc["MISSING", "composite"] == pytest.approx(SKEW_NEUTRAL)
    assert pd.isna(out.loc["MISSING", "skew"])
    # The N real names still rank on percentile (guard did NOT fire).
    assert out.loc[_name(0), "composite"] == pytest.approx(1.0)
    assert out.loc[_name(N - 1), "composite"] == pytest.approx(1 / N)


def test_skew_v1_nan_skew_leg_is_neutral():
    names = [_name(i) for i in range(N)]
    skew = {**_linear_skews(N), "NANLEG": _skew_hist([np.nan])}
    out = skew_v1(_bars_for([*names, "NANLEG"]), AS_OF, CONFIG, skew=skew)
    assert out.loc["NANLEG", "composite"] == pytest.approx(SKEW_NEUTRAL)
    # The N present names still rank around it.
    assert out.loc[_name(0), "composite"] == pytest.approx(1.0)


def test_skew_v1_none_channel_ranks_everything_neutral():
    # The live path passes skew=None; a skew ranker there must not crash and
    # simply ranks everyone neutral (fail-open -> nothing bought at threshold).
    names = [_name(i) for i in range(N)]
    out = skew_v1(_bars_for(names), AS_OF, CONFIG, skew=None)
    assert (out["composite"] == SKEW_NEUTRAL).all()


def test_skew_v1_omits_symbols_without_price_history():
    short = {"FLAT_SKEW": _trending_bars(0.004, periods=5)}
    out = skew_v1(short, AS_OF, CONFIG, skew={"FLAT_SKEW": _skew_hist([0.02])})
    assert out.empty


def test_skew_v1_thin_cross_section_is_all_neutral():
    # >= N names have price history, but only 3 carry a skew (3 < N): the
    # percentile is degenerate, so the WHOLE session is ranked neutral rather
    # than let those 3 fabricate outsized composites.
    names = [_name(i) for i in range(N)]
    skew = {
        _name(0): _skew_hist([0.02]),
        _name(1): _skew_hist([0.05]),
        _name(2): _skew_hist([0.09]),
    }
    out = skew_v1(_bars_for(names), AS_OF, CONFIG, skew=skew)
    assert (out["composite"] == SKEW_NEUTRAL).all()


def test_skew_v1_single_steep_name_is_not_a_forced_buy():
    # The reviewer's flagged case: a LONE data-name in a thin cross-section must
    # NOT score 1.0 off a degenerate one-element percentile -- even a steep
    # (unattractive) skew has to come out neutral, never a guaranteed buy.
    names = [_name(i) for i in range(N)]
    out = skew_v1(_bars_for(names), AS_OF, CONFIG, skew={_name(0): _skew_hist([0.30])})
    assert out.loc[_name(0), "composite"] == pytest.approx(SKEW_NEUTRAL)  # not 1.0
    assert (out["composite"] == SKEW_NEUTRAL).all()


# --- OPT-2: de-meaned change -----------------------------------------------


def test_skew_change_v1_uses_only_trailing_history_for_demean():
    names = [_name(i) for i in range(N)]
    out = skew_change_v1(_bars_for(names), AS_OF, CONFIG, skew=_linear_changes(N))
    assert list(out.columns) == SKEW_CHANGE_V1_COLUMNS
    # De-meaned change hand-check for S00 (history [0.10, 0.10, 0.05]).
    assert out.loc[_name(0), "skew_change"] == pytest.approx(0.05 - (0.10 + 0.10 + 0.05) / 3)
    # Below-own-mean (S00, most negative change) ranks highest; above-own-mean
    # (S{N-1}) lowest -- percentile of -(change).
    assert out.loc[_name(0), "composite"] == pytest.approx(1.0)
    assert out.loc[_name(N - 1), "composite"] == pytest.approx(1 / N)
    composites = [out.loc[_name(i), "composite"] for i in range(N)]
    assert composites == sorted(composites, reverse=True)
    # The raw as-of level is surfaced alongside the change.
    assert out.loc[_name(N - 1), "skew"] == pytest.approx(0.10 + (-0.05 + 0.01 * (N - 1)))


def test_skew_change_v1_min_obs_guard_is_neutral():
    # A name with too few obs is neutral WHILE the dense cross-section ranks --
    # the per-name min-obs guard, distinct from the session-wide thin guard.
    assert SKEW_CHANGE_MIN_OBS == 3
    names = [_name(i) for i in range(N)]
    skew = {**_linear_changes(N), "THIN": _skew_hist([0.05, 0.05])}  # only 2 obs
    out = skew_change_v1(_bars_for([*names, "THIN"]), AS_OF, CONFIG, skew=skew)
    assert pd.isna(out.loc["THIN", "skew_change"])
    assert out.loc["THIN", "composite"] == pytest.approx(SKEW_NEUTRAL)
    assert out.loc[_name(0), "composite"] == pytest.approx(1.0)  # dense names still rank


def test_skew_change_v1_demean_ignores_nan_obs():
    # A leading NaN obs is dropped before the min-obs count and the mean.
    names = [_name(i) for i in range(N)]
    skew = {**_linear_changes(N), "TGT": _skew_hist([np.nan, 0.10, 0.10, 0.02])}
    out = skew_change_v1(_bars_for([*names, "TGT"]), AS_OF, CONFIG, skew=skew)
    assert out.loc["TGT", "skew_change"] == pytest.approx(0.02 - (0.10 + 0.10 + 0.02) / 3)


def test_skew_change_v1_thin_cross_section_is_all_neutral():
    # >= N names have price history but only 2 have a valid de-meaned change
    # (< N): the session is ranked neutral, no fabricated buys.
    names = [_name(i) for i in range(N)]
    skew = {
        _name(0): _change_hist(-0.05),
        _name(1): _change_hist(0.05),
    }
    out = skew_change_v1(_bars_for(names), AS_OF, CONFIG, skew=skew)
    assert (out["composite"] == SKEW_NEUTRAL).all()
