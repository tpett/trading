"""Offline tests for the illiquidity_veto_v1 ranker (hedge-veto, then select on
option illiquidity).

Synthetic bars + hand-made skew/illiquidity frames; every asserted composite is
a hand-computable outcome so the veto-trumps-select ordering, the survivor
percentile sign, the neutral-on-missing policy, the no-lookahead cut, and the
thin-cross-section guard are pinned exactly. Nothing touches the network or the
real gathered data.

Fixtures are sized so that after the top-third hedge VETO the SURVIVOR count
still clears SKEW_MIN_CROSS_SECTION -- otherwise the guard fires and the whole
session goes neutral (its own dedicated test below). They are expressed relative
to the constant so they will not silently rot if it moves.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trading.config import SignalConfig
from trading.signals.registry import get_ranker
from trading.signals.skew import (
    ILLIQ_V1_COLUMNS,
    SKEW_MIN_CROSS_SECTION,
    SKEW_NEUTRAL,
    IVSkewPanel,
    illiquidity_veto_v1,
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
    ranker="illiquidity_veto_v1",
)

# Two clean hedge clusters: N_SURV low-hedge survivors and N_VETO high-hedge
# names. N_SURV >= SKEW_MIN_CROSS_SECTION so the survivor cross-section clears
# the guard after the veto. N_VETO is deliberately a comfortably-large chunk so
# the 2/3 hedge quantile pins to the HIGH cluster value (1.0) and only the
# high-hedge names are vetoed -- robust even when a test adds one extra low-hedge
# name (which would otherwise slide a knife-edge quantile onto the low value).
N_SURV = SKEW_MIN_CROSS_SECTION + 2
N_VETO = 8

LOW_HEDGE = 0.01   # clearly below the 2/3 hedge quantile -> a survivor
HIGH_HEDGE = 1.00  # the top-cluster hedge -> vetoed


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


def _bars_for(names: list[str]) -> dict[str, pd.DataFrame]:
    return {name: CANON_BARS for name in names}


def _hist(hedge: float, illiq: float, periods: int = 1) -> pd.DataFrame:
    """A per-symbol skew/illiquidity frame already truncated to <= as_of (what
    the panel hands a ranker). Only the LAST row is the as-of value; earlier
    rows are identical fillers. Carries the store's three columns."""
    idx = pd.date_range(end="2025-10-01", periods=periods, freq="MS", tz="UTC")
    return pd.DataFrame(
        {
            "skew_put_atm": [hedge] * periods,
            "skew_put_call": [np.nan] * periods,
            "atm_spread": [illiq] * periods,
        },
        index=idx,
    )


def _survivor_name(i: int) -> str:
    return f"S{i:02d}"


def _veto_name(i: int) -> str:
    return f"V{i:02d}"


def _base_universe(survivor_illiq, veto_illiq: float = 0.99):
    """N_SURV low-hedge survivors whose illiquidity is survivor_illiq[i], plus
    N_VETO high-hedge (top-third) names. veto_illiq is deliberately HIGHER than
    any survivor's spread so the tests prove the veto trumps a strong illiquidity
    score. Returns (names, skew_channel)."""
    skew = {}
    names = []
    for i in range(N_SURV):
        name = _survivor_name(i)
        names.append(name)
        skew[name] = _hist(LOW_HEDGE, survivor_illiq[i])
    for i in range(N_VETO):
        name = _veto_name(i)
        names.append(name)
        skew[name] = _hist(HIGH_HEDGE, veto_illiq)
    return names, skew


def test_registered_requires_skew_not_fundamentals():
    spec = get_ranker("illiquidity_veto_v1")
    assert spec.fn is illiquidity_veto_v1
    assert spec.requires_skew is True
    assert spec.requires_fundamentals is False


def test_survivors_ranked_by_illiquidity():
    # Survivor i has spread 0.01*(i+1): S00 tightest (least attractive), the last
    # survivor widest (most attractive). Composite = percentile of illiquidity
    # among the N_SURV survivors -> widest tops out at 1.0, tightest at 1/N_SURV.
    illiq = [0.01 * (i + 1) for i in range(N_SURV)]
    names, skew = _base_universe(illiq)
    out = illiquidity_veto_v1(_bars_for(names), AS_OF, CONFIG, skew=skew)
    assert list(out.columns) == ILLIQ_V1_COLUMNS
    assert out.loc[_survivor_name(N_SURV - 1), "composite"] == pytest.approx(1.0)
    assert out.loc[_survivor_name(0), "composite"] == pytest.approx(1 / N_SURV)
    # Strictly monotone in the spread (higher illiquidity -> higher composite).
    comps = [out.loc[_survivor_name(i), "composite"] for i in range(N_SURV)]
    assert comps == sorted(comps)
    # The raw ATM spread is surfaced as illiq.
    assert out.loc[_survivor_name(3), "illiq"] == pytest.approx(0.04)
    assert "raw_return_30d" in out.columns


def test_top_third_hedge_vetoed_even_if_illiquid():
    # The vetoed names carry the HIGHEST illiquidity of the whole session (0.99),
    # yet the top-third hedge veto forces composite 0.0 -- never bought, below
    # every entry_score_threshold -- and strictly below every survivor.
    illiq = [0.01 * (i + 1) for i in range(N_SURV)]
    names, skew = _base_universe(illiq, veto_illiq=0.99)
    out = illiquidity_veto_v1(_bars_for(names), AS_OF, CONFIG, skew=skew)
    for i in range(N_VETO):
        assert out.loc[_veto_name(i), "composite"] == pytest.approx(0.0)
    # The most-illiquid VETOED name (0.99) is not the winner; a survivor is.
    assert out["composite"].idxmax().startswith("S")
    assert out.loc[_survivor_name(N_SURV - 1), "composite"] > out.loc[_veto_name(0), "composite"]


def test_neutral_on_missing_hedge_not_dropped():
    # A name with NO skew frame at all (no hedge, no illiq) is neutral 0.5 --
    # never dropped (it has price history), never vetoed, never ranked.
    illiq = [0.01 * (i + 1) for i in range(N_SURV)]
    names, skew = _base_universe(illiq)
    out = illiquidity_veto_v1(_bars_for([*names, "NOHEDGE"]), AS_OF, CONFIG, skew=skew)
    assert "NOHEDGE" in out.index
    assert out.loc["NOHEDGE", "composite"] == pytest.approx(SKEW_NEUTRAL)
    assert pd.isna(out.loc["NOHEDGE", "illiq"])
    # The survivors still rank around it (guard did not fire).
    assert out.loc[_survivor_name(N_SURV - 1), "composite"] == pytest.approx(1.0)


def test_neutral_on_missing_illiquidity():
    # A name with a known (low) hedge but NaN atm_spread is a survivor with no
    # illiquidity -> neutral 0.5, not a fabricated rank, not vetoed.
    illiq = [0.01 * (i + 1) for i in range(N_SURV)]
    names, skew = _base_universe(illiq)
    skew["NOILLIQ"] = _hist(LOW_HEDGE, np.nan)
    out = illiquidity_veto_v1(_bars_for([*names, "NOILLIQ"]), AS_OF, CONFIG, skew=skew)
    assert out.loc["NOILLIQ", "composite"] == pytest.approx(SKEW_NEUTRAL)
    assert pd.isna(out.loc["NOILLIQ", "illiq"])


def test_no_lookahead_uses_only_asof_spread():
    # End-to-end PIT: a survivor's spread jumps from 0.02 (as-of) to 0.90 in a
    # LATER month. Gathered at an as_of between the two, the ranker must see only
    # the as-of 0.02, never the future 0.90 (no lookahead).
    illiq = [0.01 * (i + 1) for i in range(N_SURV)]
    names, skew = _base_universe(illiq)
    as_of = CANON_BARS.index[40]  # a decision date well inside the bar history
    # Two real decision dates bracketing as_of: one before (the as-of value) and
    # one after (a future spike the ranker must never see).
    dates = pd.DatetimeIndex([CANON_BARS.index[20], CANON_BARS.index[60]])
    target = pd.DataFrame(
        {
            "skew_put_atm": [LOW_HEDGE, LOW_HEDGE],
            "skew_put_call": [np.nan, np.nan],
            "atm_spread": [0.02, 0.90],  # as-of value, then a FUTURE spike
        },
        index=dates,
    )
    store = {**skew, "TARGET": target}
    panel = IVSkewPanel.from_store(store)
    gathered = panel.gather([*names, "TARGET"], as_of)
    out = illiquidity_veto_v1(_bars_for([*names, "TARGET"]), as_of, CONFIG, skew=gathered)
    assert out.loc["TARGET", "illiq"] == pytest.approx(0.02)  # the as-of spread, not 0.90


def test_thin_cross_section_is_all_neutral():
    # Fewer than SKEW_MIN_CROSS_SECTION survivors carry an illiquidity, so the
    # survivor percentile is degenerate: the WHOLE session is neutral -- even the
    # would-be-vetoed high-hedge names come out 0.5 (no veto, no buys).
    few = SKEW_MIN_CROSS_SECTION - 1
    skew = {}
    names = []
    for i in range(few):
        name = _survivor_name(i)
        names.append(name)
        skew[name] = _hist(LOW_HEDGE, 0.01 * (i + 1))
    for i in range(N_VETO):
        name = _veto_name(i)
        names.append(name)
        skew[name] = _hist(HIGH_HEDGE, 0.99)
    out = illiquidity_veto_v1(_bars_for(names), AS_OF, CONFIG, skew=skew)
    assert (out["composite"] == SKEW_NEUTRAL).all()


def test_none_channel_ranks_everything_neutral():
    # The live path passes skew=None; the ranker must not crash and ranks
    # everyone neutral (fail-open -> nothing bought at threshold).
    names = [_survivor_name(i) for i in range(N_SURV)]
    out = illiquidity_veto_v1(_bars_for(names), AS_OF, CONFIG, skew=None)
    assert (out["composite"] == SKEW_NEUTRAL).all()


def test_omits_symbols_without_price_history():
    short = {"THIN": _trending_bars(0.004, periods=5)}
    out = illiquidity_veto_v1(short, AS_OF, CONFIG, skew={"THIN": _hist(LOW_HEDGE, 0.05)})
    assert out.empty
