"""IV-skew signal channel: a monthly per-name options skew series, gathered
point-in-time, plus the two cross-sectional rankers that trade the STOCK on
that options SIGNAL (spec: OPT-1 level, OPT-2 de-meaned change).

WHY a separate channel (and why it mirrors fundamentals, not momentum)
----------------------------------------------------------------------
The skew for a name is a MONTHLY observation computed from an options surface
priced ON its decision date -- exactly the cadence and point-in-time shape of
a fundamentals filing, not a daily bar. So this module mirrors the M4
fundamentals plumbing rather than the price/FeaturePanel plumbing:

* ``load_skew_store`` is the analogue of ``FundamentalsStore.load`` -- it turns
  the gathered ``samples.jsonl`` into a per-symbol frame indexed by the dates
  the market told us the skew (the ``decision_date``), tolerant of a torn final
  line the way the existing loaders are.
* ``IVSkewPanel`` is the analogue of ``signals.engine.FeaturePanel``: built
  ONCE over the whole run, it answers a per-session ``gather(symbols, as_of)``
  with a searchsorted lookup instead of a rescan. Because every stored skew is
  dated ON its decision date and gather only ever returns rows with
  ``decision_date <= as_of``, the value seen at as_of is identical whether or
  not later months are present -- the same no-lookahead guarantee FeaturePanel
  gives for rolling price features. The skew is piecewise-constant between
  monthly updates (the last decision on/before as_of stands until the next),
  matching the forward-filled step function fundamentals use.

WHY the composite sign is negated
---------------------------------
The hypothesis under test: a STEEP put skew (the market paying up for downside
protection, a large positive ``skew_put_atm``) precedes LOWER forward stock
returns. So a LOW / flat skew is the attractive, buy-side end. Both rankers
therefore set ``composite = cross-sectional percentile of -skew`` -- a flat
skew lands high (buy candidate), a steep skew lands low.

Fail-open on missing data (mirrors quality/value)
-------------------------------------------------
A symbol with no skew known on/before as_of, or a null skew leg, gets the
NEUTRAL 0.5 percentile -- it neither buys nor shorts on absent data, exactly
the fundamentals rankers' policy. Symbols lacking PRICE history are omitted
(the momentum history gate in ``compute_features``), never for skew reasons.

Purity: no I/O, no clock -- as_of is the only time input to the rankers, and
truncation of both bars and skew history to <= as_of happens here.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

from trading.config import SignalConfig
from trading.research.options_iv import skew_from_cell
from trading.signals.engine import FeaturePanel, compute_features

# The raw columns a store frame carries, in order. skew_put_atm is the PRIMARY
# skew signal (iv(otm_put) - iv(atm)); skew_put_call is secondary and often null
# (needs the otm_call leg). atm_spread is the ATM option's relative bid-ask
# spread (ask-bid)/mid -- an OPTION-ILLIQUIDITY primitive read off the same
# gathered cell, NaN when the ATM contract or its bid/ask/mid is absent/<=0. It
# rides alongside the skew columns so a single store/panel serves both the skew
# rankers and the illiquidity-veto ranker (no second gather, no second panel).
SKEW_COLUMNS = ["skew_put_atm", "skew_put_call", "atm_spread"]

# One neutral policy across the whole skew channel: a missing/NaN skew ranks at
# the cross-sectional median so it cannot tilt the book (same 0.5 as
# quality/value's fundamentals neutral).
SKEW_NEUTRAL = 0.5

# Minimum cross-section for a MEANINGFUL percentile. A percentile over a tiny
# non-NaN subset is degenerate: the single data-name always scores 1.0 (a
# guaranteed buy) whatever its skew, two names give 0.5/1.0, etc. -- so a
# thin/ragged session (a name early in its life, a sparse month) would hand a
# few names outsized composites and let them dominate the long book, a
# signal-side survivorship bias. Below this many known skews we rank the WHOLE
# session neutral (no skew trades) rather than trust a degenerate cross-section.
# In steady state (OOS from 2021-07, ~100 dense monthly names) this never binds.
SKEW_MIN_CROSS_SECTION = 10

# OPT-2 de-meaning window: strip a name's STRUCTURAL skew level by subtracting
# its own trailing mean. These are fixed by design, NOT tunable hyperparameters
# -- the walk-forward surface stays exactly entry_score_threshold x
# stop_atr_multiple (same discipline as the fundamentals overlays adding no
# knobs). 12 monthly obs ~= one year of level; min 3 so a barely-seeded name
# de-means against a real (not one-point) baseline rather than trading noise.
SKEW_CHANGE_WINDOW = 12
SKEW_CHANGE_MIN_OBS = 3

SKEW_V1_COLUMNS = ["skew", "composite", "raw_return_30d"]
SKEW_CHANGE_V1_COLUMNS = ["skew", "skew_change", "composite", "raw_return_30d"]

# Fraction of the cross-section VETOED by the hedge overlay each session: the
# most downside-hedged third of names (highest skew_put_atm) are dropped before
# the illiquidity selection runs. This is a DESIGN constant, NOT a tunable
# hyperparameter -- the walk-forward surface stays exactly
# entry_score_threshold x stop_atr_multiple (same no-extra-knobs discipline the
# skew de-meaning window keeps). The veto encodes the scan finding that the
# illiquidity premium only pays AMONG names the market is not busy insuring.
VETO_HEDGE_FRACTION = 1 / 3

ILLIQ_V1_COLUMNS = ["illiq", "composite", "raw_return_30d"]


def illiquidity_veto_v1(
    bars: dict[str, pd.DataFrame],
    as_of: pd.Timestamp,
    config: SignalConfig,
    fundamentals: dict[str, pd.DataFrame] | None = None,
    *,
    skew: dict[str, pd.DataFrame] | None = None,
    panel: FeaturePanel | None = None,
) -> pd.DataFrame:
    """VETO-then-SELECT ranker: hedge-veto the most-insured names, then buy the
    most ILLIQUID options among the survivors.

    Hypothesis (from a mid-cap / S&P-400 option-data scan)
    ------------------------------------------------------
    After removing the most downside-HEDGED names, the strongest cross-sectional
    forward-return selector among the survivors is OPTION ILLIQUIDITY -- the ATM
    option's relative bid-ask spread ``(ask-bid)/mid``. A WIDER spread (a more
    neglected, under-covered, harder-to-trade name) precedes HIGHER forward
    stock returns: an illiquidity / size / neglect premium (IC t~4.1 quarterly
    OOS, corroborated by low option volume). The book therefore tilts toward the
    high-spread end -- which is exactly why the cost assumption is load-bearing:
    an illiquidity premium can be an artefact of paying that very spread, so this
    ranker is only meaningful backtested net of honest mid-cap costs (its config
    uses 15 bps slippage, NOT the 5 bps large-cap skew basis).

    Design: veto trumps select
    --------------------------
    1. HEDGE VETO. Compute the cross-sectional hedge (``skew_put_atm``)
       distribution THIS session. Any name whose hedge sits in the TOP third
       (>= the ``1 - VETO_HEDGE_FRACTION`` quantile of the non-NaN hedges) is
       VETOED: ``composite = 0.0``, below every ``entry_score_threshold`` in the
       grid, so it is never bought no matter how illiquid its option is. The
       veto trumps the illiquidity score.
    2. SELECT AMONG SURVIVORS. For the non-vetoed names, ``composite =
       cross-sectional percentile of illiquidity`` (higher spread -> higher
       composite -> more attractive), ranked only against the OTHER survivors.

    Neutral / fail-open policy (mirrors the skew rankers)
    -----------------------------------------------------
    * A name with no hedge OR no illiquidity known at as_of gets the NEUTRAL 0.5
      percentile -- never dropped for a data reason (only a name lacking PRICE
      history is dropped, by ``compute_features``, exactly like every other
      ranker). 0.5 sits ABOVE a vetoed name (0.0) but BELOW a strongly illiquid
      survivor, an acceptable fail-open middle.
    * Thin-cross-section guard (reused ``SKEW_MIN_CROSS_SECTION``): if fewer than
      that many SURVIVORS carry a non-NaN illiquidity, the survivor percentile is
      degenerate (a singleton always scores 1.0, a guaranteed buy), so the WHOLE
      session is ranked NEUTRAL rather than trusted. In steady state (a dense
      mid-cap month) this never binds.

    ``raw_return_30d`` comes from the momentum base (the M2 fee gate reads it).
    """
    base = compute_features(bars, as_of, config, panel=panel)
    if base.empty:
        return pd.DataFrame(columns=ILLIQ_V1_COLUMNS, dtype="float64")
    known = skew or {}
    hedge = {symbol: _asof_level(known.get(symbol)) for symbol in base.index}
    illiq = {symbol: _asof_column(known.get(symbol), "atm_spread") for symbol in base.index}

    hedge_series = pd.Series(hedge, dtype="float64")
    illiq_series = pd.Series(illiq, dtype="float64")
    has_hedge = hedge_series.notna()

    # HEDGE VETO: the top VETO_HEDGE_FRACTION of the non-NaN hedge distribution.
    # Only a name whose hedge is KNOWN can be vetoed (a missing hedge is neutral,
    # fail-open, not a veto). With no known hedges at all the threshold is NaN,
    # the comparison is False everywhere, and nothing is vetoed -> the survivor
    # set is empty and the guard below ranks the whole session neutral.
    veto_threshold = hedge_series.quantile(1.0 - VETO_HEDGE_FRACTION)
    vetoed = has_hedge & (hedge_series >= veto_threshold)

    # SURVIVORS = names with a KNOWN hedge that was NOT vetoed. A name missing
    # EITHER its hedge or its illiquidity is not a ranked survivor -- it takes
    # the neutral 0.5 default below (the fail-open policy: never bought at
    # threshold, never dropped for a data reason). The illiquidity percentile is
    # computed ONLY over survivors so a vetoed or data-missing name never shifts
    # a survivor's rank.
    survivor = has_hedge & ~vetoed
    survivor_illiq = illiq_series[survivor]

    # Default everyone to neutral. Guard (mirrors _rank_negated): too few
    # survivors with a known illiquidity make the percentile degenerate (a lone
    # name always scores 1.0, a guaranteed buy), so the WHOLE session stays
    # neutral -- no veto, no buys -- rather than trust a thin cross-section.
    composite = pd.Series(SKEW_NEUTRAL, index=base.index, dtype="float64")
    if survivor_illiq.notna().sum() >= SKEW_MIN_CROSS_SECTION:
        # Higher spread -> higher percentile (NO negation: illiquidity IS the
        # buy-side end). A survivor with NaN illiquidity ranks neutral 0.5.
        survivor_rank = survivor_illiq.rank(pct=True).fillna(SKEW_NEUTRAL)
        composite.loc[survivor_rank.index] = survivor_rank
        # Apply the veto LAST so it trumps any survivor/neutral score.
        composite.loc[vetoed] = 0.0

    out = pd.DataFrame(index=base.index)
    out["illiq"] = illiq_series
    out["composite"] = composite
    out["raw_return_30d"] = base["raw_return_30d"]
    return out[ILLIQ_V1_COLUMNS]


# --- Store loading ---------------------------------------------------------


def _cell_skew(cell: dict) -> tuple[float, float]:
    """The (skew_put_atm, skew_put_call) pair for one parsed cell.

    Prefers the values the gather already stored on the cell (the
    ``data/options-iv`` format carries ``skew_put_atm``/``skew_put_call`` keys,
    null when a leg was missing). Only when those keys are ENTIRELY absent -- an
    older POC cell (``data/options-poc``) that stored contracts but no
    pre-computed skew -- do we recompute from the raw legs via ``skew_from_cell``.
    A stored explicit ``null`` is respected as NaN, never silently recomputed:
    the gather already decided that leg was untradeable.
    """
    if "skew_put_atm" in cell or "skew_put_call" in cell:
        pa = cell.get("skew_put_atm")
        pc = cell.get("skew_put_call")
        return (
            float(pa) if pa is not None else math.nan,
            float(pc) if pc is not None else math.nan,
        )
    result = skew_from_cell(cell)
    if result is None:
        return (math.nan, math.nan)
    return (
        result.skew_put_atm if result.skew_put_atm is not None else math.nan,
        result.skew_put_call if result.skew_put_call is not None else math.nan,
    )


def _cell_atm_spread(cell: dict) -> float:
    """The ATM option's relative bid-ask spread ``(ask - bid) / mid`` for one
    parsed cell -- a point-in-time OPTION-ILLIQUIDITY primitive.

    Read straight off the ATM-role contract priced ON the decision date (the
    same cell the skew comes from), so it is known at the decision with no
    lookahead. NaN when the ATM contract is absent, or any of its bid/ask/mid is
    missing or non-positive: a zero/absent mid cannot normalise a spread, and a
    non-positive quote is a bad/stale print we decline to trust rather than
    fabricate an extreme illiquidity reading from. A wider spread means a more
    neglected, harder-to-trade option -- the hypothesised illiquidity premium.
    """
    atm = None
    for raw in cell.get("contracts", []):
        if str(raw.get("role", "")).lower() == "atm":
            atm = raw
            break
    if atm is None:
        return math.nan
    bid = atm.get("bid")
    ask = atm.get("ask")
    mid = atm.get("mid")
    if bid is None or ask is None or mid is None:
        return math.nan
    bid, ask, mid = float(bid), float(ask), float(mid)
    if bid <= 0.0 or ask <= 0.0 or mid <= 0.0:
        return math.nan
    return (ask - bid) / mid


def load_skew_store(path: str | Path) -> dict[str, pd.DataFrame]:
    """Parse ``samples.jsonl`` into a per-symbol skew frame.

    Each output frame is indexed by ``decision_date`` (tz-aware UTC, sorted,
    de-duplicated keeping the LAST occurrence so a re-gathered cell supersedes
    an earlier one) with float columns ``skew_put_atm`` / ``skew_put_call``
    (NaN where the gather stored null or a leg could not be inverted) and
    ``atm_spread`` -- the ATM relative bid-ask spread ``(ask-bid)/mid``, NaN when
    the ATM contract or its bid/ask/mid is missing/<=0 (see ``_cell_atm_spread``).

    Tolerant like the existing JSONL loaders: blank lines are skipped and an
    unparseable line -- e.g. the torn final line a SIGKILLed gather leaves --
    is dropped rather than sinking a store built from thousands of good cells.
    A missing file yields an empty store (the ranker then treats every symbol as
    neutral: fail-open, never an abort).
    """
    path = Path(path)
    if not path.exists():
        return {}
    rows: dict[str, list[tuple[pd.Timestamp, float, float, float]]] = {}
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                cell = json.loads(line)
            except ValueError:
                continue
            symbol = cell.get("symbol")
            decision = cell.get("decision_date")
            if not symbol or not decision:
                continue
            try:
                ts = pd.Timestamp(decision, tz="UTC")
            except (ValueError, TypeError):
                continue
            put_atm, put_call = _cell_skew(cell)
            atm_spread = _cell_atm_spread(cell)
            rows.setdefault(symbol, []).append((ts, put_atm, put_call, atm_spread))

    store: dict[str, pd.DataFrame] = {}
    for symbol, records in rows.items():
        frame = pd.DataFrame(records, columns=["decision_date", *SKEW_COLUMNS])
        frame = frame.set_index("decision_date").sort_index(kind="mergesort")
        frame = frame[~frame.index.duplicated(keep="last")]
        store[symbol] = frame.astype("float64")
    return store


# --- Point-in-time panel ---------------------------------------------------


class IVSkewPanel:
    """Precomputed per-symbol skew series answering a PIT ``gather`` cheaply.

    Analogous to ``signals.engine.FeaturePanel``: built once from the whole
    store, each per-session ``gather`` is a searchsorted + slice, and because it
    only ever returns rows dated on/before as_of, a full-store panel and a
    per-session recompute agree (no lookahead). ``gather`` hands back each
    symbol's history TRUNCATED to as_of -- the last row is the piecewise-constant
    as-of level (OPT-1); the trailing rows feed OPT-2's own-mean de-meaning.
    """

    def __init__(self, frames: dict[str, pd.DataFrame]) -> None:
        self._frames = frames
        # int64 ns index per symbol, cached for a tz-safe searchsorted.
        self._index: dict[str, np.ndarray] = {
            symbol: frame.index.asi8 for symbol, frame in frames.items()
        }

    @classmethod
    def from_store(cls, store: dict[str, pd.DataFrame]) -> IVSkewPanel:
        return cls(store)

    def gather(self, symbols: Iterable[str], as_of: pd.Timestamp) -> dict[str, pd.DataFrame]:
        """Each symbol's skew history truncated to ``decision_date <= as_of``.

        A symbol absent from the store, or with no decision on/before as_of
        (as_of predates its first gathered month), is OMITTED from the result --
        downstream that is a neutral, never a crash. side="right" includes a
        skew dated exactly as_of (the options were priced that day, so it is
        known by the decision, matching how bars <= as_of include the as_of
        bar); a skew dated AFTER as_of is never returned.
        """
        as_of_ns = as_of.value  # int64 ns, UTC
        out: dict[str, pd.DataFrame] = {}
        for symbol in symbols:
            index = self._index.get(symbol)
            if index is None:
                continue
            pos = int(np.searchsorted(index, as_of_ns, side="right"))
            if pos == 0:
                continue  # nothing on/before as_of
            out[symbol] = self._frames[symbol].iloc[:pos]
        return out


# --- Ranker helpers --------------------------------------------------------


def _asof_column(frame: pd.DataFrame | None, column: str) -> float:
    """The as-of value of ``column``: the last (piecewise-constant) entry in a
    frame already truncated to <= as_of, or NaN when the frame is absent/empty
    or does not carry the column. Used to read either ``skew_put_atm`` (the
    hedge signal) or ``atm_spread`` (the illiquidity signal) at as_of with the
    same no-lookahead guarantee ``gather`` provides."""
    if frame is None or frame.empty or column not in frame.columns:
        return math.nan
    return float(frame[column].iloc[-1])


def _asof_level(frame: pd.DataFrame | None) -> float:
    """The as-of skew_put_atm: the last (piecewise-constant) value in a frame
    already truncated to <= as_of, or NaN when absent."""
    return _asof_column(frame, "skew_put_atm")


def _demeaned_level(frame: pd.DataFrame | None) -> float:
    """as-of skew minus the symbol's own trailing-window mean (OPT-2).

    Uses ONLY the truncated (<= as_of) history: the last SKEW_CHANGE_WINDOW
    non-NaN skews, subtracting their mean from the latest. Fewer than
    SKEW_CHANGE_MIN_OBS observations -> NaN (neutral downstream): a name with
    too little history has no reliable structural level to strip.
    """
    if frame is None or frame.empty:
        return math.nan
    series = frame["skew_put_atm"].dropna()
    if len(series) < SKEW_CHANGE_MIN_OBS:
        return math.nan
    window = series.iloc[-SKEW_CHANGE_WINDOW:]
    return float(series.iloc[-1] - window.mean())


def _rank_negated(raw: dict[str, float]) -> pd.Series:
    """Cross-sectional percentile of -skew, with NaN -> NEUTRAL 0.5.

    Negated so a LOW/flat skew (the buy-side hypothesis) ranks HIGH. A NaN input
    (missing skew) drops out of the ranking and is then filled to the median so
    it neither buys nor shorts -- fail-open, matching quality/value.

    Guard: if fewer than SKEW_MIN_CROSS_SECTION names carry a skew this session,
    the percentile is degenerate (a singleton always scores 1.0), so we rank the
    WHOLE session neutral rather than let a thin cross-section fabricate strong
    buys."""
    series = pd.Series(raw, dtype="float64")
    if series.notna().sum() < SKEW_MIN_CROSS_SECTION:
        return pd.Series(SKEW_NEUTRAL, index=series.index, dtype="float64")
    return (-series).rank(pct=True).fillna(SKEW_NEUTRAL)


def skew_v1(
    bars: dict[str, pd.DataFrame],
    as_of: pd.Timestamp,
    config: SignalConfig,
    fundamentals: dict[str, pd.DataFrame] | None = None,
    *,
    skew: dict[str, pd.DataFrame] | None = None,
    panel: FeaturePanel | None = None,
) -> pd.DataFrame:
    """OPT-1: cross-sectional skew LEVEL. composite = percentile of
    -skew_put_atm (flat skew high, steep skew low). Momentum is NOT blended in
    -- we trade the stock purely on the options signal.

    The universe of ranked symbols is the momentum PRICE-history gate
    (``compute_features``): a name lacking bars is omitted, but a name with bars
    and no skew is KEPT at the neutral 0.5 composite. ``raw_return_30d`` comes
    from that same base (the M2 fee gate reads it)."""
    base = compute_features(bars, as_of, config, panel=panel)
    if base.empty:
        return pd.DataFrame(columns=SKEW_V1_COLUMNS, dtype="float64")
    known = skew or {}
    raw = {symbol: _asof_level(known.get(symbol)) for symbol in base.index}
    out = pd.DataFrame(index=base.index)
    out["skew"] = pd.Series(raw, dtype="float64")
    out["composite"] = _rank_negated(raw)
    out["raw_return_30d"] = base["raw_return_30d"]
    return out[SKEW_V1_COLUMNS]


def skew_change_v1(
    bars: dict[str, pd.DataFrame],
    as_of: pd.Timestamp,
    config: SignalConfig,
    fundamentals: dict[str, pd.DataFrame] | None = None,
    *,
    skew: dict[str, pd.DataFrame] | None = None,
    panel: FeaturePanel | None = None,
) -> pd.DataFrame:
    """OPT-2: cross-sectional skew CHANGE, each name de-meaned by its own
    trailing mean (SKEW_CHANGE_WINDOW obs, min SKEW_CHANGE_MIN_OBS). This strips
    the structural per-name skew level -- some names simply always carry a
    steeper smirk -- so what ranks is the DEVIATION from a name's own norm.
    composite = percentile of -(skew - own trailing mean); below-own-average
    skew ranks high. Missing history / too-few obs -> neutral 0.5."""
    base = compute_features(bars, as_of, config, panel=panel)
    if base.empty:
        return pd.DataFrame(columns=SKEW_CHANGE_V1_COLUMNS, dtype="float64")
    known = skew or {}
    level = {symbol: _asof_level(known.get(symbol)) for symbol in base.index}
    change = {symbol: _demeaned_level(known.get(symbol)) for symbol in base.index}
    out = pd.DataFrame(index=base.index)
    out["skew"] = pd.Series(level, dtype="float64")
    out["skew_change"] = pd.Series(change, dtype="float64")
    out["composite"] = _rank_negated(change)
    out["raw_return_30d"] = base["raw_return_30d"]
    return out[SKEW_CHANGE_V1_COLUMNS]
