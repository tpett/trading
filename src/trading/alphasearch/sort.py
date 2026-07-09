"""Portfolio-sort return generator (spec section 3.3): scores -> daily series.

Per decision date: rank the cross-section of signal scores, form equal-weight
quantile portfolios (quintiles; terciles below 50 names; skip + record below
15 -- spec section 5.4; skip + record when the cross-section has fewer
DISTINCT scores than quantile buckets in use -- a degenerate cross-section,
e.g. ind_mom on a single-sector universe, 2026-07-09 amendment), hold to the
next decision date. Skipping a date means
"don't rebalance", never "delete the period": the prior portfolio is held
through skipped dates until the next actual rebalance. Output is two dated
DAILY return series -- ls = mean(top) - mean(bottom), lo = mean(top) -- which
give ~1250 regression observations over the 5-year discovery window vs ~60
monthly. No transaction costs here (costs belong to the survivor-stage full
backtest); monthly one-way turnover of the top quantile is reported so cost
fragility is visible early.

Forward returns are computed HERE, from panel.closes -- a separate pass from
signal scoring, which only ever sees a PanelView truncated at the decision
date. That separation is the engine's no-look-ahead guarantee.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from trading.alphasearch.panel import PanelData
from trading.alphasearch.spec import SignalSpec

QUANTILES = 5
TERCILE_BELOW = 50  # cross-section thinner than this -> terciles
MIN_NAMES = 15      # thinner than this -> skip the date entirely (journaled)


class SortError(ValueError):
    """The sort could not produce a return series (empty calendar/universe)."""


@dataclass(frozen=True)
class SortResult:
    ls: pd.Series                    # daily long/short spread (top - bottom)
    lo: pd.Series                    # daily long-only (top quantile)
    turnover_monthly: float          # mean one-way turnover of the top quantile
    skipped_dates: tuple[str, ...]   # ISO dates skipped for thin cross-sections
    n_dates: int                     # decision dates attempted (incl. skipped)
    n_names_median: float            # median cross-section size on traded dates


def assign_quantiles(scores: pd.Series, quantiles: int) -> tuple[list[str], list[str]]:
    """(top, bottom) equal-weight buckets of a NaN-free score series.

    Sort ascending by (score, symbol) -- the mergesort-on-sorted-index trick
    trading.signals.engine.rank uses, so ties resolve alphabetically and the
    sort is reproducible. np.array_split makes `quantiles` near-equal buckets;
    when n % q != 0 the EARLIER (lower-score) buckets take the extras.
    """
    ordered = scores.sort_index().sort_values(kind="mergesort")
    buckets = np.array_split(ordered.index.to_numpy(), quantiles)
    return list(buckets[-1]), list(buckets[0])


def portfolio_sort(
    panel: PanelData,
    spec: SignalSpec,
    dates: Sequence[pd.Timestamp],
    end: pd.Timestamp,
    *,
    quantiles: int = QUANTILES,
    tercile_below: int = TERCILE_BELOW,
    min_names: int = MIN_NAMES,
    symbol_subset: tuple[str, ...] | None = None,
) -> SortResult:
    """Build the daily L/S and long-only series over [dates[0], end]."""
    if not dates:
        raise SortError(f"{spec.name}: no decision dates in the window")
    # Per-symbol close-to-close returns on each symbol's OWN calendar, aligned
    # to the union calendar afterwards; a symbol's missing day stays NaN and
    # mean(skipna) simply equal-weights the members that traded.
    returns = pd.DataFrame({s: c.pct_change() for s, c in panel.closes.items()})

    ls_parts: list[pd.Series] = []
    lo_parts: list[pd.Series] = []
    tops: list[set[str]] = []
    skipped: list[str] = []
    names_per_date: list[int] = []
    # A skipped date means "don't rebalance", never "delete the period": the
    # previously formed portfolio (if any) is held through the skipped period
    # until the next actual rebalance (or the window end). Leading skips have
    # no portfolio yet, so those days contribute nothing.
    current_top: list[str] | None = None
    current_bottom: list[str] | None = None

    for i, date in enumerate(dates):
        scores = spec.fn(panel.view(date), date).dropna()
        if symbol_subset is not None:
            # Piece 3 subset check: the scored cross-section is restricted to
            # the draw; every downstream rule (min_names skip, tercile fall-
            # back, distinct-score guard) then applies to the SUBSET, exactly
            # as if the universe had been this half all along.
            scores = scores[scores.index.isin(symbol_subset)]
        if len(scores) < min_names:
            skipped.append(date.date().isoformat())
        else:
            q = quantiles if len(scores) >= tercile_below else 3
            # Degenerate cross-section guard: a single-sector segment universe
            # makes ind_mom assign ONE identical value to every symbol (and
            # any other signal can coincidentally tie this thin), so there
            # are fewer distinct scores than quantile buckets. assign_
            # quantiles' mergesort would then sort purely by symbol name,
            # producing a real-looking junk trial. Skip the date -- same
            # machinery as the <min_names rule (spec section 5.4 extension);
            # if EVERY date is this degenerate, the SortError below still
            # fires and journals an honest error trial.
            if scores.nunique() < q:
                skipped.append(date.date().isoformat())
            else:
                current_top, current_bottom = assign_quantiles(scores, q)
                tops.append(set(current_top))
                names_per_date.append(len(scores))
        if current_top is None or current_bottom is None:
            continue
        hold_end = dates[i + 1] if i + 1 < len(dates) else end
        segment = returns.loc[(returns.index > date) & (returns.index <= hold_end)]
        top_mean = segment[current_top].mean(axis=1)
        bottom_mean = segment[current_bottom].mean(axis=1)
        ls_parts.append(top_mean - bottom_mean)
        lo_parts.append(top_mean)

    if not tops:
        raise SortError(
            f"{spec.name}: every decision date skipped (cross-section < {min_names})"
        )
    ls = pd.concat(ls_parts).dropna()
    lo = pd.concat(lo_parts).dropna()
    pairs = list(zip(tops, tops[1:], strict=False))
    turnover = (
        float(np.mean([1 - len(prev & cur) / len(cur) for prev, cur in pairs]))
        if pairs
        else math.nan
    )
    return SortResult(
        ls=ls,
        lo=lo,
        turnover_monthly=turnover,
        skipped_dates=tuple(skipped),
        n_dates=len(dates),
        n_names_median=float(np.median(names_per_date)),
    )
