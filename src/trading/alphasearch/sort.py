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

Concentration axis (2026-07-11 amendment, docs/superpowers/specs/2026-07-11-
concentration-axis-amendment.md section 2): `top_n` swaps the quantile
construction for a FIXED COUNT -- the N highest-signal-score names (bottom
= N lowest, for the ls diagnostic) -- at each decision date, equal-weight,
same hold-to-next-rebalance machinery. Default-off: top_n=None reproduces
the quantile path bit-for-bit (the `current_ls_valid` flag below is only
ever set False by the top_n branch, so the quintile path's ls is untouched).
The cross-section floor becomes N (min_names/tercile_below are irrelevant to
this path); a cross-section with N <= n < 2N would make top/bottom overlap,
so that holding period's ls contribution is NaN (degrade gracefully -- lo,
the gate series, is unaffected) rather than silently inflate the spread.
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


# (decision date, top, bottom) at one ACTUAL rebalance. Piece 3's arithmetic
# checks and cost/capacity analysis replay holdings from this record.
Membership = tuple[pd.Timestamp, tuple[str, ...], tuple[str, ...]]


@dataclass(frozen=True)
class SortResult:
    ls: pd.Series                    # daily long/short spread (top - bottom)
    lo: pd.Series                    # daily long-only (top quantile)
    turnover_monthly: float          # mean one-way turnover of the top quantile
    skipped_dates: tuple[str, ...]   # ISO dates skipped for thin cross-sections
    n_dates: int                     # decision dates attempted (incl. skipped)
    n_names_median: float            # median cross-section size on traded dates
    rebalances: tuple[Membership, ...] = ()  # memberships at ACTUAL rebalances


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


def assign_top_n(scores: pd.Series, n: int) -> tuple[list[str], list[str]]:
    """(top, bottom) = the n highest / n lowest scores of a NaN-free series.

    Same deterministic tie-handling as assign_quantiles: sort ascending by
    (score, symbol) -- the mergesort-on-sorted-index trick -- so ties resolve
    alphabetically and the selection is reproducible. When the cross-section
    has fewer than 2*n names, the returned top and bottom sets can share
    members; portfolio_sort's overlap guard (section 2 of the concentration
    axis amendment) handles that, not this function.
    """
    ordered = scores.sort_index().sort_values(kind="mergesort")
    return list(ordered.index[-n:]), list(ordered.index[:n])


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
    top_n: int | None = None,
) -> SortResult:
    """Build the daily L/S and long-only series over [dates[0], end].

    top_n=None (default): today's quantile construction, bit-for-bit.
    top_n=N: fixed-count top-N/bottom-N construction (concentration axis
    amendment); min_names/tercile_below/quantiles are then irrelevant -- the
    cross-section floor is N, see assign_top_n and the module docstring.
    """
    if not dates:
        raise SortError(f"{spec.name}: no decision dates in the window")
    if top_n is not None and top_n < 1:
        raise SortError(f"{spec.name}: top_n must be >= 1, got {top_n}")
    # Per-symbol close-to-close returns on each symbol's OWN calendar, aligned
    # to the union calendar afterwards; a symbol's missing day stays NaN and
    # mean(skipna) simply equal-weights the members that traded.
    returns = pd.DataFrame({s: c.pct_change() for s, c in panel.closes.items()})

    ls_parts: list[pd.Series] = []
    lo_parts: list[pd.Series] = []
    tops: list[set[str]] = []
    skipped: list[str] = []
    names_per_date: list[int] = []
    rebalances: list[Membership] = []
    # A skipped date means "don't rebalance", never "delete the period": the
    # previously formed portfolio (if any) is held through the skipped period
    # until the next actual rebalance (or the window end). Leading skips have
    # no portfolio yet, so those days contribute nothing.
    current_top: list[str] | None = None
    current_bottom: list[str] | None = None
    # Only the top_n branch below ever sets this False; the quantile branch
    # never touches it, so it stays True for the life of a top_n=None call --
    # the ls_parts.append line degrades to today's unconditional expression,
    # bit-identical (spec section 2's overlap guard is top_n-only).
    current_ls_valid = True

    for i, date in enumerate(dates):
        scores = spec.fn(panel.view(date), date).dropna()
        if symbol_subset is not None:
            # Piece 3 subset check: scores are computed above on the FULL
            # cross-section (a cross-sectional term, e.g. ind_mom's sector
            # mean, still sees every name) and only THEN restricted to the
            # draw -- a membership perturbation with the signal held fixed,
            # not a re-score of a smaller universe. Every downstream rule
            # (min_names skip, tercile fallback, distinct-score guard) then
            # applies to the SUBSET.
            scores = scores[scores.index.isin(symbol_subset)]
        if top_n is not None:
            # Concentration axis (section 2): the floor is N itself, and
            # min_names/tercile_below/the degenerate-score guard are all
            # irrelevant -- assign_top_n's tie-break is deterministic
            # regardless of how many distinct scores exist.
            if len(scores) < top_n:
                skipped.append(date.date().isoformat())
            else:
                current_top, current_bottom = assign_top_n(scores, top_n)
                # Overlap guard: a cross-section thinner than 2*N makes top
                # and bottom share members, which would silently inflate the
                # ls diagnostic. Degrade gracefully -- NaN this holding
                # period's ls contribution below, keep lo (the gate series).
                current_ls_valid = len(scores) >= 2 * top_n
                tops.append(set(current_top))
                names_per_date.append(len(scores))
                rebalances.append(
                    (date, tuple(current_top), tuple(current_bottom))
                )
        elif len(scores) < min_names:
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
                rebalances.append(
                    (date, tuple(current_top), tuple(current_bottom))
                )
        if current_top is None or current_bottom is None:
            continue
        hold_end = dates[i + 1] if i + 1 < len(dates) else end
        segment = returns.loc[(returns.index > date) & (returns.index <= hold_end)]
        top_mean = segment[current_top].mean(axis=1)
        bottom_mean = segment[current_bottom].mean(axis=1)
        ls_parts.append(
            top_mean - bottom_mean if current_ls_valid
            else pd.Series(math.nan, index=top_mean.index)
        )
        lo_parts.append(top_mean)

    if not tops:
        floor = top_n if top_n is not None else min_names
        raise SortError(
            f"{spec.name}: every decision date skipped (cross-section < {floor})"
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
        rebalances=tuple(rebalances),
    )
