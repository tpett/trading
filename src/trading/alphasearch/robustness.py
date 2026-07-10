"""Piece 3 robustness battery (design spec 2026-07-09): pre-registered,
frozen interrogation of a BH survivor BEFORE it may spend a holdout touch.

Pure composition of existing machinery: evaluate_trial_with_sort for the
re-evaluation checks (1-4, journaled as `battery`-tagged, BH-counted
discovery trials), portfolio_sort outputs for the arithmetic checks (5-6)
and the cost/capacity series, evaluate_alpha for every re-regression, and
spec.amihud_lambda for the capacity curve's impact prices. Thresholds here
are FROZEN (spec section 3); amend only in writing, prospectively. Since
the R1 amendment (2026-07-10, orchestrator-ratified) checks 1-6 score the
COST-CHARGED LO-MINUS-SPY ACTIVE RETURN series -- same frozen thresholds,
re-anchored statistic; see run_battery's docstring.

This module never reads the clock: `ts` always arrives from the CLI.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from trading.alphasearch.costs import (
    DEFAULT_SPY_CACHE_DIR,
    SPY_SYMBOL,
    apply_rebalance_charges,
    cost_charged_lo,
    load_spy_closes,
    spy_benchmark,
)
from trading.alphasearch.evaluate import (
    TRADING_DAYS,
    annualized_sharpe,
    evaluate_alpha,
    total_return,
)
from trading.alphasearch.panel import PanelData
from trading.alphasearch.sort import (
    MIN_NAMES,
    QUANTILES,
    TERCILE_BELOW,
    Membership,
    SortError,
    SortResult,
    portfolio_sort,
)
from trading.alphasearch.spec import SIGNALS, SignalSpec, amihud_lambda
from trading.alphasearch.sweep import (
    BH_Q,
    DISCOVERY_WINDOW,
    SweepError,
    UniverseSpec,
    _bh_survivor_hashes,
    _check_factor_coverage,
    _check_universe_supports,
    _hashed_params,
    _window_bounds,
    build_universe_panel,
    discovery_trials,
    evaluate_trial_with_sort,
    find_discovery_trial,
    log_trial,
    trial_config,
)
from trading.journal import Journal

# ---------------------------------------------------------------------------#
# Frozen battery parameters (spec section 3 / section 4). Amend only in
# writing, prospectively -- these are pre-registered science, not tunables.
# ---------------------------------------------------------------------------#
SUBPERIOD_MIN_ABS_T = 1.0
SUBSET_DRAWS = 5
SUBSET_SEED_BASE = 42
SUBSET_PASS_MIN = 4
JITTER_GRID = ((4, 10), (4, 20), (6, 10), (6, 20))  # (quantiles, min_names)
OFFSET_SESSIONS = 1                 # rebalance on the 2nd trading session
OFFSET_MIN_RETENTION = 0.5
NAME_EXCLUDE_TOP = 3
NAME_MIN_RETENTION = 0.5
MONTH_TOP = 3
MONTH_MAX_SHARE = 0.60
PROXY_LOADING_MULTIPLE = 2.0
PROXY_MIN_R2 = 0.5
COST_BPS = (10, 30, 50)             # one-way, per leg, per rebalance -- DIAGNOSTIC
# ONLY (spec 2026-07-10 R1 amendment): the promotion rule no longer reads a
# row from this table. Retained because the L/S cost-adjusted alpha curve is
# still a useful fragility diagnostic in the report card.
BOOK_SIZES = (10_000.0, 100_000.0, 1_000_000.0)  # $ per side


@dataclass(frozen=True)
class CheckResult:
    number: int          # spec section 3 row number (1-6)
    name: str
    passed: bool
    detail: dict         # per-check numbers, JSON-safe via log_trial


@dataclass(frozen=True)
class BatteryContext:
    """Everything the check runners share. full_active is the full-window
    annualized cost-charged LO-minus-SPY active return (%/yr) -- the baseline
    every sign/retention rule compares against (R1 amendment, 2026-07-10:
    checks re-anchored from the journaled L/S alpha to the active series;
    same frozen thresholds)."""

    journal: Journal
    panel: PanelData
    spec: SignalSpec
    factors: pd.DataFrame
    ts: str
    universe: str
    window: str          # the discovery window being interrogated
    full_active: float   # annualized active return (%/yr), the baseline
    spy_closes: pd.Series
    quantiles: int
    tercile_below: int
    min_names: int
    tag: str             # "signal:universe" -- the battery grouping tag


def subperiod_windows(window: str) -> tuple[str, str]:
    """Split a window at its calendar midpoint (floor). For the production
    discovery window this reproduces the spec section 3 literals exactly --
    pinned by test_subperiod_windows_pin_the_frozen_discovery_split."""
    start, end = _window_bounds(window)
    mid = start + pd.Timedelta((end - start).days // 2, unit="D")
    first_end = mid - pd.Timedelta(1, unit="D")
    return (
        f"{start.date().isoformat()}..{first_end.date().isoformat()}",
        f"{mid.date().isoformat()}..{end.date().isoformat()}",
    )


def subset_draw(symbols: tuple[str, ...], i: int) -> tuple[str, ...]:
    """Half-universe draw i, seed = SUBSET_SEED_BASE + i (frozen). Sorted
    input AND sorted output: the draw is a set, reproducible regardless of
    dict/tuple ordering upstream."""
    universe = sorted(symbols)
    rng = np.random.default_rng(SUBSET_SEED_BASE + i)
    picked = rng.choice(np.array(universe, dtype=object),
                        size=len(universe) // 2, replace=False)
    return tuple(sorted(str(s) for s in picked))


def signed_retention(original: float | None, perturbed: float | None) -> float:
    """perturbed / original: > 0 iff the signs match, >= r iff the magnitude
    retains an r fraction WITH the matching sign (holdout_passes' collapse,
    symmetric for negative-alpha candidates). NaN when either side is
    missing/NaN or original == 0 -- callers treat NaN as a FAIL (an
    uncomputable perturbation is not a pass, spec section 6)."""
    if original is None or perturbed is None:
        return math.nan
    original, perturbed = float(original), float(perturbed)
    if original == 0 or math.isnan(original) or math.isnan(perturbed):
        return math.nan
    return perturbed / original


# ---------------------------------------------------------------------------#
# The active series (R1 amendment, 2026-07-10, orchestrator-ratified): checks
# 1-6 are re-anchored to the COST-CHARGED LO-MINUS-SPY ACTIVE RETURN series,
# not the raw lo series -- raw-LO retention would mostly test whether the
# market regime repeated (beta), not whether the signal's edge over the
# benchmark did. Same frozen thresholds throughout.
# ---------------------------------------------------------------------------#
def active_series(charged_lo: pd.Series, spy_closes: pd.Series) -> pd.Series:
    """Daily cost-charged long-only return minus SPY's daily return, on the
    inner join of their calendars (a day either side lacks contributes
    nothing -- never a fabricated 0)."""
    spy_rets = spy_closes.pct_change().dropna()
    lo, spy = charged_lo.align(spy_rets, join="inner")
    return (lo - spy).dropna()


def annualized_active_pct(active: pd.Series) -> float:
    """Annualized active return in %/yr (mean daily active x 252 x 100) --
    the checks' retention/sign statistic. NaN for an empty series."""
    a = active.dropna()
    if len(a) == 0:
        return math.nan
    return float(a.mean()) * TRADING_DAYS * 100.0


def active_t(active: pd.Series) -> float:
    """t-statistic of the active series' mean (mean / (sd/sqrt(n))) -- check
    1's |t| >= 1.0 statistic, computed on the series' own mean/se. NaN below
    2 observations or at zero variance."""
    a = active.dropna()
    if len(a) < 2:
        return math.nan
    sd = float(a.std(ddof=1))
    if sd == 0:
        return math.nan
    return float(a.mean()) / (sd / math.sqrt(len(a)))


def _reevaluate(
    ctx: BatteryContext,
    *,
    window: str | None = None,
    quantiles: int | None = None,
    min_names: int | None = None,
    symbol_subset: tuple[str, ...] | None = None,
    calendar_offset: int = 0,
) -> tuple[dict, SortResult | None]:
    """One battery re-evaluation: journaled as a tagged, BH-counted discovery
    trial (config-hash dedupe applies as everywhere) BEFORE its check is
    judged. Errors journal an error trial exactly like run_sweep -- and the
    caller fails the check. Returns (event, sort): the sort (None on error)
    carries the raw lo series/memberships the re-anchored checks score."""
    q = ctx.quantiles if quantiles is None else quantiles
    mn = ctx.min_names if min_names is None else min_names
    w = ctx.window if window is None else window
    params = _hashed_params(q, ctx.tercile_below, mn,
                            symbol_subset=symbol_subset,
                            calendar_offset=calendar_offset)
    config = trial_config(ctx.spec.name, ctx.universe, w, params=params)
    sort: SortResult | None = None
    try:
        result, sort = evaluate_trial_with_sort(
            ctx.panel, ctx.spec, w, ctx.factors,
            quantiles=q, tercile_below=ctx.tercile_below, min_names=mn,
            symbol_subset=symbol_subset, calendar_offset=calendar_offset,
        )
        result["corrupt_cells"] = ctx.panel.corrupt_cells
        error = None
    except (SortError, ValueError, np.linalg.LinAlgError) as exc:
        result = None
        error = f"{type(exc).__name__}: {exc}"
    event = log_trial(ctx.journal, kind="discovery", config=config, ts=ctx.ts,
                      result=result, error=error, battery=ctx.tag)
    return event, sort


def _active_stats(ctx: BatteryContext, sort: SortResult) -> tuple[float, float]:
    """(annualized active %/yr, active-mean t) of one evaluation's cost-
    charged LO-minus-SPY series -- the re-anchored checks' statistics."""
    charged, _skipped = cost_charged_lo(ctx.panel, sort.lo, sort.rebalances)
    active = active_series(charged, ctx.spy_closes)
    return annualized_active_pct(active), active_t(active)


def _nan_none(value: float) -> float | None:
    return None if math.isnan(value) else value


def check_subperiods(ctx: BatteryContext) -> CheckResult:
    """Check 1 (frozen threshold; R1 re-anchor): both halves -- the active
    return's sign matches the full-window active sign AND the active series'
    |t| >= 1.0 (mean/se)."""
    halves = []
    passed = True
    for w in subperiod_windows(ctx.window):
        event, sort = _reevaluate(ctx, window=w)
        act, t = (math.nan, math.nan) if sort is None else _active_stats(ctx, sort)
        r = signed_retention(ctx.full_active, act)
        ok = (
            not math.isnan(r) and r > 0
            and not math.isnan(t) and abs(t) >= SUBPERIOD_MIN_ABS_T
        )
        passed = passed and ok
        halves.append({"window": w, "active_annual_pct": _nan_none(act),
                       "active_t": _nan_none(t),
                       "error": event.get("error"), "passed": ok})
    return CheckResult(1, "sub_period_halves", passed, {"halves": halves})


def check_subsets(ctx: BatteryContext) -> CheckResult:
    """Check 2 (frozen threshold; R1 re-anchor): 5 seeded half-universe
    draws; >= 4 of 5 active-return sign-match. A draw whose evaluation errors
    (e.g. half-universe below min_names, or missing panel data) FAILS -- the
    others proceed (spec section 6)."""
    draws = []
    n_pass = 0
    for i in range(SUBSET_DRAWS):
        subset = subset_draw(ctx.panel.symbols, i)
        event, sort = _reevaluate(ctx, symbol_subset=subset)
        act = math.nan if sort is None else _active_stats(ctx, sort)[0]
        r = signed_retention(ctx.full_active, act)
        ok = not math.isnan(r) and r > 0
        n_pass += 1 if ok else 0
        draws.append({"seed": SUBSET_SEED_BASE + i, "n_symbols": len(subset),
                      "active_annual_pct": _nan_none(act),
                      "error": event.get("error"), "passed": ok})
    return CheckResult(2, "universe_subsets", n_pass >= SUBSET_PASS_MIN,
                       {"draws": draws, "n_pass": n_pass})


def check_jitter(ctx: BatteryContext) -> CheckResult:
    """Check 3 (frozen threshold; R1 re-anchor): quantiles x min_names jitter
    grid, all 4 active-return sign-match."""
    trials = []
    passed = True
    for q, mn in JITTER_GRID:
        event, sort = _reevaluate(ctx, quantiles=q, min_names=mn)
        act = math.nan if sort is None else _active_stats(ctx, sort)[0]
        r = signed_retention(ctx.full_active, act)
        ok = not math.isnan(r) and r > 0
        passed = passed and ok
        trials.append({"quantiles": q, "min_names": mn,
                       "active_annual_pct": _nan_none(act),
                       "error": event.get("error"), "passed": ok})
    return CheckResult(3, "parameter_jitter", passed, {"trials": trials})


def check_offset(ctx: BatteryContext) -> CheckResult:
    """Check 4 (frozen threshold; R1 re-anchor): rebalance on the 2nd trading
    session; active sign matches AND |active| >= 0.5 x full-window |active|
    (the signed-ratio collapse)."""
    event, sort = _reevaluate(ctx, calendar_offset=OFFSET_SESSIONS)
    act, t = (math.nan, math.nan) if sort is None else _active_stats(ctx, sort)
    r = signed_retention(ctx.full_active, act)
    ok = not math.isnan(r) and r >= OFFSET_MIN_RETENTION
    return CheckResult(4, "decision_offset", ok, {
        "offset_sessions": OFFSET_SESSIONS, "active_annual_pct": _nan_none(act),
        "active_t": _nan_none(t), "retention": None if math.isnan(r) else r,
        "error": event.get("error"),
    })


def require_survivor(
    journal: Journal, signal_name: str, universe: str, window: str, params: dict
) -> dict:
    """The battery's admission gate (mirrors the holdout gate; refusal
    journals NOTHING). Returns the clean discovery trial being interrogated.
    Refusals: unknown signal; no clean same-params discovery trial with a
    usable L/S alpha; that EXACT trial (by config hash) not a current BH
    survivor -- the refusal lists the current survivors."""
    if signal_name not in SIGNALS:
        known = ", ".join(sorted(SIGNALS))
        raise SweepError(f"unknown signal {signal_name!r}; known: {known}")
    discovery = find_discovery_trial(journal, signal_name, universe, window,
                                     params=params)
    if discovery is None:
        raise SweepError(
            f"no discovery trial for {signal_name}:{universe} over {window} "
            f"with matching params; run the sweep first"
        )
    if discovery.get("error"):
        raise SweepError(
            f"discovery trial for {signal_name}:{universe} errored "
            f"({discovery['error']}); nothing to interrogate"
        )
    if (discovery.get("ls") or {}).get("alpha_annual_pct") is None:
        raise SweepError(
            f"discovery trial for {signal_name}:{universe} has no usable "
            f"L/S alpha (journaled as null); nothing to interrogate"
        )
    survivors = _bh_survivor_hashes(journal)
    if discovery["config_hash"] not in survivors:
        current = sorted({
            f"{t['signal']}:{t['universe']}"
            for t in discovery_trials(journal)
            if t["config_hash"] in survivors
        })
        listing = ", ".join(current) if len(current) > 0 else "none"
        raise SweepError(
            f"{signal_name}:{universe} is not a current BH survivor "
            f"(q={BH_Q}); the battery is reserved for survivors. "
            f"Current survivors: {listing}"
        )
    return discovery


# ---------------------------------------------------------------------------#
# Checks 5-7: arithmetic on already-computed series. NO new trials journaled.
# ---------------------------------------------------------------------------#
def _segments(rebalances: tuple[Membership, ...], end: pd.Timestamp):
    """(date, top, bottom, hold_end): each membership is held to the next
    ACTUAL rebalance (or the window end) -- skipped decision dates never
    truncate a holding period, matching portfolio_sort's hold-through rule."""
    for j, (date, top, bottom) in enumerate(rebalances):
        hold_end = rebalances[j + 1][0] if j + 1 < len(rebalances) else end
        yield date, top, bottom, hold_end


def ls_series(
    closes: dict[str, pd.Series],
    rebalances: tuple[Membership, ...],
    end: pd.Timestamp,
    *,
    excluded: frozenset[str] = frozenset(),
) -> pd.Series:
    """Replay the daily L/S spread from recorded memberships (bit-identical
    to portfolio_sort's ls -- proven by test), optionally excluding names
    from BOTH legs. A rebalance segment whose top or bottom leg is emptied
    by the exclusion has no defined spread and is dropped; if every segment
    empties, SortError (the caller fails the check)."""
    returns = pd.DataFrame({s: c.pct_change() for s, c in closes.items()})
    parts: list[pd.Series] = []
    for date, top, bottom, hold_end in _segments(rebalances, end):
        top_kept = [s for s in top if s not in excluded]
        bottom_kept = [s for s in bottom if s not in excluded]
        if len(top_kept) == 0 or len(bottom_kept) == 0:
            continue
        segment = returns.loc[(returns.index > date) & (returns.index <= hold_end)]
        parts.append(
            segment[top_kept].mean(axis=1) - segment[bottom_kept].mean(axis=1)
        )
    if len(parts) == 0:
        raise SortError("every rebalance segment emptied by the exclusion")
    return pd.concat(parts).dropna()


def lo_series(
    closes: dict[str, pd.Series],
    rebalances: tuple[Membership, ...],
    end: pd.Timestamp,
    *,
    excluded: frozenset[str] = frozenset(),
) -> pd.Series:
    """Replay the daily long-only (top-leg) series from recorded memberships
    -- ls_series' top-leg-only analogue, for the R1 re-anchored check 5.
    Bit-identical to portfolio_sort's lo with no exclusion (same segment
    boundaries and skipna mean). A segment whose top leg is emptied by the
    exclusion is dropped; if every segment empties, SortError (the caller
    fails the check)."""
    returns = pd.DataFrame({s: c.pct_change() for s, c in closes.items()})
    parts: list[pd.Series] = []
    for date, top, _bottom, hold_end in _segments(rebalances, end):
        top_kept = [s for s in top if s not in excluded]
        if len(top_kept) == 0:
            continue
        segment = returns.loc[(returns.index > date) & (returns.index <= hold_end)]
        parts.append(segment[top_kept].mean(axis=1))
    if len(parts) == 0:
        raise SortError("every rebalance segment emptied by the exclusion")
    return pd.concat(parts).dropna()


def top_leg_contributions(
    closes: dict[str, pd.Series],
    rebalances: tuple[Membership, ...],
    end: pd.Timestamp,
) -> pd.Series:
    """Cumulative per-name contribution to the TOP-quantile leg's daily
    return: on each held day a member contributes ret / n_live, where n_live
    counts the members with a return that day -- the same skipna denominator
    as the leg's mean(axis=1), so the contributions sum to the lo series by
    construction (a fixed 1/n_top would under-credit survivors of a
    mid-holding delisting and mis-rank check 5's exclusions).
    Descending order -- index[:3] are check 5's exclusion candidates."""
    returns = pd.DataFrame({s: c.pct_change() for s, c in closes.items()})
    totals: dict[str, float] = {}
    for date, top, _bottom, hold_end in _segments(rebalances, end):
        segment = returns.loc[(returns.index > date) & (returns.index <= hold_end)]
        leg = segment[list(top)]
        weights = 1.0 / leg.notna().sum(axis=1)
        per_name = leg.mul(weights, axis=0).sum()
        for sym, value in per_name.items():
            totals[sym] = totals.get(sym, 0.0) + float(value)
    return pd.Series(totals, dtype="float64").sort_values(ascending=False)


def check_name_concentration(ctx: BatteryContext, sort: SortResult) -> CheckResult:
    """Check 5 (frozen threshold; R1 re-anchor): replay the long-only series
    excluding the top-3 contributors to the top leg, re-charge spread costs
    on the reduced memberships, and the remaining LO-minus-SPY active return
    must retain >= 0.5 of the full-window active baseline (signed ratio --
    symmetric for negative-active candidates)."""
    end = _window_bounds(ctx.window)[1]
    contributions = top_leg_contributions(ctx.panel.closes, sort.rebalances, end)
    excluded = list(contributions.index[:NAME_EXCLUDE_TOP])
    act: float | None = None
    error: str | None = None
    try:
        reduced_lo = lo_series(ctx.panel.closes, sort.rebalances, end,
                               excluded=frozenset(excluded))
        # Costs re-charged on the REDUCED memberships (the book actually
        # held); rebalances whose top leg empties carry no book to charge.
        reduced_rebalances = tuple(
            (date, kept, bottom)
            for date, top, bottom in sort.rebalances
            if (kept := tuple(s for s in top if s not in excluded))
        )
        charged, _skipped = cost_charged_lo(ctx.panel, reduced_lo,
                                            reduced_rebalances)
        act = annualized_active_pct(active_series(charged, ctx.spy_closes))
    except (SortError, ValueError, np.linalg.LinAlgError) as exc:
        error = f"{type(exc).__name__}: {exc}"
    r = signed_retention(ctx.full_active, act)
    ok = not math.isnan(r) and r >= NAME_MIN_RETENTION
    return CheckResult(5, "name_concentration", ok, {
        "excluded": excluded,
        "active_annual_pct": None if act is None else _nan_none(act),
        "retention": None if math.isnan(r) else r, "error": error,
    })


def month_share(returns: pd.Series, top_n: int = MONTH_TOP) -> float:
    """Top-`top_n` calendar months' share of the series' cumulative log
    return. NaN when the cumulative log return is not positive (concentration
    of a non-gain is undefined; callers fail the check -- an amendment would
    be needed before running the battery on a negative-active survivor)."""
    log_returns = np.log1p(returns)
    monthly = log_returns.groupby(returns.index.strftime("%Y-%m")).sum()
    total = float(monthly.sum())
    if not total > 0:
        return math.nan
    return float(monthly.nlargest(top_n).sum()) / total


def check_month_concentration(active: pd.Series) -> CheckResult:
    """Check 6 (frozen threshold; R1 re-anchor): top-3 months' share of the
    cost-charged LO-minus-SPY active series' cumulative log return <= 60%."""
    share = month_share(active)
    ok = not math.isnan(share) and share <= MONTH_MAX_SHARE
    return CheckResult(6, "month_concentration", ok, {
        "top3_share": None if math.isnan(share) else share,
    })


def factor_proxy_flag(ls_stats: dict) -> dict:
    """Check 7 (frozen; WARNING only, never blocks): any factor loading with
    |t_loading| > 2 x |t_alpha| while R^2 > 0.5 -- the section 9 SMB-costume
    detector. Input is the discovery trial's journaled `ls` block."""
    alpha_t = ls_stats.get("alpha_t")
    r2 = ls_stats.get("r2")
    loadings_t = ls_stats.get("loadings_t") or {}
    offenders: dict[str, float] = {}
    if alpha_t is not None and r2 is not None and float(r2) > PROXY_MIN_R2:
        for name, t in loadings_t.items():
            if t is not None and abs(float(t)) > (
                PROXY_LOADING_MULTIPLE * abs(float(alpha_t))
            ):
                offenders[name] = float(t)
    return {"flagged": len(offenders) > 0, "offenders": offenders,
            "alpha_t": alpha_t, "r2": r2}


# ---------------------------------------------------------------------------#
# Cost & capacity analysis (spec section 4): arithmetic, NO new trials.
# apply_rebalance_charges now lives in costs.py (a leaf module both this file
# and sweep.py's --long-only leaderboard can import without a cycle) and is
# re-exported here under its original name -- `from
# trading.alphasearch.robustness import apply_rebalance_charges` still works.
# ---------------------------------------------------------------------------#
def _regress_charged(charged: pd.Series, factors: pd.DataFrame, row: dict) -> dict:
    try:
        alpha = evaluate_alpha(charged, factors, self_financing=True)
        row["alpha_annual_pct"] = alpha.alpha_annual_pct
        row["alpha_t"] = alpha.alpha_tstat
    except (ValueError, np.linalg.LinAlgError) as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
    return row


def cost_adjusted_table(
    ls: pd.Series,
    rebalances: tuple[Membership, ...],
    turnover_monthly: float,
    factors: pd.DataFrame,
) -> list[dict]:
    """Cost-adjusted alpha (spec section 4, frozen reading): for each one-way
    cost c in COST_BPS, EVERY rebalance -- formation included -- charges
    2 x turnover_monthly x c -- both legs trade -- against the L/S series on
    the first return day after that decision date, and the charged series is
    re-regressed. Deliberate simplification, not an equality: formation is
    really a ~100% turnover event (every position is new, from cash), but it
    is charged at the SAME mean turnover_monthly as every later rebalance
    rather than at 1.0 -- a one-time understatement of the first charge,
    traded for one consistent per-rebalance formula instead of a
    special-cased first row."""
    rows: list[dict] = []
    dates = [date for date, _top, _bottom in rebalances]
    usable = turnover_monthly is not None and not math.isnan(turnover_monthly)
    for bps in COST_BPS:
        row: dict = {"cost_bps": bps, "alpha_annual_pct": None, "alpha_t": None}
        if usable:
            per_rebalance = 2.0 * float(turnover_monthly) * bps / 1e4
            charged = apply_rebalance_charges(
                ls, [(d, per_rebalance) for d in dates]
            )
            row = _regress_charged(charged, factors, row)
        else:
            row["error"] = "turnover unavailable (single rebalance?)"
        rows.append(row)
    return rows


def capacity_curve(
    panel: PanelData,
    rebalances: tuple[Membership, ...],
    ls: pd.Series,
    factors: pd.DataFrame,
) -> list[dict]:
    """Amihud-implied capacity (spec section 4, frozen reading): for each
    book size B per side, every name ENTERING a leg is charged its own
    lambda x (B / n_new) impact on its 1/n_new-weight position -- a
    leg-return drag of lambda x B / n_new^2 -- and every EXITING name the
    analogue at the OLD leg's size (the position actually liquidated).
    lambda = spec.amihud_lambda(view.bars(sym)) at the rebalance date (PIT).
    NaN-lambda names are skipped and counted, never fabricated; final
    holdings are never charged an exit. This is a FIRST-ORDER impact model
    -- honest about being a model; for an illiquidity signal the names' own
    lambda is the most self-consistent EOD impact estimate available."""
    unit_charges: list[tuple[pd.Timestamp, float]] = []  # per $1 of book
    skipped = 0
    prev: tuple[tuple[str, ...], tuple[str, ...]] | None = None
    for date, top, bottom in rebalances:
        view = panel.view(date)
        unit = 0.0
        for leg_index, current in enumerate((top, bottom)):
            previous = prev[leg_index] if prev is not None else ()
            cur_set, prev_set = set(current), set(previous)
            for sym in sorted(cur_set - prev_set):        # entries
                lam = amihud_lambda(view.bars(sym))
                if math.isnan(lam):
                    skipped += 1
                    continue
                unit += lam / (len(current) * len(current))
            for sym in sorted(prev_set - cur_set):        # exits
                lam = amihud_lambda(view.bars(sym))
                if math.isnan(lam):
                    skipped += 1
                    continue
                unit += lam / (len(previous) * len(previous))
        unit_charges.append((date, unit))
        prev = (top, bottom)
    rows: list[dict] = []
    for book in BOOK_SIZES:
        row: dict = {
            "book_usd": book, "alpha_annual_pct": None, "alpha_t": None,
            "total_impact_charge": sum(u for _d, u in unit_charges) * book,
            "skipped_no_lambda": skipped,
        }
        charged = apply_rebalance_charges(
            ls, [(d, u * book) for d, u in unit_charges]
        )
        rows.append(_regress_charged(charged, factors, row))
    return rows


# ---------------------------------------------------------------------------#
# The battery runner: refusal gate -> checks 1-4 (journaled re-evaluations)
# -> checks 5-7 + cost/capacity (arithmetic) -> journaled verdict.
# ---------------------------------------------------------------------------#
@dataclass(frozen=True)
class BatteryOutcome:
    signal: str
    universe: str
    window: str
    checks: tuple[CheckResult, ...]   # checks 1-6, spec order
    factor_proxy: dict                # check 7: warning only, never blocks
    cost_table: list[dict]            # L/S cost-adjusted alpha: DIAGNOSTIC only
    capacity_curve: list[dict]
    long_only_gate: dict              # R1 amendment: the promotion comparator
    eligible: bool                    # the amended promotion rule (spec section 2)
    event: dict                       # the journaled kind="battery" verdict


def run_battery(
    uspec: UniverseSpec,
    journal: Journal,
    factors: pd.DataFrame,
    ts: str,
    signal_name: str,
    *,
    discovery_window: str = DISCOVERY_WINDOW,
    quantiles: int = QUANTILES,
    tercile_below: int = TERCILE_BELOW,
    min_names: int = MIN_NAMES,
    panel_factory: Callable[[UniverseSpec, pd.DataFrame | None], PanelData] = (
        build_universe_panel
    ),
    spy_closes: pd.Series | None = None,
) -> BatteryOutcome:
    """Run the frozen battery on ONE current BH survivor (spec section 3).

    Refuses non-survivors BEFORE journaling anything (require_survivor).
    Also refuses, before journaling anything, if the recomputed full-window
    alpha has drifted from the journaled discovery baseline: data caches are
    gitignored/mutable, so a stale or edited cache would otherwise mix a
    fresh numerator into every retention denominator below without ever
    raising SortError. Checks 1-4 journal battery-tagged, BH-counted
    discovery trials; checks 5-7 and the cost/capacity analysis are
    arithmetic on a locally recomputed full-window sort (identical config to
    the journaled discovery trial -- deliberately NOT journaled again). The
    verdict is one kind="battery" event per (signal, universe); re-runs
    replace by config hash.

    The promotion rule (2026-07-10 R1 amendment, spec section 2 -- amends the
    prior "checks 1-6 pass AND 30bps L/S cost t >= 2.0" rule): eligible iff
    checks 1-6 all pass AND the cost-charged long-only series' Sharpe over
    the discovery window is >= SPY buy-and-hold's Sharpe over the identical
    window AND its total return exceeds SPY's. Checks 1-6 keep their frozen
    THRESHOLDS but re-anchor their statistics (orchestrator-ratified reading
    of the spec's re-anchoring clause) onto the cost-charged LO-MINUS-SPY
    ACTIVE RETURN series -- not the raw lo series, whose retention would
    mostly test whether the market regime repeated (beta) rather than the
    signal's edge over the benchmark. `spy_closes` lets callers (tests, or a
    future non-largecap benchmark) inject the series directly; the default
    loads data/equities-tiingo/SPY.parquet and refuses loudly (no silent
    substitute) if that cache is absent. Never touches holdout state."""
    params = _hashed_params(quantiles, tercile_below, min_names)
    discovery = require_survivor(journal, signal_name, uspec.name,
                                 discovery_window, params)
    full_alpha = float(discovery["ls"]["alpha_annual_pct"])
    start, end = _window_bounds(discovery_window)
    # Stale factors refuse HERE, before the first re-evaluation journals:
    # letting evaluate_trial hit the coverage check inside the loop would
    # journal 12 predictable error trials for one fixable cache problem.
    try:
        _check_factor_coverage(factors, end)
    except ValueError as exc:
        raise SweepError(str(exc)) from exc
    # R1 amendment: SPY is the frozen promotion comparator (spec section 2).
    # Refuse pre-touch, same shape as the stale-factors check above, rather
    # than silently substituting another benchmark or crashing mid-battery.
    spy = spy_closes if spy_closes is not None else load_spy_closes(DEFAULT_SPY_CACHE_DIR)
    if spy is None:
        raise SweepError(
            f"no {SPY_SYMBOL} cache at "
            f"{DEFAULT_SPY_CACHE_DIR / f'{SPY_SYMBOL}.parquet'}; the long-only "
            "gate benchmarks every candidate against SPY buy-and-hold and "
            "refuses to substitute another proxy. Fetch SPY into that cache "
            "(same Tiingo bar-cache pipeline as the other largecap symbols) "
            "before running the battery"
        )
    panel = panel_factory(uspec, factors)
    spec = SIGNALS[signal_name]
    _check_universe_supports(panel, spec, uspec.name)
    # Full-window sort recomputed ONCE, reused for checks 5-6, cost/capacity,
    # and the active-return baseline -- same config as the journaled discovery
    # trial, so journaling it again would only append a duplicate; spec:
    # checks 5-7 journal no new trials. Computed HERE, before checks 1-4
    # journal a single battery trial, so the drift guard right below can
    # refuse pre-touch.
    try:
        sort = portfolio_sort(
            panel, spec, panel.decision_dates(start, end), end,
            quantiles=quantiles, tercile_below=tercile_below,
            min_names=min_names,
        )
    except SortError as exc:
        # The journaled discovery trial was clean, so current caches must
        # have drifted from the evidence. Refuse loudly; journal nothing new.
        raise SweepError(
            f"full-window sort failed against current data ({exc}) although "
            f"the journaled discovery trial is clean; re-run the sweep before "
            f"the battery"
        ) from exc
    # Data caches are gitignored/mutable (unlike the journal), so they can
    # drift between the discovery sweep and this battery run WITHOUT the
    # sort above ever raising -- e.g. one appended or edited bar -- silently
    # mixing a fresh numerator into every retention denominator below, which
    # is anchored to the journaled `full_alpha` baseline. Re-derive that same
    # alpha from the sort just computed and refuse, before checks 1-4
    # journal a single battery trial (a hash-replace re-run would otherwise
    # be a free re-roll channel for marginal checks), if it has moved beyond
    # floating-point noise: identical data must reproduce to within rel
    # 1e-6; a real drift is orders of magnitude larger.
    recomputed_alpha = evaluate_alpha(
        sort.ls, factors, self_financing=True
    ).alpha_annual_pct
    if not math.isclose(recomputed_alpha, full_alpha, rel_tol=1e-6):
        raise SweepError(
            "caches drifted since the discovery sweep; re-run the sweep for "
            "this trial before running its battery"
        )
    # R1 re-anchor (spec section 2, orchestrator-ratified 2026-07-10): every
    # check's retention/sign baseline is the full-window cost-charged
    # LO-minus-SPY active return -- computed once here, shared with the
    # promotion comparator below.
    charged_lo, skipped_no_spread = cost_charged_lo(panel, sort.lo, sort.rebalances)
    active_full = active_series(charged_lo, spy)
    full_active = annualized_active_pct(active_full)
    ctx = BatteryContext(
        journal=journal, panel=panel, spec=spec, factors=factors, ts=ts,
        universe=uspec.name, window=discovery_window, full_active=full_active,
        spy_closes=spy,
        quantiles=quantiles, tercile_below=tercile_below, min_names=min_names,
        tag=f"{signal_name}:{uspec.name}",
    )
    checks = [check_subperiods(ctx), check_subsets(ctx), check_jitter(ctx),
              check_offset(ctx)]
    checks.append(check_name_concentration(ctx, sort))
    checks.append(check_month_concentration(active_full))
    proxy = factor_proxy_flag(discovery.get("ls") or {})
    cost_rows = cost_adjusted_table(sort.ls, sort.rebalances,
                                    sort.turnover_monthly, factors)
    capacity_rows = capacity_curve(panel, sort.rebalances, sort.ls, factors)
    # R1 amendment (spec section 2): the promotion comparator is the
    # spread-charged long-only series vs SPY buy-and-hold, both over the
    # discovery window -- replacing the 30bps L/S cost-table row above.
    # Fix (final review, 2026-07-10): if the window's LEADING decision
    # date(s) are skipped (below min_names, or the degenerate-score guard),
    # charged_lo's first observation lands on the first ACTUAL rebalance,
    # not the nominal window start -- comparing its total return to SPY's
    # total return over the FULL nominal window would compound different
    # horizons (anti-conservative when the market fell during the skipped
    # lead-in: SPY's full-window total would be dragged down by a decline
    # the signal never had to sit through). Anchor SPY's benchmark window to
    # the first ACTUAL DECISION date, not charged_lo.index[0]: portfolio_sort
    # builds hold segments strictly after the decision date (sort.py), so
    # charged_lo.index[0] is the first REALIZED RETURN day and already
    # compounds the move from the decision date to that day -- one trading
    # day later than SPY needs to match the identical economic horizon.
    # sort.rebalances is guaranteed non-empty here: SortError above already
    # refused an empty `tops`, which populates in lockstep with rebalances.
    lo_window_start = sort.rebalances[0][0]
    lo_sharpe = annualized_sharpe(charged_lo)
    lo_total = total_return(charged_lo)
    spy_stats = spy_benchmark(spy, lo_window_start, end)
    long_only_gate_passed = (
        not math.isnan(lo_sharpe) and not math.isnan(spy_stats.sharpe_annual)
        and lo_sharpe >= spy_stats.sharpe_annual
        and not math.isnan(lo_total) and not math.isnan(spy_stats.total_return)
        and lo_total > spy_stats.total_return
    )
    long_only_gate = {
        "lo_sharpe": lo_sharpe,
        "lo_total_return": lo_total,
        "spy_sharpe": spy_stats.sharpe_annual,
        "spy_total_return": spy_stats.total_return,
        "active_annual_pct": _nan_none(full_active),  # the checks' baseline
        "skipped_no_spread": skipped_no_spread,
        "passed": long_only_gate_passed,
    }
    eligible = all(c.passed for c in checks) and long_only_gate_passed
    config = trial_config(signal_name, uspec.name, discovery_window,
                          params=params)
    verdict = {
        "checks": {
            c.name: {"number": c.number, "passed": c.passed, **c.detail}
            for c in checks
        },
        "factor_proxy": proxy,
        "cost_table": cost_rows,
        "capacity_curve": capacity_rows,
        "long_only_gate": long_only_gate,
        "eligible": eligible,
    }
    event = log_trial(journal, kind="battery", config=config, ts=ts,
                      result=verdict)
    return BatteryOutcome(
        signal=signal_name, universe=uspec.name, window=discovery_window,
        checks=tuple(checks), factor_proxy=proxy, cost_table=cost_rows,
        capacity_curve=capacity_rows, long_only_gate=long_only_gate,
        eligible=eligible, event=event,
    )
