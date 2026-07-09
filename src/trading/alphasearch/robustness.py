"""Piece 3 robustness battery (design spec 2026-07-09): pre-registered,
frozen interrogation of a BH survivor BEFORE it may spend a holdout touch.

Pure composition of existing machinery: evaluate_trial for the re-evaluation
checks (1-4, journaled as `battery`-tagged, BH-counted discovery trials),
portfolio_sort outputs for the arithmetic checks (5-6) and the cost/capacity
series, evaluate_alpha for every re-regression, and spec.amihud_lambda for
the capacity curve's impact prices. Thresholds here are FROZEN (spec section
3); amend only in writing, prospectively.

This module never reads the clock: `ts` always arrives from the CLI.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from trading.alphasearch.evaluate import evaluate_alpha
from trading.alphasearch.panel import PanelData
from trading.alphasearch.sort import Membership, SortError, SortResult
from trading.alphasearch.spec import SIGNALS, SignalSpec, amihud_lambda
from trading.alphasearch.sweep import (
    BH_Q,
    SweepError,
    _bh_survivor_hashes,
    _hashed_params,
    _window_bounds,
    discovery_trials,
    evaluate_trial,
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
COST_BPS = (10, 30, 50)             # one-way, per leg, per rebalance
ELIGIBLE_COST_BPS = 30
ELIGIBLE_MIN_COST_T = 2.0
BOOK_SIZES = (10_000.0, 100_000.0, 1_000_000.0)  # $ per side


@dataclass(frozen=True)
class CheckResult:
    number: int          # spec section 3 row number (1-6)
    name: str
    passed: bool
    detail: dict         # per-check numbers, JSON-safe via log_trial


@dataclass(frozen=True)
class BatteryContext:
    """Everything the check runners share. full_alpha is the DISCOVERY
    trial's journaled L/S four-factor alpha (%/yr) -- the baseline every
    sign/retention rule compares against."""

    journal: Journal
    panel: PanelData
    spec: SignalSpec
    factors: pd.DataFrame
    ts: str
    universe: str
    window: str          # the discovery window being interrogated
    full_alpha: float
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


def _reevaluate(
    ctx: BatteryContext,
    *,
    window: str | None = None,
    quantiles: int | None = None,
    min_names: int | None = None,
    symbol_subset: tuple[str, ...] | None = None,
    calendar_offset: int = 0,
) -> dict:
    """One battery re-evaluation: journaled as a tagged, BH-counted discovery
    trial (config-hash dedupe applies as everywhere) BEFORE its check is
    judged. Errors journal an error trial exactly like run_sweep -- and the
    caller fails the check."""
    q = ctx.quantiles if quantiles is None else quantiles
    mn = ctx.min_names if min_names is None else min_names
    w = ctx.window if window is None else window
    params = _hashed_params(q, ctx.tercile_below, mn,
                            symbol_subset=symbol_subset,
                            calendar_offset=calendar_offset)
    config = trial_config(ctx.spec.name, ctx.universe, w, params=params)
    try:
        result: dict | None = evaluate_trial(
            ctx.panel, ctx.spec, w, ctx.factors,
            quantiles=q, tercile_below=ctx.tercile_below, min_names=mn,
            symbol_subset=symbol_subset, calendar_offset=calendar_offset,
        )
        result["corrupt_cells"] = ctx.panel.corrupt_cells
        error = None
    except (SortError, ValueError, np.linalg.LinAlgError) as exc:
        result = None
        error = f"{type(exc).__name__}: {exc}"
    return log_trial(ctx.journal, kind="discovery", config=config, ts=ctx.ts,
                     result=result, error=error, battery=ctx.tag)


def _alpha_and_t(event: dict) -> tuple[float | None, float | None]:
    ls = event.get("ls") or {}
    return ls.get("alpha_annual_pct"), ls.get("alpha_t")


def check_subperiods(ctx: BatteryContext) -> CheckResult:
    """Check 1 (frozen): both halves -- alpha sign matches the full-window
    sign AND |t| >= 1.0."""
    halves = []
    passed = True
    for w in subperiod_windows(ctx.window):
        event = _reevaluate(ctx, window=w)
        alpha, t = _alpha_and_t(event)
        r = signed_retention(ctx.full_alpha, alpha)
        ok = (
            not math.isnan(r) and r > 0
            and t is not None and abs(float(t)) >= SUBPERIOD_MIN_ABS_T
        )
        passed = passed and ok
        halves.append({"window": w, "alpha_annual_pct": alpha, "alpha_t": t,
                       "error": event.get("error"), "passed": ok})
    return CheckResult(1, "sub_period_halves", passed, {"halves": halves})


def check_subsets(ctx: BatteryContext) -> CheckResult:
    """Check 2 (frozen): 5 seeded half-universe draws; >= 4 of 5 sign-match.
    A draw whose evaluation errors (e.g. half-universe below min_names, or
    missing panel data) FAILS -- the others proceed (spec section 6)."""
    draws = []
    n_pass = 0
    for i in range(SUBSET_DRAWS):
        subset = subset_draw(ctx.panel.symbols, i)
        event = _reevaluate(ctx, symbol_subset=subset)
        alpha, _t = _alpha_and_t(event)
        r = signed_retention(ctx.full_alpha, alpha)
        ok = not math.isnan(r) and r > 0
        n_pass += 1 if ok else 0
        draws.append({"seed": SUBSET_SEED_BASE + i, "n_symbols": len(subset),
                      "alpha_annual_pct": alpha, "error": event.get("error"),
                      "passed": ok})
    return CheckResult(2, "universe_subsets", n_pass >= SUBSET_PASS_MIN,
                       {"draws": draws, "n_pass": n_pass})


def check_jitter(ctx: BatteryContext) -> CheckResult:
    """Check 3 (frozen): quantiles x min_names jitter grid, all 4 sign-match."""
    trials = []
    passed = True
    for q, mn in JITTER_GRID:
        event = _reevaluate(ctx, quantiles=q, min_names=mn)
        alpha, _t = _alpha_and_t(event)
        r = signed_retention(ctx.full_alpha, alpha)
        ok = not math.isnan(r) and r > 0
        passed = passed and ok
        trials.append({"quantiles": q, "min_names": mn,
                       "alpha_annual_pct": alpha,
                       "error": event.get("error"), "passed": ok})
    return CheckResult(3, "parameter_jitter", passed, {"trials": trials})


def check_offset(ctx: BatteryContext) -> CheckResult:
    """Check 4 (frozen): rebalance on the 2nd trading session; sign matches
    AND |alpha| >= 0.5 x full-window |alpha| (the signed-ratio collapse)."""
    event = _reevaluate(ctx, calendar_offset=OFFSET_SESSIONS)
    alpha, t = _alpha_and_t(event)
    r = signed_retention(ctx.full_alpha, alpha)
    ok = not math.isnan(r) and r >= OFFSET_MIN_RETENTION
    return CheckResult(4, "decision_offset", ok, {
        "offset_sessions": OFFSET_SESSIONS, "alpha_annual_pct": alpha,
        "alpha_t": t, "retention": None if math.isnan(r) else r,
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
    """Check 5 (frozen): recompute the L/S excluding the top-3 contributors
    to the top leg; remaining four-factor alpha must retain >= 0.5 of the
    journaled original (signed ratio -- symmetric for negative alphas)."""
    end = _window_bounds(ctx.window)[1]
    contributions = top_leg_contributions(ctx.panel.closes, sort.rebalances, end)
    excluded = list(contributions.index[:NAME_EXCLUDE_TOP])
    alpha: float | None = None
    error: str | None = None
    try:
        reduced = ls_series(ctx.panel.closes, sort.rebalances, end,
                            excluded=frozenset(excluded))
        alpha = evaluate_alpha(reduced, ctx.factors, self_financing=True
                               ).alpha_annual_pct
    except (SortError, ValueError, np.linalg.LinAlgError) as exc:
        error = f"{type(exc).__name__}: {exc}"
    r = signed_retention(ctx.full_alpha, alpha)
    ok = not math.isnan(r) and r >= NAME_MIN_RETENTION
    return CheckResult(5, "name_concentration", ok, {
        "excluded": excluded, "alpha_annual_pct": alpha,
        "retention": None if math.isnan(r) else r, "error": error,
    })


def month_share(ls: pd.Series, top_n: int = MONTH_TOP) -> float:
    """Top-`top_n` calendar months' share of the cumulative L/S log return.
    NaN when the cumulative log return is not positive (concentration of a
    non-gain is undefined; callers fail the check -- an amendment would be
    needed before running the battery on a negative-alpha survivor)."""
    log_returns = np.log1p(ls)
    monthly = log_returns.groupby(ls.index.strftime("%Y-%m")).sum()
    total = float(monthly.sum())
    if not total > 0:
        return math.nan
    return float(monthly.nlargest(top_n).sum()) / total


def check_month_concentration(ls: pd.Series) -> CheckResult:
    """Check 6 (frozen): top-3 months' share of cumulative L/S log return
    <= 60%."""
    share = month_share(ls)
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
# ---------------------------------------------------------------------------#
def apply_rebalance_charges(
    ls: pd.Series, charges: list[tuple[pd.Timestamp, float]]
) -> pd.Series:
    """Deduct each charge from the first daily return strictly after its
    rebalance date (the first day the traded book exists). A charge dated at
    or after the last return day has no day to land on and is dropped (a
    final-day rebalance is never held). Returns a copy."""
    charged = ls.copy()
    for date, charge in charges:
        pos = int(charged.index.searchsorted(date, side="right"))
        if pos < len(charged):
            charged.iloc[pos] -= charge
    return charged


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
    cost c in COST_BPS, EVERY rebalance (formation included) charges
    2 x turnover_monthly x c -- both legs trade -- against the L/S series on
    the first return day after that decision date, and the charged series is
    re-regressed. turnover_monthly is exactly the leaderboard's measurement;
    charging its mean per rebalance totals the same as charging actuals."""
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
