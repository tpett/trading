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

from trading.alphasearch.panel import PanelData
from trading.alphasearch.sort import SortError
from trading.alphasearch.spec import SIGNALS, SignalSpec
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
