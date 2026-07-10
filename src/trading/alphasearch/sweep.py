"""Sweep runner, trial journal, leaderboard, holdout re-prove (spec 3.6 + 4).

The trial journal (journal/alphasearch-trials.jsonl, via trading.journal.
Journal) is the program's scientific ledger: EVERY evaluation -- success or
error -- is appended BEFORE any leaderboard is computed, and the BH-FDR /
DSR trial count is derived from this file alone. It is append-only and
committed to git; deleting or editing it invalidates the statistics.

Idempotency: an identical config re-run APPENDS a new event (append-only is
never violated) and every reader deduplicates via load_trials(), keeping the
LATEST event per (config_hash, kind) -- logical update-in-place, physical
append-only, and re-runs never inflate the trial count. Any changed parameter
changes the hash and honestly counts as a NEW trial (spec 5.6).

This module never reads the clock: `ts` always arrives from the CLI.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from trading.alphasearch import stats
from trading.alphasearch.evaluate import AlphaResult, evaluate_alpha
from trading.alphasearch.panel import PanelData, PanelError, build_panel
from trading.alphasearch.sort import (
    MIN_NAMES,
    QUANTILES,
    TERCILE_BELOW,
    SortError,
    portfolio_sort,
)
from trading.alphasearch.spec import SIGNALS, SignalSpec
from trading.journal import Journal

DISCOVERY_WINDOW = "2019-01-01..2023-12-31"   # pre-registered (spec 5.1)
HOLDOUT_START = "2024-01-01"                  # pre-registered (spec 5.3)
BH_Q = 0.10                                   # pre-registered (spec 5.2)
HOLDOUT_PASS_RATIO = 0.5                      # pre-registered (spec 3.6)
# Every parameter that can change a trial's outcome MUST appear here, or a
# re-run with a changed value would dedupe against the stale trial -- breaking
# the "any changed parameter is a NEW trial" rule. run_sweep AND run_holdout
# both build their hashed params through this one constructor so the journaled
# config always records what evaluate_trial truly ran.
#
# The Piece 3 perturbation params (symbol_subset, calendar_offset) are
# OMITTED when default-valued: an always-present new key would change the
# hash of every one of the journal's existing trials (799 at Piece 3 time),
# severing them from their dedupe identities. A default-valued perturbation
# IS the plain trial, so omission is also semantically exact. Pinned by
# test_default_config_hash_is_pinned_to_the_live_journal.
def _hashed_params(
    quantiles: int,
    tercile_below: int,
    min_names: int,
    symbol_subset: tuple[str, ...] | None = None,
    calendar_offset: int = 0,
) -> dict:
    params = {
        "quantiles": quantiles,
        "weighting": "equal",
        "cadence": "monthly",
        "tercile_below": tercile_below,
        "min_names": min_names,
    }
    if symbol_subset is not None:
        params["symbol_subset"] = sorted(symbol_subset)
    if calendar_offset != 0:
        params["calendar_offset"] = calendar_offset
    return params


DEFAULT_PARAMS = _hashed_params(QUANTILES, TERCILE_BELOW, MIN_NAMES)


class SweepError(RuntimeError):
    """A sweep/holdout invariant was violated; refuse loudly."""


def trials_journal(journal_dir: Path) -> Journal:
    return Journal(journal_dir / "alphasearch-trials.jsonl")


def trial_config(
    signal: str, universe: str, window: str, params: dict | None = None
) -> dict:
    return {
        "signal": signal,
        "universe": universe,
        "window": window,
        "params": dict(params or DEFAULT_PARAMS),
    }


def trial_config_hash(config: dict) -> str:
    payload = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def _json_safe(value: object) -> object:
    """NaN/inf -> None, recursively; numpy scalars -> Python scalars. The
    journal must stay strict JSON: json.dumps would happily emit invalid bare
    NaN, and would emit the equally non-standard Infinity/-Infinity tokens for
    +-inf, which a strict JSON reader elsewhere would choke on."""
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(v) for v in value]
    if hasattr(value, "item"):  # numpy scalar
        value = value.item()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


# Keys log_trial itself controls; a result payload containing any of these
# would silently clobber journal-controlled fields if merged in last.
RESERVED_RESULT_KEYS = frozenset(
    {"event", "kind", "config_hash", "ts", "error", "signal", "universe",
     "window", "params", "battery"}
)


def log_trial(
    journal: Journal,
    *,
    kind: str,  # "discovery" | "holdout" | "battery"
    config: dict,
    ts: str,  # ISO-8601 UTC, supplied by the CLI (the only clock reader)
    result: dict | None = None,
    error: str | None = None,
    battery: str | None = None,  # Piece 3: display/grouping tag, NEVER hashed
) -> dict:
    """Append one trial event (spec section 4 schema) and return it."""
    result = result or {}
    clobbered = RESERVED_RESULT_KEYS & result.keys()
    if clobbered:
        raise SweepError(
            f"result payload cannot set reserved journal keys: {sorted(clobbered)}"
        )
    event = {
        "event": "trial",
        "kind": kind,
        **config,
        "config_hash": trial_config_hash(config),
        "ts": ts,
        "error": error,
        **result,
    }
    if battery is not None:
        # On the EVENT only (spec Piece 3 section 5): the hash comes from
        # `config` above, so an identical evaluation made inside or outside
        # a battery stays ONE trial -- the tag is for display and grouping.
        event["battery"] = battery
    event = _json_safe(event)
    journal.append(event)
    return event


def load_trials(journal: Journal) -> list[dict]:
    """All trial events, deduplicated: latest per (config_hash, kind) wins."""
    latest: dict[tuple[str, str], dict] = {}
    for event in journal.events():
        if event.get("event") != "trial":
            continue
        latest[(event["config_hash"], event["kind"])] = event
    return list(latest.values())


def discovery_trials(journal: Journal) -> list[dict]:
    """The honest trial count for BH/DSR = len() of this list."""
    return [e for e in load_trials(journal) if e.get("kind") == "discovery"]


def prior_holdout_trial(journal: Journal, signal: str, universe: str) -> dict | None:
    """Any prior holdout event for (signal, universe) -- ANY window/params:
    the holdout is touched once per candidate, not once per configuration."""
    last: dict | None = None
    for event in journal.events():
        if (
            event.get("event") == "trial"
            and event.get("kind") == "holdout"
            and event.get("signal") == signal
            and event.get("universe") == universe
        ):
            last = event
    return last


def find_discovery_trial(
    journal: Journal,
    signal: str,
    universe: str,
    window: str = DISCOVERY_WINDOW,
    params: dict | None = None,
) -> dict | None:
    """The discovery trial for (signal, universe) under `params` (defaults:
    DEFAULT_PARAMS), by exact config hash -- the reference a holdout is
    compared against."""
    wanted = trial_config_hash(trial_config(signal, universe, window, params=params))
    for event in load_trials(journal):
        if event.get("kind") == "discovery" and event.get("config_hash") == wanted:
            return event
    return None


# --------------------------------------------------------------------------- #
# Universes (spec 3.2): Piece 1's two gathered options pools, plus Piece 2's
# segment universes (explicit symbols; samples optional). Every signal family
# in a universe is measured on the same cross-section.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class UniverseSpec:
    name: str
    cache_dir: Path
    samples: Path | None
    fundamentals_dir: Path | None
    # Form 4 insider store; None = no store, requires_insider signals are
    # refused at sweep assembly (mirrors fundamentals_dir).
    insider_dir: Path | None = None
    # Piece 2: an explicit universe (segment pools). None = derive from the
    # samples allowlist (Piece 1 behavior). The segment's identity in the
    # hashed trial config is its NAME (spec section 3.3); the symbol list is
    # derived from committed CSVs, like bar caches are for Piece 1 pools.
    symbols: tuple[str, ...] | None = None
    # Committed sic_map.csv override for sector derivation (industry-relative
    # family); None = segments.DEFAULT_SIC_MAP_CSV. Not part of the hashed
    # trial config: like bar caches, the map is committed data, and the
    # universe's identity is its NAME.
    sic_map_path: Path | None = None
    # R3 down-cap: the (band, symbol, start, end) membership CSV and the band(s)
    # this universe counts. When set, build_universe_panel loads the per-symbol
    # intervals for `bands` and hands them to build_panel, so PanelView.symbols
    # is per-date band-filtered. None (the default) = Piece 1/2 static behavior.
    membership_intervals: Path | None = None
    bands: tuple[str, ...] | None = None


def default_universes(root: Path) -> dict[str, UniverseSpec]:
    return {
        "largecap": UniverseSpec(
            "largecap",
            root / "data" / "equities-tiingo",
            root / "data" / "options-iv" / "samples.jsonl",
            root / "data" / "fundamentals" / "equities",
            insider_dir=root / "data" / "insider" / "equities",
        ),
        "midcap": UniverseSpec(
            "midcap",
            root / "data" / "equities-midcap-tiingo",
            root / "data" / "options-iv" / "samples-midcap.jsonl",
            root / "data" / "fundamentals" / "equities",
            insider_dir=root / "data" / "insider" / "equities",
        ),
    }


def _universe_sectors(sic_map_path: Path | None) -> dict[str, str]:
    """symbol -> its (unique) frozen SEGMENTS sector for every mapped symbol.
    Industry segments (biotech, banks) overlap sectors and are NOT the
    partition, so only kind == "sector" qualifies. Imported lazily:
    segments.py imports this module at module scope, so a top-level import
    back would be a cycle."""
    from trading.alphasearch.segments import SEGMENTS, load_sic_map, segments_for

    sector_of: dict[str, str] = {}
    for symbol, code in load_sic_map(sic_map_path).items():
        sector = next(
            (n for n in segments_for(code) if SEGMENTS[n].kind == "sector"), None
        )
        if sector is not None:
            sector_of[symbol] = sector
    return sector_of


def build_universe_panel(
    spec: UniverseSpec, factors: pd.DataFrame | None = None
) -> PanelData:
    membership = None
    if spec.membership_intervals is not None and spec.bands is not None:
        # Lazy import: downcap_membership imports UniverseSpec from this module,
        # so a top-level import here would be a cycle (same pattern as
        # _universe_sectors' lazy segments import).
        from trading.venues.universes.downcap_membership import load_band_membership

        membership = load_band_membership(spec.membership_intervals, frozenset(spec.bands))
    return build_panel(
        spec.cache_dir, spec.samples, spec.fundamentals_dir,
        insider_dir=spec.insider_dir,
        symbols=spec.symbols, factors=factors,
        sectors=_universe_sectors(spec.sic_map_path),
        membership=membership,
    )


def _check_universe_supports(panel: PanelData, spec: SignalSpec, universe: str) -> None:
    """Spec section 6: a universe/signal mismatch is refused at assembly time,
    never silently skipped (a silent skip would corrupt the trial count).

    The refusal names the expected store, how to populate it, and the
    zero-setup workaround -- a bare "has none" tells the operator nothing
    actionable and sends them hunting through the codebase.
    """
    if spec.requires_options and not panel.options:
        raise SweepError(
            f"signal {spec.name!r} requires options cells; universe {universe!r} "
            "has none. Gather them with `scripts/gather_options_iv.py` (writes "
            "data/options-iv/samples*.jsonl); or work around it by passing "
            "--signals with a non-options signal subset"
        )
    if spec.requires_fundamentals and not panel.fundamentals:
        raise SweepError(
            f"signal {spec.name!r} requires fundamentals; universe {universe!r} "
            "has none. Expected store: data/fundamentals/equities (the "
            "fundamentals_dir config/equities.toml points at). Populate it with "
            "`scripts/backfill_fundamentals.py`; or work around it by passing "
            "--signals with a non-fundamentals signal subset"
        )
    if spec.requires_option_volume and not panel.has_option_volume:
        raise SweepError(
            f"signal {spec.name!r} requires per-leg option volume; universe "
            f"{universe!r} cells carry none (leg volume ships only in the "
            "mid-cap gather -- see data/options-iv/samples-midcap.jsonl). "
            "Re-gather this universe's cells with a volume-carrying "
            "`scripts/gather_options_iv.py` run, or work around it by "
            "passing --signals without the option-volume family"
        )
    if spec.requires_insider and not panel.insider:
        raise SweepError(
            f"signal {spec.name!r} requires insider transactions; universe "
            f"{universe!r} has none. Expected store: data/insider/equities "
            "(UniverseSpec.insider_dir points at it). Populate it with "
            "`scripts/build_insider_store.py`; or work around it by passing "
            "--signals with a non-insider signal subset"
        )


# --------------------------------------------------------------------------- #
# One trial: panel + signal + window -> the spec section-4 result payload
# --------------------------------------------------------------------------- #
def _window_bounds(window: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    start_s, _, end_s = window.partition("..")
    if not end_s:
        raise SweepError(f"window must be 'YYYY-MM-DD..YYYY-MM-DD', got {window!r}")
    return pd.Timestamp(start_s, tz="UTC"), pd.Timestamp(end_s, tz="UTC")


# The Ken French daily-factor files publish with a short lag; a few calendar
# days of slack distinguishes "normal publication lag" from "this cache is
# actually stale and will silently truncate the regression."
FACTOR_STALENESS_TOLERANCE_DAYS = 7


def _check_factor_coverage(factors: pd.DataFrame, window_end: pd.Timestamp) -> None:
    """Refuse loudly if the factor cache does not reach the window end (spec
    section 4: run_regression's inner join would otherwise silently truncate
    to whatever dates overlap, understating the window without a trace)."""
    factors_end = factors.index.max() if len(factors) else None
    tolerance = pd.Timedelta(FACTOR_STALENESS_TOLERANCE_DAYS, unit="D")
    if factors_end is None or factors_end < (window_end - tolerance):
        have = "no data" if factors_end is None else factors_end.date().isoformat()
        raise ValueError(
            f"factor cache ends {have} but the window ends "
            f"{window_end.date().isoformat()} (tolerance "
            f"{FACTOR_STALENESS_TOLERANCE_DAYS}d); refresh the factor cache with "
            "`trading alphasearch ... --refresh-factors` (or "
            "`scripts/factor_regression.py --refresh`) before trusting this trial"
        )


def _series_moments(returns: pd.Series) -> tuple[float, float]:
    """(skew, Pearson kurtosis) of a daily series -- the DSR's inputs."""
    r = returns.dropna().to_numpy()
    if len(r) < 4:
        return math.nan, math.nan
    mean = r.mean()
    sd = r.std()  # population, per the DSR definition
    if sd == 0:
        return math.nan, math.nan
    skew = float(((r - mean) ** 3).mean() / sd**3)
    kurt = float(((r - mean) ** 4).mean() / sd**4)  # normal = 3
    return skew, kurt


def _daily_sharpe(returns: pd.Series) -> float:
    r = returns.dropna()
    if len(r) < 2:
        return math.nan
    sd = float(r.std(ddof=1))
    return float(r.mean()) / sd if sd > 0 else math.nan


def _leg_stats(alpha: AlphaResult, returns: pd.Series) -> dict:
    """The journaled per-leg payload (spec section 4 'ls'/'lo' blocks).

    Everything the leaderboard and DSR need is HERE, so `leaderboard` can be
    recomputed from the journal alone -- no panel rebuild, no factor refetch.
    """
    four = alpha.four_factor
    df = four.n - len(four.names)
    skew, kurt = _series_moments(returns)
    return {
        "alpha_annual_pct": four.alpha_annual_pct,
        "alpha_t": four.alpha_tstat,
        "p": stats.p_from_t(four.alpha_tstat, df),
        "capm_alpha_annual_pct": alpha.capm_alpha_annual_pct,
        "capm_alpha_t": alpha.capm_alpha_tstat,
        "loadings": {
            name: float(b) for name, b in zip(four.names[1:], four.beta[1:], strict=True)
        },
        "loadings_t": {
            name: float(t) for name, t in zip(four.names[1:], four.tstat[1:], strict=True)
        },
        "r2": four.r2,
        "n_obs": four.n,
        "sharpe": alpha.sharpe_annual,
        "sharpe_daily": _daily_sharpe(returns),
        "skew": skew,
        "kurt": kurt,
    }


def evaluate_trial_with_sort(
    panel: PanelData,
    spec: SignalSpec,
    window: str,
    factors: pd.DataFrame,
    *,
    quantiles: int = QUANTILES,
    tercile_below: int = TERCILE_BELOW,
    min_names: int = MIN_NAMES,
    symbol_subset: tuple[str, ...] | None = None,
    calendar_offset: int = 0,
):
    """evaluate_trial plus the underlying SortResult. The R1-amended battery
    needs each perturbed evaluation's raw lo series and memberships (to build
    the cost-charged LO-minus-SPY active series the re-anchored checks score);
    the journaled payload keeps only summary stats, so without this the
    battery would have to run every sort twice."""
    start, end = _window_bounds(window)
    _check_factor_coverage(factors, end)
    dates = panel.decision_dates(start, end, offset=calendar_offset)
    sort = portfolio_sort(
        panel, spec, dates, end,
        quantiles=quantiles, tercile_below=tercile_below, min_names=min_names,
        symbol_subset=symbol_subset,
    )
    ls_alpha = evaluate_alpha(sort.ls, factors, self_financing=True)
    lo_alpha = evaluate_alpha(sort.lo, factors, self_financing=False)
    result = {
        "n_dates": sort.n_dates,
        "n_names_median": sort.n_names_median,
        "ls": _leg_stats(ls_alpha, sort.ls),
        "lo": _leg_stats(lo_alpha, sort.lo),
        "turnover_monthly": sort.turnover_monthly,
        "skipped_dates": list(sort.skipped_dates),
    }
    return result, sort


def evaluate_trial(
    panel: PanelData,
    spec: SignalSpec,
    window: str,
    factors: pd.DataFrame,
    *,
    quantiles: int = QUANTILES,
    tercile_below: int = TERCILE_BELOW,
    min_names: int = MIN_NAMES,
    symbol_subset: tuple[str, ...] | None = None,
    calendar_offset: int = 0,
) -> dict:
    """Score -> sort -> regress. Raises SortError/ValueError/LinAlgError on
    failure; the caller journals that as an error trial. symbol_subset and
    calendar_offset are Piece 3 battery perturbations: callers that set them
    MUST hash them into the trial config via _hashed_params (they change the
    outcome)."""
    result, _sort = evaluate_trial_with_sort(
        panel, spec, window, factors,
        quantiles=quantiles, tercile_below=tercile_below, min_names=min_names,
        symbol_subset=symbol_subset, calendar_offset=calendar_offset,
    )
    return result


# --------------------------------------------------------------------------- #
# Leaderboard: recomputed from the journal ALONE (the auditable view)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class LeaderboardRow:
    signal: str
    universe: str
    window: str
    alpha_annual_pct: float | None
    alpha_t: float | None
    p: float | None
    bh_pass: bool
    dsr: float | None
    capm_alpha_annual_pct: float | None
    capm_alpha_t: float | None
    loadings: dict
    turnover_monthly: float | None
    lo_alpha_t: float | None
    n_names_median: float | None
    n_dates: int | None
    skipped_dates: int
    error: str | None


def _pval(trial: dict) -> float:
    p = (trial.get("ls") or {}).get("p")
    return float("nan") if p is None else float(p)


def _abs_t_key(row: LeaderboardRow) -> float:
    if row.alpha_t is None:
        return 0.0
    t = float(row.alpha_t)
    return -abs(t) if not math.isnan(t) else 0.0


def build_leaderboard(journal: Journal) -> tuple[list[LeaderboardRow], int]:
    """(rows sorted by |4F L/S t| desc, honest discovery-trial count).

    BH is computed across EVERY journaled discovery trial -- prior sweeps
    included -- never just the current run (spec 3.5). Error trials carry
    p=NaN -> 1.0: they cannot pass but they raise the bar for everyone.
    """
    trials = discovery_trials(journal)
    n_trials = len(trials)
    if n_trials == 0:
        return [], 0
    mask = stats.bh_fdr(np.array([_pval(t) for t in trials]), q=BH_Q)
    daily_sharpes = [
        t["ls"]["sharpe_daily"]
        for t in trials
        if t.get("ls") and t["ls"].get("sharpe_daily") is not None
    ]
    var_sr = float(np.var(daily_sharpes, ddof=1)) if len(daily_sharpes) >= 2 else 0.0
    rows: list[LeaderboardRow] = []
    for trial, passed in zip(trials, mask, strict=True):
        ls = trial.get("ls") or {}
        lo = trial.get("lo") or {}
        dsr = None
        if passed and ls.get("sharpe_daily") is not None:
            dsr = stats.deflated_sharpe(
                sr=float(ls["sharpe_daily"]),
                n_obs=int(ls["n_obs"]),
                skew=float(ls["skew"]) if ls.get("skew") is not None else 0.0,
                kurt=float(ls["kurt"]) if ls.get("kurt") is not None else 3.0,
                n_trials=n_trials,
                var_trials_sr=var_sr,
            )
            if math.isnan(dsr):
                dsr = None  # keep leaderboard rows strictly JSON-serializable
        rows.append(
            LeaderboardRow(
                signal=trial["signal"],
                universe=trial["universe"],
                window=trial["window"],
                alpha_annual_pct=ls.get("alpha_annual_pct"),
                alpha_t=ls.get("alpha_t"),
                p=ls.get("p"),
                bh_pass=bool(passed),
                dsr=dsr,
                capm_alpha_annual_pct=ls.get("capm_alpha_annual_pct"),
                capm_alpha_t=ls.get("capm_alpha_t"),
                loadings=ls.get("loadings") or {},
                turnover_monthly=trial.get("turnover_monthly"),
                lo_alpha_t=lo.get("alpha_t"),
                n_names_median=trial.get("n_names_median"),
                n_dates=trial.get("n_dates"),
                skipped_dates=len(trial.get("skipped_dates") or []),
                error=trial.get("error"),
            )
        )
    rows.sort(key=_abs_t_key)
    return rows, n_trials


def _bh_survivor_hashes(journal: Journal) -> set[str]:
    """Config hashes of the discovery trials that CURRENTLY clear the BH gate.

    Hash-keyed on purpose: a holdout gate matching on (signal, universe) alone
    would let an unrelated exploratory trial (different window/params) that
    passed BH qualify a holdout for the FAILED canonical trial."""
    trials = discovery_trials(journal)
    if not trials:
        return set()
    mask = stats.bh_fdr(np.array([_pval(t) for t in trials]), q=BH_Q)
    return {t["config_hash"] for t, ok in zip(trials, mask, strict=True) if ok}


def battery_verdict(journal: Journal, config_hash: str) -> dict | None:
    """The latest kind="battery" verdict event for a discovery config hash
    (Piece 3). load_trials' (config_hash, kind) dedupe makes re-runs replace
    in place; None when the battery has never been run for this config."""
    for event in load_trials(journal):
        if event.get("kind") == "battery" and event.get("config_hash") == config_hash:
            return event
    return None


# --------------------------------------------------------------------------- #
# --long-only leaderboard (R1 gate amendment spec section 4 deliverable 2):
# re-derives EVERY journaled discovery trial's cost-charged long-only series
# from CURRENT data and ranks it against SPY buy-and-hold over the trial's
# own window. A DISPLAY, never a re-journaling -- no trial is re-scored, no
# event is written, no touch is spent. Only the raw daily lo series can't be
# recovered from the journal (the journal keeps summary stats, not the
# series), so this necessarily re-runs the sort; trials whose signal/universe
# no longer resolves, or whose data can't reproduce it, show honestly as n/a.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class LongOnlyRow:
    signal: str
    universe: str
    window: str
    config_hash: str
    lo_sharpe: float | None
    lo_total_return: float | None
    spy_sharpe: float | None
    spy_total_return: float | None
    beats_spy: bool | None      # None when re-derivation failed (n/a)
    skipped_no_spread: int | None
    error: str | None           # set -> every numeric field above is None


def _rederive_long_only_row(
    trial: dict,
    universes: dict[str, UniverseSpec],
    panels: dict[str, PanelData],
    spy_closes: pd.Series,
) -> LongOnlyRow:
    from trading.alphasearch.costs import cost_charged_lo, spy_benchmark
    from trading.alphasearch.evaluate import annualized_sharpe, total_return

    signal_name, universe_name, window = trial["signal"], trial["universe"], trial["window"]
    config_hash = trial["config_hash"]

    def _na(error: str) -> LongOnlyRow:
        return LongOnlyRow(
            signal=signal_name, universe=universe_name, window=window,
            config_hash=config_hash, lo_sharpe=None, lo_total_return=None,
            spy_sharpe=None, spy_total_return=None, beats_spy=None,
            skipped_no_spread=None, error=error,
        )

    if signal_name not in SIGNALS:
        return _na(f"unknown signal {signal_name!r} (registry has changed)")
    if universe_name not in universes:
        return _na(f"unknown universe {universe_name!r} (not resolved for this view)")
    panel = panels.get(universe_name)
    if panel is None:
        return _na(f"panel for universe {universe_name!r} could not be assembled")
    spec = SIGNALS[signal_name]
    params = trial.get("params") or {}
    subset = params.get("symbol_subset")
    try:
        start, end = _window_bounds(window)
        dates = panel.decision_dates(start, end, offset=int(params.get("calendar_offset", 0)))
        sort = portfolio_sort(
            panel, spec, dates, end,
            quantiles=int(params.get("quantiles", QUANTILES)),
            tercile_below=int(params.get("tercile_below", TERCILE_BELOW)),
            min_names=int(params.get("min_names", MIN_NAMES)),
            symbol_subset=tuple(subset) if subset is not None else None,
        )
        charged_lo, skipped = cost_charged_lo(panel, sort.lo, sort.rebalances)
        # Fix (final review, 2026-07-10): same alignment as run_battery's
        # long_only_gate -- anchor SPY's benchmark window to the first ACTUAL
        # DECISION date (sort.rebalances[0][0]), not charged_lo.index[0]:
        # portfolio_sort builds hold segments strictly after the decision
        # date, so charged_lo.index[0] is the first REALIZED RETURN day and
        # already compounds the move from the decision date to that day --
        # one trading day later than SPY needs to match the identical
        # economic horizon. sort.rebalances is non-empty whenever portfolio_
        # sort succeeds (SortError on empty tops is caught below).
        lo_window_start = sort.rebalances[0][0]
        lo_sharpe = annualized_sharpe(charged_lo)
        lo_total = total_return(charged_lo)
        spy_stats = spy_benchmark(spy_closes, lo_window_start, end)
    except (SweepError, SortError, ValueError, IndexError, np.linalg.LinAlgError) as exc:
        return _na(f"{type(exc).__name__}: {exc}")
    beats = (
        not math.isnan(lo_sharpe) and not math.isnan(spy_stats.sharpe_annual)
        and lo_sharpe >= spy_stats.sharpe_annual
        and not math.isnan(lo_total) and not math.isnan(spy_stats.total_return)
        and lo_total > spy_stats.total_return
    )
    return LongOnlyRow(
        signal=signal_name, universe=universe_name, window=window,
        config_hash=config_hash,
        lo_sharpe=None if math.isnan(lo_sharpe) else lo_sharpe,
        lo_total_return=None if math.isnan(lo_total) else lo_total,
        spy_sharpe=(
            None if math.isnan(spy_stats.sharpe_annual) else spy_stats.sharpe_annual
        ),
        spy_total_return=(
            None if math.isnan(spy_stats.total_return) else spy_stats.total_return
        ),
        beats_spy=beats, skipped_no_spread=skipped, error=None,
    )


def build_long_only_leaderboard(
    journal: Journal,
    universes: dict[str, UniverseSpec],
    factors: pd.DataFrame,
    spy_closes: pd.Series,
    *,
    panel_factory: Callable[[UniverseSpec, pd.DataFrame | None], PanelData] = (
        build_universe_panel
    ),
) -> list[LongOnlyRow]:
    """Every journaled discovery trial, re-derived and ranked by cost-charged
    long-only Sharpe vs SPY buy-and-hold (spec section 4 deliverable 2).
    Rows sorted by lo_sharpe descending, n/a (unrankable) rows last."""
    trials = discovery_trials(journal)
    panels: dict[str, PanelData] = {}
    for name, uspec in universes.items():
        try:
            panels[name] = panel_factory(uspec, factors)
        except PanelError:
            continue   # every trial in this universe honestly shows n/a below
    rows = [
        _rederive_long_only_row(trial, universes, panels, spy_closes)
        for trial in trials
    ]
    rows.sort(key=lambda r: (r.lo_sharpe is None, -(r.lo_sharpe or 0.0)))
    return rows


# --------------------------------------------------------------------------- #
# The sweep runner (spec 3.6)
# --------------------------------------------------------------------------- #
def run_sweep(
    universes: dict[str, UniverseSpec],
    journal: Journal,
    factors: pd.DataFrame,
    ts: str,
    *,
    signals: dict[str, SignalSpec] | None = None,
    window: str = DISCOVERY_WINDOW,
    quantiles: int = QUANTILES,
    tercile_below: int = TERCILE_BELOW,
    min_names: int = MIN_NAMES,
    panel_factory: Callable[[UniverseSpec, pd.DataFrame | None], PanelData] = (
        build_universe_panel
    ),
) -> tuple[list[LeaderboardRow], int]:
    """Enumerate signals x universes serially; build each panel once; journal
    EVERY trial BEFORE the leaderboard is computed (spec 3.6) so a crash
    mid-sweep can never yield counted-but-unjournaled trials."""
    if signals is not None and not signals:
        # `signals or SIGNALS` would silently expand an explicitly-empty
        # selection to the full registry; sweeping nothing is a caller bug.
        raise SweepError("no signals selected")
    chosen = SIGNALS if signals is None else signals
    params = _hashed_params(quantiles, tercile_below, min_names)
    # Validate the FULL signal x universe cross-product BEFORE any trial runs
    # (spec section 6: refused at sweep-ASSEMBLY time). Checking per-universe
    # inside the trial loop would abort mid-sweep, making "which trials got
    # journaled" depend on universe sort order. All-or-nothing: one SweepError
    # naming every incompatible pair, zero trials journaled.
    panels: dict[str, PanelData] = {}
    mismatches: list[str] = []
    for uname, uspec in sorted(universes.items()):
        panels[uname] = panel_factory(uspec, factors)
        for name in sorted(chosen):
            try:
                _check_universe_supports(panels[uname], chosen[name], uspec.name)
            except SweepError as exc:
                mismatches.append(str(exc))
    if mismatches:
        raise SweepError("; ".join(mismatches))
    for uname, uspec in sorted(universes.items()):
        panel = panels[uname]
        for name in sorted(chosen):
            config = trial_config(name, uspec.name, window, params=params)
            try:
                result: dict | None = evaluate_trial(
                    panel, chosen[name], window, factors,
                    quantiles=quantiles, tercile_below=tercile_below,
                    min_names=min_names,
                )
                # Spec section 6: corrupt cells are skipped AND counted; the
                # count rides on every trial event so coverage loss is audible.
                result["corrupt_cells"] = panel.corrupt_cells
                error = None
            except (SortError, ValueError, np.linalg.LinAlgError) as exc:
                result = None
                error = f"{type(exc).__name__}: {exc}"
            log_trial(journal, kind="discovery", config=config, ts=ts,
                      result=result, error=error)
    return build_leaderboard(journal)


# --------------------------------------------------------------------------- #
# Holdout re-prove (spec 3.6 + 5.3): touched once per (signal, universe),
# journal-enforced, mirroring backtest/experiments.py::prior_holdout.
# --------------------------------------------------------------------------- #
RERUN_CONFIRMATION = "RERUN HOLDOUT"

# Ken French publishes the daily factor files with a multi-week lag, so even a
# just-refreshed cache routinely ends BEFORE the latest equity bar. run_holdout
# therefore clamps its window end to min(latest bar, factor end) instead of
# refusing -- a staleness refusal there would brick the holdout for most of
# every month, and --refresh-factors could never satisfy it. But factors
# covering less than this many calendar days of the holdout window make the
# clamped evaluation statistically meaningless (too few daily observations to
# re-prove an annualized alpha), and THAT is refused pre-touch. ~6 months
# (~126 daily obs) is the floor: below it the alpha t-stat is noise.
HOLDOUT_MIN_FACTOR_SPAN_DAYS = 182


def holdout_passes(discovery_alpha: float, holdout_alpha: float) -> bool:
    """Pre-registered pass rule: same sign AND the holdout point estimate
    retains >= HOLDOUT_PASS_RATIO of the discovery magnitude. Both conditions
    collapse to the signed ratio (positive iff same-signed), which applies the
    magnitude test symmetrically for negative-alpha candidates."""
    if (
        discovery_alpha == 0
        or math.isnan(discovery_alpha)
        or math.isnan(holdout_alpha)
    ):
        return False
    return holdout_alpha / discovery_alpha >= HOLDOUT_PASS_RATIO


def latest_bar_date(panel: PanelData) -> pd.Timestamp:
    return max(series.index.max() for series in panel.closes.values())


@dataclass(frozen=True)
class HoldoutOutcome:
    event: dict
    passed: bool | None          # None when the holdout evaluation errored
    discovery_alpha: float
    holdout_alpha: float | None
    window: str


def run_holdout(
    uspec: UniverseSpec,
    journal: Journal,
    factors: pd.DataFrame,
    ts: str,
    signal_name: str,
    *,
    holdout_start: str = HOLDOUT_START,
    discovery_window: str = DISCOVERY_WINDOW,
    confirm: Callable[[], str] = lambda: "",
    quantiles: int = QUANTILES,
    tercile_below: int = TERCILE_BELOW,
    min_names: int = MIN_NAMES,
    min_factor_span_days: int = HOLDOUT_MIN_FACTOR_SPAN_DAYS,
    panel_factory: Callable[[UniverseSpec, pd.DataFrame | None], PanelData] = (
        build_universe_panel
    ),
) -> HoldoutOutcome:
    """Evaluate ONE BH survivor on the reserved holdout window.

    Refusals (SweepError) protect the once-only holdout: unknown signal; no
    clean same-params discovery trial with a usable alpha; that EXACT trial
    (by config hash, never merely the (signal, universe) pair) not a current
    BH survivor; already holdout-touched unless confirm() returns the literal
    RERUN_CONFIRMATION; factor cache covering < min_factor_span_days of the
    holdout; battery not passed (Piece 3). The realized window end --
    min(latest bar, factor end), i.e. exactly what evaluate_trial sees -- is
    journaled so the evaluation is exactly reproducible.
    """
    if signal_name not in SIGNALS:
        known = ", ".join(sorted(SIGNALS))
        raise SweepError(f"unknown signal {signal_name!r}; known: {known}")
    params = _hashed_params(quantiles, tercile_below, min_names)
    discovery = find_discovery_trial(
        journal, signal_name, uspec.name, discovery_window, params=params
    )
    if discovery is None:
        raise SweepError(
            f"no discovery trial for {signal_name}:{uspec.name} over "
            f"{discovery_window} with matching params; run the sweep first"
        )
    if discovery.get("error"):
        raise SweepError(
            f"discovery trial for {signal_name}:{uspec.name} errored "
            f"({discovery['error']}); nothing to re-prove"
        )
    discovery_alpha_raw = (discovery.get("ls") or {}).get("alpha_annual_pct")
    if discovery_alpha_raw is None:
        # Journaled NaN -> null: no usable baseline. Refuse HERE, before the
        # once-only touch is spent -- crashing after log_trial would burn it.
        raise SweepError(
            f"discovery trial for {signal_name}:{uspec.name} has no usable "
            f"L/S alpha (journaled as null); nothing to re-prove against"
        )
    discovery_alpha = float(discovery_alpha_raw)
    # Bind the gate to the EXACT trial being re-proven: an unrelated
    # (signal, universe) row that survived BH under a different window/params
    # must not spend the holdout against THIS trial's baseline alpha.
    if discovery["config_hash"] not in _bh_survivor_hashes(journal):
        raise SweepError(
            f"discovery trial {discovery['config_hash']} for "
            f"{signal_name}:{uspec.name} over {discovery_window} is not a "
            f"current BH survivor (q={BH_Q}); the once-only holdout is "
            f"reserved for survivors"
        )
    # Piece 3 battery gate -- a WRITTEN PROSPECTIVE AMENDMENT to Piece 1 spec
    # 3.6, recorded in docs/superpowers/specs/2026-07-09-robustness-battery-
    # design.md section 3: no holdout may be spent on a survivor that has not
    # passed its robustness battery. No holdout had ever been spent when this
    # gate was added, so nothing is affected retroactively. Hash-keyed to the
    # EXACT discovery trial, like the BH gate above. What "battery-passed"
    # MEANS was itself amended prospectively (R1, docs/superpowers/specs/
    # 2026-07-10-longonly-gate-amendment.md section 2): the verdict's
    # `eligible` bit now requires cost-charged long-only Sharpe >= SPY's AND
    # total return > SPY's over discovery, replacing the 30bps L/S cost gate
    # -- this check just reads that bit, so its own logic is unchanged.
    verdict = battery_verdict(journal, discovery["config_hash"])
    if verdict is None or verdict.get("eligible") is not True:
        state = ("has not been run" if verdict is None
                 else "did not pass (not holdout-eligible)")
        raise SweepError(
            f"robustness battery for {signal_name}:{uspec.name} {state}; the "
            f"once-only holdout is reserved for battery-passed survivors. "
            f"Run `trading alphasearch robustness {signal_name}:{uspec.name}` "
            f"first"
        )
    prior = prior_holdout_trial(journal, signal_name, uspec.name)
    if prior is not None and confirm() != RERUN_CONFIRMATION:
        raise SweepError(
            f"holdout for {signal_name}:{uspec.name} already evaluated at "
            f"{prior['ts']}; rerunning invalidates the evidence — aborted"
        )

    panel = panel_factory(uspec, factors)
    spec = SIGNALS[signal_name]
    _check_universe_supports(panel, spec, uspec.name)
    # The FF publication lag means the factor cache routinely ends before the
    # latest bar (even freshly refreshed), so the window end is clamped to
    # what BOTH datasets cover: the journaled window must equal the window
    # actually evaluated -- journaling latest-bar while the regression only
    # reached the factor end would misrecord the evidence. Refuse pre-touch
    # (BEFORE the once-only touch is journaled) only when factors are truly
    # stale: covering < min_factor_span_days of the holdout leaves too few
    # observations to re-prove anything, and only a refresh can fix that.
    factors_end = factors.index.max() if len(factors) else None
    min_factor_end = pd.Timestamp(holdout_start, tz="UTC") + pd.Timedelta(
        min_factor_span_days, unit="D"
    )
    if factors_end is None or factors_end < min_factor_end:
        have = "no data" if factors_end is None else factors_end.date().isoformat()
        raise SweepError(
            f"factor cache ends {have} but the holdout starting {holdout_start} "
            f"needs factors through at least {min_factor_end.date().isoformat()} "
            f"({min_factor_span_days}d minimum span); refresh the factor cache "
            "with `trading alphasearch ... --refresh-factors` (or "
            "`scripts/factor_regression.py --refresh`)"
        )
    end = min(latest_bar_date(panel), factors_end)
    window = f"{holdout_start}..{end.date().isoformat()}"
    config = trial_config(signal_name, uspec.name, window, params=params)
    try:
        result: dict | None = evaluate_trial(
            panel, spec, window, factors,
            quantiles=quantiles, tercile_below=tercile_below, min_names=min_names,
        )
        error = None
    except (SortError, ValueError, np.linalg.LinAlgError) as exc:
        result = None
        error = f"{type(exc).__name__}: {exc}"
    except Exception as exc:
        # Any OTHER exception means holdout data was already read; journal an
        # error-kind event before re-raising so the once-only touch is never
        # spent without a journal record (the leak this guards against).
        error = f"{type(exc).__name__}: {exc}"
        log_trial(journal, kind="holdout", config=config, ts=ts,
                  result=None, error=error)
        raise
    event = log_trial(journal, kind="holdout", config=config, ts=ts,
                      result=result, error=error)

    if result is None:
        return HoldoutOutcome(event, None, discovery_alpha, None, window)
    holdout_alpha = float(result["ls"]["alpha_annual_pct"])
    return HoldoutOutcome(
        event,
        holdout_passes(discovery_alpha, holdout_alpha),
        discovery_alpha,
        holdout_alpha,
        window,
    )
