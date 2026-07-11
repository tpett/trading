"""Phase-A frozen GO/NO-GO gate + human report (R3 spec section 4). Pure
computation over the A4 diagnostics artifact -- no re-fetching. Thresholds
are FROZEN before any sweep so the decision is not post-hoc.

The dollar-volume-only fallback (shares-coverage < 70%) is a
developer-pre-approved path, but it is NOT automatic: it is resolved by a
written prospective amendment (spec section 4). `compute_gate(...,
require_cap_band=False)` is the tool that amendment invokes -- it produces a
REAL verdict on the dollar-volume-only universe (a single band, tradeability
screens 2-3 without the cap bound), not a re-run of the cap-mode gate. See
`render_amendment` for the executable next step."""

from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

from trading.venues.universes.downcap_membership import FALLBACK_BAND

SURVIVORSHIP_MIN = 0.15
SHARES_COVERAGE_MIN = 0.70
BREADTH_MIN = 15

# The name used for the single dollar-volume-only universe's breadth record
# and (per the written amendment, once triggered) its eventual UniverseSpec.
# Distinct from FALLBACK_BAND (the band LABEL written into the fallback
# membership CSV, imported above) -- this is the display/registration name.
FALLBACK_UNIVERSE_NAME = "downcap-dv"

# The three universes and the band(s) each counts (spec section 2).
_UNIVERSE_BANDS = {
    "downcap": {"micro", "small"},
    "downcap:small": {"small"},
    "downcap:micro": {"micro"},
}


@dataclass(frozen=True)
class UniverseBreadth:
    name: str
    min_month_count: int
    ok: bool
    # "pass" (>= BREADTH_MIN every month), "sub15" (nonempty band, but some
    # month fell below BREADTH_MIN), or "empty" (zero tradeable in-band
    # candidate-months anywhere in the window). All non-"pass" reasons are
    # gate failures (ok=False) -- "empty" is the extreme case of "sub15", not
    # an exemption from it (spec section 4: a universe with any sub-15 month,
    # including zero, is dropped from the sweep and RECORDED).
    reason: str


@dataclass(frozen=True)
class GateResult:
    survivorship_pct: float
    shares_coverage_pct: float
    spread_median: float
    spread_iqr: float
    spread_pct_le_2: float
    breadth: list[UniverseBreadth]
    survivorship_ok: bool
    shares_coverage_ok: bool
    breadth_ok: bool
    fallback_triggered: bool
    go: bool
    # Which mode produced this result. True (default) = the three cap-banded
    # universes, shares-coverage gates. False = the single dollar-volume-only
    # fallback universe, shares-coverage does NOT gate (see compute_gate).
    require_cap_band: bool = True
    # The shares-exclusion skew report (spec section 4's "report the
    # size/vintage skew of the dropped names" deliverable): among tradeable
    # candidate-months, those dropped for missing PIT shares (has_shares is
    # False) vs those kept, compared by dollar-volume (a size proxy).
    # Vintage is not present in the diagnostics artifact (see render_report).
    dropped_shares_count: int = 0
    dropped_dv_median: float = float("nan")
    kept_dv_median: float = float("nan")


def _breadth_from(sub: pd.DataFrame, name: str) -> UniverseBreadth:
    if sub.empty:
        # A universe with ZERO tradeable candidate-months anywhere in the
        # window is the EXTREME case of a sub-15 month, not an exemption
        # from it: zero < BREADTH_MIN just like any other shortfall. Fail
        # the gate and record it distinctly ("empty") so the report and any
        # downstream consumer can tell "genuinely swept and passed" apart
        # from "nothing to sweep -- dropped" (spec section 4: "a universe
        # with any sub-15 month is dropped from the sweep, RECORDED, not
        # silently skipped").
        return UniverseBreadth(name, 0, False, "empty")
    per_month = sub.groupby("date")["symbol"].nunique()
    min_count = int(per_month.min())
    ok = min_count >= BREADTH_MIN
    return UniverseBreadth(name, min_count, ok, "pass" if ok else "sub15")


def _breadth_for(tradeable_in_band: pd.DataFrame, name: str, bands: set[str]) -> UniverseBreadth:
    sub = tradeable_in_band[tradeable_in_band["band"].isin(bands)]
    return _breadth_from(sub, name)


def _shares_skew(tradeable: pd.DataFrame) -> tuple[int, float, float]:
    """The dropped-vs-kept dollar-volume skew among TRADEABLE candidate-months
    (spec section 4: "report the size/vintage skew of the dropped names").
    Vintage (listing date) is not carried in the diagnostics artifact, so
    only the dollar-volume (size) proxy is available here -- disclosed in
    render_report rather than silently omitted."""
    if tradeable.empty:
        return 0, float("nan"), float("nan")
    dropped = tradeable[~tradeable["has_shares"]]
    kept = tradeable[tradeable["has_shares"]]
    dropped_dv_median = float(dropped["dollar_volume"].median()) if len(dropped) else float("nan")
    kept_dv_median = float(kept["dollar_volume"].median()) if len(kept) else float("nan")
    return int(len(dropped)), dropped_dv_median, kept_dv_median


def _empty_result(require_cap_band: bool) -> GateResult:
    # A total-backfill-failure diagnostics artifact (zero rows) must FAIL
    # CLOSED, not crash: boolean-indexing an all-object zero-row frame
    # (e.g. `diagnostics[diagnostics["tradeable"]]`) raises a spurious
    # `KeyError` on an unrelated column in this pandas version, since it
    # drops columns when building the empty result. Short-circuit before any
    # of that filtering with a clean, honest NO-GO.
    if require_cap_band:
        breadth = [UniverseBreadth(name, 0, False, "empty") for name in _UNIVERSE_BANDS]
        shares_coverage_ok = False
        fallback_triggered = True
    else:
        # In fallback mode shares-coverage never gates (see compute_gate) --
        # that holds even on an empty artifact, so shares_coverage_ok is True
        # here too; there is no further fallback to trigger from fallback
        # mode itself, so fallback_triggered is False.
        breadth = [UniverseBreadth(FALLBACK_UNIVERSE_NAME, 0, False, "empty")]
        shares_coverage_ok = True
        fallback_triggered = False
    return GateResult(
        survivorship_pct=0.0,
        shares_coverage_pct=0.0,
        spread_median=float("nan"),
        spread_iqr=float("nan"),
        spread_pct_le_2=float("nan"),
        breadth=breadth,
        survivorship_ok=False,
        shares_coverage_ok=shares_coverage_ok,
        breadth_ok=False,
        fallback_triggered=fallback_triggered,
        go=False,
        require_cap_band=require_cap_band,
    )


def compute_gate(diagnostics: pd.DataFrame, *, require_cap_band: bool = True) -> GateResult:
    """Compute the Phase-A GO/NO-GO verdict.

    `require_cap_band=True` (default, unchanged): the three market-cap-banded
    universes (downcap, downcap:small, downcap:micro). Shares-coverage gates
    -- below SHARES_COVERAGE_MIN is a NO-GO on the cap band and
    `fallback_triggered` is set.

    `require_cap_band=False`: the developer-pre-approved dollar-volume-only
    fallback (spec section 4), invoked via a written amendment after a
    cap-mode NO-GO on shares-coverage -- NEVER silently. This is a SINGLE
    universe with no cap sub-bands: "in-universe" candidate-months are simply
    the tradeable ones (`tradeable == True`, the `band` column -- which is
    always cap-derived, regardless of mode -- is ignored). Shares-coverage
    does NOT gate here (the whole reason we're in fallback is that it
    failed); the raw fraction is still reported for the record. Survivorship
    and breadth are measured over this same tradeable population and DO
    still gate, so this call produces a genuine, independent verdict rather
    than reproducing the cap-mode NO-GO.
    """
    if diagnostics.empty:
        return _empty_result(require_cap_band)

    tradeable = diagnostics[diagnostics["tradeable"]]
    # Shares-coverage: among TRADEABLE candidate-months, fraction with shares.
    # Reported in both modes (raw fraction); only gates in cap mode.
    shares_coverage_pct = float(tradeable["has_shares"].mean()) if len(tradeable) else 0.0
    dropped_shares_count, dropped_dv_median, kept_dv_median = _shares_skew(tradeable)

    if require_cap_band:
        in_band = diagnostics[diagnostics["band"].notna()]
        # Breadth counts TRADEABLE in-band candidate-months only. In real A4
        # output `band` is non-None only when has_shares AND tradeable (see
        # downcap_band.evaluate_band's BandEval docstring), so this filter is
        # a no-op on production diagnostics -- but the gate must not silently
        # rely on that invariant, since an untradeable name can't actually be
        # traded and must not count toward the breadth floor.
        tradeable_in_band = in_band[in_band["tradeable"]]

        # Survivorship: delisted share of IN-BAND candidate-months. NOTE:
        # `delisted` over-counts ticker renames as delistings (see
        # downcap_roster.delisted_symbols) -- survivorship_pct is therefore a
        # lower bound on true survivorship-freeness, the safe direction for
        # a >= floor check.
        survivorship_pct = float(in_band["delisted"].mean()) if len(in_band) else 0.0
        # Spread realism: distribution across in-band rows.
        spreads = in_band["spread"].dropna()
        breadth = [
            _breadth_for(tradeable_in_band, name, bands) for name, bands in _UNIVERSE_BANDS.items()
        ]
        survivorship_ok = survivorship_pct >= SURVIVORSHIP_MIN
        shares_coverage_ok = shares_coverage_pct >= SHARES_COVERAGE_MIN
        breadth_ok = all(b.ok for b in breadth)
        fallback_triggered = not shares_coverage_ok
        # GO on the market-cap band iff every frozen criterion holds. A
        # fallback is NOT a GO on the market-cap band -- it is a recorded
        # amendment to the dollar-volume-only construction, re-verified on
        # its own call (require_cap_band=False) below.
        go = survivorship_ok and shares_coverage_ok and breadth_ok
    else:
        # Dollar-volume-only universe: no cap sub-bands, no `band` filter.
        # Survivorship + breadth are measured over the SAME tradeable
        # population shares-coverage was measured over above. Same
        # rename-over-count caveat as the cap-mode branch above applies.
        survivorship_pct = float(tradeable["delisted"].mean()) if len(tradeable) else 0.0
        spreads = tradeable["spread"].dropna()
        breadth = [_breadth_from(tradeable, FALLBACK_UNIVERSE_NAME)]
        survivorship_ok = survivorship_pct >= SURVIVORSHIP_MIN
        # Non-gating by design: the fallback exists BECAUSE shares-coverage
        # failed under the cap band. Re-imposing it here would make the
        # fallback dead (an automatic re-NO-GO), defeating its purpose.
        shares_coverage_ok = True
        breadth_ok = all(b.ok for b in breadth)
        # There is no further fallback to trigger from within fallback mode.
        fallback_triggered = False
        go = survivorship_ok and breadth_ok

    spread_median = float(spreads.median()) if len(spreads) else float("nan")
    spread_iqr = (
        float(spreads.quantile(0.75) - spreads.quantile(0.25)) if len(spreads) else float("nan")
    )
    spread_pct_le_2 = float((spreads <= 0.02).mean()) if len(spreads) else float("nan")

    return GateResult(
        survivorship_pct=survivorship_pct,
        shares_coverage_pct=shares_coverage_pct,
        spread_median=spread_median,
        spread_iqr=spread_iqr,
        spread_pct_le_2=spread_pct_le_2,
        breadth=breadth,
        survivorship_ok=survivorship_ok,
        shares_coverage_ok=shares_coverage_ok,
        breadth_ok=breadth_ok,
        fallback_triggered=fallback_triggered,
        go=go,
        require_cap_band=require_cap_band,
        dropped_shares_count=dropped_shares_count,
        dropped_dv_median=dropped_dv_median,
        kept_dv_median=kept_dv_median,
    )


def _fmt_dollars(x: float) -> str:
    return "n/a" if math.isnan(x) else f"${x:,.0f}"


def _dropped_names_section(gate: GateResult) -> str:
    return (
        f"- dropped for missing shares (tradeable, no PIT shares FILED <= D): "
        f"{gate.dropped_shares_count} candidate-months; median dollar-volume "
        f"{_fmt_dollars(gate.dropped_dv_median)} vs {_fmt_dollars(gate.kept_dv_median)} "
        "for shares-kept candidate-months (dollar-volume skew, a size proxy; "
        "vintage/listing-date is not carried in the diagnostics artifact so is "
        "not reported here)"
    )


def render_report(gate: GateResult) -> str:
    lines = [
        "# R3 down-cap universe -- Phase A verification report",
        "",
        "MODE: "
        + (
            "market-cap band (three universes)"
            if gate.require_cap_band
            else "DOLLAR-VOLUME-ONLY FALLBACK (cap bound dropped, single universe)"
        ),
        "",
        f"VERDICT: {'GO' if gate.go else 'NO-GO'}",
        "",
        f"- survivorship (delisted share of {'in-band' if gate.require_cap_band else 'tradeable'} "
        f"months): {gate.survivorship_pct:.1%} "
        f"(>= {SURVIVORSHIP_MIN:.0%}? {'PASS' if gate.survivorship_ok else 'FAIL'})",
    ]
    if gate.require_cap_band:
        lines.append(
            f"- shares-coverage (tradeable months with PIT shares): {gate.shares_coverage_pct:.1%} "
            f"(>= {SHARES_COVERAGE_MIN:.0%}? {'PASS' if gate.shares_coverage_ok else 'FAIL'})"
        )
    else:
        lines.append(
            "- shares-coverage: N/A for gating (fallback: cap bound dropped) -- "
            f"raw fraction, for the record: {gate.shares_coverage_pct:.1%}"
        )
    lines += [
        f"- spread: median {gate.spread_median:.4f}, IQR {gate.spread_iqr:.4f}, "
        f"% <= 2% {gate.spread_pct_le_2:.1%}",
        _dropped_names_section(gate),
        "- breadth (min tradeable names / month, each universe):",
    ]
    for b in gate.breadth:
        if b.reason == "pass":
            state = "PASS"
        elif b.reason == "empty":
            state = "DROP (empty: no tradeable candidate-months in window)"
        else:
            state = f"DROP (sub-15 month, min {b.min_month_count}/month)"
        lines.append(f"    {b.name}: min {b.min_month_count}/month -- {state}")
    if gate.fallback_triggered:
        lines += ["", "AMENDMENT TRIGGERED:", render_amendment(gate)]
    return "\n".join(lines)


def render_amendment(gate: GateResult) -> str:
    """The written prospective amendment spec section 4 requires when
    shares-coverage fails the cap band. EXECUTABLE: names the concrete
    re-run that produces a real, independent fallback verdict (never a
    re-run that reproduces the same cap-mode NO-GO), and states the
    fallback-sweep registration as the documented NEXT action -- not
    something this gate performs automatically."""
    return (
        f"shares-coverage {gate.shares_coverage_pct:.1%} < {SHARES_COVERAGE_MIN:.0%} measured: "
        "NO-GO on the market-cap band. Developer-pre-approved fallback (spec section 4): "
        "run `scripts/downcap_verify.py --fallback <diagnostics>` "
        "(compute_gate(diagnostics, require_cap_band=False)) to compute a REAL "
        "dollar-volume-only verdict -- tradeability screens 2-3 without the cap bound, "
        "a single universe, shares-coverage non-gating (this does NOT re-run the "
        "cap-mode gate and does NOT reproduce this NO-GO). Report that verdict "
        "(GO/NO-GO with its own survivorship + breadth) plus the size/vintage skew of "
        "the shares-dropped names (above) in experiments.md. If, and only if, the "
        "fallback verdict is GO, the next action -- NOT auto-run by this gate -- is to "
        "rebuild membership with `scripts/build_downcap_membership.py --no-cap-band` "
        f"and register a single dollar-volume-only `{FALLBACK_UNIVERSE_NAME}` UniverseSpec "
        f"from it (band label {FALLBACK_BAND!r}) for the sweep."
    )
