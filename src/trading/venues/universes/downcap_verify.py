"""Phase-A frozen GO/NO-GO gate + human report (R3 spec section 4). Pure
computation over the A4 diagnostics artifact -- no re-fetching. Thresholds
are FROZEN before any sweep so the decision is not post-hoc. The
dollar-volume-only fallback (shares-coverage < 70%) is an AUTOMATIC,
developer-pre-approved path recorded as a written amendment, never a silent
re-tune."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

SURVIVORSHIP_MIN = 0.15
SHARES_COVERAGE_MIN = 0.70
BREADTH_MIN = 15

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


def _breadth_for(tradeable_in_band: pd.DataFrame, name: str, bands: set[str]) -> UniverseBreadth:
    sub = tradeable_in_band[tradeable_in_band["band"].isin(bands)]
    if sub.empty:
        # A band with ZERO tradeable in-band candidate-months anywhere in the
        # window is the EXTREME case of a sub-15 month, not an exemption from
        # it: zero < BREADTH_MIN just like any other shortfall. Fail the
        # gate and record it distinctly ("empty") so the report and any
        # downstream consumer can tell "genuinely swept and passed" apart
        # from "nothing to sweep -- dropped" (spec section 4: "a universe
        # with any sub-15 month is dropped from the sweep, RECORDED, not
        # silently skipped").
        return UniverseBreadth(name, 0, False, "empty")
    per_month = sub.groupby("date")["symbol"].nunique()
    min_count = int(per_month.min())
    ok = min_count >= BREADTH_MIN
    return UniverseBreadth(name, min_count, ok, "pass" if ok else "sub15")


def compute_gate(diagnostics: pd.DataFrame) -> GateResult:
    if diagnostics.empty:
        # A total-backfill-failure diagnostics artifact (zero rows) must FAIL
        # CLOSED, not crash: boolean-indexing an all-object zero-row frame
        # (e.g. `diagnostics[diagnostics["tradeable"]]`) raises a spurious
        # `KeyError` on an unrelated column in this pandas version, since it
        # drops columns when building the empty result. Short-circuit before
        # any of that filtering with a clean, honest NO-GO.
        breadth = [
            UniverseBreadth(name, 0, False, "empty") for name in _UNIVERSE_BANDS
        ]
        return GateResult(
            survivorship_pct=0.0,
            shares_coverage_pct=0.0,
            spread_median=float("nan"),
            spread_iqr=float("nan"),
            spread_pct_le_2=float("nan"),
            breadth=breadth,
            survivorship_ok=False,
            shares_coverage_ok=False,
            breadth_ok=False,
            fallback_triggered=True,
            go=False,
        )
    in_band = diagnostics[diagnostics["band"].notna()]
    tradeable = diagnostics[diagnostics["tradeable"]]
    # Breadth counts TRADEABLE in-band candidate-months only. In real A4
    # output `band` is non-None only when has_shares AND tradeable (see
    # downcap_band.evaluate_band's BandEval docstring), so this filter is a
    # no-op on production diagnostics -- but the gate must not silently rely
    # on that invariant, since an untradeable name can't actually be traded
    # and must not count toward the breadth floor.
    tradeable_in_band = in_band[in_band["tradeable"]]

    # Survivorship: delisted share of IN-BAND candidate-months.
    survivorship_pct = (
        float(in_band["delisted"].mean()) if len(in_band) else 0.0
    )
    # Shares-coverage: among TRADEABLE candidate-months, fraction with shares.
    shares_coverage_pct = (
        float(tradeable["has_shares"].mean()) if len(tradeable) else 0.0
    )
    # Spread realism: distribution across in-band rows.
    spreads = in_band["spread"].dropna()
    spread_median = float(spreads.median()) if len(spreads) else float("nan")
    spread_iqr = (
        float(spreads.quantile(0.75) - spreads.quantile(0.25)) if len(spreads) else float("nan")
    )
    spread_pct_le_2 = float((spreads <= 0.02).mean()) if len(spreads) else float("nan")
    # Breadth: per-universe minimum monthly tradeable in-band name count.
    breadth = [
        _breadth_for(tradeable_in_band, name, bands) for name, bands in _UNIVERSE_BANDS.items()
    ]

    survivorship_ok = survivorship_pct >= SURVIVORSHIP_MIN
    shares_coverage_ok = shares_coverage_pct >= SHARES_COVERAGE_MIN
    breadth_ok = all(b.ok for b in breadth)
    fallback_triggered = not shares_coverage_ok
    # GO on the market-cap band iff every frozen criterion holds. A fallback is
    # NOT a GO on the market-cap band -- it is a recorded amendment to the
    # dollar-volume-only construction, re-verified on its own rebuild.
    go = survivorship_ok and shares_coverage_ok and breadth_ok
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
    )


def render_report(gate: GateResult) -> str:
    lines = [
        "# R3 down-cap universe -- Phase A verification report",
        "",
        f"VERDICT: {'GO' if gate.go else 'NO-GO'}",
        "",
        f"- survivorship (delisted share of in-band months): {gate.survivorship_pct:.1%} "
        f"(>= {SURVIVORSHIP_MIN:.0%}? {'PASS' if gate.survivorship_ok else 'FAIL'})",
        f"- shares-coverage (tradeable months with PIT shares): {gate.shares_coverage_pct:.1%} "
        f"(>= {SHARES_COVERAGE_MIN:.0%}? {'PASS' if gate.shares_coverage_ok else 'FAIL'})",
        f"- spread: median {gate.spread_median:.4f}, IQR {gate.spread_iqr:.4f}, "
        f"% <= 2% {gate.spread_pct_le_2:.1%}",
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
    return (
        f"shares-coverage {gate.shares_coverage_pct:.1%} < {SHARES_COVERAGE_MIN:.0%}: "
        "NO-GO on the market-cap band. Developer-pre-approved fallback to a "
        "dollar-volume-only band (tradeability screens 2-3 without the cap "
        "bound). Rebuild membership with "
        "`scripts/build_downcap_membership.py --no-cap-band`, re-run this gate on "
        "the fallback diagnostics, and record the size/vintage skew of the "
        "shares-dropped names in experiments.md."
    )
