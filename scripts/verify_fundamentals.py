"""One-shot verification of the M4 fundamentals backfill. Run AFTER
scripts/backfill_fundamentals.py (re-parses the cached ZIPs; no network).

Check 1 -- AAPL PIT spot-check, TTM basis. The xbrl-scout reported AAPL's
2023-02-03 10-Q SINGLE-QUARTER gross profitability as 0.1452 =
(117,154 - 66,822) / 346,747 ($M). On the locked TTM basis the expectation
recomputes as (all $M, all from ORIGINAL filings):

    FY2022 10-K (filed 2022-10-28):  revenue 394,328  cogs 223,546
    Q1 FY22 10-Q (filed 2022-01-28): revenue 123,945  cogs  69,702
    Q2 FY22 10-Q (filed 2022-04-29): revenue  97,278  cogs  54,719
    Q3 FY22 10-Q (filed 2022-07-29): revenue  82,959  cogs  47,074
    derived Q4 FY22 = FY - (Q1+Q2+Q3): revenue 90,146  cogs 52,051
    Q1 FY23 10-Q (filed 2023-02-03): revenue 117,154  cogs  66,822; assets 346,747

    TTM revenue = 97,278 + 82,959 + 90,146 + 117,154 = 387,537
    TTM cogs    = 54,719 + 47,074 + 52,051 +  66,822 = 220,666
    gross profitability = (387,537 - 220,666) / 346,747 = 0.4813

Check 1c -- pre-ASC-606 spot-check (Best Buy FY2018 10-K). BBY reports revenue
under the legacy SalesRevenueNet tag and COGS under the legacy CostOfGoodsSold
tag; both were added to edgar.REVENUE_TAGS / COGS_TAGS as post-606 fallbacks
to recover early-history quality coverage. The check confirms the parser now
computes BBY's revenue_ttm 42,151 / cogs_ttm 32,275 / assets 13,049 ($M) ->
gross profitability 0.7568 AND resolves them via those exact legacy tags.

Check 1b -- AAPL value primitives at the same 2023-02-03 filing:

    shares outstanding: 15,821,946,000 (dei:EntityCommonStockSharesOutstanding
    cover-page tag, as of 2023-01-20 -- verbatim on the filed 10-Q cover:
    "15,821,946,000 shares of common stock were issued and outstanding as
    of January 20, 2023")
    book equity: $56,727M (condensed balance sheet, total shareholders'
    equity at 2022-12-31)
    TTM net income ($M): Q1 FY23 29,998 + derived Q4 FY22
    (99,803 - 34,630 - 25,010 - 19,442 = 20,721) + Q3 FY22 19,442
    + Q2 FY22 25,010 = 95,171

    Earnings-yield composition sanity at the PINNED raw close of $154.50
    (2023-02-03): market cap = 15,821,946,000 x 154.50 = $2.4445T ->
    earnings yield = 95.171e9 / 2.4445e12 = 0.0389. This is pure arithmetic
    over store primitives -- the live ranker uses adjusted closes from bars,
    so the pinned-close number validates the primitives, not yfinance.

Check 2 -- restatement regression against REAL data. The invariant is
VISIBILITY TIMING: for every fiscal period a cik re-filed (same period end
+ form, multiple plain 10-K/10-Q accessions across the raw ZIPs), any
store row for that period must sit at the EARLIEST filing's filed date --
history never becomes visible later than the original filing (amendment
forms are excluded structurally and cannot appear at all). Accession
provenance is deliberately NOT part of the invariant; see
check_restatements' docstring for the companyfacts attribution quirk.

Check 3 -- ticker-recycling reconciliation (Task 4 review item, routed here):
for every cik_map.csv symbol/CIK membership interval, the backfilled store
should show at least one filing with a FILED date inside that interval's
overlap with the backfill's coverage window. A CIK that produced ZERO
filings across its whole membership window is either a foreign private
issuer (20-F/40-F, structurally excluded -- edgar.py accepts only 10-K/10-Q)
or evidence the symbol was mapped to the wrong CIK (e.g. ticker recycling).
Either way it lands on an audit list -- this check observes, it does not
gate the backfill.

Check 4 -- shares_outstanding coverage gate. The whole point of switching
the backfill's primary source to companyfacts (see trading.fundamentals.
backfill) was fixing shares_outstanding coverage; this check holds that
regime to account. Across CURRENT equities_membership.csv members, the
LATEST stored row's shares_outstanding must be non-NaN for at least
SHARES_COVERAGE_MIN (85%; the ceiling is structural, see the constant's
comment) of them, or the run fails loudly.

Check 5 -- neutral-fraction coverage table (report-only, no gate). Missing
fundamentals data doesn't error -- it flows through as a NEUTRAL 0.5
percentile in the quality/value rankers (spec). That's silent by design,
so this prints, at NEUTRAL_FRACTION_SAMPLE_DATES, what fraction of current
members would see a GENUINE (non-neutral) quality / earnings-yield /
book-to-market component at each date, using the store's step-function
"last row as-of" read the rankers themselves use. Earnings-yield and
book-to-market also need shares_outstanding (the market-cap term); this
check treats that as part of "genuine" for them, same as the ranker would.
Any date where quality genuine coverage sits below QUALITY_GENUINE_FLOOR
(30%) gets an explicit NOTE line: a mostly-neutral era ranks mostly on
momentum, and experiment interpretation must account for that.

Usage: uv run python scripts/verify_fundamentals.py
"""

from __future__ import annotations

import datetime
import math
import sys
import tomllib
from pathlib import Path

import pandas as pd

from trading.fundamentals.cik_map import load_cik_map
from trading.fundamentals.edgar import load_quarter_facts
from trading.fundamentals.store import FundamentalsStore

# scripts/ has no __init__.py (not a package); when this file runs as
# `uv run python scripts/verify_fundamentals.py`, Python already puts its own
# directory on sys.path, but the explicit insert also makes the import work
# under pytest (which imports this module by path, not by running it).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from backfill_fundamentals import SOURCE_MARKER, _read_source_marker  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "edgar-raw"
MEMBERSHIP_CSV = ROOT / "src" / "trading" / "venues" / "universes" / "equities_membership.csv"

AAPL_EXPECTED_GP = (387_537e6 - 220_666e6) / 346_747e6  # 0.4813, derivation above
AAPL_TOLERANCE = 0.002
AAPL_EXPECTED_VALUE = {  # derivations in the module docstring (Check 1b)
    # 15,821,946,000 is the dei cover-page count as of 2023-01-20, verified
    # verbatim against the filed 10-Q document (aapl-20221231.htm). The
    # PREVIOUS expectation here, 15,842,407,000, was the us-gaap
    # CommonStockSharesOutstanding balance-sheet instant at 2022-12-31 --
    # the FALLBACK tag the retired ZIP path resolved because FSDS strips
    # the dei cover-page fact -- and was wrongly attributed to the cover
    # page. The companyfacts-primary rebuild resolves the locked chain's
    # primary tag, so the expectation moved to the true cover-page value
    # (sanctioned update, final-review fix wave).
    "shares_outstanding": 15_821_946_000.0,
    "book_equity": 56_727e6,
    "ttm_net_income": 95_171e6,
}
VALUE_REL_TOLERANCE = 1e-3
# Pre-ASC-606 spot-check: Best Buy's FY2018 10-K (fiscal year ended
# 2018-02-03, filed 2018-04-02, accn 0000764478-18-000013) reports revenue
# under the legacy SalesRevenueNet tag and COGS under the legacy
# CostOfGoodsSold tag -- NEITHER of which the pre-fix chain resolved, so BBY's
# quality was neutral-0.5 across all of early history. Values verified against
# companyfacts (which mirrors the filed 10-K): net sales $42,151M, cost of
# sales $32,275M, total assets $13,049M -> gross profitability
# (42,151 - 32,275) / 13,049 = 0.7568. This is the TTM basis (a 10-K's four
# derived quarters sum to the full year), so revenue_ttm/cogs_ttm equal the
# annual figures. Guards the pre-606 fallback tags the same way check_aapl
# guards the post-606 chain.
BBY_FILED = "2018-04-02"
BBY_EXPECTED = {
    "revenue_ttm": 42_151e6,
    "cogs_ttm": 32_275e6,
    "assets": 13_049e6,
    "gross_profitability": (42_151e6 - 32_275e6) / 13_049e6,  # 0.7568
}
BBY_EXPECTED_TAGS = {"revenue_tag": "SalesRevenueNet", "cogs_tag": "CostOfGoodsSold"}
BBY_REL_TOLERANCE = 1e-3
AAPL_CLOSE_2023_02_03 = 154.50  # pinned raw close for the composition check
# 95.171e9 / (15,821,946,000 x 154.50 = 2.4445e12) = 0.03893; the shares
# correction above moved this by +0.00003 (was 0.03890 at the old share
# count), well inside the tolerance, so the rounded expectation stands.
AAPL_EXPECTED_EARNINGS_YIELD = 0.0389
EARNINGS_YIELD_TOLERANCE = 0.001
# Reconciliation coverage window: the locked backfill start (see
# scripts/backfill_fundamentals.py) through today. An interval that never
# overlaps this window (e.g. a company delisted before 2018) has nothing to
# reconcile and is skipped.
RECONCILIATION_START = pd.Timestamp("2018-01-01", tz="UTC")
# Check 4: the regime-mismatch problem the companyfacts-primary switch fixes
# (see trading.fundamentals.backfill) is specifically shares_outstanding, so
# this is the one primitive with a hard coverage floor rather than a
# report-only table. 85%, not higher, for a STRUCTURAL reason: ~77 current
# members are multi-class filers (META, BRK-B, CMCSA, ACN, ABNB, BF-B,
# CHTR, ...) whose cover-page share counts are tagged per share class with
# a class dimension, and the companyfacts API serves consolidated
# (undimensioned) facts only -- so no source in this pipeline can resolve
# them and they rank value-neutral (0.5) by design. Measured ceiling at the
# 2026-07-06 rebuild: 89.7% (823/918). Named follow-up (per-class
# summation): see PROVENANCE.md's fundamentals section.
SHARES_COVERAGE_MIN = 0.85
# Check 5: three points across the backfill's history -- shortly after the
# TTM warm-up completes (2019), roughly mid-history, and today (whatever the
# store's latest coverage looks like right now).
NEUTRAL_FRACTION_SAMPLE_DATES = [
    pd.Timestamp("2019-07-01", tz="UTC"),
    pd.Timestamp("2022-06-30", tz="UTC"),
    pd.Timestamp(datetime.date.today(), tz="UTC"),
]
# Below this, quality's genuine coverage is sparse enough that a
# quality_momentum_v1 experiment over that era is mostly momentum in
# disguise (nearly everyone neutral-0.5); the table flags such dates so
# experiment interpretation has to account for it. Early history was
# historically thin because the pre-ASC-606 revenue/COGS tags
# (SalesRevenueNet, CostOfGoodsSold) sat OUTSIDE the chain; those single-value
# fallbacks are now IN the chain (edgar.REVENUE_TAGS / COGS_TAGS), which
# materially lifts 2018-2019 quality coverage on the next backfill. A residual
# gap remains for filers that report ONLY the split parts with no total
# (SalesRevenueGoodsNet + SalesRevenueServicesNet, CostOfServices) -- those
# need per-part summation, a deliberately deferred follow-up.
QUALITY_GENUINE_FLOOR = 0.30


def current_member_symbols() -> list[str]:
    df = pd.read_csv(MEMBERSHIP_CSV, comment="#", dtype=str).fillna("")
    return sorted(set(df.loc[df["end"] == "", "symbol"]))


def check_aapl(store: FundamentalsStore) -> None:
    frame = store.read("AAPL")
    ts = pd.Timestamp("2023-02-03", tz="UTC")
    if ts not in frame.index:
        sys.exit("FATAL: AAPL has no row at filed date 2023-02-03; backfill incomplete?")
    row = frame.loc[ts]
    gp = float(row["gross_profitability"])
    print(
        f"AAPL @ 2023-02-03: gp={gp:.4f} (expected ~{AAPL_EXPECTED_GP:.4f}), "
        f"revenue_ttm={row['revenue_ttm']:.0f}, cogs_ttm={row['cogs_ttm']:.0f}, "
        f"assets={row['assets']:.0f}, adsh={row['adsh']}, tags="
        f"{row['revenue_tag']}/{row['cogs_tag']}/{row['assets_tag']}"
    )
    if abs(gp - AAPL_EXPECTED_GP) > AAPL_TOLERANCE:
        sys.exit(f"FATAL: AAPL TTM gross profitability {gp:.4f} != {AAPL_EXPECTED_GP:.4f}")
    for key, expected in AAPL_EXPECTED_VALUE.items():
        got = float(row[key])
        print(f"AAPL @ 2023-02-03: {key}={got:.0f} (expected {expected:.0f})")
        if abs(got - expected) > abs(expected) * VALUE_REL_TOLERANCE:
            sys.exit(f"FATAL: AAPL {key} {got:.0f} != {expected:.0f}")
    market_cap = float(row["shares_outstanding"]) * AAPL_CLOSE_2023_02_03
    earnings_yield = float(row["ttm_net_income"]) / market_cap
    print(
        f"AAPL @ 2023-02-03: earnings yield at pinned close "
        f"${AAPL_CLOSE_2023_02_03:.2f} = {earnings_yield:.4f} "
        f"(expected ~{AAPL_EXPECTED_EARNINGS_YIELD})"
    )
    if abs(earnings_yield - AAPL_EXPECTED_EARNINGS_YIELD) > EARNINGS_YIELD_TOLERANCE:
        sys.exit(
            f"FATAL: AAPL earnings yield {earnings_yield:.4f} != {AAPL_EXPECTED_EARNINGS_YIELD}"
        )


def check_bestbuy(store: FundamentalsStore) -> None:
    """Check 1c -- pre-ASC-606 spot-check. Best Buy's FY2018 10-K resolved
    revenue via SalesRevenueNet and COGS via CostOfGoodsSold, the two legacy
    single-value tags added to the chain to recover early-history quality
    coverage. Confirms the parser now computes BBY's revenue/COGS/gross
    profitability (matching the filed 10-K) AND resolves them via the expected
    legacy tags -- the counterpart to check_aapl's post-606 guard."""
    frame = store.read("BBY")
    ts = pd.Timestamp(BBY_FILED, tz="UTC")
    if ts not in frame.index:
        sys.exit(
            f"FATAL: BBY has no row at filed date {BBY_FILED}; backfill incomplete "
            "or pre-ASC-606 fallback tags not yet in the store?"
        )
    row = frame.loc[ts]
    print(
        f"BBY @ {BBY_FILED}: gp={float(row['gross_profitability']):.4f} "
        f"(expected ~{BBY_EXPECTED['gross_profitability']:.4f}), "
        f"revenue_ttm={row['revenue_ttm']:.0f}, cogs_ttm={row['cogs_ttm']:.0f}, "
        f"assets={row['assets']:.0f}, adsh={row['adsh']}, tags="
        f"{row['revenue_tag']}/{row['cogs_tag']}/{row['assets_tag']}"
    )
    for key, expected in BBY_EXPECTED.items():
        got = float(row[key])
        if abs(got - expected) > abs(expected) * BBY_REL_TOLERANCE:
            sys.exit(f"FATAL: BBY {key} {got:.4f} != {expected:.4f} (pre-606 fallback broken)")
    for key, expected_tag in BBY_EXPECTED_TAGS.items():
        if row[key] != expected_tag:
            sys.exit(
                f"FATAL: BBY {key} resolved as {row[key]!r}, expected {expected_tag!r} "
                "-- the pre-ASC-606 fallback did not win as intended"
            )


def check_restatements(store_root: Path, cik_map: pd.DataFrame) -> None:
    """Check 2: for every re-filed report in the raw ZIPs -- same cik, same
    fiscal period END, same form, multiple plain 10-K/10-Q accessions --
    any store row for that period must be dated at the EARLIEST filing's
    filed date. The invariant is visibility TIMING: history never becomes
    visible later than the original filing.

    Accession provenance is deliberately NOT asserted. SEC's companyfacts
    dedupes identical facts across original + re-filed submissions and
    attributes the surviving entry to the NEWEST accession number while
    keeping the ORIGINAL filed date (verified live: CRM's re-filed Q1 FY21
    10-Q, accn 0001108524-22-000007, carries filed=2020-06-01 -- the
    original date -- with the original values; when a re-filing CHANGES a
    value, companyfacts keeps both entries as separate accessions and the
    earliest-filed dedup still picks the original). A store adsh naming a
    re-file accession at the original filed date is therefore expected on
    a companyfacts-built store, not a leak. A store row dated at the
    RE-FILING's later date is the real violation this check catches.

    Groups are keyed by (cik, period, form) rather than FSDS (fy, fp)
    labels: label noise (fiscal-year-change transitions, 52/53-week
    filers) can give two genuinely different reporting periods one (fy,
    fp) label, which are not restatements at all. FSDS also rounds some
    period ends to month end where companyfacts keeps the exact date, so
    odd-calendar filers' groups may simply match no store row -- they are
    skipped, and a vacuous run (zero matched rows) fails loudly."""
    ciks = set(cik_map["cik"])
    parts = []
    for zip_path in sorted(RAW_DIR.glob("*.zip")):
        facts = load_quarter_facts(zip_path, ciks)
        if not facts.empty:
            parts.append(facts[["cik", "adsh", "form", "period", "filed"]].drop_duplicates("adsh"))
    filings = pd.concat(parts, ignore_index=True).drop_duplicates("adsh")
    original_filed: dict[tuple[int, str, str], pd.Timestamp] = {}
    for (cik, period, form), group in filings.groupby(["cik", "period", "form"]):
        if group["adsh"].nunique() > 1:
            ordered = group.sort_values(["filed", "adsh"], kind="mergesort")
            key = (int(cik), period.date().isoformat(), str(form))
            original_filed[key] = ordered["filed"].iloc[0]
    if not original_filed:
        sys.exit("FATAL: no re-filed (cik, period, form) found across 2018+; scan is broken")

    symbols_by_cik: dict[int, list[str]] = {
        int(cik): sorted(set(group["symbol"])) for cik, group in cik_map.groupby("cik")
    }
    frames: dict[str, pd.DataFrame] = {}  # symbol -> store frame, read once
    store = FundamentalsStore(store_root)
    violations: list[str] = []
    checked = 0
    for (cik, period_iso, form), orig_filed in original_filed.items():
        for symbol in symbols_by_cik.get(cik, []):
            if symbol not in frames:
                frames[symbol] = store.read(symbol)
            frame = frames[symbol]
            if frame.empty:
                continue
            rows = frame[(frame["period"] == period_iso) & (frame["form"] == form)]
            for filed, row in rows.iterrows():
                checked += 1
                if filed.tz_localize(None) > orig_filed:
                    violations.append(
                        f"{symbol} cik={cik} {form} period={period_iso}: store row filed "
                        f"{filed.date()} > original filing {orig_filed.date()} "
                        f"(adsh={row['adsh']})"
                    )
    if violations:
        sys.exit(
            "FATAL: PIT violation -- store visibility later than the original filing:\n  "
            + "\n  ".join(violations[:10])
        )
    if checked == 0:
        sys.exit("FATAL: restatement scan matched zero store rows; the check ran vacuously")
    print(
        f"restatement invariant OK: {len(original_filed)} re-filed (cik, period, form) "
        f"groups in the raw data; {checked} matching store row(s) all visible at the "
        f"original filing date"
    )


def check_recycling_reconciliation(
    store: FundamentalsStore, cik_map: pd.DataFrame
) -> list[dict[str, object]]:
    """Ticker-recycling audit routed into this script from Task 4's review:
    a mapped symbol whose CIK produced NO filings across its whole
    membership period is suspicious -- either a foreign private issuer
    excluded by the 10-K/10-Q form filter, or a wrong-company mapping.
    Prints and returns the audit list; never exits nonzero (observational)."""
    today = pd.Timestamp(datetime.date.today(), tz="UTC")
    audit: list[dict[str, object]] = []
    for row in cik_map.itertuples():
        start = pd.Timestamp(row.start, tz="UTC")
        end = pd.Timestamp(row.end, tz="UTC") if row.end else today
        window_start = max(start, RECONCILIATION_START)
        window_end = min(end, today)
        if window_start >= window_end:
            continue  # interval doesn't overlap the backfilled coverage window
        frame = store.read(row.symbol)
        covered = False
        if not frame.empty:
            in_window = (frame.index >= window_start) & (frame.index < window_end)
            covered = bool(in_window.any())
        if not covered:
            audit.append(
                {
                    "symbol": row.symbol,
                    "cik": row.cik,
                    "start": row.start,
                    "end": row.end or "(current)",
                }
            )
    if audit:
        print(
            f"RECONCILIATION AUDIT: {len(audit)} symbol/CIK mapping(s) with zero filings in "
            f"their backfilled membership window (foreign private issuer excluded by the "
            f"form filter, or a possible wrong-company mapping):"
        )
        for item in audit:
            print(f"  {item['symbol']} (cik={item['cik']}, {item['start']}..{item['end']})")
    else:
        print("RECONCILIATION AUDIT: 0 symbol/CIK mappings with zero filings in their window")
    return audit


def check_shares_coverage(store: FundamentalsStore, members: list[str]) -> None:
    """Check 4: the companyfacts-primary switch exists to fix
    shares_outstanding coverage (trading.fundamentals.backfill), so this is
    a hard gate, not a report-only table -- across CURRENT members, the
    latest stored row's shares_outstanding must resolve at least
    SHARES_COVERAGE_MIN of the time."""
    covered = 0
    missing: list[str] = []
    for symbol in members:
        frame = store.read(symbol)
        if not frame.empty and not math.isnan(frame.iloc[-1]["shares_outstanding"]):
            covered += 1
        else:
            missing.append(symbol)
    coverage = covered / len(members) if members else 0.0
    print(
        f"shares_outstanding coverage (current members, latest stored row): "
        f"{coverage:.1%} ({covered}/{len(members)})"
    )
    if coverage < SHARES_COVERAGE_MIN:
        sys.exit(
            f"FATAL: shares_outstanding coverage {coverage:.1%} < {SHARES_COVERAGE_MIN:.0%} "
            f"across current members; missing (first 20): {sorted(missing)[:20]}"
        )


def _as_of_row(frame: pd.DataFrame, as_of: pd.Timestamp) -> pd.Series | None:
    """The step-function "last row as-of" read the quality/value rankers
    use: the latest filing visible by `as_of`, or None before any filing."""
    window = frame[frame.index <= as_of]
    return window.iloc[-1] if not window.empty else None


def check_neutral_fraction_coverage(store: FundamentalsStore, members: list[str]) -> None:
    """Check 5 (report-only): at each NEUTRAL_FRACTION_SAMPLE_DATES, what
    fraction of current members have a GENUINE (non-neutral) value for each
    ranker component, versus falling through to the neutral 0.5 percentile
    (missing filing, NaN metric, or -- for the two ratios -- missing
    shares_outstanding, which the ranker also needs for market cap)."""
    print("neutral-fraction coverage (genuine vs neutral-0.5, current members):")
    header = f"  {'date':<12} {'quality':>9} {'earnings_yield':>16} {'book_to_market':>16}"
    print(header)
    sparse_quality: list[tuple[str, float]] = []
    for as_of in NEUTRAL_FRACTION_SAMPLE_DATES:
        quality_ok = earnings_yield_ok = book_to_market_ok = 0
        for symbol in members:
            row = _as_of_row(store.read(symbol), as_of)
            if row is None:
                continue
            shares = row["shares_outstanding"]
            if not math.isnan(row["gross_profitability"]):
                quality_ok += 1
            if not math.isnan(row["ttm_net_income"]) and not math.isnan(shares):
                earnings_yield_ok += 1
            if not math.isnan(row["book_equity"]) and not math.isnan(shares):
                book_to_market_ok += 1
        n = len(members)
        quality_frac = quality_ok / n
        print(
            f"  {as_of.date().isoformat():<12} {quality_frac:>9.1%} "
            f"{earnings_yield_ok / n:>16.1%} {book_to_market_ok / n:>16.1%}"
        )
        if quality_frac < QUALITY_GENUINE_FLOOR:
            sparse_quality.append((as_of.date().isoformat(), quality_frac))
    for date_iso, frac in sparse_quality:
        print(
            f"  NOTE: quality genuine coverage {frac:.1%} < {QUALITY_GENUINE_FLOOR:.0%} at "
            f"{date_iso} -- mostly-neutral era (residual after the pre-ASC-606 "
            f"SalesRevenueNet/CostOfGoodsSold fallbacks: split-only filers still need "
            f"per-part summation); interpret quality-ranker results over this period accordingly"
        )


def check_source_regime(store_root: Path) -> None:
    """This suite's expectations -- Check 1's AAPL TTM primitives (the
    dei cover-page shares count only companyfacts resolves) and Check 4's
    85% shares_outstanding coverage floor -- are locked to the
    companyfacts-primary regime (see trading.fundamentals.backfill's module
    docstring). A store built with --source zips has materially different
    (much lower) shares_outstanding coverage and would fail those checks for
    a reason that has nothing to do with a real regression, surfacing as a
    confusing AAPL FATAL deep in check_aapl rather than naming the real
    problem. Read the marker up front instead and fail loudly with a clear
    message. A missing marker (legacy store predating the marker, or one
    whose backfill was interrupted before the marker write) is not itself
    disqualifying -- warn and continue assuming companyfacts, rather than
    block a store that may well be fine."""
    marker = _read_source_marker(store_root)
    if marker is not None and marker != "companyfacts":
        sys.exit(
            "FATAL: this verification suite assumes a companyfacts-built store "
            f"(found: {marker}); rebuild with --source companyfacts"
        )
    if marker is None:
        print(
            f"WARNING: no {SOURCE_MARKER} marker at {store_root} (legacy store predating "
            "the marker, or a backfill interrupted before it was written); assuming "
            "companyfacts regime and continuing"
        )


def main() -> None:
    data_cfg = tomllib.loads((ROOT / "config" / "equities.toml").read_text())["data"]
    store_root = ROOT / data_cfg["fundamentals_dir"]
    check_source_regime(store_root)
    store = FundamentalsStore(store_root)
    cik_map = load_cik_map()
    members = current_member_symbols()
    check_aapl(store)
    check_bestbuy(store)
    check_restatements(store_root, cik_map)
    check_recycling_reconciliation(store, cik_map)
    check_shares_coverage(store, members)
    check_neutral_fraction_coverage(store, members)
    n_files = len(list(store_root.glob("*.parquet")))
    print(f"store coverage: {n_files} symbols with fundamentals under {store_root}")


if __name__ == "__main__":
    main()
