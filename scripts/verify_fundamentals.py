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

Check 1b -- AAPL value primitives at the same 2023-02-03 filing:

    shares outstanding: 15,842,407,000 (10-Q cover page, as of 2023-01-20)
    book equity: $56,727M (condensed balance sheet, total shareholders'
    equity at 2022-12-31)
    TTM net income ($M): Q1 FY23 29,998 + derived Q4 FY22
    (99,803 - 34,630 - 25,010 - 19,442 = 20,721) + Q3 FY22 19,442
    + Q2 FY22 25,010 = 95,171

    Earnings-yield composition sanity at the PINNED raw close of $154.50
    (2023-02-03): market cap = 15,842,407,000 x 154.50 = $2.448T ->
    earnings yield = 95.171e9 / 2.448e12 = 0.0389. This is pure arithmetic
    over store primitives -- the live ranker uses adjusted closes from bars,
    so the pinned-close number validates the primitives, not yfinance.

Check 2 -- restatement regression against REAL data: every (cik, fy, fp)
filed more than once as a plain 10-K/10-Q across 2018+ must appear in the
store ONLY via its earliest accession; later re-filings never leak
(amendment forms are excluded structurally and cannot appear at all).

Check 3 -- ticker-recycling reconciliation (Task 4 review item, routed here):
for every cik_map.csv symbol/CIK membership interval, the backfilled store
should show at least one filing with a FILED date inside that interval's
overlap with the backfill's coverage window. A CIK that produced ZERO
filings across its whole membership window is either a foreign private
issuer (20-F/40-F, structurally excluded -- edgar.py accepts only 10-K/10-Q)
or evidence the symbol was mapped to the wrong CIK (e.g. ticker recycling).
Either way it lands on an audit list -- this check observes, it does not
gate the backfill.

Usage: uv run python scripts/verify_fundamentals.py
"""

from __future__ import annotations

import datetime
import sys
import tomllib
from pathlib import Path

import pandas as pd

from trading.fundamentals.cik_map import load_cik_map
from trading.fundamentals.edgar import load_quarter_facts
from trading.fundamentals.store import FundamentalsStore

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "edgar-raw"

AAPL_EXPECTED_GP = (387_537e6 - 220_666e6) / 346_747e6  # 0.4813, derivation above
AAPL_TOLERANCE = 0.002
AAPL_EXPECTED_VALUE = {  # derivations in the module docstring (Check 1b)
    "shares_outstanding": 15_842_407_000.0,
    "book_equity": 56_727e6,
    "ttm_net_income": 95_171e6,
}
VALUE_REL_TOLERANCE = 1e-3
AAPL_CLOSE_2023_02_03 = 154.50  # pinned raw close for the composition check
AAPL_EXPECTED_EARNINGS_YIELD = 0.0389
EARNINGS_YIELD_TOLERANCE = 0.001
# Reconciliation coverage window: the locked backfill start (see
# scripts/backfill_fundamentals.py) through today. An interval that never
# overlaps this window (e.g. a company delisted before 2018) has nothing to
# reconcile and is skipped.
RECONCILIATION_START = pd.Timestamp("2018-01-01", tz="UTC")


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


def check_restatements(store_root: Path, cik_map: pd.DataFrame) -> None:
    ciks = set(cik_map["cik"])
    parts = []
    for zip_path in sorted(RAW_DIR.glob("*.zip")):
        facts = load_quarter_facts(zip_path, ciks)
        if not facts.empty:
            parts.append(facts[["cik", "adsh", "fy", "fp", "filed"]].drop_duplicates("adsh"))
    filings = pd.concat(parts, ignore_index=True).drop_duplicates("adsh")
    later_adshes: set[str] = set()
    dup_groups = 0
    for _, group in filings.groupby(["cik", "fy", "fp"]):
        if group["adsh"].nunique() > 1:
            dup_groups += 1
            ordered = group.sort_values(["filed", "adsh"], kind="mergesort")
            later_adshes.update(ordered["adsh"].iloc[1:])
    if dup_groups == 0:
        sys.exit("FATAL: no re-filed (cik, fy, fp) found across 2018+; scan is broken")
    stored: set[str] = set()
    for path in sorted(store_root.glob("*.parquet")):
        stored.update(pd.read_parquet(path, columns=["adsh"])["adsh"])
    leaked = sorted(later_adshes & stored)
    if leaked:
        sys.exit(f"FATAL: PIT violation -- later re-filings leaked into the store: {leaked[:10]}")
    print(
        f"restatement invariant OK: {dup_groups} re-filed fiscal periods found in the raw "
        f"data; zero later accessions present in the store"
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


def main() -> None:
    data_cfg = tomllib.loads((ROOT / "config" / "equities.toml").read_text())["data"]
    store_root = ROOT / data_cfg["fundamentals_dir"]
    store = FundamentalsStore(store_root)
    cik_map = load_cik_map()
    check_aapl(store)
    check_restatements(store_root, cik_map)
    check_recycling_reconciliation(store, cik_map)
    n_files = len(list(store_root.glob("*.parquet")))
    print(f"store coverage: {n_files} symbols with fundamentals under {store_root}")


if __name__ == "__main__":
    main()
