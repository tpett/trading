"""SEC EDGAR Financial Statement Data Set parsing (spec: M4 fundamentals overlay).

Parses one quarterly ZIP (sub.txt + num.txt) into the normalized facts table
(FACT_COLUMNS) that trading.fundamentals.metrics consumes. Pure file parsing:
downloads live in scripts/backfill_fundamentals.py, and the companyfacts JSON
top-up (trading.fundamentals.companyfacts) normalizes to this SAME table so
backfill and top-up cannot diverge.

Tag fallback chains are LOCKED to the xbrl-scout findings (2023q1 census:
the three revenue tags cover 76.5% of 10-K/10-Q filers, the three COGS tags
~48.5%, Assets 99.5%), extended with the value primitives (net income,
stockholders' equity, shares outstanding). Consolidated rows only (empty
segments + coreg). Roughly half of filers (banks, insurers) report no COGS
concept at all -> their quality metric is NaN downstream and ranks neutral
(0.5) by design, while their net income/equity/shares still resolve.

Shares date rule is PER TAG: the cover-date relaxation (ddate >= period,
latest wins) applies only to dei:EntityCommonStockSharesOutstanding, whose
post-period dating is a dei cover-page artifact; the us-gaap
CommonStockSharesOutstanding fallback is a normal balance-sheet instant held
to strict ddate == period (a later-dated instant may belong to the next
period's comparative).

Forms accepted are EXACTLY 10-K and 10-Q: amendments (10-K/A, 10-Q/A) never
enter the facts table, which is half of the PIT original-filing discipline
(the other half is metrics' earliest-filed dedup per (cik, fy, fp)).
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

USER_AGENT = "trading-system travis@launchsupply.com"  # mandatory on every SEC request

REVENUE_TAGS = [
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "Revenues",
    "RevenueFromContractWithCustomerIncludingAssessedTax",
]
COGS_TAGS = [
    "CostOfGoodsAndServicesSold",
    "CostOfRevenue",
    "CostOfGoodsAndServiceExcludingDepreciationDepletionAndAmortization",
]
ASSETS_TAGS = ["Assets"]
NET_INCOME_TAGS = ["NetIncomeLoss"]
EQUITY_TAGS = [
    "StockholdersEquity",
    "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
]
# dei:EntityCommonStockSharesOutstanding is the cover-page share count (dei
# taxonomy); CommonStockSharesOutstanding is the us-gaap balance-sheet
# fallback. num.txt keys facts by tag name only, so the taxonomy split
# matters solely to the companyfacts path (_TAXONOMY_BY_TAG there).
# Date rule is PER TAG: the cover-date relaxation (ddate >= period, latest
# wins) applies ONLY to the dei tag -- its post-period dating is a dei
# cover-page artifact. The us-gaap fallback is a normal balance-sheet
# instant held to strict ddate == period (a later-dated instant may belong
# to the next period's comparative).
_DEI_SHARES_TAG = "EntityCommonStockSharesOutstanding"
SHARES_TAGS = [_DEI_SHARES_TAG, "CommonStockSharesOutstanding"]
TAG_PRIORITY: dict[str, list[str]] = {
    "revenue": REVENUE_TAGS,
    "cogs": COGS_TAGS,
    "assets": ASSETS_TAGS,
    "net_income": NET_INCOME_TAGS,
    "equity": EQUITY_TAGS,
    "shares": SHARES_TAGS,
}
# Instant (balance-sheet / cover-page) concepts report qtrs=0; flows follow
# the form's duration (10-K qtrs=4, 10-Q qtrs=1).
INSTANT_CONCEPTS = frozenset({"assets", "equity", "shares"})
# Facts are USD except share counts.
UOM_BY_CONCEPT = {"shares": "shares"}

FACT_COLUMNS = [
    "cik",
    "adsh",
    "form",
    "fy",
    "fp",
    "period",
    "filed",
    "concept",
    "tag",
    "qtrs",
    "value",
]

_SUB_COLS = ["adsh", "cik", "form", "period", "fy", "fp", "filed"]
_NUM_COLS = ["adsh", "tag", "ddate", "qtrs", "uom", "segments", "coreg", "value"]


_FACT_DTYPES = {
    "cik": "int64",
    "qtrs": "int64",
    "value": "float64",
    "period": "datetime64[ns]",
    "filed": "datetime64[ns]",
}


def empty_facts() -> pd.DataFrame:
    return pd.DataFrame(columns=FACT_COLUMNS).astype(_FACT_DTYPES)


def load_quarter_facts(zip_path: Path, ciks: set[int] | None = None) -> pd.DataFrame:
    """One quarterly ZIP -> normalized facts: per (filing, concept) the single
    best fact by tag priority, consolidated only, the filing's OWN fiscal
    period only (ddate == sub.period), duration matched to the form (10-K
    qtrs=4 full year, 10-Q qtrs=1; instants qtrs=0). Exception: the dei
    cover-page share count is dated AFTER the period end, so THAT TAG ONLY
    takes the latest qtrs=0 fact at-or-after the period; the us-gaap shares
    fallback stays strict (ddate == period). All output rows carry the
    FILING's fiscal period in the `period` column regardless."""
    with zipfile.ZipFile(zip_path) as zf, zf.open("sub.txt") as fh:
        sub = pd.read_csv(fh, sep="\t", dtype=str, usecols=_SUB_COLS, encoding="latin-1")
    sub = sub[sub["form"].isin(["10-K", "10-Q"])]
    sub = sub.dropna(subset=["cik", "period", "fy", "fp", "filed"])
    sub = sub.copy()
    sub["cik"] = sub["cik"].astype(int)
    if ciks is not None:
        sub = sub[sub["cik"].isin(ciks)]
    if sub.empty:
        return empty_facts()

    all_tags = {tag for tags in TAG_PRIORITY.values() for tag in tags}
    wanted_adsh = set(sub["adsh"])
    chunks: list[pd.DataFrame] = []
    # num.txt is ~500MB / ~3.4M rows per quarter: stream in chunks and keep
    # only our filings + tags so the working set stays small.
    with zipfile.ZipFile(zip_path) as zf, zf.open("num.txt") as fh:
        for chunk in pd.read_csv(
            fh, sep="\t", dtype=str, usecols=_NUM_COLS, encoding="latin-1", chunksize=250_000
        ):
            keep = chunk[chunk["adsh"].isin(wanted_adsh) & chunk["tag"].isin(all_tags)]
            if not keep.empty:
                chunks.append(keep)
    if not chunks:
        return empty_facts()
    num = pd.concat(chunks, ignore_index=True)
    for col in ("segments", "coreg"):
        num[col] = num[col].fillna("")
    num = num[(num["segments"] == "") & (num["coreg"] == "") & num["value"].notna()].copy()
    if num.empty:
        return empty_facts()
    num["qtrs"] = num["qtrs"].astype(int)
    num["value"] = num["value"].astype(float)
    num = num.merge(sub, on="adsh", how="inner")

    parts: list[pd.DataFrame] = []
    for concept, tags in TAG_PRIORITY.items():
        uom = UOM_BY_CONCEPT.get(concept, "USD")
        sel = num[num["tag"].isin(tags) & (num["uom"] == uom)].copy()
        if concept == "shares":
            # The dei cover-page share count is dated "as of" a day AFTER the
            # fiscal period end (FSDS rounds it to month end), so requiring
            # ddate == period would drop it: for the dei tag ONLY, accept the
            # LATEST instant dated at-or-after the period end instead (stale
            # comparatives lose). The us-gaap fallback is a normal
            # balance-sheet instant: strict ddate == period (a later-dated
            # instant may belong to the next period's comparative).
            sel = sel[sel["qtrs"] == 0]
            is_dei = sel["tag"] == _DEI_SHARES_TAG
            sel = sel[
                (is_dei & (sel["ddate"] >= sel["period"]))
                | (~is_dei & (sel["ddate"] == sel["period"]))
            ]
        elif concept in INSTANT_CONCEPTS:
            sel = sel[(sel["qtrs"] == 0) & (sel["ddate"] == sel["period"])]
        else:  # flows: the filing's own fiscal period at the form's duration
            sel = sel[sel["ddate"] == sel["period"]]
            sel = sel[sel["qtrs"] == np.where(sel["form"] == "10-K", 4, 1)]
        if sel.empty:
            continue
        sel["_priority"] = sel["tag"].map({t: i for i, t in enumerate(tags)})
        if concept == "shares":
            # Highest-priority tag first, then the LATEST cover date wins.
            sel = sel.sort_values(
                ["adsh", "_priority", "ddate"], ascending=[True, True, False], kind="mergesort"
            )
        else:
            sel = sel.sort_values(["adsh", "_priority"], kind="mergesort")
        sel = sel.drop_duplicates("adsh", keep="first")
        sel["concept"] = concept
        parts.append(sel)
    if not parts:
        return empty_facts()
    facts = pd.concat(parts, ignore_index=True)
    facts["period"] = pd.to_datetime(facts["period"], format="%Y%m%d")
    facts["filed"] = pd.to_datetime(facts["filed"], format="%Y%m%d")
    return facts[FACT_COLUMNS].reset_index(drop=True)
