"""Point-in-time fundamentals series (spec: M4 fundamentals overlay).

Quality metric: gross profitability = trailing-4-quarter (Revenue - COGS) /
latest Assets (Novy-Marx). Value primitives: TTM net income (same
trailing-4-quarter mechanics, NaN-independent of COGS), book equity and
shares outstanding as latest-instant values from each filing. The series is
PRICE-FREE by design: ratios (earnings yield, book-to-market) live in the
ranker, so this store never rewrites when price data refreshes.

PIT discipline (non-negotiable):

- A value becomes VISIBLE at its FILING date, never its fiscal-period date
  (2023q1 census: filing lag min 19d / median 44d / max 59d -- a period-date
  join would silently look ahead across that gap).
- Per (cik, fy, fp) only the ORIGINAL filing counts: earliest filed, tie-break
  lowest accession. Later re-filings/restatements never rewrite history
  (amendment forms never even reach this module -- edgar.py accepts exactly
  10-K / 10-Q).
- Incomplete trailing-4-quarter window -> NaN metric. The row is still
  emitted with provenance so "a filing happened, metric unknown" stays
  visible downstream; the ranker reads the LAST row as-of a session (a step
  function on FILED dates), so a NaN latest filing means neutral, never a
  silent reach-back to a stale value.
"""

from __future__ import annotations

import math

import pandas as pd

TTM_QUARTERS = 4
# Four consecutive fiscal quarter-ends span ~273 calendar days; the slack
# absorbs 53-week fiscal years and shifted quarter-ends. Beyond it the
# window has a hole (skipped quarter) and the TTM sum would be wrong.
MAX_TTM_SPAN_DAYS = 330

PROVENANCE_COLUMNS = [
    "adsh",
    "form",
    "fy",
    "fp",
    "period",
    "revenue_tag",
    "cogs_tag",
    "assets_tag",
    "net_income_tag",
    "equity_tag",
    "shares_tag",
]
SERIES_COLUMNS = [
    "gross_profitability",
    "ttm_net_income",
    "book_equity",
    "shares_outstanding",
    "revenue_ttm",
    "cogs_ttm",
    "assets",
    *PROVENANCE_COLUMNS,
]


def empty_series() -> pd.DataFrame:
    frame = pd.DataFrame(columns=SERIES_COLUMNS)
    frame.index = pd.DatetimeIndex([], tz="UTC", name="filed")
    return frame


def compute_pit_series(facts: pd.DataFrame) -> dict[int, pd.DataFrame]:
    """Normalized facts (edgar.FACT_COLUMNS) -> per-cik series indexed by
    tz-aware UTC FILED date. Safe to feed overlapping quarterly files: facts
    are deduped on (adsh, concept) first, then per (cik, fy, fp) only the
    original filing survives."""
    if facts.empty:
        return {}
    facts = facts.drop_duplicates(["adsh", "concept"], keep="first")
    facts = _original_filings(facts)
    out: dict[int, pd.DataFrame] = {}
    for cik, group in facts.groupby("cik"):
        series = _cik_series(group)
        if not series.empty:
            out[int(cik)] = series
    return out


def _original_filings(facts: pd.DataFrame) -> pd.DataFrame:
    """Earliest-filed filing per (cik, fy, fp) wins, tie-break lowest adsh:
    the ORIGINAL filing's values are frozen and later filings for the same
    fiscal period are discarded entirely (PIT: history never rewrites).

    Trust assumption: (fy, fp) keying takes SEC's FSDS fiscal labels at face
    value. A transition period (fiscal-year-end change) can make SEC assign
    two genuinely DIFFERENT reporting periods the same (fy, fp) label; this
    dedup would then keep only the earlier one. We accept that -- the
    alternative (re-deriving fiscal alignment from period dates) invents
    labels SEC did not assign, a worse silent-corruption surface."""
    filings = (
        facts[["cik", "adsh", "fy", "fp", "filed"]]
        .drop_duplicates("adsh")
        .sort_values(["filed", "adsh"], kind="mergesort")
    )
    keep = set(filings.drop_duplicates(["cik", "fy", "fp"], keep="first")["adsh"])
    return facts[facts["adsh"].isin(keep)]


def _derive_q4(fy_total: float, q123: list[dict | None], key: str) -> float:
    if math.isnan(fy_total) or any(q is None or math.isnan(q[key]) for q in q123):
        return math.nan
    return fy_total - sum(q[key] for q in q123)


def _ttm(quarters: dict[tuple[str, str], dict]) -> tuple[float, float, float]:
    """(revenue_ttm, cogs_ttm, net_income_ttm). One shared 4-quarter window;
    each metric's NaN quarters propagate through sum() INDEPENDENTLY, so a
    COGS-less financial still gets a real TTM net income."""
    known = sorted(quarters.values(), key=lambda q: q["period"])
    last4 = known[-TTM_QUARTERS:]
    if len(last4) < TTM_QUARTERS:
        return math.nan, math.nan, math.nan
    if (last4[-1]["period"] - last4[0]["period"]).days > MAX_TTM_SPAN_DAYS:
        return math.nan, math.nan, math.nan  # ragged window: a quarter is missing inside
    return (
        sum(q["revenue"] for q in last4),
        sum(q["cogs"] for q in last4),
        sum(q["net_income"] for q in last4),
    )


def _cik_series(group: pd.DataFrame) -> pd.DataFrame:
    filings: dict[str, dict] = {}
    for row in group.itertuples():
        filing = filings.setdefault(
            row.adsh,
            {
                "adsh": row.adsh,
                "form": row.form,
                "fy": row.fy,
                "fp": row.fp,
                "period": row.period,
                "filed": row.filed,
                "revenue": math.nan,
                "cogs": math.nan,
                "assets": math.nan,
                "net_income": math.nan,
                "equity": math.nan,
                "shares": math.nan,
                "revenue_tag": "",
                "cogs_tag": "",
                "assets_tag": "",
                "net_income_tag": "",
                "equity_tag": "",
                "shares_tag": "",
            },
        )
        filing[row.concept] = row.value
        filing[f"{row.concept}_tag"] = row.tag

    # Same-day dual filings (a backlog filer submitting a prior-quarter 10-Q
    # together with its 10-K) are legitimate: BOTH filings' facts enter the
    # quarter store (they cover different fiscal periods, so there is no true
    # collision) and ONE row is emitted per filed date, computed from the full
    # post-ingest state. Within a day, filings ingest in ascending period
    # order (adsh tie-break) so a prior-quarter 10-Q always lands before the
    # 10-K's Q4 subtraction -- the outcome is deterministic regardless of
    # intra-day input ordering. The provenance schema is single-adsh: the
    # day's row carries the LATEST-period filing of the batch (the 10-K in
    # the dual-filing case), whose instants (assets/equity/shares) are also
    # the freshest balance sheet of the day.
    quarters: dict[tuple[str, str], dict] = {}  # (fy, fp) -> single-quarter values
    ordered = sorted(filings.values(), key=lambda f: (f["filed"], f["period"], f["adsh"]))
    batches: dict[pd.Timestamp, list[dict]] = {}
    for f in ordered:
        batches.setdefault(f["filed"], []).append(f)
    rows: list[dict] = []
    for filed, batch in batches.items():
        for f in batch:
            if f["form"] == "10-Q":
                quarters[(f["fy"], f["fp"])] = {
                    "period": f["period"],
                    "revenue": f["revenue"],
                    "cogs": f["cogs"],
                    "net_income": f["net_income"],
                }
            else:  # 10-K reports the FULL year (qtrs=4): derive Q4 = FY - (Q1+Q2+Q3)
                q123 = [quarters.get((f["fy"], fp)) for fp in ("Q1", "Q2", "Q3")]
                quarters[(f["fy"], "Q4")] = {
                    "period": f["period"],
                    "revenue": _derive_q4(f["revenue"], q123, "revenue"),
                    "cogs": _derive_q4(f["cogs"], q123, "cogs"),
                    "net_income": _derive_q4(f["net_income"], q123, "net_income"),
                }
        f = batch[-1]  # latest period of the day: provenance + instants
        revenue_ttm, cogs_ttm, ttm_net_income = _ttm(quarters)
        assets = f["assets"]
        gp = math.nan
        if (
            not math.isnan(revenue_ttm)
            and not math.isnan(cogs_ttm)
            and assets
            and not math.isnan(assets)
        ):
            gp = (revenue_ttm - cogs_ttm) / assets
        rows.append(
            {
                "filed": filed,
                "gross_profitability": gp,
                "ttm_net_income": ttm_net_income,
                "book_equity": f["equity"],
                "shares_outstanding": f["shares"],
                "revenue_ttm": revenue_ttm,
                "cogs_ttm": cogs_ttm,
                "assets": assets,
                "adsh": f["adsh"],
                "form": f["form"],
                "fy": f["fy"],
                "fp": f["fp"],
                "period": f["period"].date().isoformat(),
                "revenue_tag": f["revenue_tag"],
                "cogs_tag": f["cogs_tag"],
                "assets_tag": f["assets_tag"],
                "net_income_tag": f["net_income_tag"],
                "equity_tag": f["equity_tag"],
                "shares_tag": f["shares_tag"],
            }
        )
    frame = pd.DataFrame(rows).set_index("filed")
    frame.index = frame.index.tz_localize("UTC")
    frame.index.name = "filed"
    return frame.sort_index(kind="mergesort")[SERIES_COLUMNS]
