"""Current-quarter fundamentals top-up from data.sec.gov companyfacts (M4).

The quarterly ZIPs trail by a full quarter; companyfacts serves each filer's
complete XBRL history the day after it files. Entries are normalized to the
SAME facts table edgar.py produces and pushed through the SAME
compute_pit_series, so backfill and top-up cannot diverge -- and the
append-only store guarantees already-visible history is never rewritten by a
top-up (PIT: the earliest write for a filed date wins forever).

Fail-open (the earnings pattern): a per-symbol failure degrades -- that
symbol keeps its stored (possibly stale) values, the run continues, and the
caller journals the degraded flag as a warning. Never crash a run over
fundamentals.
"""

from __future__ import annotations

import datetime
import json
import time
import urllib.request
from collections.abc import Callable, Iterable

import pandas as pd

from trading.fundamentals.cik_map import cik_for, interval_slice
from trading.fundamentals.edgar import (
    FACT_COLUMNS,
    INSTANT_CONCEPTS,
    TAG_PRIORITY,
    UOM_BY_CONCEPT,
    USER_AGENT,
    empty_facts,
)
from trading.fundamentals.metrics import compute_pit_series, empty_series
from trading.fundamentals.store import FundamentalsStore

COMPANYFACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
_MIN_REQUEST_INTERVAL_S = 0.11  # SEC ceiling is 10 req/s; stay under it
# Duration windows for classifying a fact's period length from (start, end):
# a fiscal quarter is ~90-98 days, a fiscal year 357-371 (53-week years).
_QUARTER_DAYS = (80, 100)
_YEAR_DAYS = (350, 380)
# companyfacts nests facts by taxonomy: the cover-page share count lives
# under "dei", everything else under "us-gaap".
_TAXONOMY_BY_TAG = {"EntityCommonStockSharesOutstanding": "dei"}

_last_request_monotonic = 0.0


def _http_get_json(url: str) -> dict:
    """Network touchpoint, isolated for monkeypatching; throttled + UA per SEC policy."""
    global _last_request_monotonic
    wait = _MIN_REQUEST_INTERVAL_S - (time.monotonic() - _last_request_monotonic)
    if wait > 0:
        time.sleep(wait)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    _last_request_monotonic = time.monotonic()
    return payload


def facts_from_companyfacts(payload: dict, cik: int) -> pd.DataFrame:
    """companyfacts JSON -> the normalized facts table (edgar.FACT_COLUMNS)."""
    taxonomies = payload.get("facts", {})
    records: list[dict] = []
    for concept, tags in TAG_PRIORITY.items():
        unit = UOM_BY_CONCEPT.get(concept, "USD")
        for priority, tag in enumerate(tags):
            taxonomy = taxonomies.get(_TAXONOMY_BY_TAG.get(tag, "us-gaap"), {})
            for entry in taxonomy.get(tag, {}).get("units", {}).get(unit, []):
                if entry.get("form") not in ("10-K", "10-Q"):
                    continue  # amendments and other forms never parse (PIT)
                if entry.get("fy") is None or not entry.get("fp"):
                    continue
                end = pd.Timestamp(entry["end"])
                if concept in INSTANT_CONCEPTS:
                    qtrs = 0
                else:  # flows: revenue, cogs, net income
                    if "start" not in entry:
                        continue
                    days = (end - pd.Timestamp(entry["start"])).days + 1
                    if _QUARTER_DAYS[0] <= days <= _QUARTER_DAYS[1]:
                        qtrs = 1
                    elif _YEAR_DAYS[0] <= days <= _YEAR_DAYS[1]:
                        qtrs = 4
                    else:
                        continue  # YTD and other durations are not single quarters/years
                    if qtrs != (4 if entry["form"] == "10-K" else 1):
                        continue
                records.append(
                    {
                        "cik": cik,
                        "adsh": entry["accn"],
                        "form": entry["form"],
                        "fy": str(entry["fy"]),
                        "fp": "FY" if entry["form"] == "10-K" else entry["fp"],
                        "period": end,
                        "filed": pd.Timestamp(entry["filed"]),
                        "concept": concept,
                        "tag": tag,
                        "qtrs": qtrs,
                        "value": float(entry["val"]),
                        "_priority": priority,
                    }
                )
    if not records:
        return empty_facts()
    facts = pd.DataFrame.from_records(records)
    # companyfacts carries COMPARATIVE periods (a 10-K re-reports prior
    # years); keep only each filing's OWN fiscal period -- the latest period
    # end among its NON-shares facts (the ZIP path's ddate == sub.period
    # twin). The cover-page share count is dated AFTER the period end, so it
    # must neither define the filing's own period nor be dropped by it.
    core = facts[facts["concept"] != "shares"].copy()
    if core.empty:
        return empty_facts()
    own_period = core.groupby("adsh")["period"].max()
    core = core[core["period"] == core["adsh"].map(own_period)]
    core = core.sort_values(["adsh", "concept", "_priority"], kind="mergesort")
    core = core.drop_duplicates(["adsh", "concept"], keep="first")
    shares = facts[facts["concept"] == "shares"].copy()
    shares = shares[shares["adsh"].isin(own_period.index)]
    shares = shares[shares["period"] >= shares["adsh"].map(own_period)]
    if shares.empty:
        facts = core
    else:
        # Highest-priority tag, then the LATEST cover date wins; the row is
        # re-dated to the filing's own period so the schema matches the ZIP path.
        shares = shares.sort_values(
            ["adsh", "_priority", "period"], ascending=[True, True, False], kind="mergesort"
        ).drop_duplicates("adsh", keep="first")
        shares["period"] = shares["adsh"].map(own_period)
        facts = pd.concat([core, shares], ignore_index=True)
    return facts[FACT_COLUMNS].reset_index(drop=True)


def refresh_fundamentals(
    store: FundamentalsStore,
    cik_map: pd.DataFrame,
    symbols: Iterable[str],
    as_of: datetime.date,
    fetch_json: Callable[[str], dict] = _http_get_json,
) -> tuple[int, bool]:
    """Top up `symbols` from companyfacts. Returns (rows appended, degraded).
    Unmapped symbols are skipped silently (no EDGAR ticker -> neutral rank);
    per-symbol failures set degraded and never raise."""
    appended = 0
    degraded = False
    series_by_cik: dict[int, pd.DataFrame] = {}
    for symbol in symbols:
        cik = cik_for(cik_map, symbol, as_of)
        if cik is None:
            continue
        try:
            if cik not in series_by_cik:  # GOOG/GOOGL share one CIK: fetch once
                payload = fetch_json(COMPANYFACTS_URL.format(cik=cik))
                facts = facts_from_companyfacts(payload, cik)
                series_by_cik[cik] = compute_pit_series(facts).get(cik, empty_series())
            rows = cik_map[(cik_map["symbol"] == symbol) & (cik_map["cik"] == cik)]
            for row in rows.itertuples():
                appended += store.append(
                    symbol, interval_slice(series_by_cik[cik], row.start, row.end)
                )
        except Exception:
            degraded = True  # fail-open: stored values serve this run (earnings pattern)
    return appended, degraded
