"""Part A of the clean PEAD test (spec: docs/superpowers/specs/2026-07-11-clean-pead-test.md).

Fetch REAL earnings-announcement dates from SEC EDGAR: for every distinct CIK
in src/trading/fundamentals/cik_map.csv, pull the full submissions history
(filings.recent + paginated filings.files entries), keep 8-K filings whose
`items` string contains "2.02" (Results of Operations and Financial
Condition), restrict to filingDate in 2019-01-01..2023-12-31, dedup same-name
amendments within 5 trading days into one event per quarter per symbol, and
attach the symbol that was active (per cik_map's [start,end) interval) on
that filing date.

Data-only research script. No repo engine changes. Not committed.

Usage:
    .venv/bin/python scripts/research/fetch_earnings_dates.py \
        --out /path/to/earnings_dates.parquet \
        [--limit-ciks N] [--workers 1]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from trading.fundamentals.cik_map import load_cik_map  # noqa: E402

USER_AGENT = "trading-research travis@launchsupply.com"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
FILES_BASE = "https://data.sec.gov/submissions/{name}"
REQUEST_SPACING_S = 0.11  # SEC ceiling is 10 req/s; stay under it
START_DATE = "2019-01-01"
END_DATE = "2023-12-31"
DEDUP_TRADING_DAYS = 5

_last_request_monotonic = 0.0


def _throttled_get(session: requests.Session, url: str, retries: int = 3) -> dict | None:
    global _last_request_monotonic
    for attempt in range(retries):
        wait = REQUEST_SPACING_S - (time.monotonic() - _last_request_monotonic)
        if wait > 0:
            time.sleep(wait)
        try:
            resp = session.get(url, headers={"User-Agent": USER_AGENT}, timeout=30)
        except requests.RequestException as exc:
            _last_request_monotonic = time.monotonic()
            if attempt == retries - 1:
                print(f"WARNING: request failed permanently: {url}: {exc}", file=sys.stderr)
                return None
            continue
        _last_request_monotonic = time.monotonic()
        if resp.status_code == 404:
            return None
        if resp.status_code == 429:
            time.sleep(1.0)
            continue
        if resp.status_code != 200:
            print(f"WARNING: HTTP {resp.status_code} for {url}", file=sys.stderr)
            return None
        try:
            return resp.json()
        except json.JSONDecodeError:
            print(f"WARNING: bad JSON for {url}", file=sys.stderr)
            return None
    return None


def _extract_8k_202(columnar: dict, cik: int, source: str) -> list[dict]:
    """columnar submissions block (either filings.recent or a filings.files
    page) -> list of 8-K/Item-2.02 events with filingDate in range."""
    forms = columnar.get("form", [])
    items = columnar.get("items", [])
    filing_dates = columnar.get("filingDate", [])
    accessions = columnar.get("accessionNumber", [])
    out = []
    n = len(forms)
    for i in range(n):
        form = forms[i]
        if form != "8-K":
            continue
        item_str = items[i] if i < len(items) else ""
        if not item_str or "2.02" not in item_str:
            continue
        fdate = filing_dates[i] if i < len(filing_dates) else None
        if fdate is None or not (START_DATE <= fdate <= END_DATE):
            continue
        out.append(
            {
                "cik": cik,
                "filingDate": fdate,
                "items": item_str,
                "accessionNumber": accessions[i] if i < len(accessions) else None,
                "source": source,
            }
        )
    return out


def fetch_events_for_cik(session: requests.Session, cik: int) -> tuple[list[dict], str | None]:
    """Returns (events, error). error is None on success (even if 0 events)."""
    payload = _throttled_get(session, SUBMISSIONS_URL.format(cik=cik))
    if payload is None:
        return [], "submissions fetch failed (404 or error)"
    events: list[dict] = []
    filings = payload.get("filings", {})
    recent = filings.get("recent", {})
    events.extend(_extract_8k_202(recent, cik, "recent"))

    # CRITICAL PAGINATION: filings.recent holds only the last ~1000 filings;
    # older history (needed for 2019-2023) lives in filings.files, an array
    # of {name, filingCount, filingFrom, filingTo} pointing at additional
    # columnar JSON pages with the SAME schema as filings.recent.
    for file_entry in filings.get("files", []):
        name = file_entry.get("name")
        if not name:
            continue
        # Skip pages entirely outside our date window when the index gives
        # us a range (cheap prefilter; still validate per-row on fetch).
        f_from = file_entry.get("filingFrom")
        f_to = file_entry.get("filingTo")
        if f_to and f_to < START_DATE:
            continue
        if f_from and f_from > END_DATE:
            continue
        page = _throttled_get(session, FILES_BASE.format(name=name))
        if page is None:
            continue
        events.extend(_extract_8k_202(page, cik, name))
    return events, None


def _symbol_for_cik_date(cik_rows: pd.DataFrame, cik: int, date: str) -> str | None:
    rows = cik_rows[cik_rows["cik"] == cik]
    for _, row in rows.iterrows():
        if row["start"] <= date and (row["end"] == "" or date < row["end"]):
            return row["symbol"]
    return None


def dedup_events(events: pd.DataFrame, trading_days: pd.DatetimeIndex) -> pd.DataFrame:
    """Collapse 8-K/2.02 filings within DEDUP_TRADING_DAYS trading days of
    each other (per symbol) into one event -- keeps the EARLIEST filing
    (the original release; later ones are typically amendments/corrections)."""
    if events.empty:
        return events
    # Map each filingDate to its trading-day integer position (searchsorted
    # against the SPY calendar) so "5 trading days apart" is measured on the
    # actual market calendar, not calendar days.
    positions = trading_days.searchsorted(pd.to_datetime(events["filingDate"]))
    events = events.assign(_pos=positions).sort_values(["symbol", "filingDate"])
    keep_rows = []
    last_symbol = None
    last_pos = None
    for _, row in events.iterrows():
        if row["symbol"] != last_symbol or last_pos is None or row["_pos"] - last_pos > DEDUP_TRADING_DAYS:
            keep_rows.append(row)
            last_symbol = row["symbol"]
            last_pos = row["_pos"]
        else:
            # within window of the kept event: treat as amendment, drop.
            last_symbol = row["symbol"]
            # keep last_pos anchored to the FIRST kept event of the cluster
    kept = pd.DataFrame(keep_rows).drop(columns="_pos")
    return kept.reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit-ciks", type=int, default=None, help="debug: only fetch first N CIKs")
    ap.add_argument(
        "--spy-parquet",
        type=Path,
        default=ROOT / "data" / "equities-tiingo" / "SPY.parquet",
        help="trading-day calendar source",
    )
    args = ap.parse_args()

    cik_map = load_cik_map()
    distinct_ciks = sorted(cik_map["cik"].unique().tolist())
    if args.limit_ciks:
        distinct_ciks = distinct_ciks[: args.limit_ciks]
    print(f"{len(distinct_ciks)} distinct CIKs to fetch", file=sys.stderr)

    spy = pd.read_parquet(args.spy_parquet)
    trading_days = pd.DatetimeIndex(spy.index.tz_localize(None) if spy.index.tz else spy.index)

    session = requests.Session()
    all_events: list[dict] = []
    failures: list[int] = []
    t_start = time.monotonic()
    for i, cik in enumerate(distinct_ciks):
        events, err = fetch_events_for_cik(session, cik)
        if err:
            failures.append(cik)
        all_events.extend(events)
        if (i + 1) % 50 == 0 or (i + 1) == len(distinct_ciks):
            elapsed = time.monotonic() - t_start
            print(
                f"[{i + 1}/{len(distinct_ciks)}] {len(all_events)} raw events so far, "
                f"{len(failures)} failures, {elapsed:.0f}s elapsed",
                file=sys.stderr,
            )

    print(f"Total raw 8-K/2.02 events (pre-dedup, pre-symbol-attach): {len(all_events)}", file=sys.stderr)
    print(f"CIK fetch failures: {len(failures)} -> {failures[:20]}", file=sys.stderr)

    events_df = pd.DataFrame(all_events)
    if events_df.empty:
        raise SystemExit("ERROR: zero events fetched -- something is broken upstream")

    events_df["symbol"] = events_df.apply(
        lambda r: _symbol_for_cik_date(cik_map, r["cik"], r["filingDate"]), axis=1
    )
    unmapped = events_df["symbol"].isna().sum()
    print(f"Events with no symbol mapped for their filing date: {unmapped}", file=sys.stderr)
    events_df = events_df.dropna(subset=["symbol"])

    deduped = dedup_events(events_df, trading_days)
    print(f"Events after 5-trading-day dedup: {len(deduped)}", file=sys.stderr)

    per_name = deduped.groupby("symbol").size()
    print(
        f"Names with >=1 event: {per_name.shape[0]}; median events/name: {per_name.median()}; "
        f"mean: {per_name.mean():.2f}",
        file=sys.stderr,
    )

    out = deduped.rename(columns={"filingDate": "earnings_date"})[
        ["symbol", "cik", "earnings_date", "accessionNumber", "items", "source"]
    ].sort_values(["symbol", "earnings_date"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out, index=False)
    out.to_csv(args.out.with_suffix(".csv"), index=False)
    print(f"Wrote {len(out)} events to {args.out}", file=sys.stderr)

    # Spot-check known names.
    for sym in ["AAPL", "MSFT", "NVDA"]:
        sub = out[out["symbol"] == sym]
        print(f"--- {sym} ({len(sub)} events) ---", file=sys.stderr)
        print(sub[["earnings_date"]].to_string(index=False), file=sys.stderr)


if __name__ == "__main__":
    main()
