"""Download SEC Financial Statement Data Set quarterly ZIPs (default 2018q1 ->
the last complete quarter) into data/edgar-raw/ (gitignored via /data/) and
build the per-symbol fundamentals store at [data] fundamentals_dir.

Rerun-safe: ZIPs already on disk are not re-downloaded, and the store is
append-only so reprocessing appends 0 rows. The in-progress quarter has no
ZIP yet (404 -> warn + skip); the weekly companyfacts top-up covers it.

Locked default span is 2018q1 (see plan); TTM needs 4 trailing quarters, so
most metrics stay NaN/neutral until the FY-2018 10-K wave in early 2019.
Pass --from-quarter 2017q1 to fill that warm-up deliberately.

Usage: uv run python scripts/build_cik_map.py            (first, once)
       uv run python scripts/backfill_fundamentals.py [--from-quarter 2018q1]
"""

from __future__ import annotations

import argparse
import datetime
import os
import time
import tomllib
import urllib.error
import urllib.request
from pathlib import Path

from trading.fundamentals.backfill import backfill_quarters, last_complete_quarter, quarter_range
from trading.fundamentals.cik_map import load_cik_map
from trading.fundamentals.edgar import USER_AGENT
from trading.fundamentals.store import FundamentalsStore

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "edgar-raw"
ZIP_URL = "https://www.sec.gov/files/dera/data/financial-statement-data-sets/{quarter}.zip"
REQUEST_SPACING_S = 0.11  # SEC ceiling is 10 req/s; stay under it


def download(quarter: str) -> Path | None:
    dest = RAW_DIR / f"{quarter}.zip"
    if dest.exists():
        return dest
    url = ZIP_URL.format(quarter=quarter)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    print(f"downloading {url}")
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            data = resp.read()
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            print(f"WARNING: {quarter}.zip not published yet; skipping (top-up covers it)")
            return None
        raise
    tmp = dest.with_suffix(".zip.tmp")
    tmp.write_bytes(data)
    os.replace(tmp, dest)
    time.sleep(REQUEST_SPACING_S)
    return dest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-quarter", default="2018q1", help="first quarterly ZIP (YYYYqN)")
    args = parser.parse_args()

    data_cfg = tomllib.loads((ROOT / "config" / "equities.toml").read_text())["data"]
    store = FundamentalsStore(ROOT / data_cfg["fundamentals_dir"])
    cik_map = load_cik_map()
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    quarters = quarter_range(args.from_quarter, last_complete_quarter(datetime.date.today()))
    zips = [path for quarter in quarters if (path := download(quarter)) is not None]
    print(f"parsing {len(zips)} quarterly ZIPs for {len(set(cik_map['cik']))} CIKs ...")
    stats = backfill_quarters(zips, cik_map, store)
    print(
        f"done: {stats['filers']} filers -> {stats['symbols']} symbols, "
        f"{stats['rows']} rows appended"
    )


if __name__ == "__main__":
    main()
