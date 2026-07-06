"""Build the per-symbol fundamentals store at [data] fundamentals_dir.

Two sources, selected by --source (default: companyfacts):

- companyfacts (PRIMARY): one companyfacts fetch per unique cik_map.csv cik
  -- 100% shares_outstanding coverage for names the ZIP path misses entirely
  (JPM, META, BRK-B), because the FSDS ZIPs strip nearly all dei cover-page
  facts (a census found dei:EntityCommonStockSharesOutstanding on 1 of 5631
  ZIP filings). No download step: data.sec.gov serves each filer's full
  history directly. Since a source-switching rebuild would otherwise mix
  regimes silently (append-only history never rewrites, so old ZIP-sourced
  rows with NaN shares would survive under a symbol that also gets NEW
  companyfacts-sourced rows for filed dates the ZIPs never covered), this
  path REFUSES to run against a non-empty store -- delete
  [data] fundamentals_dir's contents first.
- zips (RETIRED-PRIMARY): downloads 2018q1 -> the last complete quarter's
  Financial Statement Data Set ZIPs into data/edgar-raw/ (gitignored via
  /data/) and parses them. Rerun-safe (ZIPs already on disk are not
  re-downloaded, and the store is append-only so reprocessing appends 0
  rows); kept for bulk revenue/COGS/assets/net-income/equity coverage
  without a per-cik network round trip, and because it needs no live
  network access to rerun once the ZIPs are cached. The in-progress quarter
  has no ZIP yet (404 on the NEWEST quarter only -> warn + skip); the
  weekly companyfacts top-up covers it regardless of which source built the
  store. A 404 on any OLDER quarter -- one SEC has definitely published --
  means a broken URL, so it aborts nonzero instead of silently producing an
  incomplete backfill. Locked default span is 2018q1 (see plan); TTM needs
  4 trailing quarters, so most metrics stay NaN/neutral until the FY-2018
  10-K wave in early 2019. Pass --from-quarter 2017q1 to fill that warm-up
  deliberately. Unlike companyfacts, this path has no empty-store trigger of
  its own (it is deliberately rerun-safe against whatever store already
  exists) -- so the reverse mixing direction (running --source zips on top
  of a companyfacts-built store, silently appending NaN-shares rows for any
  CIK the companyfacts run's per-cik fetch failed on) is instead guarded by
  a [data] fundamentals_dir/.source marker file: it REFUSES (naming the
  marker) when the marker says the store was built by companyfacts.

Usage: uv run python scripts/build_cik_map.py                     (first, once)
       uv run python scripts/backfill_fundamentals.py              (companyfacts, default)
       uv run python scripts/backfill_fundamentals.py --source zips [--from-quarter 2018q1]
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

from trading.fundamentals.backfill import (
    backfill_from_companyfacts,
    backfill_quarters,
    last_complete_quarter,
    quarter_range,
)
from trading.fundamentals.cik_map import load_cik_map
from trading.fundamentals.edgar import USER_AGENT
from trading.fundamentals.store import FundamentalsStore

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "edgar-raw"
ZIP_URL = "https://www.sec.gov/files/dera/data/financial-statement-data-sets/{quarter}.zip"
REQUEST_SPACING_S = 0.11  # SEC ceiling is 10 req/s; stay under it
PROGRESS_EVERY = 50  # print a progress line every N cik fetches (companyfacts source)
SOURCE_MARKER = ".source"  # records which --source built [data] fundamentals_dir


def download(quarter: str, allow_missing: bool = False) -> Path | None:
    """Fetch one quarterly ZIP into RAW_DIR (cached: on-disk files are never
    re-downloaded). A 404 is legitimate ONLY for the newest quarter (SEC
    publication lag; allow_missing=True): warn and skip. Any other 404 -- an
    old, definitely-published quarter -- or any non-404 HTTP error means a
    broken URL or server problem: abort nonzero with the URL rather than
    silently producing an incomplete backfill."""
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
        if exc.code == 404 and allow_missing:
            print(f"WARNING: {quarter}.zip not published yet; skipping (top-up covers it)")
            return None
        raise SystemExit(f"ERROR: HTTP {exc.code} downloading {url}") from exc
    tmp = dest.with_suffix(".zip.tmp")
    tmp.write_bytes(data)
    os.replace(tmp, dest)
    time.sleep(REQUEST_SPACING_S)
    return dest


def download_range(quarters: list[str]) -> list[Path]:
    """Download every quarter in order; only the NEWEST (last) quarter in the
    range may be missing upstream (publication lag) and skip."""
    zips: list[Path] = []
    for quarter in quarters:
        path = download(quarter, allow_missing=quarter == quarters[-1])
        if path is not None:
            zips.append(path)
    return zips


def _ensure_empty_for_rebuild(store_root: Path) -> None:
    """The companyfacts rebuild REQUIRES an empty store: append-only
    semantics mean any rows already on disk for a filed date are never
    replaced, so rebuilding on top of an existing (e.g. ZIP-sourced, partial
    shares coverage) store would silently keep the OLD values for every
    filed date the old source also covered instead of the new source's
    (better) ones. Abort loudly instead of producing a store that quietly
    mixes two regimes."""
    existing = sorted(store_root.glob("*.parquet"))
    if existing:
        raise SystemExit(
            f"ERROR: {store_root} already has {len(existing)} symbol file(s). The "
            "companyfacts rebuild (--source companyfacts, the default) must start "
            "from an EMPTY store -- append-only semantics would silently keep any "
            "stale/incomplete rows already there (e.g. NaN shares_outstanding from "
            "a prior ZIP-based backfill) instead of the new source's values. Delete "
            "the directory's contents first, then rerun."
        )


def _read_source_marker(store_root: Path) -> str | None:
    """Which `--source` last (re)built `store_root`, if any run has recorded
    one yet (older stores, or a store wiped before this guard existed, have
    no marker)."""
    path = store_root / SOURCE_MARKER
    if not path.exists():
        return None
    return path.read_text().strip()


def _write_source_marker(store_root: Path, source: str) -> None:
    """Record which `--source` built `store_root`. Atomic tmp+os.replace,
    same pattern as FundamentalsStore.mark_refreshed: always overwrite rather
    than check-then-skip, so a companyfacts rebuild (which _ensure_empty_
    for_rebuild has just guaranteed starts from a truly empty store) always
    leaves the marker correctly saying "companyfacts", even if a stale marker
    from a prior source happened to survive alongside deleted *.parquet
    files."""
    path = store_root / SOURCE_MARKER
    tmp = path.with_suffix(".tmp")
    tmp.write_text(source)
    os.replace(tmp, path)


def _ensure_zips_allowed(store_root: Path) -> None:
    """The mirror image of _ensure_empty_for_rebuild: --source zips has no
    empty-store trigger of its own (it is deliberately rerun-safe against
    whatever store already exists), so it relies on the .source marker
    instead -- refuse loudly if a companyfacts rebuild already owns this
    store, rather than silently appending NaN shares_outstanding rows for
    any CIK that rebuild's per-cik fetch failed on."""
    marker = _read_source_marker(store_root)
    if marker == "companyfacts":
        raise SystemExit(
            f"ERROR: {store_root} was built with --source companyfacts (its "
            f"{SOURCE_MARKER} marker file says so). Running --source zips on top "
            "would silently append NaN shares_outstanding rows for any CIK the "
            "companyfacts run's per-cik fetch failed on, mixing source regimes. "
            "Delete the directory's contents (including the marker) first, then "
            "rerun with --source zips if that is really what you want."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        choices=["companyfacts", "zips"],
        default="companyfacts",
        help="companyfacts (default; primary, 100%% shares coverage) or zips "
        "(retired-primary, tested but incomplete shares coverage)",
    )
    parser.add_argument(
        "--from-quarter", default="2018q1", help="first quarterly ZIP (YYYYqN); --source zips only"
    )
    args = parser.parse_args()

    data_cfg = tomllib.loads((ROOT / "config" / "equities.toml").read_text())["data"]
    store_root = ROOT / data_cfg["fundamentals_dir"]
    cik_map = load_cik_map()

    if args.source == "companyfacts":
        _ensure_empty_for_rebuild(store_root)
        store = FundamentalsStore(store_root)  # creates store_root if needed
        # Written BEFORE the per-CIK loop starts, not after it completes: the
        # marker's job is guarding against a --source zips run mixing
        # regimes on top of THIS store, and that risk exists the moment this
        # rebuild starts writing rows, not only once it finishes. An
        # interrupt partway through the loop below must still leave the
        # store guarded -- see test_interrupted_companyfacts_backfill_
        # still_leaves_marker_guarding_zips.
        _write_source_marker(store_root, "companyfacts")
        n_ciks = len(set(cik_map["cik"]))
        print(f"backfilling {n_ciks} CIKs from companyfacts (primary source) ...")

        def progress(done: int, total: int) -> None:
            if done % PROGRESS_EVERY == 0 or done == total:
                print(f"  {done}/{total} CIKs fetched")

        stats = backfill_from_companyfacts(cik_map, store, on_progress=progress)
        print(
            f"done: {stats['filers']} filers -> {stats['symbols']} symbols, "
            f"{stats['rows']} rows appended ({stats['dropped']} rows outside every symbol "
            f"interval, {stats['failed']} CIK fetch failures)"
        )
    else:
        _ensure_zips_allowed(store_root)
        store = FundamentalsStore(store_root)
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        quarters = quarter_range(args.from_quarter, last_complete_quarter(datetime.date.today()))
        zips = download_range(quarters)
        print(f"parsing {len(zips)} quarterly ZIPs for {len(set(cik_map['cik']))} CIKs ...")
        stats = backfill_quarters(zips, cik_map, store)
        _write_source_marker(store_root, "zips")
        print(
            f"done: {stats['filers']} filers -> {stats['symbols']} symbols, "
            f"{stats['rows']} rows appended ({stats['dropped']} rows outside every symbol interval)"
        )


if __name__ == "__main__":
    main()
