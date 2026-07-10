"""Build the per-symbol Form 4 insider-transaction store at
data/insider/equities/ (spec: 2026-07-09 insider pipeline, sections 2/4/5).

Downloads the SEC DERA "Insider Transactions Data Sets" quarterly ZIPs
2018q3 -> the last complete quarter (the 2018q3 start gives trailing-90d
windows full coverage from 2019-01) into data/edgar-insider-raw/ (gitignored
scratch; cached -- reruns never re-fetch), parses each with
trading.fundamentals.form345.parse_quarter (open-market P/S rows only), maps
ISSUERCIK -> symbol through the committed cik_map.csv FILED-date intervals
(unmapped ciks are COUNTED, never guessed), and regenerates the store WHOLE:
one parquet per symbol (filed index, form345.INSIDER_COLUMNS), atomic
tmp+os.replace writes, and a .source marker ("form345") like the bar caches.

Error handling (spec section 5): a quarter whose download or parse fails is
a NAMED coverage GAP -- the build continues, the gap is printed loudly in
the final report, the exit code is 1 so the orchestrator cannot miss it, and
the .source marker becomes "form345 GAPS:<quarters>" so the incomplete build
is detectable from disk after the launch log is gone.
The NEWEST quarter 404ing is publication lag, not a gap. The store must be
empty before a rebuild (whole-regeneration semantics: a rebuild on top of an
older build could leave stale symbols behind) -- delete its contents first.

Coverage report (spec section 4): per-quarter row/P/S/skipped counts,
per-year row counts (a missing quarter shows as a hole), unmapped row/cik
counts, and the window-membership coverage rate (symbols with >= 1 stored
row / membership symbols overlapping the discovery window; quiet companies
legitimately lack rows, so this is context, not a gate).

Usage: uv run python scripts/build_insider_store.py
(~31 quarterly ZIPs, a few MB each; minutes, throttled per SEC fair access)
"""

from __future__ import annotations

import datetime
import os
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from trading.fundamentals.backfill import last_complete_quarter, quarter_range
from trading.fundamentals.cik_map import load_cik_map
from trading.fundamentals.edgar import USER_AGENT
from trading.fundamentals.form345 import INSIDER_COLUMNS, parse_quarter

sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_sic_map import WINDOW_END, WINDOW_START  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "edgar-insider-raw"
STORE_DIR = ROOT / "data" / "insider" / "equities"
MEMBERSHIP = ROOT / "src" / "trading" / "venues" / "universes" / "equities_membership.csv"
ZIP_URL = (
    "https://www.sec.gov/files/structureddata/data/"
    "insider-transactions-data-sets/{quarter}_form345.zip"
)
FIRST_QUARTER = "2018q3"  # spec section 2: trailing-90d windows full from 2019-01
REQUEST_SPACING_S = 0.11  # SEC ceiling is 10 req/s; stay under it
SOURCE_MARKER = ".source"


def download(quarter: str, newest: bool) -> Path | None:
    """Fetch one quarterly ZIP into RAW_DIR (cached: on-disk files are never
    re-downloaded). Returns None on a 404 of the NEWEST quarter only
    (publication lag -- normal, not a gap). Any other failure raises and
    main() records it as a coverage gap."""
    dest = RAW_DIR / f"{quarter}_form345.zip"
    if dest.exists():
        return dest
    url = ZIP_URL.format(quarter=quarter)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    print(f"downloading {url}")
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            data = resp.read()
    except urllib.error.HTTPError as exc:
        if exc.code == 404 and newest:
            print(f"WARNING: {quarter}_form345.zip not published yet; skipping")
            return None
        raise
    tmp = dest.with_suffix(".zip.tmp")
    tmp.write_bytes(data)
    os.replace(tmp, dest)
    time.sleep(REQUEST_SPACING_S)
    return dest


def gap_line(quarter: str, exc: Exception) -> str:
    return f"{quarter}: {type(exc).__name__}: {exc}"


def map_to_symbols(
    transactions: pd.DataFrame, cik_map: pd.DataFrame
) -> tuple[dict[str, pd.DataFrame], int, int]:
    """Slice each issuer cik's rows into its cik_map symbol intervals by
    FILED date (the fundamentals rule: a renamed company's pre-rename filings
    land on the old symbol). Returns (per-symbol filed-indexed frames,
    unmapped row count, ciks with >= 1 unmapped row). A row can map to
    SEVERAL symbols (GOOG/GOOGL share one cik) -- both get it, like the
    fundamentals backfill; it is unmapped only when NO interval covers its
    (cik, filed)."""
    indexed = transactions.set_index("filed").sort_index(kind="mergesort")
    by_cik = {int(cik): frame for cik, frame in indexed.groupby("issuer_cik")}
    covered = {cik: np.zeros(len(frame), dtype=bool) for cik, frame in by_cik.items()}
    pieces: dict[str, list[pd.DataFrame]] = {}
    for row in cik_map.itertuples():
        frame = by_cik.get(int(row.cik))
        if frame is None:
            continue
        # A DatetimeIndex comparison already yields an ndarray[bool].
        mask = np.asarray(frame.index >= pd.Timestamp(row.start, tz="UTC"))
        if row.end:
            mask &= np.asarray(frame.index < pd.Timestamp(row.end, tz="UTC"))
        if mask.any():
            pieces.setdefault(row.symbol, []).append(frame[mask])
            covered[int(row.cik)] |= mask
    frames = {
        symbol: pd.concat(parts).sort_index(kind="mergesort")[INSIDER_COLUMNS]
        for symbol, parts in pieces.items()
    }
    unmapped_rows = sum(int((~flags).sum()) for flags in covered.values())
    unmapped_ciks = sum(1 for flags in covered.values() if not flags.all())
    return frames, unmapped_rows, unmapped_ciks


def write_store(
    frames: dict[str, pd.DataFrame],
    store_root: Path,
    gap_quarters: list[str] | None = None,
) -> None:
    """Whole-store write: per-symbol parquet, atomic tmp+os.replace, then the
    .source marker (the bar-cache convention). A gapped build stamps
    "form345 GAPS:<quarters>" instead of plain "form345": incomplete builds
    must be detectable from DISK, not just the launch log's exit code (the
    load path warns on a GAPS marker)."""
    store_root.mkdir(parents=True, exist_ok=True)
    for symbol in sorted(frames):
        path = store_root / f"{symbol.replace('/', '-')}.parquet"
        tmp = path.with_suffix(".parquet.tmp")
        frames[symbol].to_parquet(tmp)
        os.replace(tmp, path)
    text = "form345"
    if gap_quarters:
        text += " GAPS:" + ",".join(gap_quarters)
    marker = store_root / SOURCE_MARKER
    tmp = marker.with_suffix(".tmp")
    tmp.write_text(text)
    os.replace(tmp, marker)


def ensure_empty(store_root: Path) -> None:
    """The store is regenerated WHOLE: refuse a rebuild on top of an existing
    store (stale symbols from an older cik_map could silently survive)."""
    existing = sorted(store_root.glob("*.parquet"))
    if existing:
        raise SystemExit(
            f"ERROR: {store_root} already has {len(existing)} symbol file(s). "
            "The insider store is regenerated whole; a rebuild on top could "
            "leave stale symbols behind. Delete the directory's contents "
            f"(including {SOURCE_MARKER}) first, then rerun."
        )


def window_members(membership_path: Path) -> set[str]:
    """Membership symbols whose interval overlaps the discovery window.
    An empty result means a broken membership CSV, not a real universe:
    refuse loudly rather than divide by zero in the coverage report."""
    df = pd.read_csv(membership_path, comment="#", dtype=str).fillna("")
    overlap = (df["start"] <= WINDOW_END) & ((df["end"] == "") | (df["end"] > WINDOW_START))
    members = set(df.loc[overlap, "symbol"])
    if not members:
        raise SystemExit(
            f"ERROR: no membership symbol in {membership_path} overlaps the "
            f"discovery window {WINDOW_START}..{WINDOW_END}; the membership "
            "CSV is broken or empty (the coverage denominator would be zero)."
        )
    return members


def main() -> None:
    ensure_empty(STORE_DIR)
    cik_map = load_cik_map()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    quarters = quarter_range(FIRST_QUARTER, last_complete_quarter(datetime.date.today()))

    quarter_frames: list[pd.DataFrame] = []
    gaps: list[str] = []
    gap_quarters: list[str] = []
    total_skipped = 0
    for quarter in quarters:
        try:
            zip_path = download(quarter, newest=quarter == quarters[-1])
            if zip_path is None:
                continue  # newest not published yet: lag, not a gap
            tx, skipped = parse_quarter(zip_path)
        except (OSError, zipfile.BadZipFile, ValueError, KeyError) as exc:
            gaps.append(gap_line(quarter, exc))
            gap_quarters.append(quarter)
            print(f"ERROR: {quarter} failed ({exc}); continuing -- GAP RECORDED")
            continue
        total_skipped += skipped
        n_p = int((tx["code"] == "P").sum())
        print(f"parsed {quarter}: {len(tx)} P/S rows ({n_p} P / {len(tx) - n_p} S), "
              f"{skipped} skipped")
        if not tx.empty:  # never concat an empty frame (warnings-as-errors)
            quarter_frames.append(tx)
    if not quarter_frames:
        sys.exit("FATAL: no quarter parsed; nothing written")

    transactions = pd.concat(quarter_frames, ignore_index=True)
    frames, unmapped_rows, unmapped_ciks = map_to_symbols(transactions, cik_map)
    write_store(frames, STORE_DIR, gap_quarters)

    # ---- coverage report (spec section 4) ----
    n_p = int((transactions["code"] == "P").sum())
    mapped_rows = len(transactions) - unmapped_rows
    members = window_members(MEMBERSHIP)
    with_rows = members & set(frames)
    print(f"\nrows parsed: {len(transactions)} ({n_p} P / {len(transactions) - n_p} S); "
          f"{total_skipped} unparseable rows skipped")
    print(f"mapped to >=1 symbol: {mapped_rows} rows -> {len(frames)} symbols; "
          f"unmapped: {unmapped_rows} rows across {unmapped_ciks} ciks "
          "(non-members + the known ~2% cik_map residual; never guessed)")
    print(f"window-membership coverage: {len(with_rows)}/{len(members)} "
          f"({len(with_rows) / len(members):.1%}) members with >= 1 insider row "
          "(quiet companies legitimately lack rows)")
    print("\nrows per FILED year (a missing quarter shows as a hole):")
    per_year = transactions["filed"].dt.year.value_counts().sort_index()
    for year, count in per_year.items():
        print(f"  {year}: {count}")
    if gaps:
        print("\nCOVERAGE GAPS (spec section 5: loud, never silent):")
        for gap in gaps:
            print(f"  {gap}")
        sys.exit(1)
    print(f"\nwrote {len(frames)} symbol parquets to {STORE_DIR}")


if __name__ == "__main__":
    main()
