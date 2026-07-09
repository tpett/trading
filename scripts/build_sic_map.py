"""Build the committed symbol -> SIC classification map
(src/trading/venues/universes/sic_map.csv) for the Piece 2 segment universes.

Source: https://data.sec.gov/submissions/CIK##########.json `sic` +
`sicDescription` -- each filer's CURRENT code, applied backward over the
discovery window (a disclosed Piece 2 caveat) -- joined through the committed
cik_map.csv. A symbol with multiple CIK intervals uses the interval
overlapping the discovery window 2019-01-01..2023-12-31. Mirrors
scripts/build_cik_map.py's conventions: stdlib urllib only, the companyfacts
throttle seam (process-global 0.11 s + mandatory User-Agent), self-validating
main() that refuses to emit a bad map.

A symbol whose fetch fails twice (retry-once, spec section 6) or whose filer
carries no SIC is recorded unmapped and belongs to NO segment -- never
guessed. main() prints the coverage report and exits non-zero below 90% of
window membership. Update
src/trading/venues/universes/sources/PROVENANCE.md on every regeneration.

Usage: uv run python scripts/build_sic_map.py   (~1130 requests, ~2-3 min)
"""

from __future__ import annotations

import csv
import datetime
import sys
from collections.abc import Callable
from pathlib import Path

import pandas as pd

from trading.fundamentals.companyfacts import http_get_json

ROOT = Path(__file__).resolve().parent.parent
CIK_MAP = ROOT / "src" / "trading" / "fundamentals" / "cik_map.csv"
MEMBERSHIP = ROOT / "src" / "trading" / "venues" / "universes" / "equities_membership.csv"
OUTPUT = ROOT / "src" / "trading" / "venues" / "universes" / "sic_map.csv"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
# The pre-registered discovery window (sweep.DISCOVERY_WINDOW); intervals are
# start-inclusive, end-exclusive, empty end = current.
WINDOW_START, WINDOW_END = "2019-01-01", "2023-12-31"
MIN_COVERAGE = 0.90

Row = tuple[str, int, int, str, str]  # symbol, cik, sic, sic_description, fetched_at


def window_pairs(cik_map: pd.DataFrame) -> list[tuple[str, int]]:
    """(symbol, cik) pairs via the interval overlapping the discovery window.

    A symbol with no overlapping interval was never a window-era filer under
    any mapped CIK and is skipped. If several intervals overlap, the FIRST in
    file order wins -- dict-in-insertion-order dedupe, deterministic.
    """
    chosen: dict[str, int] = {}
    for row in cik_map.itertuples():
        overlaps = row.start <= WINDOW_END and (row.end == "" or row.end > WINDOW_START)
        if overlaps and row.symbol not in chosen:
            chosen[row.symbol] = int(row.cik)
    return sorted(chosen.items())


def fetch_sic(
    cik: int, fetch_json: Callable[[str], dict] = http_get_json
) -> tuple[int, str] | None:
    """(sic, sicDescription) for one CIK, or None -> unmapped (the fetch
    failed twice, or the filer has no SIC on record)."""
    payload: dict | None = None
    for _attempt in range(2):  # retry-once per spec section 6
        try:
            payload = fetch_json(SUBMISSIONS_URL.format(cik=cik))
            break
        except Exception:
            continue
    if payload is None:
        return None
    sic_raw = str(payload.get("sic") or "").strip()
    if not sic_raw.isdigit():
        return None
    return int(sic_raw), str(payload.get("sicDescription") or "")


def build_rows(
    pairs: list[tuple[str, int]],
    fetch_json: Callable[[str], dict] = http_get_json,
    *,
    fetched_at: str,
) -> tuple[list[Row], list[str]]:
    """One row per symbol; a CIK shared by several symbols (GOOG/GOOGL,
    FB/META) is fetched exactly once."""
    by_cik: dict[int, tuple[int, str] | None] = {}
    rows: list[Row] = []
    unmapped: list[str] = []
    for symbol, cik in pairs:
        if cik not in by_cik:
            by_cik[cik] = fetch_sic(cik, fetch_json)
        got = by_cik[cik]
        if got is None:
            unmapped.append(symbol)
            continue
        rows.append((symbol, cik, got[0], got[1], fetched_at))
    return rows, unmapped


def membership_window_symbols() -> set[str]:
    """Membership symbols whose interval overlaps the discovery window -- the
    honest coverage denominator (a 2017-2018-only member can never appear in
    a window segment, so it must not dilute the ratio)."""
    df = pd.read_csv(MEMBERSHIP, comment="#", dtype=str).fillna("")
    overlap = (df["start"] <= WINDOW_END) & ((df["end"] == "") | (df["end"] > WINDOW_START))
    return set(df.loc[overlap, "symbol"])


def validate(rows: list[Row], members: set[str]) -> None:
    mapped = {r[0] for r in rows} & members
    ratio = len(mapped) / len(members)
    if ratio < MIN_COVERAGE:
        sys.exit(
            f"FATAL: only {ratio:.1%} of {len(members)} window membership symbols "
            f"mapped to a SIC (need >= {MIN_COVERAGE:.0%}); do not commit this map"
        )
    print(f"coverage OK: {ratio:.1%} of {len(members)} window membership symbols mapped")


def write_csv(rows: list[Row], output: Path = OUTPUT) -> None:
    """csv.writer, NOT string-join: SEC sicDescription values contain commas
    (e.g. 'Services-Computer Programming, Data Processing, Etc.')."""
    with output.open("w", newline="") as fh:
        fh.write("# symbol -> SIC classification. GENERATED by scripts/build_sic_map.py\n")
        fh.write("# Source + caveats: src/trading/venues/universes/sources/PROVENANCE.md.\n")
        fh.write("# CURRENT SEC code applied backward over the window (disclosed caveat).\n")
        writer = csv.writer(fh)
        writer.writerow(["symbol", "cik", "sic", "sic_description", "fetched_at"])
        writer.writerows(rows)


def main() -> None:
    cik_map = pd.read_csv(CIK_MAP, comment="#", dtype=str).fillna("")
    pairs = window_pairs(cik_map)
    distinct_ciks = len({cik for _, cik in pairs})
    print(f"fetching SIC for {len(pairs)} symbols ({distinct_ciks} CIKs, ~0.11s/req)...")
    fetched_at = datetime.datetime.now(datetime.UTC).date().isoformat()
    rows, unmapped = build_rows(pairs, fetched_at=fetched_at)
    validate(rows, membership_window_symbols())
    write_csv(rows)
    print(f"wrote {OUTPUT} ({len(rows)} symbols mapped; {len(unmapped)} unmapped)")
    if unmapped:
        print("unmapped (belong to NO segment, never guessed): " + ", ".join(unmapped))


if __name__ == "__main__":
    main()
