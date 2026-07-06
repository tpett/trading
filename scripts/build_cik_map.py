"""Build the committed CIK<->symbol point-in-time interval map
(src/trading/fundamentals/cik_map.csv) for the M4 fundamentals overlay.

Sources: SEC company_tickers.json (current ticker -> CIK, fetched live with
the mandatory User-Agent) + the reviewed RENAMES table below (membership
symbols that changed ticker since 2017, cross-checked against the membership
CSV's remove/add dates) + src/trading/venues/universes/equities_membership.csv
(whose symbols define what needs mapping).

Primary-class selection is implicit: only symbols present in the membership
CSV are emitted, so a CIK's non-member share classes never appear.

Self-validating: aborts unless FB and META resolve to one CIK (1326801) with
the boundary at 2022-06-09, ABC and COR share CIK 1140859, and >= 95% of
CURRENT members map. Unmapped symbols (mostly acquired/delisted companies
EDGAR no longer lists a ticker for) are printed: they get no fundamentals and
rank neutral -- extend RENAMES deliberately if one matters. Update
src/trading/venues/universes/sources/PROVENANCE.md on every regeneration.

Usage: uv run python scripts/build_cik_map.py
"""

from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MEMBERSHIP = ROOT / "src" / "trading" / "venues" / "universes" / "equities_membership.csv"
OUTPUT = ROOT / "src" / "trading" / "fundamentals" / "cik_map.csv"
TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
USER_AGENT = "trading-system travis@launchsupply.com"
SINCE = "2017-01-01"  # same pad as the membership build

# (old_symbol, new_symbol, change_date). Reviewed ticker renames among 2017+
# membership symbols; the boundary date decides which symbol a filing filed
# near it attaches to, so day-exactness is low-stakes but should match the
# membership CSV's remove/add transition. Chains (A->B->C) are supported.
RENAMES = [
    ("DWDP", "DD", "2019-06-03"),
    ("HCP", "PEAK", "2019-11-05"),
    ("HRS", "LHX", "2019-07-01"),
    ("UTX", "RTX", "2020-04-03"),
    ("MYL", "VTRS", "2020-11-16"),
    ("WLTW", "WTW", "2022-01-05"),
    ("FB", "META", "2022-06-09"),
    ("ANTM", "ELV", "2022-06-28"),
    ("FBHS", "FBIN", "2022-12-19"),
    ("PKI", "RVTY", "2023-05-16"),
    ("FISV", "FI", "2023-06-06"),
    ("RE", "EG", "2023-07-10"),
    ("ABC", "COR", "2023-08-30"),
    ("PEAK", "DOC", "2024-03-04"),
]


def normalize(symbol: str) -> str:
    return str(symbol).strip().upper().replace(".", "-")


def fetch_company_tickers() -> dict[str, int]:
    req = urllib.request.Request(TICKERS_URL, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as resp:
        raw = json.loads(resp.read().decode("utf-8"))
    return {normalize(v["ticker"]): int(v["cik_str"]) for v in raw.values()}


def membership_symbols() -> tuple[set[str], set[str]]:
    """(all symbols ever in the membership CSV, symbols currently a member)."""
    df = pd.read_csv(MEMBERSHIP, comment="#", dtype=str).fillna("")
    current = set(df.loc[df["end"] == "", "symbol"])
    return set(df["symbol"]), current


def build_rows(
    symbols: set[str], current_tickers: dict[str, int]
) -> tuple[list[tuple[str, int, str, str]], list[str]]:
    new_by_old = {old: (new, date) for old, new, date in RENAMES}
    renamed_starts = {new: date for _, new, date in RENAMES}
    rows: list[tuple[str, int, str, str]] = []
    unmapped: list[str] = []
    for symbol in sorted(symbols):
        # Follow the rename chain forward until we hit a current EDGAR ticker.
        cursor, end, seen = symbol, "", set()
        while cursor not in current_tickers:
            if cursor in seen or cursor not in new_by_old:
                cursor = None
                break
            seen.add(cursor)
            new, date = new_by_old[cursor]
            if end == "":
                end = date  # this symbol stopped being the live ticker here
            cursor = new
        if cursor is None:
            unmapped.append(symbol)
            continue
        start = max(SINCE, renamed_starts.get(symbol, SINCE))
        rows.append((symbol, current_tickers[cursor], start, end))
    return rows, unmapped


def validate(rows: list[tuple[str, int, str, str]], current_members: set[str]) -> None:
    by_symbol = {s: (cik, start, end) for s, cik, start, end in rows}
    fb, meta = by_symbol.get("FB"), by_symbol.get("META")
    if (
        fb is None
        or meta is None
        or fb[0] != 1326801
        or meta[0] != 1326801
        or fb[2] != "2022-06-09"
        or meta[1] != "2022-06-09"
    ):
        sys.exit(f"FATAL: FB/META rename mapping wrong: FB={fb}, META={meta}")
    abc, cor = by_symbol.get("ABC"), by_symbol.get("COR")
    if abc is None or cor is None or abc[0] != 1140859 or cor[0] != 1140859:
        sys.exit(f"FATAL: ABC/COR rename mapping wrong: ABC={abc}, COR={cor}")
    mapped_current = current_members & set(by_symbol)
    ratio = len(mapped_current) / len(current_members)
    if ratio < 0.95:
        sys.exit(f"FATAL: only {ratio:.1%} of current members map to a CIK (need >= 95%)")
    print(f"validation OK: {ratio:.1%} of {len(current_members)} current members mapped")


def main() -> None:
    symbols, current_members = membership_symbols()
    rows, unmapped = build_rows(symbols, fetch_company_tickers())
    validate(rows, current_members)
    lines = [
        "# CIK<->symbol point-in-time intervals. GENERATED by scripts/build_cik_map.py",
        "# Sources + validation: see src/trading/venues/universes/sources/PROVENANCE.md.",
        "# start inclusive, end exclusive, empty end = current EDGAR ticker.",
        "symbol,cik,start,end",
    ]
    lines += [f"{s},{cik},{start},{end}" for s, cik, start, end in rows]
    OUTPUT.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUTPUT} ({len(rows)} intervals; {len(unmapped)} membership symbols unmapped)")
    print("unmapped (no fundamentals -> neutral rank): " + ", ".join(unmapped))


if __name__ == "__main__":
    main()
