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
from collections.abc import Callable
from pathlib import Path

import pandas as pd

from trading.fundamentals.companyfacts import COMPANYFACTS_URL, _http_get_json

ROOT = Path(__file__).resolve().parent.parent
MEMBERSHIP = ROOT / "src" / "trading" / "venues" / "universes" / "equities_membership.csv"
OUTPUT = ROOT / "src" / "trading" / "fundamentals" / "cik_map.csv"
TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
USER_AGENT = "trading-system travis@launchsupply.com"
SINCE = "2017-01-01"  # same pad as the membership build

# Symbols whose CURRENT company_tickers.json entry is a confirmed
# ticker-recycling squatter: the real (historical) membership-era company
# was acquired or went private and delisted -- with no RENAMES entry to
# follow -- so build_rows' current-ticker lookup would otherwise attach an
# unrelated, later company's CIK to the historical membership interval
# (investigated live via data.sec.gov/submissions/CIK##########.json; see
# src/trading/venues/universes/sources/PROVENANCE.md for the full writeup).
# Excluded symbols are always unmapped (fail-open, neutral rank) rather than
# silently mismapped. Extend deliberately: a new entry here should cite the
# specific squatter CIK/name it was found resolving to.
EXCLUSIONS: dict[str, str] = {
    "APC": (
        "ticker recycled: real APC (Anadarko Petroleum, sp500 member "
        "2017-01-01..2019-08-09) was acquired by Occidental; company_tickers.json "
        "now maps APC to CIK 2080921 (ARKO Petroleum Corp), an unrelated company"
    ),
    "BID": (
        "ticker recycled: real BID (Sotheby's, sp400 member "
        "2019-01-01..2019-10-03) went private; company_tickers.json now maps BID "
        "to CIK 2094919 (Tribeca Strategic Acquisition Corp), an unrelated SPAC"
    ),
    "CONE": (
        "ticker recycled: real CONE (CyrusOne, sp400 member "
        "2019-01-01..2022-03-30) went private; company_tickers.json now maps "
        "CONE to CIK 2103884 (Compass Sub North, Inc.), an unrelated merger shell"
    ),
}

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


def fetch_company_tickers_raw() -> dict[str, dict]:
    """One company_tickers.json fetch -> {ticker: {"cik": int, "title": str}}.
    fetch_company_tickers() (the CIK-only view build_rows uses) and
    check_identity_mismatches (which also wants the registered title) both
    derive from this single parse -- no extra network cost for the audit."""
    req = urllib.request.Request(TICKERS_URL, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as resp:
        raw = json.loads(resp.read().decode("utf-8"))
    return {
        normalize(v["ticker"]): {"cik": int(v["cik_str"]), "title": v.get("title", "")}
        for v in raw.values()
    }


def fetch_company_tickers() -> dict[str, int]:
    return {ticker: info["cik"] for ticker, info in fetch_company_tickers_raw().items()}


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
        if symbol in EXCLUSIONS:
            unmapped.append(symbol)
            continue
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
    excluded_present = set(EXCLUSIONS) & set(by_symbol)
    if excluded_present:
        sys.exit(
            f"FATAL: excluded (recycled-ticker) symbols leaked into cik_map: "
            f"{sorted(excluded_present)}"
        )
    mapped_current = current_members & set(by_symbol)
    ratio = len(mapped_current) / len(current_members)
    if ratio < 0.95:
        sys.exit(f"FATAL: only {ratio:.1%} of current members map to a CIK (need >= 95%)")
    print(f"validation OK: {ratio:.1%} of {len(current_members)} current members mapped")


def check_identity_mismatches(
    rows: list[tuple[str, int, str, str]],
    current_tickers_raw: dict[str, dict],
    fetch_json: Callable[[str], dict] = _http_get_json,
) -> list[dict[str, object]]:
    """Heuristic, report-only identity audit (recycling defenses, part b):
    the zero-filing reconciliation audit (scripts/verify_fundamentals.py)
    cannot catch a mismap whose wrongly-assigned CIK DOES have filings in
    the historical window -- only a name mismatch can. There is no
    independently-derivable "what this historical symbol's company was
    named" ground truth in this repo (that is exactly why a recycled ticker
    is dangerous), so this compares the ONE thing that IS derivable without
    one: a mapped symbol that is ALSO, independently, a live ticker in
    TODAY's company_tickers.json under a DIFFERENT cik than the one
    build_rows() assigned. That shape means two SEC-recognized identities
    (the direct current listing and the RENAMES-chain-resolved cik) disagree
    about who this symbol belongs to -- always worth a human look, even
    though it cannot happen for the EXCLUSIONS-covered case above (a plain,
    non-chain symbol's assigned cik IS its direct current listing by
    construction, so there is nothing here to compare against for that
    shape; the zero-filing audit is this repo's only defense there)."""
    audit: list[dict[str, object]] = []
    for symbol, cik, _start, _end in rows:
        direct = current_tickers_raw.get(symbol)
        if direct is None or direct["cik"] == cik:
            continue  # no independent current listing, or it agrees: nothing to compare
        try:
            payload = fetch_json(COMPANYFACTS_URL.format(cik=cik))
            mapped_entity_name = str(payload.get("entityName", ""))
        except Exception:
            mapped_entity_name = "(companyfacts fetch failed)"
        audit.append(
            {
                "symbol": symbol,
                "mapped_cik": cik,
                "mapped_entity_name": mapped_entity_name,
                "symbol_direct_cik": direct["cik"],
                "symbol_direct_title": direct["title"],
            }
        )
    if audit:
        print(f"IDENTITY AUDIT: {len(audit)} symbol(s) with a disagreeing direct listing:")
        for item in audit:
            print(
                f"  {item['symbol']}: mapped to cik={item['mapped_cik']} "
                f"({item['mapped_entity_name']!r}) but company_tickers.json's OWN "
                f"'{item['symbol']}' entry is cik={item['symbol_direct_cik']} "
                f"({item['symbol_direct_title']!r})"
            )
    else:
        print("IDENTITY AUDIT: 0 symbols with a disagreeing direct listing")
    return audit


def main() -> None:
    symbols, current_members = membership_symbols()
    current_tickers_raw = fetch_company_tickers_raw()
    current_tickers = {ticker: info["cik"] for ticker, info in current_tickers_raw.items()}
    rows, unmapped = build_rows(symbols, current_tickers)
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
    excluded_unmapped = [s for s in unmapped if s in EXCLUSIONS]
    if excluded_unmapped:
        print(f"of which {len(excluded_unmapped)} deliberately excluded (recycled tickers):")
        for symbol in excluded_unmapped:
            print(f"  {symbol}: {EXCLUSIONS[symbol]}")
    check_identity_mismatches(rows, current_tickers_raw)


if __name__ == "__main__":
    main()
