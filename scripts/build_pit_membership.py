"""Build the point-in-time equities membership file (spec: Venue Model).

Merges two sources into src/trading/venues/universes/equities_membership.csv:
- S&P 500: the snapshotted fja05680/sp500 dataset (date -> full ticker list).
- NDX: Wikipedia's Nasdaq-100 current constituents + yearly change tables,
  reconstructed backward from today.

Self-validating: aborts unless (a) every reconstructed date has a plausible
member count and (b) the merged current membership approximately matches the
M1 snapshot (universes/equities.csv). Treat the output as frozen data --
regenerate deliberately and review the diff.

Usage: uv run python scripts/build_pit_membership.py
"""

from __future__ import annotations

import datetime
import io
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
UNIVERSES = ROOT / "src" / "trading" / "venues" / "universes"
SP500_SNAPSHOT = UNIVERSES / "sources" / "sp500_history.csv"
CURRENT_SNAPSHOT = UNIVERSES / "equities.csv"
OUTPUT = UNIVERSES / "equities_membership.csv"
WIKI_NDX_URL = "https://en.wikipedia.org/wiki/Nasdaq-100"
SINCE = "2017-01-01"  # a year of pad before the 2018 backtest span


def normalize(symbol: str) -> str:
    return str(symbol).strip().upper().replace(".", "-")


def sp500_intervals() -> dict[str, list[list[str]]]:
    """Snapshot rows are (date, full ticker list) -> per-symbol [start, end) intervals."""
    df = pd.read_csv(SP500_SNAPSHOT)
    date_col = next(c for c in df.columns if "date" in c.lower())
    tick_col = next(c for c in df.columns if "ticker" in c.lower())
    snapshots: list[tuple[str, set[str]]] = []
    for _, row in df.iterrows():
        raw = str(row[tick_col]).replace(";", ",")
        symbols = {normalize(s) for s in raw.split(",") if s.strip()}
        snapshots.append((str(row[date_col])[:10], symbols))
    snapshots.sort()

    intervals: dict[str, list[list[str]]] = {}
    active: dict[str, list[str]] = {}
    for date_iso, symbols in snapshots:
        for symbol in symbols - set(active):
            interval = [date_iso, ""]
            active[symbol] = interval
            intervals.setdefault(symbol, []).append(interval)
        for symbol in set(active) - symbols:
            active.pop(symbol)[1] = date_iso  # end exclusive: gone as of this snapshot
    return intervals


def _fetch_html(url: str) -> str:
    """Wikipedia 403s requests without a User-Agent; urllib's default is blocked."""
    import urllib.request

    req = urllib.request.Request(url, headers={"User-Agent": "pit-membership-build-script/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read().decode("utf-8", errors="replace")


def fetch_ndx() -> tuple[set[str], list[tuple[str, str, str]]]:
    """Returns (current members, [(date_iso, 'added'|'removed', symbol), ...])."""
    tables = pd.read_html(io.StringIO(_fetch_html(WIKI_NDX_URL)))
    current: set[str] = set()
    changes: list[tuple[str, str, str]] = []
    for table in tables:
        cols = [
            " ".join(str(part) for part in c) if isinstance(c, tuple) else str(c)
            for c in table.columns
        ]
        table.columns = cols  # flatten MultiIndex columns so row[col] lookups work below
        lower = [c.lower() for c in cols]
        ticker_cols = [i for i, c in enumerate(lower) if "ticker" in c or "symbol" in c]
        date_cols = [i for i, c in enumerate(lower) if "date" in c]
        if ticker_cols and not date_cols and len(table) > 80:
            current = {normalize(s) for s in table[cols[ticker_cols[0]]].astype(str)}
            continue
        if not date_cols:
            continue
        date_col = cols[date_cols[0]]
        for action in ("added", "removed"):
            for i, c in enumerate(lower):
                if action in c and ("ticker" in c or "symbol" in c):
                    for _, row in table.iterrows():
                        when = pd.to_datetime(row[date_col], errors="coerce")
                        symbol = normalize(row[cols[i]])
                        if pd.isna(when) or not symbol or symbol in ("NAN", "-"):
                            continue
                        changes.append((when.date().isoformat(), action, symbol))
    if not current:
        sys.exit(
            "FATAL: could not locate the NDX constituents table on Wikipedia; "
            "inspect pd.read_html output and adjust column matching."
        )
    return current, sorted(changes)


def ndx_intervals(
    current: set[str], changes: list[tuple[str, str, str]]
) -> dict[str, list[list[str]]]:
    """Walk changes newest -> oldest, reconstructing membership backward from today."""
    intervals: dict[str, list[list[str]]] = {}
    end_open: dict[str, str] = {s: "" for s in current}  # symbol -> interval end (exclusive)
    for date_iso, action, symbol in sorted(changes, reverse=True):
        if date_iso < SINCE:
            break
        if action == "added" and symbol in end_open:
            intervals.setdefault(symbol, []).append([date_iso, end_open.pop(symbol)])
        elif action == "removed" and symbol not in end_open:
            end_open[symbol] = date_iso  # was a member until (exclusive) date_iso
    for symbol, end in end_open.items():
        intervals.setdefault(symbol, []).append([SINCE, end])  # member since before SINCE
    return intervals


def validate(rows: list[tuple[str, str, str, str]]) -> None:
    def members_on(day: str) -> set[str]:
        return {s for s, _, start, end in rows if start <= day and (end == "" or day < end)}

    for day, low, high in [
        ("2018-06-01", 450, 650),
        ("2022-06-01", 450, 650),
        ("2026-07-01", 450, 650),
    ]:
        count = len(members_on(day))
        if not low <= count <= high:
            sys.exit(f"FATAL: {count} members on {day}, expected {low}..{high}")
    snapshot = {
        normalize(s) for s in pd.read_csv(CURRENT_SNAPSHOT, comment="#")["symbol"].astype(str)
    }
    drift = len(snapshot ^ members_on(datetime.date.today().isoformat()))
    if drift > 15:
        sys.exit(f"FATAL: current membership differs from the M1 snapshot by {drift} symbols")
    print(f"validation OK: drift vs M1 snapshot = {drift} symbols")


def main() -> None:
    merged: list[tuple[str, str, str, str]] = []
    for symbol, spans in sp500_intervals().items():
        for start, end in spans:
            if end == "" or end >= SINCE:
                merged.append((symbol, "sp500", max(start, SINCE), end))
    ndx_current, ndx_changes = fetch_ndx()
    for symbol, spans in ndx_intervals(ndx_current, ndx_changes).items():
        for start, end in spans:
            merged.append((symbol, "ndx", start, end))
    merged.sort()
    validate(merged)
    lines = [
        "# Point-in-time S&P 500 + NDX membership. GENERATED by scripts/build_pit_membership.py",
        "# Sources + licences: see sources/PROVENANCE.md. start inclusive, end "
        "exclusive, empty end = current.",
        "symbol,index,start,end",
    ]
    lines += [f"{s},{idx},{start},{end}" for s, idx, start, end in merged]
    OUTPUT.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUTPUT} ({len(merged)} intervals)")


if __name__ == "__main__":
    main()
