"""Build the point-in-time equities membership file (spec: Venue Model).

Merges three sources into src/trading/venues/universes/equities_membership.csv:
- S&P 500: the snapshotted fja05680/sp500 dataset (date -> full ticker list).
- NDX: Wikipedia's Nasdaq-100 current constituents + yearly change tables,
  reconstructed backward from today.
- S&P 400 (MidCap): Wikipedia's "List of S&P 400 companies" current
  constituents + changes table, reconstructed backward the same way. That
  table only goes back to 2019 with confidence, so sp400 intervals never
  start before SP400_SINCE (2019-01-01) regardless of how far back a symbol's
  membership actually goes.

Self-validating: aborts unless (a) every reconstructed date has a plausible
member count, (b) two known sp400 removals land on their exact dates, and
(c) the merged sp500+ndx current membership approximately matches the M1
snapshot (universes/equities.csv) -- sp400 is opt-in (see venue config) so it
is deliberately excluded from that drift check. Treat the output as frozen
data -- regenerate deliberately and review the diff.

Usage: uv run python scripts/build_pit_membership.py
"""

from __future__ import annotations

import datetime
import io
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
UNIVERSES = ROOT / "src" / "trading" / "venues" / "universes"
SP500_SNAPSHOT = UNIVERSES / "sources" / "sp500_history.csv"
CURRENT_SNAPSHOT = UNIVERSES / "equities.csv"
OUTPUT = UNIVERSES / "equities_membership.csv"
WIKI_NDX_URL = "https://en.wikipedia.org/wiki/Nasdaq-100"
WIKI_SP400_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies"
SINCE = "2017-01-01"  # a year of pad before the 2018 backtest span
SP400_SINCE = "2019-01-01"  # the sp400 changes table is only reliable back to 2019


def normalize(symbol: str) -> str:
    return str(symbol).strip().upper().replace(".", "-")


def split_cell(cell: object) -> list[str]:
    """One Wikipedia ticker cell -> normalized symbols. Dual-class rows pack
    several tickers into one cell ("UAA/UA", or comma-separated); each is a
    real, separately-priced symbol and needs its own interval row."""
    parts = re.split(r"[/,]", str(cell))
    symbols = [normalize(p) for p in parts if p.strip()]
    return [s for s in symbols if s not in ("NAN", "-")]


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


def fetch_change_table(url: str) -> tuple[set[str], list[tuple[str, str, str]]]:
    """Returns (current members, [(date_iso, 'added'|'removed', symbol), ...]).

    Shared by NDX (Nasdaq-100) and sp400 (List of S&P 400 companies): both
    Wikipedia pages carry one "current constituents" table (a Ticker/Symbol
    column, no Date column, >80 rows) plus one or more change tables (a Date
    column and Added/Removed Ticker columns).
    """
    tables = pd.read_html(io.StringIO(_fetch_html(url)))
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
            current = {
                s for cell in table[cols[ticker_cols[0]]].astype(str) for s in split_cell(cell)
            }
            continue
        if not date_cols:
            continue
        date_col = cols[date_cols[0]]
        for action in ("added", "removed"):
            for i, c in enumerate(lower):
                if action in c and ("ticker" in c or "symbol" in c):
                    for _, row in table.iterrows():
                        when = pd.to_datetime(row[date_col], errors="coerce")
                        if pd.isna(when):
                            continue
                        for symbol in split_cell(row[cols[i]]):
                            changes.append((when.date().isoformat(), action, symbol))
    if not current:
        sys.exit(
            f"FATAL: could not locate the constituents table at {url}; "
            "inspect pd.read_html output and adjust column matching."
        )
    return current, sorted(changes)


def changes_intervals(
    current: set[str], changes: list[tuple[str, str, str]], since: str
) -> dict[str, list[list[str]]]:
    """Walk changes newest -> oldest, reconstructing membership backward from today.

    `since` is the floor: reconstruction stops there, and any symbol still
    open past it is treated as "member since before `since`".
    """
    intervals: dict[str, list[list[str]]] = {}
    end_open: dict[str, str] = {s: "" for s in current}  # symbol -> interval end (exclusive)
    for date_iso, action, symbol in sorted(changes, reverse=True):
        if date_iso < since:
            break
        if action == "added" and symbol in end_open:
            intervals.setdefault(symbol, []).append([date_iso, end_open.pop(symbol)])
        elif action == "removed" and symbol not in end_open:
            end_open[symbol] = date_iso  # was a member until (exclusive) date_iso
    for symbol, end in end_open.items():
        intervals.setdefault(symbol, []).append([since, end])  # member since before `since`
    return intervals


def validate(rows: list[tuple[str, str, str, str]]) -> None:
    # Symbol cleanliness: normalize() yields A-Z/0-9/'-' only. Anything else
    # (slashes, spaces, footnote markers, unicode) is an unsplit or unparsed
    # Wikipedia cell -- a data bug, not a ticker.
    dirty = sorted({s for s, _, _, _ in rows if not re.fullmatch(r"[A-Z0-9-]+", s)})
    if dirty:
        sys.exit(f"FATAL: non-ticker symbol artifacts in output: {dirty}")

    def members_on(day: str, indices: set[str] | None = None) -> set[str]:
        return {
            s
            for s, idx, start, end in rows
            if (indices is None or idx in indices) and start <= day and (end == "" or day < end)
        }

    for day, low, high in [
        ("2018-06-01", 450, 650),
        ("2022-06-01", 450, 650),
        ("2026-07-01", 450, 650),
    ]:
        count = len(members_on(day, {"sp500", "ndx"}))
        if not low <= count <= high:
            sys.exit(f"FATAL: {count} sp500+ndx members on {day}, expected {low}..{high}")

    for day, low, high in [
        ("2019-06-01", 380, 420),
        ("2022-06-01", 380, 420),
        ("2026-07-01", 380, 420),
    ]:
        count = len(members_on(day, {"sp400"}))
        if not low <= count <= high:
            sys.exit(f"FATAL: {count} sp400 members on {day}, expected {low}..{high}")

    # Spot-check anchors (scout-verified against the Wikipedia changes table):
    # exact known removal dates, not just a plausible count.
    def removed_on(symbol: str, index: str) -> str | None:
        ends = [end for s, idx, _, end in rows if s == symbol and idx == index and end]
        return max(ends) if ends else None

    cdk_removed = removed_on("CDK", "sp400")
    if cdk_removed != "2022-07-06":
        sys.exit(
            f"FATAL: CDK Global sp400 removal date wrong: {cdk_removed!r}, expected 2022-07-06"
        )
    mdp_removed = removed_on("MDP", "sp400")
    if mdp_removed != "2020-04-27":
        sys.exit(f"FATAL: Meredith sp400 removal date wrong: {mdp_removed!r}, expected 2020-04-27")

    snapshot = {
        normalize(s) for s in pd.read_csv(CURRENT_SNAPSHOT, comment="#")["symbol"].astype(str)
    }
    today = datetime.date.today().isoformat()
    # sp400 is opt-in (see venue [universe] config), so the M1 snapshot -- the
    # live sp500+ndx universe -- is compared against sp500+ndx only.
    drift = len(snapshot ^ members_on(today, {"sp500", "ndx"}))
    if drift > 15:
        sys.exit(f"FATAL: current membership differs from the M1 snapshot by {drift} symbols")
    sp400_today = len(members_on(today, {"sp400"}))
    print(
        f"validation OK: drift vs M1 snapshot = {drift} symbols; "
        f"sp400 members today = {sp400_today}; "
        f"CDK removed {cdk_removed}, MDP removed {mdp_removed}"
    )


def main() -> None:
    merged: list[tuple[str, str, str, str]] = []
    for symbol, spans in sp500_intervals().items():
        for start, end in spans:
            if end == "" or end >= SINCE:
                merged.append((symbol, "sp500", max(start, SINCE), end))
    ndx_current, ndx_changes = fetch_change_table(WIKI_NDX_URL)
    for symbol, spans in changes_intervals(ndx_current, ndx_changes, SINCE).items():
        for start, end in spans:
            merged.append((symbol, "ndx", start, end))
    sp400_current, sp400_changes = fetch_change_table(WIKI_SP400_URL)
    for symbol, spans in changes_intervals(sp400_current, sp400_changes, SP400_SINCE).items():
        for start, end in spans:
            merged.append((symbol, "sp400", start, end))
    merged.sort()
    validate(merged)
    lines = [
        "# Point-in-time S&P 500 + NDX + S&P 400 membership. GENERATED by "
        "scripts/build_pit_membership.py",
        "# Sources + licences: see sources/PROVENANCE.md. start inclusive, end "
        "exclusive, empty end = current.",
        "symbol,index,start,end",
    ]
    lines += [f"{s},{idx},{start},{end}" for s, idx, start, end in merged]
    OUTPUT.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUTPUT} ({len(merged)} intervals)")


if __name__ == "__main__":
    main()
