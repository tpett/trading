"""Resolve CIKs for HISTORICAL (delisted/acquired) membership symbols that
scripts/build_cik_map.py cannot map, and merge them into the committed
src/trading/fundamentals/cik_map.csv.

build_cik_map.py resolves a symbol only if it chains (via RENAMES) to a
ticker alive in TODAY's company_tickers.json, so a company acquired, taken
private, or delisted with no successor ticker (XLNX, CELG, TWTR, FRC, ...)
gets no row at all -- and Piece 2's segment map needs exactly those
window-era members. This script recovers them from the SEC's Financial
Statement Data Sets (FSDS) quarterly ZIPs: each sub.txt row's `instance`
column is the filing's XBRL instance filename, whose prefix is the ticker
the filer ITSELF used at filing time (xlnx-20201226.xml -> XLNX), alongside
the filer's cik and name -- a point-in-time ticker->CIK record that includes
dead companies.

Resolution is deliberately conservative -- never guess:
- only 10-K/10-Q rows count, and a candidate CIK qualifies only if it FILED
  during the symbol's membership tenure plus a 90-day grace tail (an acquired
  company's final report often lands a few weeks after index removal; the
  grace is far too short for a recycler to IPO and file);
- several qualifying CIKs resolve only via two reviewed tie-breaks: the
  symbol's RENAMES successor already maps to one of them in the committed
  map (subsidiary co-filers like Qwest/Level 3 share Lumen's ctl- instance
  prefix), or SEC's own submissions JSON attributes the symbol's ticker to
  exactly ONE of them; otherwise the symbol stays unmapped (reported);
- every resolution is verified against that CIK's live submissions JSON
  (throttled companyfacts seam): the symbol must appear in its `tickers`
  array, or the FSDS filer name must match its name/formerNames -- else
  dropped;
- every RENAMES pair whose two ends are both mapped (here or in the
  committed map) must agree on the CIK, or the run aborts;
- EXCLUSIONS (confirmed recycled tickers, e.g. APC/BID/CONE) are never
  resolved -- tests pin them absent from the committed map;
- appended intervals are end-dated (RENAMES change date, else the successor
  row's start, else the symbol's last membership end) so a later recycler
  of the ticker can never inherit the interval.

Outputs: src/trading/fundamentals/cik_map_historical.csv (committed source
artifact, reviewed like RENAMES; build_cik_map.py re-merges it on every
regeneration) and the same rows appended to cik_map.csv -- existing rows are
preserved byte-for-byte, never rewritten or reordered. Aborts below 90%
window-membership coverage. Update
src/trading/venues/universes/sources/PROVENANCE.md on every regeneration.

Usage: uv run python scripts/build_cik_map_historical.py
(downloads ~21 FSDS quarterly ZIPs, ~50-100MB each, cached in
data/edgar-raw/ shared with backfill_fundamentals.py --source zips)
"""

from __future__ import annotations

import datetime
import re
import sys
import zipfile
from collections.abc import Callable
from pathlib import Path

import pandas as pd

from trading.fundamentals.backfill import quarter_range
from trading.fundamentals.companyfacts import http_get_json
from trading.symbols import RENAMES, normalize

sys.path.insert(0, str(Path(__file__).resolve().parent))

from backfill_fundamentals import RAW_DIR, download  # noqa: E402
from build_cik_map import EXCLUSIONS  # noqa: E402
from build_sic_map import WINDOW_END, WINDOW_START  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
MEMBERSHIP = ROOT / "src" / "trading" / "venues" / "universes" / "equities_membership.csv"
CIK_MAP = ROOT / "src" / "trading" / "fundamentals" / "cik_map.csv"
OUTPUT = ROOT / "src" / "trading" / "fundamentals" / "cik_map_historical.csv"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
# FSDS quarters spanning the discovery window (2019-01-01..2023-12-31) plus
# one trailing quarter so a filing FILED just after the window still counts.
QUARTERS = quarter_range("2019q1", "2024q1")
SINCE = "2017-01-01"  # same interval floor as build_cik_map.py
MIN_COVERAGE = 0.90
FAR_FUTURE = "9999-12-31"  # stand-in for an open interval end when comparing
GRACE_DAYS = 90  # filed-date tail past membership end (final straggler reports)

# (symbol, cik, start, end, name) -- name kept as review evidence
HistRow = tuple[str, int, str, str, str]


def _overlaps_window(start: str, end: str) -> bool:
    return start <= WINDOW_END and (end == "" or end > WINDOW_START)


def target_symbols(cik_map: pd.DataFrame, membership: pd.DataFrame) -> list[str]:
    """Window-membership symbols with NO window-overlapping cik_map interval,
    minus the deliberately-excluded recycled tickers."""
    covered = {
        row.symbol
        for row in cik_map.itertuples()
        if _overlaps_window(row.start, row.end)
    }
    members = {
        row.symbol
        for row in membership.itertuples()
        if _overlaps_window(row.start, row.end)
    }
    return sorted(members - covered - set(EXCLUSIONS))


_INSTANCE_PREFIX = re.compile(r"^([a-zA-Z][a-zA-Z0-9]{0,5})-")


def ticker_from_instance(instance: str) -> str | None:
    """xlnx-20201226.xml -> XLNX; None when the prefix is not ticker-shaped
    (older accession-numbered or ad-hoc instance names)."""
    match = _INSTANCE_PREFIX.match(str(instance))
    return match.group(1).upper() if match else None


def parse_sub(zip_path: Path) -> pd.DataFrame:
    """One FSDS ZIP's sub.txt -> (cik, name, filed, ticker) for 10-K/10-Q
    rows whose instance filename carries a ticker prefix."""
    cols = ["cik", "name", "form", "filed", "instance"]
    with zipfile.ZipFile(zip_path) as zf, zf.open("sub.txt") as fh:
        sub = pd.read_csv(fh, sep="\t", dtype=str, usecols=cols, encoding="latin-1")
    sub = sub[sub["form"].isin(["10-K", "10-Q"])].dropna(subset=["cik", "filed", "instance"])
    sub = sub.copy()
    sub["ticker"] = sub["instance"].map(ticker_from_instance)
    sub = sub.dropna(subset=["ticker"])
    sub["cik"] = sub["cik"].astype(int)
    # filed is YYYYMMDD -> ISO for direct string comparison with intervals
    sub["filed"] = sub["filed"].str.replace(
        r"^(\d{4})(\d{2})(\d{2})$", r"\1-\2-\3", regex=True
    )
    sub["name"] = sub["name"].fillna("")
    return sub[["cik", "name", "filed", "ticker"]]


def candidates_for(sub: pd.DataFrame, targets: set[str]) -> dict[str, dict[int, dict]]:
    """{symbol: {cik: {name, filed: sorted dates}}} restricted to targets.
    The name kept is the LATEST filing's (a filer's registered name can
    change across the window)."""
    out: dict[str, dict[int, dict]] = {}
    hits = sub[sub["ticker"].isin(targets)]
    for (ticker, cik), group in hits.groupby(["ticker", "cik"], sort=True):
        filed = sorted(group["filed"])
        latest_name = group.sort_values("filed").iloc[-1]["name"]
        out.setdefault(str(ticker), {})[int(cik)] = {"name": latest_name, "filed": filed}
    return out


def membership_tenure(membership: pd.DataFrame) -> dict[str, tuple[str, str]]:
    """{symbol: (first start, last end)} across all of a symbol's membership
    intervals; an open interval yields a FAR_FUTURE end."""
    tenure: dict[str, tuple[str, str]] = {}
    for row in membership.itertuples():
        end = row.end if row.end else FAR_FUTURE
        if row.symbol in tenure:
            lo, hi = tenure[row.symbol]
            tenure[row.symbol] = (min(lo, row.start), max(hi, end))
        else:
            tenure[row.symbol] = (row.start, end)
    return tenure


def qualifying_candidates(
    symbol_candidates: dict[int, dict], tenure: tuple[str, str], grace_days: int = 0
) -> list[tuple[int, str]]:
    """Candidate CIKs with a 10-K/10-Q FILED inside the symbol's membership
    tenure plus a grace tail (a straggling final report), sorted by cik."""
    lo, hi = tenure
    if grace_days and hi != FAR_FUTURE:
        hi = (datetime.date.fromisoformat(hi) + datetime.timedelta(days=grace_days)).isoformat()
    return [
        (cik, info["name"])
        for cik, info in sorted(symbol_candidates.items())
        if any(lo <= filed <= hi for filed in info["filed"])
    ]


def successor_cik(symbol: str, committed: pd.DataFrame) -> int | None:
    """The committed CIK of the symbol's RENAMES successor at the rename
    date -- reviewed-table evidence for tie-breaking co-filer prefix
    collisions (Lumen's subsidiaries also file under the ctl- prefix)."""
    for old, new, date in RENAMES:
        if old != symbol:
            continue
        for row in committed.itertuples():
            if row.symbol == new and row.start <= date and (row.end == "" or date < row.end):
                return int(row.cik)
    return None


def sec_ticker_tiebreak(
    symbol: str,
    qualifying: list[tuple[int, str]],
    fetch_json: Callable[[str], dict] = http_get_json,
) -> tuple[int, str] | None:
    """The single qualifying candidate whose live submissions JSON attributes
    the symbol's ticker to it (retry-once per candidate); several -- or none,
    or any fetch failing twice -- resolves nothing."""
    matches: list[tuple[int, str]] = []
    for cik, name in qualifying:
        payload: dict | None = None
        for _attempt in range(2):
            try:
                payload = fetch_json(SUBMISSIONS_URL.format(cik=cik))
                break
            except Exception:
                continue
        if payload is None:
            return None  # cannot rule this candidate out: stay unmapped
        if normalize(symbol) in {normalize(t) for t in payload.get("tickers") or []}:
            matches.append((cik, name))
    return matches[0] if len(matches) == 1 else None


def resolve_target(
    symbol: str,
    symbol_candidates: dict[int, dict],
    tenure: tuple[str, str],
    committed: pd.DataFrame,
    fetch_json: Callable[[str], dict] = http_get_json,
) -> tuple[int, str, str] | None:
    """(cik, name, rule) for one target symbol, or None (never guess).
    Rules, most to least direct: a UNIQUE candidate filed strictly in tenure
    (checked BEFORE the grace tail, so grace can widen but never break a
    clean strict answer); a unique candidate within tenure+grace; the
    RENAMES successor's committed CIK among the qualifying; SEC's own ticker
    attribution naming exactly one of them."""
    strict = qualifying_candidates(symbol_candidates, tenure)
    if len(strict) == 1:
        return (*strict[0], "unique")
    qualifying = qualifying_candidates(symbol_candidates, tenure, GRACE_DAYS)
    if len(qualifying) == 1:
        return (*qualifying[0], "unique-grace")
    if len(qualifying) > 1:
        successor = successor_cik(symbol, committed)
        by_cik = dict(qualifying)
        if successor in by_cik:
            return (successor, by_cik[successor], "renames-successor")
        tiebreak = sec_ticker_tiebreak(symbol, qualifying, fetch_json)
        if tiebreak is not None:
            return (*tiebreak, "sec-tickers")
    return None


def choose_interval(
    symbol: str, existing: pd.DataFrame, tenure_end: str
) -> tuple[str, str] | None:
    """[start, end) for the appended row. End precedence: the symbol's
    RENAMES change date (the ticker stopped being live there), else the
    earliest existing cik_map interval start for the same symbol (e.g. DOC's
    window-era filer hands off to the committed successor row), else the
    symbol's last membership end. A degenerate interval resolves to None."""
    renamed_away = {old: date for old, _new, date in RENAMES}
    if symbol in renamed_away:
        end = renamed_away[symbol]
    else:
        same = existing[existing["symbol"] == symbol]
        if len(same):
            end = min(same["start"])
        elif tenure_end != FAR_FUTURE:
            end = tenure_end
        else:
            return None  # open-ended tenure with no successor row: leave unmapped
    return (SINCE, end) if SINCE < end else None


def _canon(name: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(name).upper())


def verify_resolution(
    symbol: str,
    cik: int,
    fsds_name: str,
    fetch_json: Callable[[str], dict] = http_get_json,
) -> bool:
    """Live submissions-JSON check (retry-once): the symbol appears in the
    CIK's tickers array, or the FSDS filer name matches its registered
    name/formerNames. A fetch that fails twice verifies nothing -> False."""
    payload: dict | None = None
    for _attempt in range(2):
        try:
            payload = fetch_json(SUBMISSIONS_URL.format(cik=cik))
            break
        except Exception:
            continue
    if payload is None:
        return False
    tickers = {normalize(t) for t in payload.get("tickers") or []}
    if normalize(symbol) in tickers:
        return True
    names = [payload.get("name", "")]
    names += [fn.get("name", "") for fn in payload.get("formerNames") or []]
    want = _canon(fsds_name)
    return bool(want) and any(
        want in _canon(have) or _canon(have) in want for have in names if _canon(have)
    )


def check_rename_consistency(rows: list[HistRow], committed: pd.DataFrame) -> list[str]:
    """RENAMES pairs mapped on both sides of the rename BOUNDARY (this run or
    the committed map) but to DIFFERENT CIKs -- must be empty.

    Compared AT the boundary, not per symbol: a rename target can carry a
    different, earlier company's interval before the handoff (DOC was
    Physicians Realty until Healthpeak took the ticker on 2024-03-04) and
    that is not a conflict."""
    intervals = [
        (row.symbol, int(row.cik), row.start, row.end) for row in committed.itertuples()
    ]
    intervals += [(s, cik, start, end) for s, cik, start, end, _name in rows]

    def cik_at(symbol: str, day: str) -> int | None:
        for sym, cik, start, end in intervals:
            if sym == symbol and start <= day and (end == "" or day < end):
                return cik
        return None

    conflicts: list[str] = []
    for old, new, date in RENAMES:
        day_before = (
            datetime.date.fromisoformat(date) - datetime.timedelta(days=1)
        ).isoformat()
        old_cik, new_cik = cik_at(old, day_before), cik_at(new, date)
        if old_cik is not None and new_cik is not None and old_cik != new_cik:
            conflicts.append(f"{old}->{new}@{date}: {old_cik} != {new_cik}")
    return conflicts


def append_to_cik_map(rows: list[HistRow], cik_map_path: Path = CIK_MAP) -> None:
    """Append (symbol,cik,start,end) lines; existing content is preserved
    byte-for-byte (never rewritten or reordered). Refuses a re-run double
    append: no appended (symbol, start) may already be present."""
    text = cik_map_path.read_text()
    existing = {
        (line.split(",")[0], line.split(",")[2])
        for line in text.splitlines()
        if line and not line.startswith(("#", "symbol,"))
    }
    dupes = [s for s, _cik, start, _end, _name in rows if (s, start) in existing]
    if dupes:
        sys.exit(f"FATAL: rows already appended to {cik_map_path.name}: {dupes}")
    if not text.endswith("\n"):
        text += "\n"
    lines = [f"{s},{cik},{start},{end}" for s, cik, start, end, _name in rows]
    cik_map_path.write_text(text + "\n".join(lines) + "\n")


def write_historical_csv(rows: list[HistRow], output: Path = OUTPUT) -> None:
    lines = [
        "# Historical (delisted/acquired) symbol->CIK intervals resolved from SEC",
        "# FSDS sub.txt instance prefixes. GENERATED by",
        "# scripts/build_cik_map_historical.py; build_cik_map.py re-merges these on",
        "# every regeneration. Sources + verification: see",
        "# src/trading/venues/universes/sources/PROVENANCE.md. The name column is",
        "# review evidence (the filer's FSDS registered name), not consumed by code.",
        "symbol,cik,start,end,name",
    ]
    lines += [f'{s},{cik},{start},{end},"{name}"' for s, cik, start, end, name in rows]
    output.write_text("\n".join(lines) + "\n")


def window_coverage(cik_map: pd.DataFrame, membership: pd.DataFrame) -> tuple[int, int]:
    """(mapped, members): window members with a window-overlapping interval."""
    members = {
        row.symbol for row in membership.itertuples() if _overlaps_window(row.start, row.end)
    }
    covered = {
        row.symbol for row in cik_map.itertuples() if _overlaps_window(row.start, row.end)
    }
    return len(members & covered), len(members)


def main() -> None:
    membership = pd.read_csv(MEMBERSHIP, comment="#", dtype=str).fillna("")
    cik_map = pd.read_csv(CIK_MAP, comment="#", dtype=str).fillna("")
    targets = target_symbols(cik_map, membership)
    before_mapped, members = window_coverage(cik_map, membership)
    print(
        f"coverage before: {before_mapped}/{members} "
        f"({before_mapped / members:.1%}) window members mapped; "
        f"{len(targets)} targets (+{len(set(EXCLUSIONS))} excluded recycled tickers)"
    )

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    frames = []
    for quarter in QUARTERS:
        zip_path = download(quarter)
        frames.append(parse_sub(zip_path))
        print(f"parsed {quarter}: {len(frames[-1])} ticker-prefixed 10-K/10-Q rows")
    sub = pd.concat(frames, ignore_index=True)

    cands = candidates_for(sub, set(targets))
    tenure = membership_tenure(membership)
    resolved: dict[str, tuple[int, str, str]] = {}
    unresolved: list[str] = []
    absent: list[str] = []
    for symbol in targets:
        if symbol not in cands:
            absent.append(symbol)
            continue
        got = resolve_target(symbol, cands[symbol], tenure[symbol], cik_map)
        if got is None:
            unresolved.append(symbol)
            continue
        resolved[symbol] = got

    print(f"resolved {len(resolved)} symbols; verifying against submissions JSON ...")
    dropped_verify: list[str] = []
    dropped_interval: list[str] = []
    rows: list[HistRow] = []
    rules: dict[str, str] = {}
    for symbol in sorted(resolved):
        cik, name, rule = resolved[symbol]
        if not verify_resolution(symbol, cik, name):
            dropped_verify.append(symbol)
            continue
        interval = choose_interval(symbol, cik_map, tenure[symbol][1])
        if interval is None:
            dropped_interval.append(symbol)
            continue
        rows.append((symbol, cik, interval[0], interval[1], name))
        rules[symbol] = rule

    conflicts = check_rename_consistency(rows, cik_map)
    if conflicts:
        sys.exit("FATAL: RENAMES pairs disagree on CIK: " + "; ".join(conflicts))

    print(f"\n{'symbol':<8}{'cik':>9}  {'interval':<25}{'rule':<19}name")
    for symbol, cik, start, end, name in rows:
        print(f"{symbol:<8}{cik:>9}  {start}..{end}   {rules[symbol]:<19}{name}")

    appended = pd.concat(
        [
            cik_map,
            pd.DataFrame(
                [(s, str(c), st, e) for s, c, st, e, _n in rows],
                columns=["symbol", "cik", "start", "end"],
            ),
        ],
        ignore_index=True,
    )
    after_mapped, _ = window_coverage(appended, membership)
    print(
        f"\ncoverage after: {after_mapped}/{members} ({after_mapped / members:.1%}); "
        f"unmapped remain: absent-from-FSDS={sorted(absent)} "
        f"unresolved={sorted(unresolved)} dropped-verification={sorted(dropped_verify)} "
        f"dropped-degenerate-interval={sorted(dropped_interval)} "
        f"excluded={sorted(EXCLUSIONS)}"
    )
    if after_mapped / members < MIN_COVERAGE:
        sys.exit(
            f"FATAL: only {after_mapped / members:.1%} of {members} window membership "
            f"symbols would be mapped (need >= {MIN_COVERAGE:.0%}); nothing written"
        )
    write_historical_csv(rows)
    append_to_cik_map(rows)
    print(f"wrote {OUTPUT} and appended {len(rows)} rows to {CIK_MAP}")


if __name__ == "__main__":
    main()
