"""Roster-scoped CIK-resolution + companyfacts-shares pipeline for the R3
down-cap universe.

The committed cik_map.csv / cik_map_historical.csv pipeline is scoped to the
INDEX membership (equities_membership.csv). The down-cap universe is instead
defined by a survivorship-free ROSTER (trading.venues.universes.downcap_roster,
Tiingo supported_tickers) whose ~10.6k discovery-span candidates only ~48%
resolve via SEC's current company_tickers.json; the other ~52% are delisted
and need the FSDS historical resolution. This orchestrator runs the SAME
guarded resolution used by the index pipeline against the ROSTER's symbols +
listing intervals, producing a down-cap CIK map and a SEPARATE down-cap
companyfacts shares store that
`scripts/build_downcap_membership.py --fundamentals-dir data/fundamentals/equities-downcap`
consumes.

REUSED GUARDS (imported unchanged -- wrong ticker->CIK = wrong company's
shares = contaminated market cap, so the anti-wrong-CIK guards are
load-bearing and are NOT reimplemented here):
- build_cik_map: fetch_company_tickers_raw, build_rows, EXCLUSIONS,
  merge_historical (+ RENAMES via build_rows' module globals).
- build_cik_map_historical: candidates_for, resolve_target, verify_resolution,
  choose_interval, check_rename_consistency, ticker_from_instance, parse_sub,
  QUARTERS (FSDS span 2019q1..2024q1).
- backfill_fundamentals: download (cached FSDS ZIP fetch), RAW_DIR,
  SOURCE_MARKER, _read/_write_source_marker.
- trading.fundamentals.backfill.backfill_from_companyfacts (companyfacts ->
  shares store) + the append-only FundamentalsStore.

ROSTER -> GUARDED-FUNCTION SHAPES (the only new logic here; see the report):
- candidates = roster_symbols(roster, 2019-01-01, 2023-12-31) -- every ticker
  a roster candidate on ANY month-start in the discovery window (reuses the
  down-cap bar-backfill probe). Fed to build_rows as the symbol set.
- tenure = {symbol: (startDate, min(endDate, discovery_end))} -- the SAME
  (lo, hi) tuple shape membership_tenure yields, consumed by resolve_target /
  choose_interval. endDate empty or beyond the window clamps to discovery_end,
  so hi is always a real date (never FAR_FUTURE) and the grace tail applies.

STAGES (--stage): each writes a file the next stage reads, so the heavy
network stages run/resume independently on the run host.
  current    -> data/fundamentals/downcap/cik_map_current.csv  (+ unmapped.csv)
                Live company_tickers.json fetch; build_rows over the candidates.
  historical -> data/fundamentals/downcap/cik_map_historical.csv
                Downloads ~21 FSDS ZIPs (cached, shared data/edgar-raw/;
                already-downloaded ZIPs skipped) and runs the guarded FSDS
                resolution for the ~52% unmapped by `current`. Resumable.
  merge      -> data/fundamentals/downcap/cik_map.csv
                merge_historical(current_rows, historical) -- guarded overlap /
                EXCLUSIONS skip. Reports resolved/unresolved + coverage %.
  shares     -> data/fundamentals/equities-downcap/ (append-only store, its
                OWN .source=companyfacts marker; NEVER mixed with the index
                store data/fundamentals/equities/). Resumable (append-only).
  all        -> current -> historical -> merge -> shares, in order.

--validate spot-checks the built map against known CIKs (AAPL 320193 current,
XLNX 743988, ATVI 718877 historical) and prints them, so the run operator can
confirm resolution correctness before trusting the shares. It does not need
the network (reads the merged map).

RUN-BOOK (on the run host, in order):
    uv run python scripts/build_downcap_fundamentals.py --stage current
    uv run python scripts/build_downcap_fundamentals.py --stage historical
    uv run python scripts/build_downcap_fundamentals.py --stage merge
    uv run python scripts/build_downcap_fundamentals.py --validate
    uv run python scripts/build_downcap_fundamentals.py --stage shares
    # then feed the store to the membership build:
    uv run python scripts/build_downcap_membership.py \
        --fundamentals-dir data/fundamentals/equities-downcap
(`--stage all` runs current->historical->merge->shares in one process; the
split above is preferred on the host so each network stage can be resumed
independently. Re-running current/historical/merge is safe: they overwrite
their outputs deterministically. The shares store is append-only, which gives
mid-run RESUME within one clean run -- but a rerun AFTER a cik_map change (a
corrected resolution, extended EXCLUSIONS, a dropped symbol) must start from an
EMPTY store, else the changed symbol's parquet keeps its stale (wrong-company)
rows. So before re-running --stage shares after any current/historical/merge
change, CLEAR the store first:
    rm -rf data/fundamentals/equities-downcap
stage_shares REFUSES a non-empty store rather than silently mixing regimes.)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from backfill_fundamentals import (  # noqa: E402
    RAW_DIR,
    SOURCE_MARKER,
    _ensure_empty_for_rebuild,
    _read_source_marker,
    _write_source_marker,
    download,
)
from build_cik_map import (  # noqa: E402
    EXCLUSIONS,
    build_rows,
    check_identity_mismatches,
    fetch_company_tickers_raw,
    merge_historical,
)
from build_cik_map_historical import (  # noqa: E402
    QUARTERS,
    candidates_for,
    check_rename_consistency,
    choose_interval,
    parse_sub,
    resolve_target,
    verify_resolution,
)

from trading.fundamentals.backfill import backfill_from_companyfacts  # noqa: E402
from trading.fundamentals.cik_map import load_cik_map  # noqa: E402
from trading.fundamentals.store import FundamentalsStore  # noqa: E402
from trading.venues.universes.downcap_backfill import roster_symbols  # noqa: E402
from trading.venues.universes.downcap_roster import (  # noqa: E402
    parse_supported_tickers,
    structural_roster,
)

ROOT = Path(__file__).resolve().parent.parent

# Discovery window (R3 spec): candidates = roster names listed at any month in
# it; matches build_sic_map.WINDOW_START/END and the down-cap membership build.
DISCOVERY_START = "2019-01-01"
DISCOVERY_END = "2023-12-31"

DEFAULT_ROSTER_ZIP = ROOT / "data" / "tiingo_supported_tickers.zip"

# Output artifacts live under data/ (regenerable, ~10k rows -- NOT committed
# alongside the reviewed index cik_map.csv under src/).
OUT_DIR = ROOT / "data" / "fundamentals" / "downcap"
CURRENT_MAP = OUT_DIR / "cik_map_current.csv"
UNMAPPED_CSV = OUT_DIR / "unmapped.csv"
HISTORICAL_MAP = OUT_DIR / "cik_map_historical.csv"
CIK_MAP = OUT_DIR / "cik_map.csv"

# Down-cap shares store, ISOLATED from the index store: its own directory and
# its own .source marker. backfill_from_companyfacts / FundamentalsStore never
# see the index store data/fundamentals/equities/.
STORE_DIR = ROOT / "data" / "fundamentals" / "equities-downcap"
INDEX_STORE_DIR = ROOT / "data" / "fundamentals" / "equities"

# Known-good resolutions for --validate (run operator's confidence check).
KNOWN_CIKS = {
    "AAPL": (320193, "current (company_tickers.json)"),
    "XLNX": (743988, "historical (FSDS; Xilinx, acquired by AMD)"),
    "ATVI": (718877, "historical (FSDS; Activision Blizzard, acquired by MSFT)"),
}


# --- roster -> guarded-function shapes -------------------------------------


def load_roster(roster_zip: Path) -> pd.DataFrame:
    """Structurally-filtered survivorship-free roster (US common stock on the
    frozen venue set) from the Tiingo supported_tickers ZIP."""
    roster, _report = structural_roster(parse_supported_tickers(roster_zip))
    return roster


def roster_candidates(roster: pd.DataFrame) -> list[str]:
    """Every roster ticker listed on ANY month-start in the discovery window
    -- the survivorship-free candidate set, reusing the down-cap bar-backfill
    probe (roster_symbols) so CIK resolution covers exactly the same names the
    bar cache warms."""
    import datetime

    return roster_symbols(
        roster,
        datetime.date.fromisoformat(DISCOVERY_START),
        datetime.date.fromisoformat(DISCOVERY_END),
    )


def roster_tenure(
    roster: pd.DataFrame, candidates: list[str]
) -> dict[str, tuple[str, str]]:
    """{symbol: (start, end)} in the SAME (lo, hi) shape membership_tenure
    yields for resolve_target / choose_interval. start = the roster listing's
    startDate; end = min(endDate, discovery_end), so hi is always a concrete
    date <= the window end. Note (downcap_roster.py): Tiingo's endDate is the
    file BUILD date even for still-active names (never empty under live data),
    which for an active down-cap name sits after the window -- the `end >
    DISCOVERY_END` clamp pins those to the discovery end. The `endDate or
    DISCOVERY_END` fallback only guards a degenerate empty-endDate row (an
    anomaly, not the normal case). A symbol with multiple roster rows takes
    (min start, max end), matching membership_tenure's aggregation across a
    symbol's intervals."""
    wanted = set(candidates)
    tenure: dict[str, tuple[str, str]] = {}
    for row in roster.itertuples():
        symbol = row.ticker
        if symbol not in wanted:
            continue
        end = row.endDate or DISCOVERY_END
        if end > DISCOVERY_END:
            end = DISCOVERY_END
        start = row.startDate
        if symbol in tenure:
            lo, hi = tenure[symbol]
            tenure[symbol] = (min(lo, start), max(hi, end))
        else:
            tenure[symbol] = (start, end)
    return tenure


def historical_targets(candidates: list[str], current_map: pd.DataFrame) -> list[str]:
    """Candidates with NO current-stage interval, minus the deliberately
    excluded recycled tickers -- the analog of build_cik_map_historical.
    target_symbols, but over the roster candidate set instead of window
    membership."""
    mapped = set(current_map["symbol"])
    return sorted(set(candidates) - mapped - set(EXCLUSIONS))


# --- CSV I/O ---------------------------------------------------------------


def _write_map(path: Path, rows: list[tuple[str, int, str, str]], header: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = list(header)
    lines.append("symbol,cik,start,end")
    lines += [f"{s},{cik},{start},{end}" for s, cik, start, end in rows]
    path.write_text("\n".join(lines) + "\n")


# --- stages ----------------------------------------------------------------


def stage_current(roster_zip: Path) -> None:
    roster = load_roster(roster_zip)
    candidates = roster_candidates(roster)
    print(f"roster: {len(roster)} rows; {len(candidates)} discovery-window candidates")
    current_tickers_raw = fetch_company_tickers_raw()  # live SEC fetch
    current_tickers = {t: info["cik"] for t, info in current_tickers_raw.items()}
    rows, unmapped = build_rows(set(candidates), current_tickers)
    _write_map(
        CURRENT_MAP,
        rows,
        [
            "# Down-cap CURRENT-ticker CIK intervals. GENERATED by",
            "# scripts/build_downcap_fundamentals.py --stage current",
            "# (build_cik_map.build_rows over the down-cap roster candidates).",
        ],
    )
    UNMAPPED_CSV.parent.mkdir(parents=True, exist_ok=True)
    UNMAPPED_CSV.write_text("symbol\n" + "\n".join(unmapped) + "\n")
    pct = len(rows) / len(candidates) * 100 if candidates else 0.0
    print(
        f"current: {len(rows)} mapped ({pct:.1f}% of candidates), "
        f"{len(unmapped)} unmapped -> need FSDS historical resolution"
    )
    print(f"wrote {CURRENT_MAP} and {UNMAPPED_CSV}")
    # Report-only identity audit (reused from build_cik_map): surface any
    # symbol whose resolved CIK disagrees with its OWN direct current listing.
    # More valuable on the ~10x-larger, less-curated roster; does NOT filter
    # rows (the file is already written above).
    check_identity_mismatches(rows, current_tickers_raw)


def stage_historical(roster_zip: Path) -> None:
    if not CURRENT_MAP.exists():
        sys.exit(f"ERROR: {CURRENT_MAP} missing; run --stage current first")
    roster = load_roster(roster_zip)
    candidates = roster_candidates(roster)
    tenure = roster_tenure(roster, candidates)
    current_map = load_cik_map(CURRENT_MAP)
    targets = historical_targets(candidates, current_map)
    print(
        f"historical: {len(targets)} targets (+{len(set(EXCLUSIONS))} excluded "
        f"recycled tickers); downloading/parsing {len(QUARTERS)} FSDS quarters ..."
    )

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    frames = []
    for quarter in QUARTERS:
        zip_path = download(quarter)  # cached: on-disk ZIPs are not re-downloaded
        frames.append(parse_sub(zip_path))
        print(f"  parsed {quarter}: {len(frames[-1])} ticker-prefixed 10-K/10-Q rows")
    sub = pd.concat(frames, ignore_index=True)

    cands = candidates_for(sub, set(targets))
    resolved: dict[str, tuple[int, str, str]] = {}
    unresolved: list[str] = []
    absent: list[str] = []
    for symbol in targets:
        if symbol not in cands:
            absent.append(symbol)
            continue
        got = resolve_target(symbol, cands[symbol], tenure[symbol], current_map)
        if got is None:
            unresolved.append(symbol)
            continue
        resolved[symbol] = got

    print(f"  resolved {len(resolved)} symbols; verifying against submissions JSON ...")
    rows: list[tuple[str, int, str, str, str]] = []
    dropped_verify: list[str] = []
    dropped_interval: list[str] = []
    for i, symbol in enumerate(sorted(resolved), start=1):
        cik, name, _rule = resolved[symbol]
        if not verify_resolution(symbol, cik, name):
            dropped_verify.append(symbol)
            continue
        interval = choose_interval(symbol, current_map, tenure[symbol][1])
        if interval is None:
            dropped_interval.append(symbol)
            continue
        rows.append((symbol, cik, interval[0], interval[1], name))
        if i % 50 == 0:
            print(f"    verified {i}/{len(resolved)}")

    conflicts = check_rename_consistency(rows, current_map)
    if conflicts:
        sys.exit("FATAL: RENAMES pairs disagree on CIK: " + "; ".join(conflicts))

    HISTORICAL_MAP.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Down-cap HISTORICAL (delisted/acquired) symbol->CIK intervals resolved",
        "# from SEC FSDS sub.txt instance prefixes. GENERATED by",
        "# scripts/build_downcap_fundamentals.py --stage historical. The name",
        "# column is FSDS review evidence, not consumed by code.",
        "symbol,cik,start,end,name",
    ]
    lines += [f'{s},{cik},{start},{end},"{name}"' for s, cik, start, end, name in rows]
    HISTORICAL_MAP.write_text("\n".join(lines) + "\n")
    print(
        f"historical: resolved {len(rows)} of {len(targets)} targets "
        f"(absent-from-FSDS={len(absent)}, unresolved={len(unresolved)}, "
        f"dropped-verification={len(dropped_verify)}, "
        f"dropped-degenerate-interval={len(dropped_interval)})"
    )
    print(f"wrote {HISTORICAL_MAP}")


def stage_merge() -> None:
    if not CURRENT_MAP.exists():
        sys.exit(f"ERROR: {CURRENT_MAP} missing; run --stage current first")
    current_map = load_cik_map(CURRENT_MAP)
    current_rows = [
        (row.symbol, int(row.cik), row.start, row.end) for row in current_map.itertuples()
    ]
    if HISTORICAL_MAP.exists():
        historical = pd.read_csv(HISTORICAL_MAP, comment="#", dtype=str).fillna("")
    else:
        print(f"WARNING: {HISTORICAL_MAP} missing; merging current-stage rows only")
        historical = pd.DataFrame(columns=["symbol", "cik", "start", "end"])
    merged, skipped = merge_historical(current_rows, historical)
    _write_map(
        CIK_MAP,
        merged,
        [
            "# Down-cap CIK<->symbol point-in-time intervals. GENERATED by",
            "# scripts/build_downcap_fundamentals.py --stage merge",
            "# (merge_historical of the current + FSDS-historical stages).",
            "# start inclusive, end exclusive, empty end = current EDGAR ticker.",
        ],
    )
    n_current = len(current_rows)
    n_hist = len(merged) - n_current
    print(
        f"merge: {n_current} current + {n_hist} historical = {len(merged)} intervals"
        + (f" (skipped {len(skipped)}: {sorted(set(skipped))})" if skipped else "")
    )
    print(f"wrote {CIK_MAP}")


def stage_shares() -> None:
    """companyfacts shares backfill into the isolated down-cap store.

    Like the index companyfacts path, this REFUSES to run against a non-empty
    store (_ensure_empty_for_rebuild): FundamentalsStore.append() is
    append-only, so a rebuild on top of existing rows would silently KEEP the
    old CIK's shares for every already-stored filed date instead of the new
    map's. That is a correctness hazard specifically across a resolution
    CORRECTION: if a rerun of current/historical/merge drops or re-points a
    symbol (extended EXCLUSIONS, a corrected FSDS resolution), the symbol's
    parquet would still hold the STALE (wrong-company) rows. So a rerun after
    any cik_map change MUST start from an empty store -- clear
    data/fundamentals/equities-downcap/ first (see the run-book). Within a
    SINGLE clean run the append-only store still gives mid-run resume."""
    if not CIK_MAP.exists():
        sys.exit(f"ERROR: {CIK_MAP} missing; run --stage merge first")
    if STORE_DIR.resolve() == INDEX_STORE_DIR.resolve():
        sys.exit(f"FATAL: down-cap store {STORE_DIR} collides with the index store")
    marker = _read_source_marker(STORE_DIR)
    if marker is not None and marker != "companyfacts":
        sys.exit(
            f"FATAL: {STORE_DIR} was built with --source {marker!r} "
            f"(its {SOURCE_MARKER} says so); refusing to mix regimes"
        )
    _ensure_empty_for_rebuild(STORE_DIR)  # non-empty store -> refuse (stale-CIK guard)
    cik_map = load_cik_map(CIK_MAP)
    store = FundamentalsStore(STORE_DIR)  # creates STORE_DIR if needed
    _write_source_marker(STORE_DIR, "companyfacts")
    n_ciks = len(set(cik_map["cik"]))
    print(f"shares: backfilling {n_ciks} CIKs from companyfacts into {STORE_DIR} ...")

    def progress(done: int, total: int) -> None:
        if done % 50 == 0 or done == total:
            print(f"  {done}/{total} CIKs fetched")

    stats = backfill_from_companyfacts(cik_map, store, on_progress=progress)
    print(
        f"shares done: {stats['filers']} filers -> {stats['symbols']} symbols, "
        f"{stats['rows']} rows appended ({stats['dropped']} outside every interval, "
        f"{stats['failed']} CIK fetch failures)"
    )


def run_validate() -> int:
    """Spot-check the merged map against known CIKs. Returns process exit code
    (0 = all known resolutions present & correct)."""
    if not CIK_MAP.exists():
        print(f"validate: {CIK_MAP} not built yet; run --stage merge first")
        return 1
    cik_map = load_cik_map(CIK_MAP)
    by_symbol = {row.symbol: int(row.cik) for row in cik_map.itertuples()}
    ok = True
    print("validate: known-CIK spot check")
    for symbol, (want, note) in KNOWN_CIKS.items():
        got = by_symbol.get(symbol)
        status = "OK" if got == want else "MISMATCH"
        if got != want:
            ok = False
        print(f"  [{status}] {symbol} -> {got} (expected {want}; {note})")
    print("validate: " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--stage",
        choices=["current", "historical", "merge", "shares", "all"],
        help="pipeline stage to run (see the module docstring / run-book)",
    )
    parser.add_argument(
        "--roster-zip",
        type=Path,
        default=DEFAULT_ROSTER_ZIP,
        help=f"Tiingo supported_tickers ZIP (default {DEFAULT_ROSTER_ZIP})",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="spot-check the merged map against known CIKs and exit",
    )
    args = parser.parse_args()

    if args.validate:
        return run_validate()
    if args.stage is None:
        parser.error("one of --stage or --validate is required")

    if args.stage == "current":
        stage_current(args.roster_zip)
    elif args.stage == "historical":
        stage_historical(args.roster_zip)
    elif args.stage == "merge":
        stage_merge()
    elif args.stage == "shares":
        stage_shares()
    elif args.stage == "all":
        stage_current(args.roster_zip)
        stage_historical(args.roster_zip)
        stage_merge()
        stage_shares()
    return 0


if __name__ == "__main__":
    sys.exit(main())
