"""Backfill RAW (unadjusted) daily closes for the options-IV gather universe.

Why a SEPARATE raw cache from ``data/equities-tiingo``
-----------------------------------------------------
The main equities cache stores split+dividend-ADJUSTED closes (AAPL 2019-02-01
prints ~$39.51). ThetaData option strikes, however, are RAW/unadjusted (real
2019 AAPL strikes sit near $165), and Black-Scholes must invert a raw strike
against a raw spot. Feeding the gather an adjusted spot mis-snaps the strike
ladder by the cumulative adjustment factor -- an "ATM" strike lands ~$110
in-the-money, every quote is all-intrinsic, and every IV comes back null.

This script pulls Tiingo's RAW ``close`` field for exactly the names the gather
will process and caches one ``close_raw`` parquet per symbol. The gather reads
it for SPOT ONLY; the adjusted cache still supplies the decision calendar and
the dollar-volume ranking (both adjustment-invariant).

    uv run python scripts/backfill_options_underlying_raw.py --limit-symbols AAPL

Incremental: a symbol whose parquet already exists is skipped unless --force, so
a restart resumes cheaply.
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import sys
from collections.abc import Callable
from pathlib import Path

import pandas as pd

from trading.research.options_gather import build_universe
from trading.venues.equities import (
    _TIINGO_URL,
    DEFAULT_MEMBERSHIP_CSV,
    _tiingo_get_retrying,
)

log = logging.getLogger("scripts.backfill_options_underlying_raw")

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT_DIR = ROOT / "data" / "options-iv" / "underlying_raw"
DEFAULT_CACHE_DIR = ROOT / "data" / "equities-tiingo"


def parse_raw_close(rows: list[dict]) -> pd.DataFrame:
    """Tiingo daily/prices rows -> a single-column ``close_raw`` frame.

    ``date`` (ISO) becomes a UTC-midnight DatetimeIndex named ``Date`` and the
    RAW (unadjusted) ``close`` field becomes ``close_raw`` -- the same index
    shape ``load_raw_close`` expects.
    """
    df = pd.DataFrame(rows)
    idx = pd.to_datetime(df["date"], utc=True).dt.normalize()
    out = pd.DataFrame({"close_raw": df["close"].astype(float).to_numpy()}, index=idx)
    out.index.name = "Date"
    return out


def run_backfill(
    symbols: list[str],
    out_dir: Path,
    start: datetime.date,
    end: datetime.date,
    *,
    force: bool = False,
    fetch: Callable[[str, dict], tuple[int, bytes]] = _tiingo_get_retrying,
) -> dict:
    """Pull raw closes for ``symbols`` into ``out_dir`` (one parquet each).

    Returns a summary dict. Per-symbol failures never abort the pass: a non-200
    status or a 200-but-empty response is logged and counted, and the loop
    continues -- exactly the fail-open the overnight backfill needs.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    counters = {"pulled": 0, "skipped": 0, "empty": 0, "errors": 0}
    params = {"startDate": start.isoformat(), "endDate": end.isoformat(), "format": "json"}
    total = len(symbols)
    for i, symbol in enumerate(symbols, 1):
        path = out_dir / f"{symbol}.parquet"
        if path.exists() and not force:
            counters["skipped"] += 1
            continue
        status, body = fetch(_TIINGO_URL.format(symbol=symbol), params)
        if status != 200:
            counters["errors"] += 1
            log.warning("%s: HTTP %d -- skipping (%r)", symbol, status, body[:120])
            continue
        rows = json.loads(body)
        if not rows:
            counters["empty"] += 1
            log.warning("%s: 200 but no rows in range -- skipping", symbol)
            continue
        parse_raw_close(rows).to_parquet(path)
        counters["pulled"] += 1
        if i % 25 == 0 or i == total:
            log.info(
                "progress %d/%d (pulled=%d skipped=%d empty=%d errors=%d)",
                i,
                total,
                counters["pulled"],
                counters["skipped"],
                counters["empty"],
                counters["errors"],
            )
    summary = {"symbols": total, **counters}
    log.info("raw-underlying backfill done: %s", summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--universe-size", type=int, default=100, help="top-N liquid names")
    parser.add_argument(
        "--limit-symbols",
        nargs="+",
        default=None,
        help="explicit symbol list (bypasses universe ranking, like the gather)",
    )
    parser.add_argument(
        "--start-date",
        default="2018-06-01",
        help="first raw bar to pull (pad before the 2019 decision window)",
    )
    parser.add_argument("--end-date", default="2026-07-31", help="last raw bar to pull")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="raw close cache dir")
    parser.add_argument(
        "--membership", type=Path, default=DEFAULT_MEMBERSHIP_CSV, help="PIT membership CSV"
    )
    parser.add_argument(
        "--cache-dir", type=Path, default=DEFAULT_CACHE_DIR, help="adjusted Tiingo parquets"
    )
    parser.add_argument("--force", action="store_true", help="re-pull symbols already cached")
    parser.add_argument(
        "--indices",
        nargs="+",
        default=["sp500", "ndx"],
        help="membership indices for the universe (e.g. sp400 for mid-caps); match the gather",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s"
    )

    if args.limit_symbols is not None:
        symbols = args.limit_symbols
    else:
        # Same universe the gather uses, so the raw cache covers exactly the
        # names that will be gathered.
        symbols = build_universe(
            args.membership, args.cache_dir, args.universe_size, indices=tuple(args.indices)
        )
        log.info("built universe of %d symbols", len(symbols))

    summary = run_backfill(
        symbols,
        args.out_dir,
        datetime.date.fromisoformat(args.start_date),
        datetime.date.fromisoformat(args.end_date),
        force=args.force,
    )
    print(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
