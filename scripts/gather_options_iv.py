"""CLI: gather historical EOD option quotes + IV/skew into samples.jsonl.

Drives ``trading.research.options_gather`` against a local ThetaData v3 terminal.
The library holds all the logic (universe ranking, expiration/strike selection,
quote fetch, IV inversion, resumable append); this script is just argument
parsing, logging setup, and a wrapped ``ThetaClient``.

Typical overnight run (top 100 liquid names, 2019-2025 monthly decisions):

    uv run python scripts/gather_options_iv.py

Smoke test against one name (still hits the terminal, but tiny):

    uv run python scripts/gather_options_iv.py --limit-symbols AAPL

The run is resumable: re-invoking with the same --out skips (symbol,
decision_date) cells already written, so a killed job just picks up where it
left off.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from trading.research.options_gather import ThetaClient, run_gather
from trading.venues.equities import DEFAULT_MEMBERSHIP_CSV

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "data" / "options-iv" / "samples.jsonl"
DEFAULT_CACHE_DIR = ROOT / "data" / "equities-tiingo"
DEFAULT_RAW_DIR = ROOT / "data" / "options-iv" / "underlying_raw"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-url", default="http://127.0.0.1:25503", help="ThetaData v3 terminal"
    )
    parser.add_argument("--universe-size", type=int, default=100, help="top-N liquid names")
    parser.add_argument("--start-month", default="2019-01", help="first decision month YYYY-MM")
    parser.add_argument("--end-month", default="2025-12", help="last decision month (YYYY-MM)")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="samples.jsonl output path")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR, help="Tiingo parquets")
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=DEFAULT_RAW_DIR,
        help="raw (unadjusted) close cache for spot (backfill_options_underlying_raw)",
    )
    parser.add_argument(
        "--membership", type=Path, default=DEFAULT_MEMBERSHIP_CSV, help="PIT membership CSV"
    )
    parser.add_argument(
        "--limit-symbols",
        nargs="+",
        default=None,
        help="explicit symbol list (smoke test); bypasses universe ranking",
    )
    parser.add_argument("--max-workers", type=int, default=4, help="in-flight cap (terminal max 4)")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s"
    )

    client = ThetaClient(args.base_url)
    summary = run_gather(
        client,
        args.out,
        symbols=args.limit_symbols,
        universe_size=args.universe_size,
        start_month=args.start_month,
        end_month=args.end_month,
        cache_dir=args.cache_dir,
        raw_dir=args.raw_dir,
        membership_csv=args.membership,
        max_workers=args.max_workers,
    )
    print(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
