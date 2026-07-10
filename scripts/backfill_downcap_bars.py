"""Warm data/equities-downcap-tiingo/ for every roster candidate in the
discovery span, from Tiingo. Run overnight on the always-on mini (Tiingo
rate-limited). Coverage gaps are printed, never imputed.

Prereq: fetch the roster ZIP once, e.g.
    uv run python -c "from pathlib import Path; \
from trading.venues.universes.downcap_roster import fetch_supported_tickers; \
fetch_supported_tickers(Path('data/tiingo_supported_tickers.zip'))"

    uv run python scripts/backfill_downcap_bars.py --throttle-s 0.5
"""

from __future__ import annotations

import argparse
import datetime
import sys
from pathlib import Path

from trading.config import load_venue_config
from trading.data.cache import OhlcvCache
from trading.venues import make_adapter
from trading.venues.universes.downcap_backfill import roster_symbols, run_backfill
from trading.venues.universes.downcap_roster import (
    parse_supported_tickers,
    structural_roster,
)

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", default="config/experiments/tiingo")
    parser.add_argument("--roster-zip", default="data/tiingo_supported_tickers.zip")
    parser.add_argument("--cache-dir", default="data/equities-downcap-tiingo")
    parser.add_argument("--start", default="2018-01-01")  # 1yr pre-roll for trailing windows
    parser.add_argument("--throttle-s", type=float, default=0.0)
    parser.add_argument("--rate-limit-wait-s", type=float, default=300.0)
    parser.add_argument("--min-coverage", type=float, default=0.60)
    args = parser.parse_args()

    config = load_venue_config("equities", Path(args.config_dir))
    if config.data.bar_source != "tiingo":
        print(
            f"FATAL: down-cap roster is Tiingo-only (fetched from Tiingo's "
            f"supported_tickers ZIP), but {args.config_dir} sets "
            f"data.bar_source={config.data.bar_source!r}",
            file=sys.stderr,
        )
        return 1
    adapter = make_adapter(config)
    cache = OhlcvCache(Path(args.cache_dir), config.data.refetch_days, source="tiingo")

    roster, report = structural_roster(parse_supported_tickers(Path(args.roster_zip)))
    print(f"roster: {len(roster)} candidates; exchange report:")
    print(f"  kept:    {report['kept']}")
    print(f"  dropped: {report['dropped']}")

    start = datetime.date.fromisoformat(args.start)
    end = datetime.date.today()
    symbols = roster_symbols(roster, start, end)
    print(f"backfilling {len(symbols)} symbols into {args.cache_dir} ({start}..{end})")

    def progress(i, n, rep):
        if i % 50 == 0:
            print(f"  {i}/{n} ({rep.fetched} ok, {len(rep.missing)} missing)", flush=True)

    result = run_backfill(
        symbols, cache, adapter, start, end,
        throttle_s=args.throttle_s, rate_limit_wait_s=args.rate_limit_wait_s,
        on_progress=progress,
    )
    print(f"\ncoverage: {result.fetched}/{result.total} = {result.coverage:.1%}")
    if result.missing:
        print(f"source lacks bars for {len(result.missing)} symbol(s) (recorded gaps)")
    if result.errors:
        print(f"\n{len(result.errors)} hard error(s):")
        for line in result.errors[:20]:
            print(f"  {line}")
    if result.coverage < args.min_coverage:
        print(f"\nWARN: coverage {result.coverage:.1%} < {args.min_coverage:.0%} floor")
        return 1
    print("\nbackfill done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
