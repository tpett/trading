"""Warm the OHLCV cache for every PIT-universe symbol from the configured
bar source, and REPORT coverage explicitly.

Why a dedicated pass instead of letting the walk-forward fetch lazily: on a
COLD cache a symbol the source cannot serve raises DataFetchError, which the
engine absorbs as "missing" -- a silent coverage gap. For a survivorship-
free re-run that silence is the exact failure we are trying to remove, so we
fetch every historical member up front and print which symbols the source
lacks. Only once coverage is known-complete should the walk-forward run on
the warm cache (where DataFetchError is then correctly treated as a gap, not
a loss -- see OhlcvCache.fetch).

    uv run python scripts/backfill_bars.py --config-dir config/experiments/tiingo
"""

from __future__ import annotations

import argparse
import datetime
import sys
from pathlib import Path

import pandas as pd

from trading.config import load_venue_config
from trading.data.cache import OhlcvCache
from trading.symbols import resolution_collisions
from trading.venues import make_adapter
from trading.venues.base import DataFetchError, RateLimitError
from trading.venues.equities import DEFAULT_MEMBERSHIP_CSV

ROOT = Path(__file__).resolve().parent.parent


def _sleep(seconds: float) -> None:
    """Sleep touchpoint, isolated for tests."""
    import time

    time.sleep(seconds)


def _fetch_waiting_on_rate_limit(cache, adapter, symbol, start, end, wait_s):
    """cache.fetch, but a rate-limit rejection waits and retries the SAME
    symbol instead of surfacing -- so a metered plan slows the backfill
    rather than punching coverage holes. A genuine miss (404 -> empty ->
    DataFetchError, which is NOT a RateLimitError) propagates immediately."""
    while True:
        try:
            return cache.fetch(symbol, start, end, adapter.fetch_ohlcv)
        except RateLimitError:
            print(f"  rate-limited on {symbol}; waiting {wait_s:.0f}s", flush=True)
            _sleep(wait_s)


def historical_symbols(adapter, start: datetime.date, end: datetime.date) -> list[str]:
    """Every symbol that was a universe member on ANY month-start in the
    window -- the survivorship-free set, including names delisted mid-window.
    Monthly sampling is enough: membership intervals are far longer than a
    month, so no member is missed between samples."""
    seen: set[str] = set()
    day = start
    while day <= end:
        seen.update(info.symbol for info in adapter.universe(day))
        year, month = (day.year + 1, 1) if day.month == 12 else (day.year, day.month + 1)
        day = datetime.date(year, month, 1)
    return sorted(seen)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", default="config")
    parser.add_argument("--venue", default="equities")
    parser.add_argument(
        "--min-coverage",
        type=float,
        default=0.95,
        help="fail if fewer than this fraction of symbols returned bars",
    )
    parser.add_argument(
        "--throttle-s",
        type=float,
        default=0.0,
        help="min seconds between symbol fetches; set ~75 to stay under a "
        "50 req/hour free-tier cap (0 = full speed for an unmetered plan)",
    )
    parser.add_argument(
        "--rate-limit-wait-s",
        type=float,
        default=300.0,
        help="on a 429 (hourly cap hit), wait this long and RETRY the same "
        "symbol rather than dropping it into a silent coverage gap",
    )
    args = parser.parse_args()

    config = load_venue_config(args.venue, Path(args.config_dir))
    adapter = make_adapter(config)
    cache = OhlcvCache(
        Path(config.data.cache_dir), config.data.refetch_days, config.data.bar_source
    )

    # Warn loudly if the active universe has entity collisions (a company that
    # enters the ranking twice -- cross-index membership double-listing, a few
    # even in the default sp500+ndx universe, plus more under sp400). Surfaced,
    # not yet auto-fixed; see trading.symbols.resolution_collisions.
    if args.venue == "equities" and config.data.bar_source == "tiingo":
        membership = pd.read_csv(DEFAULT_MEMBERSHIP_CSV, comment="#", dtype=str).fillna("")
        collisions = resolution_collisions(membership, config.universe.indices)
        if collisions:
            print(
                f"WARNING: {len(collisions)} rename/successor membership collision(s) under "
                f"indices {config.universe.indices} -- the same company will be double-counted "
                "in the ranking (static resolution is not point-in-time). Affected:",
                flush=True,
            )
            for c in collisions:
                print(f"  {c['old']}->{c['new']}: {c['old_window']} overlaps {c['new_window']}")

    start = config.backtest.start
    end = datetime.date.today()
    fetch_start = start - datetime.timedelta(days=config.data.history_days)
    benchmark = config.benchmark
    symbols = historical_symbols(adapter, start, end)
    if benchmark not in symbols:
        symbols.append(benchmark)

    print(
        f"backfilling {len(symbols)} symbols from {config.data.bar_source} "
        f"into {config.data.cache_dir} ({fetch_start}..{end})"
    )
    fetched = 0
    missing: list[str] = []
    errors: list[str] = []
    for i, symbol in enumerate(symbols, 1):
        if args.throttle_s and i > 1:
            _sleep(args.throttle_s)
        try:
            df = _fetch_waiting_on_rate_limit(
                cache, adapter, symbol, fetch_start, end, args.rate_limit_wait_s
            )
            fetched += 1 if not df.empty else 0
            if df.empty:
                missing.append(symbol)
        except DataFetchError:
            # NOT a RateLimitError (those wait+retry inside): a genuine
            # "no such ticker" (404 -> empty -> raised by the adapter).
            missing.append(symbol)
        except Exception as exc:  # noqa: BLE001 - report, don't abort the whole pass
            errors.append(f"{symbol}: {type(exc).__name__}: {exc}")
        if i % 25 == 0:
            print(f"  {i}/{len(symbols)} ({fetched} ok, {len(missing)} missing)", flush=True)

    coverage = fetched / len(symbols) if symbols else 0.0
    print(f"\ncoverage: {fetched}/{len(symbols)} = {coverage:.1%}")
    if missing:
        print(f"source lacks bars for {len(missing)} symbol(s): {', '.join(missing[:40])}")
        if len(missing) > 40:
            print(f"  ... and {len(missing) - 40} more")
    if errors:
        print(f"\n{len(errors)} hard error(s) (investigate before trusting the run):")
        for line in errors[:20]:
            print(f"  {line}")
    if benchmark in missing:
        print(f"\nFATAL: benchmark {benchmark} has no bars; the walk-forward cannot run")
        return 1
    if coverage < args.min_coverage:
        print(f"\nFAIL: coverage {coverage:.1%} < {args.min_coverage:.0%} floor")
        return 1
    print("\nbackfill OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
