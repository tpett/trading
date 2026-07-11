"""Build the down-cap band-membership CSV (the science artifact the three
UniverseSpecs consume) and the diagnostics CSV (the Phase-A gate input) from
the backfilled cache + fundamentals store.

    uv run python scripts/build_downcap_membership.py
    # dollar-volume-only fallback (only after A5 records the amendment):
    uv run python scripts/build_downcap_membership.py --no-cap-band
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from trading.alphasearch.panel import load_bars
from trading.fundamentals.store import FundamentalsStore
from trading.venues.universes.downcap_membership import (
    build_band_membership,
    write_membership,
)
from trading.venues.universes.downcap_roster import parse_supported_tickers, structural_roster

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--roster-zip", default="data/tiingo_supported_tickers.zip")
    p.add_argument("--cache-dir", default="data/equities-downcap-tiingo")
    p.add_argument("--fundamentals-dir", default="data/fundamentals/equities")
    p.add_argument("--out-membership", default="data/equities-downcap-tiingo/band_membership.csv")
    p.add_argument("--out-diagnostics", default="data/equities-downcap-tiingo/diagnostics.csv")
    p.add_argument("--no-cap-band", action="store_true", help="dollar-volume-only fallback")
    args = p.parse_args()

    roster, report = structural_roster(parse_supported_tickers(Path(args.roster_zip)))
    print(f"roster: {len(roster)} candidates; kept exchanges {report['kept']}")
    bars = load_bars(Path(args.cache_dir), sorted(roster["ticker"]))
    print(f"loaded bars for {len(bars)} candidates from {args.cache_dir}")
    store = FundamentalsStore(Path(args.fundamentals_dir))
    build = build_band_membership(roster, bars, store, require_cap_band=not args.no_cap_band)
    write_membership(build, Path(args.out_membership), Path(args.out_diagnostics))
    print(
        f"wrote {len(build.membership)} intervals -> {args.out_membership}; "
        f"{len(build.diagnostics)} candidate-month rows -> {args.out_diagnostics}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
