"""Command-line entry point (spec: CLI & README).

M1 ships `trading rankings`. Later milestones add run/status/backtest/digest/
schedule/reset-breaker on the same parser. Human-readable rich tables by
default, --json for machine consumption. The CLI is the only module allowed
to read the clock; everything below it takes as_of as a parameter.
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path

import pandas as pd

from trading.config import load_venue_config
from trading.data.cache import OhlcvCache
from trading.pipeline import PipelineDataError, RankingsResult, build_rankings
from trading.venues import make_adapter

VENUES = ["equities", "crypto"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="trading", description="Momentum swing trading system")
    sub = parser.add_subparsers(dest="command", required=True)

    rankings = sub.add_parser("rankings", help="current ranked table with sub-scores")
    rankings.add_argument("--venue", choices=VENUES, required=True)
    rankings.add_argument(
        "--as-of",
        type=datetime.date.fromisoformat,
        default=None,
        help="decision date, YYYY-MM-DD (default: today UTC)",
    )
    rankings.add_argument(
        "--top", type=int, default=25, help="rows to display, 0 = all (table output only)"
    )
    rankings.add_argument("--json", action="store_true", help="machine-readable output")
    rankings.add_argument("--config-dir", default="config", help="directory with <venue>.toml")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "rankings":
        return _cmd_rankings(args)
    return 2  # unreachable: subparsers are required


def _cmd_rankings(args: argparse.Namespace) -> int:
    config = load_venue_config(args.venue, Path(args.config_dir))
    adapter = make_adapter(config)
    cache = OhlcvCache(Path(config.data.cache_dir), config.data.refetch_days)
    as_of = args.as_of or datetime.datetime.now(datetime.UTC).date()
    try:
        result = build_rankings(config, adapter, cache, as_of)
    except PipelineDataError as exc:
        print(f"WARNING: {exc}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(_to_json(result), indent=2))
    else:
        _render(result, top=args.top)
    return 0


def _to_json(result: RankingsResult) -> dict:
    rankings = []
    for pos, (symbol, row) in enumerate(result.table.iterrows(), start=1):
        entry: dict[str, object] = {"rank": pos, "symbol": symbol, "status": row["status"]}
        for col in result.table.columns:
            if col == "status":
                continue
            value = row[col]
            entry[col] = None if pd.isna(value) else round(float(value), 4)
        rankings.append(entry)
    return {
        "venue": result.venue,
        "as_of": result.as_of.date().isoformat(),
        "regime": {
            "state": result.regime.state,
            "exposure_multiplier": result.regime.exposure_multiplier,
        },
        "coverage": {
            "requested": result.coverage.requested,
            "fetched": result.coverage.fetched,
            "ratio": round(result.coverage.ratio, 4),
        },
        "quarantined": list(result.quarantined),
        "fetch_failures": list(result.fetch_failures),
        "insufficient_history": list(result.insufficient_history),
        "rankings": rankings,
    }


def _render(result: RankingsResult, top: int) -> None:
    from rich.console import Console
    from rich.table import Table

    console = Console()
    console.print(
        f"[bold]{result.venue}[/bold] rankings as of {result.as_of.date().isoformat()} | "
        f"regime: [bold]{result.regime.state}[/bold] "
        f"(exposure x{result.regime.exposure_multiplier})"
    )
    value_columns = [c for c in result.table.columns if c != "status"]
    table = Table()
    table.add_column("#", justify="right")
    table.add_column("symbol")
    table.add_column("status")
    for col in value_columns:
        table.add_column(col, justify="right")
    rows = result.table if top == 0 else result.table.head(top)
    for pos, (symbol, row) in enumerate(rows.iterrows(), start=1):
        cells = [str(pos), str(symbol), str(row["status"])]
        cells += ["" if pd.isna(row[c]) else f"{row[c]:.3f}" for c in value_columns]
        table.add_row(*cells)
    console.print(table)
    console.print(
        f"coverage {result.coverage.fetched}/{result.coverage.requested} "
        f"({result.coverage.ratio:.0%})"
    )
    if result.quarantined:
        console.print(f"[yellow]quarantined:[/yellow] {', '.join(result.quarantined)}")
    if result.fetch_failures:
        console.print(f"[yellow]fetch failures:[/yellow] {', '.join(result.fetch_failures)}")
    if result.insufficient_history:
        console.print(
            f"[yellow]insufficient history:[/yellow] {', '.join(result.insufficient_history)}"
        )
