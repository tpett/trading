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

from trading.config import VENUES, load_venue_config
from trading.data.cache import OhlcvCache
from trading.notify import notify
from trading.pipeline import PipelineDataError, RankingsResult, build_rankings
from trading.runner import RunnerError, restore_from_journal, run_venue
from trading.venues import make_adapter


def _utcnow() -> datetime.datetime:
    """The CLI is the only module allowed to read the clock."""
    return datetime.datetime.now(datetime.UTC)


def _add_store_dirs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config-dir", default="config", help="directory with <venue>.toml")
    parser.add_argument("--state-dir", default="state", help="portfolio state root")
    parser.add_argument("--journal-dir", default="journal", help="journal root")
    parser.add_argument("--digest-dir", default="digest", help="daily digest directory")


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

    run = sub.add_parser("run", help="one live-paper cycle now")
    run.add_argument("--venue", choices=VENUES, required=True)
    run.add_argument("--json", action="store_true", help="machine-readable output")
    run.add_argument(
        "--restore-from-journal",
        action="store_true",
        help="rebuild state/<venue>/portfolio.json from the last journal snapshot (confirms)",
    )
    _add_store_dirs(run)

    digest = sub.add_parser("digest", help="print a daily digest (default: latest)")
    digest.add_argument("--date", default=None, help="digest date, YYYY-MM-DD")
    digest.add_argument("--json", action="store_true", help="machine-readable output")
    digest.add_argument("--digest-dir", default="digest", help="daily digest directory")

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    handlers = {
        "rankings": _cmd_rankings,
        "run": _cmd_run,
        "digest": _cmd_digest,
    }
    return handlers[args.command](args)


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


def _cmd_run(args: argparse.Namespace) -> int:
    config = load_venue_config(args.venue, Path(args.config_dir))
    state_root, journal_root = Path(args.state_dir), Path(args.journal_dir)

    if args.restore_from_journal:
        print(f"This will overwrite {state_root / args.venue / 'portfolio.json'} from the journal.")
        if input("Type RESTORE to confirm: ").strip() != "RESTORE":
            print("aborted")
            return 1
        try:
            print(restore_from_journal(args.venue, state_root, journal_root))
        except RunnerError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        return 0

    adapter = make_adapter(config)
    cache = OhlcvCache(Path(config.data.cache_dir), config.data.refetch_days)
    try:
        outcome = run_venue(
            config,
            adapter,
            cache,
            now=_utcnow(),
            state_root=state_root,
            journal_root=journal_root,
            notify=notify,
            digest_root=Path(args.digest_dir),
        )
    except Exception as exc:  # a silent dead pipeline is the worst failure (spec)
        notify("trading: run crashed", f"{args.venue}: {exc}")
        raise
    if args.json:
        print(
            json.dumps(
                {
                    "venue": outcome.venue,
                    "status": outcome.status,
                    "message": outcome.message,
                    "run_key": outcome.run_key,
                }
            )
        )
    else:
        print(f"{outcome.venue}: {outcome.status} — {outcome.message}")
    return 0 if outcome.status in ("ok", "noop") else 1


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


def _cmd_digest(args: argparse.Namespace) -> int:
    digest_dir = Path(args.digest_dir)
    if args.date:
        path = digest_dir / f"{args.date}.md"
    else:
        candidates = sorted(digest_dir.glob("*.md"))
        path = candidates[-1] if candidates else None
    if path is None or not path.exists():
        print("no digest found", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps({"date": path.stem, "markdown": path.read_text()}))
    else:
        print(path.read_text())
    return 0
