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

from trading.backtest.engine import BacktestError, BacktestResult, prepare, replay
from trading.backtest.experiments import (
    experiment_count,
    experiments_journal,
    log_experiment,
    prior_holdout,
)
from trading.backtest.metrics import BacktestMetrics, compute_metrics
from trading.backtest.walkforward import WalkForwardError, WalkForwardResult, run_walk_forward
from trading.config import VENUES, load_venue_config
from trading.data.cache import OhlcvCache
from trading.journal import Journal, JournalError
from trading.notify import notify
from trading.pipeline import PipelineDataError, RankingsResult, build_rankings
from trading.runner import (
    RunLock,
    RunnerError,
    load_state,
    lock_path,
    restore_from_journal,
    run_venue,
    save_state,
    state_path,
)
from trading.simulator.state import StateError
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

    status = sub.add_parser("status", help="portfolios, P&L vs benchmark, last-run health")
    status.add_argument("--json", action="store_true", help="machine-readable output")
    status.add_argument("--state-dir", default="state", help="portfolio state root")
    status.add_argument("--journal-dir", default="journal", help="journal root")

    breaker = sub.add_parser("reset-breaker", help="manually reset the circuit breaker (confirms)")
    breaker.add_argument("--venue", choices=VENUES, required=True)
    breaker.add_argument("--state-dir", default="state", help="portfolio state root")
    breaker.add_argument("--journal-dir", default="journal", help="journal root")

    backtest = sub.add_parser("backtest", help="historical replay of the live simulator")
    backtest.add_argument("--venue", choices=VENUES, required=True)
    backtest.add_argument(
        "--from",
        dest="from_date",
        type=datetime.date.fromisoformat,
        default=None,
        help="start date (default: [backtest].start)",
    )
    backtest.add_argument(
        "--to",
        dest="to_date",
        type=datetime.date.fromisoformat,
        default=None,
        help="end date (default and cap: yesterday UTC)",
    )
    backtest.add_argument(
        "--walk-forward",
        action="store_true",
        help="tune the two hyperparameters per rolling window; report stitched OOS only",
    )
    backtest.add_argument(
        "--holdout",
        action="store_true",
        help="evaluate the final holdout ONCE with current TOML params (confirms on rerun)",
    )
    backtest.add_argument("--json", action="store_true", help="machine-readable output")
    backtest.add_argument("--config-dir", default="config", help="directory with <venue>.toml")
    backtest.add_argument("--journal-dir", default="journal", help="journal root")

    sched = sub.add_parser("schedule", help="manage launchd jobs")
    sched.add_argument("action", choices=["install", "status", "remove"])
    sched.add_argument(
        "--agents-dir", default=None, help="LaunchAgents dir (default ~/Library/LaunchAgents)"
    )
    sched.add_argument("--json", action="store_true", help="machine-readable output")

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    handlers = {
        "rankings": _cmd_rankings,
        "run": _cmd_run,
        "digest": _cmd_digest,
        "status": _cmd_status,
        "reset-breaker": _cmd_reset_breaker,
        "schedule": _cmd_schedule,
        "backtest": _cmd_backtest,
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


def _venue_status(venue: str, state_dir: Path, journal_dir: Path, now: datetime.datetime) -> dict:
    info: dict[str, object] = {"venue": venue, "state": "ok"}
    try:
        state = load_state(state_path(state_dir, venue))
    except StateError:
        return {"venue": venue, "state": "corrupt"}
    if state is None:
        return {"venue": venue, "state": "not bootstrapped"}
    info["breaker_tripped"] = state.breaker_tripped
    info["positions"] = len(state.positions)

    journal = Journal(journal_dir / f"{venue}.jsonl")
    try:
        last_run = journal.last_event(types=frozenset({"run"}))
        last_ok = journal.last_event(types=frozenset({"run", "bootstrap"}))
    except JournalError:
        return {"venue": venue, "state": "journal corrupt"}
    if last_run is not None:
        snapshot = last_run["snapshot"]
        start = float(last_run["starting_balance"])
        bench = last_run["benchmark"]
        info["value"] = float(snapshot["value"])
        info["pnl_pct"] = float(snapshot["value"]) / start - 1.0
        info["benchmark_pnl_pct"] = float(bench["close"]) / float(bench["start_price"]) - 1.0
    if last_ok is not None:
        last_ts = datetime.datetime.fromisoformat(last_ok["ts"])
        info["hours_since_last_success"] = (now - last_ts).total_seconds() / 3600
    return info


def _cmd_status(args: argparse.Namespace) -> int:
    now = _utcnow()
    venues = [_venue_status(v, Path(args.state_dir), Path(args.journal_dir), now) for v in VENUES]
    if args.json:
        print(json.dumps({"as_of": now.isoformat(), "venues": venues}))
        return 0

    from rich.console import Console
    from rich.table import Table

    table = Table(title=f"trading status — {now.isoformat(timespec='seconds')}")
    for col in ["venue", "state", "value", "P&L", "benchmark", "positions", "breaker", "last run"]:
        table.add_column(col)
    for v in venues:
        table.add_row(
            str(v["venue"]),
            str(v["state"]),
            f"${v['value']:,.2f}" if "value" in v else "-",
            f"{v['pnl_pct']:+.2%}" if "pnl_pct" in v else "-",
            f"{v['benchmark_pnl_pct']:+.2%}" if "benchmark_pnl_pct" in v else "-",
            str(v.get("positions", "-")),
            "TRIPPED" if v.get("breaker_tripped") else "armed",
            f"{v['hours_since_last_success']:.1f}h ago"
            if "hours_since_last_success" in v
            else "never",
        )
    Console().print(table)
    return 0


def _cmd_reset_breaker(args: argparse.Namespace) -> int:
    state_root, journal_root = Path(args.state_dir), Path(args.journal_dir)
    # Same lock a scheduled run holds: without it, a run in flight can
    # re-persist the tripped breaker right after this command clears it, or
    # interleave journal appends and trip the torn-tail repair against the
    # other process's partial flush. Refuse loudly rather than race.
    lock = RunLock(lock_path(state_root, args.venue))
    if not lock.acquire():
        print(f"ERROR: another run is in progress for {args.venue}", file=sys.stderr)
        return 1
    try:
        path = state_path(state_root, args.venue)
        try:
            state = load_state(path)
        except StateError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        if state is None:
            print(f"no state for {args.venue}; nothing to reset", file=sys.stderr)
            return 1
        journal = Journal(journal_root / f"{args.venue}.jsonl")
        last_run = journal.last_event(types=frozenset({"run"}))
        # Same fail-safe as run_venue: mutating a state file that is behind the
        # journal would bake the divergence in. Refuse until reconciled.
        if last_run is not None and state.last_run_key != last_run["run_key"]:
            print(
                f"state file behind journal; run "
                f"'trading run --venue {args.venue} --restore-from-journal' first",
                file=sys.stderr,
            )
            return 1
        if not state.breaker_tripped:
            print(f"{args.venue}: breaker is not tripped")
            return 0
        print(f"Circuit breaker for {args.venue} tripped at {state.breaker_tripped_at}.")
        try:
            answer = input("Type RESET to re-enable entries: ").strip()
        except EOFError:  # non-interactive stdin: treat as a refusal, not a crash
            answer = ""
        if answer != "RESET":
            print("aborted")
            return 1
        if last_run is not None:
            # Rebase the high-water mark to the last marked value; otherwise the
            # unchanged HWM re-trips the breaker on the very next run.
            state.high_water_mark = float(last_run["snapshot"]["value"])
        state.breaker_tripped = False
        state.breaker_tripped_at = None
        # Journal-FIRST (same crash-safe ordering as run_venue): a crash between
        # the two writes leaves state behind the journal — the recoverable
        # direction, and restore_from_journal re-applies this event. The reverse
        # order would clear the breaker with no journal record, and a later
        # restore would silently re-trip a breaker the operator explicitly reset.
        try:
            journal.append(
                {
                    "event": "breaker_reset",
                    "venue": args.venue,
                    "ts": _utcnow().isoformat(),
                    "high_water_mark": state.high_water_mark,
                    "last_run_key": state.last_run_key,
                }
            )
        except Exception as exc:
            print(f"ERROR: journal append failed ({exc}); state untouched", file=sys.stderr)
            return 1
        try:
            save_state(path, state)
        except Exception as exc:
            print(
                f"ERROR: state write failed after journal append ({exc}); recover with "
                f"'trading run --venue {args.venue} --restore-from-journal'",
                file=sys.stderr,
            )
            return 1
        print(f"{args.venue}: breaker reset; entries re-enabled")
        return 0
    finally:
        lock.release()


def _cmd_schedule(args: argparse.Namespace) -> int:
    from trading import schedule

    agents_dir = (
        Path(args.agents_dir) if args.agents_dir else (Path.home() / "Library" / "LaunchAgents")
    )
    try:
        if args.action == "install":
            output: object = schedule.install(Path.cwd(), agents_dir)
        elif args.action == "remove":
            output = schedule.remove(agents_dir)
        else:
            output = schedule.status(agents_dir)
    except schedule.ScheduleError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(output))
    elif isinstance(output, dict):
        for venue, info in output.items():
            state = (
                "loaded"
                if info["loaded"]
                else ("installed, not loaded" if info["installed"] else "not installed")
            )
            print(f"{venue}: {state}")
    else:
        for line in output:
            print(line)
    return 0


def _cmd_backtest(args: argparse.Namespace) -> int:
    if args.walk_forward and args.holdout:
        print(
            "ERROR: --walk-forward tunes; --holdout is a single evaluation of the "
            "tuned TOML params. They cannot be combined.",
            file=sys.stderr,
        )
        return 1
    if args.holdout and (args.from_date is not None or args.to_date is not None):
        print(
            "ERROR: --holdout is a fixed final window ([backtest].holdout_start "
            "through yesterday); it cannot be combined with --from or --to (a "
            "partial evaluation would still spend the once-only holdout).",
            file=sys.stderr,
        )
        return 1
    config = load_venue_config(args.venue, Path(args.config_dir))
    bt = config.backtest
    now = _utcnow()
    yesterday = now.date() - datetime.timedelta(days=1)
    journal = experiments_journal(Path(args.journal_dir), args.venue)

    end = min(args.to_date or yesterday, yesterday)
    prior = None
    if args.holdout:
        prior = prior_holdout(journal)
        if prior is not None:
            # stderr, like the clamp note: stdout stays a pure --json channel.
            print(
                f"Holdout already evaluated at {prior['ts']} (config {prior['config_hash']}, "
                f"result journaled). The holdout is spent the first time it is read;",
                file=sys.stderr,
            )
            print("rerunning it invalidates the go-live evidence (spec).", file=sys.stderr)
            try:
                answer = input("Type RERUN HOLDOUT to run it anyway: ").strip()
            except EOFError:
                answer = ""
            if answer != "RERUN HOLDOUT":
                print("aborted", file=sys.stderr)
                return 1
        start = bt.holdout_start
    else:
        start = args.from_date or bt.start
        boundary = bt.holdout_start - datetime.timedelta(days=1)
        if start > boundary:
            print(
                f"ERROR: --from {start} is inside the final holdout "
                f"(from {bt.holdout_start}); use --holdout for its one evaluation",
                file=sys.stderr,
            )
            return 1
        if end > boundary:
            print(f"note: --to clamped to {boundary} (holdout stays untouched)", file=sys.stderr)
            end = boundary
    if start >= end:
        print(f"ERROR: empty date range {start}..{end}", file=sys.stderr)
        return 1

    adapter = make_adapter(config)
    cache = OhlcvCache(Path(config.data.cache_dir), config.data.refetch_days)
    try:
        prepared = prepare(config, adapter, cache, start, end)
    except BacktestError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if args.walk_forward:
        return _run_walk_forward_command(prepared, config, journal, args, start, end, now)
    return _run_plain_backtest_command(prepared, config, journal, args, start, end, now, prior)


def _run_plain_backtest_command(prepared, config, journal, args, start, end, now, prior) -> int:
    try:
        result = replay(prepared, config)
    except BacktestError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    metrics = compute_metrics(result, config.backtest.periods_per_year)
    kind = "holdout" if args.holdout else "backtest"
    extra: dict = {"missing_symbols": len(prepared.missing_symbols)}
    if prior is not None:
        # A confirmed rerun of a spent holdout: mark it explicitly so the
        # journal shows more than a duplicate ts, and point at what it spent.
        extra.update(rerun=True, prior_ts=prior["ts"])
    log_experiment(
        journal,
        config=config,
        kind=kind,
        start=start,
        end=end,
        metrics=metrics,
        ts=now.isoformat(),
        grid_point={
            "entry_score_threshold": config.portfolio.entry_score_threshold,
            "stop_atr_multiple": config.portfolio.stop_atr_multiple,
        },
        survivorship_ratio=result.survivorship_ratio,
        extra=extra,
    )
    count = experiment_count(journal, config.name)
    if args.json:
        print(json.dumps(_backtest_json(result, metrics, count, start, end, kind)))
    else:
        _render_backtest(result, metrics, count, kind)
    return 0


def _run_walk_forward_command(prepared, config, journal, args, start, end, now) -> int:
    try:
        wf = run_walk_forward(prepared, config, start=start, end=end)
    except (WalkForwardError, BacktestError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    for wr in wf.windows:
        log_experiment(
            journal,
            config=config,
            kind="walk_forward_window",
            start=wr.window.test_start,
            end=wr.window.test_end - datetime.timedelta(days=1),
            metrics=wr.test_metrics,
            ts=now.isoformat(),
            grid_point={
                "entry_score_threshold": wr.best.entry_score_threshold,
                "stop_atr_multiple": wr.best.stop_atr_multiple,
            },
            survivorship_ratio=wr.test_result.survivorship_ratio,
        )
    log_experiment(
        journal,
        config=config,
        kind="walk_forward",
        start=start,
        end=end,
        metrics=wf.stitched_metrics,
        ts=now.isoformat(),
        extra={
            "windows": len(wf.windows),
            "stress_segments_covered": list(wf.stress_segments_covered),
            "missing_symbols": len(prepared.missing_symbols),
        },
    )
    count = experiment_count(journal, config.name)
    if args.json:
        payload = {
            "venue": config.name,
            "from": start.isoformat(),
            "to": end.isoformat(),
            "kind": "walk_forward",
            "windows": [
                {
                    "test_from": wr.window.test_start.isoformat(),
                    "test_to": (wr.window.test_end - datetime.timedelta(days=1)).isoformat(),
                    "grid_point": {
                        "entry_score_threshold": wr.best.entry_score_threshold,
                        "stop_atr_multiple": wr.best.stop_atr_multiple,
                    },
                    "metrics": _metrics_json(wr.test_metrics),
                }
                for wr in wf.windows
            ],
            "stitched_metrics": _metrics_json(wf.stitched_metrics),
            "gate_passed": wf.stitched_metrics.gate_passed,
            "stress_segments_covered": list(wf.stress_segments_covered),
            "experiment_count": count,
        }
        print(json.dumps(payload))
    else:
        _render_walk_forward(wf, config.name, count)
    return 0


def _metrics_json(metrics: BacktestMetrics) -> dict:
    import math
    from dataclasses import asdict

    return {
        k: (None if isinstance(v, float) and math.isnan(v) else v)
        for k, v in asdict(metrics).items()
    }


def _backtest_json(
    result: BacktestResult, metrics: BacktestMetrics, count: int, start, end, kind: str
) -> dict:
    return {
        "venue": result.venue,
        "kind": kind,
        "from": start.isoformat(),
        "to": end.isoformat(),
        "metrics": _metrics_json(metrics),
        "gate_passed": metrics.gate_passed,
        "trades": len(result.trades),
        "open_positions": list(result.open_positions),
        "sessions_run": result.sessions_run,
        "sessions_skipped": len(result.sessions_skipped),
        "survivorship_ratio": round(result.survivorship_ratio, 4),
        "eligible_members": {"min": result.eligible_min, "mean": round(result.eligible_mean, 2)},
        "warnings": list(result.warnings),
        "experiment_count": count,
    }


def _fmt_pct(value: float) -> str:
    import math

    return "-" if math.isnan(value) else f"{value:+.2%}"


def _fmt_num(value: float) -> str:
    import math

    return "-" if math.isnan(value) else f"{value:.2f}"


def _render_backtest(
    result: BacktestResult, metrics: BacktestMetrics, count: int, kind: str
) -> None:
    from rich.console import Console
    from rich.table import Table

    console = Console()
    table = Table(title=f"{result.venue} {kind} {result.start} .. {result.end}")
    table.add_column("metric")
    table.add_column("value", justify="right")
    for name, value in [
        ("total return", _fmt_pct(metrics.total_return)),
        ("annualized return", _fmt_pct(metrics.annualized_return)),
        ("max drawdown", _fmt_pct(metrics.max_drawdown)),
        ("sharpe (daily, 0% cash)", _fmt_num(metrics.sharpe)),
        ("win rate", _fmt_pct(metrics.win_rate)),
        ("avg win / avg loss", f"${metrics.avg_win:,.2f} / ${metrics.avg_loss:,.2f}"),
        ("trades", str(metrics.trade_count)),
        ("turnover (annualized)", _fmt_num(metrics.turnover) + "x"),
        ("fee drag", f"${metrics.fees_paid:,.2f} ({metrics.fee_drag:.2%} of start)"),
        ("gross profit", f"${metrics.gross_profit:,.2f}"),
        ("fee drag vs gross profit (go-live: <30%)", _fmt_pct(metrics.fee_drag_vs_gross)),
        ("benchmark total return", _fmt_pct(metrics.benchmark_total_return)),
        ("benchmark sharpe", _fmt_num(metrics.benchmark_sharpe)),
        ("GATE (sharpe > benchmark AND total > 0)", "PASS" if metrics.gate_passed else "FAIL"),
    ]:
        table.add_row(name, value)
    console.print(table)
    console.print(
        f"survivorship coverage: {result.survivorship_ratio:.1%} of point-in-time "
        f"members had data; coverage gated on {result.eligible_min}-"
        f"{result.eligible_mean:.1f} (min-mean) listed members per session; "
        f"{len(result.sessions_skipped)} session(s) skipped"
    )
    for warning in result.warnings:
        console.print(f"[yellow]{warning}[/yellow]")
    console.print(f"experiments journaled for {result.venue}: {count} (this run included)")


def _render_walk_forward(wf: WalkForwardResult, venue: str, count: int) -> None:
    from rich.console import Console
    from rich.table import Table

    console = Console()
    table = Table(title=f"{venue} walk-forward — stitched OOS segments only")
    for col in ["test window", "threshold", "stop x", "sharpe", "total", "max DD", "trades"]:
        table.add_column(col, justify="right")
    for wr in wf.windows:
        m = wr.test_metrics
        table.add_row(
            f"{wr.window.test_start} .. {wr.window.test_end - datetime.timedelta(days=1)}",
            f"{wr.best.entry_score_threshold:.2f}",
            f"{wr.best.stop_atr_multiple:.1f}",
            _fmt_num(m.sharpe),
            _fmt_pct(m.total_return),
            _fmt_pct(m.max_drawdown),
            str(m.trade_count),
        )
    console.print(table)
    s = wf.stitched_metrics
    console.print(
        f"stitched OOS: sharpe {_fmt_num(s.sharpe)} vs benchmark {_fmt_num(s.benchmark_sharpe)}, "
        f"total {_fmt_pct(s.total_return)}, fee drag ${s.fees_paid:,.2f} — "
        f"GATE {'PASS' if s.gate_passed else 'FAIL'}"
    )
    console.print(f"stress segments covered: {', '.join(wf.stress_segments_covered)}")
    console.print(f"experiments journaled for {venue}: {count} (all windows + summary included)")
