"""One live-paper cycle around the pure simulator (spec: Execution Split).

Owns every side effect the simulator is not allowed to have: the lockfile,
state file (atomic write, corrupt -> refuse + notify), append-only journal,
staleness decision (the only clock-dependent rule), earnings fetch, and
failure notifications. run_venue never trades a decision bar twice: the
journal is consulted by run_key before acting.
"""

from __future__ import annotations

import datetime
import json
import os
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from trading.config import VenueConfig
from trading.data.cache import OhlcvCache
from trading.earnings import fetch_earnings_dates
from trading.journal import Journal, config_hash
from trading.pipeline import PipelineDataError, RankingsResult, build_rankings
from trading.simulator.core import StepResult, decision_bar, make_run_key, step
from trading.simulator.state import (
    PortfolioState,
    StateError,
    initial_state,
    state_from_dict,
    to_state_dict,
)
from trading.venues.base import VenueAdapter

Notifier = Callable[[str, str], None]

EARNINGS_CANDIDATE_DEPTH = 15  # top-of-ranking symbols worth an earnings lookup


class RunnerError(RuntimeError):
    pass


@dataclass(frozen=True)
class RunOutcome:
    venue: str
    status: str  # "ok" | "noop" | "skipped" | "failed"
    message: str
    run_key: str | None = None
    result: StepResult | None = None


def state_path(state_root: Path, venue: str) -> Path:
    return state_root / venue / "portfolio.json"


def lock_path(state_root: Path, venue: str) -> Path:
    return state_root / venue / ".lock"


def load_state(path: Path) -> PortfolioState | None:
    """None = not bootstrapped yet. Corruption raises StateError — the caller
    must refuse to run and notify; state is NEVER silently regenerated."""
    if not path.exists():
        return None
    try:
        return state_from_dict(json.loads(path.read_text()))
    except (json.JSONDecodeError, StateError) as exc:
        raise StateError(f"{path}: {exc}") from exc


def save_state(path: Path, state: PortfolioState) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(to_state_dict(state), indent=2, sort_keys=True))
    os.replace(tmp, path)  # atomic: never leave a torn state file


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class RunLock:
    """state/<venue>/.lock — prevents a manual run racing the scheduled job."""

    def __init__(self, path: Path):
        self._path = path

    def acquire(self) -> bool:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        for _ in range(2):
            try:
                fd = os.open(self._path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                try:
                    pid = int(self._path.read_text().strip())
                except (OSError, ValueError):
                    pid = None
                if pid is not None and _pid_alive(pid):
                    return False
                self._path.unlink(missing_ok=True)  # stale lock from a dead process
                continue
            os.write(fd, str(os.getpid()).encode())
            os.close(fd)
            return True
        return False

    def release(self) -> None:
        self._path.unlink(missing_ok=True)


def _ranking_records(table: pd.DataFrame) -> list[dict]:
    records: list[dict] = []
    for pos, (symbol, row) in enumerate(table.iterrows(), start=1):
        record: dict[str, object] = {"rank": pos, "symbol": symbol, "status": row["status"]}
        for col in table.columns:
            if col == "status":
                continue
            value = row[col]
            record[col] = None if pd.isna(value) else round(float(value), 4)
        records.append(record)
    return records


def _run_event(
    result: StepResult,
    rankings: RankingsResult,
    config: VenueConfig,
    now: datetime.datetime,
    extra_warnings: list[str],
) -> dict:
    benchmark_close = float(rankings.benchmark_bars.loc[: result.decision_ts, "close"].iloc[-1])
    return {
        "event": "run",
        "venue": config.name,
        "run_key": result.run_key,
        "ts": now.isoformat(),
        "decision_ts": result.decision_ts.isoformat(),
        "config_hash": config_hash(config),
        "regime": {
            "state": rankings.regime.state,
            "exposure_multiplier": rankings.regime.exposure_multiplier,
        },
        "coverage": {
            "requested": rankings.coverage.requested,
            "fetched": rankings.coverage.fetched,
            "ratio": round(rankings.coverage.ratio, 4),
        },
        "benchmark": {
            "symbol": config.benchmark,
            "close": benchmark_close,
            "start_price": result.state.benchmark_start_price,
        },
        "starting_balance": config.portfolio.starting_balance,
        "ranking": _ranking_records(rankings.table),
        "fills": [asdict(f) for f in result.fills],
        "new_orders": [asdict(o) for o in result.new_orders],
        "skips": [asdict(s) for s in result.skips],
        "warnings": [
            *result.warnings,
            *extra_warnings,
            *(f"quarantined: {s}" for s in rankings.quarantined),
            *(f"fetch failed: {s}" for s in rankings.fetch_failures),
        ],
        "snapshot": asdict(result.snapshot),
        "state_after": to_state_dict(result.state),
    }


def run_venue(
    config: VenueConfig,
    adapter: VenueAdapter,
    cache: OhlcvCache,
    *,
    now: datetime.datetime,
    state_root: Path,
    journal_root: Path,
    notify: Notifier,
) -> RunOutcome:
    venue = config.name
    journal = Journal(journal_root / f"{venue}.jsonl")
    lock = RunLock(lock_path(state_root, venue))
    if not lock.acquire():
        notify("trading: run skipped", f"{venue}: another run holds the lock")
        return RunOutcome(venue, "skipped", "lockfile held by a live process")
    try:
        try:
            state = load_state(state_path(state_root, venue))
        except StateError as exc:
            notify("trading: state corrupt", f"{venue}: corrupt state file, refusing to run")
            return RunOutcome(
                venue,
                "failed",
                f"corrupt state file ({exc}); recover with "
                f"'trading run --venue {venue} --restore-from-journal'",
            )

        try:
            rankings = build_rankings(config, adapter, cache, now.date())
        except PipelineDataError as exc:
            journal.append(
                {"event": "run_failed", "venue": venue, "ts": now.isoformat(), "reason": str(exc)}
            )
            notify("trading: run failed", f"{venue}: {exc}")
            return RunOutcome(venue, "failed", str(exc))

        decision_ts = decision_bar(rankings)
        run_key = make_run_key(venue, decision_ts)
        if journal.has_run(run_key):
            return RunOutcome(
                venue, "noop", f"decision bar {decision_ts.date()} already processed", run_key
            )

        if state is None:  # explicit bootstrap path, journaled (spec)
            benchmark_close = float(rankings.benchmark_bars["close"].iloc[-1])
            state = initial_state(
                venue, config.portfolio.starting_balance, benchmark_close, now.isoformat()
            )
            journal.append(
                {
                    "event": "bootstrap",
                    "venue": venue,
                    "ts": now.isoformat(),
                    "starting_balance": config.portfolio.starting_balance,
                    "benchmark_start_price": benchmark_close,
                    "config_hash": config_hash(config),
                }
            )

        # Staleness (spec: Execution Split #4): a late run still processes
        # exits and fills; entries are skipped beyond the config bound past
        # the decision bar's day boundary. No catch-up trading, ever.
        deadline = (
            decision_ts
            + pd.Timedelta(1, unit="D")
            + pd.Timedelta(config.portfolio.staleness_hours, unit="h")
        )
        allow_entries = pd.Timestamp(now) <= deadline

        earnings = None
        extra_warnings: list[str] = []
        if config.portfolio.earnings_blackout_enabled:
            candidates = list(rankings.table.index[:EARNINGS_CANDIDATE_DEPTH])
            earnings, degraded = fetch_earnings_dates(candidates)
            if degraded:
                extra_warnings.append(
                    "earnings fetch degraded: blackout filter partially disabled this run"
                )

        result = step(
            state,
            rankings,
            config,
            allow_entries=allow_entries,
            stale_reason=None if allow_entries else "stale_run_entries_skipped",
            earnings=earnings,
        )

        save_state(state_path(state_root, venue), result.state)
        journal.append(_run_event(result, rankings, config, now, extra_warnings))

        if result.breaker_tripped_now:
            notify(
                "trading: circuit breaker",
                f"{venue}: drawdown halt — entries stopped until reset-breaker",
            )

        message = f"{len(result.fills)} fill(s), {len(result.new_orders)} new order(s)"
        if not allow_entries:
            message += "; stale run: entries skipped"
        return RunOutcome(venue, "ok", message, run_key, result)
    finally:
        lock.release()


def restore_from_journal(venue: str, state_root: Path, journal_root: Path) -> str:
    """Recovery for a corrupt state file: replay the last journaled snapshot.
    The CLI gates this behind typed operator confirmation."""
    journal = Journal(journal_root / f"{venue}.jsonl")
    last = journal.last_event(types=frozenset({"run"}))
    if last is None or "state_after" not in last:
        raise RunnerError(f"no run event with state_after in {venue} journal")
    state = state_from_dict(last["state_after"])
    save_state(state_path(state_root, venue), state)
    return f"restored {venue} state from journaled run {last['run_key']}"
