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
from zoneinfo import ZoneInfo

import pandas as pd

from trading.config import VENUES, VenueConfig
from trading.data.cache import OhlcvCache
from trading.digest import collect_run_events, write_digest
from trading.earnings import fetch_earnings_dates
from trading.journal import Journal, JournalError, config_hash
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

_EQUITIES_TZ = ZoneInfo("America/New_York")
_EQUITIES_SESSION_CLOSE_HOUR = 16  # NYSE close, local time


class RunnerError(RuntimeError):
    pass


def intraday_partial_bar_reason(
    config: VenueConfig, decision_ts: pd.Timestamp, now: datetime.datetime
) -> str | None:
    """Session venues (costs.trades_24_7 = false) can have their daily bar
    served IN PROGRESS by the data provider while the market is still open --
    a launchd-coalesced run landing mid-session must never trade on it. If
    the decision bar's date is today (exchange-local) and local time hasn't
    reached session close + config.portfolio.session_close_buffer_minutes,
    the bar isn't final yet: refuse before any journal write. 24/7 venues
    (crypto) are unaffected -- there is no session to be mid-way through.
    """
    if config.costs.trades_24_7:
        return None
    now_utc = now if now.tzinfo is not None else now.replace(tzinfo=datetime.UTC)
    local_now = now_utc.astimezone(_EQUITIES_TZ)
    if decision_ts.date() != local_now.date():
        return None
    close = local_now.replace(hour=_EQUITIES_SESSION_CLOSE_HOUR, minute=0, second=0, microsecond=0)
    deadline = close + datetime.timedelta(minutes=config.portfolio.session_close_buffer_minutes)
    if local_now < deadline:
        return "run during market session; decision bar incomplete — rerun after close"
    return None


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

    def _sweep_orphaned_reclaims(self) -> None:
        """Remove <lock>.reclaim.<pid> files whose reclaimer died between
        the rename and the unlink; a live pid's reclaim is left alone."""
        for orphan in self._path.parent.glob(f"{self._path.name}.reclaim.*"):
            try:
                pid = int(orphan.name.rsplit(".", 1)[-1])
            except ValueError:
                continue
            if not _pid_alive(pid):
                orphan.unlink(missing_ok=True)

    def acquire(self) -> bool:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._sweep_orphaned_reclaims()
        for _ in range(3):
            try:
                fd = os.open(self._path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                try:
                    pid = int(self._path.read_text().strip())
                except (OSError, ValueError):
                    pid = None
                if pid is not None and _pid_alive(pid):
                    return False
                # Claim the stale lock ATOMICALLY: exactly one reclaimer wins
                # the rename; a plain unlink would let two racing processes
                # both remove it and both "acquire". Losers get
                # FileNotFoundError and retry the O_EXCL create.
                reclaim = self._path.with_name(f"{self._path.name}.reclaim.{os.getpid()}")
                try:
                    os.rename(self._path, reclaim)
                except FileNotFoundError:
                    continue  # another process won the reclaim; retry
                reclaim.unlink(missing_ok=True)
                continue
            os.write(fd, str(os.getpid()).encode())
            os.close(fd)
            return True
        return False  # bounded retries: treat persistent contention as lock-held

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
    digest_root: Path | None = None,
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

        # Fail-safe: if the journal's tail is ahead of the persisted state
        # (a state write failed and was never reconciled), proceeding would
        # silently and permanently drop the missing run's fills and orders —
        # has_run guarantees that run_key is never reprocessed. Refuse.
        last_run = journal.last_event(types=frozenset({"run"}))
        if last_run is not None and (state is None or state.last_run_key != last_run["run_key"]):
            message = (
                f"state file behind journal; run "
                f"'trading run --venue {venue} --restore-from-journal'"
            )
            notify("trading: state behind journal", f"{venue}: {message}")
            return RunOutcome(venue, "failed", message)

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

        # Guard BEFORE any journal write: a coalesced run landing mid-session
        # must never let an in-progress daily bar become the decision bar.
        # No journal event, no state mutation -- run_key stays virgin so the
        # legitimate after-close run processes the (by-then-final) bar.
        guard_reason = intraday_partial_bar_reason(config, decision_ts, now)
        if guard_reason is not None:
            notify("trading: run aborted", f"{venue}: {guard_reason}")
            return RunOutcome(venue, "failed", guard_reason, run_key)

        if state is None:  # explicit bootstrap path, journaled (spec)
            # Reuse an already-journaled bootstrap (crash between the bootstrap
            # append and the first state write) so the event is never doubled
            # and the benchmark baseline/birthdate stay those of the original.
            boot = journal.last_event(types=frozenset({"bootstrap"}))
            if boot is None:
                benchmark_close = float(rankings.benchmark_bars["close"].iloc[-1])
                created_at = now.isoformat()
                journal.append(
                    {
                        "event": "bootstrap",
                        "venue": venue,
                        "ts": created_at,
                        "starting_balance": config.portfolio.starting_balance,
                        "benchmark_start_price": benchmark_close,
                        "config_hash": config_hash(config),
                    }
                )
            else:
                benchmark_close = float(boot["benchmark_start_price"])
                created_at = boot["ts"]
            state = initial_state(
                venue, config.portfolio.starting_balance, benchmark_close, created_at
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

        # Persist journal-FIRST (crash-safe ordering): a crash between the two
        # writes leaves state one run BEHIND the journal — the anticipated,
        # recoverable condition (--restore-from-journal replays the last run
        # event's state_after). The reverse order would leave mutated state
        # with no run_key in the journal, and a restart would re-execute
        # step() on refetched (possibly different) data: double-processing.
        try:
            journal.append(_run_event(result, rankings, config, now, extra_warnings))
        except Exception as exc:
            notify("trading: journal write failed", f"{venue}: {exc}")
            return RunOutcome(
                venue, "failed", f"journal append failed ({exc}); state untouched", run_key
            )
        try:
            save_state(state_path(state_root, venue), result.state)
        except Exception as exc:
            notify("trading: state write failed", f"{venue}: {exc}")
            return RunOutcome(
                venue,
                "failed",
                f"state write failed after journal append ({exc}); state is one run "
                f"behind — recover with 'trading run --venue {venue} --restore-from-journal'",
                run_key,
            )

        # Pure reporting, only after state is durable: a digest failure must
        # never block the state save or fail the run — notify and carry on.
        if digest_root is not None:
            date_iso = now.date().isoformat()
            try:
                write_digest(
                    digest_root,
                    date_iso,
                    collect_run_events(journal_root, VENUES, date_iso),
                )
            except Exception as exc:
                notify("trading: digest write failed", f"{venue}: {exc}")

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
    The CLI gates this behind typed operator confirmation.

    Takes the venue RunLock for the full mutation: a scheduled run landing
    mid-restore could otherwise race the state write it produces, or trip
    the journal's torn-tail repair against a partial flush from the other
    process -- either way, silent corruption. Refuse loudly instead.
    """
    lock = RunLock(lock_path(state_root, venue))
    if not lock.acquire():
        raise RunnerError(f"another run is in progress for {venue}; try again after it completes")
    try:
        journal = Journal(journal_root / f"{venue}.jsonl")
        last: dict | None = None
        reset_after: dict | None = None  # breaker_reset journaled AFTER the last run
        try:
            for event in journal.events():
                kind = event.get("event")
                if kind == "run":
                    last, reset_after = event, None
                elif kind == "breaker_reset":
                    reset_after = event
        except JournalError as exc:
            raise RunnerError(f"journal corrupt: {exc}") from exc
        if last is None or "state_after" not in last:
            raise RunnerError(f"no run event with state_after in {venue} journal")
        state = state_from_dict(last["state_after"])
        if reset_after is not None:
            # A manual reset journaled after the last run must survive restore;
            # replaying only the run snapshot would silently re-trip a breaker
            # the operator explicitly reset (journal-first ordering means the
            # event can exist before the reset's state write landed).
            state.breaker_tripped = False
            state.breaker_tripped_at = None
            state.high_water_mark = float(reset_after["high_water_mark"])
        save_state(state_path(state_root, venue), state)
        return f"restored {venue} state from journaled run {last['run_key']}"
    finally:
        lock.release()
