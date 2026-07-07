"""Journal Robinhood's earnings calendar daily -- point-in-time by construction.

Why: the spec dropped the earnings entry blackout in both modes because
yfinance's dates proved stale. Robinhood's calendar is verified-quality
(report date, am/pm timing, verified flag, EPS estimate/actual), but it is
a LIVE feed with no history API -- so we build our own history by appending
what the calendar said each day to an append-only journal. Once enough
history accumulates, the entry blackout can be reinstated for live trading
and validated against this journal (never against a retroactive source).

Two windows per run: forward 14 days (upcoming reports, the blackout input)
and trailing 7 days (captures actual EPS after reports land). Each window
appends one event keyed `earnings:<date>:<+days>`; the Journal's has_run()
makes same-day re-runs no-ops, so a launchd catch-up after sleep cannot
duplicate history.

One-time setup on the machine that runs this (see rhmcp.auth_flow):
    uv run python scripts/dump_earnings_calendar.py --auth
Daily run (installed as a LaunchAgent by `trading schedule`):
    uv run python scripts/dump_earnings_calendar.py
"""

from __future__ import annotations

import argparse
import datetime
import sys
from pathlib import Path

from trading.journal import Journal
from trading.rhmcp import DEFAULT_TOKEN_PATH, McpClient, auth_flow

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_JOURNAL = ROOT / "journal" / "earnings-calendar.jsonl"
WINDOWS_DAYS = (14, -7)


def run_key(day: datetime.date, days: int) -> str:
    return f"earnings:{day.isoformat()}:{days:+d}"


def dump(client: McpClient, journal: Journal, today: datetime.date) -> list[str]:
    """Fetch each window and append its event; returns human-readable lines.
    A window that fails raises AFTER earlier windows have been journaled --
    partial progress is kept (append-only, idempotent), the failure surfaces
    via exit code in the launchd log."""
    messages = []
    for days in WINDOWS_DAYS:
        key = run_key(today, days)
        if journal.has_run(key):
            messages.append(f"{key}: already journaled, skipping")
            continue
        payload = client.call_tool(
            "get_earnings_calendar",
            {"start_date": today.isoformat(), "days": days},
        )
        results = payload["data"]["results"]
        journal.append(
            {
                "event": "earnings_calendar",
                "run_key": key,
                "fetched_at": datetime.datetime.now(datetime.UTC).isoformat(),
                "start_date": today.isoformat(),
                "days": days,
                "results": results,
            }
        )
        messages.append(f"{key}: journaled {len(results)} report(s)")
    return messages


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--auth", action="store_true", help="run one-time OAuth consent and exit")
    parser.add_argument("--journal", type=Path, default=DEFAULT_JOURNAL)
    parser.add_argument("--token-path", type=Path, default=DEFAULT_TOKEN_PATH)
    args = parser.parse_args()

    if args.auth:
        auth_flow(args.token_path)
        return 0

    client = McpClient(args.token_path)
    today = datetime.date.today()
    try:
        messages = dump(client, Journal(args.journal), today)
    except Exception as exc:  # noqa: BLE001 - launchd log is the surface
        print(f"earnings dump failed: {exc}", file=sys.stderr)
        return 1
    for message in messages:
        print(message)
    return 0


if __name__ == "__main__":
    sys.exit(main())
