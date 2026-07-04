"""Per-venue append-only JSONL journal (spec: Reporting & Operations).

Every run appends one event; state is reconstructible by replaying the
events' state_after snapshots. The journal is also the idempotency ledger:
has_run(run_key) is consulted before acting, so a decision bar that has been
traded is never traded again.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterator
from dataclasses import asdict
from pathlib import Path

from trading.config import VenueConfig


class JournalError(RuntimeError):
    """The journal is corrupt somewhere other than a torn final line."""


class Journal:
    def __init__(self, path: Path):
        self._path = path
        path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, event: dict) -> None:
        line = json.dumps(event, sort_keys=True, default=str)
        with self._path.open("a") as f:
            f.write(line + "\n")
            f.flush()
            os.fsync(f.fileno())

    def events(self) -> Iterator[dict]:
        if not self._path.exists():
            return
        lines = self._path.read_text().splitlines()
        for number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                if number == len(lines):
                    return  # torn final line: crash mid-append, ignore
                raise JournalError(f"{self._path}: corrupt journal line {number}") from exc

    def has_run(self, run_key: str) -> bool:
        return any(event.get("run_key") == run_key for event in self.events())

    def last_event(self, types: frozenset[str] | None = None) -> dict | None:
        last: dict | None = None
        for event in self.events():
            if types is None or event.get("event") in types:
                last = event
        return last


def config_hash(config: VenueConfig) -> str:
    payload = json.dumps(asdict(config), sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:12]
