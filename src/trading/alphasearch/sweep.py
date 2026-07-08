"""Sweep runner, trial journal, leaderboard, holdout re-prove (spec 3.6 + 4).

The trial journal (journal/alphasearch-trials.jsonl, via trading.journal.
Journal) is the program's scientific ledger: EVERY evaluation -- success or
error -- is appended BEFORE any leaderboard is computed, and the BH-FDR /
DSR trial count is derived from this file alone. It is append-only and
committed to git; deleting or editing it invalidates the statistics.

Idempotency: an identical config re-run APPENDS a new event (append-only is
never violated) and every reader deduplicates via load_trials(), keeping the
LATEST event per (config_hash, kind) -- logical update-in-place, physical
append-only, and re-runs never inflate the trial count. Any changed parameter
changes the hash and honestly counts as a NEW trial (spec 5.6).

This module never reads the clock: `ts` always arrives from the CLI.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from trading.journal import Journal

DISCOVERY_WINDOW = "2019-01-01..2023-12-31"   # pre-registered (spec 5.1)
HOLDOUT_START = "2024-01-01"                  # pre-registered (spec 5.3)
BH_Q = 0.10                                   # pre-registered (spec 5.2)
HOLDOUT_PASS_RATIO = 0.5                      # pre-registered (spec 3.6)
DEFAULT_PARAMS = {"quantiles": 5, "weighting": "equal", "cadence": "monthly"}


class SweepError(RuntimeError):
    """A sweep/holdout invariant was violated; refuse loudly."""


def trials_journal(journal_dir: Path) -> Journal:
    return Journal(journal_dir / "alphasearch-trials.jsonl")


def trial_config(
    signal: str, universe: str, window: str, params: dict | None = None
) -> dict:
    return {
        "signal": signal,
        "universe": universe,
        "window": window,
        "params": dict(params or DEFAULT_PARAMS),
    }


def trial_config_hash(config: dict) -> str:
    payload = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def _json_safe(value: object) -> object:
    """NaN -> None, recursively; numpy scalars -> Python scalars. The journal
    must stay strict JSON (json.dumps would happily emit invalid bare NaN)."""
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(v) for v in value]
    if hasattr(value, "item"):  # numpy scalar
        value = value.item()
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def log_trial(
    journal: Journal,
    *,
    kind: str,  # "discovery" | "holdout"
    config: dict,
    ts: str,  # ISO-8601 UTC, supplied by the CLI (the only clock reader)
    result: dict | None = None,
    error: str | None = None,
) -> dict:
    """Append one trial event (spec section 4 schema) and return it."""
    event = {
        "event": "trial",
        "kind": kind,
        **config,
        "config_hash": trial_config_hash(config),
        "ts": ts,
        "error": error,
        **(result or {}),
    }
    event = _json_safe(event)
    journal.append(event)
    return event


def load_trials(journal: Journal) -> list[dict]:
    """All trial events, deduplicated: latest per (config_hash, kind) wins."""
    latest: dict[tuple[str, str], dict] = {}
    for event in journal.events():
        if event.get("event") != "trial":
            continue
        latest[(event["config_hash"], event["kind"])] = event
    return list(latest.values())


def discovery_trials(journal: Journal) -> list[dict]:
    """The honest trial count for BH/DSR = len() of this list."""
    return [e for e in load_trials(journal) if e.get("kind") == "discovery"]


def prior_holdout_trial(journal: Journal, signal: str, universe: str) -> dict | None:
    """Any prior holdout event for (signal, universe) -- ANY window/params:
    the holdout is touched once per candidate, not once per configuration."""
    last: dict | None = None
    for event in journal.events():
        if (
            event.get("event") == "trial"
            and event.get("kind") == "holdout"
            and event.get("signal") == signal
            and event.get("universe") == universe
        ):
            last = event
    return last


def find_discovery_trial(
    journal: Journal, signal: str, universe: str, window: str = DISCOVERY_WINDOW
) -> dict | None:
    """The default-params discovery trial for (signal, universe), by exact
    config hash -- the reference a holdout is compared against."""
    wanted = trial_config_hash(trial_config(signal, universe, window))
    for event in load_trials(journal):
        if event.get("kind") == "discovery" and event.get("config_hash") == wanted:
            return event
    return None
