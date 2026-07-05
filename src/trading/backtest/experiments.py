"""Experiments journal (spec: Backtesting & Validation / Reporting).

Every backtest run -- plain, walk-forward window, walk-forward summary,
holdout -- appends config hash + grid point + metrics to a per-venue
append-only JSONL (reusing trading.journal.Journal). The experiment count is
reported alongside any quoted result: 50 experiments deep, "OOS beats SPY"
is selection, not signal. prior_holdout() backs the once-only holdout gate.
"""

from __future__ import annotations

import datetime
import math
from dataclasses import asdict
from pathlib import Path

from trading.backtest.metrics import BacktestMetrics
from trading.config import VenueConfig
from trading.journal import Journal, config_hash


def experiments_journal(journal_root: Path, venue: str) -> Journal:
    return Journal(journal_root / f"experiments-{venue}.jsonl")


def _json_safe(value: object) -> object:
    return None if isinstance(value, float) and math.isnan(value) else value


def log_experiment(
    journal: Journal,
    *,
    config: VenueConfig,
    kind: str,  # backtest | walk_forward_window | walk_forward | holdout
    start: datetime.date,
    end: datetime.date,
    metrics: BacktestMetrics,
    ts: str,  # ISO-8601 UTC, supplied by the CLI (the only clock reader)
    grid_point: dict | None = None,
    survivorship_ratio: float | None = None,
    extra: dict | None = None,
) -> None:
    journal.append(
        {
            "event": "experiment",
            "kind": kind,
            "venue": config.name,
            "ts": ts,
            "config_hash": config_hash(config),
            "grid_point": grid_point,
            "from": start.isoformat(),
            "to": end.isoformat(),
            "survivorship_ratio": survivorship_ratio,
            "metrics": {k: _json_safe(v) for k, v in asdict(metrics).items()},
            **(extra or {}),
        }
    )


def experiment_count(journal: Journal, venue: str) -> int:
    return sum(
        1
        for event in journal.events()
        if event.get("event") == "experiment" and event.get("venue") == venue
    )


def prior_holdout(journal: Journal) -> dict | None:
    last: dict | None = None
    for event in journal.events():
        if event.get("event") == "experiment" and event.get("kind") == "holdout":
            last = event
    return last
