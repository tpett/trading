import datetime
import math
from pathlib import Path

from trading.backtest.experiments import (
    experiment_count,
    experiments_journal,
    log_experiment,
    prior_holdout,
)
from trading.backtest.metrics import BacktestMetrics
from trading.config import load_venue_config
from trading.journal import config_hash

CONFIG = load_venue_config("crypto", Path("config"))


def _metrics(sharpe: float = 1.0) -> BacktestMetrics:
    return BacktestMetrics(
        total_return=0.1,
        annualized_return=0.2,
        max_drawdown=0.05,
        sharpe=sharpe,
        win_rate=0.5,
        avg_win=10.0,
        avg_loss=-5.0,
        trade_count=4,
        turnover=2.0,
        fees_paid=3.0,
        fee_drag=0.003,
        gross_profit=10.0,
        fee_drag_vs_gross=0.3,
        benchmark_total_return=0.05,
        benchmark_sharpe=0.8,
        gate_passed=True,
    )


def test_log_and_count_experiments(tmp_path):
    journal = experiments_journal(tmp_path, "crypto")
    assert experiment_count(journal, "crypto") == 0
    for kind in ("backtest", "walk_forward_window", "walk_forward"):
        log_experiment(
            journal,
            config=CONFIG,
            kind=kind,
            start=datetime.date(2020, 1, 1),
            end=datetime.date(2020, 6, 30),
            metrics=_metrics(),
            ts="2026-07-05T00:00:00+00:00",
            grid_point={"entry_score_threshold": 0.7, "stop_atr_multiple": 1.5},
            survivorship_ratio=0.95,
        )
    assert experiment_count(journal, "crypto") == 3
    assert experiment_count(journal, "equities") == 0
    events = list(journal.events())
    assert events[0]["config_hash"] == config_hash(CONFIG)
    assert events[0]["grid_point"]["entry_score_threshold"] == 0.7
    assert events[0]["metrics"]["sharpe"] == 1.0
    assert events[0]["from"] == "2020-01-01" and events[0]["to"] == "2020-06-30"
    assert (tmp_path / "experiments-crypto.jsonl").exists()


def test_nan_metrics_are_json_null(tmp_path):
    journal = experiments_journal(tmp_path, "crypto")
    log_experiment(
        journal,
        config=CONFIG,
        kind="backtest",
        start=datetime.date(2020, 1, 1),
        end=datetime.date(2020, 6, 30),
        metrics=_metrics(sharpe=math.nan),
        ts="2026-07-05T00:00:00+00:00",
    )
    event = next(journal.events())
    assert event["metrics"]["sharpe"] is None  # NaN is not valid JSON


def test_prior_holdout_found_only_for_holdout_kind(tmp_path):
    journal = experiments_journal(tmp_path, "equities")
    log_experiment(
        journal,
        config=CONFIG,
        kind="backtest",
        start=datetime.date(2020, 1, 1),
        end=datetime.date(2020, 6, 30),
        metrics=_metrics(),
        ts="2026-07-05T00:00:00+00:00",
    )
    assert prior_holdout(journal) is None
    log_experiment(
        journal,
        config=CONFIG,
        kind="holdout",
        start=datetime.date(2026, 1, 5),
        end=datetime.date(2026, 7, 4),
        metrics=_metrics(),
        ts="2026-07-05T01:00:00+00:00",
    )
    prior = prior_holdout(journal)
    assert prior is not None and prior["ts"] == "2026-07-05T01:00:00+00:00"
