"""Journal honesty: idempotent re-runs, distinct params count, holdout tracking."""

from __future__ import annotations

import math

import pytest

from trading.alphasearch.sweep import (
    DEFAULT_PARAMS,
    DISCOVERY_WINDOW,
    SweepError,
    discovery_trials,
    find_discovery_trial,
    load_trials,
    log_trial,
    prior_holdout_trial,
    trial_config,
    trial_config_hash,
    trials_journal,
)


def _journal(tmp_path):
    return trials_journal(tmp_path / "journal")


def _result(alpha_t: float = 2.0) -> dict:
    return {
        "n_dates": 60,
        "n_names_median": 97.0,
        "ls": {"alpha_annual_pct": 5.0, "alpha_t": alpha_t, "p": 0.04},
        "lo": {"alpha_annual_pct": 2.0, "alpha_t": 1.0, "p": 0.3},
        "turnover_monthly": 0.4,
        "skipped_dates": [],
    }


def test_trial_config_hash_is_deterministic_and_param_sensitive():
    base = trial_config("mom21", "largecap", DISCOVERY_WINDOW)
    again = trial_config("mom21", "largecap", DISCOVERY_WINDOW)
    assert trial_config_hash(base) == trial_config_hash(again)
    assert base["params"] == DEFAULT_PARAMS
    tercile = trial_config("mom21", "largecap", DISCOVERY_WINDOW,
                           params={**DEFAULT_PARAMS, "quantiles": 3})
    assert trial_config_hash(tercile) != trial_config_hash(base)


def test_log_trial_event_matches_spec_schema(tmp_path):
    journal = _journal(tmp_path)
    config = trial_config("mom21", "largecap", DISCOVERY_WINDOW)
    event = log_trial(journal, kind="discovery", config=config,
                      ts="2026-07-08T00:00:00+00:00", result=_result())
    stored = next(iter(journal.events()))
    assert stored == event
    assert stored["event"] == "trial"
    assert stored["kind"] == "discovery"
    assert stored["signal"] == "mom21"
    assert stored["universe"] == "largecap"
    assert stored["window"] == DISCOVERY_WINDOW
    assert stored["params"] == DEFAULT_PARAMS
    assert stored["config_hash"] == trial_config_hash(config)
    assert stored["ls"]["alpha_t"] == 2.0
    assert stored["error"] is None
    assert stored["ts"] == "2026-07-08T00:00:00+00:00"


def test_identical_rerun_appends_but_never_double_counts(tmp_path):
    journal = _journal(tmp_path)
    config = trial_config("mom21", "largecap", DISCOVERY_WINDOW)
    log_trial(journal, kind="discovery", config=config, ts="t1", result=_result(2.0))
    log_trial(journal, kind="discovery", config=config, ts="t2", result=_result(2.5))
    assert len(list(journal.events())) == 2      # append-only: nothing rewritten
    trials = discovery_trials(journal)
    assert len(trials) == 1                       # ...but it is ONE trial
    assert trials[0]["ts"] == "t2"                # latest event wins


def test_distinct_params_are_distinct_trials(tmp_path):
    journal = _journal(tmp_path)
    log_trial(journal, kind="discovery",
              config=trial_config("mom21", "largecap", DISCOVERY_WINDOW),
              ts="t1", result=_result())
    log_trial(journal, kind="discovery",
              config=trial_config("mom21", "largecap", DISCOVERY_WINDOW,
                                  params={**DEFAULT_PARAMS, "quantiles": 3}),
              ts="t2", result=_result())
    log_trial(journal, kind="discovery",
              config=trial_config("mom21", "midcap", DISCOVERY_WINDOW),
              ts="t3", result=_result())
    assert len(discovery_trials(journal)) == 3


def test_error_trial_still_counts(tmp_path):
    journal = _journal(tmp_path)
    log_trial(journal, kind="discovery",
              config=trial_config("vrp", "largecap", DISCOVERY_WINDOW),
              ts="t1", error="ValueError: need more observations (3) than parameters (5)")
    trials = discovery_trials(journal)
    assert len(trials) == 1
    assert trials[0]["error"].startswith("ValueError")
    assert "ls" not in trials[0] or trials[0].get("ls") is None


def test_nan_results_are_journaled_as_null(tmp_path):
    journal = _journal(tmp_path)
    result = _result()
    result["turnover_monthly"] = math.nan
    result["ls"]["p"] = math.nan
    log_trial(journal, kind="discovery",
              config=trial_config("mom21", "largecap", DISCOVERY_WINDOW),
              ts="t1", result=result)
    stored = next(iter(journal.events()))
    assert stored["turnover_monthly"] is None    # NaN never reaches the JSONL
    assert stored["ls"]["p"] is None


def test_inf_results_are_journaled_as_null(tmp_path):
    journal = _journal(tmp_path)
    result = _result()
    result["turnover_monthly"] = math.inf
    result["ls"]["alpha_t"] = -math.inf
    log_trial(journal, kind="discovery",
              config=trial_config("mom21", "largecap", DISCOVERY_WINDOW),
              ts="t1", result=result)
    stored = next(iter(journal.events()))
    assert stored["turnover_monthly"] is None    # +-inf never reaches the JSONL
    assert stored["ls"]["alpha_t"] is None


def test_log_trial_rejects_result_payload_that_clobbers_reserved_keys(tmp_path):
    journal = _journal(tmp_path)
    config = trial_config("mom21", "largecap", DISCOVERY_WINDOW)
    with pytest.raises(SweepError):
        log_trial(journal, kind="discovery", config=config, ts="t1",
                  result={**_result(), "ts": "hacked"})
    assert list(journal.events()) == []          # refused before any append


def test_holdout_tracking_and_discovery_lookup(tmp_path):
    journal = _journal(tmp_path)
    log_trial(journal, kind="discovery",
              config=trial_config("mom21", "largecap", DISCOVERY_WINDOW),
              ts="t1", result=_result())
    assert prior_holdout_trial(journal, "mom21", "largecap") is None
    log_trial(journal, kind="holdout",
              config=trial_config("mom21", "largecap", "2024-01-01..2026-07-07"),
              ts="t2", result=_result())
    prior = prior_holdout_trial(journal, "mom21", "largecap")
    assert prior is not None and prior["ts"] == "t2"
    assert prior_holdout_trial(journal, "mom21", "midcap") is None
    found = find_discovery_trial(journal, "mom21", "largecap")
    assert found is not None and found["kind"] == "discovery"
    assert find_discovery_trial(journal, "rev5", "largecap") is None


def test_holdout_and_discovery_never_collide_in_dedupe(tmp_path):
    journal = _journal(tmp_path)
    window = "2024-01-01..2026-07-07"
    config = trial_config("mom21", "largecap", window)
    log_trial(journal, kind="discovery", config=config, ts="t1", result=_result())
    log_trial(journal, kind="holdout", config=config, ts="t2", result=_result())
    assert len(load_trials(journal)) == 2  # same hash, different kind


def test_default_config_hash_is_pinned_to_the_live_journal():
    # journal/alphasearch-trials.jsonl carries 799 discovery trials hashed
    # through this exact params dict. This is amihud:midcap's LIVE hash (the
    # parked BH survivor, verified present in the committed journal). If
    # _hashed_params ever emits a different default dict — e.g. a new key
    # present at its default value — every existing trial silently orphans
    # from its dedupe identity. Never "fix" this by re-pinning.
    config = trial_config("amihud", "midcap", "2019-01-01..2023-12-31")
    assert trial_config_hash(config) == "4f3d0819382a"
    assert config["params"] == {
        "quantiles": 5, "weighting": "equal", "cadence": "monthly",
        "tercile_below": 50, "min_names": 15,
    }


def test_default_valued_perturbation_params_are_omitted_from_the_hash():
    from trading.alphasearch.sweep import DEFAULT_PARAMS, _hashed_params

    assert _hashed_params(5, 50, 15) == DEFAULT_PARAMS
    assert _hashed_params(5, 50, 15, symbol_subset=None, calendar_offset=0) == DEFAULT_PARAMS
    assert set(_hashed_params(5, 50, 15)) == {
        "quantiles", "weighting", "cadence", "tercile_below", "min_names",
    }


def test_subset_and_offset_change_the_hash_when_set():
    from trading.alphasearch.sweep import _hashed_params

    window = "2020-01-01..2020-06-30"
    base = trial_config_hash(trial_config("mom21", "largecap", window))
    sub = _hashed_params(5, 50, 15, symbol_subset=("B", "A"))
    assert sub["symbol_subset"] == ["A", "B"]        # sorted: draw-order-proof
    sub_hash = trial_config_hash(trial_config("mom21", "largecap", window, params=sub))
    off = _hashed_params(5, 50, 15, calendar_offset=1)
    assert off["calendar_offset"] == 1
    off_hash = trial_config_hash(trial_config("mom21", "largecap", window, params=off))
    assert len({base, sub_hash, off_hash}) == 3      # three distinct trials
