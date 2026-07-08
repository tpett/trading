"""Unit tests for the cross-sectional signal-scan core (no I/O)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from signal_scan import _cell_metrics, information_coefficient, survivors  # noqa: E402


def _panel(rows):
    return pd.DataFrame(rows, columns=["date", "symbol", "metric", "ret", "hedge"])


def test_ic_perfect_positive_is_plus_one():
    # Two months, metric rank == return rank in each -> IC 1.0.
    rows = []
    for month in ("2021-07-01", "2021-08-01"):
        d = pd.Timestamp(month, tz="UTC")
        for i in range(20):
            rows.append((d, f"S{i}", float(i), float(i) * 0.01, 0.0))
    ic, t, n = information_coefficient(_panel(rows), "metric", "ret")
    assert ic == 1.0
    assert n == 2


def test_ic_perfect_negative_is_minus_one():
    rows = []
    for month in ("2021-07-01", "2021-08-01"):
        d = pd.Timestamp(month, tz="UTC")
        for i in range(20):
            rows.append((d, f"S{i}", float(i), -float(i), 0.0))
    ic, _, _ = information_coefficient(_panel(rows), "metric", "ret")
    assert ic == -1.0


def test_ic_skips_thin_months():
    # A month with < MIN_NAMES names is ignored; only the full month counts.
    full = pd.Timestamp("2021-07-01", tz="UTC")
    thin = pd.Timestamp("2021-08-01", tz="UTC")
    rows = [(full, f"S{i}", float(i), float(i), 0.0) for i in range(20)]
    rows += [(thin, f"T{i}", float(i), float(i), 0.0) for i in range(5)]
    _, _, n = information_coefficient(_panel(rows), "metric", "ret")
    assert n == 1  # the 5-name month is dropped


def test_survivors_drops_top_hedged_third():
    d = pd.Timestamp("2021-07-01", tz="UTC")
    rows = [(d, f"S{i}", 0.0, 0.0, float(i)) for i in range(9)]  # hedge 0..8
    kept = survivors(_panel(rows), veto_frac=1 / 3)
    # top third (hedge >= 2/3 quantile = 5.33) dropped -> hedges 6,7,8 gone.
    assert set(kept["hedge"]) == {0, 1, 2, 3, 4, 5}


def test_survivors_no_veto_keeps_all():
    d = pd.Timestamp("2021-07-01", tz="UTC")
    rows = [(d, f"S{i}", 0.0, 0.0, float(i)) for i in range(9)]
    assert len(survivors(_panel(rows), veto_frac=0)) == 9


def test_cell_metrics_atm_spread_and_excite():
    cell = {
        "skew_put_atm": 0.05,
        "skew_put_call": 0.02,
        "contracts": [
            {"role": "atm", "bid": 4.0, "ask": 4.2, "mid": 4.1, "iv": 0.30, "volume": 100},
            {"role": "otm_put", "iv": 0.34, "volume": 50},
            {"role": "otm_call", "iv": 0.28, "volume": 25},
        ],
    }
    m = _cell_metrics(cell)
    assert m["atm_spread"] == (4.2 - 4.0) / 4.1
    assert m["excite"] == -0.02  # -skew_put_call
    assert m["hedge"] == 0.05


def test_cell_metrics_missing_atm_is_nan():
    cell = {"skew_put_atm": 0.05, "contracts": [{"role": "otm_put", "iv": 0.3}]}
    assert np.isnan(_cell_metrics(cell)["atm_spread"])
