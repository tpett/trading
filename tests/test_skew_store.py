"""Offline tests for the IV-skew store loader and the point-in-time panel.

Every input is fabricated in-test (a synthetic samples.jsonl); nothing touches
the network or the real gathered data directory. The no-lookahead behaviour of
IVSkewPanel.gather is the load-bearing property here and is tested directly.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from trading.signals.skew import (
    SKEW_COLUMNS,
    IVSkewPanel,
    load_skew_store,
)


def _cell(symbol: str, decision_date: str, skew_put_atm, skew_put_call=None) -> dict:
    """A samples.jsonl cell carrying pre-computed skew (the data/options-iv
    format). None -> the key is written as JSON null (a missing leg)."""
    return {
        "symbol": symbol,
        "decision_date": decision_date,
        "spot_at_decision": 100.0,
        "target_expiration": "2019-03-15",
        "days_to_expiry": 42,
        "contracts": [],
        "skew_put_atm": skew_put_atm,
        "skew_put_call": skew_put_call,
    }


def _write(path: Path, cells: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(c) for c in cells) + "\n")


def test_load_groups_by_symbol_indexed_by_decision_date_utc(tmp_path):
    path = tmp_path / "samples.jsonl"
    _write(
        path,
        [
            _cell("AAA", "2019-01-02", 0.05, 0.03),
            _cell("AAA", "2019-02-01", 0.07, 0.04),
            _cell("BBB", "2019-01-02", 0.02),
        ],
    )
    store = load_skew_store(path)
    assert sorted(store) == ["AAA", "BBB"]
    aaa = store["AAA"]
    assert list(aaa.columns) == SKEW_COLUMNS
    assert str(aaa.index.tz) == "UTC"
    assert aaa.index.is_monotonic_increasing
    assert aaa.loc[pd.Timestamp("2019-02-01", tz="UTC"), "skew_put_atm"] == 0.07


def test_null_skew_leg_becomes_nan(tmp_path):
    path = tmp_path / "samples.jsonl"
    _write(path, [_cell("AAA", "2019-01-02", None, None)])
    store = load_skew_store(path)
    row = store["AAA"].iloc[-1]
    assert pd.isna(row["skew_put_atm"])
    assert pd.isna(row["skew_put_call"])
    # skew_put_call is often null even when skew_put_atm is present.
    _write(path, [_cell("AAA", "2019-01-02", 0.06, None)])
    store = load_skew_store(path)
    row = store["AAA"].iloc[-1]
    assert row["skew_put_atm"] == 0.06
    assert pd.isna(row["skew_put_call"])


def test_tolerant_of_torn_and_blank_lines(tmp_path):
    path = tmp_path / "samples.jsonl"
    good1 = json.dumps(_cell("AAA", "2019-01-02", 0.05))
    good2 = json.dumps(_cell("AAA", "2019-02-01", 0.06))
    # A blank line, then a torn final line (killed gather) that must not sink the load.
    path.write_text(good1 + "\n\n" + good2 + '\n{"symbol": "AAA", "decis')
    store = load_skew_store(path)
    assert len(store["AAA"]) == 2
    assert store["AAA"]["skew_put_atm"].tolist() == [0.05, 0.06]


def test_duplicate_decision_keeps_last(tmp_path):
    path = tmp_path / "samples.jsonl"
    _write(
        path,
        [
            _cell("AAA", "2019-01-02", 0.05),
            _cell("AAA", "2019-01-02", 0.09),  # re-gathered: supersedes
        ],
    )
    store = load_skew_store(path)
    assert len(store["AAA"]) == 1
    assert store["AAA"]["skew_put_atm"].iloc[-1] == 0.09


def test_missing_file_yields_empty_store(tmp_path):
    assert load_skew_store(tmp_path / "nope.jsonl") == {}


def test_recompute_fallback_for_poc_cells_without_stored_skew(tmp_path):
    """A cell that stores contracts but NO skew_put_atm/skew_put_call keys (the
    older data/options-poc format) is recomputed from its raw legs, so one
    loader serves both gather formats."""
    from trading.research.options_iv import DIV_YIELD, RATE, bs_price

    dte, spot = 42.0, 100.0
    t = dte / 365.0
    atm = bs_price(spot, 100.0, t, RATE, DIV_YIELD, 0.20, True)
    otm_put = bs_price(spot, 90.0, t, RATE, DIV_YIELD, 0.28, False)  # richer downside vol
    poc_cell = {
        "symbol": "AAA",
        "decision_date": "2019-01-02",
        "spot_at_decision": spot,
        "days_to_expiry": dte,
        "contracts": [
            {"role": "atm", "type": "call", "strike": 100.0, "close": atm},
            {"role": "otm_put", "type": "put", "strike": 90.0, "close": otm_put},
        ],
    }
    path = tmp_path / "samples.jsonl"
    path.write_text(json.dumps(poc_cell) + "\n")
    store = load_skew_store(path)
    # iv(otm_put) ~ 0.28, iv(atm) ~ 0.20 -> skew_put_atm ~ 0.08 (positive smirk).
    assert store["AAA"]["skew_put_atm"].iloc[-1] > 0.0


# --- IVSkewPanel PIT gather -------------------------------------------------


def _panel(tmp_path) -> IVSkewPanel:
    path = tmp_path / "samples.jsonl"
    _write(
        path,
        [
            _cell("AAA", "2019-01-02", 0.05),
            _cell("AAA", "2019-02-01", 0.07),
            _cell("AAA", "2019-03-01", 0.09),
        ],
    )
    return IVSkewPanel.from_store(load_skew_store(path))


def test_gather_returns_asof_piecewise_constant(tmp_path):
    panel = _panel(tmp_path)
    # Mid-February: the last decision on/before is the 2019-02-01 one (0.07).
    got = panel.gather(["AAA"], pd.Timestamp("2019-02-15", tz="UTC"))
    assert got["AAA"]["skew_put_atm"].iloc[-1] == 0.07
    # Exactly on a decision date: that day is included (side="right").
    got = panel.gather(["AAA"], pd.Timestamp("2019-03-01", tz="UTC"))
    assert got["AAA"]["skew_put_atm"].iloc[-1] == 0.09


def test_gather_no_lookahead(tmp_path):
    panel = _panel(tmp_path)
    # Before the first decision -> the symbol is absent (neutral downstream).
    assert panel.gather(["AAA"], pd.Timestamp("2018-12-31", tz="UTC")) == {}
    # A decision dated AFTER as_of is never returned: at 2019-02-15 the 0.09
    # March value is invisible.
    hist = panel.gather(["AAA"], pd.Timestamp("2019-02-15", tz="UTC"))["AAA"]
    assert 0.09 not in hist["skew_put_atm"].tolist()
    assert hist.index.max() <= pd.Timestamp("2019-02-15", tz="UTC")


def test_gather_identity_vs_from_scratch_lookup(tmp_path):
    """The panel gather must equal a from-scratch as-of slice of the same store
    -- the FeaturePanel equivalence property, for skew."""
    path = tmp_path / "samples.jsonl"
    _write(
        path,
        [
            _cell("AAA", "2019-01-02", 0.05),
            _cell("AAA", "2019-02-01", 0.07),
            _cell("AAA", "2019-03-01", 0.09),
        ],
    )
    store = load_skew_store(path)
    panel = IVSkewPanel.from_store(store)
    as_of = pd.Timestamp("2019-02-20", tz="UTC")
    from_scratch = store["AAA"].loc[:as_of]
    pd.testing.assert_frame_equal(panel.gather(["AAA"], as_of)["AAA"], from_scratch)


def test_gather_absent_symbol_omitted(tmp_path):
    panel = _panel(tmp_path)
    got = panel.gather(["AAA", "ZZZ"], pd.Timestamp("2019-02-15", tz="UTC"))
    assert "ZZZ" not in got
    assert "AAA" in got
