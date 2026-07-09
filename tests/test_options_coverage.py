"""v2-vs-v1 gather coverage/agreement report (spec section 5). Pure fixtures,
no network, no real data files."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from trading.research.options_coverage import (
    IV_DRIFT_RED_FLAG,
    coverage_report,
    iv_deltas,
    load_cells,
)


def _leg(role: str, strike: float, iv: float | None, **extra) -> dict:
    return {"role": role, "type": "call" if role != "otm_put" else "put",
            "strike": strike, "bid": 1.0, "ask": 1.2, "close": 1.1, "mid": 1.1,
            "iv": iv, "volume": extra.pop("volume", None), "count": 3, **extra}


def _cell(symbol: str, day: str, *, iv: float = 0.30, volume=None, oi=False,
          far=False, expiration: str = "2019-04-18") -> dict:
    legs = [
        _leg("atm", 100.0, iv, volume=volume, **({"open_interest": 100} if oi else {})),
        _leg("otm_put", 90.0, iv + 0.05, volume=volume),
    ]
    cell = {
        "symbol": symbol, "decision_date": day, "spot_at_decision": 100.0,
        "target_expiration": expiration, "days_to_expiry": 48,
        "contracts": legs, "skew_put_atm": 0.05, "skew_put_call": None,
    }
    if far:
        cell["far"] = {"target_expiration": "2019-05-17", "days_to_expiry": 77,
                       "contracts": [_leg("atm", 100.0, iv - 0.02)],
                       "skew_put_atm": None, "skew_put_call": None}
    return cell


def test_load_cells_skips_torn_and_keyless_lines(tmp_path: Path):
    path = tmp_path / "samples.jsonl"
    path.write_text(
        json.dumps(_cell("AAA", "2019-03-01")) + "\n"
        + "{ torn line\n"
        + json.dumps({"decision_date": "2019-03-01"}) + "\n"  # no symbol
    )
    cells = load_cells(path)
    assert [c["symbol"] for c in cells] == ["AAA"]
    assert load_cells(tmp_path / "absent.jsonl") == []


def test_iv_deltas_matches_same_contract_only():
    v1 = [_cell("AAA", "2019-03-01", iv=0.300)]
    v2 = [
        _cell("AAA", "2019-03-01", iv=0.302),                        # same contract: |d|=0.002
        _cell("AAA", "2019-04-01", iv=0.500),                        # different date: no match
        _cell("AAA", "2019-03-01", iv=0.900, expiration="2019-05-17"),  # different contract
    ]
    deltas = iv_deltas(v2, v1)
    # atm AND otm_put both match on the 03-01 cell (put ivs 0.352 vs 0.350).
    assert sorted(round(d, 6) for d in deltas) == [0.002, 0.002]


def test_coverage_report_rates_and_agreement():
    v1 = [_cell("AAA", "2019-03-01", iv=0.300), _cell("BBB", "2019-03-01", iv=0.40)]
    v2 = [
        _cell("AAA", "2019-03-01", iv=0.302, volume=50, oi=True, far=True),
        _cell("BBB", "2019-03-01", iv=0.400),  # no volume/oi/far
    ]
    report = coverage_report(v2, v1)
    assert report["cells_v2"] == 2 and report["cells_v1"] == 2
    assert report["leg_volume_rate"] == 0.5   # 1 of 2 cells has a volume-bearing near leg
    assert report["oi_leg_rate"] == 0.25      # 1 of 4 near legs carries open_interest
    assert report["far_rate"] == 0.5
    assert report["iv_overlap_legs"] == 4
    assert report["iv_median_abs_delta"] == pytest.approx(0.001)  # {.002,.002,0,0} -> median .001
    assert report["iv_red_flag"] is False


def test_coverage_report_red_flags_large_iv_drift():
    v1 = [_cell("AAA", "2019-03-01", iv=0.30)]
    v2 = [_cell("AAA", "2019-03-01", iv=0.30 + 2 * IV_DRIFT_RED_FLAG)]
    report = coverage_report(v2, v1)
    assert report["iv_red_flag"] is True


def test_coverage_report_empty_inputs_yield_none_rates():
    report = coverage_report([], [])
    assert report["cells_v2"] == 0
    assert report["leg_volume_rate"] is None
    assert report["oi_leg_rate"] is None
    assert report["far_rate"] is None
    assert report["iv_median_abs_delta"] is None
    assert report["iv_red_flag"] is False
