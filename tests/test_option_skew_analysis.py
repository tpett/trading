"""Offline, synthetic tests for the IV-skew cross-sectional analysis.

Every input is fabricated in-test (a synthetic samples.jsonl + underlying
parquets); nothing touches the network or the real gathered data directory.
The centrepiece is a hand-constructed case where skew and forward return are
PERFECTLY anti-correlated, so the Spearman/tercile math has a known answer.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from option_skew_analysis import (  # noqa: E402
    build_observations,
    forward_return,
    run,
    spearman,
    tercile_spread,
)

from trading.research.options_iv import DIV_YIELD, RATE, bs_price  # noqa: E402

DTE = 30.0
T = DTE / 365.0
SPOT = 100.0
ATM_VOL = 0.30
OTM_CALL_VOL = 0.25


def _cell(symbol: str, decision_date: str, put_vol: float) -> dict:
    """A sample cell whose otm_put vol (hence skew_put_atm) is put_vol - 0.30."""
    atm = bs_price(SPOT, 100.0, T, RATE, DIV_YIELD, ATM_VOL, True)
    otm_put = bs_price(SPOT, 90.0, T, RATE, DIV_YIELD, put_vol, False)
    otm_call = bs_price(SPOT, 110.0, T, RATE, DIV_YIELD, OTM_CALL_VOL, True)
    return {
        "symbol": symbol,
        "decision_date": decision_date,
        "spot_at_decision": SPOT,
        "target_expiration": "2024-02-16",
        "days_to_expiry": DTE,
        "contracts": [
            {"role": "atm", "type": "call", "strike": 100.0, "close": atm},
            {"role": "otm_put", "type": "put", "strike": 90.0, "close": otm_put},
            {"role": "otm_call", "type": "call", "strike": 110.0, "close": otm_call},
        ],
    }


def _write_underlying(
    data_dir: Path, symbol: str, dates: pd.DatetimeIndex, fwd_return: float
) -> None:
    """Underlying parquet whose close 20 rows after the first date gives fwd_return."""
    close = [SPOT] * len(dates)
    close[20] = SPOT * (1.0 + fwd_return)
    frame = pd.DataFrame(
        {
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "volume": [1_000_000] * len(dates),
        },
        index=dates,
    )
    out = data_dir / "underlying"
    out.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(out / f"{symbol}.parquet")


@pytest.fixture
def anticorrelated_dataset(tmp_path: Path) -> Path:
    """Six names on one date: skew increases as forward return decreases.

    put vols  -> skew_put_atm    fwd return
    0.32      -> ~0.02 (flattest) +0.10 (highest)
    ...
    0.42      -> ~0.12 (steepest) -0.10 (lowest)
    """
    data_dir = tmp_path / "options-poc"
    data_dir.mkdir()
    dates = pd.bdate_range("2024-01-02", periods=25, tz="UTC")
    decision_date = dates[0].date().isoformat()

    put_vols = [0.32, 0.34, 0.36, 0.38, 0.40, 0.42]
    fwd_returns = [0.10, 0.06, 0.02, -0.02, -0.06, -0.10]

    cells = []
    for i, (pv, fr) in enumerate(zip(put_vols, fwd_returns, strict=True)):
        sym = f"SYM{i}"
        cells.append(_cell(sym, decision_date, pv))
        _write_underlying(data_dir, sym, dates, fr)

    with (data_dir / "samples.jsonl").open("w") as fh:
        for c in cells:
            fh.write(json.dumps(c) + "\n")
    return data_dir


def test_forward_return_horizon() -> None:
    dates = pd.bdate_range("2024-01-02", periods=25, tz="UTC")
    close = [SPOT] * 25
    close[20] = 110.0
    df = pd.DataFrame({"close": close}, index=dates)
    assert forward_return(df, dates[0].date().isoformat(), SPOT, horizon=20) == pytest.approx(0.10)
    # Horizon running off the end -> None.
    assert forward_return(df, dates[10].date().isoformat(), SPOT, horizon=20) is None


def test_build_observations_recovers_skew_and_return(anticorrelated_dataset: Path) -> None:
    from option_skew_analysis import load_samples

    samples = load_samples(anticorrelated_dataset / "samples.jsonl")
    obs = build_observations(samples, anticorrelated_dataset)

    assert len(obs) == 6
    # Flattest name (SYM0): skew ~0.02, return +0.10; steepest (SYM5): ~0.12, -0.10.
    flat = obs.loc[obs["symbol"] == "SYM0"].iloc[0]
    steep = obs.loc[obs["symbol"] == "SYM5"].iloc[0]
    assert flat["skew_put_atm"] == pytest.approx(0.02, abs=3e-3)
    assert steep["skew_put_atm"] == pytest.approx(0.12, abs=3e-3)
    assert flat["fwd_return"] == pytest.approx(0.10)
    assert steep["fwd_return"] == pytest.approx(-0.10)


def test_spearman_perfect_anticorrelation(anticorrelated_dataset: Path) -> None:
    from option_skew_analysis import load_samples

    samples = load_samples(anticorrelated_dataset / "samples.jsonl")
    obs = build_observations(samples, anticorrelated_dataset)

    res = spearman(obs["skew_put_atm"], obs["fwd_return"])
    assert res.n == 6
    assert res.rho == pytest.approx(-1.0)  # monotone decreasing -> exactly -1
    assert res.t_stat is not None and res.t_stat < 0


def test_tercile_spread_bottom_minus_top_positive(anticorrelated_dataset: Path) -> None:
    from option_skew_analysis import load_samples

    samples = load_samples(anticorrelated_dataset / "samples.jsonl")
    obs = build_observations(samples, anticorrelated_dataset)

    res = tercile_spread(obs, "skew_put_atm")
    assert res.n_dates == 1
    # bottom tercile (SYM0,SYM1) mean +0.08; top (SYM4,SYM5) mean -0.08.
    assert res.mean_spread == pytest.approx(0.16)
    assert res.frac_positive == pytest.approx(1.0)


def test_spearman_too_few_points_returns_none() -> None:
    res = spearman(pd.Series([1.0, 2.0]), pd.Series([3.0, 4.0]))
    assert res.rho is None
    assert res.n == 2


def test_run_no_data_is_graceful(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = run(tmp_path / "does-not-exist")
    out = capsys.readouterr().out
    assert code == 0
    assert "No gathered data" in out


def test_run_end_to_end_prints_report(
    anticorrelated_dataset: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = run(anticorrelated_dataset)
    out = capsys.readouterr().out
    assert code == 0
    assert "IV SKEW -> FORWARD RETURN" in out
    assert "VERDICT" in out
    # Perfect anti-correlation on the level test should read as a signal.
    assert "skew_put_atm (level)" in out
