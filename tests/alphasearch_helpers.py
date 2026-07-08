"""Deterministic synthetic fixtures shared by the alphasearch tests.

Not a test module (no test_ prefix): imported by test_alphasearch_sweep.py,
test_alphasearch_lookahead.py and the golden sweep test, mirroring
tests/backtest_helpers.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from trading.alphasearch.panel import PanelData, options_from_cells


def make_cell(
    symbol: str,
    date: str,
    *,
    atm_iv: float = 0.30,
    put_iv: float = 0.34,
    call_iv: float = 0.28,
    skew_put_atm: float = 0.05,
    skew_put_call: float = 0.02,
) -> dict:
    """One samples.jsonl-shaped options cell with all three legs present."""
    return {
        "symbol": symbol,
        "decision_date": date,
        "skew_put_atm": skew_put_atm,
        "skew_put_call": skew_put_call,
        "contracts": [
            {"role": "atm", "bid": 4.0, "ask": 4.2, "mid": 4.1, "iv": atm_iv,
             "volume": 100},
            {"role": "otm_put", "iv": put_iv, "volume": 50},
            {"role": "otm_call", "iv": call_iv, "volume": 25},
        ],
    }


def month_firsts(idx: pd.DatetimeIndex) -> list[pd.Timestamp]:
    """First index entry of each calendar month (the fixture's decision dates)."""
    firsts: dict[str, pd.Timestamp] = {}
    for date in idx:
        firsts.setdefault(date.strftime("%Y-%m"), date)
    return [firsts[m] for m in sorted(firsts)]


def make_panel(
    n_symbols: int = 16,
    start: str = "2020-01-02",
    periods: int = 130,
    seed: int = 7,
    with_options: bool = True,
    with_fundamentals: bool = True,
) -> PanelData:
    """Symbol S<i> drifts at (i - n/2)*2bp/day plus small seeded noise: momentum
    ranks are stable (so a momentum L/S spread has a large true alpha), values
    are bit-reproducible across runs, and 16 names >= MIN_NAMES=15 so default
    sort parameters trade every eligible date (as terciles, being < 50)."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=periods, freq="B", tz="UTC")
    names = [f"S{i:02d}" for i in range(n_symbols)]
    closes: dict[str, pd.Series] = {}
    for i, sym in enumerate(names):
        drift = (i - n_symbols / 2) * 2e-4
        rets = drift + rng.normal(0.0, 0.002, size=periods)
        closes[sym] = pd.Series(100.0 * np.cumprod(1 + rets), index=idx)
    options: dict[str, pd.DataFrame] = {}
    if with_options:
        cells = []
        for date in month_firsts(idx):
            iso = date.date().isoformat()
            for i, sym in enumerate(names):
                cells.append(make_cell(
                    sym, iso,
                    atm_iv=0.20 + 0.01 * i,
                    put_iv=0.24 + 0.01 * i,
                    call_iv=0.18 + 0.01 * i,
                    skew_put_atm=0.02 + 0.005 * i,
                    skew_put_call=0.01 + 0.002 * i,
                ))
        options = options_from_cells(cells)
    fundamentals: dict[str, pd.DataFrame] = {}
    if with_fundamentals:
        # Two filings (initial + mid-fixture) so the no-look-ahead test has
        # post-cutoff fundamentals to perturb -- one filing would make that
        # family's check vacuous.
        filed = pd.DatetimeIndex([idx[0], idx[len(idx) // 2]])
        for i, sym in enumerate(names):
            fundamentals[sym] = pd.DataFrame(
                {
                    "gross_profitability": [0.10 + 0.02 * i, 0.12 + 0.02 * i],
                    "ttm_net_income": [1e6 * (i + 1), 1.1e6 * (i + 1)],
                    "book_equity": [5e6 * (i + 1), 5.2e6 * (i + 1)],
                    "shares_outstanding": [1e6, 1e6],
                },
                index=filed,
            )
    return PanelData(closes=closes, options=options, fundamentals=fundamentals,
                     symbols=tuple(names))


def make_factors(
    start: str = "2019-12-02", periods: int = 160, seed: int = 3
) -> pd.DataFrame:
    """Synthetic Ken-French-shaped daily factors covering the fixture window.
    Uncorrelated with the fixture returns by construction (different seed), so
    the fixture's drift spread shows up as ALPHA, not loadings."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=periods, freq="B", tz="UTC")
    return pd.DataFrame(
        {
            "Mkt-RF": rng.normal(0.0004, 0.01, size=periods),
            "SMB": rng.normal(0.0, 0.005, size=periods),
            "HML": rng.normal(0.0, 0.005, size=periods),
            "Mom": rng.normal(0.0, 0.006, size=periods),
            "RF": 0.0001,
        },
        index=idx,
    )
