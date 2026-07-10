"""Deterministic synthetic fixtures shared by the alphasearch tests.

Not a test module (no test_ prefix): imported by test_alphasearch_sweep.py,
test_alphasearch_lookahead.py and the golden sweep test, mirroring
tests/backtest_helpers.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from trading.alphasearch.panel import PanelData, compute_rolling_features, options_from_cells


def make_cell(
    symbol: str,
    date: str,
    *,
    atm_iv: float = 0.30,
    put_iv: float = 0.34,
    call_iv: float = 0.28,
    skew_put_atm: float = 0.05,
    skew_put_call: float = 0.02,
    with_volume: bool = True,
    atm_oi: float | None = None,
    put_oi: float | None = None,
    call_oi: float | None = None,
    far_atm_iv: float | None = None,
) -> dict:
    """One samples.jsonl-shaped options cell with all three legs present.
    with_volume=False reproduces the largecap gather (no volume keys on any
    leg); True reproduces the mid-cap gather (volumes 100/50/25).

    2026-07-09 options-v2 batch (oi_put_call/d_oi/iv_term_slope): atm_oi/
    put_oi/call_oi set each leg's open_interest key; None (the default)
    OMITS the key entirely (leg-present-but-unmeasured), keeping every
    existing caller's cell shape OI-absent. far_atm_iv, when given, adds a
    minimal far block (role=atm only) carrying that IV; None (the default)
    omits "far" entirely."""
    contracts = [
        {"role": "atm", "bid": 4.0, "ask": 4.2, "mid": 4.1, "iv": atm_iv},
        {"role": "otm_put", "mid": 2.0, "iv": put_iv},
        {"role": "otm_call", "mid": 1.5, "iv": call_iv},
    ]
    if with_volume:
        for contract, volume in zip(contracts, (100, 50, 25), strict=True):
            contract["volume"] = volume
    for contract, oi in zip(contracts, (atm_oi, put_oi, call_oi), strict=True):
        if oi is not None:
            contract["open_interest"] = oi
    cell = {
        "symbol": symbol,
        "decision_date": date,
        "skew_put_atm": skew_put_atm,
        "skew_put_call": skew_put_call,
        "contracts": contracts,
    }
    if far_atm_iv is not None:
        cell["far"] = {"contracts": [{"role": "atm", "iv": far_atm_iv}]}
    return cell


def month_firsts(idx: pd.DatetimeIndex) -> list[pd.Timestamp]:
    """First index entry of each calendar month (the fixture's decision dates)."""
    firsts: dict[str, pd.Timestamp] = {}
    for date in idx:
        firsts.setdefault(date.strftime("%Y-%m"), date)
    return [firsts[m] for m in sorted(firsts)]


def assemble_panel(
    bars: dict[str, pd.DataFrame],
    options: dict[str, pd.DataFrame],
    fundamentals: dict[str, pd.DataFrame],
    factors: pd.DataFrame,
    *,
    insider: dict[str, pd.DataFrame] | None = None,
    has_option_volume: bool = False,
    sectors: dict[str, str] | None = None,
) -> PanelData:
    """PanelData from raw stores, deriving what build_panel derives (closes
    from bars). The lookahead test perturbs RAW stores and reassembles
    through here, so derived state (Task 2: precomputed rolling features) is
    recomputed from the perturbed inputs, never perturbed directly."""
    closes = {s: frame["close"] for s, frame in bars.items()}
    return PanelData(
        closes=closes, options=options, fundamentals=fundamentals,
        insider={} if insider is None else insider,
        symbols=tuple(sorted(bars)), bars=bars, factors=factors,
        features=compute_rolling_features(closes, factors),
        has_option_volume=has_option_volume,
        sectors={} if sectors is None else sectors,
    )


def make_panel(
    n_symbols: int = 16,
    start: str = "2020-01-02",
    periods: int = 130,
    seed: int = 7,
    with_options: bool = True,
    with_fundamentals: bool = True,
    with_option_volume: bool = True,
    with_open_interest: bool = True,
    with_insider: bool = True,
    factors: pd.DataFrame | None = None,
) -> PanelData:
    """Symbol S<i> drifts at (i - n/2)*2bp/day plus small seeded noise (same
    recipe/rng order as ever: closes are bit-identical to the pre-bars
    fixture). Bars extend the closes deterministically: open gaps up
    1bp*(i+1) from the prior close (a per-symbol overnight drift), high/low
    bracket the close at +-(0.2+0.05i)%, volume 1e5*(i+1), div_cash
    0.01*(i+1) daily, split_factor 1.0, close_raw == close (no synthetic
    split -- div_yield's raw-basis fix is exercised by its own dedicated
    fixture in test_alphasearch_tier1.py, not this general-purpose panel).
    Fundamentals file THREE times so the
    300-day YoY rule and the post-cutoff perturbation are both exercised on
    long fixtures (positions 0 / 63% / 95% of the index).

    Insider fixture (with_insider, deterministic, NO rng use -- closes must
    stay bit-identical): at every month-first FILED date, symbol i gets
    (i % 8) distinct buyers (owner_cik 1000*(i+1)+j, the first an officer,
    the second a director) buying 100*(i+1) sh @ 10, plus ONE ten-pct seller
    of 50*(i+1) sh @ 10. i % 8 == 0 symbols are covered-but-buyless
    (cluster_buys' real 0; npr -1). trans_date = filed - 20d, so month-first
    filings just after a cutoff carry trans_dates BEFORE it -- the
    straddling rows the lookahead perturbation needs to catch trans_date
    keying. 8 distinct cluster values keeps segment-free sorts
    non-degenerate (>= 3 buckets).

    with_open_interest (2026-07-09 options-v2 batch, default True so
    oi_put_call/d_oi/iv_term_slope produce real values on this general-
    purpose fixture): every leg gets a per-symbol constant open_interest
    (atm/put/call distinct so oi_put_call is non-degenerate) and every cell
    gets a far block with a distinct ATM IV -- OI is constant month to
    month, so d_oi is a real (zero) delta rather than NaN."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=periods, freq="B", tz="UTC")
    names = [f"S{i:02d}" for i in range(n_symbols)]
    bars: dict[str, pd.DataFrame] = {}
    for i, sym in enumerate(names):
        drift = (i - n_symbols / 2) * 2e-4
        rets = drift + rng.normal(0.0, 0.002, size=periods)
        close = pd.Series(100.0 * np.cumprod(1 + rets), index=idx)
        open_ = close.shift(1) * (1 + 1e-4 * (i + 1))
        open_.iloc[0] = close.iloc[0]
        span = 0.002 + 0.0005 * i
        bars[sym] = pd.DataFrame(
            {"open": open_, "high": close * (1 + span), "low": close * (1 - span),
             "close": close, "volume": 1e5 * (i + 1), "div_cash": 0.01 * (i + 1),
             "split_factor": 1.0, "close_raw": close},
            index=idx,
        )
    if factors is None:
        factors = make_factors()
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
                    with_volume=with_option_volume,
                    atm_oi=1000.0 + i if with_open_interest else None,
                    put_oi=1200.0 + i if with_open_interest else None,
                    call_oi=900.0 + i if with_open_interest else None,
                    far_atm_iv=(0.22 + 0.01 * i) if with_open_interest else None,
                ))
        options = options_from_cells(cells)
    fundamentals: dict[str, pd.DataFrame] = {}
    if with_fundamentals:
        filed = pd.DatetimeIndex(
            [idx[0], idx[(63 * len(idx)) // 100], idx[(95 * len(idx)) // 100]]
        )
        for i, sym in enumerate(names):
            fundamentals[sym] = pd.DataFrame(
                {
                    "gross_profitability": [0.10 + 0.02 * i, 0.12 + 0.02 * i,
                                            0.13 + 0.02 * i],
                    "ttm_net_income": [1e6 * (i + 1), 1.1e6 * (i + 1),
                                       1.2e6 * (i + 1)],
                    "book_equity": [5e6 * (i + 1), 5.2e6 * (i + 1),
                                    5.3e6 * (i + 1)],
                    "shares_outstanding": [1e6 * (i + 1), 1.02e6 * (i + 1),
                                           1.04e6 * (i + 1)],
                    "assets": [1e7 * (i + 1), 1.1e7 * (i + 1), 1.15e7 * (i + 1)],
                    "revenue_ttm": [2e7 * (i + 1), 2.2e7 * (i + 1),
                                    2.3e7 * (i + 1)],
                },
                index=filed,
            )
    insider: dict[str, pd.DataFrame] = {}
    if with_insider:
        for i, sym in enumerate(names):
            rows: list[dict] = []
            for date in month_firsts(idx):
                trans = date - pd.Timedelta(20, unit="D")
                for j in range(i % 8):
                    rows.append({
                        "filed": date, "trans_date": trans, "code": "P",
                        "shares": 100.0 * (i + 1), "price": 10.0,
                        "value": 1000.0 * (i + 1),
                        "owner_cik": 1000 * (i + 1) + j,
                        "is_officer": j == 0, "is_director": j == 1,
                        "is_ten_pct": False,
                    })
                rows.append({
                    "filed": date, "trans_date": trans, "code": "S",
                    "shares": 50.0 * (i + 1), "price": 10.0,
                    "value": 500.0 * (i + 1), "owner_cik": 9000 + i,
                    "is_officer": False, "is_director": False,
                    "is_ten_pct": True,
                })
            insider[sym] = pd.DataFrame(rows).set_index("filed")
    sectors = {sym: ("manufacturing-tech" if i % 2 == 0 else "finance")
               for i, sym in enumerate(names)}
    return assemble_panel(
        bars, options, fundamentals, factors,
        insider=insider,
        has_option_volume=with_options and with_option_volume,
        sectors=sectors,
    )


def make_spy_closes(
    start: str = "2015-01-02", periods: int = 4000, seed: int = 99,
    drift: float = 0.0003,
) -> pd.Series:
    """Synthetic SPY buy-and-hold close series (R1 gate amendment) spanning
    essentially any fixture window used across the alphasearch tests --
    DISCOVERY_WINDOW, the holdout, the various small WINDOW fixtures -- so
    run_battery/leaderboard tests never depend on the real (gitignored)
    data/equities-tiingo/SPY cache."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=periods, freq="B", tz="UTC")
    rets = drift + rng.normal(0.0, 0.01, size=periods)
    return pd.Series(100.0 * np.cumprod(1 + rets), index=idx)


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
