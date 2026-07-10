"""Form 4 insider family: hand-computed fixtures pinning the FROZEN spec
section 3 table -- definitions, signs (+ all three), and the NaN conventions
(cluster's 0-vs-NaN never-covered distinction; officer's raw-price basis)."""

from __future__ import annotations

import math

import pandas as pd

from trading.alphasearch.panel import BAR_COLUMNS, PanelData
from trading.alphasearch.spec import SIGNALS

AS_OF = pd.Timestamp("2020-06-30", tz="UTC")


def _insider(rows: list[tuple]) -> pd.DataFrame:
    """(filed_iso, code, shares, price, owner_cik, is_officer[, is_director])
    -> store frame. is_director defaults to False when omitted."""
    padded = [r if len(r) == 7 else (*r, False) for r in rows]
    frame = pd.DataFrame(
        padded,
        columns=["filed", "code", "shares", "price", "owner_cik", "is_officer",
                 "is_director"],
    )
    frame["filed"] = pd.to_datetime(frame["filed"]).dt.tz_localize("UTC")
    frame["trans_date"] = frame["filed"] - pd.Timedelta(2, unit="D")
    frame["value"] = frame["shares"] * frame["price"]
    frame["is_ten_pct"] = False
    return frame.set_index("filed").sort_index(kind="mergesort")


def _bars(close_raw: float = 50.0) -> pd.DataFrame:
    idx = pd.date_range("2020-06-01", periods=21, freq="B", tz="UTC")
    close = pd.Series(50.0, index=idx)
    return pd.DataFrame(
        {"open": close, "high": close, "low": close, "close": close,
         "volume": 1000.0, "div_cash": 0.0, "split_factor": 1.0,
         "close_raw": close_raw},
        index=idx,
    )


def _panel(insider: dict[str, pd.DataFrame], symbols: tuple[str, ...],
           *, with_fundamentals: bool = True, with_bars: bool = True) -> PanelData:
    filed = pd.DatetimeIndex([pd.Timestamp("2020-01-06", tz="UTC")])
    fundamentals = (
        {s: pd.DataFrame({"shares_outstanding": [1e6]}, index=filed) for s in symbols}
        if with_fundamentals else {}
    )
    bars = {s: _bars() for s in symbols} if with_bars else {}
    return PanelData(
        closes={s: b["close"] for s, b in bars.items()},
        fundamentals=fundamentals, insider=insider, bars=bars, symbols=symbols,
    )


def _score(name: str, panel: PanelData) -> pd.Series:
    return SIGNALS[name].fn(panel.view(AS_OF), AS_OF)


def test_npr_90_net_purchase_ratio_and_sign():
    insider = {
        # AAA: buys 100*10=1000, sells 50*10=500 -> (1000-500)/1500 = 1/3
        "AAA": _insider([("2020-06-15", "P", 100.0, 10.0, 1, True),
                         ("2020-06-16", "S", 50.0, 10.0, 2, False)]),
        # BBB: sells only -> -1 (net seller, ranked at the bottom: + sign)
        "BBB": _insider([("2020-06-15", "S", 50.0, 10.0, 3, False)]),
        # CCC: covered, but nothing filed in the trailing 90d -> NaN (no P/S
        # rows in window is the spec's npr NaN case, distinct from cluster's 0)
        "CCC": _insider([("2019-06-15", "P", 10.0, 10.0, 4, False)]),
    }
    scores = _score("npr_90", _panel(insider, ("AAA", "BBB", "CCC", "DDD")))
    assert math.isclose(scores["AAA"], 1.0 / 3.0, rel_tol=1e-12)
    assert scores["BBB"] == -1.0
    assert scores["AAA"] > scores["BBB"]          # net buying ranks higher
    assert math.isnan(scores["CCC"])
    assert math.isnan(scores["DDD"])              # never covered


def test_npr_90_all_nan_values_in_window_is_nan_not_zero():
    # A footnote-priced (value=NaN) window must not fabricate 0/0.
    frame = _insider([("2020-06-15", "P", 100.0, 10.0, 1, True)])
    frame["value"] = float("nan")
    scores = _score("npr_90", _panel({"AAA": frame}, ("AAA",)))
    assert math.isnan(scores["AAA"])


def test_cluster_buys_90_distinct_owners_and_the_zero_vs_nan_distinction():
    insider = {
        # Two P rows, SAME owner -> 1; the sale's owner never counts.
        "ONE": _insider([("2020-06-10", "P", 10.0, 5.0, 42, False),
                         ("2020-06-20", "P", 10.0, 5.0, 42, False),
                         ("2020-06-21", "S", 10.0, 5.0, 43, False)]),
        "TWO": _insider([("2020-06-10", "P", 10.0, 5.0, 1, False),
                         ("2020-06-20", "P", 10.0, 5.0, 2, False)]),
        # Covered (a sale in-window) but NO buys -> 0.0, a REAL value.
        "QUIET": _insider([("2020-06-10", "S", 10.0, 5.0, 9, False)]),
        # Covered only by an out-of-window old row -> still 0.0 (covered).
        "OLD": _insider([("2019-01-10", "P", 10.0, 5.0, 9, False)]),
    }
    scores = _score("cluster_buys_90", _panel(insider, ("ONE", "TWO", "QUIET", "OLD", "NEVER")))
    assert scores["ONE"] == 1.0
    assert scores["TWO"] == 2.0
    assert scores["TWO"] > scores["ONE"]          # more buyers = conviction
    assert scores["QUIET"] == 0.0                 # quiet, not missing
    assert scores["OLD"] == 0.0                   # never-covered != quiet: covered
    assert math.isnan(scores["NEVER"])            # NO row ever filed <= as_of


def test_officer_buy_90_raw_price_basis_and_nan_conventions():
    insider = {
        # Officer buys 100 sh @ 10 = 1000; a NON-officer buy of 2000 must not
        # count, NOR must a director-only buy of 3000 (is_officer=False,
        # is_director=True) -- officer_buy_90 is officer-only, not
        # officer-or-director. shares_outstanding 1e6, close_raw 50 ->
        # 1000 / 5e7 = 2e-5.
        "AAA": _insider([("2020-06-15", "P", 100.0, 10.0, 1, True),
                         ("2020-06-16", "P", 200.0, 10.0, 2, False),
                         ("2020-06-17", "P", 300.0, 10.0, 4, False, True)]),
        # Covered, no officer buying -> 0.0 (real quiet), not NaN.
        "BBB": _insider([("2020-06-16", "S", 10.0, 10.0, 3, False)]),
    }
    scores = _score("officer_buy_90", _panel(insider, ("AAA", "BBB", "NEVER")))
    assert math.isclose(scores["AAA"], 1000.0 / (1e6 * 50.0), rel_tol=1e-12)
    assert scores["BBB"] == 0.0
    assert math.isnan(scores["NEVER"])


def test_officer_buy_90_nan_without_shares_or_close_raw():
    insider = {"AAA": _insider([("2020-06-15", "P", 100.0, 10.0, 1, True)])}
    no_fund = _score("officer_buy_90", _panel(insider, ("AAA",), with_fundamentals=False))
    assert math.isnan(no_fund["AAA"])             # no shares_outstanding
    no_bars = _score("officer_buy_90", _panel(insider, ("AAA",), with_bars=False))
    assert math.isnan(no_bars["AAA"])             # no close_raw at as_of
    # A legacy narrow cache: bars exist but close_raw is NaN -> NaN, never a
    # fabricated adjusted-close basis (the div_yield lesson).
    panel = _panel(insider, ("AAA",))
    narrow = panel.bars["AAA"].copy()
    narrow["close_raw"] = float("nan")
    panel = PanelData(closes=panel.closes, fundamentals=panel.fundamentals,
                      insider=panel.insider, bars={"AAA": narrow}, symbols=("AAA",))
    assert math.isnan(_score("officer_buy_90", panel)["AAA"])
    assert list(narrow.columns) == BAR_COLUMNS


def test_insider_signals_all_nan_without_any_insider_data():
    panel = _panel({}, ("AAA", "BBB"))
    for name in ("npr_90", "cluster_buys_90", "officer_buy_90"):
        assert _score(name, panel).isna().all()
