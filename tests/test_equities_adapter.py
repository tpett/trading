import datetime
from pathlib import Path

import pandas as pd
import pytest

from trading.config import load_venue_config
from trading.venues.base import OHLCV_COLUMNS, DataFetchError, VenueConstraints
from trading.venues.equities import EquitiesAdapter

CONFIG = load_venue_config("equities", Path("config"))


def _yf_style_frame(symbol: str) -> pd.DataFrame:
    """Mimic yfinance.download output: naive index, MultiIndex (Price, Ticker) columns."""
    idx = pd.date_range("2026-01-05", periods=5, freq="B")  # naive, like yfinance
    data = {
        ("Open", symbol): [10.0, 10.5, 10.2, 10.8, 11.0],
        ("High", symbol): [10.6, 10.9, 10.7, 11.2, 11.4],
        ("Low", symbol): [9.8, 10.1, 9.9, 10.5, 10.7],
        ("Close", symbol): [10.5, 10.2, 10.6, 11.0, 11.2],
        ("Volume", symbol): [1e6, 1.1e6, 9e5, 1.3e6, 1.2e6],
    }
    return pd.DataFrame(data, index=idx)


MEMBERSHIP_CSV = """# test fixture
symbol,index,start,end
AAA,sp500,2018-01-01,
BBB,sp500,2018-01-01,2020-06-01
CCC,ndx,2020-06-01,
"""


def _adapter_with(tmp_path, text: str) -> EquitiesAdapter:
    path = tmp_path / "membership.csv"
    path.write_text(text)
    config = load_venue_config("equities", Path("config"))
    return EquitiesAdapter(config, membership_csv=path)


def test_universe_is_point_in_time(tmp_path):
    adapter = _adapter_with(tmp_path, MEMBERSHIP_CSV)
    in_2019 = {i.symbol for i in adapter.universe(datetime.date(2019, 1, 2))}
    in_2021 = {i.symbol for i in adapter.universe(datetime.date(2021, 1, 4))}
    assert in_2019 == {"AAA", "BBB"}
    assert in_2021 == {"AAA", "CCC"}
    assert all(i.status == "tradable" for i in adapter.universe(datetime.date(2021, 1, 4)))


MEMBERSHIP_CSV_WITH_SP400 = """# test fixture
symbol,index,start,end
AAA,sp500,2018-01-01,
BBB,sp500,2018-01-01,2020-06-01
CCC,ndx,2020-06-01,
DDD,sp400,2018-01-01,
"""


def test_universe_excludes_sp400_by_default(tmp_path):
    # Live/paper invariant: sp400 rows exist in the CSV, but the default
    # config.universe.indices is ("sp500", "ndx") -- unchanged behavior.
    adapter = _adapter_with(tmp_path, MEMBERSHIP_CSV_WITH_SP400)
    symbols = {i.symbol for i in adapter.universe(datetime.date(2021, 1, 4))}
    assert symbols == {"AAA", "CCC"}
    assert "DDD" not in symbols


def test_universe_includes_sp400_when_config_opts_in(tmp_path):
    from dataclasses import replace

    path = tmp_path / "membership.csv"
    path.write_text(MEMBERSHIP_CSV_WITH_SP400)
    config = replace(CONFIG, universe=replace(CONFIG.universe, indices=("sp500", "ndx", "sp400")))
    adapter = EquitiesAdapter(config, membership_csv=path)
    symbols = {i.symbol for i in adapter.universe(datetime.date(2021, 1, 4))}
    assert symbols == {"AAA", "CCC", "DDD"}


def test_membership_intervals_returns_all_indices_unfiltered(tmp_path):
    # membership_intervals is index-agnostic by design (the recycling guard
    # cares whether a ticker is a current member of ANYTHING, not just the
    # indices this particular backtest opted into).
    adapter = _adapter_with(tmp_path, MEMBERSHIP_CSV_WITH_SP400)
    assert adapter.membership_intervals("DDD") == [("2018-01-01", "")]
    assert adapter.membership_intervals("BBB") == [("2018-01-01", "2020-06-01")]
    assert adapter.membership_intervals("ZZZ") == []


def test_membership_interval_boundaries_start_inclusive_end_exclusive(tmp_path):
    adapter = _adapter_with(tmp_path, MEMBERSHIP_CSV)
    on_start = {i.symbol for i in adapter.universe(datetime.date(2020, 6, 1))}
    day_before = {i.symbol for i in adapter.universe(datetime.date(2020, 5, 31))}
    assert "CCC" in on_start and "BBB" not in on_start  # start inclusive, end exclusive
    assert "BBB" in day_before and "CCC" not in day_before


def test_committed_membership_file_sanity():
    # The real committed file: plausible sizes, and known index churn visible.
    adapter = EquitiesAdapter(load_venue_config("equities", Path("config")))
    for day, low, high in [
        (datetime.date(2018, 6, 1), 450, 650),
        (datetime.date(2022, 6, 1), 450, 650),
        (datetime.date(2026, 7, 1), 450, 650),
    ]:
        count = len(adapter.universe(day))
        assert low <= count <= high, f"{day}: {count} members"
    early = {i.symbol for i in adapter.universe(datetime.date(2018, 6, 1))}
    today = {i.symbol for i in adapter.universe(datetime.date(2026, 7, 1))}
    assert early != today
    assert len(early - today) > 20  # real churn: many 2018 members are gone
    # sp400 rows exist in the committed CSV but are excluded by the default
    # config -- live/paper universe is untouched by their addition.
    assert not {i.symbol for i in adapter.universe(datetime.date(2026, 7, 1))} & {"CDK", "MDP"}


def test_committed_membership_file_sp400_opt_in():
    from dataclasses import replace

    base = load_venue_config("equities", Path("config"))
    sp400_only = replace(base, universe=replace(base.universe, indices=("sp400",)))
    adapter = EquitiesAdapter(sp400_only)
    for day, low, high in [
        (datetime.date(2019, 6, 1), 380, 420),
        (datetime.date(2022, 6, 1), 380, 420),
        (datetime.date(2026, 7, 1), 380, 420),
    ]:
        count = len(adapter.universe(day))
        assert low <= count <= high, f"{day}: {count} sp400 members"
    # Spot-check anchors (scout-verified against the Wikipedia changes table):
    # exact known removal dates, not just a plausible count.
    before_cdk = {i.symbol for i in adapter.universe(datetime.date(2022, 7, 5))}
    after_cdk = {i.symbol for i in adapter.universe(datetime.date(2022, 7, 6))}
    assert "CDK" in before_cdk and "CDK" not in after_cdk
    before_mdp = {i.symbol for i in adapter.universe(datetime.date(2020, 4, 26))}
    after_mdp = {i.symbol for i in adapter.universe(datetime.date(2020, 4, 27))}
    assert "MDP" in before_mdp and "MDP" not in after_mdp


def test_constraints_come_from_config():
    adapter = EquitiesAdapter(CONFIG)
    assert adapter.constraints() == VenueConstraints(
        taker_fee_bps=0.0,
        maker_fee_bps=0.0,
        slippage_bps=5.0,
        settlement_days=1,
        trades_24_7=False,
    )


def test_fetch_ohlcv_normalizes_yfinance_frame(monkeypatch):
    monkeypatch.setattr(
        "trading.venues.equities._yf_download", lambda s, start, end: _yf_style_frame(s)
    )
    adapter = EquitiesAdapter(CONFIG)
    df = adapter.fetch_ohlcv("AAPL", datetime.date(2026, 1, 5), datetime.date(2026, 1, 9))
    assert list(df.columns) == OHLCV_COLUMNS
    assert str(df.index.tz) == "UTC"
    assert df.index.is_monotonic_increasing
    assert df["close"].iloc[-1] == 11.2
    assert len(df) == 5


def test_fetch_ohlcv_slices_to_requested_range(monkeypatch):
    monkeypatch.setattr(
        "trading.venues.equities._yf_download", lambda s, start, end: _yf_style_frame(s)
    )
    adapter = EquitiesAdapter(CONFIG)
    df = adapter.fetch_ohlcv("AAPL", datetime.date(2026, 1, 6), datetime.date(2026, 1, 8))
    assert df.index.min() == pd.Timestamp("2026-01-06", tz="UTC")
    assert df.index.max() == pd.Timestamp("2026-01-08", tz="UTC")


def test_fetch_ohlcv_empty_raises(monkeypatch):
    monkeypatch.setattr(
        "trading.venues.equities._yf_download", lambda s, start, end: pd.DataFrame()
    )
    adapter = EquitiesAdapter(CONFIG)
    with pytest.raises(DataFetchError):
        adapter.fetch_ohlcv("AAPL", datetime.date(2026, 1, 5), datetime.date(2026, 1, 9))
