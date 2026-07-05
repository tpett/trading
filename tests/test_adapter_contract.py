"""Shared venue-adapter data contract (spec: Testing Strategy).

Both venues run the same assertions in v1 since both paper-trade from day
one. Network touchpoints are monkeypatched; the contract covers universe
shape, constraints, and OHLCV frame invariants.
"""

import datetime
from pathlib import Path
from typing import get_args

import pandas as pd
import pytest

from trading.config import load_venue_config
from trading.venues.base import OHLCV_COLUMNS, DataFetchError, SymbolStatus
from trading.venues.crypto import CryptoAdapter
from trading.venues.equities import EquitiesAdapter

START = datetime.date(2026, 6, 1)
END = datetime.date(2026, 7, 1)
AS_OF = datetime.date(2026, 7, 1)
VALID_STATUSES = set(get_args(SymbolStatus))


def _fake_yf(symbol: str, start: datetime.date, end: datetime.date) -> pd.DataFrame:
    idx = pd.date_range(start, END, freq="B")
    return pd.DataFrame(
        {"Open": 100.0, "High": 101.0, "Low": 99.0, "Close": 100.5, "Volume": 1e6}, index=idx
    )


def _fake_kraken(pair: str, since_ms: int) -> list[list[float]]:
    idx = pd.date_range(START, END, freq="D", tz="UTC")
    return [[int(ts.timestamp() * 1000), 100.0, 101.0, 99.0, 100.5, 1e6] for ts in idx]


def make_adapter(venue: str, monkeypatch, tmp_path, empty: bool = False):
    config = load_venue_config(venue, Path("config"))
    universe = tmp_path / f"{venue}_universe.csv"
    if venue == "equities":
        universe.write_text(
            "# provenance comment line\n"
            "symbol,index,start,end\n"
            "AAA,sp500,2018-01-01,\n"
            "BBB,sp500,2018-01-01,\n"
        )
        fetch = (lambda s, a, b: pd.DataFrame()) if empty else _fake_yf
        monkeypatch.setattr("trading.venues.equities._yf_download", fetch)
        return EquitiesAdapter(config, membership_csv=universe), config, "AAA"
    universe.write_text("symbol,status\nBTC,tradable\nETH,sell_only\n")
    fetch = (lambda p, s: []) if empty else _fake_kraken
    monkeypatch.setattr("trading.venues.crypto._kraken_fetch", fetch)
    return CryptoAdapter(config, universe_csv=universe), config, "BTC"


VENUES = ["equities", "crypto"]


@pytest.mark.parametrize("venue", VENUES)
def test_universe_returns_symbol_infos_with_valid_statuses(venue, monkeypatch, tmp_path):
    adapter, _, _ = make_adapter(venue, monkeypatch, tmp_path)
    infos = adapter.universe(AS_OF)
    assert len(infos) == 2
    assert all(info.status in VALID_STATUSES for info in infos)
    assert all(isinstance(info.symbol, str) and info.symbol for info in infos)


@pytest.mark.parametrize("venue", VENUES)
def test_constraints_mirror_config_costs(venue, monkeypatch, tmp_path):
    adapter, config, _ = make_adapter(venue, monkeypatch, tmp_path)
    constraints = adapter.constraints()
    assert constraints.taker_fee_bps == config.costs.taker_fee_bps
    assert constraints.maker_fee_bps == config.costs.maker_fee_bps
    assert constraints.slippage_bps == config.costs.slippage_bps
    assert constraints.settlement_days == config.costs.settlement_days
    assert constraints.trades_24_7 == config.costs.trades_24_7


@pytest.mark.parametrize("venue", VENUES)
def test_fetch_ohlcv_returns_utc_ohlcv_frame_sliced_to_range(venue, monkeypatch, tmp_path):
    adapter, _, symbol = make_adapter(venue, monkeypatch, tmp_path)
    df = adapter.fetch_ohlcv(symbol, START, END)
    assert list(df.columns) == OHLCV_COLUMNS
    assert str(df.index.tz) == "UTC"
    assert df.index.is_monotonic_increasing
    assert (df.dtypes == "float64").all()
    assert df.index.min() >= pd.Timestamp(START, tz="UTC")
    assert df.index.max() <= pd.Timestamp(END, tz="UTC")


@pytest.mark.parametrize("venue", VENUES)
def test_fetch_ohlcv_empty_raises_data_fetch_error(venue, monkeypatch, tmp_path):
    adapter_empty, _, symbol = make_adapter(venue, monkeypatch, tmp_path, empty=True)
    with pytest.raises(DataFetchError):
        adapter_empty.fetch_ohlcv(symbol, START, END)
