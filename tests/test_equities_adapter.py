import datetime
from pathlib import Path

import pandas as pd
import pytest

from trading.config import load_venue_config
from trading.venues.base import OHLCV_COLUMNS, DataFetchError, SymbolInfo, VenueConstraints
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


def test_universe_reads_symbols_from_csv(tmp_path):
    csv = tmp_path / "universe.csv"
    csv.write_text("symbol\nAAPL\nMSFT\nNVDA\n")
    adapter = EquitiesAdapter(CONFIG, universe_csv=csv)
    infos = adapter.universe(datetime.date(2026, 7, 1))
    assert infos == [
        SymbolInfo("AAPL", "tradable"),
        SymbolInfo("MSFT", "tradable"),
        SymbolInfo("NVDA", "tradable"),
    ]


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
