import datetime
from pathlib import Path

import pandas as pd
import pytest

from trading.config import load_venue_config
from trading.venues.base import OHLCV_COLUMNS, DataFetchError, SymbolInfo, VenueConstraints
from trading.venues.crypto import DEFAULT_UNIVERSE_CSV, CryptoAdapter

CONFIG = load_venue_config("crypto", Path("config"))


def _kraken_rows(n: int, end: datetime.date) -> list[list[float]]:
    """Mimic ccxt fetch_ohlcv: [ms_timestamp, open, high, low, close, volume] rows."""
    start_ts = pd.Timestamp(end, tz="UTC") - pd.Timedelta(days=n - 1)
    return [
        [
            int((start_ts + pd.Timedelta(days=i)).timestamp() * 1000),
            100.0 + i,
            101.0 + i,
            99.0 + i,
            100.5 + i,
            1000.0,
        ]
        for i in range(n)
    ]


def test_universe_reads_symbols_and_statuses(tmp_path):
    csv = tmp_path / "universe.csv"
    csv.write_text("symbol,status\nBTC,tradable\nETH,tradable\nSOL,sell_only\n")
    adapter = CryptoAdapter(CONFIG, universe_csv=csv)
    infos = adapter.universe(datetime.date(2026, 7, 1))
    assert infos == [
        SymbolInfo("BTC", "tradable"),
        SymbolInfo("ETH", "tradable"),
        SymbolInfo("SOL", "sell_only"),
    ]


def test_universe_rejects_unknown_status(tmp_path):
    csv = tmp_path / "universe.csv"
    csv.write_text("symbol,status\nBTC,halted\n")
    adapter = CryptoAdapter(CONFIG, universe_csv=csv)
    with pytest.raises(ValueError, match="halted"):
        adapter.universe(datetime.date(2026, 7, 1))


def test_committed_universe_csv_is_valid():
    adapter = CryptoAdapter(CONFIG, universe_csv=DEFAULT_UNIVERSE_CSV)
    infos = adapter.universe(datetime.date(2026, 7, 1))
    assert len(infos) >= 80
    assert SymbolInfo("BTC", "tradable") in infos


def test_constraints_come_from_config():
    adapter = CryptoAdapter(CONFIG)
    assert adapter.constraints() == VenueConstraints(
        taker_fee_bps=95.0,
        maker_fee_bps=50.0,
        slippage_bps=5.0,
        settlement_days=0,
        trades_24_7=True,
    )


def test_fetch_ohlcv_maps_symbol_to_kraken_usd_pair(monkeypatch):
    seen: list[str] = []

    def fake_fetch(pair: str, since_ms: int) -> list[list[float]]:
        seen.append(pair)
        return _kraken_rows(10, datetime.date(2026, 7, 1))

    monkeypatch.setattr("trading.venues.crypto._kraken_fetch", fake_fetch)
    adapter = CryptoAdapter(CONFIG)
    df = adapter.fetch_ohlcv("BTC", datetime.date(2026, 6, 22), datetime.date(2026, 7, 1))
    assert seen == ["BTC/USD"]
    assert list(df.columns) == OHLCV_COLUMNS
    assert str(df.index.tz) == "UTC"
    assert df.index.is_monotonic_increasing
    assert len(df) == 10
    assert df["close"].iloc[-1] == 109.5


def test_fetch_ohlcv_slices_to_requested_range(monkeypatch):
    monkeypatch.setattr(
        "trading.venues.crypto._kraken_fetch",
        lambda pair, since_ms: _kraken_rows(10, datetime.date(2026, 7, 1)),
    )
    adapter = CryptoAdapter(CONFIG)
    df = adapter.fetch_ohlcv("BTC", datetime.date(2026, 6, 25), datetime.date(2026, 6, 30))
    assert df.index.min() == pd.Timestamp("2026-06-25", tz="UTC")
    assert df.index.max() == pd.Timestamp("2026-06-30", tz="UTC")


def test_fetch_ohlcv_empty_raises(monkeypatch):
    monkeypatch.setattr("trading.venues.crypto._kraken_fetch", lambda pair, since_ms: [])
    adapter = CryptoAdapter(CONFIG)
    with pytest.raises(DataFetchError):
        adapter.fetch_ohlcv("BTC", datetime.date(2026, 6, 22), datetime.date(2026, 7, 1))


def test_fetch_ohlcv_paginates_over_page_limit(monkeypatch):
    """Ranges longer than one Kraken page are stitched from multiple fetches."""
    all_rows = _kraken_rows(10, datetime.date(2026, 7, 1))
    day_ms = 86_400_000
    calls: list[int] = []

    def fake_fetch(pair: str, since_ms: int) -> list[list[float]]:
        calls.append(since_ms)
        # 4-row pages, overlapping one day back like a real exchange might.
        return [r for r in all_rows if r[0] >= since_ms - day_ms][:4]

    monkeypatch.setattr("trading.venues.crypto._kraken_fetch", fake_fetch)
    adapter = CryptoAdapter(CONFIG)
    df = adapter.fetch_ohlcv("BTC", datetime.date(2026, 6, 22), datetime.date(2026, 7, 1))
    assert len(calls) > 1  # actually paginated
    assert len(df) == 10  # full range covered
    assert not df.index.duplicated().any()  # page overlap deduplicated
    assert df.index.min() == pd.Timestamp("2026-06-22", tz="UTC")
    assert df.index.max() == pd.Timestamp("2026-07-01", tz="UTC")


def test_fetch_ohlcv_wraps_ccxt_errors_as_data_fetch_error(monkeypatch):
    import ccxt

    def fake_fetch(pair: str, since_ms: int) -> list[list[float]]:
        raise ccxt.BadSymbol(f"kraken does not have market symbol {pair}")

    monkeypatch.setattr("trading.venues.crypto._kraken_fetch", fake_fetch)
    adapter = CryptoAdapter(CONFIG)
    with pytest.raises(DataFetchError, match="NOPE/USD"):
        adapter.fetch_ohlcv("NOPE", datetime.date(2026, 6, 22), datetime.date(2026, 7, 1))
