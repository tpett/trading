import dataclasses

import pandas as pd
import pytest

from trading.venues.base import OHLCV_COLUMNS, SymbolInfo, VenueConstraints, validate_ohlcv


def _good_frame() -> pd.DataFrame:
    idx = pd.date_range("2026-01-05", periods=3, freq="B", tz="UTC")
    return pd.DataFrame(
        {"open": 1.0, "high": 2.0, "low": 0.5, "close": 1.5, "volume": 100.0}, index=idx
    )


def test_symbol_info_is_frozen():
    info = SymbolInfo(symbol="AAPL", status="tradable")
    with pytest.raises(dataclasses.FrozenInstanceError):
        info.status = "sell_only"


def test_venue_constraints_is_frozen():
    c = VenueConstraints(
        taker_fee_bps=95.0,
        maker_fee_bps=50.0,
        slippage_bps=5.0,
        settlement_days=0,
        trades_24_7=True,
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        c.taker_fee_bps = 0.0


def test_validate_ohlcv_accepts_contract_frame():
    df = _good_frame()
    assert validate_ohlcv(df) is df


def test_validate_ohlcv_rejects_wrong_columns():
    df = _good_frame().rename(columns={"close": "Close"})
    with pytest.raises(ValueError, match="columns"):
        validate_ohlcv(df)


def test_validate_ohlcv_rejects_naive_index():
    df = _good_frame()
    df.index = df.index.tz_localize(None)
    with pytest.raises(ValueError, match="UTC"):
        validate_ohlcv(df)


def test_validate_ohlcv_rejects_unsorted_index():
    df = _good_frame().iloc[::-1]
    with pytest.raises(ValueError, match="sorted"):
        validate_ohlcv(df)


def test_ohlcv_columns_locked():
    assert OHLCV_COLUMNS == ["open", "high", "low", "close", "volume"]
