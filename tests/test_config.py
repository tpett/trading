import dataclasses
from pathlib import Path

import pytest

from trading.config import load_venue_config

CONFIG_DIR = Path("config")


def test_load_equities_config():
    config = load_venue_config("equities", CONFIG_DIR)
    assert config.name == "equities"
    assert config.benchmark == "SPY"
    assert config.costs.slippage_bps == 5.0
    assert config.costs.settlement_days == 1
    assert config.costs.trades_24_7 is False
    assert config.signals.momentum_windows == (5, 20, 60)
    assert config.signals.calendar_days is False
    assert config.signals.breakout_windows == (20, 60)
    assert config.regime.sma_slow == 200
    assert config.portfolio.max_positions == 5
    assert config.data.min_coverage == 0.90
    assert config.data.max_daily_move == 0.40


def test_load_crypto_config():
    config = load_venue_config("crypto", CONFIG_DIR)
    assert config.benchmark == "BTC"
    assert config.costs.taker_fee_bps == 95.0
    assert config.costs.maker_fee_bps == 50.0
    assert config.costs.trades_24_7 is True
    assert config.signals.momentum_windows == (7, 30, 90)
    assert config.signals.calendar_days is True
    assert config.portfolio.max_positions == 3
    assert config.portfolio.min_raw_return_cost_multiple == 3.0


def test_config_is_frozen():
    config = load_venue_config("equities", CONFIG_DIR)
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.name = "other"
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.signals.vol_window = 99


def test_missing_config_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_venue_config("equities", tmp_path)
