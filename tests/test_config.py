import dataclasses
import datetime
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
    assert config.signals.momentum_windows == (63, 126, 252)
    assert config.signals.calendar_days is False
    assert config.signals.breakout_windows == (20, 60)
    assert config.regime.sma_slow == 200
    assert config.portfolio.max_positions == 20
    assert config.data.min_coverage == 0.90
    assert config.data.max_daily_move == 0.40
    assert config.portfolio.atr_window == 20
    # Kill switch flipped off: yfinance earnings dates proved unreliable
    # in practice (see README's "Earnings blackout" section).
    assert config.portfolio.earnings_blackout_enabled is False


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
    assert config.portfolio.earnings_blackout_enabled is False


def test_config_is_frozen():
    config = load_venue_config("equities", CONFIG_DIR)
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.name = "other"
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.signals.vol_window = 99


def test_missing_config_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_venue_config("equities", tmp_path)


def test_backfill_config_loaded():
    crypto = load_venue_config("crypto", Path("config"))
    assert crypto.data.backfill_exchange != ""
    assert crypto.data.backfill_page_limit > 0
    assert crypto.data.seam_max_gap_days > 0
    equities = load_venue_config("equities", Path("config"))
    assert equities.data.backfill_exchange == ""


def test_exit_style_loaded_as_frozen_by_default_in_real_configs():
    for venue in ("equities", "crypto"):
        config = load_venue_config(venue, Path("config"))
        assert config.portfolio.exit_style == "frozen"


def test_invalid_exit_style_raises(tmp_path):
    raw = (Path("config") / "equities.toml").read_text()
    bad = raw.replace('exit_style = "frozen"', 'exit_style = "yolo"')
    assert bad != raw
    (tmp_path / "equities.toml").write_text(bad)
    with pytest.raises(ValueError, match="exit_style"):
        load_venue_config("equities", tmp_path)


def test_fundamentals_data_keys_load_for_both_venues():
    eq = load_venue_config("equities", Path("config"))
    assert eq.data.fundamentals_dir == "data/fundamentals/equities"
    assert eq.data.fundamentals_refresh_days == 7
    assert eq.data.fundamentals_refresh_budget_s == 900
    cr = load_venue_config("crypto", Path("config"))
    assert cr.data.fundamentals_dir == ""  # no fundamentals concept for crypto
    assert cr.data.fundamentals_refresh_days == 0
    assert cr.data.fundamentals_refresh_budget_s == 0


def test_ranker_loaded_as_momentum_v1_by_default_in_real_configs():
    for venue in ("equities", "crypto"):
        config = load_venue_config(venue, Path("config"))
        assert config.signals.ranker == "momentum_v1"


def test_unknown_ranker_raises_at_load_time(tmp_path):
    raw = (Path("config") / "equities.toml").read_text()
    bad = raw.replace('ranker = "momentum_v1"', 'ranker = "bogus_ranker"')
    assert bad != raw
    (tmp_path / "equities.toml").write_text(bad)
    # The message must name the bad value AND enumerate the known rankers.
    with pytest.raises(ValueError, match="bogus_ranker") as excinfo:
        load_venue_config("equities", tmp_path)
    assert "momentum_v1" in str(excinfo.value)


def test_backtest_config_loaded():
    for venue in ("equities", "crypto"):
        config = load_venue_config(venue, Path("config"))
        bt = config.backtest
        assert bt.start == datetime.date(2018, 1, 1)
        assert bt.holdout_start == datetime.date(2026, 1, 5)
        assert 24 <= bt.train_months <= 36 and bt.test_months == 3  # spec bounds
        assert len(bt.entry_score_threshold_grid) >= 3
        assert len(bt.stop_atr_multiple_grid) >= 3
        assert 0 < bt.min_session_coverage <= 1
        assert bt.stress_segments[0] == (datetime.date(2022, 1, 1), datetime.date(2022, 12, 31))
    assert load_venue_config("equities", Path("config")).backtest.periods_per_year == 252
    assert load_venue_config("crypto", Path("config")).backtest.periods_per_year == 365


def test_universe_indices_default_to_live_sp500_and_ndx():
    # Live/paper invariant: sp400 is backtest opt-in, never a default-config
    # change (equities_membership.csv now also carries sp400 rows).
    for venue in ("equities", "crypto"):
        config = load_venue_config(venue, Path("config"))
        assert config.universe.indices == ("sp500", "ndx")


def test_universe_indices_overridable_via_config(tmp_path):
    raw = (Path("config") / "equities.toml").read_text()
    patched = raw.replace(
        "point_in_time = true", 'point_in_time = true\nindices = ["sp500", "ndx", "sp400"]'
    )
    assert patched != raw
    (tmp_path / "equities.toml").write_text(patched)
    config = load_venue_config("equities", tmp_path)
    assert config.universe.indices == ("sp500", "ndx", "sp400")


def test_membership_exit_buffer_days_defaults_to_30_days():
    for venue in ("equities", "crypto"):
        config = load_venue_config(venue, Path("config"))
        assert config.backtest.membership_exit_buffer_days == 30


def test_fundamentals_requiring_ranker_with_empty_dir_fails_at_load(tmp_path):
    text = (Path("config") / "equities.toml").read_text()
    text = text.replace('ranker = "momentum_v1"', 'ranker = "quality_momentum_v1"')
    text = text.replace('fundamentals_dir = "data/fundamentals/equities"', 'fundamentals_dir = ""')
    (tmp_path / "equities.toml").write_text(text)
    with pytest.raises(ValueError, match="fundamentals_dir"):
        load_venue_config("equities", tmp_path)


def test_fundamentals_requiring_ranker_with_zero_refresh_days_fails_at_load(tmp_path):
    text = (Path("config") / "experiments" / "quality" / "equities.toml").read_text()
    text = text.replace("fundamentals_refresh_days = 7", "fundamentals_refresh_days = 0")
    assert "fundamentals_refresh_days = 0" in text
    (tmp_path / "equities.toml").write_text(text)
    with pytest.raises(ValueError, match="fundamentals_refresh_days"):
        load_venue_config("equities", tmp_path)


def test_fundamentals_requiring_ranker_with_zero_refresh_budget_fails_at_load(tmp_path):
    text = (Path("config") / "experiments" / "quality" / "equities.toml").read_text()
    text = text.replace("fundamentals_refresh_budget_s = 900", "fundamentals_refresh_budget_s = 0")
    assert "fundamentals_refresh_budget_s = 0" in text
    (tmp_path / "equities.toml").write_text(text)
    with pytest.raises(ValueError, match="fundamentals_refresh_budget_s"):
        load_venue_config("equities", tmp_path)


def test_fundamentals_requiring_ranker_omitting_refresh_budget_loads_with_default(tmp_path):
    # A fundamentals-requiring TOML that legitimately OMITS
    # fundamentals_refresh_budget_s must still load -- it gets the
    # DataConfig field default (900s), not the raw-dict-get sentinel of 0
    # the validation used to (wrongly) check against.
    text = (Path("config") / "experiments" / "quality" / "equities.toml").read_text()
    lines = text.splitlines(keepends=True)
    text = "".join(line for line in lines if "fundamentals_refresh_budget_s" not in line)
    assert "fundamentals_refresh_budget_s" not in text
    (tmp_path / "equities.toml").write_text(text)
    config = load_venue_config("equities", tmp_path)
    assert config.data.fundamentals_refresh_budget_s == 900


def test_quality_experiment_config_loads():
    config = load_venue_config("equities", Path("config") / "experiments" / "quality")
    assert config.signals.ranker == "quality_momentum_v1"
    assert config.data.fundamentals_dir == "data/fundamentals/equities"


def test_value_experiment_config_loads():
    config = load_venue_config("equities", Path("config") / "experiments" / "value")
    assert config.signals.ranker == "value_momentum_v1"
    assert config.data.fundamentals_dir == "data/fundamentals/equities"
