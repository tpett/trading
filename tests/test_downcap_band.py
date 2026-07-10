import math

import pandas as pd
import pytest

from trading.alphasearch.panel import BAR_COLUMNS
from trading.venues.universes import downcap_band as db


def _bars(n=80, close_raw=10.0, close=10.0, high=10.05, low=9.95, volume=100_000.0):
    idx = pd.date_range("2019-01-01", periods=n, freq="B", tz="UTC")
    frame = pd.DataFrame(
        {
            "open": close, "high": high, "low": low, "close": close,
            "volume": volume, "div_cash": 0.0, "split_factor": 1.0,
            "close_raw": close_raw,
        },
        index=idx,
    )
    return frame[BAR_COLUMNS]


def test_market_cap_uses_raw_price_hand_computed():
    # 20M shares * $10 raw = $200M
    assert db.market_cap_raw(20_000_000.0, 10.0) == pytest.approx(200_000_000.0)


def test_band_of_boundaries():
    assert db.band_of(49_999_999.0) is None          # below $50M
    assert db.band_of(50_000_000.0) == "micro"       # inclusive lower
    assert db.band_of(299_999_999.0) == "micro"
    assert db.band_of(300_000_000.0) == "small"      # micro/small boundary
    assert db.band_of(2_000_000_000.0) == "small"    # inclusive upper
    assert db.band_of(2_000_000_001.0) is None       # above $2B


def test_median_dollar_volume_hand_computed():
    bars = _bars(close=10.0, volume=100_000.0)  # 10*100k = 1_000_000 every day
    assert db.median_dollar_volume(bars) == pytest.approx(1_000_000.0)


def test_median_dollar_volume_discriminates_from_mean():
    # Pins median-not-mean: 62 sessions @ $1M/day dollar-volume + 1 session
    # @ $50M/day. Mean over the trailing-63 window is ~$1.79M, but the
    # median stays at $1M. If the implementation used `.mean()` instead of
    # `.median()`, this assertion would fail.
    bars = _bars(n=63, close=10.0, volume=100_000.0)  # $1M/day baseline
    bars = bars.copy()
    bars.iloc[-1, bars.columns.get_loc("volume")] = 5_000_000.0  # $50M spike day
    mean_dv = float((bars["close"] * bars["volume"]).mean())
    assert mean_dv != pytest.approx(1_000_000.0)
    assert db.median_dollar_volume(bars) == pytest.approx(1_000_000.0)


def test_evaluate_band_in_band_micro():
    bars = _bars(close_raw=10.0, close=10.0, volume=100_000.0)
    ev = db.evaluate_band(bars, shares=20_000_000.0)  # $200M -> micro
    assert ev.band == "micro"
    assert ev.has_shares is True and ev.tradeable is True
    assert ev.market_cap == pytest.approx(200_000_000.0)


def test_missing_shares_excluded_fail_closed():
    bars = _bars()
    ev = db.evaluate_band(bars, shares=math.nan)
    assert ev.has_shares is False
    assert ev.band is None                 # never guess a cap
    assert math.isnan(ev.market_cap)
    assert ev.tradeable is True            # still measured for the audit denom


def test_spread_screen_rejects_wide_names():
    # Very wide high/low band -> CS spread well above 2%.
    bars = _bars(high=12.0, low=8.0)
    ev = db.evaluate_band(bars, shares=20_000_000.0)
    assert ev.tradeable is False
    assert ev.band is None
    assert ev.spread > db.SPREAD_CAP_PCT


def test_depth_screen_rejects_thin_names():
    bars = _bars(close=10.0, volume=1_000.0)  # 10*1000 = 10k < $50k
    ev = db.evaluate_band(bars, shares=20_000_000.0)
    assert ev.tradeable is False
    assert ev.band is None
    assert ev.dollar_volume < db.DV_FLOOR


def test_cap_ignores_future_split_lookahead_guard():
    """A future split retro-adjusts the ADJUSTED close but never close_raw.
    The cap MUST read close_raw, so a name's as-of-D cap is identical whether
    or not a later split has been applied to the adjusted column."""
    pre = _bars(close_raw=10.0, close=10.0, volume=100_000.0)
    # Simulate a vendor 2:1 retro-adjustment of the ADJUSTED close only.
    post = pre.copy()
    post["close"] = post["close"] / 2.0        # adjusted halved
    post["volume"] = post["volume"] * 2.0      # adjusted volume doubled
    ev_pre = db.evaluate_band(pre, shares=20_000_000.0)
    ev_post = db.evaluate_band(post, shares=20_000_000.0)
    assert ev_pre.market_cap == ev_post.market_cap == pytest.approx(200_000_000.0)
    # Sanity: a cap computed on the adjusted close WOULD have leaked the split.
    assert (20_000_000.0 * float(post["close"].iloc[-1])) != pytest.approx(200_000_000.0)


def test_evaluate_band_empty_bars_fail_closed():
    """A name with PIT shares_outstanding but no cached bars (e.g. PanelView
    returns an empty frame) must fail closed to NaN/None, never raise."""
    empty = _bars(n=0)
    ev = db.evaluate_band(empty, shares=20_000_000.0)
    assert ev.band is None
    assert ev.has_shares is True
    assert math.isnan(ev.market_cap)


def test_evaluate_band_nonpositive_close_raw_fail_closed():
    bars = _bars(close_raw=0.0, close=10.0, volume=100_000.0)
    ev = db.evaluate_band(bars, shares=20_000_000.0)
    assert ev.band is None
    assert math.isnan(ev.market_cap)
