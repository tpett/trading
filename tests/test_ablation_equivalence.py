"""R2 spec, §3: 'tests proving W4 ≡ the unmodified engine bit-for-bit on a
fixture.' This is the golden comparison. The expected literals below were
captured by running this exact fixture through trading.backtest.engine before
any R2 ablation code (RegimeConfig.disabled, PortfolioConfig.bare_mode,
trading.simulator.bare, the core.step branching) existed. Every new flag
defaults to False, and every non-bare code path in core.step is left
byte-for-byte unedited (only wrapped in an `if bare: ... else: <unchanged>`),
so this pins that the ablation build introduced zero behavior change for any
config that doesn't opt in.
"""

import datetime
from dataclasses import replace

from backtest_helpers import FakeBacktestAdapter, noisy_frame, small_config
from trading.backtest.engine import prepare, replay
from trading.data.cache import OhlcvCache

START = datetime.date(2025, 2, 1)
END = datetime.date(2025, 4, 30)

EXPECTED_TRADES = [
    ("AAA", "trend_break", 16.87043831, 134.22345867, 144.4185867),
    ("BBB", "stop_loss", -18.64770115, 112.80037558, 108.00277212),
    ("CCC", "trend_break", -4.17051033, 98.76588032, 99.27667477),
    ("AAA", "trend_break", 42.78688395, 146.83083857, 170.58516248),
    ("BBB", "trend_break", -12.11745599, 125.82513419, 123.27000629),
    ("DDD", "stop_loss", -21.1499604, 77.71655836, 73.8903639),
    ("BBB", "trend_break", -13.28997792, 121.65465213, 118.61300069),
    ("AAA", "trend_break", 23.13166532, 178.75331673, 196.00686826),
    ("BBB", "trend_break", -10.11461563, 124.10911166, 122.35067334),
    ("CCC", "trend_break", -5.98545754, 89.77450473, 89.72614172),
    ("CCC", "trend_break", -14.87221332, 90.80019765, 88.06003553),
    ("CCC", "trend_break", -8.05414762, 87.09676236, 86.45068232),
    ("BBB", "trend_break", -6.01339505, 121.11606664, 120.99726617),
    ("AAA", "trend_break", 64.80479434, 201.63304365, 248.93546151),
    ("BBB", "trend_break", -1.17168436, 126.53261404, 128.48221067),
    ("BBB", "trend_break", -18.04643462, 129.77883003, 124.80439631),
    ("CCC", "trend_break", 10.33013117, 86.94561047, 91.54352191),
]
EXPECTED_OPEN_POSITIONS = ("AAA",)
EXPECTED_FINAL_EQUITY = 1053.11997051
EXPECTED_EQUITY_SUM = 91108.516594
EXPECTED_FEES_PAID = 102.99808871
EXPECTED_BUY_NOTIONAL = 5514.54318846
EXPECTED_SESSIONS_RUN = 89
EXPECTED_WARNINGS = (
    "crypto universe is today's Robinhood listing: coins delisted before today are "
    "absent (survivorship bias); listing dates are inferred from data availability",
)


def _fixture_frames():
    return {
        "AAA": noisy_frame(seed=1, drift=0.01),
        "BBB": noisy_frame(seed=2, drift=0.002),
        "CCC": noisy_frame(seed=3, drift=0.0),
        "DDD": noisy_frame(seed=4, drift=-0.003),
        "BENCH": noisy_frame(seed=9, drift=0.003),
    }


def _replay_default(tmp_path, config):
    adapter = FakeBacktestAdapter(_fixture_frames(), "BENCH", None)
    cache = OhlcvCache(tmp_path / "cache", config.data.refetch_days)
    prepared = prepare(config, adapter, cache, START, END)
    return replay(prepared, config)


def test_w4_defaults_reproduce_the_pre_ablation_engine_bit_for_bit(tmp_path):
    config = replace(small_config(), benchmark="BENCH")
    assert config.regime.disabled is False  # defaults, unopted-in
    assert config.portfolio.bare_mode is False
    result = _replay_default(tmp_path, config)

    actual_trades = [
        (
            t.symbol,
            t.reason,
            round(t.realized_pnl, 8),
            round(t.entry_price, 8),
            round(t.exit_price, 8),
        )
        for t in result.trades
    ]
    assert actual_trades == EXPECTED_TRADES
    assert result.open_positions == EXPECTED_OPEN_POSITIONS
    assert round(float(result.equity_curve.iloc[-1]), 8) == EXPECTED_FINAL_EQUITY
    assert round(float(result.equity_curve.sum()), 6) == EXPECTED_EQUITY_SUM
    assert round(result.fees_paid, 8) == EXPECTED_FEES_PAID
    assert round(result.buy_notional, 8) == EXPECTED_BUY_NOTIONAL
    assert result.sessions_run == EXPECTED_SESSIONS_RUN
    assert tuple(sorted(result.warnings)) == EXPECTED_WARNINGS


def test_explicitly_setting_new_flags_to_their_defaults_is_still_identical(tmp_path):
    # Belt-and-suspenders: explicitly constructing the "off" values (rather
    # than relying on the dataclass default) must produce the identical
    # result -- the flags are genuinely inert at their default, not just
    # absent from old TOML files.
    base = replace(small_config(), benchmark="BENCH")
    config = replace(
        base,
        regime=replace(base.regime, disabled=False),
        portfolio=replace(base.portfolio, bare_mode=False),
    )
    result = _replay_default(tmp_path, config)
    actual_trades = [
        (
            t.symbol,
            t.reason,
            round(t.realized_pnl, 8),
            round(t.entry_price, 8),
            round(t.exit_price, 8),
        )
        for t in result.trades
    ]
    assert actual_trades == EXPECTED_TRADES
    assert round(float(result.equity_curve.iloc[-1]), 8) == EXPECTED_FINAL_EQUITY
