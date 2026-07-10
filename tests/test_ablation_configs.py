"""R2 spec §1: the 15-cell ablation matrix (3 signals x 5 cells) must load
and match the pre-registered wrapper settings per cell. See
docs/superpowers/specs/2026-07-10-wrapper-ablation-design.md.
"""

from pathlib import Path

import pytest

from trading.config import load_venue_config

CONFIG_ROOT = Path("config/experiments")

SIGNALS = {
    "momentum": "momentum_v1",
    "amihud": "amihud_v1",
    "skew": "skew_v1",
}

# (regime_disabled, bare_mode, entry_score_threshold, single_point_grids)
# per the spec table: W0 bare, W1 regime-only, W2 stops-only, W3
# threshold-only, W4 full/control (must equal the historical values).
EXPECTED = {
    "w0": dict(regime_disabled=True, bare_mode=True, threshold=0.0, single_point_grids=True),
    "w1": dict(regime_disabled=False, bare_mode=False, threshold=0.0, single_point_grids=True),
    "w2": dict(regime_disabled=True, bare_mode=False, threshold=0.0, single_point_grids=True),
    "w3": dict(regime_disabled=True, bare_mode=False, threshold=0.70, single_point_grids=False),
    "w4": dict(regime_disabled=False, bare_mode=False, threshold=0.70, single_point_grids=False),
}


@pytest.mark.parametrize("signal,ranker", sorted(SIGNALS.items()))
@pytest.mark.parametrize("cell", sorted(EXPECTED))
def test_ablation_cell_loads_with_the_pre_registered_wrapper_settings(signal, ranker, cell):
    config = load_venue_config("equities", CONFIG_ROOT / f"ablation-{signal}-{cell}")
    expected = EXPECTED[cell]
    assert config.signals.ranker == ranker
    assert config.regime.disabled is expected["regime_disabled"]
    assert config.portfolio.bare_mode is expected["bare_mode"]
    assert config.portfolio.entry_score_threshold == pytest.approx(expected["threshold"])
    if expected["single_point_grids"]:
        assert len(config.backtest.entry_score_threshold_grid) == 1
        assert len(config.backtest.stop_atr_multiple_grid) == 1
    else:
        assert len(config.backtest.entry_score_threshold_grid) > 1


@pytest.mark.parametrize("signal", sorted(SIGNALS))
def test_w0_bare_disables_stops_via_bare_mode_not_via_the_grid(signal):
    # W0's stop/threshold values are inert under bare_mode (trading.simulator.bare
    # never reads them) -- the flag itself is what does the work; this just
    # pins that both are set for a reader auditing the journaled config.
    config = load_venue_config("equities", CONFIG_ROOT / f"ablation-{signal}-w0")
    assert config.portfolio.bare_mode is True
    assert config.regime.disabled is True


@pytest.mark.parametrize("signal", sorted(SIGNALS))
def test_w4_control_reproduces_the_historical_wrapper_values(signal):
    config = load_venue_config("equities", CONFIG_ROOT / f"ablation-{signal}-w4")
    assert config.regime.disabled is False
    assert config.portfolio.bare_mode is False
    assert config.portfolio.stop_atr_multiple == pytest.approx(1.5)
    assert config.portfolio.regime_flush_atr_multiple == pytest.approx(1.0)
    assert config.portfolio.drawdown_halt_pct == pytest.approx(0.20)
    assert config.backtest.entry_score_threshold_grid == (0.60, 0.65, 0.70, 0.75, 0.80)
    assert config.backtest.stop_atr_multiple_grid == (1.0, 1.5, 2.0, 2.5)


@pytest.mark.parametrize("signal", sorted(SIGNALS))
def test_only_wrapper_flags_differ_from_the_source_experiment_config(signal):
    """Everything except [regime].disabled, [portfolio].bare_mode/
    entry_score_threshold/stop_atr_multiple/regime_flush_atr_multiple/
    drawdown_halt_pct, and [backtest]'s two grids must be identical to the
    historical source config across all 5 cells (spec §3: 'changing ONLY
    wrapper flags')."""
    source_by_signal = {
        "momentum": CONFIG_ROOT / "tiingo",
        "amihud": CONFIG_ROOT / "amihud-midcap",
        "skew": CONFIG_ROOT / "options-skew",
    }
    source = load_venue_config("equities", source_by_signal[signal])
    for cell in EXPECTED:
        config = load_venue_config("equities", CONFIG_ROOT / f"ablation-{signal}-{cell}")
        assert config.name == source.name
        assert config.benchmark == source.benchmark
        assert config.costs == source.costs
        assert config.universe == source.universe
        assert config.signals == source.signals
        assert config.data == source.data
        assert config.backtest.start == source.backtest.start
        assert config.backtest.holdout_start == source.backtest.holdout_start
        assert config.backtest.train_months == source.backtest.train_months
        assert config.backtest.test_months == source.backtest.test_months
        assert config.backtest.min_session_coverage == source.backtest.min_session_coverage
        assert config.backtest.periods_per_year == source.backtest.periods_per_year
        assert config.backtest.stress_segments == source.backtest.stress_segments
        # [portfolio] fields NOT part of the ablation matrix must be untouched.
        assert config.portfolio.max_positions == source.portfolio.max_positions
        assert config.portfolio.position_size_pct == source.portfolio.position_size_pct
        assert config.portfolio.time_stop_bars == source.portfolio.time_stop_bars
        assert config.portfolio.cooldown_days == source.portfolio.cooldown_days
        assert (
            config.portfolio.max_daily_deployment_pct
            == source.portfolio.max_daily_deployment_pct
        )
        assert config.portfolio.exit_style == source.portfolio.exit_style
        assert config.regime.sma_fast == source.regime.sma_fast
        assert config.regime.sma_slow == source.regime.sma_slow
        assert config.regime.exposure_risk_on == source.regime.exposure_risk_on
