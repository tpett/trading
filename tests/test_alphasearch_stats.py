"""Unit tests for the trials-aware statistics (worked reference values)."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from trading.alphasearch.stats import (
    bh_fdr,
    closed_form_sharpe_se,
    deflated_sharpe,
    p_from_t,
    sharpe_ci,
)


# --------------------------------------------------------------------------- #
# p_from_t: checked against published t-table critical values
# --------------------------------------------------------------------------- #
def test_p_from_t_matches_critical_values():
    # Two-sided 5% critical values: t(10)=2.228139, t(60)=2.000298, t(1)=12.7062.
    assert math.isclose(p_from_t(2.228139, 10), 0.05, abs_tol=1e-5)
    assert math.isclose(p_from_t(2.000298, 60), 0.05, abs_tol=1e-5)
    assert math.isclose(p_from_t(12.7062, 1), 0.05, abs_tol=1e-5)
    # Interior value: P(|T_10| >= 1.0) = 0.34089.
    assert math.isclose(p_from_t(1.0, 10), 0.34089, abs_tol=1e-4)
    # Large df converges to the normal: z=1.959964 -> 0.05.
    assert math.isclose(p_from_t(1.959964, 10**6), 0.05, abs_tol=1e-4)


def test_p_from_t_edges():
    assert p_from_t(0.0, 5) == 1.0
    assert p_from_t(-2.5, 30) == p_from_t(2.5, 30)  # two-sided symmetry
    assert math.isnan(p_from_t(float("nan"), 10))  # error trials carry no t


def test_p_from_t_rejects_bad_df():
    import pytest

    with pytest.raises(ValueError):
        p_from_t(1.0, 0)


# --------------------------------------------------------------------------- #
# bh_fdr: canonical Benjamini-Hochberg (1995) worked example, 15 p-values
# --------------------------------------------------------------------------- #
BH_PVALS = [0.0001, 0.0004, 0.0019, 0.0095, 0.0201, 0.0278, 0.0298, 0.0344,
            0.0459, 0.3240, 0.4262, 0.5719, 0.6528, 0.7590, 1.0000]


def test_bh_canonical_example_q05_rejects_four():
    mask = bh_fdr(BH_PVALS, q=0.05)
    # k = max{i: p_(i) <= (i/15)*0.05} = 4 -> the four smallest are rejected.
    assert mask.sum() == 4
    assert list(np.nonzero(mask)[0]) == [0, 1, 2, 3]


def test_bh_canonical_example_q10_rejects_nine():
    # At the pre-registered q=0.10: p_(9)=0.0459 <= 9/15*0.10=0.06 -> k=9.
    assert bh_fdr(BH_PVALS, q=0.10).sum() == 9


def test_bh_mask_alignment_is_input_order():
    # Shuffled input: the mask marks the same VALUES regardless of position.
    p = [0.5719, 0.0001, 0.3240, 0.0004]
    mask = bh_fdr(p, q=0.05)
    assert list(mask) == [False, True, False, True]


def test_bh_nan_counts_as_a_trial_but_never_passes():
    # Two borderline p-values pass alone...
    assert bh_fdr([0.05, 0.09], q=0.10).sum() == 2
    # ...but adding two error trials (NaN p) raises the bar and kills both:
    # honest trial accounting means a failed trial still spends a trial.
    mask = bh_fdr([0.05, 0.09, float("nan"), float("nan")], q=0.10)
    assert mask.sum() == 0
    assert len(mask) == 4


def test_bh_empty_input():
    assert bh_fdr([], q=0.10).sum() == 0


# --------------------------------------------------------------------------- #
# deflated_sharpe: Bailey & Lopez de Prado (2014) worked example
# --------------------------------------------------------------------------- #
def test_dsr_paper_reference_value():
    # The paper's example: annualized SR 2.5 over 1250 daily obs, skew -3,
    # kurtosis 10, best of N=100 trials with cross-trial variance of the
    # ANNUALIZED SR = 0.5. In daily units: sr=2.5/sqrt(250), var=0.5/250.
    # Published result: DSR ~= 0.9004.
    dsr = deflated_sharpe(
        sr=2.5 / math.sqrt(250), n_obs=1250, skew=-3.0, kurt=10.0,
        n_trials=100, var_trials_sr=0.5 / 250,
    )
    assert math.isclose(dsr, 0.9004, abs_tol=1e-3)


def test_dsr_single_trial_has_no_deflation():
    # n_trials=1 -> SR0=0 -> plain probabilistic Sharpe ratio; positive SR > 0.5.
    dsr = deflated_sharpe(sr=0.1, n_obs=252, skew=0.0, kurt=3.0,
                          n_trials=1, var_trials_sr=0.0)
    assert 0.5 < dsr < 1.0


def test_dsr_more_trials_deflate_harder():
    kw = dict(sr=0.1, n_obs=252, skew=0.0, kurt=3.0, var_trials_sr=0.002)
    assert deflated_sharpe(n_trials=100, **kw) < deflated_sharpe(n_trials=10, **kw)


def test_dsr_degenerate_inputs_are_nan():
    assert math.isnan(deflated_sharpe(0.1, 1, 0.0, 3.0, 10, 0.001))  # n_obs < 2
    assert math.isnan(deflated_sharpe(0.1, 252, 0.0, 3.0, 0, 0.001))  # no trials


# --------------------------------------------------------------------------- #
# sharpe_ci: stationary bootstrap CI, cross-checked against the closed-form SE
# (R6 Stage 1 market-neutral gate amendment spec section 3)
# --------------------------------------------------------------------------- #
def _iid_normal_series(n=2520, mean=0.0006, sd=0.01, seed=42) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(mean, sd, size=n))


def test_sharpe_ci_agrees_with_closed_form_se_on_iid_normal():
    # A synthetic iid-normal series (no serial correlation, so the block
    # bootstrap and the closed-form iid-normal SE should roughly agree): the
    # bootstrap's central-95% half-width should be in the same ballpark as
    # 1.96 * the closed-form SE.
    r = _iid_normal_series()
    point, lo, hi = sharpe_ci(r, seed=7, n_boot=2000, block=10)
    assert not math.isnan(point)
    se = closed_form_sharpe_se(point, len(r))
    boot_half_width = (hi - lo) / 2.0
    closed_form_half_width = 1.96 * se
    assert boot_half_width == pytest.approx(closed_form_half_width, rel=0.10)
    # The point estimate sits inside its own CI.
    assert lo < point < hi


def test_sharpe_ci_is_deterministic_given_a_seed():
    r = _iid_normal_series(seed=11)
    first = sharpe_ci(r, seed=123)
    second = sharpe_ci(r, seed=123)
    assert first == second


def test_sharpe_ci_different_seeds_can_differ():
    r = _iid_normal_series(seed=11)
    a = sharpe_ci(r, seed=1)
    b = sharpe_ci(r, seed=2)
    assert a != b


def test_sharpe_ci_near_zero_mean_series_straddles_zero():
    # The whole point of the gate (spec section 3): a near-zero-mean series
    # (no real edge) must yield a CI that straddles 0, so a naive positive
    # point Sharpe would NOT clear the "CI lower bound > 0" gate.
    r = _iid_normal_series(mean=0.0, sd=0.01, seed=5)
    _point, lo, hi = sharpe_ci(r, seed=3)
    assert lo < 0 < hi


def test_sharpe_ci_nan_below_two_observations():
    point, lo, hi = sharpe_ci(pd.Series([0.01]))
    assert math.isnan(lo) and math.isnan(hi)


def test_closed_form_sharpe_se_matches_the_frozen_formula():
    # Lo (2002): sqrt((1 + 0.5*SR^2) / T) requires SR and T in the SAME
    # frequency. `sharpe` here is ANNUALIZED and `n_obs` gives T in years,
    # so the SR^2 term must be de-annualized (divided by TRADING_DAYS=252)
    # before combining with T in years -- computed independently here from
    # the daily-frequency form, not copied from the implementation:
    #   daily SR = SR_ann / sqrt(252); T_days = n_obs
    #   SE_daily = sqrt((1 + 0.5*sr_daily^2) / n_obs)
    #   SE_ann = SE_daily * sqrt(252)   (SE scales with sqrt(frequency))
    # sr=1.2, n_obs=1260 (5 years at 252 sessions/yr):
    #   sr_daily = 1.2 / sqrt(252) = 0.07559289...
    #   SE_daily = sqrt((1 + 0.5*0.07559289^2) / 1260) = sqrt(1.0028571/1260)
    #            = 0.02821203...
    #   SE_ann = 0.02821203 * sqrt(252) = 0.44785202...
    sr, n_obs = 1.2, 1260  # 5 years at 252 sessions/yr
    sr_daily = sr / math.sqrt(252)
    se_daily = math.sqrt((1.0 + 0.5 * sr_daily * sr_daily) / n_obs)
    expected = se_daily * math.sqrt(252)
    assert expected == pytest.approx(0.44785201637530736)
    assert closed_form_sharpe_se(sr, n_obs) == pytest.approx(expected)


def test_closed_form_sharpe_se_high_sharpe_discriminates_old_bug():
    # At high annualized Sharpe the old (buggy) formula -- which mixed
    # annualized SR with T in years without de-annualizing SR^2 -- was
    # ~72% too wide. sr=2.0, T=4 years (n_obs=1008), independently derived
    # the same way as the case above:
    #   sr_daily = 2.0 / sqrt(252) = 0.12598816...
    #   SE_daily = sqrt((1 + 0.5*0.12598816^2) / 1008) = sqrt(1.0079365/1008)
    #            = 0.03162178...
    #   SE_ann = 0.03162178 * sqrt(252) = 0.50198021...
    # The buggy formula sqrt((1+0.5*4)/4) = sqrt(0.75) = 0.86603 would FAIL
    # this assertion (0.86603 vs ~0.502, a ratio of 1.725x).
    sr, n_obs = 2.0, 1008  # 4 years at 252 sessions/yr
    sr_daily = sr / math.sqrt(252)
    se_daily = math.sqrt((1.0 + 0.5 * sr_daily * sr_daily) / n_obs)
    expected = se_daily * math.sqrt(252)
    assert expected == pytest.approx(0.5019802057692384)
    assert closed_form_sharpe_se(sr, n_obs) == pytest.approx(expected)


def test_closed_form_sharpe_se_nan_inputs():
    assert math.isnan(closed_form_sharpe_se(float("nan"), 252))
    assert math.isnan(closed_form_sharpe_se(1.0, 0))
