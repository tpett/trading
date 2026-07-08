"""Unit tests for the trials-aware statistics (worked reference values)."""

from __future__ import annotations

import math

import numpy as np

from trading.alphasearch.stats import bh_fdr, deflated_sharpe, p_from_t


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
