"""Trials-aware statistics: t -> p-value, BH-FDR gate, Deflated Sharpe Ratio.

This repo has no scipy (deliberately), so the Student-t two-sided p-value is
computed from the regularized incomplete beta function, implemented with
math.lgamma plus the standard continued fraction (Numerical Recipes 6.4,
modified Lentz's method). The normal CDF / inverse CDF needed by the DSR come
from stdlib statistics.NormalDist. Every function is verified against
published reference values in tests/test_alphasearch_stats.py.
"""

from __future__ import annotations

import math
from statistics import NormalDist

import numpy as np

_NORMAL = NormalDist()
_EULER_GAMMA = 0.5772156649015329  # Euler-Mascheroni, in the DSR E[max] term
_TINY = 1e-30
_MAX_ITER = 200
_CF_EPS = 1e-12


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the incomplete beta (modified Lentz's method)."""
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < _TINY:
        d = _TINY
    d = 1.0 / d
    h = d
    for m in range(1, _MAX_ITER + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < _TINY:
            d = _TINY
        c = 1.0 + aa / c
        if abs(c) < _TINY:
            c = _TINY
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < _TINY:
            d = _TINY
        c = 1.0 + aa / c
        if abs(c) < _TINY:
            c = _TINY
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < _CF_EPS:
            return h
    raise ArithmeticError("incomplete beta continued fraction did not converge")


def _betainc(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    ln_front = (
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
        + a * math.log(x) + b * math.log1p(-x)
    )
    front = math.exp(ln_front)
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def p_from_t(t: float, df: int) -> float:
    """Two-sided p-value of a Student-t statistic with df degrees of freedom.

    P(|T| >= |t|) = I_{df/(df+t^2)}(df/2, 1/2). NaN in -> NaN out (a journaled
    error trial has no t-stat; bh_fdr maps that NaN to p=1.0 itself).
    """
    if df <= 0:
        raise ValueError(f"df must be positive, got {df}")
    if math.isnan(t):
        return float("nan")
    return _betainc(df / 2.0, 0.5, df / (df + t * t))


def bh_fdr(pvals: np.ndarray | list[float], q: float = 0.10) -> np.ndarray:
    """Benjamini-Hochberg step-up: boolean pass mask aligned to input order.

    Reject H_(1)..H_(k) where k = max{i : p_(i) <= (i/n) * q}. NaN p-values
    (journaled error trials) are treated as p=1.0: they can never pass, but
    they DO count in n -- an error trial still spends a trial (spec section 6).
    """
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    if n == 0:
        return np.zeros(0, dtype=bool)
    p = np.where(np.isnan(p), 1.0, p)
    order = np.argsort(p, kind="stable")
    passed = p[order] <= q * np.arange(1, n + 1) / n
    mask = np.zeros(n, dtype=bool)
    if passed.any():
        k = int(np.nonzero(passed)[0].max())
        mask[order[: k + 1]] = True
    return mask


def deflated_sharpe(
    sr: float,
    n_obs: int,
    skew: float,
    kurt: float,
    n_trials: int,
    var_trials_sr: float,
) -> float:
    """Deflated Sharpe Ratio (Bailey & Lopez de Prado 2014). Advisory only.

    Probability that the candidate's true Sharpe exceeds zero after accounting
    for having selected the best of n_trials and for non-normal returns.
    ALL Sharpe inputs are PER-PERIOD (daily, non-annualized): `sr` is the
    candidate's daily Sharpe, `var_trials_sr` the cross-trial variance of the
    daily Sharpes. `kurt` is Pearson kurtosis (normal = 3).
    """
    if n_trials < 1 or n_obs < 2:
        return float("nan")
    if n_trials == 1:
        sr0 = 0.0  # inv_cdf(0) is -inf; a single trial has no selection bias
    else:
        sr0 = math.sqrt(max(var_trials_sr, 0.0)) * (
            (1 - _EULER_GAMMA) * _NORMAL.inv_cdf(1 - 1 / n_trials)
            + _EULER_GAMMA * _NORMAL.inv_cdf(1 - 1 / (n_trials * math.e))
        )
    denom = 1 - skew * sr + (kurt - 1) / 4.0 * sr * sr
    if denom <= 0:
        return float("nan")
    return _NORMAL.cdf((sr - sr0) * math.sqrt(n_obs - 1) / math.sqrt(denom))
