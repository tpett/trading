"""Factor regression -> alpha: the reusable core behind scripts/factor_regression.py.

ALPHA is the piece of a strategy's return that known, freely-buyable factor
exposures do NOT explain -- the intercept of an OLS regression of the daily
return series on the canonical Ken French daily factors (Mkt-RF, SMB, HML,
Mom, with RF for excess returns). A positive, statistically significant alpha
is the only trustworthy "real edge" measure; Sharpe-vs-benchmark is not
(docs/experiments.md section 9). This module always fits BOTH the four-factor
model and market-only CAPM: the gap between their alphas is the size/value/
momentum tilt a single-factor model mis-reads as skill.

RF handling is the caller's declaration, not a guess: a long/short spread is
self-financing, so its regression return is the RAW spread
(run_regression(..., subtract_rf=False)); a long-only series regresses its
excess return over RF (the default). evaluate_alpha() packages both models
plus the raw annualized Sharpe into an AlphaResult for the alphasearch sweep.

Standard errors are classical OLS (Newey-West is a documented v2; see the
script). Promoted from scripts/factor_regression.py -- the script remains the
CLI and re-imports everything from here, so there is exactly one copy of the
statistics.
"""

from __future__ import annotations

import datetime
import io
import math
import re
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

RESEARCH_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    "F-F_Research_Data_Factors_daily_CSV.zip"
)
MOMENTUM_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    "F-F_Momentum_Factor_daily_CSV.zip"
)
RESEARCH_CSV = "F-F_Research_Data_Factors_daily.csv"
MOMENTUM_CSV = "F-F_Momentum_Factor_daily.csv"

ALL_FACTORS = ("Mkt-RF", "SMB", "HML", "Mom")
TRADING_DAYS = 252
# Ken French encodes missing observations as these percent sentinels; any real
# daily factor return is far above -99%, so a single threshold catches both.
MISSING_SENTINEL = -99.0
USER_AGENT = "trading-factor-regression/1.0 (research; stdlib urllib)"
_DATE_RE = re.compile(r"^\d{8}$")


# --------------------------------------------------------------------------- #
# OLS core (hand-rolled; no statsmodels/scipy in this repo)
# --------------------------------------------------------------------------- #
def ols(
    x: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, int]:
    """Classical OLS. ``x`` is the full design matrix (constant column included).

    Returns ``(beta, se, tstat, r2, n)``:
      beta  = (X'X)^-1 X'y
      e     = y - X beta                      (residuals)
      s2    = e'e / (n - k)                    (unbiased error variance)
      Var(beta) = s2 (X'X)^-1 ; se = sqrt(diag) ; t = beta / se
      r2    = 1 - e'e / TSS
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n, k = x.shape
    if n <= k:
        raise ValueError(f"need more observations ({n}) than parameters ({k})")
    xtx_inv = np.linalg.inv(x.T @ x)
    beta = xtx_inv @ (x.T @ y)
    resid = y - x @ beta
    ss_res = float(resid @ resid)
    sigma2 = ss_res / (n - k)
    se = np.sqrt(np.diag(sigma2 * xtx_inv))
    tstat = beta / se
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return beta, se, tstat, r2, n


# --------------------------------------------------------------------------- #
# Fama-French factor parsing / fetching
# --------------------------------------------------------------------------- #
def parse_ff_csv(text: str) -> pd.DataFrame:
    """Parse a Ken French daily-factor CSV into a UTC-indexed decimal DataFrame.

    Format quirks handled: a multi-line text preamble, a header row like
    ``,Mkt-RF,SMB,HML,RF``, daily rows ``YYYYMMDD, <space-padded numbers>``,
    and -- LATER in the same file -- a blank line followed by an *annual* block
    (4-digit year rows). We keep ONLY rows whose first field is an 8-digit date,
    which excludes the annual block outright. Values are in PERCENT and are
    divided by 100; the -99.99/-999 missing sentinels become NaN. Header
    whitespace is stripped (the momentum file uses ``   Mom   ``).
    """
    header: list[str] | None = None
    dates: list[pd.Timestamp] = []
    rows: list[list[float]] = []
    for line in text.splitlines():
        fields = [f.strip() for f in line.split(",")]
        first = fields[0]
        if _DATE_RE.match(first):
            when = datetime.datetime.strptime(first, "%Y%m%d").replace(tzinfo=datetime.UTC)
            values = [float(f) for f in fields[1:] if f != ""]
            dates.append(pd.Timestamp(when))
            rows.append(values)
        elif header is None and first == "":
            names = [f for f in fields[1:] if f != ""]
            if names:  # first ",Mkt-RF,..." style header wins; annual re-header ignored
                header = names
    if header is None or not rows:
        raise ValueError("no Fama-French header or daily rows found")
    width = len(header)
    # Guard against ragged rows: keep exactly the header's worth of columns.
    trimmed = [r[:width] for r in rows]
    frame = pd.DataFrame(trimmed, index=pd.DatetimeIndex(dates, name="date"), columns=header)
    frame = frame.where(frame > MISSING_SENTINEL)  # sentinels -> NaN
    return frame / 100.0  # percent -> decimal


def _fetch_ff_csv(url: str, cache_path: Path, refresh: bool) -> str:
    """Download+extract a Ken French zip's CSV, caching the extracted text.

    After the first successful fetch the cache makes re-runs fully offline;
    ``--refresh`` forces a re-download.
    """
    if cache_path.exists() and not refresh:
        return cache_path.read_text()
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request) as response:  # noqa: S310 (trusted canonical host)
        payload = response.read()
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        members = [n for n in archive.namelist() if n.lower().endswith(".csv")]
        if not members:
            raise ValueError(f"no CSV inside zip at {url}")
        text = archive.read(members[0]).decode("latin-1")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(text)
    return text


def load_factors(factors_dir: Path, refresh: bool = False) -> pd.DataFrame:
    """Return a UTC-indexed frame with columns Mkt-RF, SMB, HML, RF, Mom (decimals)."""
    research = parse_ff_csv(
        _fetch_ff_csv(RESEARCH_URL, factors_dir / RESEARCH_CSV, refresh)
    )
    momentum = parse_ff_csv(
        _fetch_ff_csv(MOMENTUM_URL, factors_dir / MOMENTUM_CSV, refresh)
    )
    return research.join(momentum, how="inner")


# --------------------------------------------------------------------------- #
# Alignment + regression
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class RegressionResult:
    names: list[str]  # ["alpha", <factor>, ...]
    beta: np.ndarray
    se: np.ndarray
    tstat: np.ndarray
    r2: float
    adj_r2: float
    n: int

    @property
    def alpha_daily(self) -> float:
        return float(self.beta[0])

    @property
    def alpha_annual_pct(self) -> float:
        return self.alpha_daily * TRADING_DAYS * 100.0

    @property
    def alpha_tstat(self) -> float:
        return float(self.tstat[0])


def run_regression(
    returns: pd.DataFrame,
    factors: pd.DataFrame,
    factor_names: list[str],
    *,
    subtract_rf: bool = True,
) -> RegressionResult:
    """Inner-join returns with factors, form the regression return, and regress.

    subtract_rf=True (default, unchanged behavior): regress the EXCESS return
    strategy - RF -- correct for a long-only or benchmark-relative series.
    subtract_rf=False: regress the raw series -- correct for a self-financing
    long/short spread, which already nets out the financing leg. RF must be
    present and non-NaN either way so both modes see identical observations.
    """
    merged = returns.join(factors, how="inner")
    needed = list(factor_names) + ["RF", "strategy"]
    missing = [c for c in needed if c not in merged.columns]
    if missing:
        raise ValueError(f"missing columns after join: {missing}")
    merged = merged.dropna(subset=needed)
    if merged.empty:
        raise ValueError("no overlapping dates between returns and factors")
    if subtract_rf:
        y = (merged["strategy"] - merged["RF"]).to_numpy()
    else:
        y = merged["strategy"].to_numpy()
    columns = [np.ones(len(merged))] + [merged[f].to_numpy() for f in factor_names]
    x = np.column_stack(columns)
    beta, se, tstat, r2, n = ols(x, y)
    k = x.shape[1]
    adj_r2 = 1.0 - (1.0 - r2) * (n - 1) / (n - k) if n > k else float("nan")
    return RegressionResult(
        names=["alpha", *factor_names],
        beta=beta,
        se=se,
        tstat=tstat,
        r2=r2,
        adj_r2=adj_r2,
        n=n,
    )


# --------------------------------------------------------------------------- #
# AlphaResult: the alphasearch evaluation unit (spec section 3.4)
# --------------------------------------------------------------------------- #
def annualized_sharpe(returns: pd.Series) -> float:
    """Raw annualized Sharpe of a daily return series (0% cash rate).

    Reported for tradability context only; the gate statistic is the
    four-factor alpha t, never this number.
    """
    r = returns.dropna()
    if len(r) < 2:
        return float("nan")
    sd = float(r.std(ddof=1))
    if sd == 0:
        return float("nan")
    return float(r.mean()) / sd * math.sqrt(TRADING_DAYS)


@dataclass(frozen=True)
class AlphaResult:
    """CAPM + four-factor decomposition of one return series."""

    four_factor: RegressionResult
    capm: RegressionResult
    sharpe_annual: float

    @property
    def alpha_annual_pct(self) -> float:
        return self.four_factor.alpha_annual_pct

    @property
    def alpha_tstat(self) -> float:
        return self.four_factor.alpha_tstat

    @property
    def capm_alpha_annual_pct(self) -> float:
        return self.capm.alpha_annual_pct

    @property
    def capm_alpha_tstat(self) -> float:
        return self.capm.alpha_tstat

    @property
    def n(self) -> int:
        return self.four_factor.n


def evaluate_alpha(
    returns: pd.Series, factors: pd.DataFrame, *, self_financing: bool
) -> AlphaResult:
    """Regress one daily return series on CAPM and the Carhart four factors.

    self_financing=True (the L/S spread): regress the raw series.
    self_financing=False (long-only): regress returns - RF.
    """
    frame = returns.rename("strategy").to_frame()
    subtract = not self_financing
    four = run_regression(frame, factors, list(ALL_FACTORS), subtract_rf=subtract)
    capm = run_regression(frame, factors, ["Mkt-RF"], subtract_rf=subtract)
    return AlphaResult(
        four_factor=four, capm=capm, sharpe_annual=annualized_sharpe(returns)
    )
