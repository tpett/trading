"""Decompose a strategy's returns into factor exposures (betas) plus ALPHA.

The core question this answers: *is a strategy's edge real, or is it just paid
factor beta?* Any long book earns some return simply by being exposed to
well-known, broadly-compensated risk factors -- the overall market, small-cap
(size), cheap/value, and recent-winner (momentum) tilts. Those exposures are
freely available (buy an index / a factor ETF); they are not evidence of skill.

ALPHA is the piece of the strategy's *excess* return (over the risk-free rate)
that the factors do NOT explain -- the intercept of a regression of the
strategy's daily excess return on the factor returns. A positive, statistically
significant alpha is the return the strategy earns after stripping out every
factor tilt it happens to carry. That is the number that says "real edge".

We regress on the canonical Fama-French factors (Ken French's data library):
``Mkt-RF`` (market minus risk-free), ``SMB`` (small minus big / size),
``HML`` (high minus low book-to-market / value), and ``Mom`` (momentum). These
are the generally-accepted, whole-US-market factor definitions -- the reference
everyone benchmarks against -- so a loading here is an honest "you are being paid
for exposure X" statement, not an artifact of a bespoke factor construction.

Contrast is the point. This tool always prints BOTH the full four-factor model
AND a market-only (CAPM) regression. A strategy can show a big CAPM alpha yet a
near-zero four-factor alpha: that gap is the alpha the single-factor model
mis-attributed to skill but which is actually a size/value/momentum tilt. E.g. a
mid-cap momentum book will load heavily on SMB and Mom, and its "CAPM alpha"
largely evaporates once those factors are in the model.

Standard errors here are classical OLS SEs. Daily strategy returns are mildly
autocorrelated, which biases classical SEs; a Newey-West HAC correction is the
documented, well-understood refinement for that and is a natural v2 -- it is not
needed to answer the first-order "is there any alpha at all?" question and is
intentionally out of scope for v1.

Two-step workflow to decompose a strategy:

    trading backtest --venue equities --walk-forward --dump-returns rets.csv ...
    uv run python scripts/factor_regression.py --returns rets.csv

The first command writes daily strategy/benchmark returns; this script fetches +
caches the factors (offline after the first run) and prints the decomposition.
"""

from __future__ import annotations

import argparse
import datetime
import io
import re
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_FACTORS_DIR = ROOT / "data" / "factors"

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
    returns: pd.DataFrame, factors: pd.DataFrame, factor_names: list[str]
) -> RegressionResult:
    """Inner-join returns with factors, form the excess return, and regress.

    Strategy EXCESS return = strategy - RF. The design matrix is a constant
    (the alpha intercept) plus the selected factor columns.
    """
    merged = returns.join(factors, how="inner")
    needed = list(factor_names) + ["RF", "strategy"]
    missing = [c for c in needed if c not in merged.columns]
    if missing:
        raise ValueError(f"missing columns after join: {missing}")
    merged = merged.dropna(subset=needed)
    if merged.empty:
        raise ValueError("no overlapping dates between returns and factors")
    y = (merged["strategy"] - merged["RF"]).to_numpy()
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
# Reporting
# --------------------------------------------------------------------------- #
def load_returns(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "date" not in frame.columns or "strategy" not in frame.columns:
        raise ValueError("returns CSV must have at least 'date' and 'strategy' columns")
    frame["date"] = pd.to_datetime(frame["date"], utc=True)
    return frame.set_index("date")


def _verdict(result: RegressionResult) -> str:
    sig = "significant (t>2)" if abs(result.alpha_tstat) > 2 else "not distinguishable from zero"
    return (
        f"ALPHA: {result.alpha_annual_pct:+.1f}%/yr, "
        f"t={result.alpha_tstat:+.1f} — {sig}"
    )


def _print_model(title: str, result: RegressionResult) -> None:
    print(f"\n{title}  (N={result.n})")
    print(f"  {'term':<10}{'coef':>14}{'t-stat':>10}   note")
    for i, name in enumerate(result.names):
        if name == "alpha":
            coef = f"{result.alpha_annual_pct:+.2f}%/yr"
            note = "annualized intercept (daily x 252)"
        else:
            coef = f"{result.beta[i]:+.4f}"
            flag = "significant loading" if abs(result.tstat[i]) > 2 else ""
            note = flag
        print(f"  {name:<10}{coef:>14}{result.tstat[i]:>+10.2f}   {note}")
    print(f"  R^2 = {result.r2:.3f}   adj R^2 = {result.adj_r2:.3f}")
    print(f"  {_verdict(result)}")
    loaders = [
        n for n, t in zip(result.names[1:], result.tstat[1:], strict=True) if abs(t) > 2
    ]
    print(f"  significant factor loadings: {', '.join(loaders) if loaders else 'none'}")


def report(returns: pd.DataFrame, factors: pd.DataFrame, factor_names: list[str]) -> None:
    full = run_regression(returns, factors, factor_names)
    _print_model(f"Factor model: {', '.join(factor_names)}", full)
    # Always show the market-only (CAPM) line for contrast: the gap between its
    # alpha and the full-model alpha is exactly the return the extra factors
    # explain (i.e. tilt, not skill).
    if factor_names != ["Mkt-RF"]:
        capm = run_regression(returns, factors, ["Mkt-RF"])
        _print_model("CAPM (market only)", capm)
        print(
            f"\nCAPM alpha {capm.alpha_annual_pct:+.1f}%/yr vs "
            f"{len(factor_names)}-factor alpha {full.alpha_annual_pct:+.1f}%/yr: "
            "the difference is the size/value/momentum tilt the market-only model "
            "mis-reads as alpha."
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--returns",
        type=Path,
        required=True,
        help="returns CSV (date,strategy[,benchmark])",
    )
    parser.add_argument(
        "--factors",
        nargs="+",
        default=list(ALL_FACTORS),
        choices=list(ALL_FACTORS),
        help="factors to include (default: all four; e.g. 'Mkt-RF' alone for CAPM)",
    )
    parser.add_argument("--refresh", action="store_true", help="re-download the factor CSVs")
    parser.add_argument(
        "--factors-dir", type=Path, default=DEFAULT_FACTORS_DIR, help="factor cache directory"
    )
    args = parser.parse_args(argv)

    returns = load_returns(args.returns)
    factors = load_factors(args.factors_dir, refresh=args.refresh)
    print(
        f"loaded {len(returns)} return rows "
        f"({returns.index.min().date()}..{returns.index.max().date()}); "
        f"factors {factors.index.min().date()}..{factors.index.max().date()}"
    )
    report(returns, factors, list(args.factors))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
