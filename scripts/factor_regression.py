"""Decompose a strategy's returns into factor exposures (betas) plus ALPHA.

Thin CLI over trading.alphasearch.evaluate, which owns the statistics (OLS,
Ken French factor fetching/parsing, RegressionResult) -- see that module's
docstring for the full "why alpha, not Sharpe-vs-benchmark" story. This
script only loads a returns CSV, fetches/caches factors, and prints the
four-factor vs CAPM contrast.

Two-step workflow to decompose a strategy:

    trading backtest --venue equities --walk-forward --dump-returns rets.csv ...
    uv run python scripts/factor_regression.py --returns rets.csv

The first command writes daily strategy/benchmark returns; this script fetches +
caches the factors (offline after the first run) and prints the decomposition.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from trading.alphasearch.evaluate import (
    ALL_FACTORS as ALL_FACTORS,
)
from trading.alphasearch.evaluate import (
    MISSING_SENTINEL as MISSING_SENTINEL,
)
from trading.alphasearch.evaluate import (
    TRADING_DAYS as TRADING_DAYS,
)
from trading.alphasearch.evaluate import (
    RegressionResult as RegressionResult,
)
from trading.alphasearch.evaluate import (
    load_factors as load_factors,
)
from trading.alphasearch.evaluate import (
    ols as ols,
)
from trading.alphasearch.evaluate import (
    parse_ff_csv as parse_ff_csv,
)
from trading.alphasearch.evaluate import (
    run_regression as run_regression,
)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_FACTORS_DIR = ROOT / "data" / "factors"


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
