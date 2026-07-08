"""Cross-sectional study: does IV skew predict forward stock returns?

The hypothesis under test is the classic one: a steep put skew (the market
paying up for downside protection) precedes LOWER forward returns. If that
holds up cross-sectionally -- consistently, with the right (negative) sign --
the signal is worth paying a vendor for a real options surface and building
properly. A null result on this small gathered sample is *weak evidence*, not
a refutation: we simply do not have the n to detect a modest edge.

Data it reads (gathered by a SEPARATE process into ``data/options-poc/``):

* ``samples.jsonl`` -- one JSON object per (symbol, decision_date):
    {"symbol","decision_date","spot_at_decision","target_expiration",
     "days_to_expiry","contracts":[{"role","type","strike","close",...}]}
* ``underlying/<SYMBOL>.parquet`` -- daily OHLCV, UTC DatetimeIndex, used for
  the forward return.

The analysis:

1. Per cell, compute skew (via ``trading.research.options_iv``) and the forward
   20-trading-day underlying return. Skip cells missing either.
2. Cross-sectional tests, for both ``skew_put_atm`` and ``skew_put_call``:
     * Spearman rank correlation of skew vs forward return (expect NEGATIVE),
       with n and an approximate t.
     * Per-date tercile spread: mean forward return of the bottom (flat-skew)
       tercile minus the top (steep-skew) tercile, averaged across dates, plus
       the fraction of dates where that (bottom - top) spread is positive.
3. The same tests on skew CHANGE -- each name de-meaned by its own average skew
   -- to strip out structural per-name skew levels.

Run (once the gathering process has populated the data directory):

    uv run python scripts/option_skew_analysis.py

The core is factored into importable functions (loading, skew/return assembly,
Spearman, terciles) so the math is unit-tested without any real data.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from trading.research.options_iv import skew_from_cell

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = ROOT / "data" / "options-poc"
FORWARD_HORIZON = 20  # trading rows

# Skew columns we run the whole battery over.
SKEW_COLS = ("skew_put_atm", "skew_put_call")


# --- Loading ---------------------------------------------------------------


def load_samples(path: Path) -> list[dict]:
    """Parse samples.jsonl into a list of cell dicts.

    Blank lines are ignored, and an unparseable line (e.g. a torn final line
    left by a SIGKILLed gather) is skipped rather than aborting the whole
    analysis -- one bad line must not sink a study over thousands of good cells.
    """
    cells: list[dict] = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                cells.append(json.loads(line))
            except ValueError:
                continue
    return cells


def load_underlying(symbol: str, data_dir: Path) -> pd.DataFrame | None:
    """Load a symbol's daily OHLCV parquet, or None if absent."""
    path = data_dir / "underlying" / f"{symbol}.parquet"
    if not path.exists():
        return None
    return pd.read_parquet(path)


def forward_return(
    underlying: pd.DataFrame,
    decision_date: str | dt.date,
    spot_at_decision: float,
    horizon: int = FORWARD_HORIZON,
) -> float | None:
    """Forward return: close ``horizon`` trading rows after the decision date,
    divided by the spot at decision, minus one.

    The decision row is located as the first bar on/after ``decision_date``;
    the forward bar is ``horizon`` rows further on. Returns None when there is
    not enough forward data (the horizon runs off the end of the series) or the
    decision date is past the series entirely.
    """
    if underlying is None or underlying.empty:
        return None
    ts = pd.Timestamp(decision_date)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    idx = underlying.index
    pos = idx.searchsorted(ts, side="left")
    if pos >= len(idx):
        return None
    fwd_pos = pos + horizon
    if fwd_pos >= len(idx):
        return None
    fwd_close = float(underlying["close"].iloc[fwd_pos])
    if spot_at_decision <= 0:
        return None
    return fwd_close / spot_at_decision - 1.0


def build_observations(
    samples: list[dict],
    data_dir: Path,
    horizon: int = FORWARD_HORIZON,
) -> pd.DataFrame:
    """Assemble one row per usable cell.

    Columns: symbol, decision_date (datetime64), skew_put_atm, skew_put_call,
    fwd_return. Cells whose skew cannot be computed for a measure carry NaN for
    that measure; cells lacking forward data are dropped entirely (no return =
    nothing to correlate against).
    """
    underlying_cache: dict[str, pd.DataFrame | None] = {}
    rows: list[dict] = []
    for cell in samples:
        symbol = cell["symbol"]
        if symbol not in underlying_cache:
            underlying_cache[symbol] = load_underlying(symbol, data_dir)
        underlying = underlying_cache[symbol]

        fwd = forward_return(
            underlying, cell["decision_date"], float(cell["spot_at_decision"]), horizon
        )
        if fwd is None:
            continue

        skew = skew_from_cell(cell)
        skew_put_atm = skew.skew_put_atm if skew is not None else None
        skew_put_call = skew.skew_put_call if skew is not None else None
        if skew_put_atm is None and skew_put_call is None:
            continue

        rows.append(
            {
                "symbol": symbol,
                "decision_date": pd.Timestamp(cell["decision_date"]).normalize(),
                "skew_put_atm": skew_put_atm,
                "skew_put_call": skew_put_call,
                "fwd_return": fwd,
            }
        )

    return pd.DataFrame(
        rows,
        columns=["symbol", "decision_date", "skew_put_atm", "skew_put_call", "fwd_return"],
    )


def add_skew_change(obs: pd.DataFrame) -> pd.DataFrame:
    """Add per-name de-meaned skew columns (``<col>_change``).

    ``skew_change = skew - mean(skew for that symbol across its own dates)``.
    This strips each name's structural skew level (some names are chronically
    skewed) so the test sees only *deviations* from a name's own norm.
    """
    out = obs.copy()
    for col in SKEW_COLS:
        name_mean = out.groupby("symbol")[col].transform("mean")
        out[f"{col}_change"] = out[col] - name_mean
    return out


# --- Statistics (hand-rolled; no scipy) ------------------------------------


@dataclass(frozen=True)
class SpearmanResult:
    rho: float | None
    n: int
    t_stat: float | None

    @property
    def approx_significant(self) -> bool:
        """|t| > ~2 is the usual eyeball threshold for p<0.05 at these n."""
        return self.t_stat is not None and abs(self.t_stat) > 2.0


def spearman(x: pd.Series, y: pd.Series) -> SpearmanResult:
    """Spearman rank correlation with an approximate t-statistic.

    Pairs where either side is NaN are dropped. Spearman is Pearson on the
    average-ranked values; the significance read is the standard
    ``t = rho * sqrt((n-2)/(1-rho^2))`` on n-2 df.
    """
    paired = pd.DataFrame({"x": x, "y": y}).dropna()
    n = len(paired)
    if n < 3:
        return SpearmanResult(rho=None, n=n, t_stat=None)
    rx = paired["x"].rank()
    ry = paired["y"].rank()
    if rx.std(ddof=0) == 0 or ry.std(ddof=0) == 0:
        return SpearmanResult(rho=None, n=n, t_stat=None)
    rho = float(rx.corr(ry))
    if abs(rho) >= 1.0:
        t_stat = math.inf * (1.0 if rho > 0 else -1.0)
    else:
        t_stat = rho * math.sqrt((n - 2) / (1.0 - rho * rho))
    return SpearmanResult(rho=rho, n=n, t_stat=t_stat)


@dataclass(frozen=True)
class TercileResult:
    mean_spread: float | None  # mean over dates of (bottom - top) forward return
    frac_positive: float | None  # fraction of dates with a positive spread
    n_dates: int  # dates with enough symbols to form terciles


def tercile_spread(
    obs: pd.DataFrame,
    skew_col: str,
    ret_col: str = "fwd_return",
    date_col: str = "decision_date",
) -> TercileResult:
    """Per-date bottom-minus-top tercile forward-return spread.

    Within each date, symbols are ranked by ``skew_col``; the bottom tercile is
    the flattest-skew names, the top tercile the steepest. The reported spread
    is ``mean(bottom fwd return) - mean(top fwd return)``. Under the hypothesis
    (steep skew -> lower return) this is POSITIVE. Dates with fewer than 3
    usable names are skipped. ``frac_positive`` measures consistency across
    dates.
    """
    spreads: list[float] = []
    for _, group in obs.groupby(date_col):
        g = group[[skew_col, ret_col]].dropna()
        n = len(g)
        if n < 3:
            continue
        k = max(1, n // 3)
        ordered = g.sort_values(skew_col)
        bottom = ordered.head(k)[ret_col].mean()  # flattest skew
        top = ordered.tail(k)[ret_col].mean()  # steepest skew
        spreads.append(float(bottom - top))

    if not spreads:
        return TercileResult(mean_spread=None, frac_positive=None, n_dates=0)
    series = pd.Series(spreads)
    return TercileResult(
        mean_spread=float(series.mean()),
        frac_positive=float((series > 0).mean()),
        n_dates=len(series),
    )


@dataclass(frozen=True)
class MeasureReport:
    label: str
    skew_col: str
    spearman: SpearmanResult
    tercile: TercileResult


def analyze_measure(obs: pd.DataFrame, skew_col: str, label: str) -> MeasureReport:
    return MeasureReport(
        label=label,
        skew_col=skew_col,
        spearman=spearman(obs[skew_col], obs["fwd_return"]),
        tercile=tercile_spread(obs, skew_col),
    )


def run_all_measures(obs: pd.DataFrame) -> list[MeasureReport]:
    """Level and change tests for both skew measures, side by side."""
    obs = add_skew_change(obs)
    reports: list[MeasureReport] = []
    for col in SKEW_COLS:
        reports.append(analyze_measure(obs, col, f"{col} (level)"))
        reports.append(analyze_measure(obs, f"{col}_change", f"{col} (change)"))
    return reports


# --- Reporting -------------------------------------------------------------


def _fmt(value: float | None, width: int, number_spec: str) -> str:
    """Format a numeric cell to a fixed width, right-justifying "n/a" for None
    so the table columns stay aligned whether or not a stat is available.

    ``number_spec`` is the sign/precision/type tail (e.g. ``+.3f``); the width
    is inserted between the sign flag and the precision as the format grammar
    requires.
    """
    if value is None:
        return "n/a".rjust(width)
    sign, rest = (
        (number_spec[0], number_spec[1:]) if number_spec[:1] in "+- " else ("", number_spec)
    )
    return format(value, f"{sign}{width}{rest}")


def format_report(obs: pd.DataFrame, reports: list[MeasureReport]) -> str:
    n_obs = len(obs)
    n_symbols = obs["symbol"].nunique()
    n_dates = obs["decision_date"].nunique()

    lines: list[str] = []
    lines.append("=" * 78)
    lines.append("IV SKEW -> FORWARD RETURN  (cross-sectional POC)")
    lines.append("=" * 78)
    lines.append(
        f"observations: {n_obs}   symbols: {n_symbols}   decision dates: {n_dates}"
        f"   forward horizon: {FORWARD_HORIZON} trading days"
    )
    lines.append("")
    lines.append("Hypothesis: steep put skew -> LOWER forward return, i.e. NEGATIVE Spearman")
    lines.append("rho and a POSITIVE (bottom-minus-top) tercile spread, consistently across dates.")
    lines.append("")

    header = (
        f"{'measure':<26} {'n':>4} {'spearman':>9} {'~t':>7} "
        f"{'terc.spread':>12} {'%dates>0':>9} {'#dates':>7}"
    )
    lines.append(header)
    lines.append("-" * len(header))
    for r in reports:
        s, t = r.spearman, r.tercile
        lines.append(
            f"{r.label:<26} {s.n:>4} "
            f"{_fmt(s.rho, 9, '+.3f')} {_fmt(s.t_stat, 7, '+.2f')} "
            f"{_fmt(t.mean_spread, 12, '+.4f')} {_fmt(t.frac_positive, 9, '.2f')} "
            f"{t.n_dates:>7}"
        )
    lines.append("")
    lines.append(_verdict(obs, reports))
    lines.append("=" * 78)
    return "\n".join(lines)


def _verdict(obs: pd.DataFrame, reports: list[MeasureReport]) -> str:
    """Plain-language read of the battery.

    A "signal" here is the hypothesised sign showing up *and* holding together:
    a negative Spearman rho with a credible t, plus a positive tercile spread
    on a majority of dates. Anything else on this sample size is read as weak
    evidence, not a refutation.
    """
    n_obs = len(obs)

    def is_signal(r: MeasureReport) -> bool:
        s, t = r.spearman, r.tercile
        neg_corr = s.rho is not None and s.rho < 0 and s.approx_significant
        pos_spread = (
            t.mean_spread is not None
            and t.mean_spread > 0
            and t.frac_positive is not None
            and t.frac_positive > 0.5
        )
        return neg_corr and pos_spread

    hits = [r.label for r in reports if is_signal(r)]
    lines = ["VERDICT:"]
    if hits:
        lines.append("  Consistent NEGATIVE skew->return relationship on: " + ", ".join(hits) + ".")
        lines.append("  This is the sign and consistency we hoped for -- worth paying a vendor")
        lines.append("  for a real options surface and building the signal properly. Confirm on")
        lines.append(f"  a larger sample before sizing any capital (current n={n_obs}).")
    else:
        leaning = [r.label for r in reports if r.spearman.rho is not None and r.spearman.rho < 0]
        lines.append(
            f"  No consistent, significant negative relationship on n={n_obs} observations."
        )
        if leaning:
            lines.append(
                "  Some measures lean the right way (negative rho) but not significantly: "
                + ", ".join(leaning)
                + "."
            )
        lines.append("  Read this as WEAK EVIDENCE given the small sample, NOT a refutation of the")
        lines.append(
            "  skew hypothesis. A larger gather is the cheapest next step before dropping it."
        )
    return "\n".join(lines)


# --- Entry point -----------------------------------------------------------


def run(data_dir: Path) -> int:
    samples_path = data_dir / "samples.jsonl"
    if not samples_path.exists():
        print(f"No gathered data at {samples_path}.")
        print("Run the options-POC gathering process first to populate")
        print(f"  {data_dir}/samples.jsonl  and  {data_dir}/underlying/<SYMBOL>.parquet")
        return 0

    samples = load_samples(samples_path)
    if not samples:
        print(f"{samples_path} is present but empty -- nothing to analyze yet.")
        return 0

    obs = build_observations(samples, data_dir)
    if obs.empty:
        print(
            f"Parsed {len(samples)} sample cell(s) but none yielded both a skew and "
            f"{FORWARD_HORIZON}-day forward data."
        )
        print("This usually means the underlying parquets are missing or too short.")
        return 0

    reports = run_all_measures(obs)
    print(format_report(obs, reports))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"gathered options-POC directory (default: {DEFAULT_DATA_DIR})",
    )
    args = parser.parse_args(argv)
    return run(args.data_dir)


if __name__ == "__main__":
    sys.exit(main())
