"""Systematic cross-sectional signal scan (docs/experiments.md, options track).

The question this answers: among a filtered pool of candidate stocks (e.g. the
"survivors" left after a risk veto), does ANY metric we already have predict the
cross-section of forward returns? Rather than hand-test signals one at a time,
this scans a whole panel at once and ranks them by information coefficient.

For each metric it computes the **IC** = the mean, across monthly decision
dates, of the cross-sectional Spearman rank-correlation between the metric (known
at the decision date) and the forward stock return. A positive IC means "high
metric -> high forward return." The **t-stat** across months is what matters --
a high mean IC on few, noisy months is not signal. Each metric is reported at
two horizons and both out-of-sample and full-sample, because a real signal
should hold in BOTH; a hit that only shows up in one cut is likely noise.

MULTIPLE-TESTING WARNING: scanning N metrics will, by chance, throw up ~N*0.05
hits at |t|>2. Treat this as HYPOTHESIS GENERATION, not confirmation: a metric
is only credible if it is consistent across horizons AND across the OOS/FULL
split, and even then it must clear a full walk-forward backtest (with costs) and
the reserved holdout before it is believed. Do not cherry-pick the top row.

All metrics are point-in-time: option metrics are as-of the decision-date cell;
price metrics use only trailing bars up to the decision date. Forward returns use
the survivorship-free adjusted-close cache (total return).

Run:  uv run python scripts/signal_scan.py \
          --samples data/options-iv/samples-midcap.jsonl \
          --cache-dir data/equities-midcap-tiingo
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SAMPLES = ROOT / "data" / "options-iv" / "samples.jsonl"
DEFAULT_CACHE = ROOT / "data" / "equities-tiingo"
DEFAULT_OOS = "2021-07-01"
MIN_NAMES = 15  # skip a month whose (filtered) cross-section is thinner than this

# The candidate panel. Option metrics are read from the cell; price metrics are
# computed from the bar cache. Extend this list to test new ideas (a fundamentals
# column would slot in the same way, keyed by symbol+date).
OPTION_METRICS = ("excite", "atm_iv", "otm_put_iv", "otm_call_iv", "smile",
                  "cp_vol", "wing_vol", "tot_vol", "atm_spread")
PRICE_METRICS = ("mom21", "mom63", "mom126", "mom252", "rev5", "rvol21", "vrp", "disthigh")


def _cell_metrics(cell: dict) -> dict:
    """The option-derived metrics for one samples.jsonl cell (NaN when a leg is
    missing so a partial cell never fabricates a value)."""
    d = {c["role"]: c for c in cell.get("contracts", [])}

    def iv(role):
        return d.get(role, {}).get("iv")

    def vol(role):
        return d.get(role, {}).get("volume") or 0

    atm = d.get("atm", {})
    put_iv, call_iv, atm_iv = iv("otm_put"), iv("otm_call"), iv("atm")
    rr = cell.get("skew_put_call")
    spread = ((atm["ask"] - atm["bid"]) / atm["mid"]
              if atm.get("mid") and atm.get("bid") is not None and atm.get("ask") is not None
              else np.nan)
    return {
        "hedge": cell.get("skew_put_atm"),
        "excite": (-rr if rr is not None else np.nan),  # call-vs-put IV richness
        "atm_iv": atm_iv,
        "otm_put_iv": put_iv,
        "otm_call_iv": call_iv,
        "smile": ((put_iv + call_iv) / 2 - atm_iv
                  if None not in (put_iv, call_iv, atm_iv) else np.nan),
        "cp_vol": np.log((vol("atm") + vol("otm_call") + 1) / (vol("otm_put") + 1)),
        "wing_vol": np.log((vol("otm_call") + 1) / (vol("otm_put") + 1)),
        "tot_vol": vol("atm") + vol("otm_put") + vol("otm_call"),
        "atm_spread": spread,
    }


def load_panel(samples: Path, cache_dir: Path, horizons=(21, 63)) -> pd.DataFrame:
    """One row per (symbol, decision date): option metrics + price metrics +
    forward returns for each horizon."""
    rows = []
    for line in samples.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        cell = json.loads(line)
        row = {"symbol": cell["symbol"], "date": pd.Timestamp(cell["decision_date"], tz="UTC")}
        row.update(_cell_metrics(cell))
        rows.append(row)
    df = pd.DataFrame(rows)
    closes = {}
    for sym in df["symbol"].unique():
        p = cache_dir / f"{sym}.parquet"
        if p.exists():
            closes[sym] = pd.read_parquet(p)["close"]

    def prior_pos(sym, date):
        s = closes.get(sym)
        if s is None:
            return None, None
        return s, s.index.searchsorted(date, side="right") - 1

    def fwd(sym, date, horizon):
        s = closes.get(sym)
        if s is None:
            return np.nan
        p0 = s.index.searchsorted(date, side="left")
        if p0 < 0 or p0 + horizon >= len(s):
            return np.nan
        return s.iloc[p0 + horizon] / s.iloc[p0] - 1

    def trail(sym, date, window):
        s, p = prior_pos(sym, date)
        if s is None or p is None or p - window < 0:
            return np.nan
        return s.iloc[p] / s.iloc[p - window] - 1

    def rvol(sym, date, window=21):
        s, p = prior_pos(sym, date)
        if s is None or p is None or p < window:
            return np.nan
        return s.iloc[p - window:p].pct_change().std() * np.sqrt(252)

    def disthigh(sym, date, window=252):
        s, p = prior_pos(sym, date)
        if s is None or p is None:
            return np.nan
        return s.iloc[p] / s.iloc[max(0, p - window):p + 1].max() - 1

    sd = list(zip(df["symbol"], df["date"], strict=True))
    for h in horizons:
        df[f"f{h}"] = [fwd(s, d, h) for s, d in sd]
    for w in (21, 63, 126, 252):
        df[f"mom{w}"] = [trail(s, d, w) for s, d in sd]
    df["rev5"] = [trail(s, d, 5) for s, d in sd]
    df["rvol21"] = [rvol(s, d) for s, d in sd]
    df["vrp"] = df["atm_iv"] - df["rvol21"]
    df["disthigh"] = [disthigh(s, d) for s, d in sd]
    return df


def information_coefficient(frame: pd.DataFrame, metric: str, ret: str, min_names: int = MIN_NAMES):
    """Mean monthly cross-sectional Spearman(metric, ret) and its t-stat.

    Factored out (no I/O) so the aggregation math is unit-tested directly.
    Returns (mean_ic, tstat, n_months).
    """
    ics = []
    for _, g in frame.groupby("date"):
        g = g.dropna(subset=[metric, ret])
        if len(g) < min_names:
            continue
        ics.append(g[metric].rank().corr(g[ret].rank()))
    ics = np.asarray([x for x in ics if x == x], dtype="float64")
    if len(ics) == 0:
        return np.nan, np.nan, 0
    se = ics.std(ddof=1) / np.sqrt(len(ics)) if len(ics) > 1 else 0.0
    t = ics.mean() / se if se > 0 else np.nan  # se==0 (identical ICs) -> t undefined
    return float(ics.mean()), float(t), len(ics)


def survivors(frame: pd.DataFrame, veto_frac: float) -> pd.DataFrame:
    """Drop the most downside-hedged names (top ``veto_frac`` of skew_put_atm per
    month) -- the risk veto. veto_frac<=0 keeps everyone (scan the raw universe)."""
    if veto_frac <= 0:
        return frame
    keep = []
    for _, g in frame.groupby("date"):
        h = g["hedge"].dropna()
        if h.empty:
            keep.append(g)
            continue
        keep.append(g[g["hedge"] < h.quantile(1 - veto_frac)])
    return pd.concat(keep) if keep else frame


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Rank candidate metrics by cross-sectional IC.")
    ap.add_argument("--samples", type=Path, default=DEFAULT_SAMPLES)
    ap.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    ap.add_argument("--oos-start", default=DEFAULT_OOS)
    ap.add_argument("--veto-frac", type=float, default=1 / 3,
                    help="drop this top fraction of hedged names before scanning (0 = no veto)")
    args = ap.parse_args(argv)

    df = load_panel(args.samples, args.cache_dir)
    pool = survivors(df, args.veto_frac)
    oos = pd.Timestamp(args.oos_start, tz="UTC")
    metrics = [m for m in (*OPTION_METRICS, *PRICE_METRICS) if m in pool.columns]

    print(f"{args.samples.name}: {len(df)} cells, {df['symbol'].nunique()} symbols; "
          f"veto_frac={args.veto_frac} -> {len(pool)} survivor-cells")
    print("IC = mean monthly Spearman(metric, forward return); ranked by |t| at 63d OOS.")
    cols = ("H21 OOS t", "H63 OOS t", "H63 FULL t", "IC63 OOS")
    print(f"{'metric':11} | {cols[0]:>9} | {cols[1]:>9} | {cols[2]:>10} | {cols[3]:>9}")
    results = []
    for m in metrics:
        _, t21o, _ = information_coefficient(pool[pool["date"] >= oos], m, "f21")
        ic63o, t63o, _ = information_coefficient(pool[pool["date"] >= oos], m, "f63")
        _, t63f, _ = information_coefficient(pool, m, "f63")
        results.append((m, t21o, t63o, t63f, ic63o))
    results.sort(key=lambda r: -abs(r[2]) if r[2] == r[2] else 0)
    for m, t21o, t63o, t63f, ic63o in results:
        print(f"{m:11} | {t21o:>+9.1f} | {t63o:>+9.1f} | {t63f:>+10.1f} | {ic63o:>+9.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
