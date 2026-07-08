"""Model-free cross-sectional test of the IV-skew premium (docs/experiments.md §9).

A long-only backtest cannot separate a skew *premium* from market *beta*. This
script measures the skew -> forward-return relationship directly, so it says
whether steep put-skew actually precedes lower cross-sectional returns
independent of market direction.

For each monthly decision date it ranks the gathered names by ``skew_put_atm``
and reports, per holding horizon:

* the **tercile spread** = mean forward total-return of the LOW-skew third minus
  the HIGH-skew third -- exactly what a market-neutral long-low / short-high book
  would earn (hypothesis: POSITIVE, flat skew outperforms steep skew);
* the **Spearman** rank correlation of skew vs forward return (hypothesis:
  NEGATIVE);

each with a t-stat across months. Forward returns come from the survivorship-free
Tiingo adjusted-close cache (total return), so this is a clean signal measurement.

Run:  uv run python scripts/skew_premium_study.py
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
# Horizons in trading days: 21 ~ the backtest's monthly rebalance, 42/63 test
# whether the (slow) skew signal needs a longer hold.
HORIZONS = (21, 42, 63)
MIN_NAMES = 15          # skip a month with too thin a cross-section
OOS_START = "2021-07-01"  # first OOS test date in the walk-forward (§9)


def load_cells(samples: Path) -> pd.DataFrame:
    rows = []
    for line in samples.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        c = json.loads(line)
        if c.get("skew_put_atm") is not None:
            rows.append((c["symbol"], pd.Timestamp(c["decision_date"], tz="UTC"), float(c["skew_put_atm"])))
    return pd.DataFrame(rows, columns=["symbol", "date", "skew"])


def load_closes(symbols, cache: Path) -> dict[str, pd.Series]:
    out = {}
    for sym in symbols:
        p = cache / f"{sym}.parquet"
        if p.exists():
            out[sym] = pd.read_parquet(p)["close"]
    return out


def forward_return(closes, sym, date, horizon):
    s = closes.get(sym)
    if s is None:
        return np.nan
    pos = s.index.searchsorted(date, side="left")
    if pos >= len(s) or pos + horizon >= len(s):
        return np.nan
    return s.iloc[pos + horizon] / s.iloc[pos] - 1.0


def _tstat(a):
    a = np.asarray(a)
    return a.mean() / (a.std(ddof=1) / np.sqrt(len(a))) if len(a) > 1 else np.nan


def study(frame, horizon, label):
    spreads, spearmans, lows, highs, ns = [], [], [], [], []
    for _, g in frame.groupby("date"):
        if len(g) < MIN_NAMES:
            continue
        q = g["skew"].quantile([1 / 3, 2 / 3])
        low = g[g["skew"] <= q.iloc[0]]["fwd"].mean()   # flat skew -> buy
        high = g[g["skew"] >= q.iloc[1]]["fwd"].mean()  # steep skew -> avoid/short
        spreads.append(low - high)
        spearmans.append(g["skew"].rank().corr(g["fwd"].rank()))
        lows.append(low)
        highs.append(high)
        ns.append(len(g))
    spreads, spearmans = np.array(spreads), np.array(spearmans)
    print(
        f"H={horizon:>2}d {label:<20} months={len(spreads):>2} names/mo~{np.mean(ns):.0f} | "
        f"tercile spread {spreads.mean() * 100:+.3f}%/period t={_tstat(spreads):+.2f} "
        f"hit={np.mean(spreads > 0) * 100:.0f}% | "
        f"Spearman {spearmans.mean():+.4f} t={_tstat(spearmans):+.2f} | "
        f"low {np.mean(lows) * 100:+.2f}% vs high {np.mean(highs) * 100:+.2f}%"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--samples", type=Path, default=DEFAULT_SAMPLES)
    ap.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    args = ap.parse_args()

    df = load_cells(args.samples)
    closes = load_closes(df["symbol"].unique(), args.cache_dir)
    print(f"{len(df)} skew cells, {df['symbol'].nunique()} symbols, "
          f"{df['date'].min().date()}..{df['date'].max().date()}")
    oos_start = pd.Timestamp(OOS_START, tz="UTC")
    for horizon in HORIZONS:
        df["fwd"] = [forward_return(closes, s, d, horizon) for s, d in zip(df["symbol"], df["date"], strict=False)]
        usable = df.dropna(subset=["fwd"])
        study(usable, horizon, "FULL 2019-2025")
        study(usable[usable["date"] >= oos_start], horizon, "OOS 2021-07+")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
