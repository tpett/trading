"""Part B of the clean PEAD test (spec: docs/superpowers/specs/2026-07-11-clean-pead-test.md).

Consumes the real (symbol, earnings_date) table from Part A
(scripts/research/fetch_earnings_dates.py) plus survivorship-clean tiingo
bars, and runs the frozen §2-§4 event study: model-free earnings-day
surprise, market-adjusted forward drift entered AFTER the reaction (t0
close), surprise terciles x liquidity thirds x momentum terciles, with/
without COVID, largecap/midcap split, mean AND median with 95%
stationary-bootstrap CIs.

Data-only research script. No repo engine changes. Not committed.

Usage:
    .venv/bin/python scripts/research/pead_event_study.py \
        --events /path/to/earnings_dates.parquet \
        --out-dir /path/to/output_dir
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
LARGECAP_DIR = ROOT / "data" / "equities-tiingo"
MIDCAP_DIR = ROOT / "data" / "equities-midcap-tiingo"
SPY_PATH = LARGECAP_DIR / "SPY.parquet"

HORIZONS = (5, 21, 42, 63)
MOM_START, MOM_END = 252, 21  # 12-1 month momentum: t0-252 .. t0-21
LIQ_WINDOW = 63  # trailing dollar-volume window, ending t0-1
COVID_START = dt.date(2020, 2, 15)
COVID_END = dt.date(2020, 4, 30)
N_BOOT = 2000
BOOT_BLOCK = 5  # mean block length (events), preserves local event-clustering correlation
SEED = 42

_bars_cache: dict[str, pd.Series | None] = {}
_vol_cache: dict[str, pd.Series | None] = {}
_pool_cache: dict[str, str | None] = {}


def _date_indexed_close(path: Path) -> pd.Series:
    df = pd.read_parquet(path, columns=["close"])
    idx = df.index
    if idx.tz is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    dates = pd.DatetimeIndex(idx).normalize().date
    s = pd.Series(df["close"].to_numpy(), index=pd.Index(dates))
    s = s[~s.index.duplicated(keep="last")].sort_index()
    return s


def _date_indexed_dollar_volume(path: Path) -> pd.Series:
    df = pd.read_parquet(path, columns=["close", "volume"])
    idx = df.index
    if idx.tz is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    dates = pd.DatetimeIndex(idx).normalize().date
    dv = (df["close"] * df["volume"]).to_numpy()
    s = pd.Series(dv, index=pd.Index(dates))
    s = s[~s.index.duplicated(keep="last")].sort_index()
    return s


def symbol_pool_and_path(symbol: str) -> tuple[str | None, Path | None]:
    """largecap (equities-tiingo) wins on overlap; documented data caveat --
    120 symbols have bars cached under BOTH pools (multi-index membership
    history), and we do not attempt point-in-time index-membership
    reclassification here."""
    lc = LARGECAP_DIR / f"{symbol}.parquet"
    mc = MIDCAP_DIR / f"{symbol}.parquet"
    if lc.exists():
        return "largecap", lc
    if mc.exists():
        return "midcap", mc
    return None, None


def get_close_series(symbol: str) -> pd.Series | None:
    if symbol in _bars_cache:
        return _bars_cache[symbol]
    pool, path = symbol_pool_and_path(symbol)
    _pool_cache[symbol] = pool
    if path is None:
        _bars_cache[symbol] = None
        return None
    s = _date_indexed_close(path)
    _bars_cache[symbol] = s
    return s


def get_dollar_volume_series(symbol: str) -> pd.Series | None:
    if symbol in _vol_cache:
        return _vol_cache[symbol]
    _, path = symbol_pool_and_path(symbol)
    if path is None:
        _vol_cache[symbol] = None
        return None
    s = _date_indexed_dollar_volume(path)
    _vol_cache[symbol] = s
    return s


def nearest_calendar_pos(target: dt.date, calendar: np.ndarray) -> int | None:
    """Position in a sorted array of dates nearest to `target` (ties -> the
    LATER date, i.e. the next trading session)."""
    pos = np.searchsorted(calendar, target)
    if pos == 0:
        return 0
    if pos >= len(calendar):
        return len(calendar) - 1
    before = calendar[pos - 1]
    after = calendar[pos]
    if (target - before) < (after - target):
        return pos - 1
    return pos  # tie or after is closer -> forward


def lookup_close(series: pd.Series, date: dt.date, tolerance_days: int = 3) -> float:
    """Exact-date close, or nearest PRIOR trading day within tolerance (halts /
    minor calendar mismatch). NaN if nothing within tolerance."""
    if date in series.index:
        return float(series.loc[date])
    idx = series.index.to_numpy()
    pos = np.searchsorted(idx, date)
    # look backward for the nearest prior session within tolerance
    if pos > 0:
        prior = idx[pos - 1]
        if (date - prior).days <= tolerance_days:
            return float(series.iloc[pos - 1])
    return float("nan")


def build_events(events_path: Path) -> pd.DataFrame:
    events = pd.read_parquet(events_path)
    events["earnings_date"] = pd.to_datetime(events["earnings_date"]).dt.date

    spy_close = _date_indexed_close(SPY_PATH)
    spy_calendar = spy_close.index.to_numpy()

    rows = []
    missing_bars = []
    for rec in events.itertuples(index=False):
        symbol = rec.symbol
        stock_close = get_close_series(symbol)
        pool = _pool_cache.get(symbol)
        if stock_close is None:
            missing_bars.append(symbol)
            continue
        stock_vol = get_dollar_volume_series(symbol)

        raw_t0 = rec.earnings_date
        pos = nearest_calendar_pos(raw_t0, spy_calendar)
        t0 = spy_calendar[pos]

        t0_close = lookup_close(stock_close, t0)
        prior_date = spy_calendar[pos - 1] if pos - 1 >= 0 else None
        prior_close = lookup_close(stock_close, prior_date) if prior_date else float("nan")
        if np.isnan(t0_close) or np.isnan(prior_close) or prior_close == 0:
            continue
        reaction = t0_close / prior_close - 1.0

        # momentum: t0-252 .. t0-21 (trailing, excludes t0)
        momentum = float("nan")
        if pos - MOM_START >= 0:
            mstart_date = spy_calendar[pos - MOM_START]
            mend_date = spy_calendar[pos - MOM_END]
            c0 = lookup_close(stock_close, mstart_date)
            c1 = lookup_close(stock_close, mend_date)
            if not np.isnan(c0) and not np.isnan(c1) and c0 != 0:
                momentum = c1 / c0 - 1.0

        # liquidity: trailing 63d median dollar volume, ending t0-1 (excludes t0)
        liquidity = float("nan")
        if stock_vol is not None and pos - LIQ_WINDOW >= 0:
            window_dates = spy_calendar[pos - LIQ_WINDOW : pos]
            vals = [stock_vol.loc[d] for d in window_dates if d in stock_vol.index]
            if len(vals) >= LIQ_WINDOW // 2:  # require at least half the window present
                liquidity = float(np.nanmedian(vals))

        row = {
            "symbol": symbol,
            "pool": pool,
            "earnings_date_raw": raw_t0,
            "t0": t0,
            "t0_shifted": t0 != raw_t0,
            "reaction": reaction,
            "momentum_12_1": momentum,
            "liquidity_dollar_vol": liquidity,
        }
        for h in HORIZONS:
            fwd_pos = pos + h
            if fwd_pos >= len(spy_calendar):
                row[f"drift_{h}"] = float("nan")
                continue
            fwd_date = spy_calendar[fwd_pos]
            stock_fwd = lookup_close(stock_close, fwd_date)
            spy_fwd = float(spy_close.loc[fwd_date])
            spy_t0 = float(spy_close.loc[t0])
            if np.isnan(stock_fwd):
                row[f"drift_{h}"] = float("nan")
            else:
                stock_ret = stock_fwd / t0_close - 1.0
                spy_ret = spy_fwd / spy_t0 - 1.0
                row[f"drift_{h}"] = stock_ret - spy_ret
        rows.append(row)

    df = pd.DataFrame(rows)
    print(f"Missing bars entirely for {len(set(missing_bars))} symbols: {sorted(set(missing_bars))[:30]}", file=sys.stderr)
    print(f"Events with usable reaction: {len(df)} / {len(events)} raw events", file=sys.stderr)
    n_shifted = int(df["t0_shifted"].sum())
    print(f"Events where t0 was shifted off a weekend/holiday to nearest trading day: {n_shifted}", file=sys.stderr)
    return df


def bucket_terciles(values: pd.Series) -> tuple[float, float]:
    """(33rd, 67th) percentile breakpoints, ignoring NaN."""
    v = values.dropna()
    return float(np.percentile(v, 33.333)), float(np.percentile(v, 66.667))


def label_tercile(value: float, lo: float, hi: float, labels=("T1", "T2", "T3")) -> str | float:
    if pd.isna(value):
        return float("nan")
    if value <= lo:
        return labels[0]
    if value <= hi:
        return labels[1]
    return labels[2]


# --- stationary bootstrap (Politis & Romano 1994), reimplemented standalone
# for this self-contained research script (same distribution as
# trading.alphasearch.stats._stationary_bootstrap_indices -- a sequence of
# blocks, each starting at a uniform-random position and running for a
# GEOMETRIC(1/block) length, wrapping circularly): applied to the event
# sequence SORTED BY t0 DATE so temporally-clustered events (e.g. many names
# reporting the same week) keep their local correlation structure under
# resampling, rather than being treated as iid draws.
#
# Vectorized across all n_boot replicates at once (one (n_boot, n) index
# matrix built via n sequential steps -- "continue previous block (prob
# 1-p)" vs "restart at a uniform random position (prob p)" -- each step
# vectorized over n_boot with numpy, not a per-replicate Python loop). This
# is exactly the same geometric-block-length stationary bootstrap as the
# reference _stationary_bootstrap_indices, just generated breadth-first
# across replicates instead of one replicate at a time: a pure-Python loop
# of n_boot=2000 replicates x n up to several thousand events per cell was
# too slow (>10min and still on the first COVID variant) for this study's
# ~700+ cells; this form is O(n) numpy-vectorized steps per cell instead of
# O(n_boot) Python-level while-loops per cell.


def _stationary_bootstrap_index_matrix(
    n: int, block: int, n_boot: int, rng: np.random.Generator
) -> np.ndarray:
    if n == 0:
        return np.zeros((n_boot, 0), dtype=np.int64)
    p = 1.0 / block
    idx = np.empty((n_boot, n), dtype=np.int64)
    idx[:, 0] = rng.integers(0, n, size=n_boot)
    if n > 1:
        restart = rng.random((n_boot, n - 1)) < p
        random_starts = rng.integers(0, n, size=(n_boot, n - 1))
        for j in range(1, n):
            cont = (idx[:, j - 1] + 1) % n
            idx[:, j] = np.where(restart[:, j - 1], random_starts[:, j - 1], cont)
    return idx


def bootstrap_stats(
    values: pd.Series,
    dates: pd.Series,
    n_boot: int = N_BOOT,
    block: int = BOOT_BLOCK,
    seed: int = SEED,
) -> dict:
    """mean AND median point + 95% stationary-bootstrap CI in one pass (one
    shared index matrix -> one gather -> both statistics), over the
    time-ordered event sequence."""
    d = pd.DataFrame({"v": values, "t": dates}).dropna(subset=["v"]).sort_values("t")
    v = d["v"].to_numpy()
    n = len(v)
    out = {"n": n, "mean": float("nan"), "mean_ci_lo": float("nan"), "mean_ci_hi": float("nan"),
           "median": float("nan"), "median_ci_lo": float("nan"), "median_ci_hi": float("nan")}
    if n == 0:
        return out
    out["mean"] = float(v.mean())
    out["median"] = float(np.median(v))
    if n < 3:
        return out
    rng = np.random.default_rng(seed)
    idx = _stationary_bootstrap_index_matrix(n, min(block, n), n_boot, rng)
    samples = v[idx]  # (n_boot, n)
    boot_means = samples.mean(axis=1)
    boot_medians = np.median(samples, axis=1)
    out["mean_ci_lo"], out["mean_ci_hi"] = (float(x) for x in np.percentile(boot_means, [2.5, 97.5]))
    out["median_ci_lo"], out["median_ci_hi"] = (float(x) for x in np.percentile(boot_medians, [2.5, 97.5]))
    return out


def bootstrap_spread_stats(
    values_hi: pd.Series,
    dates_hi: pd.Series,
    values_lo: pd.Series,
    dates_lo: pd.Series,
    n_boot: int = N_BOOT,
    block: int = BOOT_BLOCK,
    seed: int = SEED,
) -> dict:
    """mean AND median spread (stat(hi) - stat(lo)) + 95% CI, each side
    block-bootstrapped independently over its own time-ordered sequence per
    replicate."""
    dh = pd.DataFrame({"v": values_hi, "t": dates_hi}).dropna(subset=["v"]).sort_values("t")
    dl = pd.DataFrame({"v": values_lo, "t": dates_lo}).dropna(subset=["v"]).sort_values("t")
    vh, vl = dh["v"].to_numpy(), dl["v"].to_numpy()
    nh, nl = len(vh), len(vl)
    out = {"n_hi": nh, "n_lo": nl, "mean": float("nan"), "mean_ci_lo": float("nan"), "mean_ci_hi": float("nan"),
           "median": float("nan"), "median_ci_lo": float("nan"), "median_ci_hi": float("nan")}
    if nh == 0 or nl == 0:
        return out
    out["mean"] = float(vh.mean() - vl.mean())
    out["median"] = float(np.median(vh) - np.median(vl))
    if nh < 3 or nl < 3:
        return out
    rng = np.random.default_rng(seed)
    idx_h = _stationary_bootstrap_index_matrix(nh, min(block, nh), n_boot, rng)
    idx_l = _stationary_bootstrap_index_matrix(nl, min(block, nl), n_boot, rng)
    sh, sl = vh[idx_h], vl[idx_l]
    boot_mean_spread = sh.mean(axis=1) - sl.mean(axis=1)
    boot_median_spread = np.median(sh, axis=1) - np.median(sl, axis=1)
    out["mean_ci_lo"], out["mean_ci_hi"] = (float(x) for x in np.percentile(boot_mean_spread, [2.5, 97.5]))
    out["median_ci_lo"], out["median_ci_hi"] = (float(x) for x in np.percentile(boot_median_spread, [2.5, 97.5]))
    return out


def summarize_cell(df: pd.DataFrame, horizon_col: str) -> dict:
    return bootstrap_stats(df[horizon_col], df["t0"])


def fmt_pct(x: float) -> str:
    return "nan" if pd.isna(x) else f"{x * 100:+.2f}%"


def fmt_cell(c: dict) -> str:
    n = 0 if pd.isna(c["n"]) else int(c["n"])
    return (
        f"n={n:4d}  mean={fmt_pct(c['mean']):>8s} "
        f"[{fmt_pct(c['mean_ci_lo']):>8s},{fmt_pct(c['mean_ci_hi']):>8s}]   "
        f"median={fmt_pct(c['median']):>8s} "
        f"[{fmt_pct(c['median_ci_lo']):>8s},{fmt_pct(c['median_ci_hi']):>8s}]"
    )


def run_analysis(df: pd.DataFrame, label: str, out_dir: Path) -> pd.DataFrame:
    """Runs the full §4 battery on `df` (already COVID-filtered or not) and
    returns a long-format results table; also prints the frozen-decision-rule
    focused view to stdout."""
    print(f"\n{'=' * 100}\n{label}  (n_events={len(df)})\n{'=' * 100}")

    # --- global bucket breakpoints (computed ONCE on this df's positive/valid
    # subset, held fixed for every slice below so pool/liquidity/momentum
    # buckets mean the same thing across all cuts within this COVID variant).
    pos = df[df["reaction"] > 0].copy()
    neg = df[df["reaction"] < 0].copy()
    surprise_lo, surprise_hi = bucket_terciles(pos["reaction"])
    mom_lo, mom_hi = bucket_terciles(df["momentum_12_1"])
    liq_lo, liq_hi = bucket_terciles(df["liquidity_dollar_vol"])

    pos["surprise_tercile"] = pos["reaction"].apply(lambda v: label_tercile(v, surprise_lo, surprise_hi))
    df["momentum_tercile"] = df["momentum_12_1"].apply(
        lambda v: label_tercile(v, mom_lo, mom_hi, labels=("LOW", "MID", "HIGH"))
    )
    df["liquidity_tercile"] = df["liquidity_dollar_vol"].apply(
        lambda v: label_tercile(v, liq_lo, liq_hi, labels=("LOW", "MID", "HIGH"))
    )
    # re-merge tercile columns onto pos/neg via index alignment (pos/neg are
    # row-subsets of df, so just reuse df's computed columns via the index)
    pos["momentum_tercile"] = df.loc[pos.index, "momentum_tercile"]
    pos["liquidity_tercile"] = df.loc[pos.index, "liquidity_tercile"]
    neg["momentum_tercile"] = df.loc[neg.index, "momentum_tercile"]
    neg["liquidity_tercile"] = df.loc[neg.index, "liquidity_tercile"]

    print(
        f"breakpoints: surprise(pos) 33/67 = {surprise_lo * 100:.2f}%/{surprise_hi * 100:.2f}%  "
        f"momentum 33/67 = {mom_lo * 100:.1f}%/{mom_hi * 100:.1f}%  "
        f"liquidity($vol) 33/67 = {liq_lo:,.0f}/{liq_hi:,.0f}"
    )
    print(f"positive-surprise events: {len(pos)}   negative-surprise events: {len(neg)}")

    long_rows: list[dict] = []

    pools = ["ALL", "largecap", "midcap"]
    for pool_name in pools:
        pos_p = pos if pool_name == "ALL" else pos[pos["pool"] == pool_name]
        neg_p = neg if pool_name == "ALL" else neg[neg["pool"] == pool_name]
        df_p = df if pool_name == "ALL" else df[df["pool"] == pool_name]

        for h in HORIZONS:
            col = f"drift_{h}"

            # (1) surprise-tercile x liquidity-third table, positive leg
            for st in ["T1", "T2", "T3"]:
                for lt in ["LOW", "MID", "HIGH"]:
                    cell = pos_p[(pos_p["surprise_tercile"] == st) & (pos_p["liquidity_tercile"] == lt)]
                    c = summarize_cell(cell, col)
                    long_rows.append(
                        {"cut": "surprise_x_liquidity", "pool": pool_name, "horizon": h,
                         "surprise_tercile": st, "liquidity_tercile": lt, "momentum_tercile": "ALL", **c}
                    )

            # (2) T3-T1 spread (all liquidity), positive leg
            t1 = pos_p[pos_p["surprise_tercile"] == "T1"]
            t3 = pos_p[pos_p["surprise_tercile"] == "T3"]
            sp = bootstrap_spread_stats(t3[col], t3["t0"], t1[col], t1["t0"])
            long_rows.append(
                {"cut": "T3_minus_T1_spread", "pool": pool_name, "horizon": h, **sp}
            )

            # (3) neutral-momentum-tercile drift, positive leg (all surprise terciles pooled, and per-tercile)
            neutral = pos_p[pos_p["momentum_tercile"] == "MID"]
            c = summarize_cell(neutral, col)
            long_rows.append(
                {"cut": "neutral_momentum_positive_leg", "pool": pool_name, "horizon": h,
                 "surprise_tercile": "ALL", "liquidity_tercile": "ALL", "momentum_tercile": "MID", **c}
            )
            for st in ["T1", "T2", "T3"]:
                cell = pos_p[(pos_p["surprise_tercile"] == st) & (pos_p["momentum_tercile"] == "MID")]
                c = summarize_cell(cell, col)
                long_rows.append(
                    {"cut": "neutral_momentum_by_surprise_tercile", "pool": pool_name, "horizon": h,
                     "surprise_tercile": st, "liquidity_tercile": "ALL", "momentum_tercile": "MID", **c}
                )

            # (4) surprise x momentum cross, positive leg
            for st in ["T1", "T2", "T3"]:
                for mt in ["LOW", "MID", "HIGH"]:
                    cell = pos_p[(pos_p["surprise_tercile"] == st) & (pos_p["momentum_tercile"] == mt)]
                    c = summarize_cell(cell, col)
                    long_rows.append(
                        {"cut": "surprise_x_momentum", "pool": pool_name, "horizon": h,
                         "surprise_tercile": st, "liquidity_tercile": "ALL", "momentum_tercile": mt, **c}
                    )

            # (5) negative-surprise leg: overall, momentum-neutral, and x liquidity
            c = summarize_cell(neg_p, col)
            long_rows.append(
                {"cut": "negative_leg_overall", "pool": pool_name, "horizon": h,
                 "surprise_tercile": "NEG", "liquidity_tercile": "ALL", "momentum_tercile": "ALL", **c}
            )
            neg_neutral = neg_p[neg_p["momentum_tercile"] == "MID"]
            c = summarize_cell(neg_neutral, col)
            long_rows.append(
                {"cut": "negative_leg_momentum_neutral", "pool": pool_name, "horizon": h,
                 "surprise_tercile": "NEG", "liquidity_tercile": "ALL", "momentum_tercile": "MID", **c}
            )
            for lt in ["LOW", "MID", "HIGH"]:
                cell = neg_p[neg_p["liquidity_tercile"] == lt]
                c = summarize_cell(cell, col)
                long_rows.append(
                    {"cut": "negative_leg_x_liquidity", "pool": pool_name, "horizon": h,
                     "surprise_tercile": "NEG", "liquidity_tercile": lt, "momentum_tercile": "ALL", **c}
                )

    results = pd.DataFrame(long_rows)
    out_path = out_dir / f"pead_results_{label}.csv"
    results.to_csv(out_path, index=False)
    print(f"Wrote full results table ({len(results)} rows) to {out_path}")

    # --- focused decision-rule printout: 42d/63d, ALL pool + per-pool
    print("\n--- (c) T3 x HIGH-liquidity, positive leg (42d/63d) ---")
    for pool_name in pools:
        for h in (42, 63):
            row = results[
                (results["cut"] == "surprise_x_liquidity") & (results["pool"] == pool_name)
                & (results["horizon"] == h) & (results["surprise_tercile"] == "T3")
                & (results["liquidity_tercile"] == "HIGH")
            ]
            if len(row):
                r = row.iloc[0].to_dict()
                print(f"  [{pool_name:>9s}] {h}d: {fmt_cell(r)}")

    print("\n--- T3-T1 monotonicity spread, positive leg, all liquidity (42d/63d) ---")
    for pool_name in pools:
        for h in (42, 63):
            row = results[
                (results["cut"] == "T3_minus_T1_spread") & (results["pool"] == pool_name)
                & (results["horizon"] == h)
            ]
            if len(row):
                r = row.iloc[0]
                for stat in ["mean", "median"]:
                    print(
                        f"  [{pool_name:>9s}] {h}d {stat:>6s}: spread={fmt_pct(r[stat])} "
                        f"[{fmt_pct(r[f'{stat}_ci_lo'])},{fmt_pct(r[f'{stat}_ci_hi'])}]  "
                        f"n_t3={r['n_hi']:.0f} n_t1={r['n_lo']:.0f}"
                    )

    print("\n--- (e) neutral-momentum-tercile drift, positive leg (42d/63d) ---")
    for pool_name in pools:
        for h in (42, 63):
            row = results[
                (results["cut"] == "neutral_momentum_positive_leg") & (results["pool"] == pool_name)
                & (results["horizon"] == h)
            ]
            if len(row):
                print(f"  [{pool_name:>9s}] {h}d: {fmt_cell(row.iloc[0].to_dict())}")

    print("\n--- (f) negative-surprise leg, overall + momentum-neutral (42d/63d) ---")
    for pool_name in pools:
        for h in (42, 63):
            for cut in ["negative_leg_overall", "negative_leg_momentum_neutral"]:
                row = results[
                    (results["cut"] == cut) & (results["pool"] == pool_name) & (results["horizon"] == h)
                ]
                if len(row):
                    print(f"  [{pool_name:>9s}] {h}d {cut:>30s}: {fmt_cell(row.iloc[0].to_dict())}")

    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = build_events(args.events)
    df.to_csv(args.out_dir / "pead_events_full.csv", index=False)

    covid_mask = df["t0"].apply(lambda d: COVID_START <= d <= COVID_END)
    print(f"\nEvents inside COVID window [{COVID_START}, {COVID_END}]: {int(covid_mask.sum())}", file=sys.stderr)

    run_analysis(df.copy(), "with_covid", args.out_dir)
    run_analysis(df[~covid_mask].copy(), "ex_covid", args.out_dir)


if __name__ == "__main__":
    main()
