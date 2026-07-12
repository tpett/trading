#!/usr/bin/env python
"""
Data-only feasibility study: does selling a short OTM strangle (held to expiry)
harvest a positive expected P&L net of the option bid/ask spread?

Frozen methodology per task spec. Analysis only, no repo changes.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/Users/travis/Source/personal/trading")

POOLS = {
    "largecap": {
        "jsonl": REPO / "data/options-iv/samples.jsonl",
        "eq_dir": REPO / "data/equities-tiingo",
    },
    "midcap": {
        "jsonl": REPO / "data/options-iv/samples-midcap.jsonl",
        "eq_dir": REPO / "data/equities-midcap-tiingo",
    },
}

LIQ_MAX_SPREAD_PCT = 0.15
LIQ_MIN_OI = 100
SPLIT_GUARD = 0.6  # |S_exp/spot_at_decision - 1| > this -> drop as implausible

RNG_SEED = 42
N_BOOT = 2000


def load_equity_cache(eq_dir):
    cache = {}

    def get(symbol):
        if symbol in cache:
            return cache[symbol]
        path = eq_dir / f"{symbol}.parquet"
        if not path.exists():
            cache[symbol] = None
            return None
        df = pd.read_parquet(path, columns=["close_raw"])
        df = df.sort_index()
        cache[symbol] = df
        return df

    return get


def close_on_or_before(df, target_expiration_ts):
    sub = df.loc[:target_expiration_ts]
    if sub.empty:
        return None
    return float(sub["close_raw"].iloc[-1])


def process_pool(pool_name, jsonl_path, eq_dir):
    get_eq = load_equity_cache(eq_dir)

    rows = []
    n_total = 0
    n_skipped_missing = 0
    n_dropped_split_guard = 0

    with open(jsonl_path) as f:
        for line in f:
            n_total += 1
            rec = json.loads(line)
            symbol = rec["symbol"]
            spot = rec["spot_at_decision"]
            target_exp = rec["target_expiration"]

            contracts = {c["role"]: c for c in rec["contracts"]}
            if "otm_put" not in contracts or "otm_call" not in contracts:
                n_skipped_missing += 1
                continue
            put = contracts["otm_put"]
            call = contracts["otm_call"]

            df = get_eq(symbol)
            if df is None:
                n_skipped_missing += 1
                continue

            target_ts = pd.Timestamp(target_exp, tz="UTC")
            s_exp = close_on_or_before(df, target_ts)
            if s_exp is None:
                n_skipped_missing += 1
                continue

            # split / data-error guard
            if abs(s_exp / spot - 1) > SPLIT_GUARD:
                n_dropped_split_guard += 1
                continue

            credit_bid = put["bid"] + call["bid"]
            credit_mid = put["mid"] + call["mid"]
            payoff = max(0.0, put["strike"] - s_exp) + max(0.0, s_exp - call["strike"])

            pnl_bid = credit_bid - payoff
            pnl_mid = credit_mid - payoff
            pnl_ret_bid = pnl_bid / spot
            pnl_ret_mid = pnl_mid / spot

            # open_interest is occasionally absent from the source record (not just 0);
            # treat missing OI as unknown/illiquid -> fails the >=100 threshold.
            def leg_ok(c):
                if c["bid"] <= 0 or c["mid"] <= 0:
                    return False
                spread_pct = (c["ask"] - c["bid"]) / c["mid"]
                oi = c.get("open_interest", 0)
                return spread_pct <= LIQ_MAX_SPREAD_PCT and oi >= LIQ_MIN_OI

            tradeable = leg_ok(put) and leg_ok(call)

            put_spread_pct = (put["ask"] - put["bid"]) / put["mid"] if put["mid"] else np.nan
            call_spread_pct = (call["ask"] - call["bid"]) / call["mid"] if call["mid"] else np.nan

            rows.append(
                dict(
                    symbol=symbol,
                    decision_date=rec["decision_date"],
                    target_expiration=target_exp,
                    spot=spot,
                    s_exp=s_exp,
                    pnl_ret_bid=pnl_ret_bid,
                    pnl_ret_mid=pnl_ret_mid,
                    tradeable=tradeable,
                    put_spread_pct=put_spread_pct,
                    call_spread_pct=call_spread_pct,
                    put_oi=put.get("open_interest", np.nan),
                    call_oi=call.get("open_interest", np.nan),
                )
            )

    df_all = pd.DataFrame(rows)
    meta = dict(
        n_total=n_total,
        n_skipped_missing=n_skipped_missing,
        n_dropped_split_guard=n_dropped_split_guard,
        n_kept=len(df_all),
    )
    return df_all, meta


def bootstrap_ci_mean(x, n_boot=N_BOOT, seed=RNG_SEED):
    rng = np.random.default_rng(seed)
    x = np.asarray(x)
    n = len(x)
    if n == 0:
        return (np.nan, np.nan)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = x[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return (lo, hi)


def summarize(subset_name, x_bid, x_mid):
    x_bid = np.asarray(x_bid, dtype=float)
    x_mid = np.asarray(x_mid, dtype=float)
    n = len(x_bid)
    out = {"subset": subset_name, "n": n}
    if n == 0:
        return out

    mean_bid = x_bid.mean()
    med_bid = np.median(x_bid)
    std_bid = x_bid.std(ddof=1) if n > 1 else np.nan
    pct_profit = (x_bid > 0).mean() * 100

    ci_lo, ci_hi = bootstrap_ci_mean(x_bid)
    t_stat = mean_bid / (std_bid / np.sqrt(n)) if n > 1 and std_bid > 0 else np.nan

    mean_mid = x_mid.mean()
    spread_bite = mean_mid - mean_bid

    worst_1pct = np.percentile(x_bid, 1)
    worst_5pct = np.percentile(x_bid, 5)
    # mean excluding worst 5% of cells (by value, dropped from below)
    thresh = np.percentile(x_bid, 5)
    ex_worst5 = x_bid[x_bid > thresh]
    mean_ex_worst5 = ex_worst5.mean() if len(ex_worst5) else np.nan

    out.update(
        dict(
            mean_net=mean_bid,
            median_net=med_bid,
            std_net=std_bid,
            pct_profitable=pct_profit,
            ci95_lo=ci_lo,
            ci95_hi=ci_hi,
            t_stat=t_stat,
            mean_gross=mean_mid,
            spread_bite=spread_bite,
            worst_1pct=worst_1pct,
            worst_5pct=worst_5pct,
            mean_ex_worst5pct=mean_ex_worst5,
            annualized_x12=mean_bid * 12,
        )
    )
    return out


def fmt_pct(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    return f"{x * 100:.3f}%"


def fmt_num(x, d=3):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    return f"{x:.{d}f}"


def print_summary_block(pool_name, subset_name, s):
    print(f"\n--- {pool_name} / {subset_name} ---")
    print(f"  n = {s.get('n')}")
    if s.get("n", 0) == 0:
        print("  (no cells)")
        return
    print(f"  NET(at-bid) mean pnl_ret   = {fmt_pct(s['mean_net'])}")
    print(f"  NET(at-bid) median pnl_ret = {fmt_pct(s['median_net'])}")
    print(f"  NET(at-bid) std pnl_ret    = {fmt_pct(s['std_net'])}")
    print(f"  %% profitable               = {s['pct_profitable']:.2f}%")
    print(f"  bootstrap 95% CI on mean   = [{fmt_pct(s['ci95_lo'])}, {fmt_pct(s['ci95_hi'])}]")
    print(f"  t-stat                     = {fmt_num(s['t_stat'])}")
    print(f"  GROSS(at-mid) mean pnl_ret = {fmt_pct(s['mean_gross'])}")
    print(f"  spread_bite (gross-net)    = {fmt_pct(s['spread_bite'])}")
    print(f"  worst 1% pnl_ret           = {fmt_pct(s['worst_1pct'])}")
    print(f"  worst 5% pnl_ret           = {fmt_pct(s['worst_5pct'])}")
    print(f"  mean net EXCLUDING worst 5%= {fmt_pct(s['mean_ex_worst5pct'])}")
    print(f"  crude annualized (x12)     = {fmt_pct(s['annualized_x12'])}")


def main():
    all_results = {}
    for pool_name, cfg in POOLS.items():
        df_all, meta = process_pool(pool_name, cfg["jsonl"], cfg["eq_dir"])
        print(f"\n=== POOL: {pool_name} ===")
        print(f"  n_total_cells (jsonl lines)      = {meta['n_total']}")
        print(f"  n_skipped (missing expiry price) = {meta['n_skipped_missing']}")
        print(f"  n_dropped (split guard >|{SPLIT_GUARD}|) = {meta['n_dropped_split_guard']}")
        print(f"  n_kept (analyzed)                = {meta['n_kept']}")

        if len(df_all) == 0:
            continue

        n_tradeable = int(df_all["tradeable"].sum())
        n_illiquid = int((~df_all["tradeable"]).sum())
        print(f"  n_tradeable (liquidity filter passed) = {n_tradeable}")
        print(f"  n_illiquid (liquidity filter failed)  = {n_illiquid}")

        for subset_name, mask in [
            ("tradeable", df_all["tradeable"]),
            ("illiquid", ~df_all["tradeable"]),
        ]:
            sub = df_all[mask]
            s = summarize(subset_name, sub["pnl_ret_bid"].values, sub["pnl_ret_mid"].values)
            print_summary_block(pool_name, subset_name, s)
            all_results[(pool_name, subset_name)] = s

        # largecap-tradeable liquidity survival stats
        if pool_name == "largecap":
            trad = df_all[df_all["tradeable"]]
            print(f"\n  [largecap/tradeable liquidity-survival stats]")
            print(f"    median otm_put spread%  = {fmt_pct(trad['put_spread_pct'].median())}")
            print(f"    median otm_call spread% = {fmt_pct(trad['call_spread_pct'].median())}")
            print(f"    median otm_put OI       = {trad['put_oi'].median():.0f}")
            print(f"    median otm_call OI      = {trad['call_oi'].median():.0f}")

    # compact final table
    print("\n\n==================== COMPACT SUMMARY TABLE ====================")
    header = (
        f"{'pool':10s} {'subset':10s} {'n':>6s} {'net_mean':>10s} {'ci95_lo':>10s} {'ci95_hi':>10s} "
        f"{'t_stat':>8s} {'%prof':>8s} {'gross_mean':>11s} {'spread_bite':>12s} "
        f"{'worst5%':>9s} {'mean_ex5%':>10s} {'ann_x12':>9s}"
    )
    print(header)
    for (pool_name, subset_name), s in all_results.items():
        if s.get("n", 0) == 0:
            print(f"{pool_name:10s} {subset_name:10s} {0:6d} (no cells)")
            continue
        print(
            f"{pool_name:10s} {subset_name:10s} {s['n']:6d} "
            f"{fmt_pct(s['mean_net']):>10s} {fmt_pct(s['ci95_lo']):>10s} {fmt_pct(s['ci95_hi']):>10s} "
            f"{fmt_num(s['t_stat'],2):>8s} {s['pct_profitable']:7.2f}% "
            f"{fmt_pct(s['mean_gross']):>11s} {fmt_pct(s['spread_bite']):>12s} "
            f"{fmt_pct(s['worst_5pct']):>9s} {fmt_pct(s['mean_ex_worst5pct']):>10s} {fmt_pct(s['annualized_x12']):>9s}"
        )


if __name__ == "__main__":
    main()
