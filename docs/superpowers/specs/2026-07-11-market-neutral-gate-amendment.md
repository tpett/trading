# R6 Stage 1 — The Market-Neutral (Long/Short) Gate (pre-registered amendment)

**Status:** pre-registered 2026-07-11, BEFORE any run. Motivated by a decision
to design for a shorting-capable API venue (IBKR-class). Amends the engine's
evaluation to gate the LONG/SHORT (market-neutral) series as a tradeable
strategy — the construction the long-only Robinhood constraint forbade and the
engine has silently computed all along.

## 1. Why

Every gate to date (R1 `--long-only`, R3, R5 concentration) scored the
long-only `lo` series, because the live account cannot short. But the engine's
`portfolio_sort` has always produced the `ls` series (`mean(top) − mean(bottom)`)
as the 4F-alpha input. On a shorting-capable venue that series is a **tradeable
market-neutral strategy**. Critically: `amihud` is the program's lone BH-FDR
survivor (L/S alpha t=8.34 midcap, DSR 0.999) and was refuted **only in its
long-only tradeable form** (walk-forward 0.16). Its market-neutral form — long
the illiquid names, short the liquid ones — has never been evaluated as a
tradeable, cost-charged, out-of-sample strategy. This is the single most direct
test of "what does shorting buy us."

## 2. The frozen construction

- **Series:** the engine's long/short book — long the top bucket, short the
  bottom bucket, equal-weight, monthly rebalance, rank-exit (the existing `ls`
  machinery). At $12,500 a diversified L/S book cannot hold hundreds of names,
  so the pre-registered book is **concentrated: top-N long / bottom-N short,
  N ∈ {10, 20}** (reusing the R5 `top_n` axis, now applied symmetrically to
  both legs). The quintile L/S is also computed as a reference.
- **Costs (frozen, charged to the market-neutral daily series):**
  - **Both legs** pay the Corwin-Schultz half-spread on turnover (the existing
    R1 `cost_charged` machinery, applied to the long AND the short leg).
  - **Short-borrow cost:** each short name accrues a borrow rate on its
    notional, charged daily. Frozen model: a general-collateral floor of
    **0.5%/yr**, scaled up by illiquidity — `borrow_bps = clamp(50, 1500,
    50 + k·(percentile of Amihud-illiquidity of the shorted name))` so
    hard-to-borrow (illiquid) shorts cost more, capped at 15%/yr. (For amihud
    the short leg is the *liquid* names → cheap borrow, a genuine tailwind; the
    cost then concentrates in the *long* illiquid leg's spread, which we already
    know is punishing — this is the honest test.) `k` fixed so the median
    shorted name pays ~1%/yr.
- **Benchmark:** CASH (market-neutral has no market benchmark). The gate is an
  **absolute** risk-adjusted return: annualized Sharpe of the cost-charged
  market-neutral series.

## 3. The frozen gate — WITH error bars (the statistical-power fix)

The prior program's methodological gap was gating on a naked Sharpe point
estimate. This gate does not repeat it:

- **Promotion statistic:** cost-charged market-neutral annualized Sharpe **and
  its two-sided confidence interval** (via a stationary bootstrap on the daily
  series, and cross-checked with the closed-form Sharpe SE
  `≈ √((1 + ½·SR²)/T)`).
- **Gate:** a candidate is promotion-eligible ONLY if the **lower bound of the
  95% Sharpe CI is > 0** — i.e. the market-neutral edge is statistically
  distinguishable from zero after costs. A point Sharpe of 0.9 whose CI includes
  0 does NOT pass. This is applied out-of-sample (§4).
- BH-FDR continues across the journal on the L/S alpha p-values as before; DSR
  advisory. The once-only 2024+ holdout stays reserved and is spent only on an
  explicit developer decision.

## 4. Out-of-sample discipline

- **Sub-period halves** of discovery (both must show a CI lower bound > 0).
- **A true walk-forward** where feasible: train windows select the sign/params,
  stitched OOS test segments carry the verdict (mirrors the R5 concentration
  OOS test that killed that lead). An in-sample market-neutral Sharpe is
  necessary, not sufficient — the same overfitting caveat that killed
  concentration applies, now guarded by the CI + OOS.

## 5. The frozen read

- **Signals:** re-derive and rank the whole journaled ledger's market-neutral
  series; the **pre-registered primary hypothesis is `amihud`** (the lone
  L/S-significant survivor). Others are exploratory, BH-counted.
- **If cost-charged market-neutral amihud clears the gate (CI lower bound > 0)
  out of sample** → shorting genuinely unlocks a tradeable edge the long-only
  constraint buried; it then earns the battery and, only on a developer
  decision, the holdout.
- **If it does not** (the honest base case: the long illiquid leg's spread eats
  the L/S alpha, exactly as it ate the long-only form) → then shorting does NOT
  rescue our one real factor, and "long-only was the constraint" is falsified —
  the constraint was costs, not the short leg. Either result is decisive and
  worth knowing.

## 6. Build scope

- A `--market-neutral` (long/short) leaderboard/eval path re-deriving the `ls`
  series from journaled params (incl. `top_n`), charging both legs' CS spread +
  the frozen borrow model, computing Sharpe + a bootstrap CI. Reuses R1
  `cost_charged`, the R5 `top_n` axis, and the sort machinery. Additive,
  default-off; existing long-only/quintile gates and the golden test unchanged.
- Tests: both-legs cost charging, the borrow model, the Sharpe CI (hand-checked
  against the closed-form SE), the OOS sub-period + walk-forward reads, and the
  default-off bit-identity guarantee.

## 7. Out of scope

The multi-leg options / straddle / VRP track (R6 Stage 2, separate spec — starts
with an IV-crush execution-cost data study); the live IBKR API integration
(none built); spending the holdout.
