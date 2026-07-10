# R1 — The Long-Only Gate Amendment (written prospective amendment)

**Status:** approved direction (strategy audit 2026-07-10, developer-approved
sequence). Amends Piece 1 §5 and the Piece 3 promotion rule, prospectively:
no already-journaled trial is re-scored retroactively; the new gate applies
to all FUTURE promotion decisions, and the leaderboard gains a re-READ of
existing trials under the new lens (a display, not a re-journaling).

## 1. Why (the audit's central finding)

The engine's gate — four-factor LONG/SHORT alpha — certifies returns in a
construction this account cannot trade, after stripping out the four premia
a long-only investor actually collects. Twice the program proved a signal
"real" under that gate and watched it die in tradeable form. "Beating the
market" from this seat means: the long-only portfolio, after realistic
costs, outperforms SPY. That is what the gate must measure.

## 2. The amended gate (frozen)

- **Promotion statistic:** the signal's LONG-ONLY top-quantile portfolio
  (the engine's existing `lo` series: equal weight, monthly, quintiles),
  charged spread-based costs (§3), evaluated over the discovery window:
  **annualized Sharpe and total return vs SPY buy-and-hold over the
  identical window.** A candidate is promotion-eligible when cost-charged
  long-only Sharpe ≥ SPY's over discovery AND total return > SPY's.
- **The 4F regression is retained as a mandatory DIAGNOSTIC** (printed with
  every candidate: alpha, loadings, R²) — know what you're being paid for —
  but no longer a filter. The BH-FDR machinery continues to run on the L/S
  alpha p-values exactly as before (the multiple-testing record and its
  honesty rules are untouched); BH survivorship becomes a reported property,
  not the promotion gate.
- **The robustness battery still gates the holdout**, with its checks
  re-anchored to the long-only cost-charged series (same frozen thresholds;
  the 30bps cost row is superseded by the §3 spread model at the account's
  book size). *[Dated note, 2026-07-10 — ratified statistic: the re-anchor
  target is the COST-CHARGED LO-MINUS-SPY ACTIVE RETURN series, not the raw
  `lo` series. Checks 1-5 score the signed retention of its annualized
  value; check 1's |t| ≥ 1.0 is the active series' mean/se t; check 6's
  month concentration runs on its monthly log returns. Raw-LO retention
  would mostly test whether the market regime repeated (beta), not the
  signal's edge over the benchmark. See Ratifications below.]*
- The once-only holdout, trial journal, pre-registration discipline, and
  PIT machinery are all unchanged.

## 3. Spread-based cost model (frozen)

- Per-name effective spread estimated from daily OHLC via the
  **Corwin-Schultz (2012) high-low estimator** (two-day HL ratios; negative
  estimates floored at 0), averaged over the trailing 21 sessions, floored
  at 2 bps (large-cap reality) and capped at 5% (data sanity).
- Cost per rebalance per name = half-spread × turnover participation (the
  existing turnover machinery), charged to the `lo` series on the first
  return day after each decision date. Book-size impact at $1k is
  negligible by construction (fractional shares, no commissions) and is NOT
  modeled beyond the spread — documented, revisit if the account grows.
- The estimator gets hand-computable unit fixtures + a sanity check against
  known-liquid names (AAPL estimated spread must land in single-digit bps).

## 4. Deliverables

1. `evaluate.py`/`robustness.py`: Corwin-Schultz estimator + `lo`-series
   cost charging (reuses `apply_rebalance_charges` shape with per-name
   spreads instead of flat bps).
2. Leaderboard: a `--long-only` view ranking every journaled trial by
   cost-charged long-only Sharpe vs SPY (SPY series from the cached bars;
   computed at read time from the stored trial record + re-derived `lo`
   series where needed — trials whose data can't re-derive show honestly as
   n/a).
3. Battery promotion rule swap per §2; holdout pre-check text updated.
4. Spec §5 amendment notes in the Piece 1/Piece 3 docs + glossary update +
   experiments.md amendment record.
5. Tests per repo discipline; the standing no-look-ahead/golden guarantees
   unchanged.

## 5. Out of scope

Re-running any sweep (existing journal is re-READ, not re-run); R3
down-cap universes; R4 deployment; wrapper changes (R2 owns that).

## Ratifications (2026-07-10, orchestrator — recorded per the amend-only-in-writing rule)

1. **Battery re-anchor statistic (§2, third bullet):** the re-anchored
   checks 1-6 score the **cost-charged LO-minus-SPY active return series**
   (daily charged `lo` return minus SPY's daily return, inner-joined
   calendars), NOT the raw `lo` series, at the same frozen thresholds.
   Rationale: raw-LO retention across sub-periods, subsets, and jitter
   would predominantly test whether the market regime repeated — a beta
   property SPY exhibits too — rather than whether the signal's edge over
   the benchmark persisted, which is precisely what the §2 promotion
   statistic certifies. Checks 1-5 use the signed retention of the
   annualized active return; check 1's |t| ≥ 1.0 is the active series'
   mean/se t; check 6's ≤ 60% month concentration runs on the active
   series' monthly log returns.
2. **Corwin-Schultz component-averaging (§3, first bullet):** the
   implementation averages the β/γ two-day components over the trailing 21
   sessions and applies the (alpha → spread) transform once, DEVIATING from
   this section's literal per-day floor-then-average text (and from CS
   2012's own per-pair baseline). Ratified on the Monte Carlo evidence: on
   a simulated KNOWN-zero-spread price path, per-pair floor-then-average
   reads ~75bps of phantom spread out of pure noise while
   component-averaging reads ~12bps (a residual, conservative bias mostly
   absorbed by the 2bps floor) — and only the component-averaged form
   satisfies this section's own acceptance criterion (AAPL in single-digit
   bps). The omitted overnight-gap adjustment is mildly anti-conservative
   for gappy names and immaterial under the floor.
