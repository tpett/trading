# R5 — The Concentration Axis (pre-registered amendment)

**Status:** pre-registered 2026-07-11, BEFORE any run, in response to an
independent adversarial audit (recorded in the session narrative) of the
R1/R2/R3 negative verdict. Amends the alpha-search engine to add a
concentration axis and re-tests the momentum verdict at the construction a
~$1k account actually uses.

## 1. Why (the audit's central finding)

Every "long-only vs SPY" number in the program (R1/R2/R3, 1,314 of 1,326
journaled trials) was a **top-QUINTILE equal-weight** book — 20 to *hundreds*
of names — which by construction converges to market beta (R2's bare momentum
Sharpe 0.96 = SPY). The account holds ~5–20 names. Concentration was never a
search axis. R3's capacity-edge thesis ("a $1k account can hold a concentrated
handful of the best names institutions can't accumulate") was tested with the
*opposite* construction — a hundreds-name diluted quintile — so its "momentum
ties SPY in down-cap" result is confounded by construction. Momentum's premium
is tail-concentrated (winner decile ≫ quintile); the diluted book averages the
tail away. **The recorded negative verdict is therefore narrower than stated:
it is "monthly, equal-weight top-quintile, single-signal long-only doesn't beat
SPY on 2019–2023," not "no long-only construction beats SPY."** This amendment
tests the one construction that matters most and was never run.

## 2. The frozen construction (new)

- **Top-N long-only book:** equal-weight mean of the **N highest-signal-score
  names** in the tradeable cross-section, at each monthly decision date, held
  to the next (rank-exit) — identical machinery to the existing `lo` quintile
  series, but a fixed COUNT of names instead of the top 1/quantiles fraction.
  (Fixed count ≠ top-decile: a decile of the 4,443-name down-cap universe is
  ~444 names, still diluted; N=10 is the account's real book.)
- **Frozen N values (pre-registered, no other N tested — N is a forking-paths
  hazard): N ∈ {10, 20}.** Both are realistic $1k-account book sizes.
- **Weighting: equal-weight only** (a $1k account holds ~equal fractional
  positions; rank/score weighting is out of scope for this amendment).
- **4F diagnostic:** the long/short series for the mandatory 4F regression is
  `mean(top-N) − mean(bottom-N)` (symmetric), unchanged in role (diagnostic,
  not gate).
- **Cross-section floor:** a decision date is skipped if fewer than **N**
  tradeable names exist (replaces the quintile's 15-name floor for top-N
  trials). Skips are hold-through, as today.
- **Trial identity:** `top_n` enters the hashed trial params, so top-N trials
  are distinct journaled trials with their own config hashes (never collide
  with the quintile trials).

## 3. The frozen test (pre-registered hypotheses)

- **Primary signal:** `mom_12_2` (classic 12-minus-1-month academic momentum).
  **Secondary:** `mom252` (12-month). No other signals in THIS amendment (a
  full-battery concentration sweep would be a separate pre-registration).
- **Universes:** `largecap` (the account can actually trade these; the clean
  tradeable case) and `downcap-dv` (the survivorship-free dollar-volume
  universe from R3 — the capacity-edge thesis, now tested at true
  concentration).
- **Gate (unchanged, R1):** cost-charged (Corwin-Schultz spread) long-only
  top-N Sharpe **and** total return vs SPY buy-and-hold over the discovery
  window 2019-01-01..2023-12-31. 4F alpha printed as diagnostic. BH-FDR runs
  across the whole journal as always; these trials are counted.
- **Matrix:** 2 signals × 2 N × 2 universes = **8 pre-registered top-N
  trials**, read alongside the existing top-quintile baselines for the same
  signals/universes.

## 4. Pre-registered readings

- **If concentrated (top-10) momentum beats SPY where quintile only tied** —
  cost-charged long-only Sharpe ≥ SPY AND total > SPY — then the R3/R2
  "momentum ties SPY" conclusion was a construction artifact, and the
  concentrated book is a genuine candidate: it then earns the robustness
  battery and, only on a developer decision, the once-only holdout.
- **If top-10 and top-20 still only tie or trail SPY** (in-sample), the
  negative verdict is robust to concentration and stands — now actually
  tested, not assumed.
- **Honesty guard (unchanged):** a discovery-window beat is the overfitting
  surface (the R1 §13 re-read found ~45% of all trials beat SPY in-sample), so
  an in-sample top-N win is NECESSARY, not SUFFICIENT. No promotion without the
  battery; the 2024+ holdout stays reserved and is spent only on an explicit
  developer decision. Classical-OLS-SE caveat unchanged.

## 5. Build scope (small, additive)

- `sort.py`: a `top_n: int | None = None` param on `portfolio_sort` +
  `assign_quantiles` (or a sibling) producing top-N/bottom-N selection when
  set; the `lo`/`ls` series then mean the N names; the min-names floor becomes
  `max(min_names_default, top_n)` → effectively N for top-N trials. Default-off:
  `top_n=None` reproduces today's quintile behavior bit-for-bit.
- `sweep.py`: `top_n` threaded into `_hashed_params` (identity) + a way to
  request top-N trials (a `--top-n` sweep option, or a params override), on the
  frozen signals/universes.
- Leaderboard `--long-only`: re-derives top-N trials from their journaled
  params exactly as it does quintile trials (params carry `top_n`).
- Tests: top-N selection is hand-checked (N highest scores), the min-N skip,
  the hash distinctness, and the default-off bit-identity guard (top_n=None ≡
  today). No look-ahead change; the standing golden/PIT guarantees hold.

## 6. Out of scope

Rank/score weighting; other signals or a full-battery concentration sweep
(separate pre-registrations); the holding-horizon axis and PEAD signal (the
audit's other two recommendations — separate specs); spending the holdout.
