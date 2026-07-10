# Options-v2 + Insider Batch — the combined pre-registered sweep (design spec)

**Status:** APPROVED by the developer 2026-07-09; sweep authorized. Data prerequisites are COMPLETE and verified: gather v2
(largecap 7,399 cells / midcap 11,235; IV median |Δ| = 0.0 vs v1 on ~50k
overlap legs; volume 100% both pools; OI 99.9% of legs; far blocks 81% / 50%)
and the Form 4 store (925k rows, spot-checked vs public filings).

## 1. Purpose

One pre-registered sweep covering the two Tier-2 signal families the new data
enables, plus the large-cap enablement of the two volume-gated Tier-1 options
signals. One leaderboard read. Registry goes 40 → 43.

## 2. New frozen signals (sign = higher more attractive)

| name | definition | sign rationale | universes |
|---|---|---|---|
| `oi_put_call` | log(1+OTM-put leg OI) − log(1+(ATM + OTM-call leg OI)) at the current cell | − heavy put OI positioning predicts negative returns (Fodor-Krieger-Doran family) | options universes |
| `d_oi` | Δ log(1+total near-leg OI) current vs prior month's cell (45d staleness, prior-cell machinery) | − rising single-name option OI predicts lower stock returns (Fodor-Krieger-Doran 2011) | options universes |
| `iv_term_slope` | far-block ATM IV − near ATM IV | + upward-sloping IV term structure = near-term calm (term-structure family, Vasquez 2017; the WEAKEST literature anchor in this batch — flagged exploratory) | options universes with far blocks |

All three: NaN when the required legs/blocks/prior cells are absent (never
imputed); `d_oi` reuses `option_row_prior`; `iv_term_slope` requires the far
block (mid-cap cross-sections will be ~half the pool — above the 15-name
floor, disclosed).

## 3. Already-frozen signals entering new territory

- `cp_vol`, `osv` (Tier-1 definitions unchanged): the v2 re-gather closed the
  large-cap leg-volume gap, so `requires_option_volume` now passes on
  large-cap options universes — their large-cap trials are NEW (new hashes).
- The 3 insider signals (`npr_90`, `cluster_buys_90`, `officer_buy_90`,
  frozen in the insider spec): first sweep anywhere.

## 4. Data-replacement disclosure (pre-registered honesty)

The re-gather REPLACED both samples files (v1 backed up beside them). Trial
identity is the universe NAME, so re-running existing options-family trials
journals under their EXISTING hashes — the leaderboard's numbers for those
trials update to v2-data values (hash-dedupe keeps the count honest; the
append-only journal preserves v1-era rows as history). This is the designed
cache-refresh semantics; the Piece 3 drift guard will refuse batteries whose
baseline no longer reproduces, forcing a re-sweep first — which this sweep
IS. Both v1 files are retained for forensics.

## 5. The sweep (after approval; one leaderboard read)

Same locked mechanics (discovery 2019-01-01..2023-12-31, quintiles, BH
q=0.10 across the whole journal, DSR advisory, holdout untouched). Passes:
(i) new + re-run options families across options universes; (ii) cp_vol/osv
large-cap enablement; (iii) insider family everywhere a store attaches
(~28 universes). Estimated ~110-130 genuinely new trials (journal ~835 →
~950-965) plus hash-replacements of existing options trials; the exact count
is whatever the journal records. Survivors (if any) require battery +
developer sign-off before any holdout, per standing rules.

## 6. Build scope (small)

Register the 3 new signals + tests (sign conventions, NaN/absence rules,
prior-cell staleness, far-block requirement); registry-count updates (43);
golden additions per family; no engine changes (all accessors exist).
