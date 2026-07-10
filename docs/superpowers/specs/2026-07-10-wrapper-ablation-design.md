# R2 — Strategy-Wrapper Ablation (pre-registered experiment protocol)

**Status:** approved direction (strategy audit 2026-07-10, developer-approved
sequence R2 → R1 → R3, R4 alongside). This spec pre-registers the ablation
BEFORE any run.
**Question:** how much of every "tradeable form" verdict is the signal, and
how much is the inherited momentum-era wrapper (regime gate, ATR stops,
entry-score threshold, per-window hyperparameter grid)? No bare portfolio has
ever been tested; the trailing-exit disaster (0.59→0.09) proves wrapper
components can dominate outcomes.

## 1. The ablation matrix (pre-registered; run once, all cells journaled)

Three signals with established life — `momentum_v1` (best honest 0.45),
`amihud_v1` (factor real, wrapper-form 0.16), `skew_v1` (§9's 63d premium) —
each run over the IDENTICAL stitched OOS span the historical verdicts used
(walk-forward windows, 2018 start, holdout auto-clamped), on their historical
universes (momentum: sp500+ndx; amihud: sp400 midcap; skew: options pool),
at 15 bps (midcap) / 5 bps (largecap) — the historical cost conventions,
held fixed so ONLY the wrapper varies:

| cell | regime gate | stops | entry threshold | grid |
|---|---|---|---|---|
| W0 (bare) | off | off | off (rank-only, top-N equal weight, monthly) | none |
| W1 | ON | off | off | none |
| W2 | off | ON | off | none |
| W3 | off | off | ON | historical grid |
| W4 (full, control) | ON | ON | ON | historical grid — must reproduce the recorded verdicts |

15 runs (3 signals × 5 cells). All journaled in
`journal/experiments-equities.jsonl` as usual; recorded in experiments.md
§13 regardless of outcome.

## 2. Pre-registered readings

- If W0(momentum) ≈ or > the recorded 0.45 → the wrapper was NOT the
  bottleneck; the momentum conclusion stands.
- If W0(momentum) materially exceeds 0.45 (≥ +0.1 Sharpe) → the record's
  central conclusion is partly a wrapper artifact; experiments.md gets a
  written correction.
- **Component attribution (amended pre-run, 2026-07-10, before any cell
  executed):** which component drags is read from **Wi vs W4** (each Wi
  differs from the full wrapper by exactly one named axis; time-stop and
  trend-break exits — unnamed engine components discovered during the build
  — are held constant across W1-W4 and are therefore excluded from
  attribution). W0 vs W4 measures the WHOLE wrapper including those unnamed
  components; W0-anchored per-component attribution is NOT valid with these
  cells and is not claimed. Known coupling, disclosed: W2 (stops on, regime
  off) cannot exercise the regime-flush stop — regime.disabled forces
  risk-on, so only the plain ATR stop and time-stop act there.
- Same logic per signal. W4 must reproduce the historical numbers within
  noise or the ablation harness itself is broken (STOP, debug, never
  reinterpret).
- No cell result may be promoted to live from this experiment alone — this
  is instrument calibration, not discovery. Promotion continues to require
  the (post-R1) gate + battery + holdout discipline.

## 3. Implementation (minimal)

Ablation flags must be CONFIG-ONLY if the engine already supports them
(regime disable, stop disable, threshold bypass, single-point grid);
otherwise the smallest possible engine change, additive and default-off,
with tests proving W4 ≡ the unmodified engine bit-for-bit on a fixture.
Config dirs `config/experiments/ablation-<signal>-<cell>/` generated from
the historical experiment configs, changing ONLY wrapper flags. Bare mode
(W0): hold top-N by rank (N = the historical max_positions), equal weight,
rebalance monthly, sell only on rank exit.

## 4. Out of scope

The R1 gate amendment (separate spec, after this runs); spread-based cost
models (R1); down-cap universes (R3); any live/paper deployment (R4).
