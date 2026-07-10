# R2 Wrapper-Ablation — Build Report

Spec: `docs/superpowers/specs/2026-07-10-wrapper-ablation-design.md`.
Branch: `tpett/ai/ablation`.

## 1. Investigation findings

### Config-only components (no engine change needed)

- **Entry threshold**: `[portfolio].entry_score_threshold = 0.0` admits
  every tradable candidate — `signals.engine`'s composite is a
  cross-sectional percentile blend, always in `(0, 1]`, so `0.0` is a
  guaranteed-safe "off" sentinel (`src/trading/simulator/entries.py:63`).
- **Grid**: single-point `entry_score_threshold_grid` /
  `stop_atr_multiple_grid` lists collapse `walkforward.grid_points()` to one
  combination — no tuning occurs, no engine change (`src/trading/backtest/
  walkforward.py:94`).
- **ATR stop (entry-frozen and trailing)**: `stop_atr_multiple` set to a
  large sentinel (`1000.0`) makes `stop_price` deeply negative
  (`entry_price − multiple·ATR`); since prices can't go negative,
  `last_close <= stop_price` never trips (`src/trading/simulator/fills.py:99`,
  `exits.py:98/114`). Config-only.
- **Regime-flush ratchet**: a *second* ATR-based stop
  (`regime_flush_atr_multiple`), triggered when `regime.state == "risk_off"`
  (`exits.py:80-116`). Also config-only-disableable via the same
  large-sentinel trick — but it is a fifth, previously-unnamed lever (see
  below).
- **Circuit breaker / drawdown halt**: `drawdown_halt_pct` set above `1.0`
  (used `10.0`) can never trip since a long-only, no-margin book's drawdown
  is bounded at 100% (`src/trading/simulator/core.py:135-143`). Config-only.

### NOT config-only: regime gate

`entries.py:43` checks `rankings.regime.state == "risk_off"` **directly**,
not the exposure multiplier — so even setting
`exposure_risk_on=exposure_neutral=exposure_risk_off=1.0` leaves entries
hard-blocked whenever `compute_regime` classifies the market risk-off, and
`compute_regime`'s risk-off branch (`last < sma_slow`) is driven by real
price action, not a tunable numeric threshold. No config combination can
force "always risk_on." **Required a genuine engine change**: added
`RegimeConfig.disabled: bool = False`; when true, `compute_regime`
short-circuits to `Regime("risk_on", 1.0)` before touching the bars
(`src/trading/signals/regime.py`).

### Hidden wrapper components identified (beyond the 4 named)

1. **`time_stop_bars`** (`exits.py:127-129`): forces an exit after N bars
   held near breakeven, unconditional on regime/stops/threshold.
2. **`trend_break` exit** (`exits.py:118-125`, frozen `exit_style` only):
   sells a held name once it falls into the ranked bottom half AND below
   its 20-day mean. A momentum-era rule, not one of the 4 named axes.
3. **`regime_flush_atr_multiple`** ratchet (see above) — a second ATR stop,
   distinct from `stop_atr_multiple`, gated by regime state.
4. **`drawdown_halt_pct` circuit breaker** (`core.py:132-144`): halts ALL
   entries venue-wide on a portfolio-level drawdown, independent of
   regime/stops/threshold.
5. **`cooldown_days`** (`entries.py:71-74`, set in `fills.py:125-127`):
   re-entry lockout, but only ever *set* on a `stop_loss` fill — inert
   automatically whenever stops are off, needs no separate handling.
6. **`max_daily_deployment_pct`** (`entries.py:106-108`): 25%-of-portfolio
   daily cap — incompatible with W0's "hold top-N, rebalance monthly"
   requirement (a single rebalance day must be able to fill all N slots),
   so bare mode bypasses it entirely.
7. **Position sizing model**: `position_size_pct` is a *fixed fraction of
   current portfolio value per buy*, sized once at entry and never
   rebalanced — structurally different from W0's "equal weight, monthly
   rebalance" (which implies a moving 1/N target). The simulator has no
   partial-trim/add primitive at all (positions are atomic lots).
8. **Daily cadence**: the entire engine trades every session; there is no
   existing "only decide on rebalance days" concept anywhere in
   `core.step`/`walkforward`.

Items 1–2 (and daily-vs-monthly cadence, equal-weight sizing) meant W0
could not be reached by flipping existing knobs — it needed a distinct
execution mode, not just larger stop/threshold numbers.

## 2. What was built

### Config-only (documented via large sentinel values in the generated
TOMLs, no code change): entry threshold, grid, ATR stop, regime-flush
ratchet, drawdown halt.

### Two new additive, default-`False` config flags

- **`[regime] disabled`** (`src/trading/config.py`,
  `src/trading/signals/regime.py`): bypasses the gate entirely — always
  `risk_on`/`1.0×`. The only way to defeat the price-driven risk-off branch.
- **`[portfolio] bare_mode`** (`src/trading/config.py`,
  new module `src/trading/simulator/bare.py`, wired into
  `src/trading/simulator/core.py::step`): W0's dedicated execution mode.
  - `rank_only_top_n`: top-`max_positions` tradable, non-NaN-composite,
    liquid (dollar-volume floor, held constant — it's a universe
    definition, not a wrapper) symbols, by rank order. No threshold.
  - `evaluate_exits_bare`: forced exits (delisted/untradable/quarantined/
    fetch-failed — data integrity, always active) fire every session;
    `rank_exit` (fell out of the top-N) fires **only on a monthly
    rebalance session**.
  - `evaluate_entries_bare`: on a rebalance session, buys every top-N
    symbol not already held at **equal weight** (`portfolio_value /
    max_positions`), respecting cash but *not* the daily-deployment cap.
  - Monthly cadence: `PortfolioState` gained one backward-compatible field,
    `bare_last_rebalance_month` (`"YYYY-MM"`, `None` default,
    `payload.get(...)` on load so old state files are unaffected); a
    session is a rebalance session iff its month differs from the stored
    one.
  - The circuit breaker (`state.breaker_tripped`) still gates bare-mode
    entries — held as an orthogonal safety mechanism, not part of the
    ablation (though its threshold is set to the disabling sentinel in
    every non-W4 cell anyway).

**Design choice, explicitly not spec-named**: `time_stop_bars` and
`trend_break` are only stripped by `bare_mode` (W0). For W1–W3 they remain
active exactly as in the historical engine, since the spec's matrix names
only 4 axes and W1–W3 are NOT bare mode — only W0 is. This is documented
in the ablation config headers and in `bare.py`'s module docstring.

### W4-equivalence proof

`core.step`'s non-bare branch is **byte-for-byte the original code**,
merely wrapped in `if bare: ... else: <unchanged lines>`; `compute_regime`'s
non-disabled path is likewise untouched. `tests/test_ablation_equivalence.py`
pins a golden fixture captured from `trading.backtest.engine.replay()`
**before** any of this code existed (trades, realized P&L, entry/exit
prices, equity curve, fees, warnings — 8+ independent fields) and asserts
the post-change result is identical both at the dataclass defaults and when
the new flags are explicitly set to their off values. Combined with the
full existing suite (1121 tests, all still passing with `-W error`), this
is the "bit-for-bit on a fixture" proof the spec requires.

## 3. Config dirs generated

`config/experiments/ablation-<signal>-<cell>/equities.toml` for
`signal ∈ {momentum, amihud, skew}` × `cell ∈ {w0, w1, w2, w3, w4}` = 15
dirs, each a generated clone of the historical config with **only** the
wrapper-matrix fields changed (`[regime].disabled`, `[portfolio].bare_mode`
/ `entry_score_threshold` / `stop_atr_multiple` /
`regime_flush_atr_multiple` / `drawdown_halt_pct`, `[backtest]`'s two
grids). `tests/test_ablation_configs.py` parametrically asserts every cell
matches the spec table and that every OTHER field (costs, universe,
signals, data, non-ablated portfolio/backtest fields) is byte-identical to
its source.

| signal | cloned from | ranker | cost convention |
|---|---|---|---|
| momentum | `config/experiments/tiingo` | `momentum_v1` | 5 bps, sp500+ndx (the "best honest 0.45" config) |
| amihud | `config/experiments/amihud-midcap` | `amihud_v1` | 15 bps, sp400 (the battery-passed, 0.16-wrapper-form config) |
| skew | `config/experiments/options-skew` | `skew_v1` | 5 bps, options-covered allowlist (the §9 OPT-1 config, 0.79) |

**Skew provenance**: `config/experiments/options-skew/equities.toml`
already IS the OPT-1 setup docs/experiments.md §9 describes (ranker
`skew_v1`, `data.skew_samples`/`universe.symbols_allowlist_path` both
pointing at `data/options-iv/samples.jsonl`, `backtest.start = 2019-01-01`
matching the documented 2019-01..2025-12 skew-cell span) — no construction
needed, just cloned like the other two.

**Sentinel values** (documented in every generated file's `[portfolio]`
comments): `stop_atr_multiple = regime_flush_atr_multiple = 1000.0`,
`drawdown_halt_pct = 10.0`. W2 ("stops ON") instead fixes them at the
historical `1.5` / `1.0` (single-point, no search). W4 is an exact clone
(verified via `diff` against the source configs — only the two new inert
flag lines and a `0.20`→`0.2` float formatting no-op differ).

## 4. What was NOT done

The 15 actual walk-forward runs (journal entries, experiments.md §13
write-up) were **not** executed — the task scope here was building the
capability and generating the pre-registered configs, not running the
experiment. Running needs live Tiingo/options-IV data fetches and
walk-forward compute time well beyond this build step; that's the natural
next task once this is reviewed.

## 5. Verification

- `uv run pytest -q -W error`: **1121 passed**
- `uv run ruff check .`: **All checks passed**
