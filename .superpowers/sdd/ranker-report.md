# Ranker registry: implementation report

## What changed

Made the ranking-strategy selection pluggable via a registry, so the
experiment cadence can swap in new rankers (other momentum variants, a
future fundamentals ranker, per-universe strategies) via config instead of
touching the engine.

- New module `src/trading/signals/registry.py`: a `RANKERS: dict[str, Ranker]`
  registry mapping a name to a callable matching `compute_features`'s exact
  signature `(bars, as_of, config: SignalConfig) -> pd.DataFrame`. The current
  `compute_features` is registered directly (not wrapped) under the key
  `"momentum_v1"`, so behavior is bit-identical by construction (same
  function object, not a re-implementation). `get_ranker(name)` returns the
  callable or raises `ValueError` listing all known ranker names, sorted.
  The module docstring documents the contract a new ranker must uphold:
  purity (no I/O, no clock reads — `as_of` is always explicit), that
  truncation to `as_of` is the ranker's own responsibility (traced to
  `compute_features`'s `df.loc[:as_of]` "structural no-lookahead cut"), the
  exact output column contract (`OUTPUT_COLUMNS`), and NaN semantics
  (a symbol may appear with NaN feature/composite values; symbols with
  insufficient history are omitted from the index entirely). The shared
  `rank()` sort step is explicitly called out as NOT part of a ranker's job.

- `SignalConfig` (`src/trading/config.py`) gains a required `ranker: str`
  field. `load_venue_config` validates it against
  `trading.signals.registry.RANKERS` at config-load time (not at first
  pipeline run), raising `ValueError` naming the bad value and listing known
  rankers. The import of `registry` inside `load_venue_config` is a
  deferred/local import — `registry.py` imports `SignalConfig` from
  `trading.config`, so importing `registry` at `config.py`'s module level
  would be a circular import. Deferring it to call time breaks the cycle
  cleanly (by call time, `trading.config`'s module body, including the
  `SignalConfig` class, has already finished executing).

- `src/trading/pipeline.py`: `assemble_rankings` now resolves the ranker via
  `get_ranker(config.signals.ranker)` and calls it in place of importing and
  calling `compute_features` directly. `compute_features` is no longer
  imported by `pipeline.py` at all — grepping `src/` confirms the only
  remaining references to `compute_features` are inside
  `signals/engine.py` (its definition) and `signals/registry.py` (its
  registration + docstring mention). The shared `rank()` sort step is
  unchanged and still runs on the ranker's output afterward.

- Added `ranker = "momentum_v1"` to `config/equities.toml`,
  `config/crypto.toml`, and `tests/golden/golden.toml` (all three TOML files
  that flow through `load_venue_config`, including the golden fixture).

- `README.md` "Where things live" section gains two sentences on the
  `[signals].ranker` config key and how to register a new ranker.

## Files touched (absolute paths)

- `/Users/travis/Source/personal/trading/worktrees/ranker-registry/src/trading/signals/registry.py` (new)
- `/Users/travis/Source/personal/trading/worktrees/ranker-registry/src/trading/config.py`
- `/Users/travis/Source/personal/trading/worktrees/ranker-registry/src/trading/pipeline.py`
- `/Users/travis/Source/personal/trading/worktrees/ranker-registry/config/equities.toml`
- `/Users/travis/Source/personal/trading/worktrees/ranker-registry/config/crypto.toml`
- `/Users/travis/Source/personal/trading/worktrees/ranker-registry/tests/golden/golden.toml`
- `/Users/travis/Source/personal/trading/worktrees/ranker-registry/tests/test_registry.py` (new)
- `/Users/travis/Source/personal/trading/worktrees/ranker-registry/tests/test_config.py`
- `/Users/travis/Source/personal/trading/worktrees/ranker-registry/tests/test_engine.py`
- `/Users/travis/Source/personal/trading/worktrees/ranker-registry/README.md`

## Tests added (TDD: written first, confirmed red, then made green)

- `tests/test_registry.py`
  - `test_get_ranker_unknown_name_raises_listing_known_names` — `get_ranker("bogus")` raises
    `ValueError` mentioning `"momentum_v1"`.
  - `test_get_ranker_returns_registered_callable` — `get_ranker("momentum_v1") is RANKERS["momentum_v1"]`.
  - `test_momentum_v1_output_matches_compute_features_exactly` — `pd.testing.assert_frame_equal`
    on a realistic trending-bars fixture, comparing the registry-resolved
    ranker's output against `compute_features` called directly.
- `tests/test_config.py`
  - `test_ranker_loaded_as_momentum_v1_by_default_in_real_configs` — both real venue TOMLs load
    `signals.ranker == "momentum_v1"`.
  - `test_unknown_ranker_raises_at_load_time` — a TOML with `ranker = "bogus_ranker"` raises
    `ValueError` matching `"bogus_ranker"` at `load_venue_config` time.
- `tests/test_engine.py` — only a one-line fixture change: added
  `ranker="momentum_v1"` to the existing `CONFIG = SignalConfig(...)`
  construction (now a required field). No behavioral test changes.

Confirmed red before implementation: `uv run pytest tests/test_registry.py -q`
failed collection with `ModuleNotFoundError: No module named
'trading.signals.registry'` prior to creating the module.

## Test results

- Baseline (before any change): `320 passed in 29.81s`, 0 warnings.
- After implementation: `325 passed in 17.64s` (320 baseline + 5 new: 3 in
  `test_registry.py`, 2 in `test_config.py`), 0 warnings, 0 failures. Full
  suite run via `uv run pytest -q`.
- `tests/test_golden_backtest.py` run in isolation: all 4 tests pass
  (`test_golden_backtest_matches_committed_expected`,
  `test_golden_fixture_actually_trades`,
  `test_golden_expectation_covers_every_exit_path`,
  `test_golden_skips_are_data_quality_not_listing`).

## Golden invariant verification

- `git diff --stat tests/golden/expected.json` and
  `git status --porcelain tests/golden/expected.json` both show no output:
  the file is byte-for-byte untouched. Only `tests/golden/golden.toml` was
  edited (added the `ranker = "momentum_v1"` key), and the golden test still
  passes against the pre-existing, unmodified `expected.json`. This confirms
  the registry indirection is behaviorally transparent for the full
  backtest engine path (`assemble_rankings` -> ranker -> `rank()`), not just
  for the isolated `compute_features` comparison in `test_registry.py`.

## No-lookahead / purity property test

`tests/test_engine.py::test_no_lookahead_property` is unmodified in
substance (only the fixture's `SignalConfig(...)` construction gained the
new required `ranker="momentum_v1"` argument) and still passes. It exercises
`compute_features` directly, which is exactly what's registered under
`"momentum_v1"`, so its guarantee (perturbing post-`as_of` data does not
change output at `as_of`) continues to hold for the registered ranker
without any new property test being strictly necessary — the registry is
a pure dict lookup / delegation, it does not touch data or time itself.

## Ruff

`uv run ruff check src/ tests/` — all checks passed after one fix: the
initial `registry.py` draft used `typing.Callable` (ruff UP035: prefer
`collections.abc.Callable`) and had two lines over the 100-char limit (E501)
from an inline `Callable[[...], ...]` annotation repeated in both `RANKERS`
and `get_ranker`. Fixed by importing from `collections.abc` and factoring
the annotation into a `Ranker` type alias, reused in both places.

## Design decisions / ambiguities resolved

- **`ranker` field location**: the plan's step 3 wording said
  "`registry.get_ranker(config.ranker)`" where `config` is `VenueConfig` in
  `assemble_rankings`'s signature. The field was specified to live on
  `SignalConfig` (per step 2), so the actual call is
  `get_ranker(config.signals.ranker)` — consistent with how
  `compute_features` was already being called with `config.signals`
  (not `config`) before this change. No behavior ambiguity, just read the
  plan's two steps together.
- **Registering `compute_features` directly vs. a wrapper**: the plan
  allowed either "wrap or reference directly." I registered the function
  object directly (`"momentum_v1": compute_features`) rather than writing an
  adapter, since `compute_features`'s existing signature is already an exact
  match for the `Ranker` contract — no argument-shape translation needed.
  This guarantees bit-identical behavior by construction (verified by
  `test_momentum_v1_output_matches_compute_features_exactly` and, at the
  system level, by the untouched golden `expected.json` still matching).
- **Circular import between `config.py` and `registry.py`**: `registry.py`
  needs `SignalConfig` from `config.py` for its type contract, but
  `config.py` needs `registry.RANKERS` to validate at load time. Resolved
  with a deferred (function-body) import in `load_venue_config`, documented
  inline with a comment explaining why.
- **`ranker` field default**: made it a required (no-default) field on
  `SignalConfig`, matching every other field in that dataclass (the codebase
  convention per its module docstring: "Unknown or missing TOML keys raise
  TypeError via dataclass construction"). This meant updating the one other
  direct `SignalConfig(...)` construction site (`tests/test_engine.py`) to
  pass `ranker="momentum_v1"`.

## Concerns / follow-ups

- None outstanding. The registry is intentionally minimal (a dict + a
  lookup function) since the spec only asked for pluggability, not a
  plugin-discovery mechanism, entry points, or per-ranker config schemas —
  those would be reasonable future extensions if/when a second ranker with
  materially different config needs actually arrives (e.g. a fundamentals
  ranker needing new SignalConfig fields would need those fields added to
  the shared dataclass, since all rankers currently share one `SignalConfig`
  shape).

## Polish pass (post-review, 2026-07-05)

Review approved with three Minor items; all applied in one commit:

1. `load_venue_config` no longer duplicates the known-rankers enumeration:
   it now calls `get_ranker(signals["ranker"])` and lets the registry's
   ValueError propagate, making `registry.get_ranker` the single source of
   truth for the unknown-ranker message. Fail-fast-at-load is preserved
   (the call still happens inside `load_venue_config`, before any pipeline
   run); the deferred-import comment was updated accordingly.
2. `test_unknown_ranker_raises_at_load_time` now additionally asserts the
   error message enumerates the known rankers (`"momentum_v1" in str(...)`),
   pinning the message-content contract end to end through config load.
3. `registry.py`'s module docstring gained an explicit INPUT contract
   section: `bars` values are OHLCV frames with columns exactly
   [open, high, low, close, volume] on a sorted tz-aware UTC DatetimeIndex
   (the trading.venues.base.validate_ohlcv shape), frames may extend past
   as_of, and truncation is the RANKER's responsibility -- with the exact
   location of momentum_v1's cut named (`window = df.loc[:as_of]`, first
   step of compute_features' per-symbol loop). A float64-dtype claim was
   deliberately NOT made: validate_ohlcv does not enforce dtype, only
   columns/index shape.

Re-verified after the pass: full suite 325 passed, 0 warnings; ruff clean.
`tests/golden/expected.json` remains untouched.
