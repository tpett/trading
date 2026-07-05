# M3: Backtester & Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A user can run `trading backtest --venue equities|crypto [--from --to] [--walk-forward] [--holdout] [--json]` and get a 2018-present replay of the exact live-paper simulator over point-in-time universes, with walk-forward validation of the two tunable hyperparameters, a once-only final holdout, a full metrics report vs buy-and-hold benchmark, and every experiment journaled.

**Architecture:** A `trading.backtest` package with two halves: `prepare()` owns all I/O (point-in-time universe, deep-history cached fetch, per-session rankings precomputed with the SAME `assemble_rankings` core the live pipeline uses) and `replay()` — pure, no I/O, no clock — drives the SAME `simulator.core.step()` as live-paper, session by session (equities: NYSE sessions = SPY bar dates; crypto: UTC daily bars), filling at next-bar opens. Rankings depend only on data + signal config, never on the two tunable hyperparameters, so one `prepare()` serves every walk-forward grid point. Walk-forward, metrics, and the experiments journal are thin pure layers on top; the CLI owns the clock, journaling, and the holdout once-only gate.

**Tech Stack:** Python 3.12, uv, pandas, ccxt, yfinance (all pinned), argparse + rich, pytest (warnings-as-errors), ruff. **No new runtime dependencies** — the NYSE calendar is the benchmark's own bar dates; deep crypto history comes through ccxt (already a dependency).

## Global Constraints

From the approved spec (`docs/superpowers/specs/2026-07-04-momentum-swing-system-design.md`) and locked decisions. Every task's requirements implicitly include this section.

- Repo root for all commands and relative paths: `/Users/travis/Source/personal/trading/worktrees/backtester` (git worktree, branch `tpett/ai/backtester`).
- Python 3.12, uv (`uv sync`, `uv run ...`). Before every commit: `uv run ruff check . && uv run ruff format .` and the affected tests. pytest runs with warnings-as-errors (already configured in `pyproject.toml`).
- **One engine, replayed:** the backtester calls `trading.simulator.core.step()` verbatim. Never fork, copy, or re-implement simulator logic. Any simulator change (Task 2's entry fee) must keep live/paper behavior identical apart from the deliberate, spec'd fix, with existing tests updated deliberately — never weakened.
- **Pure modules stay pure:** `replay()`, `walkforward.py`, `metrics.py` do no I/O and read no clock — the backtest clock is the replay loop's parameter (the session timestamps). I/O lives in `prepare()`, `experiments.py` (journal writes), and the CLI. Only `cli.py` reads the wall clock.
- **No lookahead:** decisions at session T use bars sliced to ≤ T; entries fill at the next bar's open exactly as live-paper (the fill engine already does this — pending orders fill at the first bar strictly after their decision bar).
- **Point-in-time universe (equities):** `universe(as_of)` returns historical S&P 500 + NDX membership as-of that date. Backtesting today's members over the past is prohibited (spec).
- **Tunable surface is exactly two hyperparameters:** `entry_score_threshold` and `stop_atr_multiple`, grid-searched from TOML-configured grids. Nothing else is fitted.
- **Final holdout** (`[backtest].holdout_start` onward) is touched exactly once, via `trading backtest --holdout`, which refuses to run twice without typed confirmation. Non-holdout backtests are clamped to end before it.
- Every new number lives in TOML (`config/<venue>.toml`), never as a code constant.
- All timestamps UTC everywhere. Span target: 2018-present.
- Commit after every task, one logical change per commit, message tagged `[AI]`, with footer `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.
- Baseline before starting: 239 tests green (`uv run pytest -q`), ruff clean.

## File Structure

```
config/equities.toml                       MODIFY: [data] backfill fields (disabled), [backtest] section, half-day note
config/crypto.toml                         MODIFY: [data] backfill fields, [backtest] section
src/trading/config.py                      MODIFY: DataConfig backfill fields; new BacktestConfig; VenueConfig.backtest
src/trading/simulator/state.py             MODIFY: Position.entry_fee (default 0.0)
src/trading/simulator/fills.py             MODIFY: store entry fee at fill; realized_pnl includes it
src/trading/pipeline.py                    MODIFY: extract pure assemble_rankings() from build_rankings()
src/trading/venues/crypto.py               MODIFY: deep-history backfill via second ccxt exchange; pagination refactor
src/trading/venues/equities.py             MODIFY: universe(as_of) reads point-in-time membership CSV
src/trading/venues/universes/equities_membership.csv    NEW: committed point-in-time membership (generated)
src/trading/venues/universes/sources/sp500_history.csv  NEW: committed raw dataset snapshot (provenance)
src/trading/venues/universes/sources/PROVENANCE.md      NEW: source URLs, commit SHA, licence, retrieval date
scripts/build_pit_membership.py            NEW: builds equities_membership.csv from snapshot + Wikipedia NDX changes
scripts/gen_golden_fixture.py              NEW: seeded golden fixture + expected-output generator
src/trading/backtest/__init__.py           NEW: empty package marker
src/trading/backtest/engine.py             NEW: prepare(), replay(), SessionPlan, PreparedBacktest, BacktestResult, TradeRecord
src/trading/backtest/metrics.py            NEW: BacktestMetrics, sharpe_ratio, max_drawdown, compute_metrics, gate
src/trading/backtest/experiments.py        NEW: experiments journal helpers (reuses trading.journal.Journal)
src/trading/backtest/walkforward.py        NEW: windows, grid search, OOS stitching, stress-segment check
src/trading/cli.py                         MODIFY: `trading backtest` subcommand
tests/test_fills.py                        MODIFY: weekend T+1 test (Task 1), entry-fee tests (Task 2)
tests/test_entries.py                      MODIFY: fee-gate + deployment-cap boundary tests (Task 1)
tests/test_journal.py                      MODIFY: all-torn-file branch test (Task 1)
tests/test_runner.py                       MODIFY: DST session-guard test (Task 1)
tests/test_state.py                        MODIFY: entry_fee round-trip + legacy-state load (Task 2)
tests/test_config.py                       MODIFY: backfill + backtest config assertions (Tasks 4, 6)
tests/test_crypto_adapter.py               MODIFY: backfill splice tests (Task 4)
tests/test_equities_adapter.py             MODIFY: point-in-time universe tests (Task 5)
tests/test_pipeline.py                     MODIFY: assemble_rankings direct test (Task 3)
tests/backtest_helpers.py                  NEW: noisy frames, FakeBacktestAdapter, small_config, hand-built sessions
tests/test_backtest_engine.py              NEW
tests/test_backtest_metrics.py             NEW
tests/test_experiments.py                  NEW
tests/test_walkforward.py                  NEW
tests/test_backtest_cli.py                 NEW
tests/golden/golden.toml                   NEW: frozen venue config for the golden backtest
tests/golden/bars/*.csv                    NEW: committed fixture bars (generated, seeded)
tests/golden/expected.json                 NEW: committed expected output
tests/golden_helpers.py                    NEW: GoldenAdapter + run_golden()
tests/test_golden_backtest.py              NEW
README.md                                  MODIFY: backtest commands, data provenance, survivorship caveats
```

Locked design decisions for this plan (referenced by tasks):

- **Session calendar = benchmark bar dates.** Equities sessions are SPY's actual yfinance bar dates (the realized NYSE calendar, half-days and holidays included); crypto sessions are BTC's UTC daily bars. No calendar dependency.
- **Session semantics:** the session at bar-timestamp T means "the run after bar T completed". Decisions use bars ≤ T; the previous session's pending orders fill at bar T's open (first bar strictly after their decision bar T-1) — identical to live.
- **`--to` defaults to yesterday (UTC)** and is clamped there: today's daily bar may still be in progress on either venue; the backtest never sees a possibly-incomplete bar.
- **Deep-history precedence (crypto):** Kraken owns `[end - backfill_before_days, end]`; the backfill exchange owns older rows; on any overlap Kraken wins (`keep="last"` after concat with Kraken last). Live 500-day requests never touch the backfill path.
- **Survivorship:** sessions where fewer than `[backtest].min_session_coverage` of point-in-time members have data are skipped (state carries over, like a live coverage-failed run that touches nothing). The mean per-session coverage ratio is annotated on every result; missing symbols are counted in the experiments journal. Crypto results always carry the delisted-coins caveat string.
- **Index-removed held names:** a held symbol that leaves the point-in-time universe but still has data is injected into the session table with status `untradable` → the simulator's existing forced-exit path sells it next bar (spec: dropped from the venue universe → force-exit). A held symbol whose data ends entirely stays as a reported open position marked at its last close, with a warning — we do not fork fill semantics to invent a liquidation print.
- **Walk-forward OOS state:** each test window replays from a fresh `initial_state`; stitched OOS = the concatenated daily returns of the test windows, compounded. Grid selection on the train window = highest Sharpe (tiebreak: higher total return, then lower threshold, then lower stop multiple — fully deterministic).
- **Holdout boundary is a fixed TOML date** (`holdout_start`), not a rolling 6-months — deterministic and auditable.

---

### Task 1: Deferred M2 review test batch + NYSE half-day note

Regression tests deferred from the M2 reviews. These codify EXISTING behavior — each test is expected to PASS immediately; if one fails, stop and investigate before touching any code (that would be a real M2 bug, worth its own fix commit).

**Files:**
- Modify: `tests/test_fills.py` (append)
- Modify: `tests/test_entries.py` (append)
- Modify: `tests/test_journal.py` (append)
- Modify: `tests/test_runner.py` (append)
- Modify: `config/equities.toml` (comment only)

**Interfaces:**
- Consumes: `apply_fills(state, bars, config)`, `release_settlements(state, decision_date)` from `trading.simulator.fills`; `evaluate_entries(state, rankings, config, decision_ts, portfolio_value)` from `trading.simulator.entries`; `Journal` from `trading.journal`; `intraday_partial_bar_reason(config, decision_ts, now)` from `trading.runner`; `frame`, `make_state`, `make_table`, `make_rankings`, `EQ`, `CR`, `AS_OF` from `tests/sim_helpers.py`.
- Produces: nothing new — regression coverage only.

- [ ] **Step 1: Append the weekend-crossing T+1 settlement test to `tests/test_fills.py`**

```python
def test_settlement_crosses_weekend_t_plus_1():
    # Sell decided Thu 2026-06-04 fills Fri 2026-06-05 -> available_on Sat
    # 2026-06-06. Cash must stay unspendable through Friday's session and be
    # released by Monday's (2026-06-08) decision date.
    bars = {"AAA": frame(end="2026-06-05", periods=5)}  # Mon..Fri, freq="B"
    state = make_state(EQ, positions={"AAA": _position()}, cash=0.0)
    state.pending_orders = [
        PendingOrder(
            symbol="AAA",
            side="sell",
            notional=0.0,
            decision_ts="2026-06-04T00:00:00+00:00",
            reason="trend_break",
        )
    ]
    fills, _ = apply_fills(state, bars, EQ)
    proceeds = fills[0].qty * fills[0].price  # zero commission on equities
    assert fills[0].bar_ts == "2026-06-05T00:00:00+00:00"
    assert state.settlements[0].available_on == "2026-06-06"  # Saturday

    release_settlements(state, datetime.date(2026, 6, 5))  # still Friday
    assert state.cash == 0.0
    release_settlements(state, datetime.date(2026, 6, 8))  # Monday's decision date
    assert state.cash == pytest.approx(proceeds)
    assert state.settlements == []
```

Add `import datetime` to the imports at the top of `tests/test_fills.py` if not present.

- [ ] **Step 2: Run it — expected PASS (characterizes existing behavior)**

Run: `uv run pytest tests/test_fills.py::test_settlement_crosses_weekend_t_plus_1 -v`
Expected: PASS. If it fails, stop — that is an M2 settlement bug; investigate before proceeding.

- [ ] **Step 3: Append boundary-equality tests to `tests/test_entries.py`**

The fee gate uses strict `<` (equality passes) and the deployment cap uses strict `>` (equality passes). Pin both boundaries:

```python
def test_fee_gate_boundary_equality_passes():
    # Crypto round-trip cost = 2 * (95 + 5) bps = 2.0%; gate = 3.0x = 6.0%.
    # A raw 30d return of EXACTLY 6.0% passes (strict <).
    round_trip = 2 * (CR.costs.taker_fee_bps + CR.costs.slippage_bps) / 1e4
    exactly_at_gate = CR.portfolio.min_raw_return_cost_multiple * round_trip
    bars = {"BTC": frame(end="2026-07-01")}
    table = make_table(
        {"BTC": {"status": "tradable", "composite": 0.9, "raw_return_30d": exactly_at_gate}}
    )
    rankings = make_rankings(CR, bars, table)
    state = make_state(CR)
    orders, skips = evaluate_entries(state, rankings, CR, AS_OF, 1000.0)
    assert [o.symbol for o in orders] == ["BTC"]
    assert ("BTC", "fee_gate") not in [(s.symbol, s.reason) for s in skips]


def test_deployment_cap_boundary_equality_passes():
    # With max_daily_deployment_pct == position_size_pct the first entry's
    # notional EQUALS the budget: it must pass (strict >); the second
    # candidate then faces a zero budget and is capped.
    config = replace(
        EQ, portfolio=replace(EQ.portfolio, max_daily_deployment_pct=EQ.portfolio.position_size_pct)
    )
    bars = {"AAA": frame(end="2026-07-01"), "BBB": frame(end="2026-07-01")}
    table = make_table(
        {
            "AAA": {"status": "tradable", "composite": 0.9, "raw_return_30d": 0.5},
            "BBB": {"status": "tradable", "composite": 0.8, "raw_return_30d": 0.5},
        }
    )
    rankings = make_rankings(config, bars, table)
    state = make_state(config)
    orders, skips = evaluate_entries(state, rankings, config, AS_OF, 1000.0)
    assert [o.symbol for o in orders] == ["AAA"]
    assert ("BBB", "daily_deployment_cap") in [(s.symbol, s.reason) for s in skips]
```

Add `from dataclasses import replace` to the imports at the top of `tests/test_entries.py` if not present.

- [ ] **Step 4: Run them — expected PASS**

Run: `uv run pytest tests/test_entries.py -v -k boundary`
Expected: 2 passed.

- [ ] **Step 5: Append the all-torn-file branch test to `tests/test_journal.py`**

A journal whose ONLY line is torn (no trailing newline, invalid JSON): `events()` must yield nothing (torn final line), and `append()` must truncate the file to empty before writing — never concatenate onto the torn fragment.

```python
def test_append_to_fully_torn_single_line_file(tmp_path):
    path = tmp_path / "torn.jsonl"
    path.write_bytes(b'{"event": "run", "torn')  # no newline anywhere in the file
    journal = Journal(path)
    assert list(journal.events()) == []

    journal.append({"event": "after_repair"})
    events = list(journal.events())
    assert [e["event"] for e in events] == ["after_repair"]
    # The torn fragment is gone entirely, not merged into the new line.
    assert path.read_text().count("\n") == 1
    assert "torn" not in path.read_text()
```

- [ ] **Step 6: Run it — expected PASS**

Run: `uv run pytest tests/test_journal.py::test_append_to_fully_torn_single_line_file -v`
Expected: PASS.

- [ ] **Step 7: Append the DST-transition session-guard test to `tests/test_runner.py`**

Same UTC wall-clock time, opposite outcomes across the DST boundary — proves the guard converts through America/New_York rather than assuming a fixed UTC offset. US DST ends 2026-11-01; deadline is 16:00 local + 90 min buffer = 17:30 local.

```python
def test_session_guard_is_dst_aware():
    eq = load_venue_config("equities", Path("config"))
    # July (EDT, UTC-4): 21:45 UTC = 17:45 local -> past the 17:30 deadline: run allowed.
    summer_bar = pd.Timestamp("2026-07-06", tz="UTC")
    summer_now = datetime.datetime(2026, 7, 6, 21, 45, tzinfo=datetime.UTC)
    assert intraday_partial_bar_reason(eq, summer_bar, summer_now) is None
    # November (EST, UTC-5): 21:45 UTC = 16:45 local -> BEFORE the deadline: refuse.
    winter_bar = pd.Timestamp("2026-11-02", tz="UTC")
    winter_now = datetime.datetime(2026, 11, 2, 21, 45, tzinfo=datetime.UTC)
    assert intraday_partial_bar_reason(eq, winter_bar, winter_now) is not None
```

Reuse the existing imports in `tests/test_runner.py` (it already imports `intraday_partial_bar_reason`, `load_venue_config`, `datetime`, `pd`, `Path` — add any of these that are missing).

- [ ] **Step 8: Run it — expected PASS**

Run: `uv run pytest tests/test_runner.py::test_session_guard_is_dst_aware -v`
Expected: PASS.

- [ ] **Step 9: Document the NYSE half-day behavior as conservative**

In `config/equities.toml`, extend the comment block above `session_close_buffer_minutes` (keep the existing lines, append these):

```toml
# NYSE half-days (e.g. day after Thanksgiving, 13:00 ET close) are NOT special-
# cased: the guard still waits for 16:00 ET + buffer. That is conservative --
# it can only delay a half-day run by ~3h, never trade a partial bar. If that
# delay ever matters, the fix is a session-calendar lookup feeding this guard,
# not a smaller buffer.
```

- [ ] **Step 10: Run the full suite, lint, commit**

Run: `uv run pytest -q && uv run ruff check . && uv run ruff format --check .`
Expected: 244 passed (239 + 5 new), ruff clean.

```bash
git add tests/test_fills.py tests/test_entries.py tests/test_journal.py tests/test_runner.py config/equities.toml
git commit -m "$(cat <<'EOF'
Backfill deferred M2 review tests; document half-day guard [AI]

Weekend-crossing T+1 settlement, fee-gate and deployment-cap boundary
equality, journal all-torn-file repair, DST-aware session guard. The
NYSE half-day close is documented as conservatively handled (guard
waits for 16:00 ET + buffer; can delay, never trades a partial bar).

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: Entry fee on Position; per-trade realized P&L includes it

Spec Open Item: the buy-side fee is deducted from cash at fill time but not reflected in `realized_pnl`, so per-trade P&L understates round-trip cost by the entry fee. Fix: store the entry fee on the Position at fill; subtract it in the sell's `realized_pnl`. Cash accounting is already exact and must not change. Live/paper behavior is otherwise identical.

**Files:**
- Modify: `src/trading/simulator/state.py` (Position dataclass)
- Modify: `src/trading/simulator/fills.py` (buy stores fee; sell subtracts it)
- Modify: `tests/test_fills.py`, `tests/test_state.py`

**Interfaces:**
- Consumes: `Position`, `apply_fills` as they exist today.
- Produces: `Position.entry_fee: float = 0.0` (new field, default keeps old state files loadable — STATE_VERSION stays 1); `Fill.realized_pnl` on sells now equals `proceeds - qty * entry_price - entry_fee`. Task 7's TradeRecord relies on this.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_fills.py`:

```python
def test_buy_fill_stores_entry_fee_on_position():
    bars = {"BTC": frame(end="2026-07-01")}
    state = make_state(CR, pending_orders=[_buy_order(symbol="BTC", notional=300.0)])
    apply_fills(state, bars, CR)
    assert state.positions["BTC"].entry_fee == pytest.approx(300.0 * 95.0 / 1e4)


def test_sell_realized_pnl_includes_entry_fee():
    bars = {"BTC": frame(end="2026-07-01")}
    position = _position(symbol="BTC")
    position = replace(position, entry_fee=2.85)
    state = make_state(CR, positions={"BTC": position}, cash=0.0)
    state.pending_orders = [
        PendingOrder(symbol="BTC", side="sell", notional=0.0, decision_ts=DECISION, reason="time_stop")
    ]
    fills, _ = apply_fills(state, bars, CR)
    price = 100.0 * (1 - 5.0 / 1e4)
    gross = 2.0 * price
    proceeds = gross - gross * 95.0 / 1e4
    assert fills[0].realized_pnl == pytest.approx(proceeds - 2.0 * 90.0 - 2.85)
```

Add `from dataclasses import replace` to `tests/test_fills.py` imports.

Append to `tests/test_state.py` (inside/alongside the existing round-trip tests):

```python
def test_position_without_entry_fee_loads_as_zero():
    # Old (pre-M3) state files have no entry_fee; they must load with 0.0.
    state = _state()  # the module's existing populated-state builder
    payload = to_state_dict(state)
    for position in payload["positions"].values():
        del position["entry_fee"]
    restored = state_from_dict(payload)
    assert all(p.entry_fee == 0.0 for p in restored.positions.values())
```

(If `tests/test_state.py` names its builder differently, use that module's existing populated-state helper — the point is a payload with at least one position.)

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_fills.py -k entry_fee tests/test_state.py -k entry_fee -v`
Expected: FAIL — `TypeError: __init__() got an unexpected keyword argument 'entry_fee'` / `AttributeError: 'Position' object has no attribute 'entry_fee'`.

- [ ] **Step 3: Add the field and use it in fills**

In `src/trading/simulator/state.py`, add the field at the END of `Position` (default keeps old JSON loadable; STATE_VERSION stays 1):

```python
@dataclass(frozen=True)
class Position:
    symbol: str
    qty: float
    entry_price: float  # fill price including slippage
    entry_ts: str  # ISO-8601 UTC timestamp of the fill bar
    entry_atr: float  # ATR frozen at entry (bars through the entry decision bar)
    stop_price: float
    flushed: bool  # one-way regime-flush ratchet already applied
    entry_composite: float  # ranking evidence at decision time (journal/digest rationale)
    entry_rank: int
    entry_fee: float = 0.0  # buy-side fee paid at fill; folded into realized_pnl on exit
```

In `src/trading/simulator/fills.py`, in the buy branch of `apply_fills`, add `entry_fee=fee` to the `Position(...)` construction:

```python
            state.positions[order.symbol] = Position(
                symbol=order.symbol,
                qty=qty,
                entry_price=price,
                entry_ts=bar_ts.isoformat(),
                entry_atr=order.atr_at_decision,
                stop_price=price - config.portfolio.stop_atr_multiple * order.atr_at_decision,
                flushed=False,
                entry_composite=order.composite,
                entry_rank=order.rank,
                entry_fee=fee,
            )
```

In the sell branch, change the realized-P&L expression (last argument of the sell `Fill(...)`) from `proceeds - position.qty * position.entry_price` to:

```python
                    proceeds - position.qty * position.entry_price - position.entry_fee,
```

- [ ] **Step 4: Run the full suite — deliberate-update check**

Run: `uv run pytest -q`
Expected: all pass. The pre-existing sell tests (`test_equities_sell_settles_t_plus_1_and_realizes_pnl`, `test_crypto_sell_is_immediately_settled`) construct positions via `_position()` which now defaults `entry_fee=0.0`, so their expectations are unchanged — that is the deliberate, documented outcome (equities pay zero commission; the crypto test's position predates the fee being stored). Do NOT loosen any assertion. If anything else fails, the failure is telling you about a real behavioral coupling — investigate.

- [ ] **Step 5: Lint and commit**

Run: `uv run ruff check . && uv run ruff format --check .`

```bash
git add src/trading/simulator/state.py src/trading/simulator/fills.py tests/test_fills.py tests/test_state.py
git commit -m "$(cat <<'EOF'
Fold the entry fee into per-trade realized P&L [AI]

Spec Open Item: cash accounting was exact but realized_pnl understated
round-trip cost by the buy-side fee. The fee is now frozen on the
Position at fill (entry_fee, default 0.0 so pre-M3 state files still
load; STATE_VERSION unchanged) and subtracted from the sell fill's
realized_pnl. Live/paper cash behavior is unchanged.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: Extract the pure rankings core (`assemble_rankings`)

`build_rankings` mixes per-symbol fetching (I/O) with the pure coverage → quarantine → regime → features → rank sequence. The backtester must run that pure sequence over pre-fetched, pre-sliced frames thousands of times. Extract it as `assemble_rankings()`; `build_rankings` becomes fetch + delegate. Zero behavior change.

**Files:**
- Modify: `src/trading/pipeline.py`
- Modify: `tests/test_pipeline.py` (append)

**Interfaces:**
- Consumes: `check_coverage`, `quarantine_outliers`, `compute_regime`, `compute_features`, `rank` (unchanged).
- Produces: `assemble_rankings(config: VenueConfig, infos: list[SymbolInfo], bars: dict[str, pd.DataFrame], benchmark_bars: pd.DataFrame, as_of: datetime.date, fetch_failures: tuple[str, ...] = ()) -> RankingsResult` — raises `PipelineDataError` exactly where `build_rankings` did (coverage below floor, benchmark sanity failure). Task 7's `prepare()` calls this per session.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_pipeline.py`:

Constant-drift `sim_helpers.frame` series produce NaN vol-adjusted features, but the symbols still appear in the table (with NaN composite) because their history length is sufficient — this test asserts structure, which is all the extraction can change:

```python
def test_assemble_rankings_is_pure_and_matches_build_rankings_semantics():
    # No adapter, no cache: hand it frames directly and get a full RankingsResult.
    from trading.pipeline import assemble_rankings

    config = load_venue_config("equities", Path("config"))
    bars = {"AAA": frame(periods=300), "BBB": frame(periods=300)}
    benchmark = frame(periods=300)
    infos = [
        SymbolInfo(symbol="AAA", status="tradable"),
        SymbolInfo(symbol="BBB", status="sell_only"),
    ]
    result = assemble_rankings(config, infos, bars, benchmark, datetime.date(2026, 7, 1))
    assert set(result.table.index) == {"AAA", "BBB"}
    assert result.table.loc["BBB", "status"] == "sell_only"
    assert result.coverage.ratio == 1.0
    assert result.bars.keys() == bars.keys()  # nothing quarantined
    assert result.venue == "equities"
```

Reuse `tests/test_pipeline.py`'s existing imports (`frame` via sim_helpers, `SymbolInfo`, `load_venue_config`, `datetime`, `Path`) — add any missing.

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_pipeline.py::test_assemble_rankings_is_pure_and_matches_build_rankings_semantics -v`
Expected: FAIL — `ImportError: cannot import name 'assemble_rankings'`.

- [ ] **Step 3: Extract the function**

In `src/trading/pipeline.py`, add `from trading.venues.base import SymbolInfo` to the imports, then add `assemble_rankings` and rewrite `build_rankings` to delegate. The extracted body is a verbatim move of lines 74–132 of the current file (coverage check onward), parameterized:

```python
def assemble_rankings(
    config: VenueConfig,
    infos: list[SymbolInfo],
    bars: dict[str, pd.DataFrame],
    benchmark_bars: pd.DataFrame,
    as_of: datetime.date,
    fetch_failures: tuple[str, ...] = (),
) -> RankingsResult:
    """Pure rankings core: coverage -> quarantine -> regime -> features -> rank.

    No I/O, no clock. build_rankings (live) and the M3 backtester's prepare()
    both call this, so backtest and live-paper rank identically by construction.
    """
    coverage = check_coverage([i.symbol for i in infos], bars, config.data.min_coverage)
    if not coverage.ok:
        raise PipelineDataError(
            f"universe coverage {coverage.ratio:.0%} below "
            f"{config.data.min_coverage:.0%}; missing: {', '.join(coverage.missing)}"
        )

    clean, quarantined = quarantine_outliers(
        bars, config.data.max_daily_move, config.data.quarantine_window_days
    )

    # A corrupt benchmark print must not silently flip venue-wide exposure/
    # regime: run it through the same recent-window sanity check universe
    # symbols get, but fail loudly instead of quietly excluding it.
    _, benchmark_quarantined = quarantine_outliers(
        {config.benchmark: benchmark_bars},
        config.data.max_daily_move,
        config.data.quarantine_window_days,
    )
    if benchmark_quarantined:
        raise PipelineDataError(
            f"benchmark {config.benchmark} failed data-sanity check: outlier move "
            f"within the trailing {config.data.quarantine_window_days}d window"
        )

    as_of_ts = pd.Timestamp(as_of, tz="UTC")
    regime = compute_regime(benchmark_bars, as_of_ts, config.regime)
    features = compute_features(clean, as_of_ts, config.signals)
    table = rank(features).copy()

    statuses = {i.symbol: i.status for i in infos}
    table.insert(0, "status", [statuses[s] for s in table.index])
    insufficient = tuple(sorted(set(clean) - set(table.index)))

    return RankingsResult(
        venue=config.name,
        as_of=as_of_ts,
        regime=regime,
        table=table,
        coverage=coverage,
        quarantined=quarantined,
        fetch_failures=fetch_failures,
        insufficient_history=insufficient,
        bars=clean,
        benchmark_bars=benchmark_bars,
    )
```

`build_rankings` keeps its fetch loop, `drop_incomplete_last_bar` handling, and benchmark fetch (through the `except ... raise PipelineDataError` around the benchmark), then ends with:

```python
    return assemble_rankings(
        config, infos, bars, benchmark, as_of, fetch_failures=tuple(sorted(failures))
    )
```

Delete the now-duplicated tail of `build_rankings` (the coverage check through the final `return RankingsResult(...)`, and the benchmark-sanity block — all of it now lives in `assemble_rankings`). The benchmark FETCH (and its reuse-from-bars shortcut and incomplete-bar drop) stays in `build_rankings`.

- [ ] **Step 4: Run the full suite**

Run: `uv run pytest -q`
Expected: all pass — this is a pure extraction; any pipeline test failure means the move dropped behavior.

- [ ] **Step 5: Lint and commit**

Run: `uv run ruff check . && uv run ruff format --check .`

```bash
git add src/trading/pipeline.py tests/test_pipeline.py
git commit -m "$(cat <<'EOF'
Extract pure assemble_rankings core from build_rankings [AI]

The M3 backtester replays the rankings sequence over pre-fetched,
pre-sliced frames thousands of times; extracting the pure core (no
fetching) lets backtest and live-paper rank through literally the same
function. No behavior change.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: Crypto deep history — ccxt backfill exchange behind the same adapter

Kraken's public OHLC endpoint serves only the ~720 most-recent daily candles (spec Open Item, live-verified). 2018+ backtests need a deeper source. Add a backfill path inside `CryptoAdapter.fetch_ohlcv` — same adapter interface, used automatically only when the requested `start` predates Kraken's window. Recent data still comes from Kraken. Precedence (documented, tested): **Kraken owns `[end - backfill_before_days, end]`; the backfill exchange owns older rows; Kraken wins any overlap.**

**Files:**
- Modify: `config/crypto.toml`, `config/equities.toml` (`[data]` fields)
- Modify: `src/trading/config.py` (DataConfig)
- Modify: `src/trading/venues/crypto.py`
- Modify: `tests/test_crypto_adapter.py`, `tests/test_config.py`

**Interfaces:**
- Consumes: existing `_kraken_fetch(pair, since_ms)`, `validate_ohlcv`, `DataFetchError`.
- Produces: `CryptoAdapter.fetch_ohlcv` (unchanged signature) now serves 2018+ ranges; `_backfill_fetch(exchange_id: str, pair: str, since_ms: int, limit: int) -> list[list[float]]` (network touchpoint for monkeypatching); `DataConfig` gains `backfill_exchange: str`, `backfill_page_limit: int`, `backfill_before_days: int`.

- [ ] **Step 1: Verify the backfill exchange live (one-off, no code)**

The spec leaves the exact ccxt exchange id open ("coinbase" spot vs "coinbaseexchange"; fall back to Bitstamp). Verify which serves public daily OHLCV without API keys, back to 2018, with forward pagination:

```bash
uv run python - <<'EOF'
import ccxt
SINCE_2018 = 1514764800000  # 2018-01-01T00:00:00Z
for name in ["coinbase", "coinbaseexchange", "bitstamp"]:
    try:
        exchange = getattr(ccxt, name)({"enableRateLimit": True})
        rows = exchange.fetch_ohlcv("BTC/USD", timeframe="1d", since=SINCE_2018, limit=300)
        first = rows[0][0] if rows else None
        print(f"{name}: {len(rows)} rows, first={first} ({'OK' if first and first <= SINCE_2018 + 7*86400000 else 'NOT FROM 2018'})")
    except Exception as exc:
        print(f"{name}: FAILED {type(exc).__name__}: {exc}")
EOF
```

Expected: at least one exchange prints `N rows, first=~1514764800000 (OK)` — rows starting at/near 2018-01-01, meaning it honors `since` and paginates forward. **Pick the first that passes, in the order tried.** Record the winner and its observed page size; they become `backfill_exchange` and `backfill_page_limit` in Step 4. If a candidate returns rows anchored to the present (ignores `since`), it fails the check — move on. If all three fail, stop and consult the developer (spec Open Item escalation), do not improvise a fourth source.

- [ ] **Step 2: Write the failing tests**

Append to `tests/test_crypto_adapter.py` (it already has a monkeypatch pattern for `_kraken_fetch`; follow it):

```python
def _daily_rows(start: str, periods: int, price: float) -> list[list[float]]:
    base = pd.Timestamp(start, tz="UTC")
    return [
        [int((base + pd.Timedelta(i, unit="D")).timestamp() * 1000), price, price, price, price, 1e6]
        for i in range(periods)
    ]


def test_deep_request_splices_backfill_and_kraken_with_kraken_precedence(monkeypatch):
    config = load_venue_config("crypto", Path("config"))
    # Kraken serves only from the boundary; backfill serves 2018 up to and
    # INCLUDING the boundary day at a different price -- Kraken must win it.
    boundary = datetime.date(2026, 7, 1) - datetime.timedelta(
        days=config.data.backfill_before_days
    )
    kraken_rows = _daily_rows(boundary.isoformat(), 30, price=200.0)
    deep_rows = _daily_rows("2018-01-01", (boundary - datetime.date(2018, 1, 1)).days + 1, price=100.0)

    def fake_kraken(pair, since_ms):
        return [r for r in kraken_rows if r[0] >= since_ms][:720]

    def fake_backfill(exchange_id, pair, since_ms, limit):
        assert exchange_id == config.data.backfill_exchange
        return [r for r in deep_rows if r[0] >= since_ms][:limit]

    monkeypatch.setattr("trading.venues.crypto._kraken_fetch", fake_kraken)
    monkeypatch.setattr("trading.venues.crypto._backfill_fetch", fake_backfill)
    adapter = CryptoAdapter(config)
    # end = 2026-07-01 so the adapter's internal boundary (end - backfill_before_days)
    # equals the `boundary` computed above.
    df = adapter.fetch_ohlcv("BTC", datetime.date(2018, 1, 1), datetime.date(2026, 7, 1))

    assert df.index[0] == pd.Timestamp("2018-01-01", tz="UTC")
    boundary_ts = pd.Timestamp(boundary, tz="UTC")
    assert float(df.loc[boundary_ts, "close"]) == 200.0  # Kraken wins the overlap
    assert float(df.loc[boundary_ts - pd.Timedelta(1, unit="D"), "close"]) == 100.0
    assert df.index.is_monotonic_increasing and not df.index.duplicated().any()


def test_recent_request_never_touches_backfill(monkeypatch):
    config = load_venue_config("crypto", Path("config"))
    kraken_rows = _daily_rows("2026-05-01", 62, price=200.0)

    def fake_kraken(pair, since_ms):
        return [r for r in kraken_rows if r[0] >= since_ms][:720]

    def forbidden(exchange_id, pair, since_ms, limit):
        raise AssertionError("backfill must not be called for a recent window")

    monkeypatch.setattr("trading.venues.crypto._kraken_fetch", fake_kraken)
    monkeypatch.setattr("trading.venues.crypto._backfill_fetch", forbidden)
    adapter = CryptoAdapter(config)
    df = adapter.fetch_ohlcv("BTC", datetime.date(2026, 5, 1), datetime.date(2026, 7, 1))
    assert len(df) == 62


def test_backfill_pair_missing_falls_back_to_kraken_only(monkeypatch):
    config = load_venue_config("crypto", Path("config"))
    kraken_rows = _daily_rows("2026-05-01", 62, price=200.0)

    def fake_kraken(pair, since_ms):
        return [r for r in kraken_rows if r[0] >= since_ms][:720]

    def missing_pair(exchange_id, pair, since_ms, limit):
        raise DataFetchError(f"{pair} not listed on {exchange_id}")

    monkeypatch.setattr("trading.venues.crypto._kraken_fetch", fake_kraken)
    monkeypatch.setattr("trading.venues.crypto._backfill_fetch", missing_pair)
    adapter = CryptoAdapter(config)
    # Deep request, but the pair only exists on Kraken: short history, no error.
    df = adapter.fetch_ohlcv("BTC", datetime.date(2018, 1, 1), datetime.date(2026, 7, 1))
    assert df.index[0] == pd.Timestamp("2026-05-01", tz="UTC")
```

Add missing imports to the test file (`datetime`, `DataFetchError` from `trading.venues.base`).

- [ ] **Step 3: Run to verify they fail**

Run: `uv run pytest tests/test_crypto_adapter.py -v -k "backfill or splices"`
Expected: FAIL — `AttributeError: ... has no attribute '_backfill_fetch'` / `TypeError` on unknown `[data]` keys (config not yet updated).

- [ ] **Step 4: Add the TOML fields and DataConfig fields**

`config/crypto.toml`, append to `[data]` (substitute the exchange id and page limit verified in Step 1):

```toml
# --- M3 deep-history backfill (spec Open Item: Kraken caps daily history) ---
# Kraken's public OHLC endpoint serves only the ~720 most-recent daily candles.
# Requests starting earlier than (end - backfill_before_days) fetch the older
# rows from backfill_exchange via ccxt. Precedence: Kraken owns
# [end - backfill_before_days, end]; the backfill exchange owns older rows;
# Kraken wins any overlap. Live rankings (history_days = 500) never hit this.
backfill_exchange = "coinbase"    # ccxt exchange id verified at implementation; "" disables
backfill_page_limit = 300         # candles per page on the backfill exchange
backfill_before_days = 700        # inside Kraken's ~720-candle window, with margin
```

`config/equities.toml`, append to `[data]`:

```toml
# Deep-history backfill is a crypto concern (Kraken's 720-candle cap);
# yfinance serves full adjusted history directly. Disabled here.
backfill_exchange = ""
backfill_page_limit = 0
backfill_before_days = 0
```

`src/trading/config.py`, extend `DataConfig`:

```python
@dataclass(frozen=True)
class DataConfig:
    cache_dir: str
    refetch_days: int
    min_coverage: float
    max_daily_move: float
    history_days: int
    quarantine_window_days: int
    drop_incomplete_last_bar: bool
    backfill_exchange: str  # ccxt exchange id for pre-Kraken-window rows; "" disables
    backfill_page_limit: int
    backfill_before_days: int
```

Append to `tests/test_config.py`'s existing per-venue assertions:

```python
def test_backfill_config_loaded():
    crypto = load_venue_config("crypto", Path("config"))
    assert crypto.data.backfill_exchange != ""
    assert crypto.data.backfill_page_limit > 0
    assert crypto.data.backfill_before_days > 0
    equities = load_venue_config("equities", Path("config"))
    assert equities.data.backfill_exchange == ""
```

- [ ] **Step 5: Implement the backfill path in `src/trading/venues/crypto.py`**

Replace the module's fetch section (keep `_KRAKEN_DAILY_LIMIT` and `_kraken_fetch` exactly as they are) with a shared pagination helper, the new touchpoint, and the split `fetch_ohlcv`:

```python
def _backfill_fetch(exchange_id: str, pair: str, since_ms: int, limit: int) -> list[list[float]]:
    """One page of daily candles from the deep-history exchange (spec Open
    Item: Kraken caps daily history at ~720 candles). Network touchpoint,
    isolated for monkeypatching."""
    import ccxt

    exchange = getattr(ccxt, exchange_id)({"enableRateLimit": True})
    try:
        return exchange.fetch_ohlcv(pair, timeframe="1d", since=since_ms, limit=limit)
    except ccxt.BaseError as e:
        raise DataFetchError(f"{exchange_id} fetch failed for {pair}: {e}") from e


def _paginate(
    fetch_page: Callable[[int], list[list[float]]],
    start: datetime.date,
    end: datetime.date,
) -> list[list[float]]:
    """Stitch forward-paged OHLCV rows over [start, end]. Progress is
    guaranteed: a page adding nothing new ends the loop."""
    since_ms = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
    end_ms = int(pd.Timestamp(end, tz="UTC").timestamp() * 1000)
    rows: list[list[float]] = []
    while True:
        page = fetch_page(since_ms)
        new = [r for r in page if not rows or r[0] > rows[-1][0]]
        if not new:
            break
        rows.extend(new)
        if new[-1][0] >= end_ms:
            break
        since_ms = int(new[-1][0]) + 1
    return rows


def _rows_to_frame(rows: list[list[float]]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=["timestamp", *OHLCV_COLUMNS])
    df.index = pd.to_datetime(df.pop("timestamp"), unit="ms", utc=True)
    df.index.name = None
    return df.astype("float64").sort_index()
```

`fetch_ohlcv` becomes:

```python
    def fetch_ohlcv(self, symbol: str, start: datetime.date, end: datetime.date) -> pd.DataFrame:
        import ccxt

        pair = f"{symbol}/USD"
        cfg = self._config.data
        boundary = end - datetime.timedelta(days=cfg.backfill_before_days)
        frames: list[pd.DataFrame] = []
        kraken_start = start

        if cfg.backfill_exchange and start < boundary:
            # Deep request: rows before the boundary come from the backfill
            # exchange; Kraken owns [boundary, end]. A pair missing there is
            # not an error -- it simply has Kraken-depth history only.
            try:
                deep_rows = _paginate(
                    lambda since_ms: _backfill_fetch(
                        cfg.backfill_exchange, pair, since_ms, cfg.backfill_page_limit
                    ),
                    start,
                    boundary,
                )
                if deep_rows:
                    frames.append(_rows_to_frame(deep_rows))
            except DataFetchError:
                pass
            kraken_start = boundary

        try:
            kraken_rows = _paginate(
                lambda since_ms: _kraken_fetch(pair, since_ms), kraken_start, end
            )
        except ccxt.BaseError as e:
            raise DataFetchError(f"kraken fetch failed for {pair}: {e}") from e
        if kraken_rows:
            frames.append(_rows_to_frame(kraken_rows))

        if not frames:
            raise DataFetchError(f"no crypto data for {pair}")
        df = pd.concat(frames)
        # Kraken appended last: keep="last" makes Kraken win any overlap (the
        # documented splice precedence; both sources are spot USD prices).
        df = df[~df.index.duplicated(keep="last")].sort_index()
        df = df.loc[pd.Timestamp(start, tz="UTC") : pd.Timestamp(end, tz="UTC")]
        return validate_ohlcv(df)
```

Add `from collections.abc import Callable` to the module imports. Note the existing `_kraken_fetch` already raises `ccxt.BaseError` subclasses; the wrap above preserves the current error message shape. The cache layer (`OhlcvCache`) needs no change: it persists whatever fetch returns, and its trailing-`refetch_days` refetch window (30d) sits entirely inside the Kraken-owned range, so warm refreshes never re-hit the backfill exchange.

- [ ] **Step 6: Run the tests**

Run: `uv run pytest tests/test_crypto_adapter.py tests/test_config.py -v`
Expected: all pass, including the pre-existing Kraken pagination tests (the pagination loop moved but its semantics are identical).

- [ ] **Step 7: Full suite, lint, commit**

Run: `uv run pytest -q && uv run ruff check . && uv run ruff format --check .`

```bash
git add config/crypto.toml config/equities.toml src/trading/config.py src/trading/venues/crypto.py tests/test_crypto_adapter.py tests/test_config.py
git commit -m "$(cat <<'EOF'
Add deep-history crypto backfill behind the same adapter [AI]

Kraken serves only ~720 recent daily candles (spec Open Item); 2018+
backtests fetch older rows from a second ccxt exchange inside
CryptoAdapter.fetch_ohlcv. Precedence: Kraken owns the trailing
backfill_before_days window, backfill owns older rows, Kraken wins
overlaps. Live 500-day requests never touch the backfill path.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 5: Point-in-time equities universe

`universe(as_of)` must return actual S&P 500 + NDX membership as-of any date since 2018 — backtesting today's members over the past is prohibited (spec). Sources: S&P 500 from the free `fja05680/sp500` GitHub dataset (raw file snapshotted in-repo); NDX reconstructed from Wikipedia's Nasdaq-100 change history by a committed, self-validating build script. Output is one committed intervals CSV the adapter reads.

**Files:**
- Create: `src/trading/venues/universes/sources/sp500_history.csv` (downloaded snapshot)
- Create: `src/trading/venues/universes/sources/PROVENANCE.md`
- Create: `scripts/build_pit_membership.py`
- Create: `src/trading/venues/universes/equities_membership.csv` (generated, committed)
- Modify: `src/trading/venues/equities.py`
- Modify: `tests/test_equities_adapter.py`

**Interfaces:**
- Consumes: `SymbolInfo` from `trading.venues.base`.
- Produces: `EquitiesAdapter.universe(as_of: datetime.date) -> list[SymbolInfo]` now point-in-time; `EquitiesAdapter.__init__(config, membership_csv: Path | None = None)` (the old `universe_csv` parameter is removed — the static snapshot is no longer an adapter input). Membership CSV columns: `symbol,index,start,end` (`start` inclusive, `end` exclusive, empty `end` = current member; `index` is `sp500` or `ndx`, informational).

- [ ] **Step 1: Snapshot the S&P 500 dataset and record provenance**

```bash
mkdir -p src/trading/venues/universes/sources
gh api repos/fja05680/sp500/contents --jq '.[].name'
```

Expected: a file list including one named like `S&P 500 Historical Components & Changes(MM-DD-YYYY).csv` (the date varies). Download that exact file and capture the pinned commit:

```bash
gh api "repos/fja05680/sp500/commits?per_page=1" --jq '.[0].sha'
curl -L -o src/trading/venues/universes/sources/sp500_history.csv \
  "https://raw.githubusercontent.com/fja05680/sp500/master/<URL-ENCODED-FILENAME-FROM-ABOVE>"
head -2 src/trading/venues/universes/sources/sp500_history.csv
```

Expected: a header row containing a date column and a tickers column, then rows like `1996-01-02,"AAPL,ABT,..."`. Also check the repo licence:

```bash
gh api repos/fja05680/sp500/license --jq '.license.spdx_id' || echo "NO LICENSE FILE"
```

Write `src/trading/venues/universes/sources/PROVENANCE.md` with exactly this structure, filling in the observed values:

```markdown
# Point-in-time universe data provenance

## S&P 500 membership
- Source: https://github.com/fja05680/sp500
- File: `S&P 500 Historical Components & Changes(<date>).csv` -> snapshotted as `sp500_history.csv`
- Pinned commit: `<sha from gh api>`
- Retrieved: 2026-07-05 (UTC)
- Licence: <spdx id, or "none declared — community-maintained dataset, snapshotted for
  personal research use; not redistributed beyond this private repo">

## Nasdaq-100 membership
- Source: https://en.wikipedia.org/wiki/Nasdaq-100 (current constituents + yearly
  change tables), retrieved by scripts/build_pit_membership.py
- Page revision: <permanent oldid URL from the page's "View history" at retrieval time>
- Retrieved: 2026-07-05 (UTC)
- Licence: CC BY-SA 4.0 (Wikipedia text/data)

## Output
- `../equities_membership.csv` — merged intervals, regenerated only by
  `uv run python scripts/build_pit_membership.py`; treat as frozen data, review diffs.

## Known limitations (annotated on backtest results)
- Ticker renames (FB->META, ANTM->ELV, ...) appear as remove+add: the backtest
  force-exits on the rename date. Conservative, and rare enough to accept.
- Residual survivorship: delisted tickers absent from yfinance are counted per
  session and reported as the coverage ratio on every equities result.
```

- [ ] **Step 2: Write the failing adapter tests**

Replace the universe-related tests in `tests/test_equities_adapter.py` (keep the `fetch_ohlcv` tests untouched) with:

```python
MEMBERSHIP_CSV = """# test fixture
symbol,index,start,end
AAA,sp500,2018-01-01,
BBB,sp500,2018-01-01,2020-06-01
CCC,ndx,2020-06-01,
"""


def _adapter_with(tmp_path, text: str) -> EquitiesAdapter:
    path = tmp_path / "membership.csv"
    path.write_text(text)
    config = load_venue_config("equities", Path("config"))
    return EquitiesAdapter(config, membership_csv=path)


def test_universe_is_point_in_time(tmp_path):
    adapter = _adapter_with(tmp_path, MEMBERSHIP_CSV)
    in_2019 = {i.symbol for i in adapter.universe(datetime.date(2019, 1, 2))}
    in_2021 = {i.symbol for i in adapter.universe(datetime.date(2021, 1, 4))}
    assert in_2019 == {"AAA", "BBB"}
    assert in_2021 == {"AAA", "CCC"}
    assert all(i.status == "tradable" for i in adapter.universe(datetime.date(2021, 1, 4)))


def test_membership_interval_boundaries_start_inclusive_end_exclusive(tmp_path):
    adapter = _adapter_with(tmp_path, MEMBERSHIP_CSV)
    on_start = {i.symbol for i in adapter.universe(datetime.date(2020, 6, 1))}
    day_before = {i.symbol for i in adapter.universe(datetime.date(2020, 5, 31))}
    assert "CCC" in on_start and "BBB" not in on_start  # start inclusive, end exclusive
    assert "BBB" in day_before and "CCC" not in day_before


def test_committed_membership_file_sanity():
    # The real committed file: plausible sizes, and known index churn visible.
    adapter = EquitiesAdapter(load_venue_config("equities", Path("config")))
    for day, low, high in [
        (datetime.date(2018, 6, 1), 450, 650),
        (datetime.date(2022, 6, 1), 450, 650),
        (datetime.date(2026, 7, 1), 450, 650),
    ]:
        count = len(adapter.universe(day))
        assert low <= count <= high, f"{day}: {count} members"
    early = {i.symbol for i in adapter.universe(datetime.date(2018, 6, 1))}
    today = {i.symbol for i in adapter.universe(datetime.date(2026, 7, 1))}
    assert early != today
    assert len(early - today) > 20  # real churn: many 2018 members are gone
```

Add `import datetime` and `from pathlib import Path` / `load_venue_config` imports as needed.

- [ ] **Step 3: Run to verify they fail**

Run: `uv run pytest tests/test_equities_adapter.py -v`
Expected: FAIL — `TypeError: __init__() got an unexpected keyword argument 'membership_csv'`, plus a failure for the missing committed file.

- [ ] **Step 4: Implement `universe(as_of)` in `src/trading/venues/equities.py`**

Replace the constructor and `universe` (delete `DEFAULT_UNIVERSE_CSV` and the old CSV read; `scripts/build_equities_universe.py` and `universes/equities.csv` remain in-repo as the build script's current-membership cross-check, but the adapter no longer reads them):

```python
DEFAULT_MEMBERSHIP_CSV = Path(__file__).parent / "universes" / "equities_membership.csv"


class EquitiesAdapter:
    def __init__(self, config: VenueConfig, membership_csv: Path | None = None):
        self._config = config
        self._membership_csv = membership_csv or DEFAULT_MEMBERSHIP_CSV
        self._membership: pd.DataFrame | None = None

    def _load_membership(self) -> pd.DataFrame:
        # Cached in memory: the backtester calls universe() once per session
        # (~2100 times per prepared span); re-reading the CSV each call is waste.
        if self._membership is None:
            df = pd.read_csv(self._membership_csv, comment="#", dtype=str).fillna("")
            self._membership = df
        return self._membership

    def universe(self, as_of: datetime.date) -> list[SymbolInfo]:
        """Point-in-time S&P 500 + NDX membership as-of the given date (spec:
        backtesting today's members over the past is prohibited)."""
        df = self._load_membership()
        iso = as_of.isoformat()
        active = df[(df["start"] <= iso) & ((df["end"] == "") | (iso < df["end"]))]
        return [SymbolInfo(symbol=s, status="tradable") for s in sorted(set(active["symbol"]))]
```

ISO-8601 date strings compare correctly as strings — no datetime parsing needed on the hot path.

- [ ] **Step 5: Write `scripts/build_pit_membership.py`**

```python
"""Build the point-in-time equities membership file (spec: Venue Model).

Merges two sources into src/trading/venues/universes/equities_membership.csv:
- S&P 500: the snapshotted fja05680/sp500 dataset (date -> full ticker list).
- NDX: Wikipedia's Nasdaq-100 current constituents + yearly change tables,
  reconstructed backward from today.

Self-validating: aborts unless (a) every reconstructed date has a plausible
member count and (b) the merged current membership approximately matches the
M1 snapshot (universes/equities.csv). Treat the output as frozen data --
regenerate deliberately and review the diff.

Usage: uv run python scripts/build_pit_membership.py
"""

from __future__ import annotations

import datetime
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
UNIVERSES = ROOT / "src" / "trading" / "venues" / "universes"
SP500_SNAPSHOT = UNIVERSES / "sources" / "sp500_history.csv"
CURRENT_SNAPSHOT = UNIVERSES / "equities.csv"
OUTPUT = UNIVERSES / "equities_membership.csv"
WIKI_NDX_URL = "https://en.wikipedia.org/wiki/Nasdaq-100"
SINCE = "2017-01-01"  # a year of pad before the 2018 backtest span


def normalize(symbol: str) -> str:
    return str(symbol).strip().upper().replace(".", "-")


def sp500_intervals() -> dict[str, list[list[str]]]:
    """Snapshot rows are (date, full ticker list) -> per-symbol [start, end) intervals."""
    df = pd.read_csv(SP500_SNAPSHOT)
    date_col = next(c for c in df.columns if "date" in c.lower())
    tick_col = next(c for c in df.columns if "ticker" in c.lower())
    snapshots: list[tuple[str, set[str]]] = []
    for _, row in df.iterrows():
        raw = str(row[tick_col]).replace(";", ",")
        symbols = {normalize(s) for s in raw.split(",") if s.strip()}
        snapshots.append((str(row[date_col])[:10], symbols))
    snapshots.sort()

    intervals: dict[str, list[list[str]]] = {}
    active: dict[str, list[str]] = {}
    for date_iso, symbols in snapshots:
        for symbol in symbols - set(active):
            interval = [date_iso, ""]
            active[symbol] = interval
            intervals.setdefault(symbol, []).append(interval)
        for symbol in set(active) - symbols:
            active.pop(symbol)[1] = date_iso  # end exclusive: gone as of this snapshot
    return intervals


def fetch_ndx() -> tuple[set[str], list[tuple[str, str, str]]]:
    """Returns (current members, [(date_iso, 'added'|'removed', symbol), ...])."""
    tables = pd.read_html(WIKI_NDX_URL)
    current: set[str] = set()
    changes: list[tuple[str, str, str]] = []
    for table in tables:
        cols = [
            " ".join(str(part) for part in c) if isinstance(c, tuple) else str(c)
            for c in table.columns
        ]
        lower = [c.lower() for c in cols]
        ticker_cols = [i for i, c in enumerate(lower) if "ticker" in c or "symbol" in c]
        date_cols = [i for i, c in enumerate(lower) if "date" in c]
        if ticker_cols and not date_cols and len(table) > 80:
            current = {normalize(s) for s in table[cols[ticker_cols[0]]].astype(str)}
            continue
        if not date_cols:
            continue
        date_col = cols[date_cols[0]]
        for action in ("added", "removed"):
            for i, c in enumerate(lower):
                if action in c and ("ticker" in c or "symbol" in c):
                    for _, row in table.iterrows():
                        when = pd.to_datetime(row[date_col], errors="coerce")
                        symbol = normalize(row[cols[i]])
                        if pd.isna(when) or not symbol or symbol in ("NAN", "-"):
                            continue
                        changes.append((when.date().isoformat(), action, symbol))
    if not current:
        sys.exit("FATAL: could not locate the NDX constituents table on Wikipedia; "
                 "inspect pd.read_html output and adjust column matching.")
    return current, sorted(changes)


def ndx_intervals(current: set[str], changes: list[tuple[str, str, str]]) -> dict[str, list[list[str]]]:
    """Walk changes newest -> oldest, reconstructing membership backward from today."""
    intervals: dict[str, list[list[str]]] = {}
    end_open: dict[str, str] = {s: "" for s in current}  # symbol -> interval end (exclusive)
    for date_iso, action, symbol in sorted(changes, reverse=True):
        if date_iso < SINCE:
            break
        if action == "added" and symbol in end_open:
            intervals.setdefault(symbol, []).append([date_iso, end_open.pop(symbol)])
        elif action == "removed" and symbol not in end_open:
            end_open[symbol] = date_iso  # was a member until (exclusive) date_iso
    for symbol, end in end_open.items():
        intervals.setdefault(symbol, []).append([SINCE, end])  # member since before SINCE
    return intervals


def validate(rows: list[tuple[str, str, str, str]]) -> None:
    def members_on(day: str) -> set[str]:
        return {s for s, _, start, end in rows if start <= day and (end == "" or day < end)}

    for day, low, high in [("2018-06-01", 450, 650), ("2022-06-01", 450, 650), ("2026-07-01", 450, 650)]:
        count = len(members_on(day))
        if not low <= count <= high:
            sys.exit(f"FATAL: {count} members on {day}, expected {low}..{high}")
    snapshot = {
        normalize(s)
        for s in pd.read_csv(CURRENT_SNAPSHOT, comment="#")["symbol"].astype(str)
    }
    drift = len(snapshot ^ members_on(datetime.date.today().isoformat()))
    if drift > 15:
        sys.exit(f"FATAL: current membership differs from the M1 snapshot by {drift} symbols")


def main() -> None:
    merged: list[tuple[str, str, str, str]] = []
    for symbol, spans in sp500_intervals().items():
        for start, end in spans:
            if end == "" or end >= SINCE:
                merged.append((symbol, "sp500", max(start, SINCE), end))
    ndx_current, ndx_changes = fetch_ndx()
    for symbol, spans in ndx_intervals(ndx_current, ndx_changes).items():
        for start, end in spans:
            merged.append((symbol, "ndx", start, end))
    merged.sort()
    validate(merged)
    lines = [
        "# Point-in-time S&P 500 + NDX membership. GENERATED by scripts/build_pit_membership.py",
        "# Sources + licences: see sources/PROVENANCE.md. start inclusive, end exclusive, empty end = current.",
        "symbol,index,start,end",
    ]
    lines += [f"{s},{idx},{start},{end}" for s, idx, start, end in merged]
    OUTPUT.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUTPUT} ({len(merged)} intervals)")


if __name__ == "__main__":
    main()
```

Note: `pd.read_html` needs `lxml`, already pulled in transitively? It is NOT a declared dependency — check with `uv run python -c "import lxml"`. If that fails, add `lxml==5.4.0` to `[dependency-groups].dev` in `pyproject.toml` (it is a build-time script dependency, not a runtime one) and run `uv sync`.

- [ ] **Step 6: Generate the membership file and eyeball it**

```bash
uv run python scripts/build_pit_membership.py
head -8 src/trading/venues/universes/equities_membership.csv
grep -c "" src/trading/venues/universes/equities_membership.csv
grep "^META\|^FB," src/trading/venues/universes/equities_membership.csv
```

Expected: `wrote ... (N intervals)` with N roughly 700–1200; header comments then rows; FB shows an interval ending 2022 and META one starting 2022 (the rename appears as remove+add). If the script exits FATAL on the Wikipedia parse, inspect `pd.read_html(WIKI_NDX_URL)` interactively, adjust the column matching in `fetch_ndx`, and record the final matching logic — do not hand-edit the output CSV. Duplicate `(symbol,start)` rows across sp500/ndx are fine — the adapter dedupes by symbol.

- [ ] **Step 7: Run the adapter tests**

Run: `uv run pytest tests/test_equities_adapter.py -v`
Expected: all pass, including `test_committed_membership_file_sanity` against the real generated file. If the churn assertion (`len(early - today) > 20`) fails, the reconstruction is wrong — S&P 500 alone replaced ~130 names since 2018; investigate the interval builder, do not relax the test.

- [ ] **Step 8: Full suite, lint, commit**

Run: `uv run pytest -q && uv run ruff check . && uv run ruff format --check .`
Note: `tests/sim_helpers.py` and pipeline/runner/CLI tests construct adapters via `make_adapter(config)` with no extra args — unaffected. Any test that passed `universe_csv=` must be updated to the new fixture pattern from Step 2 (deliberate update).

```bash
git add src/trading/venues/universes/sources/ src/trading/venues/universes/equities_membership.csv scripts/build_pit_membership.py src/trading/venues/equities.py tests/test_equities_adapter.py pyproject.toml uv.lock
git commit -m "$(cat <<'EOF'
Point-in-time S&P 500 + NDX universe for equities [AI]

universe(as_of) now reads a committed membership-intervals CSV built
from the snapshotted fja05680/sp500 dataset plus Wikipedia's NDX change
history (provenance + licences recorded in sources/PROVENANCE.md).
Backtesting today's members over the past is now structurally
impossible (spec). Live behavior: as_of=today yields current members.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 6: `[backtest]` config section

All backtest numbers live in TOML: span start, holdout boundary, walk-forward window sizes, the two hyperparameter grids, session-coverage floor, annualization factor, stress segments.

**Files:**
- Modify: `src/trading/config.py`
- Modify: `config/equities.toml`, `config/crypto.toml`
- Modify: `tests/test_config.py`

**Interfaces:**
- Produces: `BacktestConfig` dataclass; `VenueConfig.backtest: BacktestConfig`. Consumed by Tasks 7–13. Field names used verbatim later: `start`, `holdout_start`, `train_months`, `test_months`, `entry_score_threshold_grid`, `stop_atr_multiple_grid`, `min_session_coverage`, `periods_per_year`, `stress_segments`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_config.py`:

```python
def test_backtest_config_loaded():
    for venue in ("equities", "crypto"):
        config = load_venue_config(venue, Path("config"))
        bt = config.backtest
        assert bt.start == datetime.date(2018, 1, 1)
        assert bt.holdout_start == datetime.date(2026, 1, 5)
        assert 24 <= bt.train_months <= 36 and bt.test_months == 3  # spec bounds
        assert len(bt.entry_score_threshold_grid) >= 3
        assert len(bt.stop_atr_multiple_grid) >= 3
        assert 0 < bt.min_session_coverage <= 1
        assert bt.stress_segments[0] == (datetime.date(2022, 1, 1), datetime.date(2022, 12, 31))
    assert load_venue_config("equities", Path("config")).backtest.periods_per_year == 252
    assert load_venue_config("crypto", Path("config")).backtest.periods_per_year == 365
```

Add `import datetime` to the test file if missing.

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_config.py::test_backtest_config_loaded -v`
Expected: FAIL — `AttributeError: 'VenueConfig' object has no attribute 'backtest'` (after a `KeyError: 'backtest'` once the loader change lands first — either failure shape is fine).

- [ ] **Step 3: Add the dataclass, loader, and TOML sections**

`src/trading/config.py` — add `import datetime` at the top, then:

```python
@dataclass(frozen=True)
class BacktestConfig:
    """Spec: Backtesting & Validation. The tunable surface is exactly two
    hyperparameters (entry_score_threshold, stop_atr_multiple); their grids
    live here. Everything else is set by design, not fitted."""

    start: datetime.date
    holdout_start: datetime.date  # final 6 months; touched exactly once via --holdout
    train_months: int
    test_months: int
    entry_score_threshold_grid: tuple[float, ...]
    stop_atr_multiple_grid: tuple[float, ...]
    min_session_coverage: float  # skip a session when fewer members have data
    periods_per_year: int  # Sharpe annualization: 252 sessions / 365 UTC days
    stress_segments: tuple[tuple[datetime.date, datetime.date], ...]
```

Extend `VenueConfig` with `backtest: BacktestConfig` (after `data`). In `load_venue_config`, before the `return`:

```python
    backtest = dict(raw["backtest"])
    backtest["entry_score_threshold_grid"] = tuple(backtest["entry_score_threshold_grid"])
    backtest["stop_atr_multiple_grid"] = tuple(backtest["stop_atr_multiple_grid"])
    backtest["stress_segments"] = tuple(
        (datetime.date.fromisoformat(a), datetime.date.fromisoformat(b))
        for a, b in backtest["stress_segments"]
    )
```

and add `backtest=BacktestConfig(**backtest),` to the `VenueConfig(...)` construction. (Bare TOML dates like `start = 2018-01-01` parse to `datetime.date` natively via tomllib; the nested stress-segment strings need explicit parsing.)

`config/equities.toml`, append:

```toml
[backtest]  # spec: Backtesting & Validation
start = 2018-01-01
# Final holdout boundary: FIXED date (not rolling) so the untouched period is
# deterministic and auditable. Everything from here on is spent the first time
# `trading backtest --holdout` reads it.
holdout_start = 2026-01-05
train_months = 30            # spec: tune on 24-36 months
test_months = 3              # spec: test on the following 3 months, roll
entry_score_threshold_grid = [0.60, 0.65, 0.70, 0.75, 0.80]
stop_atr_multiple_grid = [1.0, 1.5, 2.0, 2.5]
# yfinance lacks most delisted tickers: early point-in-time sessions will not
# reach the live 90% coverage bar. Below this floor a session is skipped
# (state untouched, like a live coverage-failed run); the per-session ratio is
# annotated on every result (spec: survivorship).
min_session_coverage = 0.60
periods_per_year = 252
# The stitched OOS must fully cover at least one of these (spec: a bear market
# that only appears in training data does not count as tested).
stress_segments = [["2022-01-01", "2022-12-31"]]
```

`config/crypto.toml`, append:

```toml
[backtest]  # spec: Backtesting & Validation
start = 2018-01-01
holdout_start = 2026-01-05   # fixed boundary; see equities.toml note
train_months = 30
test_months = 3
entry_score_threshold_grid = [0.60, 0.65, 0.70, 0.75, 0.80]
stop_atr_multiple_grid = [1.0, 1.5, 2.0, 2.5]
min_session_coverage = 0.90  # deep history exists for listed coins; gaps mean data trouble
periods_per_year = 365       # 24/7 venue: UTC daily bars
stress_segments = [["2022-01-01", "2022-12-31"]]
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_config.py -v`
Expected: all pass. Then `uv run pytest -q` — all pass (`config_hash` now covers the new section, which is intentional: backtest-config changes are distinct experiments; nothing keys off a stored old hash).

- [ ] **Step 5: Lint and commit**

Run: `uv run ruff check . && uv run ruff format --check .`

```bash
git add src/trading/config.py config/equities.toml config/crypto.toml tests/test_config.py
git commit -m "$(cat <<'EOF'
Add [backtest] config section [AI]

Span start, fixed holdout boundary, walk-forward window sizes, the two
hyperparameter grids (the entire tunable surface, per spec), session
coverage floor, Sharpe annualization, and stress segments -- every
backtest number in TOML, none in code.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 7: Replay engine — `prepare()` + `replay()`

The heart of M3. `prepare()` (I/O): benchmark bars define the session calendar; point-in-time universe per session; one cached full-span fetch per symbol; per-session rankings precomputed via `assemble_rankings` (rankings never depend on the two tunable hyperparameters, so one `prepare()` serves every grid point). `replay()` (pure): fresh state, then `step()` per session over bars re-sliced to ≤ that session — the exact live loop, clocked by the session list.

**Files:**
- Create: `src/trading/backtest/__init__.py` (empty)
- Create: `src/trading/backtest/engine.py`
- Create: `tests/backtest_helpers.py`
- Create: `tests/test_backtest_engine.py`

**Interfaces:**
- Consumes: `assemble_rankings` (Task 3), `PipelineDataError`, `RankingsResult`, `OhlcvCache.fetch(symbol, start, end, fetch_fn)`, `step(state, rankings, config)`, `initial_state(venue, starting_balance, benchmark_start_price, created_at)`, `Fill`, `VenueAdapter`, `config.backtest` (Task 6).
- Produces (used by Tasks 8–13):
  - `prepare(config: VenueConfig, adapter: VenueAdapter, cache: OhlcvCache, start: datetime.date, end: datetime.date) -> PreparedBacktest`
  - `replay(prepared: PreparedBacktest, config: VenueConfig, *, start: datetime.date | None = None, end: datetime.date | None = None) -> BacktestResult`
  - `SessionPlan(ts, rankings, clean_symbols, survivorship_ratio, skip_reason)`
  - `PreparedBacktest(venue, start, end, sessions, bars, benchmark_bars, missing_symbols)`
  - `TradeRecord(symbol, qty, entry_ts, exit_ts, entry_price, exit_price, entry_fee, exit_fee, realized_pnl, reason)`
  - `BacktestResult(venue, start, end, equity_curve, benchmark_curve, trades, open_positions, fees_paid, buy_notional, sessions_run, sessions_skipped, survivorship_ratio, warnings)`
  - `BacktestError`, `CRYPTO_SURVIVORSHIP_CAVEAT`

- [ ] **Step 1: Create the test helpers (`tests/backtest_helpers.py`)**

`sim_helpers.frame` produces constant-drift (zero-volatility) series whose features are NaN — useless for driving real entries. Backtest tests need noisy frames, a date-aware fake adapter, a small-window config, and a hand-built `PreparedBacktest` builder:

```python
"""Shared builders for backtest tests: noisy frames, a date-aware fake
adapter, small-window configs, and hand-built PreparedBacktest fixtures."""

from __future__ import annotations

import datetime
from collections.abc import Callable
from dataclasses import replace

import numpy as np
import pandas as pd
from sim_helpers import CR, make_rankings

from trading.backtest.engine import PreparedBacktest, SessionPlan
from trading.config import VenueConfig
from trading.venues.base import SymbolInfo, VenueConstraints


def noisy_frame(
    *,
    seed: int,
    drift: float = 0.0,
    periods: int = 120,
    start: str = "2025-01-01",
    start_price: float = 100.0,
) -> pd.DataFrame:
    """Seeded random-walk OHLCV, daily UTC bars (24/7 style)."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=periods, freq="D", tz="UTC")
    returns = rng.normal(loc=drift, scale=0.02, size=periods)
    close = start_price * np.cumprod(1.0 + returns)
    open_ = np.concatenate([[start_price], close[:-1]])
    return pd.DataFrame(
        {
            "open": open_,
            "high": np.maximum(open_, close) * 1.01,
            "low": np.minimum(open_, close) * 0.99,
            "close": close,
            "volume": rng.uniform(5e5, 1.5e6, size=periods),
        },
        index=idx,
    )


def small_config(base: VenueConfig = CR, **backtest_overrides) -> VenueConfig:
    """Shrink every window so 120-bar fixtures rank and trade."""
    signals = replace(
        base.signals,
        momentum_windows=(3, 5, 8),
        vol_window=5,
        volume_week=3,
        volume_baseline=10,
        breakout_windows=(5, 8),
        rsi_window=5,
        mean_window=5,
        raw_return_days=5,
    )
    regime = replace(base.regime, sma_fast=5, sma_slow=15, vol_lookback=30)
    portfolio = replace(
        base.portfolio,
        atr_window=5,
        time_stop_bars=8,
        entry_score_threshold=0.55,
        min_raw_return_cost_multiple=0.0,
    )
    backtest_defaults = {
        "start": datetime.date(2025, 2, 1),
        "holdout_start": datetime.date(2027, 1, 1),
        "min_session_coverage": 0.5,
        "periods_per_year": 365,
    }
    backtest = replace(base.backtest, **{**backtest_defaults, **backtest_overrides})
    data = replace(base.data, history_days=25, backfill_exchange="")
    return replace(
        base, signals=signals, regime=regime, portfolio=portfolio, backtest=backtest, data=data
    )


class FakeBacktestAdapter:
    """In-memory venue: membership may vary by date; fetch slices stored frames."""

    def __init__(
        self,
        frames: dict[str, pd.DataFrame],
        benchmark: str,
        members_on: Callable[[datetime.date], list[str]] | None = None,
    ):
        self._frames = frames
        self._benchmark = benchmark
        self._members_on = members_on  # date -> list[str]; None = all non-benchmark symbols

    def universe(self, as_of: datetime.date) -> list[SymbolInfo]:
        if self._members_on is not None:
            names = self._members_on(as_of)
        else:
            names = [s for s in sorted(self._frames) if s != self._benchmark]
        return [SymbolInfo(symbol=s, status="tradable") for s in names]

    def constraints(self) -> VenueConstraints:
        return VenueConstraints(
            taker_fee_bps=95.0, maker_fee_bps=50.0, slippage_bps=5.0,
            settlement_days=0, trades_24_7=True,
        )

    def fetch_ohlcv(self, symbol: str, start: datetime.date, end: datetime.date) -> pd.DataFrame:
        df = self._frames[symbol]
        return df.loc[pd.Timestamp(start, tz="UTC") : pd.Timestamp(end, tz="UTC")]


def prepared_from_sessions(
    config: VenueConfig,
    session_specs: list[tuple[str, pd.DataFrame]],
    bars: dict[str, pd.DataFrame],
    benchmark: pd.DataFrame,
) -> PreparedBacktest:
    """Hand-built PreparedBacktest: full control over each session's table.

    session_specs: [(iso_date, table)] built with sim_helpers.make_table.
    """
    sessions = []
    for iso, table in session_specs:
        ts = pd.Timestamp(iso, tz="UTC")
        rankings = make_rankings(config, {s: bars[s].loc[:ts] for s in bars}, table)
        slim = replace(rankings, bars={}, benchmark_bars=benchmark.iloc[0:0])
        sessions.append(
            SessionPlan(
                ts=ts, rankings=slim, clean_symbols=tuple(bars),
                survivorship_ratio=1.0, skip_reason=None,
            )
        )
    return PreparedBacktest(
        venue=config.name,
        start=sessions[0].ts.date(),
        end=sessions[-1].ts.date(),
        sessions=tuple(sessions),
        bars=bars,
        benchmark_bars=benchmark,
        missing_symbols=(),
    )
```

(`make_rankings` fixes `as_of=AS_OF`; `step()` never reads `rankings.as_of`, so that is harmless here.)

- [ ] **Step 2: Write the failing engine tests (`tests/test_backtest_engine.py`)**

```python
import datetime
import math

import pandas as pd
import pytest
from backtest_helpers import (
    FakeBacktestAdapter,
    noisy_frame,
    prepared_from_sessions,
    small_config,
)
from sim_helpers import make_table

from trading.backtest.engine import (
    CRYPTO_SURVIVORSHIP_CAVEAT,
    BacktestError,
    prepare,
    replay,
)
from trading.data.cache import OhlcvCache

START = datetime.date(2025, 2, 1)
END = datetime.date(2025, 4, 30)


def _fixture_frames() -> dict[str, pd.DataFrame]:
    return {
        "AAA": noisy_frame(seed=1, drift=0.01),   # strong: should get entered
        "BBB": noisy_frame(seed=2, drift=0.002),
        "CCC": noisy_frame(seed=3, drift=0.0),
        "DDD": noisy_frame(seed=4, drift=-0.003),
        "BENCH": noisy_frame(seed=9, drift=0.003),
    }


def _prepare(tmp_path, frames=None, members_on=None):
    config = small_config()
    config = _with_benchmark(config)
    adapter = FakeBacktestAdapter(frames or _fixture_frames(), "BENCH", members_on)
    cache = OhlcvCache(tmp_path / "cache", config.data.refetch_days)
    return config, prepare(config, adapter, cache, START, END)


def _with_benchmark(config):
    from dataclasses import replace

    return replace(config, benchmark="BENCH")


def test_prepare_builds_a_session_per_benchmark_bar(tmp_path):
    config, prepared = _prepare(tmp_path)
    assert prepared.venue == config.name
    session_dates = [s.ts.date() for s in prepared.sessions]
    assert session_dates[0] >= START and session_dates[-1] <= END
    assert len(session_dates) == len(set(session_dates))
    assert all(s.rankings is not None or s.skip_reason for s in prepared.sessions)
    assert prepared.missing_symbols == ()


def test_replay_is_deterministic_and_trades(tmp_path):
    config, prepared = _prepare(tmp_path)
    first = replay(prepared, config)
    second = replay(prepared, config)
    assert first.trades == second.trades
    assert first.equity_curve.equals(second.equity_curve)
    assert first.sessions_run > 0
    assert len(first.equity_curve) == first.sessions_run
    # The fixture has a strongly drifting symbol and a risk-on benchmark:
    # the replay must actually trade. If this fails, raise AAA's drift.
    assert first.trades or first.open_positions


def test_no_lookahead_perturbing_future_data_never_changes_past_decisions(tmp_path):
    cutoff = pd.Timestamp("2025-03-15", tz="UTC")
    frames = _fixture_frames()
    perturbed = {}
    for symbol, df in frames.items():
        bumped = df.copy()
        after = bumped.index > cutoff
        bumped.loc[after, ["open", "high", "low", "close"]] *= 1.5
        perturbed[symbol] = bumped

    config, prepared_a = _prepare(tmp_path / "a", frames)
    _, prepared_b = _prepare(tmp_path / "b", perturbed)
    result_a = replay(prepared_a, config)
    result_b = replay(prepared_b, config)

    def fills_through(result):
        return [t for t in result.trades if pd.Timestamp(t.exit_ts) <= cutoff]

    assert fills_through(result_a) == fills_through(result_b)
    curve_a = result_a.equity_curve[result_a.equity_curve.index <= cutoff]
    curve_b = result_b.equity_curve[result_b.equity_curve.index <= cutoff]
    assert curve_a.equals(curve_b)


def test_member_leaving_universe_is_force_exited(tmp_path):
    drop_after = datetime.date(2025, 3, 15)

    def members_on(as_of: datetime.date) -> list[str]:
        names = ["AAA", "BBB", "CCC", "DDD"]
        return [n for n in names if n != "AAA" or as_of <= drop_after]

    config, prepared = _prepare(tmp_path, members_on=members_on)
    result = replay(prepared, config)
    forced = [t for t in result.trades if t.reason == "forced_exit"]
    assert any(t.symbol == "AAA" for t in forced), (
        "AAA (strongest drift) should be held when it leaves the universe and "
        "then force-exited; if it was never entered, raise its drift in the fixture"
    )


def test_thin_session_is_skipped_and_state_carries_over(tmp_path):
    thin_day = datetime.date(2025, 3, 10)

    ghosts = ["GHOST1", "GHOST2", "GHOST3", "GHOST4", "GHOST5"]

    def members_on(as_of: datetime.date) -> list[str]:
        if as_of == thin_day:
            return ["AAA", "BBB", "CCC", "DDD", *ghosts]  # 4 of 9 have data -> 0.44 < 0.5 floor
        return ["AAA", "BBB", "CCC", "DDD"]

    config, prepared = _prepare(tmp_path, members_on=members_on)
    skipped = [s for s in prepared.sessions if s.skip_reason is not None]
    assert [s.ts.date() for s in skipped] == [thin_day]
    assert skipped[0].survivorship_ratio == pytest.approx(4 / 9)
    result = replay(prepared, config)
    assert any(str(thin_day) in entry for entry in result.sessions_skipped)
    # GHOST symbols never had data: counted as survivorship gaps.
    assert set(prepared.missing_symbols) == set(ghosts)


def test_crypto_results_carry_survivorship_caveat(tmp_path):
    config, prepared = _prepare(tmp_path)  # small_config is crypto-based
    result = replay(prepared, config)
    assert CRYPTO_SURVIVORSHIP_CAVEAT in result.warnings
    assert 0.0 < result.survivorship_ratio <= 1.0


def test_replay_window_slicing_runs_fresh_state_per_window(tmp_path):
    config, prepared = _prepare(tmp_path)
    late = replay(prepared, config, start=datetime.date(2025, 4, 1))
    assert late.equity_curve.index[0] >= pd.Timestamp("2025-04-01", tz="UTC")
    # Fresh state: the window's first marked value is the starting balance
    # (no fills can exist on the first session -- there are no pending orders),
    # not the full run's marked value on that date.
    assert late.equity_curve.iloc[0] == pytest.approx(config.portfolio.starting_balance)


def test_replay_with_no_sessions_raises(tmp_path):
    config, prepared = _prepare(tmp_path)
    try:
        replay(prepared, config, start=datetime.date(2030, 1, 1), end=datetime.date(2030, 2, 1))
        raise AssertionError("expected BacktestError")
    except BacktestError:
        pass


def test_entry_exit_round_trip_pairs_into_trade_records():
    # Hand-built sessions: deterministic entry then stop-out, no feature math.
    config = _with_benchmark(small_config())
    bars = {s: noisy_frame(seed=i, drift=0.001) for i, s in enumerate(["AAA", "BBB"], start=1)}
    bench = noisy_frame(seed=9, drift=0.001)
    enter = make_table({
        "AAA": {"status": "tradable", "composite": 0.9, "raw_return_30d": 0.5},
        "BBB": {"status": "tradable", "composite": 0.2, "raw_return_30d": 0.1},
    })
    neutral = make_table({
        "AAA": {"status": "tradable", "composite": 0.9, "raw_return_30d": 0.5},
        "BBB": {"status": "tradable", "composite": 0.2, "raw_return_30d": 0.1},
    })
    force = make_table({
        "AAA": {"status": "untradable", "composite": 0.9, "raw_return_30d": 0.5},
        "BBB": {"status": "tradable", "composite": 0.2, "raw_return_30d": 0.1},
    })
    prepared = prepared_from_sessions(
        config,
        [("2025-03-01", enter), ("2025-03-02", neutral), ("2025-03-03", force),
         ("2025-03-04", neutral)],
        bars,
        bench,
    )
    result = replay(prepared, config)
    assert [t.symbol for t in result.trades] == ["AAA"]
    trade = result.trades[0]
    assert trade.reason == "forced_exit"
    assert trade.entry_fee > 0.0  # crypto taker fee, frozen at entry (Task 2)
    assert math.isclose(
        trade.realized_pnl,
        trade.qty * trade.exit_price - trade.exit_fee - trade.qty * trade.entry_price - trade.entry_fee,
        rel_tol=1e-9,
    )
    assert result.fees_paid > 0.0 and result.buy_notional > 0.0
```

- [ ] **Step 3: Run to verify failure**

Run: `uv run pytest tests/test_backtest_engine.py -v 2>&1 | tail -5`
Expected: collection error — `ModuleNotFoundError: No module named 'trading.backtest'`.

- [ ] **Step 4: Implement `src/trading/backtest/engine.py`** (and an empty `src/trading/backtest/__init__.py`)

```python
"""Backtest replay harness (spec: Backtesting & Validation).

One engine, replayed: prepare() owns all I/O (point-in-time universe, cached
deep-history fetch) and precomputes per-session rankings with the SAME
assemble_rankings the live pipeline uses; replay() is pure -- no I/O, no
clock -- and drives the SAME simulator step() as live-paper, session by
session, filling at next-bar opens. The session calendar is the benchmark's
own bar dates (equities: SPY trading days = the realized NYSE calendar;
crypto: BTC UTC daily bars). No lookahead: every input handed to the
simulator is sliced to bars <= that session's decision bar. Rankings depend
only on data + signal config -- never on the two tunable hyperparameters --
so one prepare() serves every walk-forward grid point.
"""

from __future__ import annotations

import datetime
import math
from dataclasses import dataclass, replace

import pandas as pd

from trading.config import VenueConfig
from trading.data.cache import OhlcvCache
from trading.pipeline import PipelineDataError, RankingsResult, assemble_rankings
from trading.simulator.core import step
from trading.simulator.fills import Fill
from trading.simulator.state import initial_state
from trading.venues.base import VenueAdapter

CRYPTO_SURVIVORSHIP_CAVEAT = (
    "crypto universe is today's Robinhood listing: coins delisted before today are "
    "absent (survivorship bias); listing dates are inferred from data availability"
)


class BacktestError(RuntimeError):
    pass


@dataclass(frozen=True)
class SessionPlan:
    ts: pd.Timestamp
    rankings: RankingsResult | None  # bars/benchmark_bars stripped (memory); None = skipped
    clean_symbols: tuple[str, ...]  # quarantine-passed members this session
    survivorship_ratio: float  # members-with-data / point-in-time members
    skip_reason: str | None


@dataclass(frozen=True)
class PreparedBacktest:
    venue: str
    start: datetime.date
    end: datetime.date
    sessions: tuple[SessionPlan, ...]
    bars: dict[str, pd.DataFrame]  # full-span frames; replay re-slices per session
    benchmark_bars: pd.DataFrame
    missing_symbols: tuple[str, ...]  # members no data source can serve (survivorship gaps)


@dataclass(frozen=True)
class TradeRecord:
    symbol: str
    qty: float
    entry_ts: str
    exit_ts: str
    entry_price: float
    exit_price: float
    entry_fee: float
    exit_fee: float
    realized_pnl: float  # includes the entry fee (spec Open Item, Task 2)
    reason: str


@dataclass(frozen=True)
class BacktestResult:
    venue: str
    start: datetime.date
    end: datetime.date
    equity_curve: pd.Series  # marked portfolio value per session run
    benchmark_curve: pd.Series  # buy-and-hold, scaled to the same starting balance
    trades: tuple[TradeRecord, ...]
    open_positions: tuple[str, ...]
    fees_paid: float
    buy_notional: float
    sessions_run: int
    sessions_skipped: tuple[str, ...]
    survivorship_ratio: float  # mean per-session coverage of point-in-time members
    warnings: tuple[str, ...]


def prepare(
    config: VenueConfig,
    adapter: VenueAdapter,
    cache: OhlcvCache,
    start: datetime.date,
    end: datetime.date,
) -> PreparedBacktest:
    warmup_start = start - datetime.timedelta(days=config.data.history_days)
    try:
        benchmark = cache.fetch(config.benchmark, warmup_start, end, adapter.fetch_ohlcv)
    except Exception as exc:
        raise BacktestError(f"benchmark {config.benchmark} fetch failed: {exc}") from exc
    session_index = [ts for ts in benchmark.index if start <= ts.date() <= end]
    if not session_index:
        raise BacktestError(f"no {config.benchmark} sessions between {start} and {end}")

    members_by_session = {ts: adapter.universe(ts.date()) for ts in session_index}
    union = sorted({i.symbol for infos in members_by_session.values() for i in infos})
    bars: dict[str, pd.DataFrame] = {}
    missing: list[str] = []
    for symbol in union:
        try:
            frame = cache.fetch(symbol, warmup_start, end, adapter.fetch_ohlcv)
        except Exception:
            frame = pd.DataFrame()
        if frame.empty:
            missing.append(symbol)  # survivorship gap: counted + annotated (spec)
        else:
            bars[symbol] = frame

    sessions: list[SessionPlan] = []
    for ts in session_index:
        infos = members_by_session[ts]
        available = [i for i in infos if i.symbol in bars and not bars[i.symbol].loc[:ts].empty]
        ratio = len(available) / len(infos) if infos else 0.0
        if ratio < config.backtest.min_session_coverage:
            sessions.append(
                SessionPlan(
                    ts=ts,
                    rankings=None,
                    clean_symbols=(),
                    survivorship_ratio=ratio,
                    skip_reason=(
                        f"coverage {ratio:.0%} below "
                        f"{config.backtest.min_session_coverage:.0%}"
                    ),
                )
            )
            continue
        sliced = {i.symbol: bars[i.symbol].loc[:ts] for i in available}
        try:
            rankings = assemble_rankings(config, available, sliced, benchmark.loc[:ts], ts.date())
        except PipelineDataError as exc:
            sessions.append(SessionPlan(ts, None, (), ratio, str(exc)))
            continue
        slim = replace(rankings, bars={}, benchmark_bars=benchmark.iloc[0:0])
        sessions.append(SessionPlan(ts, slim, tuple(rankings.bars), ratio, None))

    return PreparedBacktest(
        venue=config.name,
        start=start,
        end=end,
        sessions=tuple(sessions),
        bars=bars,
        benchmark_bars=benchmark,
        missing_symbols=tuple(missing),
    )


def replay(
    prepared: PreparedBacktest,
    config: VenueConfig,
    *,
    start: datetime.date | None = None,
    end: datetime.date | None = None,
) -> BacktestResult:
    """Pure: no I/O, no clock. The session list IS the backtest clock."""
    start = start or prepared.start
    end = end or prepared.end
    sessions = [s for s in prepared.sessions if start <= s.ts.date() <= end]
    if not sessions:
        raise BacktestError(f"no sessions between {start} and {end}")

    bench = prepared.benchmark_bars
    bench_window = bench[(bench.index >= sessions[0].ts) & (bench.index <= sessions[-1].ts)]
    if bench_window.empty:
        raise BacktestError("benchmark has no bars in the replay window")
    bench_start_close = float(bench_window["close"].iloc[0])
    state = initial_state(
        config.name,
        config.portfolio.starting_balance,
        bench_start_close,
        sessions[0].ts.isoformat(),
    )

    open_lots: dict[str, Fill] = {}
    trades: list[TradeRecord] = []
    values: dict[pd.Timestamp, float] = {}
    skipped: list[str] = []
    warnings: set[str] = set()
    fees_paid = 0.0
    buy_notional = 0.0

    for plan in sessions:
        if plan.rankings is None:
            # Live parity: a run below the coverage floor touches nothing.
            skipped.append(f"{plan.ts.date().isoformat()}: {plan.skip_reason}")
            continue
        held = set(state.positions) | {o.symbol for o in state.pending_orders}
        bars: dict[str, pd.DataFrame] = {}
        for symbol in set(plan.clean_symbols) | held:
            frame = prepared.bars.get(symbol)
            if frame is None:
                continue
            window = frame.loc[: plan.ts]
            if not window.empty:
                bars[symbol] = window
        table = plan.rankings.table
        extras = sorted((held - set(table.index)) & set(bars))
        if extras:
            # Held names that left the point-in-time universe but still trade:
            # inject as untradable -> the simulator's own forced-exit path sells
            # them next bar (spec: dropped from the venue universe). Appended
            # LAST with NaN composite so entry iteration is unaffected.
            table = table.copy()
            for symbol in extras:
                row = {column: math.nan for column in table.columns}
                row["status"] = "untradable"
                table.loc[symbol] = pd.Series(row)
        rankings = replace(
            plan.rankings, table=table, bars=bars, benchmark_bars=bench.loc[: plan.ts]
        )
        result = step(state, rankings, config)
        state = result.state
        values[plan.ts] = result.snapshot.value
        warnings.update(result.warnings)
        for fill in result.fills:
            fees_paid += fill.fee
            if fill.side == "buy":
                open_lots[fill.symbol] = fill
                buy_notional += fill.qty * fill.price
            else:
                lot = open_lots.pop(fill.symbol)
                trades.append(
                    TradeRecord(
                        symbol=fill.symbol,
                        qty=fill.qty,
                        entry_ts=lot.bar_ts,
                        exit_ts=fill.bar_ts,
                        entry_price=lot.price,
                        exit_price=fill.price,
                        entry_fee=lot.fee,
                        exit_fee=fill.fee,
                        realized_pnl=float(fill.realized_pnl),
                        reason=fill.reason,
                    )
                )

    if not values:
        raise BacktestError("every session in the window was skipped; nothing to report")

    last_ts = sessions[-1].ts
    for symbol in state.positions:
        frame = prepared.bars.get(symbol)
        if frame is None or frame.loc[: last_ts].index[-1] < last_ts:
            warnings.add(
                f"{symbol}: held at end with no current bar; marked at its last close "
                "(no liquidation print invented)"
            )
    if config.name == "crypto":
        warnings.add(CRYPTO_SURVIVORSHIP_CAVEAT)

    equity_curve = pd.Series(values).sort_index()
    benchmark_curve = (
        bench_window["close"] / bench_start_close * config.portfolio.starting_balance
    )
    ratios = [s.survivorship_ratio for s in sessions]
    return BacktestResult(
        venue=prepared.venue,
        start=start,
        end=end,
        equity_curve=equity_curve,
        benchmark_curve=benchmark_curve,
        trades=tuple(trades),
        open_positions=tuple(sorted(state.positions)),
        fees_paid=fees_paid,
        buy_notional=buy_notional,
        sessions_run=len(values),
        sessions_skipped=tuple(skipped),
        survivorship_ratio=sum(ratios) / len(ratios),
        warnings=tuple(sorted(warnings)),
    )
```

- [ ] **Step 5: Run the engine tests**

Run: `uv run pytest tests/test_backtest_engine.py -v`
Expected: all pass. Two tests assert the seeded fixture actually trades (`test_replay_is_deterministic_and_trades`, `test_member_leaving_universe_is_force_exited`); if either fails on "no trades", tune the FIXTURE (raise AAA's drift, lower `entry_score_threshold` in `small_config`) — never bend engine code to a fixture.

- [ ] **Step 6: Full suite, lint, commit**

Run: `uv run pytest -q && uv run ruff check . && uv run ruff format --check .`

Performance expectations (document only, no action): a full 2018-present equities `prepare()` is dominated by the first-run network fetch (hundreds of symbols) and per-session feature computation — expect tens of minutes cold, minutes warm; each `replay()` over the prepared span is ~1–2 minutes (bars re-slicing dominates). Walk-forward grid search reuses one `prepare()` — that reuse is the design's load-bearing optimization.

```bash
git add src/trading/backtest/ tests/backtest_helpers.py tests/test_backtest_engine.py
git commit -m "$(cat <<'EOF'
Add backtest replay engine: prepare() + replay() [AI]

prepare() (I/O) builds the session calendar from benchmark bar dates,
fetches point-in-time members once through the cache, and precomputes
per-session rankings via the same assemble_rankings live uses. replay()
(pure) drives the unmodified simulator step() per session over bars
sliced to <= that session: one engine, replayed, no lookahead. Held
names that leave the universe are injected as untradable so the
simulator's own forced-exit path handles them. Survivorship coverage
is tracked per session and annotated on every result.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 8: Metrics + gate

Per venue (spec): total/annualized return, max drawdown, annualized Sharpe of daily returns (0% cash), win rate, avg win/loss, turnover, fee drag as its own line, vs buy-and-hold benchmark over the identical period. Gate = higher Sharpe than benchmark AND positive total return.

**Files:**
- Create: `src/trading/backtest/metrics.py`
- Create: `tests/test_backtest_metrics.py`

**Interfaces:**
- Consumes: `BacktestResult`, `TradeRecord` (Task 7).
- Produces (used by Tasks 9–13): `BacktestMetrics` (fields: `total_return, annualized_return, max_drawdown, sharpe, win_rate, avg_win, avg_loss, trade_count, turnover, fees_paid, fee_drag, benchmark_total_return, benchmark_sharpe, gate_passed`); `sharpe_ratio(curve: pd.Series, periods_per_year: int) -> float`; `max_drawdown(curve: pd.Series) -> float`; `metrics_from_curves(equity, benchmark, trades, buy_notional, fees_paid, periods_per_year) -> BacktestMetrics`; `compute_metrics(result: BacktestResult, periods_per_year: int) -> BacktestMetrics`.

- [ ] **Step 1: Write the failing tests (`tests/test_backtest_metrics.py`)**

```python
import datetime
import math

import pandas as pd
import pytest

from trading.backtest.engine import BacktestResult, TradeRecord
from trading.backtest.metrics import (
    compute_metrics,
    max_drawdown,
    metrics_from_curves,
    sharpe_ratio,
)


def _curve(values: list[float], start: str = "2025-01-01") -> pd.Series:
    idx = pd.date_range(start, periods=len(values), freq="D", tz="UTC")
    return pd.Series(values, index=idx, dtype="float64")


def _trade(pnl: float) -> TradeRecord:
    return TradeRecord(
        symbol="AAA", qty=1.0, entry_ts="2025-01-02T00:00:00+00:00",
        exit_ts="2025-01-05T00:00:00+00:00", entry_price=100.0, exit_price=100.0 + pnl,
        entry_fee=0.5, exit_fee=0.5, realized_pnl=pnl, reason="stop_loss",
    )


def test_sharpe_hand_computed():
    # Daily returns: +1%, -1%, +1% -> mean 1/300, std known; 0% cash (spec).
    curve = _curve([100.0, 101.0, 99.99, 100.9899])
    returns = curve.pct_change().dropna()
    expected = float(returns.mean() / returns.std()) * math.sqrt(252)
    assert sharpe_ratio(curve, 252) == pytest.approx(expected)


def test_sharpe_degenerate_inputs_are_nan():
    assert math.isnan(sharpe_ratio(_curve([100.0]), 252))          # one point
    assert math.isnan(sharpe_ratio(_curve([100.0, 100.0, 100.0]), 252))  # zero vol


def test_max_drawdown_hand_computed():
    assert max_drawdown(_curve([100.0, 120.0, 90.0, 110.0])) == pytest.approx(0.25)
    assert max_drawdown(_curve([100.0, 110.0, 121.0])) == pytest.approx(0.0)


def test_metrics_from_curves_hand_computed():
    equity = _curve([1000.0, 1010.0, 1005.0, 1030.0])
    benchmark = _curve([1000.0, 1001.0, 1002.0, 1003.0])
    trades = (_trade(+30.0), _trade(+10.0), _trade(-15.0))
    m = metrics_from_curves(equity, benchmark, trades, buy_notional=2000.0,
                            fees_paid=12.5, periods_per_year=252)
    assert m.total_return == pytest.approx(0.03)
    years = 4 / 252
    assert m.annualized_return == pytest.approx(1.03 ** (1 / years) - 1)
    assert m.max_drawdown == pytest.approx((1010.0 - 1005.0) / 1010.0)
    assert m.win_rate == pytest.approx(2 / 3)
    assert m.avg_win == pytest.approx(20.0)
    assert m.avg_loss == pytest.approx(-15.0)
    assert m.trade_count == 3
    assert m.turnover == pytest.approx(2000.0 / float(equity.mean()) / years)
    assert m.fees_paid == 12.5
    assert m.fee_drag == pytest.approx(12.5 / 1000.0)
    assert m.benchmark_total_return == pytest.approx(0.003)


def test_gate_requires_higher_sharpe_and_positive_total_return():
    strong = _curve([1000.0, 1012.0, 1008.0, 1035.0])
    weak_bench = _curve([1000.0, 1001.0, 999.0, 1002.0])
    m = metrics_from_curves(strong, weak_bench, (), 0.0, 0.0, 252)
    assert m.gate_passed is True

    losing = _curve([1000.0, 995.0, 998.0, 990.0])
    m2 = metrics_from_curves(losing, weak_bench, (), 0.0, 0.0, 252)
    assert m2.gate_passed is False  # negative total return fails regardless of Sharpe

    flat = _curve([1000.0, 1000.0, 1000.0, 1000.0])  # NaN sharpe
    m3 = metrics_from_curves(flat, weak_bench, (), 0.0, 0.0, 252)
    assert m3.gate_passed is False


def test_compute_metrics_delegates_from_result():
    equity = _curve([1000.0, 1010.0, 1005.0, 1030.0])
    benchmark = _curve([1000.0, 1001.0, 1002.0, 1003.0])
    result = BacktestResult(
        venue="crypto", start=datetime.date(2025, 1, 1), end=datetime.date(2025, 1, 4),
        equity_curve=equity, benchmark_curve=benchmark, trades=(_trade(5.0),),
        open_positions=(), fees_paid=3.0, buy_notional=500.0, sessions_run=4,
        sessions_skipped=(), survivorship_ratio=1.0, warnings=(),
    )
    m = compute_metrics(result, 365)
    assert m.trade_count == 1
    assert m.fee_drag == pytest.approx(3.0 / 1000.0)
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_backtest_metrics.py -v 2>&1 | tail -3`
Expected: `ModuleNotFoundError: No module named 'trading.backtest.metrics'`.

- [ ] **Step 3: Implement `src/trading/backtest/metrics.py`**

```python
"""Backtest metrics + go/no-go gate (spec: Backtesting & Validation).

Pure math over equity/benchmark curves and the trade list. The gate metric is
defined by the spec: annualized Sharpe of daily returns (cash yielding 0%) vs
the benchmark over the identical period; "beats" = higher Sharpe AND positive
total return.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

from trading.backtest.engine import BacktestResult, TradeRecord


@dataclass(frozen=True)
class BacktestMetrics:
    total_return: float
    annualized_return: float
    max_drawdown: float
    sharpe: float
    win_rate: float
    avg_win: float
    avg_loss: float
    trade_count: int
    turnover: float  # annualized: total buy notional / mean equity / years
    fees_paid: float
    fee_drag: float  # fees as a fraction of starting equity (spec: own line)
    benchmark_total_return: float
    benchmark_sharpe: float
    gate_passed: bool  # sharpe > benchmark_sharpe AND total_return > 0 (spec)


def sharpe_ratio(curve: pd.Series, periods_per_year: int) -> float:
    """Annualized Sharpe of daily returns, cash yielding 0% (spec)."""
    returns = curve.pct_change().dropna()
    if len(returns) < 2:
        return math.nan
    std = float(returns.std())
    if std == 0.0:
        return math.nan
    return float(returns.mean()) / std * math.sqrt(periods_per_year)


def max_drawdown(curve: pd.Series) -> float:
    if curve.empty:
        return math.nan
    return float((1.0 - curve / curve.cummax()).max())


def metrics_from_curves(
    equity: pd.Series,
    benchmark: pd.Series,
    trades: tuple[TradeRecord, ...],
    buy_notional: float,
    fees_paid: float,
    periods_per_year: int,
) -> BacktestMetrics:
    total = float(equity.iloc[-1] / equity.iloc[0]) - 1.0
    years = len(equity) / periods_per_year
    annualized = (1.0 + total) ** (1.0 / years) - 1.0 if years > 0 and total > -1.0 else math.nan
    wins = [t.realized_pnl for t in trades if t.realized_pnl > 0]
    losses = [t.realized_pnl for t in trades if t.realized_pnl <= 0]
    sharpe = sharpe_ratio(equity, periods_per_year)
    benchmark_sharpe = sharpe_ratio(benchmark, periods_per_year)
    gate = (
        not math.isnan(sharpe)
        and not math.isnan(benchmark_sharpe)
        and sharpe > benchmark_sharpe
        and total > 0.0
    )
    return BacktestMetrics(
        total_return=total,
        annualized_return=annualized,
        max_drawdown=max_drawdown(equity),
        sharpe=sharpe,
        win_rate=len(wins) / len(trades) if trades else math.nan,
        avg_win=sum(wins) / len(wins) if wins else 0.0,
        avg_loss=sum(losses) / len(losses) if losses else 0.0,
        trade_count=len(trades),
        turnover=buy_notional / float(equity.mean()) / years if years > 0 else math.nan,
        fees_paid=fees_paid,
        fee_drag=fees_paid / float(equity.iloc[0]),
        benchmark_total_return=float(benchmark.iloc[-1] / benchmark.iloc[0]) - 1.0,
        benchmark_sharpe=benchmark_sharpe,
        gate_passed=gate,
    )


def compute_metrics(result: BacktestResult, periods_per_year: int) -> BacktestMetrics:
    return metrics_from_curves(
        result.equity_curve,
        result.benchmark_curve,
        result.trades,
        result.buy_notional,
        result.fees_paid,
        periods_per_year,
    )
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_backtest_metrics.py -v`
Expected: all pass.

- [ ] **Step 5: Full suite, lint, commit**

Run: `uv run pytest -q && uv run ruff check . && uv run ruff format --check .`

```bash
git add src/trading/backtest/metrics.py tests/test_backtest_metrics.py
git commit -m "$(cat <<'EOF'
Add backtest metrics and the spec gate metric [AI]

Total/annualized return, max drawdown, annualized Sharpe of daily
returns at 0% cash, win rate, avg win/loss, turnover, fee drag as its
own line, benchmark comparison over the identical period. Gate =
higher Sharpe than benchmark AND positive total return (spec).

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 9: Experiments journal

Every backtest run appends config hash + grid point + metrics to a per-venue experiments journal (reusing `trading.journal.Journal`); the experiment count is reported alongside any quoted result (spec: "50 experiments deep, 'OOS beats SPY' is selection, not signal"). The holdout once-only guard reads this journal.

**Files:**
- Create: `src/trading/backtest/experiments.py`
- Create: `tests/test_experiments.py`

**Interfaces:**
- Consumes: `Journal`, `config_hash` from `trading.journal`; `BacktestMetrics` (Task 8).
- Produces (used by Tasks 10–13): `experiments_journal(journal_root: Path, venue: str) -> Journal` (file `journal/experiments-<venue>.jsonl`); `log_experiment(journal, *, config, kind, start, end, metrics, ts, grid_point=None, survivorship_ratio=None, extra=None) -> None`; `experiment_count(journal: Journal, venue: str) -> int`; `prior_holdout(journal: Journal) -> dict | None`. Valid `kind` values: `"backtest"`, `"walk_forward_window"`, `"walk_forward"`, `"holdout"`.

- [ ] **Step 1: Write the failing tests (`tests/test_experiments.py`)**

```python
import datetime
import math
from pathlib import Path

from trading.backtest.experiments import (
    experiment_count,
    experiments_journal,
    log_experiment,
    prior_holdout,
)
from trading.backtest.metrics import BacktestMetrics
from trading.config import load_venue_config
from trading.journal import config_hash

CONFIG = load_venue_config("crypto", Path("config"))


def _metrics(sharpe: float = 1.0) -> BacktestMetrics:
    return BacktestMetrics(
        total_return=0.1, annualized_return=0.2, max_drawdown=0.05, sharpe=sharpe,
        win_rate=0.5, avg_win=10.0, avg_loss=-5.0, trade_count=4, turnover=2.0,
        fees_paid=3.0, fee_drag=0.003, benchmark_total_return=0.05,
        benchmark_sharpe=0.8, gate_passed=True,
    )


def test_log_and_count_experiments(tmp_path):
    journal = experiments_journal(tmp_path, "crypto")
    assert experiment_count(journal, "crypto") == 0
    for kind in ("backtest", "walk_forward_window", "walk_forward"):
        log_experiment(
            journal, config=CONFIG, kind=kind,
            start=datetime.date(2020, 1, 1), end=datetime.date(2020, 6, 30),
            metrics=_metrics(), ts="2026-07-05T00:00:00+00:00",
            grid_point={"entry_score_threshold": 0.7, "stop_atr_multiple": 1.5},
            survivorship_ratio=0.95,
        )
    assert experiment_count(journal, "crypto") == 3
    assert experiment_count(journal, "equities") == 0
    events = list(journal.events())
    assert events[0]["config_hash"] == config_hash(CONFIG)
    assert events[0]["grid_point"]["entry_score_threshold"] == 0.7
    assert events[0]["metrics"]["sharpe"] == 1.0
    assert events[0]["from"] == "2020-01-01" and events[0]["to"] == "2020-06-30"
    assert (tmp_path / "experiments-crypto.jsonl").exists()


def test_nan_metrics_are_json_null(tmp_path):
    journal = experiments_journal(tmp_path, "crypto")
    log_experiment(
        journal, config=CONFIG, kind="backtest",
        start=datetime.date(2020, 1, 1), end=datetime.date(2020, 6, 30),
        metrics=_metrics(sharpe=math.nan), ts="2026-07-05T00:00:00+00:00",
    )
    event = next(journal.events())
    assert event["metrics"]["sharpe"] is None  # NaN is not valid JSON


def test_prior_holdout_found_only_for_holdout_kind(tmp_path):
    journal = experiments_journal(tmp_path, "equities")
    log_experiment(
        journal, config=CONFIG, kind="backtest",
        start=datetime.date(2020, 1, 1), end=datetime.date(2020, 6, 30),
        metrics=_metrics(), ts="2026-07-05T00:00:00+00:00",
    )
    assert prior_holdout(journal) is None
    log_experiment(
        journal, config=CONFIG, kind="holdout",
        start=datetime.date(2026, 1, 5), end=datetime.date(2026, 7, 4),
        metrics=_metrics(), ts="2026-07-05T01:00:00+00:00",
    )
    prior = prior_holdout(journal)
    assert prior is not None and prior["ts"] == "2026-07-05T01:00:00+00:00"
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_experiments.py -v 2>&1 | tail -3`
Expected: `ModuleNotFoundError: No module named 'trading.backtest.experiments'`.

- [ ] **Step 3: Implement `src/trading/backtest/experiments.py`**

```python
"""Experiments journal (spec: Backtesting & Validation / Reporting).

Every backtest run -- plain, walk-forward window, walk-forward summary,
holdout -- appends config hash + grid point + metrics to a per-venue
append-only JSONL (reusing trading.journal.Journal). The experiment count is
reported alongside any quoted result: 50 experiments deep, "OOS beats SPY"
is selection, not signal. prior_holdout() backs the once-only holdout gate.
"""

from __future__ import annotations

import datetime
import math
from dataclasses import asdict
from pathlib import Path

from trading.backtest.metrics import BacktestMetrics
from trading.config import VenueConfig
from trading.journal import Journal, config_hash


def experiments_journal(journal_root: Path, venue: str) -> Journal:
    return Journal(journal_root / f"experiments-{venue}.jsonl")


def _json_safe(value: object) -> object:
    return None if isinstance(value, float) and math.isnan(value) else value


def log_experiment(
    journal: Journal,
    *,
    config: VenueConfig,
    kind: str,  # backtest | walk_forward_window | walk_forward | holdout
    start: datetime.date,
    end: datetime.date,
    metrics: BacktestMetrics,
    ts: str,  # ISO-8601 UTC, supplied by the CLI (the only clock reader)
    grid_point: dict | None = None,
    survivorship_ratio: float | None = None,
    extra: dict | None = None,
) -> None:
    journal.append(
        {
            "event": "experiment",
            "kind": kind,
            "venue": config.name,
            "ts": ts,
            "config_hash": config_hash(config),
            "grid_point": grid_point,
            "from": start.isoformat(),
            "to": end.isoformat(),
            "survivorship_ratio": survivorship_ratio,
            "metrics": {k: _json_safe(v) for k, v in asdict(metrics).items()},
            **(extra or {}),
        }
    )


def experiment_count(journal: Journal, venue: str) -> int:
    return sum(
        1
        for event in journal.events()
        if event.get("event") == "experiment" and event.get("venue") == venue
    )


def prior_holdout(journal: Journal) -> dict | None:
    last: dict | None = None
    for event in journal.events():
        if event.get("event") == "experiment" and event.get("kind") == "holdout":
            last = event
    return last
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_experiments.py -v`
Expected: all pass.

- [ ] **Step 5: Full suite, lint, commit**

Run: `uv run pytest -q && uv run ruff check . && uv run ruff format --check .`

```bash
git add src/trading/backtest/experiments.py tests/test_experiments.py
git commit -m "$(cat <<'EOF'
Add per-venue experiments journal [AI]

Every backtest run appends config hash + grid point + metrics to
journal/experiments-<venue>.jsonl (reusing the M2 Journal). The count
is the selection-pressure denominator quoted with every result; the
holdout once-only gate reads prior_holdout() from the same file.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 10: Walk-forward validation

Tune on `train_months` (24–36 per config), test on the following `test_months` untouched, roll. Grid = exactly the two TOML grids. Report stitched OOS segments only. The stitched OOS must fully cover at least one configured stress segment (2022 bear) or the run refuses.

**Files:**
- Create: `src/trading/backtest/walkforward.py`
- Create: `tests/test_walkforward.py`

**Interfaces:**
- Consumes: `PreparedBacktest`, `replay`, `BacktestError` (Task 7); `compute_metrics`, `metrics_from_curves`, `BacktestMetrics` (Task 8); `config.backtest` (Task 6).
- Produces (used by Task 12's CLI): `GridPoint(entry_score_threshold, stop_atr_multiple)`; `Window(train_start, train_end, test_start, test_end)` (ends exclusive); `WindowResult(window, best, train_metrics, test_result, test_metrics)`; `WalkForwardResult(windows, stitched_equity, stitched_benchmark, stitched_metrics, stress_segments_covered)`; `add_months(day, months) -> date`; `generate_windows(start, end, train_months, test_months) -> list[Window]`; `grid_points(config) -> list[GridPoint]`; `apply_grid_point(config, point) -> VenueConfig`; `run_walk_forward(prepared, config, *, start, end) -> WalkForwardResult`; `WalkForwardError`. Pure module: no I/O, no clock, no journal — the CLI journals.

- [ ] **Step 1: Write the failing tests (`tests/test_walkforward.py`)**

```python
import datetime

import pandas as pd
import pytest
from backtest_helpers import FakeBacktestAdapter, noisy_frame, small_config

from trading.backtest.engine import prepare
from trading.backtest.walkforward import (
    GridPoint,
    WalkForwardError,
    add_months,
    apply_grid_point,
    generate_windows,
    grid_points,
    run_walk_forward,
)
from trading.data.cache import OhlcvCache


def test_add_months_rolls_years_and_rejects_late_month_days():
    assert add_months(datetime.date(2018, 1, 1), 30) == datetime.date(2020, 7, 1)
    assert add_months(datetime.date(2025, 11, 15), 3) == datetime.date(2026, 2, 15)
    with pytest.raises(ValueError):
        add_months(datetime.date(2018, 1, 31), 1)


def test_generate_windows_rolls_by_test_months_and_only_full_windows():
    windows = generate_windows(
        datetime.date(2018, 1, 1), datetime.date(2021, 1, 1), train_months=24, test_months=3
    )
    # 2019-01-01 would need test_end 2021-04-01 > end: excluded (full windows only).
    assert [w.train_start for w in windows] == [
        datetime.date(2018, 1, 1),
        datetime.date(2018, 4, 1),
        datetime.date(2018, 7, 1),
        datetime.date(2018, 10, 1),
    ]
    for w in windows:
        assert w.train_end == add_months(w.train_start, 24)
        assert w.test_start == w.train_end
        assert w.test_end == add_months(w.test_start, 3)
        assert w.test_end <= datetime.date(2021, 1, 1)


def test_grid_points_are_the_two_toml_grids_only():
    config = small_config()
    points = grid_points(config)
    assert len(points) == len(config.backtest.entry_score_threshold_grid) * len(
        config.backtest.stop_atr_multiple_grid
    )
    tuned = apply_grid_point(config, GridPoint(0.61, 2.5))
    assert tuned.portfolio.entry_score_threshold == 0.61
    assert tuned.portfolio.stop_atr_multiple == 2.5
    # Nothing else moved: the tunable surface is exactly two hyperparameters.
    assert tuned.portfolio.position_size_pct == config.portfolio.position_size_pct
    assert tuned.signals == config.signals


def _prepared(tmp_path):
    # 14 months of daily bars so train=6 + test=2 rolls at least twice.
    frames = {
        "AAA": noisy_frame(seed=1, drift=0.008, periods=430, start="2025-01-01"),
        "BBB": noisy_frame(seed=2, drift=0.001, periods=430, start="2025-01-01"),
        "CCC": noisy_frame(seed=3, drift=-0.002, periods=430, start="2025-01-01"),
        "BENCH": noisy_frame(seed=9, drift=0.002, periods=430, start="2025-01-01"),
    }
    config = small_config(
        train_months=6,
        test_months=2,
        entry_score_threshold_grid=(0.5, 0.7),
        stop_atr_multiple_grid=(1.0, 2.0),
        stress_segments=((datetime.date(2025, 8, 1), datetime.date(2025, 9, 30)),),
    )
    from dataclasses import replace

    config = replace(config, benchmark="BENCH")
    adapter = FakeBacktestAdapter(frames, "BENCH")
    cache = OhlcvCache(tmp_path / "cache", config.data.refetch_days)
    prepared = prepare(
        config, adapter, cache, datetime.date(2025, 2, 1), datetime.date(2026, 2, 28)
    )
    return config, prepared


def test_walk_forward_stitches_oos_segments_only(tmp_path):
    config, prepared = _prepared(tmp_path)
    wf = run_walk_forward(
        prepared, config, start=datetime.date(2025, 2, 1), end=datetime.date(2026, 2, 28)
    )
    assert len(wf.windows) >= 2
    for wr in wf.windows:
        # Stitched curve contains ONLY test-window dates (OOS), never train dates.
        curve = wr.test_result.equity_curve
        assert curve.index[0].date() >= wr.window.test_start
        assert curve.index[-1].date() < wr.window.test_end
        assert wr.best in grid_points(config)
    stitched_dates = [ts.date() for ts in wf.stitched_equity.index]
    first_test = wf.windows[0].window.test_start
    assert min(stitched_dates) >= first_test
    assert wf.stitched_metrics.trade_count == sum(
        w.test_metrics.trade_count for w in wf.windows
    )
    assert wf.stress_segments_covered  # 2025-08..09 lies inside the OOS stitch
    assert len(wf.stitched_equity) == len(wf.stitched_benchmark)


def test_walk_forward_refuses_without_stress_coverage(tmp_path):
    config, prepared = _prepared(tmp_path)
    from dataclasses import replace

    no_stress_in_span = replace(
        config,
        backtest=replace(
            config.backtest,
            stress_segments=((datetime.date(2022, 1, 1), datetime.date(2022, 12, 31)),),
        ),
    )
    with pytest.raises(WalkForwardError, match="stress"):
        run_walk_forward(
            prepared,
            no_stress_in_span,
            start=datetime.date(2025, 2, 1),
            end=datetime.date(2026, 2, 28),
        )


def test_walk_forward_span_too_short_raises(tmp_path):
    config, prepared = _prepared(tmp_path)
    with pytest.raises(WalkForwardError, match="shorter"):
        run_walk_forward(
            prepared, config, start=datetime.date(2025, 2, 1), end=datetime.date(2025, 6, 1)
        )
```

Note: `small_config(**backtest_overrides)` passes overrides into `replace(base.backtest, ...)` — `train_months`, `test_months`, the grids, and `stress_segments` are all `BacktestConfig` fields, so this works as written (Task 7 defined the helper that way).

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_walkforward.py -v 2>&1 | tail -3`
Expected: `ModuleNotFoundError: No module named 'trading.backtest.walkforward'`.

- [ ] **Step 3: Implement `src/trading/backtest/walkforward.py`**

```python
"""Walk-forward validation (spec: Backtesting & Validation).

Tune on train_months, test on the following test_months untouched, roll by
test_months. The tunable surface is EXACTLY two hyperparameters, grid-searched
from TOML grids; selection on the train window is highest Sharpe (tiebreak:
higher total return, then lower threshold, then lower stop multiple --
deterministic). Only stitched OOS segments are reported; each test window
replays from a fresh initial state and the stitch chains their daily returns.
The stitched OOS must fully cover at least one configured stress segment.
Pure module: no I/O, no clock -- the CLI journals every window as an
experiment.
"""

from __future__ import annotations

import datetime
import math
from dataclasses import dataclass, replace

import pandas as pd

from trading.backtest.engine import BacktestResult, PreparedBacktest, replay
from trading.backtest.metrics import BacktestMetrics, compute_metrics, metrics_from_curves
from trading.config import VenueConfig


class WalkForwardError(RuntimeError):
    pass


@dataclass(frozen=True)
class GridPoint:
    entry_score_threshold: float
    stop_atr_multiple: float


@dataclass(frozen=True)
class Window:
    train_start: datetime.date
    train_end: datetime.date  # exclusive
    test_start: datetime.date
    test_end: datetime.date  # exclusive


@dataclass(frozen=True)
class WindowResult:
    window: Window
    best: GridPoint
    train_metrics: BacktestMetrics
    test_result: BacktestResult
    test_metrics: BacktestMetrics


@dataclass(frozen=True)
class WalkForwardResult:
    windows: tuple[WindowResult, ...]
    stitched_equity: pd.Series
    stitched_benchmark: pd.Series
    stitched_metrics: BacktestMetrics
    stress_segments_covered: tuple[str, ...]


def add_months(day: datetime.date, months: int) -> datetime.date:
    if day.day > 28:
        raise ValueError("walk-forward dates must use day-of-month <= 28")
    month = day.month - 1 + months
    return datetime.date(day.year + month // 12, month % 12 + 1, day.day)


def generate_windows(
    start: datetime.date, end: datetime.date, train_months: int, test_months: int
) -> list[Window]:
    windows: list[Window] = []
    cursor = start
    while True:
        train_end = add_months(cursor, train_months)
        test_end = add_months(train_end, test_months)
        if test_end > end:
            break  # only FULL test windows count as OOS
        windows.append(Window(cursor, train_end, train_end, test_end))
        cursor = add_months(cursor, test_months)
    return windows


def grid_points(config: VenueConfig) -> list[GridPoint]:
    return [
        GridPoint(threshold, stop)
        for threshold in config.backtest.entry_score_threshold_grid
        for stop in config.backtest.stop_atr_multiple_grid
    ]


def apply_grid_point(config: VenueConfig, point: GridPoint) -> VenueConfig:
    portfolio = replace(
        config.portfolio,
        entry_score_threshold=point.entry_score_threshold,
        stop_atr_multiple=point.stop_atr_multiple,
    )
    return replace(config, portfolio=portfolio)


def _selection_key(point: GridPoint, metrics: BacktestMetrics) -> tuple:
    sharpe = -math.inf if math.isnan(metrics.sharpe) else metrics.sharpe
    total = -math.inf if math.isnan(metrics.total_return) else metrics.total_return
    return (-sharpe, -total, point.entry_score_threshold, point.stop_atr_multiple)


def _stitch(curves: list[pd.Series], starting_balance: float) -> pd.Series:
    returns = pd.concat([c.pct_change().dropna() for c in curves]).sort_index()
    stitched = starting_balance * (1.0 + returns).cumprod()
    anchor = pd.Series([starting_balance], index=[curves[0].index[0]])
    return pd.concat([anchor, stitched]).sort_index()


def _covered_segments(config: VenueConfig, stitched: pd.Series) -> tuple[str, ...]:
    first, last = stitched.index[0].date(), stitched.index[-1].date()
    return tuple(
        f"{seg_start.isoformat()}..{seg_end.isoformat()}"
        for seg_start, seg_end in config.backtest.stress_segments
        if first <= seg_start and seg_end <= last
    )


def run_walk_forward(
    prepared: PreparedBacktest,
    config: VenueConfig,
    *,
    start: datetime.date,
    end: datetime.date,
) -> WalkForwardResult:
    bt = config.backtest
    windows = generate_windows(start, end, bt.train_months, bt.test_months)
    if not windows:
        raise WalkForwardError(
            f"span {start}..{end} is shorter than one train+test window "
            f"({bt.train_months}+{bt.test_months} months)"
        )
    one_day = datetime.timedelta(days=1)
    results: list[WindowResult] = []
    for window in windows:
        scored: dict[GridPoint, BacktestMetrics] = {}
        for point in grid_points(config):
            train = replay(
                prepared,
                apply_grid_point(config, point),
                start=window.train_start,
                end=window.train_end - one_day,
            )
            scored[point] = compute_metrics(train, bt.periods_per_year)
        best = min(scored, key=lambda p: _selection_key(p, scored[p]))
        test = replay(
            prepared,
            apply_grid_point(config, best),
            start=window.test_start,
            end=window.test_end - one_day,
        )
        results.append(
            WindowResult(window, best, scored[best], test, compute_metrics(test, bt.periods_per_year))
        )

    stitched_equity = _stitch(
        [r.test_result.equity_curve for r in results], config.portfolio.starting_balance
    )
    stitched_benchmark = _stitch(
        [r.test_result.benchmark_curve for r in results], config.portfolio.starting_balance
    )
    covered = _covered_segments(config, stitched_equity)
    if not covered:
        raise WalkForwardError(
            "stitched OOS does not fully cover any configured stress segment; a bear "
            "market that only ever appears in training data does not count as tested (spec)"
        )
    trades = tuple(t for r in results for t in r.test_result.trades)
    stitched_metrics = metrics_from_curves(
        stitched_equity,
        stitched_benchmark,
        trades,
        sum(r.test_result.buy_notional for r in results),
        sum(r.test_result.fees_paid for r in results),
        bt.periods_per_year,
    )
    return WalkForwardResult(
        windows=tuple(results),
        stitched_equity=stitched_equity,
        stitched_benchmark=stitched_benchmark,
        stitched_metrics=stitched_metrics,
        stress_segments_covered=covered,
    )
```

Note the stitched benchmark: equity and benchmark curves cover the same session dates, so their stitched lengths match (`test_walk_forward_stitches_oos_segments_only` pins this).

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_walkforward.py -v`
Expected: all pass. The `_prepared` fixture must produce ≥ 2 windows; if `generate_windows` yields fewer because the fixture span is off-by-one against month arithmetic, extend `periods` in the fixture — do not touch window logic to fit the fixture.

- [ ] **Step 5: Full suite, lint, commit**

Run: `uv run pytest -q && uv run ruff check . && uv run ruff format --check .`

```bash
git add src/trading/backtest/walkforward.py tests/test_walkforward.py
git commit -m "$(cat <<'EOF'
Add walk-forward validation with stitched OOS and stress gate [AI]

Rolls train/test windows from TOML sizes, grid-searches exactly the
two spec hyperparameters per train window (deterministic selection:
Sharpe, then total return, then smaller params), replays each test
window fresh, and reports stitched OOS only. Refuses to report a
stitch that does not fully cover a configured stress segment.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 11: `trading backtest` CLI (plain, `--walk-forward`, `--holdout` once-only)

Wire everything into the CLI following the existing patterns: subparser on the same parser, `--json` for machine output, rich tables otherwise, typed confirmations for dangerous actions, `cli.py` as the only clock reader. The CLI journals every run as an experiment and prints the experiment count with every result.

**Files:**
- Modify: `src/trading/cli.py`
- Create: `tests/test_backtest_cli.py`

**Interfaces:**
- Consumes: `prepare`, `replay`, `BacktestError`, `BacktestResult` (Task 7); `compute_metrics`, `BacktestMetrics` (Task 8); `experiments_journal`, `log_experiment`, `experiment_count`, `prior_holdout` (Task 9); `run_walk_forward`, `WalkForwardError`, `WalkForwardResult` (Task 10); `make_adapter`, `OhlcvCache`, `load_venue_config` (existing).
- Produces: `trading backtest --venue equities|crypto [--from DATE] [--to DATE] [--walk-forward] [--holdout] [--json] [--config-dir DIR] [--journal-dir DIR]`. Exit 0 on success (gate FAIL is information, not an error), 1 on any refusal/error.

Date semantics (locked): default `--from` = `[backtest].start`; default `--to` = **yesterday UTC** (today's daily bar may be in progress on either venue), and any later `--to` is clamped to yesterday. Non-holdout runs are additionally clamped to `holdout_start - 1 day` (with a printed note); a non-holdout `--from` on/after `holdout_start` is an error. `--holdout` runs `[holdout_start, --to]`, refuses if a prior holdout event exists unless the operator types `RERUN HOLDOUT`, and is journaled as kind `holdout`. `--walk-forward --holdout` together is an error (the holdout is a single evaluation with the tuned TOML params, not another search).

- [ ] **Step 1: Write the failing CLI tests (`tests/test_backtest_cli.py`)**

Follow `tests/test_cli.py`'s existing style (invoke `main([...])`, monkeypatch adapters and capture stdout via capsys):

```python
import datetime
import json
from pathlib import Path

import pytest
from backtest_helpers import FakeBacktestAdapter, noisy_frame, small_config

import trading.cli as cli
from trading.backtest.experiments import experiments_journal
from trading.cli import main


@pytest.fixture()
def backtest_env(tmp_path, monkeypatch):
    """A crypto-venue config dir + fake adapter wired through the real CLI."""
    frames = {
        "AAA": noisy_frame(seed=1, drift=0.008, periods=430, start="2025-01-01"),
        "BBB": noisy_frame(seed=2, drift=0.001, periods=430, start="2025-01-01"),
        "CCC": noisy_frame(seed=3, drift=-0.002, periods=430, start="2025-01-01"),
        "BTC": noisy_frame(seed=9, drift=0.002, periods=430, start="2025-01-01"),
    }
    config = small_config(
        train_months=6,
        test_months=2,
        entry_score_threshold_grid=(0.5, 0.7),
        stop_atr_multiple_grid=(1.0, 2.0),
        stress_segments=((datetime.date(2025, 8, 1), datetime.date(2025, 9, 30)),),
        start=datetime.date(2025, 2, 1),
        holdout_start=datetime.date(2025, 12, 1),
    )
    # Write the small config as a real TOML the CLI can load.
    _write_venue_toml(tmp_path / "config" / "crypto.toml", config, cache_dir=str(tmp_path / "cache"))
    adapter = FakeBacktestAdapter(frames, "BTC")
    monkeypatch.setattr(cli, "make_adapter", lambda config: adapter)
    monkeypatch.setattr(
        cli, "_utcnow", lambda: datetime.datetime(2026, 1, 15, 3, 0, tzinfo=datetime.UTC)
    )
    return tmp_path


def _write_venue_toml(path: Path, config, cache_dir: str) -> None:
    """Serialize a VenueConfig back to TOML for CLI-level tests."""
    from dataclasses import asdict

    raw = asdict(config)
    raw["data"]["cache_dir"] = cache_dir
    lines = ["[venue]", f'name = "{config.name}"', f'benchmark = "BTC"']
    for section in ("costs", "universe", "signals", "regime", "portfolio", "data", "backtest"):
        lines.append(f"[{section}]")
        for key, value in raw[section].items():
            lines.append(f"{key} = {_toml_value(value)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def _toml_value(value) -> str:
    # Two container shapes exist in VenueConfig: flat number lists (grids,
    # windows) and date-pair lists (stress_segments, rendered as string pairs).
    import datetime as dt

    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return f'"{value}"'
    if isinstance(value, dt.date):
        return value.isoformat()
    if isinstance(value, (list, tuple)):
        if value and isinstance(value[0], (list, tuple)):
            inner = ", ".join(f'["{a.isoformat()}", "{b.isoformat()}"]' for a, b in value)
        else:
            inner = ", ".join(_toml_value(v) for v in value)
        return f"[{inner}]"
    return str(value)


def test_backtest_json_reports_metrics_gate_and_experiment_count(backtest_env, capsys):
    rc = main([
        "backtest", "--venue", "crypto", "--json",
        "--from", "2025-03-01", "--to", "2025-06-30",
        "--config-dir", str(backtest_env / "config"),
        "--journal-dir", str(backtest_env / "journal"),
    ])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["venue"] == "crypto"
    assert "sharpe" in payload["metrics"] and "fee_drag" in payload["metrics"]
    assert payload["gate_passed"] in (True, False)
    assert payload["experiment_count"] == 1
    assert 0.0 < payload["survivorship_ratio"] <= 1.0
    journal = experiments_journal(backtest_env / "journal", "crypto")
    event = next(journal.events())
    assert event["kind"] == "backtest"
    assert event["grid_point"]["entry_score_threshold"] == 0.55  # TOML value, journaled


def test_backtest_clamps_to_before_holdout(backtest_env, capsys):
    rc = main([
        "backtest", "--venue", "crypto", "--json",
        "--from", "2025-03-01", "--to", "2026-01-10",  # inside the holdout
        "--config-dir", str(backtest_env / "config"),
        "--journal-dir", str(backtest_env / "journal"),
    ])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["to"] == "2025-11-30"  # holdout_start - 1 day


def test_backtest_from_inside_holdout_is_an_error(backtest_env, capsys):
    rc = main([
        "backtest", "--venue", "crypto",
        "--from", "2025-12-05",
        "--config-dir", str(backtest_env / "config"),
        "--journal-dir", str(backtest_env / "journal"),
    ])
    assert rc == 1
    assert "holdout" in capsys.readouterr().err.lower()


def test_walk_forward_journals_every_window_plus_summary(backtest_env, capsys):
    rc = main([
        "backtest", "--venue", "crypto", "--walk-forward", "--json",
        "--from", "2025-02-01", "--to", "2025-11-30",
        "--config-dir", str(backtest_env / "config"),
        "--journal-dir", str(backtest_env / "journal"),
    ])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["windows"], "expected at least one OOS window"
    assert payload["stress_segments_covered"]
    journal = experiments_journal(backtest_env / "journal", "crypto")
    kinds = [e["kind"] for e in journal.events()]
    assert kinds.count("walk_forward_window") == len(payload["windows"])
    assert kinds.count("walk_forward") == 1
    assert payload["experiment_count"] == len(kinds)


def test_holdout_runs_once_then_requires_typed_confirmation(backtest_env, capsys, monkeypatch):
    args = [
        "backtest", "--venue", "crypto", "--holdout", "--json",
        "--to", "2026-01-14",
        "--config-dir", str(backtest_env / "config"),
        "--journal-dir", str(backtest_env / "journal"),
    ]
    assert main(args) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["from"] == "2025-12-01"  # holdout_start
    journal = experiments_journal(backtest_env / "journal", "crypto")
    assert [e["kind"] for e in journal.events()] == ["holdout"]

    # Second invocation refuses without the typed phrase.
    monkeypatch.setattr("builtins.input", lambda prompt="": "no")
    assert main(args) == 1
    assert [e["kind"] for e in journal.events()] == ["holdout"]

    # And proceeds with it.
    monkeypatch.setattr("builtins.input", lambda prompt="": "RERUN HOLDOUT")
    assert main(args) == 0
    capsys.readouterr()
    assert [e["kind"] for e in journal.events()] == ["holdout", "holdout"]


def test_walk_forward_and_holdout_together_is_an_error(backtest_env, capsys):
    rc = main([
        "backtest", "--venue", "crypto", "--walk-forward", "--holdout",
        "--config-dir", str(backtest_env / "config"),
        "--journal-dir", str(backtest_env / "journal"),
    ])
    assert rc == 1
    assert "holdout" in capsys.readouterr().err.lower()
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_backtest_cli.py -v 2>&1 | tail -5`
Expected: `SystemExit: 2` from argparse — `invalid choice: 'backtest'`.

- [ ] **Step 3: Implement the subcommand in `src/trading/cli.py`**

Add imports:

```python
from trading.backtest.engine import BacktestError, BacktestResult, prepare, replay
from trading.backtest.experiments import (
    experiment_count,
    experiments_journal,
    log_experiment,
    prior_holdout,
)
from trading.backtest.metrics import BacktestMetrics, compute_metrics
from trading.backtest.walkforward import WalkForwardError, WalkForwardResult, run_walk_forward
```

In `build_parser()`, after the `rankings` block:

```python
    backtest = sub.add_parser("backtest", help="historical replay of the live simulator")
    backtest.add_argument("--venue", choices=VENUES, required=True)
    backtest.add_argument(
        "--from", dest="from_date", type=datetime.date.fromisoformat, default=None,
        help="start date (default: [backtest].start)",
    )
    backtest.add_argument(
        "--to", dest="to_date", type=datetime.date.fromisoformat, default=None,
        help="end date (default and cap: yesterday UTC)",
    )
    backtest.add_argument(
        "--walk-forward", action="store_true",
        help="tune the two hyperparameters per rolling window; report stitched OOS only",
    )
    backtest.add_argument(
        "--holdout", action="store_true",
        help="evaluate the final holdout ONCE with current TOML params (confirms on rerun)",
    )
    backtest.add_argument("--json", action="store_true", help="machine-readable output")
    backtest.add_argument("--config-dir", default="config", help="directory with <venue>.toml")
    backtest.add_argument("--journal-dir", default="journal", help="journal root")
```

Register `"backtest": _cmd_backtest` in `main()`'s handlers dict. Then the handler and its helpers:

```python
def _cmd_backtest(args: argparse.Namespace) -> int:
    if args.walk_forward and args.holdout:
        print(
            "ERROR: --walk-forward tunes; --holdout is a single evaluation of the "
            "tuned TOML params. They cannot be combined.",
            file=sys.stderr,
        )
        return 1
    config = load_venue_config(args.venue, Path(args.config_dir))
    bt = config.backtest
    now = _utcnow()
    yesterday = now.date() - datetime.timedelta(days=1)
    journal = experiments_journal(Path(args.journal_dir), args.venue)

    end = min(args.to_date or yesterday, yesterday)
    if args.holdout:
        prior = prior_holdout(journal)
        if prior is not None:
            print(
                f"Holdout already evaluated at {prior['ts']} (config {prior['config_hash']}, "
                f"result journaled). The holdout is spent the first time it is read;"
            )
            print("rerunning it invalidates the go-live evidence (spec).")
            try:
                answer = input("Type RERUN HOLDOUT to run it anyway: ").strip()
            except EOFError:
                answer = ""
            if answer != "RERUN HOLDOUT":
                print("aborted")
                return 1
        start = bt.holdout_start
    else:
        start = args.from_date or bt.start
        boundary = bt.holdout_start - datetime.timedelta(days=1)
        if start > boundary:
            print(
                f"ERROR: --from {start} is inside the final holdout "
                f"(from {bt.holdout_start}); use --holdout for its one evaluation",
                file=sys.stderr,
            )
            return 1
        if end > boundary:
            print(f"note: --to clamped to {boundary} (holdout stays untouched)")
            end = boundary
    if start >= end:
        print(f"ERROR: empty date range {start}..{end}", file=sys.stderr)
        return 1

    adapter = make_adapter(config)
    cache = OhlcvCache(Path(config.data.cache_dir), config.data.refetch_days)
    try:
        prepared = prepare(config, adapter, cache, start, end)
    except BacktestError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if args.walk_forward:
        return _run_walk_forward_command(prepared, config, journal, args, start, end, now)
    return _run_plain_backtest_command(prepared, config, journal, args, start, end, now)


def _run_plain_backtest_command(prepared, config, journal, args, start, end, now) -> int:
    try:
        result = replay(prepared, config)
    except BacktestError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    metrics = compute_metrics(result, config.backtest.periods_per_year)
    kind = "holdout" if args.holdout else "backtest"
    log_experiment(
        journal,
        config=config,
        kind=kind,
        start=start,
        end=end,
        metrics=metrics,
        ts=now.isoformat(),
        grid_point={
            "entry_score_threshold": config.portfolio.entry_score_threshold,
            "stop_atr_multiple": config.portfolio.stop_atr_multiple,
        },
        survivorship_ratio=result.survivorship_ratio,
        extra={"missing_symbols": len(prepared.missing_symbols)},
    )
    count = experiment_count(journal, config.name)
    if args.json:
        print(json.dumps(_backtest_json(result, metrics, count, start, end, kind)))
    else:
        _render_backtest(result, metrics, count, kind)
    return 0


def _run_walk_forward_command(prepared, config, journal, args, start, end, now) -> int:
    try:
        wf = run_walk_forward(prepared, config, start=start, end=end)
    except (WalkForwardError, BacktestError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    for wr in wf.windows:
        log_experiment(
            journal,
            config=config,
            kind="walk_forward_window",
            start=wr.window.test_start,
            end=wr.window.test_end - datetime.timedelta(days=1),
            metrics=wr.test_metrics,
            ts=now.isoformat(),
            grid_point={
                "entry_score_threshold": wr.best.entry_score_threshold,
                "stop_atr_multiple": wr.best.stop_atr_multiple,
            },
            survivorship_ratio=wr.test_result.survivorship_ratio,
        )
    log_experiment(
        journal,
        config=config,
        kind="walk_forward",
        start=start,
        end=end,
        metrics=wf.stitched_metrics,
        ts=now.isoformat(),
        extra={
            "windows": len(wf.windows),
            "stress_segments_covered": list(wf.stress_segments_covered),
            "missing_symbols": len(prepared.missing_symbols),
        },
    )
    count = experiment_count(journal, config.name)
    if args.json:
        payload = {
            "venue": config.name,
            "from": start.isoformat(),
            "to": end.isoformat(),
            "kind": "walk_forward",
            "windows": [
                {
                    "test_from": wr.window.test_start.isoformat(),
                    "test_to": (wr.window.test_end - datetime.timedelta(days=1)).isoformat(),
                    "grid_point": {
                        "entry_score_threshold": wr.best.entry_score_threshold,
                        "stop_atr_multiple": wr.best.stop_atr_multiple,
                    },
                    "metrics": _metrics_json(wr.test_metrics),
                }
                for wr in wf.windows
            ],
            "stitched_metrics": _metrics_json(wf.stitched_metrics),
            "gate_passed": wf.stitched_metrics.gate_passed,
            "stress_segments_covered": list(wf.stress_segments_covered),
            "experiment_count": count,
        }
        print(json.dumps(payload))
    else:
        _render_walk_forward(wf, config.name, count)
    return 0


def _metrics_json(metrics: BacktestMetrics) -> dict:
    import math
    from dataclasses import asdict

    return {
        k: (None if isinstance(v, float) and math.isnan(v) else v)
        for k, v in asdict(metrics).items()
    }


def _backtest_json(
    result: BacktestResult, metrics: BacktestMetrics, count: int, start, end, kind: str
) -> dict:
    return {
        "venue": result.venue,
        "kind": kind,
        "from": start.isoformat(),
        "to": end.isoformat(),
        "metrics": _metrics_json(metrics),
        "gate_passed": metrics.gate_passed,
        "trades": len(result.trades),
        "open_positions": list(result.open_positions),
        "sessions_run": result.sessions_run,
        "sessions_skipped": len(result.sessions_skipped),
        "survivorship_ratio": round(result.survivorship_ratio, 4),
        "warnings": list(result.warnings),
        "experiment_count": count,
    }


def _fmt_pct(value: float) -> str:
    import math

    return "-" if math.isnan(value) else f"{value:+.2%}"


def _fmt_num(value: float) -> str:
    import math

    return "-" if math.isnan(value) else f"{value:.2f}"


def _render_backtest(
    result: BacktestResult, metrics: BacktestMetrics, count: int, kind: str
) -> None:
    from rich.console import Console
    from rich.table import Table

    console = Console()
    table = Table(title=f"{result.venue} {kind} {result.start} .. {result.end}")
    table.add_column("metric")
    table.add_column("value", justify="right")
    for name, value in [
        ("total return", _fmt_pct(metrics.total_return)),
        ("annualized return", _fmt_pct(metrics.annualized_return)),
        ("max drawdown", _fmt_pct(metrics.max_drawdown)),
        ("sharpe (daily, 0% cash)", _fmt_num(metrics.sharpe)),
        ("win rate", _fmt_pct(metrics.win_rate)),
        ("avg win / avg loss", f"${metrics.avg_win:,.2f} / ${metrics.avg_loss:,.2f}"),
        ("trades", str(metrics.trade_count)),
        ("turnover (annualized)", _fmt_num(metrics.turnover) + "x"),
        ("fee drag", f"${metrics.fees_paid:,.2f} ({metrics.fee_drag:.2%} of start)"),
        ("benchmark total return", _fmt_pct(metrics.benchmark_total_return)),
        ("benchmark sharpe", _fmt_num(metrics.benchmark_sharpe)),
        ("GATE (sharpe > benchmark AND total > 0)", "PASS" if metrics.gate_passed else "FAIL"),
    ]:
        table.add_row(name, value)
    console.print(table)
    console.print(
        f"survivorship coverage: {result.survivorship_ratio:.1%} of point-in-time "
        f"members had data; {len(result.sessions_skipped)} session(s) skipped"
    )
    for warning in result.warnings:
        console.print(f"[yellow]{warning}[/yellow]")
    console.print(f"experiments journaled for {result.venue}: {count} (this run included)")


def _render_walk_forward(wf: WalkForwardResult, venue: str, count: int) -> None:
    from rich.console import Console
    from rich.table import Table

    console = Console()
    table = Table(title=f"{venue} walk-forward — stitched OOS segments only")
    for col in ["test window", "threshold", "stop x", "sharpe", "total", "max DD", "trades"]:
        table.add_column(col, justify="right")
    for wr in wf.windows:
        m = wr.test_metrics
        table.add_row(
            f"{wr.window.test_start} .. {wr.window.test_end - datetime.timedelta(days=1)}",
            f"{wr.best.entry_score_threshold:.2f}",
            f"{wr.best.stop_atr_multiple:.1f}",
            _fmt_num(m.sharpe),
            _fmt_pct(m.total_return),
            _fmt_pct(m.max_drawdown),
            str(m.trade_count),
        )
    console.print(table)
    s = wf.stitched_metrics
    console.print(
        f"stitched OOS: sharpe {_fmt_num(s.sharpe)} vs benchmark {_fmt_num(s.benchmark_sharpe)}, "
        f"total {_fmt_pct(s.total_return)}, fee drag ${s.fees_paid:,.2f} — "
        f"GATE {'PASS' if s.gate_passed else 'FAIL'}"
    )
    console.print(f"stress segments covered: {', '.join(wf.stress_segments_covered)}")
    console.print(f"experiments journaled for {venue}: {count} (all windows + summary included)")
```

- [ ] **Step 4: Run the CLI tests**

Run: `uv run pytest tests/test_backtest_cli.py -v`
Expected: all pass.

- [ ] **Step 5: Full suite, lint, commit**

Run: `uv run pytest -q && uv run ruff check . && uv run ruff format --check .`

```bash
git add src/trading/cli.py tests/test_backtest_cli.py
git commit -m "$(cat <<'EOF'
Add trading backtest CLI with once-only holdout gate [AI]

trading backtest --venue X [--from --to] [--walk-forward] [--holdout]
[--json]. Every run is journaled as an experiment and the count prints
with every result. --to defaults to (and is capped at) yesterday UTC;
non-holdout runs are clamped to before the fixed holdout boundary;
--holdout refuses a second evaluation without a typed RERUN HOLDOUT.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 12: Golden backtest

A small frozen fixture dataset (committed) with committed expected output; the suite fails if results drift (spec: Testing Strategy). Regeneration is a deliberate act with a reviewed diff, never automatic.

**Files:**
- Create: `scripts/gen_golden_fixture.py`
- Create: `tests/golden/golden.toml`
- Create: `tests/golden/bars/*.csv` (generated by the script, committed)
- Create: `tests/golden/expected.json` (generated by the script, committed)
- Create: `tests/golden_helpers.py`
- Create: `tests/test_golden_backtest.py`

**Interfaces:**
- Consumes: `prepare`, `replay`, `compute_metrics`, `load_venue_config`, `OhlcvCache`, `validate_ohlcv`, `SymbolInfo`, `VenueConstraints`.
- Produces: `run_golden(cache_dir: Path) -> dict` in `tests/golden_helpers.py` — used by both the test and the regeneration script so they can never disagree.

- [ ] **Step 1: Write the frozen golden config (`tests/golden/golden.toml`)**

A complete venue TOML with tiny windows so 150 daily bars produce rankings, entries, stops, and fees (crypto-style: 24/7, taker fee, instant settlement):

```toml
[venue]
name = "golden"
benchmark = "BENCH"

[costs]
taker_fee_bps = 25.0
maker_fee_bps = 0.0
slippage_bps = 5.0
settlement_days = 0
trades_24_7 = true

[universe]
min_dollar_volume = 0.0

[signals]
momentum_windows = [3, 5, 8]
calendar_days = true
vol_window = 5
volume_week = 3
volume_baseline = 10
breakout_windows = [5, 8]
rsi_window = 5
mean_window = 5
raw_return_days = 5

[regime]
sma_fast = 5
sma_slow = 15
vol_window = 5
vol_lookback = 30
vol_high_percentile = 0.9
exposure_risk_on = 1.0
exposure_neutral = 0.5
exposure_risk_off = 0.0

[portfolio]
max_positions = 3
position_size_pct = 0.30
starting_balance = 1000.0
time_stop_bars = 8
stop_atr_multiple = 1.5
regime_flush_atr_multiple = 1.0
cooldown_days = 3
max_daily_deployment_pct = 0.35
drawdown_halt_pct = 0.50
entry_score_threshold = 0.55
min_raw_return_cost_multiple = 0.0
earnings_blackout_sessions = 0
earnings_blackout_enabled = false
staleness_hours = 24
atr_window = 5
session_close_buffer_minutes = 0

[data]
cache_dir = "unused-tests-pass-their-own-cache"
refetch_days = 5
min_coverage = 0.9
max_daily_move = 0.5
history_days = 40
quarantine_window_days = 10
drop_incomplete_last_bar = false
backfill_exchange = ""
backfill_page_limit = 0
backfill_before_days = 0

[backtest]
start = 2025-02-15
holdout_start = 2027-01-01
train_months = 1
test_months = 1
entry_score_threshold_grid = [0.5, 0.6]
stop_atr_multiple_grid = [1.0, 2.0]
min_session_coverage = 0.9
periods_per_year = 365
stress_segments = [["2025-03-01", "2025-03-31"]]
```

- [ ] **Step 2: Write `tests/golden_helpers.py`**

```python
"""Golden backtest: load the committed fixture and run the real engine.

Shared by tests/test_golden_backtest.py and scripts/gen_golden_fixture.py so
the expected output and the assertion can never diverge in HOW they run.
"""

from __future__ import annotations

import datetime
import math
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from trading.backtest.engine import prepare, replay
from trading.backtest.metrics import compute_metrics
from trading.config import load_venue_config
from trading.data.cache import OhlcvCache
from trading.venues.base import SymbolInfo, VenueConstraints, validate_ohlcv

GOLDEN = Path(__file__).parent / "golden"
GOLDEN_END = datetime.date(2025, 5, 30)


class GoldenAdapter:
    def __init__(self) -> None:
        self._frames = {
            path.stem: self._load(path) for path in sorted((GOLDEN / "bars").glob("*.csv"))
        }

    @staticmethod
    def _load(path: Path) -> pd.DataFrame:
        df = pd.read_csv(path, index_col=0)
        df.index = pd.DatetimeIndex(pd.to_datetime(df.index, utc=True))
        return validate_ohlcv(
            df[["open", "high", "low", "close", "volume"]].astype("float64")
        )

    def universe(self, as_of: datetime.date) -> list[SymbolInfo]:
        return [
            SymbolInfo(symbol=s, status="tradable") for s in sorted(self._frames) if s != "BENCH"
        ]

    def constraints(self) -> VenueConstraints:
        return VenueConstraints(
            taker_fee_bps=25.0, maker_fee_bps=0.0, slippage_bps=5.0,
            settlement_days=0, trades_24_7=True,
        )

    def fetch_ohlcv(self, symbol: str, start: datetime.date, end: datetime.date) -> pd.DataFrame:
        return self._frames[symbol].loc[
            pd.Timestamp(start, tz="UTC") : pd.Timestamp(end, tz="UTC")
        ]


def _round(value: object) -> object:
    if isinstance(value, float):
        return None if math.isnan(value) else round(value, 8)
    return value


def run_golden(cache_dir: Path) -> dict:
    config = load_venue_config("golden", GOLDEN)
    adapter = GoldenAdapter()
    cache = OhlcvCache(cache_dir, config.data.refetch_days)
    prepared = prepare(config, adapter, cache, config.backtest.start, GOLDEN_END)
    result = replay(prepared, config)
    metrics = compute_metrics(result, config.backtest.periods_per_year)
    return {
        "final_value": _round(float(result.equity_curve.iloc[-1])),
        "sessions_run": result.sessions_run,
        "sessions_skipped": len(result.sessions_skipped),
        "open_positions": list(result.open_positions),
        "trades": [
            {key: _round(value) for key, value in asdict(trade).items()}
            for trade in result.trades
        ],
        "metrics": {key: _round(value) for key, value in asdict(metrics).items()},
    }
```

- [ ] **Step 3: Write the generator (`scripts/gen_golden_fixture.py`)**

```python
"""Golden-backtest fixture generator (spec: Testing Strategy / Golden backtest).

Writes tests/golden/bars/<SYMBOL>.csv (seeded random walks -- fully
deterministic) and, with --write-expected, tests/golden/expected.json from the
CURRENT engine. Regenerating expected.json is a deliberate act: do it only
when a behavior change is intended, and review the diff in the commit. The
golden test exists to fail when results drift unintentionally.

Usage:
  uv run python scripts/gen_golden_fixture.py                  # bars only
  uv run python scripts/gen_golden_fixture.py --write-expected # bars + expected
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
GOLDEN = ROOT / "tests" / "golden"
SYMBOLS = {"AAA": 0.006, "BBB": 0.002, "CCC": 0.0, "DDD": -0.002, "EEE": 0.004, "FFF": -0.001}
PERIODS = 150
START = "2025-01-01"


def build_frame(seed: int, drift: float) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range(START, periods=PERIODS, freq="D", tz="UTC")
    returns = rng.normal(loc=drift, scale=0.02, size=PERIODS)
    close = 100.0 * np.cumprod(1.0 + returns)
    open_ = np.concatenate([[100.0], close[:-1]])
    return pd.DataFrame(
        {
            "open": open_,
            "high": np.maximum(open_, close) * 1.01,
            "low": np.minimum(open_, close) * 0.99,
            "close": close,
            "volume": rng.uniform(5e5, 1.5e6, size=PERIODS),
        },
        index=idx,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-expected", action="store_true")
    args = parser.parse_args()
    bars_dir = GOLDEN / "bars"
    bars_dir.mkdir(parents=True, exist_ok=True)
    for n, (symbol, drift) in enumerate(sorted(SYMBOLS.items()), start=1):
        build_frame(seed=n, drift=drift).to_csv(bars_dir / f"{symbol}.csv")
    build_frame(seed=99, drift=0.002).to_csv(bars_dir / "BENCH.csv")
    print(f"wrote {len(SYMBOLS) + 1} fixture frames to {bars_dir}")
    if args.write_expected:
        sys.path.insert(0, str(ROOT / "tests"))
        from golden_helpers import run_golden

        with tempfile.TemporaryDirectory() as tmp:
            expected = run_golden(Path(tmp) / "cache")
        (GOLDEN / "expected.json").write_text(json.dumps(expected, indent=2, sort_keys=True) + "\n")
        print(f"wrote {GOLDEN / 'expected.json'} ({len(expected['trades'])} trades)")
        if not expected["trades"] and not expected["open_positions"]:
            sys.exit("FATAL: golden fixture produced zero trades -- raise drifts/lower threshold")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Write the failing test (`tests/test_golden_backtest.py`)**

```python
import json

from golden_helpers import GOLDEN, run_golden


def test_golden_backtest_matches_committed_expected(tmp_path):
    expected = json.loads((GOLDEN / "expected.json").read_text())
    actual = run_golden(tmp_path / "cache")
    assert actual == expected, (
        "Golden backtest drifted. If the change is INTENDED, regenerate with "
        "'uv run python scripts/gen_golden_fixture.py --write-expected' and "
        "explain the drift in the commit message."
    )


def test_golden_fixture_actually_trades(tmp_path):
    # Guards against the golden test passing vacuously on an empty run.
    actual = run_golden(tmp_path / "cache")
    assert actual["trades"] or actual["open_positions"]
    assert actual["sessions_run"] > 60
```

- [ ] **Step 5: Run to verify failure, then generate**

Run: `uv run pytest tests/test_golden_backtest.py -v 2>&1 | tail -3`
Expected: FAIL — no `tests/golden/bars/` / `expected.json` yet.

```bash
uv run python scripts/gen_golden_fixture.py --write-expected
ls tests/golden/bars/ && head -c 400 tests/golden/expected.json
```

Expected: 7 CSVs; expected.json with nonzero trades (the FATAL guard enforces it — if it fires, raise the drifts in `SYMBOLS`, regenerate, and note the tweak in the commit).

- [ ] **Step 6: Run the golden test**

Run: `uv run pytest tests/test_golden_backtest.py -v`
Expected: 2 passed. Run it TWICE to prove determinism: `uv run pytest tests/test_golden_backtest.py -q` again — same result.

- [ ] **Step 7: Full suite, lint, commit (fixture + expected are committed — that is the point)**

Run: `uv run pytest -q && uv run ruff check . && uv run ruff format --check .`

```bash
git add scripts/gen_golden_fixture.py tests/golden/ tests/golden_helpers.py tests/test_golden_backtest.py
git commit -m "$(cat <<'EOF'
Add golden backtest with committed fixture and expected output [AI]

Seeded fixture bars + frozen golden.toml run through the real
prepare/replay/metrics path; the suite fails if any engine change
drifts the committed expected.json. Regeneration is deliberate:
scripts/gen_golden_fixture.py --write-expected, with the drift
explained in the commit.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 13: README, docs, and final validation sweep

Document the backtest surface in the README (spec: the README is the only required reading), cross-check the spec's M3 requirements, and leave the branch green.

**Files:**
- Modify: `README.md`
- Modify: `docs/superpowers/specs/2026-07-04-momentum-swing-system-design.md` (Open Items only)

**Interfaces:** none — documentation and verification.

- [ ] **Step 1: Extend the README**

Add a `## Backtesting` section after the existing command documentation, covering exactly:

````markdown
## Backtesting

```
trading backtest --venue equities|crypto [--from DATE] [--to DATE] [--json]
trading backtest --venue equities --walk-forward     # stitched OOS, tuned per window
trading backtest --venue equities --holdout          # final 6 months, evaluated ONCE
```

- One engine, replayed: the backtest drives the same simulator `step()` as
  `trading run`, session by session, filling at next-bar opens. Decisions at
  session T never see data after T.
- `--to` defaults to yesterday (UTC) — today's daily bar may still be forming.
- Non-holdout runs stop before `[backtest].holdout_start`. `--holdout` spends
  the holdout: a second invocation demands a typed `RERUN HOLDOUT` and both
  evaluations stay journaled forever.
- `--walk-forward` tunes exactly two hyperparameters (`entry_score_threshold`,
  `stop_atr_multiple`) on rolling train windows and reports stitched
  out-of-sample segments only; it refuses to report a stitch that skips every
  configured stress segment (2022 bear).
- Every run appends config hash + grid point + metrics to
  `journal/experiments-<venue>.jsonl`; the experiment count prints with every
  result. Quote results WITH their experiment count.
- Gate: annualized Sharpe of daily returns (0% cash) above buy-and-hold
  SPY/BTC over the identical period AND positive total return.

### Data caveats (read before trusting a number)

- **Equities universe is point-in-time** (S&P 500 + NDX membership as-of each
  session; sources + licences in
  `src/trading/venues/universes/sources/PROVENANCE.md`). Residual survivorship
  (delisted tickers missing from yfinance) is measured per session and printed
  as the coverage ratio on every equities result; sessions below
  `[backtest].min_session_coverage` are skipped, not faked.
- **Crypto universe is today's Robinhood listing**: coins delisted before
  today are absent (survivorship bias — annotated on every crypto result).
  Listing dates come from data availability. Deep history (pre-Kraken-window)
  is spliced from a second exchange; Kraken wins overlaps
  (see `[data]` in `config/crypto.toml`).
- NYSE half-days are handled conservatively by the live session guard (waits
  for 16:00 ET + buffer); the backtest calendar is SPY's actual bar dates, so
  half-days are traded normally there.
````

Also update the README's command list (the block copied from the spec) so `trading backtest` shows its real flags.

- [ ] **Step 2: Tick off the spec's Open Items**

In `docs/superpowers/specs/2026-07-04-momentum-swing-system-design.md`, update three Open Items (do not rewrite the spec body):

- Kraken history cap → append: `**Resolved (M3):** deep history via the ccxt exchange verified in Task 4 (named in config/crypto.toml [data].backfill_exchange), behind the same adapter; Kraken wins overlaps.`
- Point-in-time constituent dataset → append: `**Resolved (M3):** fja05680/sp500 snapshot + Wikipedia NDX changes; provenance + licences in src/trading/venues/universes/sources/PROVENANCE.md.`
- Per-trade realized P&L excludes entry fee → append: `**Resolved (M3):** entry fee frozen on Position at fill and folded into realized_pnl; cash accounting unchanged.`

- [ ] **Step 3: Spec-coverage sweep (manual checklist — fix anything missing before the final commit)**

Walk the spec's "Backtesting & Validation" section line by line and confirm:
- Span 2018-present → `[backtest].start = 2018-01-01`, deep crypto history (Task 4), `--to` yesterday cap (Task 11).
- Walk-forward 24–36mo train / 3mo test / stitched OOS only → Task 10 (+ config test pins the bounds).
- Two hyperparameters only → `grid_points`/`apply_grid_point` touch nothing else (pinned by `test_grid_points_are_the_two_toml_grids_only`).
- Final holdout once + experiment count reported → Tasks 9, 11.
- Gate metric defined → Task 8 (`gate_passed`).
- Metrics list incl. fee drag own line + benchmark → Tasks 8, 11 render.
- Survivorship structural + coverage-ratio annotation → Tasks 5, 7, 11.
- Golden backtest → Task 12. No-lookahead property test → Task 7.
- Stress segment in OOS stitch → Task 10.

- [ ] **Step 4: Full validation, then run a real smoke backtest**

```bash
uv run pytest -q && uv run ruff check . && uv run ruff format --check .
uv run trading backtest --venue crypto --from 2024-01-01 --to 2024-06-30 2>&1 | tee /tmp/claude-m3-smoke.log
```

Expected: suite green; the smoke run prints a metrics table, a survivorship line, the crypto caveat, and `experiments journaled for crypto: 1`. (First run fetches real data — allow several minutes. If the network is unavailable, note it and skip the smoke run rather than faking it.)

- [ ] **Step 5: Commit**

```bash
git add README.md docs/superpowers/specs/2026-07-04-momentum-swing-system-design.md
git commit -m "$(cat <<'EOF'
Document backtesting surface and resolve M3 spec Open Items [AI]

README gains the backtest commands, the holdout discipline, the
experiment-count convention, and the data caveats (point-in-time
equities provenance, crypto survivorship, deep-history splice).

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
EOF
)"
```

---

## Self-review notes (run before handoff)

1. **Spec coverage:** every "Backtesting & Validation" requirement maps to a task (see Task 13 Step 3 checklist). Deferred M2 items and both spec Open Items owned by M3 (Kraken cap, entry-fee P&L) are Tasks 1, 2, 4.
2. **Placeholder scan:** no TBD/TODO; every code step shows complete code; data-dependent fixtures (membership spot-checks, seeded-drift trade assertions) carry explicit tune-the-fixture-not-the-code instructions.
3. **Type consistency:** signatures cross-checked against the real codebase (`step`, `assemble_rankings`, `OhlcvCache.fetch`, `initial_state`, `Journal`, `config_hash`, `SymbolInfo`, `VenueConstraints`) and between tasks (Task 7's dataclasses consumed verbatim by Tasks 8–12).
