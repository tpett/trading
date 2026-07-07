"""CEILING TEST -- bound the MAXIMUM benefit any quality/junk filter could
ever give the momentum strategy.

************************  LOOK-AHEAD DIAGNOSTIC  ************************
* This script USES FUTURE INFORMATION (which names later terminally     *
* failed) to remove them from the universe BEFORE they crash. It is a   *
* measurement tool ONLY and must NEVER be presented, deployed, or       *
* interpreted as a tradeable strategy: no live screen can know in       *
* advance which names will die. Every number it prints is an UPPER      *
* BOUND on realizable benefit, not an achievable result.                *
***********************************************************************

Why: the survivorship-free backtest scored 0.45 OOS Sharpe (vs 0.59 on
survivor-only data) because delisted/failing names entered the pool and
momentum sometimes picked them (a dead-cat-bounce rally ranks high) right
before they crashed. This re-runs that backtest with a HINDSIGHT filter that
removes the names that terminally failed, and measures how much of the
0.45 -> 0.59 gap (0.14 Sharpe) that recovers:

  * recovers LITTLE  -> no realizable quality filter can help (clean, strong
    negative: the survivorship gap is not a junk-avoidance problem).
  * recovers a LOT   -> quantifies the headroom a smarter (e.g. sector-aware)
    screen could chase, even though this hindsight ceiling is itself
    unreachable.

Design (see the module CLAUDE.md task for the full rationale):

  1. Terminal-failure detection from the Tiingo bar cache: a symbol
     "terminally failed" if its cached bars END > LAST_BAR_GAP_DAYS trading
     days before the backtest `end` (delisted, not a recent gap) AND its
     trailing-TRAILING_RETURN_DAYS return into its last bar is below the
     distress threshold (a decline -- distinguishes a failure from an
     acquisition, which exits flat-to-up at a takeover premium). A second,
     looser distress threshold is also reported for sensitivity.
  2. Exclusion window: a failed symbol is removed from the universe for the
     EXCLUSION_WINDOW_DAYS trading days before its last bar (the pre-failure
     decline). Before that window it participates normally.
  3. Injection via a thin wrapping adapter (the engine is NOT modified): it
     delegates everything to the real EquitiesAdapter except universe(as_of),
     which drops any symbol whose as_of falls inside its exclusion window.
  4. Run baseline (real adapter) and ceiling (wrapping adapter), and report,
     per distress threshold: baseline / ceiling stitched OOS Sharpe, absolute
     recovery, and recovery as a fraction of the 0.14 survivorship gap.

Run (on the warm Tiingo cache -- this is a ~2h compute job):

    uv run python scripts/ceiling_test.py --config-dir config/experiments/tiingo
"""

from __future__ import annotations

import argparse
import datetime
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from trading.backtest.engine import prepare
from trading.backtest.walkforward import run_walk_forward
from trading.config import load_venue_config
from trading.data.cache import OhlcvCache
from trading.venues import make_adapter
from trading.venues.base import SymbolInfo

# --- Terminal-failure thresholds (module-level so they are auditable) --------
# A symbol's cached bars must end more than this many TRADING days before the
# backtest `end` to count as delisted (vs a merely-recent data gap).
LAST_BAR_GAP_DAYS = 30
# Trailing return window (~6 months of trading days) measured into the last bar.
TRAILING_RETURN_DAYS = 126
# Primary distress cutoff: a <-20% 6-month decline into the last bar reads as a
# failure, not an acquisition (which exits flat-to-up at a takeover premium).
DISTRESS_RETURN = -0.20
# Looser sensitivity cutoff, reported alongside the primary one.
DISTRESS_RETURN_LOOSE = -0.40
# Exclusion window (~12 months of trading days) before a failed name's last
# bar: the pre-failure decline period a dying name's rally could rank high on.
EXCLUSION_WINDOW_DAYS = 252

# The survivorship gap this test tries to recover: survivor-only OOS Sharpe
# (0.59) minus survivorship-free OOS Sharpe (0.45).
SURVIVOR_ONLY_SHARPE = 0.59
SURVIVORSHIP_FREE_SHARPE = 0.45
SURVIVORSHIP_GAP = SURVIVOR_ONLY_SHARPE - SURVIVORSHIP_FREE_SHARPE
# Sanity band on the reproduced baseline: if it lands outside this, stop.
BASELINE_SHARPE_TOLERANCE = 0.10


@dataclass(frozen=True)
class TerminalFailure:
    """A name flagged as having terminally failed (look-ahead: uses its full
    cached history, including bars after any backtest decision date)."""

    symbol: str
    last_bar: datetime.date
    six_month_return: float
    window_start: datetime.date  # start of the 252-trading-day exclusion window


def trailing_return(close: pd.Series, days: int) -> float | None:
    """Return of the last close over the close `days` trading rows earlier.

    Uses the earliest available bar when the history is shorter than `days`.
    None when there is too little data or a non-positive base to divide by.
    """
    if len(close) < 2:
        return None
    idx = max(0, len(close) - 1 - days)
    base = float(close.iloc[idx])
    if base <= 0.0:
        return None
    return float(close.iloc[-1]) / base - 1.0


def detect_terminal_failure(
    symbol: str,
    frame: pd.DataFrame | None,
    sessions: pd.DatetimeIndex,
    distress_return: float,
) -> TerminalFailure | None:
    """Flag `symbol` as a terminal failure, or return None.

    `sessions` is the benchmark trading-day calendar bounded at the backtest
    `end`; the last-bar gap is measured in those sessions so "trading days" is
    the realized NYSE calendar, not raw calendar days. LOOK-AHEAD: this reads
    the symbol's full cached history and the fact that trading later stopped.
    """
    if frame is None or frame.empty:
        return None
    frame = frame.sort_index()
    last_ts = frame.index[-1]
    # (a) delisting: trading sessions strictly after the last bar, up to `end`.
    gap = int((sessions > last_ts).sum())
    if gap <= LAST_BAR_GAP_DAYS:
        return None  # still trading into the backtest end: survivor / recent gap
    # (b) distress: a decline into the last bar (an acquisition exits flat-to-up).
    ret = trailing_return(frame["close"], TRAILING_RETURN_DAYS)
    if ret is None or ret >= distress_return:
        return None
    n = len(frame)
    window_start_idx = max(0, n - 1 - EXCLUSION_WINDOW_DAYS)
    return TerminalFailure(
        symbol=symbol,
        last_bar=last_ts.date(),
        six_month_return=ret,
        window_start=frame.index[window_start_idx].date(),
    )


def exclusion_windows(
    failures: dict[str, TerminalFailure],
) -> dict[str, tuple[datetime.date, datetime.date]]:
    """{symbol: (window_start_date, last_bar_date)} for the wrapping adapter."""
    return {sym: (f.window_start, f.last_bar) for sym, f in failures.items()}


class HindsightFilterAdapter:
    """Thin wrapper over the real EquitiesAdapter that removes hindsight-known
    terminal failures from the universe during their pre-failure decline.

    LOOK-AHEAD: the exclusion windows are derived from future information. Every
    method OTHER than universe() delegates straight to the inner adapter, so the
    engine sees identical bars, constraints and membership intervals -- only the
    universe(as_of) membership is filtered.
    """

    def __init__(
        self,
        inner: object,
        exclusion_windows: dict[str, tuple[datetime.date, datetime.date]],
    ):
        self._inner = inner
        self._windows = dict(exclusion_windows)

    def universe(self, as_of: datetime.date) -> list[SymbolInfo]:
        return [i for i in self._inner.universe(as_of) if not self._excluded(i.symbol, as_of)]

    def _excluded(self, symbol: str, as_of: datetime.date) -> bool:
        window = self._windows.get(symbol)
        if window is None:
            return False
        window_start, last_bar = window
        return window_start <= as_of <= last_bar

    def __getattr__(self, name: str) -> object:
        # Only reached for attributes this wrapper does not define (fetch_ohlcv,
        # constraints, membership_intervals, ...): delegate them unchanged.
        return getattr(self._inner, name)


def _load_frame(cache: OhlcvCache, symbol: str) -> pd.DataFrame | None:
    path = cache.path_for(symbol)
    if not path.exists():
        return None
    return pd.read_parquet(path)


def _stitched_sharpe(prepared, config, start: datetime.date, end: datetime.date) -> float:
    return run_walk_forward(prepared, config, start=start, end=end).stitched_metrics.sharpe


def _print_failures(label: str, failures: dict[str, TerminalFailure]) -> None:
    print(f"\n{label}: {len(failures)} symbol(s) flagged as terminal failures")
    sample = sorted(failures.values(), key=lambda f: f.six_month_return)[:15]
    print("  sample (most-distressed first) -- eyeball these as real failures, not M&A:")
    print(f"    {'symbol':<10} {'last bar':<12} {'6mo return':>10}")
    for f in sample:
        print(f"    {f.symbol:<10} {f.last_bar.isoformat():<12} {f.six_month_return:>9.1%}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", default="config/experiments/tiingo")
    parser.add_argument("--venue", default="equities")
    args = parser.parse_args()

    print(__doc__.split("\n\n")[0])  # re-emit the LOOK-AHEAD banner at runtime
    print("\n!!! LOOK-AHEAD DIAGNOSTIC -- results are an UPPER BOUND, not tradeable !!!\n")

    config = load_venue_config(args.venue, Path(args.config_dir))
    adapter = make_adapter(config)
    # Offline: serve purely from the warm parquet cache, no network -- the
    # tiingo config's refetch_days must not re-hit Tiingo during prepare().
    cache = OhlcvCache(
        Path(config.data.cache_dir),
        config.data.refetch_days,
        config.data.bar_source,
        offline=True,
    )

    start = config.backtest.start
    # Clamp end to just before the holdout: the walk-forward clamps internally
    # too, but be explicit -- the one-time holdout must never be touched here.
    end = config.backtest.holdout_start - datetime.timedelta(days=1)
    print(f"span: {start} .. {end} (holdout_start={config.backtest.holdout_start}, untouched)")

    # Benchmark trading-day calendar up to `end`, for the last-bar gap measure.
    warmup_start = start - datetime.timedelta(days=config.data.history_days)
    benchmark = cache.fetch(config.benchmark, warmup_start, end, adapter.fetch_ohlcv)
    end_ts = pd.Timestamp(end, tz="UTC")
    sessions = benchmark.index[benchmark.index <= end_ts]

    # Candidate symbols = everything the backfill warmed into the cache dir
    # (the survivorship-free set + benchmark). Equities tickers carry no '/'.
    symbols = sorted(p.stem for p in Path(config.data.cache_dir).glob("*.parquet"))
    print(f"scanning {len(symbols)} cached symbols for terminal failures")

    # --- Baseline: real adapter, must reproduce ~0.45 (sanity gate) ----------
    print("\n[1/3] preparing + walk-forward: BASELINE (real universe)")
    baseline_prepared = prepare(config, adapter, cache, start, end)
    if baseline_prepared.missing_symbols:
        miss = baseline_prepared.missing_symbols
        print(
            f"  offline cache served no bars for {len(miss)} member(s) "
            f"(reported, not silently dropped): {', '.join(miss[:20])}"
            + (f" ... +{len(miss) - 20} more" if len(miss) > 20 else "")
        )
    baseline_sharpe = _stitched_sharpe(baseline_prepared, config, start, end)
    print(f"  baseline stitched OOS Sharpe = {baseline_sharpe:.4f}")
    if abs(baseline_sharpe - SURVIVORSHIP_FREE_SHARPE) > BASELINE_SHARPE_TOLERANCE:
        print(
            f"\nSTOP: baseline {baseline_sharpe:.3f} is outside "
            f"{SURVIVORSHIP_FREE_SHARPE:.2f} +/- {BASELINE_SHARPE_TOLERANCE:.2f}; the "
            "survivorship-free backtest did not reproduce. Investigate the cache/config "
            "before trusting any ceiling number."
        )
        return 1

    # --- Ceiling, per distress threshold -------------------------------------
    rows: list[tuple[float, int, float, float, float]] = []
    for step, distress in enumerate((DISTRESS_RETURN, DISTRESS_RETURN_LOOSE), start=2):
        failures = {
            sym: failure
            for sym in symbols
            if (
                failure := detect_terminal_failure(sym, _load_frame(cache, sym), sessions, distress)
            )
            is not None
        }
        _print_failures(f"distress<{distress:.0%}", failures)
        print(f"\n[{step}/3] preparing + walk-forward: CEILING (distress<{distress:.0%})")
        wrapped = HindsightFilterAdapter(adapter, exclusion_windows(failures))
        ceiling_prepared = prepare(config, wrapped, cache, start, end)
        ceiling_sharpe = _stitched_sharpe(ceiling_prepared, config, start, end)
        recovery = ceiling_sharpe - baseline_sharpe
        fraction = recovery / SURVIVORSHIP_GAP if SURVIVORSHIP_GAP else float("nan")
        print(f"  ceiling stitched OOS Sharpe = {ceiling_sharpe:.4f}")
        rows.append((distress, len(failures), ceiling_sharpe, recovery, fraction))

    # --- Report --------------------------------------------------------------
    print("\n" + "=" * 78)
    print("CEILING TEST RESULTS  (LOOK-AHEAD upper bound -- NOT a tradeable strategy)")
    print("=" * 78)
    print(
        f"survivorship gap being chased: {SURVIVOR_ONLY_SHARPE:.2f} (survivor-only) - "
        f"{SURVIVORSHIP_FREE_SHARPE:.2f} (surv-free) = {SURVIVORSHIP_GAP:.2f} Sharpe"
    )
    print(f"baseline (reproduced survivorship-free) OOS Sharpe: {baseline_sharpe:.4f}")
    print(f"\n  {'distress':>9} {'#failed':>8} {'ceiling':>9} {'recovery':>9} {'% of gap':>9}")
    for distress, n_failed, ceiling_sharpe, recovery, fraction in rows:
        print(
            f"  {distress:>8.0%} {n_failed:>8d} {ceiling_sharpe:>9.4f} "
            f"{recovery:>+9.4f} {fraction:>8.0%}"
        )
    print(
        "\nInterpretation: a small recovery means NO realizable quality filter can close "
        "the survivorship gap; a large one quantifies the (unreachable) headroom for a "
        "smarter, sector-aware screen. Either way this is hindsight -- do not trade it."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
