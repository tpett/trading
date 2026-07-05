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
    # members-with-a-bar-this-session / ALL point-in-time members (reported;
    # the skip gate uses the listing-aware eligible denominator instead)
    survivorship_ratio: float
    eligible_members: int  # the gate denominator; surfaced so shrinkage is visible
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
    equity_curve: pd.Series  # marked value per session (skipped: carried forward, flat)
    benchmark_curve: pd.Series  # buy-and-hold, scaled to the same starting balance
    trades: tuple[TradeRecord, ...]
    open_positions: tuple[str, ...]
    fees_paid: float
    buy_notional: float
    sessions_run: int
    sessions_skipped: tuple[str, ...]
    survivorship_ratio: float  # mean per-session coverage of point-in-time members
    # Coverage-gate denominator across sessions: on a today-snapshot universe
    # (crypto) the eligible set shrinks going back in time, so its size is
    # surfaced rather than blended into one ratio.
    eligible_min: int
    eligible_mean: float
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
        if config.universe.point_in_time:
            # The venue knows listings independently (equities PIT membership
            # intervals): universe(as_of) already excludes not-yet-listed
            # symbols, so every member counts. A member whose bars start
            # after this session -- or never arrive -- is a DATA outage, not
            # a listing, and must degrade coverage visibly.
            eligible = list(infos)
        else:
            # No independent listing source (crypto universe is TODAY'S
            # snapshot): listing dates are inferred from data availability
            # (plan-sanctioned). Members whose first available bar postdates
            # this session -- or with no data in the fetched window at all --
            # cannot count against coverage.
            eligible = [i for i in infos if i.symbol in bars and bars[i.symbol].index[0] <= ts]
        # A member that HAS listed but lacks a bar covering this session
        # (delisted mid-window, exchange hole) still drags coverage down.
        available = [i for i in eligible if i.symbol in bars and ts in bars[i.symbol].index]
        coverage = len(available) / len(eligible) if eligible else 0.0
        # The reported survivorship ratio keeps ALL point-in-time members in
        # the denominator so the shrinking historical universe stays visible
        # on every result (spec caveat), even though it no longer gates skips.
        ratio = len(available) / len(infos) if infos else 0.0
        if coverage < config.backtest.min_session_coverage:
            sessions.append(
                SessionPlan(
                    ts=ts,
                    rankings=None,
                    clean_symbols=(),
                    survivorship_ratio=ratio,
                    eligible_members=len(eligible),
                    skip_reason=(
                        f"coverage {coverage:.0%} below {config.backtest.min_session_coverage:.0%}"
                    ),
                )
            )
            continue
        sliced = {i.symbol: bars[i.symbol].loc[:ts] for i in available}
        try:
            rankings = assemble_rankings(config, available, sliced, benchmark.loc[:ts], ts.date())
        except PipelineDataError as exc:
            sessions.append(SessionPlan(ts, None, (), ratio, len(eligible), str(exc)))
            continue
        slim = replace(rankings, bars={}, benchmark_bars=benchmark.iloc[0:0])
        sessions.append(SessionPlan(ts, slim, tuple(rankings.bars), ratio, len(eligible), None))

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
    sessions_run = 0
    last_value = config.portfolio.starting_balance

    for plan in sessions:
        if plan.rankings is None:
            # Live parity: a run below the coverage floor touches nothing --
            # but the equity curve still gets a point (the last marked value
            # carried forward: flat, nothing traded) so both curves stay
            # defined on the SAME session index.
            skipped.append(f"{plan.ts.date().isoformat()}: {plan.skip_reason}")
            values[plan.ts] = last_value
            continue
        held = set(state.positions) | {o.symbol for o in state.pending_orders}
        # Live's RankingsResult.bars is the quarantine-passed universe only --
        # a held symbol quarantined THIS session gets no bar there, so fills
        # defer and marking falls back to entry price. Excluding quarantined
        # symbols here (even when held) keeps replay on that same seam;
        # evaluate_exits' quarantine check (keyed off rankings.quarantined,
        # untouched by this dict) still warns-and-holds them regardless.
        bars: dict[str, pd.DataFrame] = {}
        for symbol in (set(plan.clean_symbols) | held) - set(plan.rankings.quarantined):
            frame = prepared.bars.get(symbol)
            if frame is None:
                continue
            window = frame.loc[: plan.ts]
            if not window.empty:
                bars[symbol] = window
        table = plan.rankings.table
        extras = sorted((held - set(table.index)) & set(bars))
        if extras:
            # Held names with a bar this session but absent from the ranking
            # table: they left the point-in-time universe (delisted/dropped)
            # or lack sufficient signal history. Same-session-quarantined
            # holds never reach here -- they're excluded from `bars` above,
            # so evaluate_exits' quarantine check preempts and holds them
            # (live parity). Inject extras as untradable -> the simulator's
            # own forced-exit path sells them next bar (spec: dropped from
            # the venue universe). Appended LAST with NaN composite so entry
            # iteration is unaffected.
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
        values[plan.ts] = last_value = result.snapshot.value
        sessions_run += 1
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

    if sessions_run == 0:
        raise BacktestError("every session in the window was skipped; nothing to report")

    last_ts = sessions[-1].ts
    for symbol in state.positions:
        frame = prepared.bars.get(symbol)
        if frame is None or frame.loc[:last_ts].index[-1] < last_ts:
            warnings.add(
                f"{symbol}: held at end with no current bar; marked at its last close "
                "(no liquidation print invented)"
            )
    if config.name == "crypto":
        warnings.add(CRYPTO_SURVIVORSHIP_CAVEAT)

    equity_curve = pd.Series(values).sort_index()
    benchmark_curve = bench_window["close"] / bench_start_close * config.portfolio.starting_balance
    if not equity_curve.index.equals(benchmark_curve.index):
        # Cheap invariant, loud beats silent: downstream index-aligned
        # arithmetic must never see NaNs from mismatched session indexes.
        raise BacktestError("equity and benchmark curves diverged in index; engine bug")
    ratios = [s.survivorship_ratio for s in sessions]
    eligible_counts = [s.eligible_members for s in sessions]
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
        sessions_run=sessions_run,
        sessions_skipped=tuple(skipped),
        survivorship_ratio=sum(ratios) / len(ratios),
        eligible_min=min(eligible_counts),
        eligible_mean=sum(eligible_counts) / len(eligible_counts),
        warnings=tuple(sorted(warnings)),
    )
