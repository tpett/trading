"""Paper-portfolio state (spec: Portfolio Simulator, State).

Pure data + dict round-trip only. File persistence (atomic write, corrupt
detection) lives in trading.runner; the simulator core never touches disk.
All timestamps are ISO-8601 UTC strings so the state survives JSON exactly.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal

STATE_VERSION = 1

Side = Literal["buy", "sell"]


class StateError(RuntimeError):
    """State payload is corrupt or structurally invalid."""


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
    peak_close: float | None = None  # trailing-exit high-water close; None on old saved state


@dataclass(frozen=True)
class PendingOrder:
    symbol: str
    side: Side
    notional: float  # buys: committed dollars (fee charged on top); sells: 0.0 (full qty)
    decision_ts: str  # ISO-8601 UTC decision-bar timestamp; fills at first bar strictly after
    reason: str  # entry | stop_loss | trend_break | time_stop | forced_exit
    atr_at_decision: float = 0.0  # buys only: ATR to freeze at entry
    composite: float = 0.0  # buys only
    rank: int = 0  # buys only


@dataclass(frozen=True)
class Settlement:
    amount: float
    available_on: str  # ISO date; settled once the decision-bar date reaches it


@dataclass(frozen=True)
class Skip:
    symbol: str  # "*" for venue-wide skips
    action: str  # entry | exit | fill
    reason: str


@dataclass
class PortfolioState:
    venue: str
    cash: float  # settled cash only
    settlements: list[Settlement] = field(default_factory=list)
    positions: dict[str, Position] = field(default_factory=dict)
    pending_orders: list[PendingOrder] = field(default_factory=list)
    cooldowns: dict[str, str] = field(default_factory=dict)  # symbol -> ISO date re-entry allowed
    high_water_mark: float = 0.0
    breaker_tripped: bool = False
    breaker_tripped_at: str | None = None
    benchmark_start_price: float = 0.0  # benchmark close at bootstrap (buy-and-hold baseline)
    created_at: str = ""
    last_run_key: str | None = None


def initial_state(
    venue: str, starting_balance: float, benchmark_start_price: float, created_at: str
) -> PortfolioState:
    return PortfolioState(
        venue=venue,
        cash=starting_balance,
        high_water_mark=starting_balance,
        benchmark_start_price=benchmark_start_price,
        created_at=created_at,
    )


def to_state_dict(state: PortfolioState) -> dict:
    return {"version": STATE_VERSION, **asdict(state)}


def state_from_dict(payload: dict) -> PortfolioState:
    try:
        version = payload["version"]
        if version != STATE_VERSION:
            raise StateError(f"unsupported state version {version!r}")
        return PortfolioState(
            venue=payload["venue"],
            cash=float(payload["cash"]),
            settlements=[Settlement(**s) for s in payload["settlements"]],
            positions={k: Position(**p) for k, p in payload["positions"].items()},
            pending_orders=[PendingOrder(**o) for o in payload["pending_orders"]],
            cooldowns=dict(payload["cooldowns"]),
            high_water_mark=float(payload["high_water_mark"]),
            breaker_tripped=bool(payload["breaker_tripped"]),
            breaker_tripped_at=payload["breaker_tripped_at"],
            benchmark_start_price=float(payload["benchmark_start_price"]),
            created_at=payload["created_at"],
            last_run_key=payload["last_run_key"],
        )
    except StateError:
        raise
    except (KeyError, TypeError, ValueError) as exc:
        raise StateError(f"corrupt portfolio state: {exc}") from exc
