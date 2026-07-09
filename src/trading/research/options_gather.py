"""Historical EOD option-quote gather + IV/skew derivation (ThetaData v3).

Why this module exists
----------------------
``options_iv.py`` can invert a Black-Scholes price for implied volatility and
form crude skew measures, but it is fed by a samples file that some *other*
process has to produce. This module is that producer: for a universe of liquid
US equities it walks a monthly decision calendar, snaps a small ring of
option contracts (ATM, a 10%-OTM put, a 10%-OTM call) around each decision-date
spot, pulls their EOD quotes from a local ThetaData v3 terminal, computes IV
from the quote MID, and writes one JSON cell per (symbol, decision-month) that
``option_skew_analysis.py`` / ``options_iv.skew_from_cell`` already know how to
consume -- extended with the ``mid``/``iv`` fields we can now compute at gather
time because we hold the full quote.

Design choices worth stating once
----------------------------------
* **MID, not close, drives IV.** The bid/ask midpoint is a cleaner mark than the
  last print for a thinly-traded contract, so ``implied_vol`` is fed
  ``(bid+ask)/2``. ``close`` is still carried through for the older analysis
  path and for eyeballing.
* **The tape double-prints.** ThetaData's EOD history returns several rows for
  one trade date that differ only in their ``created`` snapshot timestamp; the
  OHLC/bid/ask are identical. We collapse to one bar per *trade date* (derived
  from ``last_trade``), keeping the last row seen -- see ``dedup_bars_by_trade_date``.
* **Concurrency is capped at 4** because the terminal rejects a 5th in-flight
  request. We run one worker per (symbol, month) cell and let each worker issue
  its handful of requests sequentially, so a pool of 4 workers can never have
  more than 4 requests outstanding. Expiration/strike lists are memoised per
  symbol so re-scanning 84 months of one name is a couple of list calls, not
  hundreds.
* **Resumable.** Cells are appended (and flushed) as they finish; a re-run reads
  back the existing (symbol, decision_date) keys and skips them, so the
  hours-long overnight job survives a restart without duplicating work.

Everything network-facing is isolated behind ``ThetaClient`` (or any object
exposing the same three methods) so tests inject a fake and never touch the
terminal.
"""

from __future__ import annotations

import json
import logging
import math
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path
from typing import Protocol

import pandas as pd

from trading.research.options_iv import DIV_YIELD, RATE, implied_vol

log = logging.getLogger("trading.research.options_gather")

# --- Gather knobs (defaults; overridable at the call site) -----------------
# A standard monthly option expires the 3rd Friday of its month. For a
# first-of-month decision date the SAME-month monthly is only ~15-20 DTE (too
# near), so we always reach for the NEXT month's monthly, which lands ~45-53 DTE.
# The band is therefore 25-55 (target ~35 as a tiebreak): the lower bound rejects
# the same-month monthly, the upper bound must clear 53 or a few first-of-month
# dates whose next monthly falls at 51-53 DTE would find nothing in band and be
# dropped (observed: 2019-05-01 -> 2019-06-21 = 51 DTE). Skew is a same-expiry IV
# difference, so the ~8-day DTE spread across months is immaterial.
TARGET_DTE = 35
MIN_DTE = 25
MAX_DTE = 55
# OTM legs sit a nominal 10% either side of spot; snapped to the real ladder.
OTM_PUT_MULT = 0.90
OTM_CALL_MULT = 1.10
# A listed strike can be dataless: ThetaData lists half-dollar strikes (e.g. AAPL
# 167.5) that /history/eod returns empty (HTTP 472) for, while the neighbouring
# whole-dollar strike has real bars. So a leg walks outward from its nominal
# target to the nearest strike that ACTUALLY returns a bar, up to this many
# candidates before giving the leg up as absent.
MAX_STRIKE_CANDIDATES = 4
# EOD history is requested over decision +/- this many calendar days; the bar
# used is the one ON the decision date, else the nearest PRIOR trade date still
# inside the window.
QUOTE_WINDOW_DAYS = 3

# Universe: symbols that were EVER an sp500/ndx member anywhere in this span AND
# are present in the Tiingo cache are candidates, ranked by dollar volume.
UNIVERSE_MEMBERSHIP_WINDOW = ("2019-01-01", "2026-12-31")

# --- v2 enrichment (open interest, far expiration) ---------------------------
# Endpoint path VERIFIED against the live Standard-tier terminal (options-gather-v2
# plan Task 1 discovery, 2026-07): the EOD tape itself carries no inline
# open_interest field, but this dedicated route serves it (HTTP 200, one row/day,
# columns symbol,expiration,strike,right,timestamp,open_interest -- same query
# params as history/eod). Greeks endpoints (`/v3/option/history/greeks`,
# `/v3/option/history/implied_volatility`) both 404 on this tier, so no greeks
# capture path exists here -- re-run discovery if the terminal build changes.
OI_PATH = "/v3/option/history/open_interest"
# Contract keys the gather may ADD to a leg; only extractor-vetted values ever
# carry them (absent field -> absent key, never 0).
_ENRICHMENT_KEYS = ("open_interest",)
# Far (term-structure) leg: the NEXT standard monthly after the near target.
# Near sits 25..55 DTE, so the following monthly lands ~55..85; 90 caps a
# straggler month. Spec: docs/superpowers/specs/2026-07-09-options-gather-v2-design.md.
FAR_MIN_DTE = 55
FAR_MAX_DTE = 90


# --- HTTP client -----------------------------------------------------------


class OptionsClient(Protocol):
    """The three terminal reads the gather needs. ``ThetaClient`` implements it;
    tests pass a fake with the same surface so nothing hits the network."""

    def list_expirations(self, symbol: str) -> list[str]: ...
    def list_strikes(self, symbol: str, expiration: str) -> list[float]: ...
    def history_eod(
        self,
        symbol: str,
        expiration: str,
        strike: float,
        right: str,
        start_date: str,
        end_date: str,
    ) -> list[dict]: ...
    def history_open_interest(
        self,
        symbol: str,
        expiration: str,
        strike: float,
        right: str,
        start_date: str,
        end_date: str,
    ) -> list[dict]: ...


def _fmt_strike(strike: float) -> str:
    """Render a strike for the query string.

    The ladder comes back as floats (``185.0``, ``157.5``); the terminal expects
    the bare integer for whole-dollar strikes (``strike=185``, per the confirmed
    API shape) and the decimal form otherwise. Formatting here keeps the wire
    value canonical regardless of how the caller carried the number.
    """
    if float(strike).is_integer():
        return str(int(strike))
    return repr(float(strike))


# ThetaData returns HTTP 472 for a well-formed request that simply has no data
# in range -- e.g. an illiquid deep-OTM half-strike with no EOD prints in the
# +/-3d window. That is a normal "empty", NOT a failure: an optional leg (the
# OTM call) should be omitted, not sink the whole cell. Any OTHER non-200 stays
# a retryable error.
_THETA_NO_DATA_STATUS = 472


def _contract_params(
    symbol: str, expiration: str, strike: float, right: str, start_date: str, end_date: str
) -> dict:
    """The canonical per-contract query params every history endpoint takes."""
    return {
        "symbol": symbol,
        "expiration": expiration,
        "strike": _fmt_strike(strike),
        "right": right,
        "start_date": start_date,
        "end_date": end_date,
        "format": "json",
    }


def _flatten_rows(payload: dict) -> list[dict]:
    """v3 envelope -> flat row list. Entries either nest rows under "data"
    (the history/eod shape) or ARE the rows; tolerate both so the OI parser
    survives either serialization."""
    rows: list[dict] = []
    for entry in payload.get("response", []):
        if not isinstance(entry, dict):
            continue
        data = entry.get("data")
        if isinstance(data, list):
            rows.extend(row for row in data if isinstance(row, dict))
        else:
            rows.append(entry)
    return rows


class ThetaClient:
    """Thin stdlib-``urllib`` wrapper over the local ThetaData v3 terminal.

    Only GET/JSON, only the three endpoints the gather uses. Transient failures
    (connection refused, non-200, malformed body) are retried with exponential
    backoff; the ``sleep`` callable is injectable so tests exercise the retry
    path without wall-clock delay. After the retries are exhausted the last
    exception propagates -- the orchestrator catches it per cell so one bad leg
    never sinks the run.
    """

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:25503",
        *,
        timeout: float = 30.0,
        max_retries: int = 3,
        backoff_base: float = 0.5,
        sleep=time.sleep,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._max_retries = max_retries
        self._backoff_base = backoff_base
        self._sleep = sleep

    def _get_json(self, path: str, params: dict) -> dict:
        query = urllib.parse.urlencode(params)
        url = f"{self._base_url}{path}?{query}"
        last_exc: Exception | None = None
        for attempt in range(self._max_retries):
            try:
                req = urllib.request.Request(url, headers={"Accept": "application/json"})
                with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                # 472 = "no data in range" for a valid request: treat as empty,
                # no retry, so an optional dataless leg is skipped cleanly.
                if exc.code == _THETA_NO_DATA_STATUS:
                    return {"response": []}
                last_exc = exc
                if attempt < self._max_retries - 1:
                    self._sleep(self._backoff_base * (2**attempt))
            except (urllib.error.URLError, OSError, ValueError) as exc:
                # URLError covers connection refused; OSError covers socket
                # timeouts; ValueError covers a truncated/invalid body.
                last_exc = exc
                if attempt < self._max_retries - 1:
                    self._sleep(self._backoff_base * (2**attempt))
        assert last_exc is not None
        raise last_exc

    def list_expirations(self, symbol: str) -> list[str]:
        payload = self._get_json(
            "/v3/option/list/expirations", {"symbol": symbol, "format": "json"}
        )
        return [row["expiration"] for row in payload.get("response", [])]

    def list_strikes(self, symbol: str, expiration: str) -> list[float]:
        payload = self._get_json(
            "/v3/option/list/strikes",
            {"symbol": symbol, "expiration": expiration, "format": "json"},
        )
        return [float(row["strike"]) for row in payload.get("response", [])]

    def history_eod(
        self,
        symbol: str,
        expiration: str,
        strike: float,
        right: str,
        start_date: str,
        end_date: str,
    ) -> list[dict]:
        payload = self._get_json(
            "/v3/option/history/eod",
            _contract_params(symbol, expiration, strike, right, start_date, end_date),
        )
        # One (strike,right) query returns a single contract entry, but flatten
        # defensively across all entries' bars.
        bars: list[dict] = []
        for entry in payload.get("response", []):
            bars.extend(entry.get("data", []))
        return bars

    def history_open_interest(
        self,
        symbol: str,
        expiration: str,
        strike: float,
        right: str,
        start_date: str,
        end_date: str,
    ) -> list[dict]:
        payload = self._get_json(
            OI_PATH, _contract_params(symbol, expiration, strike, right, start_date, end_date)
        )
        return _flatten_rows(payload)


class _CachingClient:
    """Memoises the per-symbol expiration list and per-(symbol,expiration) strike
    ladder so 84 monthly cells for one name cost a couple of list calls instead
    of one per cell. History (unique per leg/date) is passed straight through.

    Fetches happen outside the lock -- a rare concurrent double-fetch is
    harmless and cheaper than serialising every worker behind one mutex.
    """

    def __init__(self, client: OptionsClient) -> None:
        self._client = client
        self._exp: dict[str, list[str]] = {}
        self._strikes: dict[tuple[str, str], list[float]] = {}
        self._lock = threading.Lock()

    def list_expirations(self, symbol: str) -> list[str]:
        with self._lock:
            cached = self._exp.get(symbol)
        if cached is not None:
            return cached
        value = self._client.list_expirations(symbol)
        with self._lock:
            return self._exp.setdefault(symbol, value)

    def list_strikes(self, symbol: str, expiration: str) -> list[float]:
        key = (symbol, expiration)
        with self._lock:
            cached = self._strikes.get(key)
        if cached is not None:
            return cached
        value = self._client.list_strikes(symbol, expiration)
        with self._lock:
            return self._strikes.setdefault(key, value)

    def history_eod(self, *args, **kwargs) -> list[dict]:
        return self._client.history_eod(*args, **kwargs)

    def history_open_interest(self, *args, **kwargs) -> list[dict]:
        return self._client.history_open_interest(*args, **kwargs)


# --- Calendar / spot helpers ----------------------------------------------


def _as_date(value: str | date) -> date:
    return value if isinstance(value, date) else date.fromisoformat(str(value)[:10])


def first_trading_days(
    index: pd.DatetimeIndex, start_month: str, end_month: str
) -> list[pd.Timestamp]:
    """First bar of each month in ``[start_month, end_month]`` from a symbol's own
    calendar (its Tiingo index). Months with no bars (a name not yet listed) are
    simply absent from the result -- the decision date does not exist for them.
    """
    tz = index.tz
    out: list[pd.Timestamp] = []
    for period in pd.period_range(start_month, end_month, freq="M"):
        lo = pd.Timestamp(period.start_time, tz=tz)
        hi = pd.Timestamp(period.end_time, tz=tz)
        bars = index[(index >= lo) & (index <= hi)]
        if len(bars):
            out.append(bars[0])
    return out


def load_raw_close(symbol: str, raw_dir: Path) -> pd.Series | None:
    """RAW (unadjusted) daily close series for ``symbol``, or None if absent.

    Reads ``<raw_dir>/<SYM>.parquet`` (single ``close_raw`` column, UTC index)
    produced by ``scripts/backfill_options_underlying_raw.py``. This is a
    SEPARATE cache from ``data/equities-tiingo`` on purpose: that one holds
    split+dividend-ADJUSTED closes, but ThetaData option strikes and the BS
    inversion both live in RAW price space, so spot MUST be the unadjusted close.
    """
    path = raw_dir / f"{symbol}.parquet"
    if not path.exists():
        return None
    return pd.read_parquet(path, columns=["close_raw"])["close_raw"]


def raw_spot_at(raw_close: pd.Series, decision_ts: pd.Timestamp) -> float | None:
    """Raw close on the last bar on/before ``decision_ts`` (searchsorted
    side='right' minus one). Returns None if the decision date precedes the raw
    series entirely -- the caller then skips the cell rather than falling back to
    an adjusted (and therefore strike-mismatched) price."""
    idx = raw_close.index
    pos = idx.searchsorted(decision_ts, side="right") - 1
    if pos < 0:
        return None
    return float(raw_close.iloc[pos])


# --- Expiration & strike selection ----------------------------------------


def is_standard_monthly(d: date) -> bool:
    """Is ``d`` a standard monthly option expiration?

    Standard monthlies expire the 3rd Friday of the month, which always falls on
    the 15th-21st. The one wrinkle: when that Friday is a market holiday the
    expiration shifts one day earlier to the Thursday -- exactly the spec's own
    ``2019-04-18`` example (the 19th was Good Friday). We therefore accept a
    Thursday OR Friday inside the 3rd-Friday week and reject everything else.
    Weeklies live on the other Fridays of the month, which are outside the 15-21
    window, so they are excluded without a holiday calendar.
    """
    return 15 <= d.day <= 21 and d.weekday() in (3, 4)


def select_expiration(
    expirations: list[str | date],
    decision_date: date,
    *,
    target_dte: int = TARGET_DTE,
    min_dte: int = MIN_DTE,
    max_dte: int = MAX_DTE,
) -> date | None:
    """Pick the standard-monthly expiration nearest ``decision + target_dte``,
    restricted to the ``[min_dte, max_dte]`` DTE band. Non-standard (weekly /
    quarterly-EOM) expirations are ignored. Among in-window candidates a true
    3rd-FRIDAY monthly is preferred; a Thursday (Good-Friday shift) is taken only
    when no Friday monthly is in-window. Ties break to the earlier date so the
    choice is deterministic. Returns None when nothing sits in-window.
    """
    target = decision_date + timedelta(days=target_dte)
    candidates: list[date] = []
    for raw in expirations:
        d = _as_date(raw)
        if not is_standard_monthly(d):
            continue
        dte = (d - decision_date).days
        if min_dte <= dte <= max_dte:
            candidates.append(d)
    if not candidates:
        return None
    fridays = [d for d in candidates if d.weekday() == 4]
    pool = fridays or candidates  # only fall back to a Thursday when no Friday is in-window
    return min(pool, key=lambda d: (abs((d - target).days), d))


def select_far_expiration(
    expirations: list[str | date],
    decision_date: date,
    near_expiration: date,
    *,
    min_dte: int = FAR_MIN_DTE,
    max_dte: int = FAR_MAX_DTE,
) -> date | None:
    """The NEXT standard monthly strictly after ``near_expiration`` inside the
    far DTE band ``[min_dte, max_dte]``, or None. "Next" means EARLIEST -- the
    term-structure far leg is the adjacent monthly, so there is no
    target-distance tiebreak: the first in-band monthly after the near one
    wins, deterministically.
    """
    candidates: list[date] = []
    for raw in expirations:
        d = _as_date(raw)
        if not is_standard_monthly(d):
            continue
        if d <= near_expiration:
            continue
        if min_dte <= (d - decision_date).days <= max_dte:
            candidates.append(d)
    if len(candidates) == 0:
        return None
    return min(candidates)


def snap_strikes(
    strikes: list[float],
    spot: float,
    *,
    put_mult: float = OTM_PUT_MULT,
    call_mult: float = OTM_CALL_MULT,
) -> dict[str, float]:
    """Snap the ATM / OTM-put / OTM-call targets onto the real strike ladder.

    ATM is the strike nearest spot; the OTM legs are nearest ``put_mult*spot``
    and ``call_mult*spot``. On a sparse ladder two roles can land on the same
    strike -- that collapse is left intact here (the roles keep their identity)
    and de-duplicated at the *fetch* level by (strike, right), so the same
    contract is never pulled twice. Ties break to the lower strike for
    determinism.
    """
    ladder = sorted({float(s) for s in strikes})

    def nearest(target: float) -> float:
        return min(ladder, key=lambda s: (abs(s - target), s))

    return {
        "atm": nearest(spot),
        "otm_put": nearest(spot * put_mult),
        "otm_call": nearest(spot * call_mult),
    }


def rank_strikes_by_distance(strikes: list[float], target: float) -> list[float]:
    """Ladder strikes ordered by distance to ``target``, ties broken by strike
    (ascending) so the ordering is deterministic. This is the walk order a leg
    tries when its nearest strike turns out to be dataless."""
    return sorted({float(s) for s in strikes}, key=lambda s: (abs(s - target), s))


# --- Bar dedup & decision-date selection -----------------------------------


def dedup_bars_by_trade_date(bars: list[dict]) -> dict[date, dict]:
    """Collapse the tape's duplicate rows to one bar per trade date.

    The trade date is the date part of ``last_trade``; the terminal emits
    several rows for one date that differ only in ``created`` but carry identical
    OHLC/bid/ask, so keeping the last row seen is safe. Bars without a parseable
    ``last_trade`` carry no usable trade date and are dropped.
    """
    out: dict[date, dict] = {}
    for bar in bars:
        last_trade = bar.get("last_trade")
        if not last_trade:
            continue
        try:
            trade_date = date.fromisoformat(str(last_trade)[:10])
        except ValueError:
            continue
        out[trade_date] = bar  # last row wins; duplicates are identical for our fields
    return out


def select_bar_for_decision(
    bars: list[dict], decision_date: date, *, window_days: int = QUOTE_WINDOW_DAYS
) -> dict | None:
    """The bar ON the decision date, else the nearest PRIOR trade date still
    inside ``[decision - window_days, decision)``. Returns None if neither
    exists (e.g. the contract had no prints in the window)."""
    by_date = dedup_bars_by_trade_date(bars)
    if decision_date in by_date:
        return by_date[decision_date]
    lo = decision_date - timedelta(days=window_days)
    prior = [d for d in by_date if lo <= d < decision_date]
    if not prior:
        return None
    return by_date[max(prior)]


# --- v2 enrichment row selection & extraction --------------------------------


def row_trade_date(row: dict) -> date | None:
    """Trade date of an enrichment/EOD row. The live OI tape stamps its rows
    ``timestamp`` (e.g. ``2023-05-15T06:30:10.000`` -- per Task 1 discovery),
    the EOD tape ``last_trade``; ``date`` is kept as a defensive alias. Only
    the DATE part is used: matching is by calendar date, never exact-timestamp
    equality. None when nothing parses -- an undated row can never be matched
    to a decision bar."""
    for key in ("timestamp", "date", "last_trade"):
        value = row.get(key)
        if value is None:
            continue
        try:
            return date.fromisoformat(str(value)[:10])
        except ValueError:
            continue
    return None


def select_row_for_date(rows: list[dict], trade_date: date) -> dict | None:
    """The row stamped exactly ``trade_date`` (last one wins -- the tape
    double-prints; same rationale as dedup_bars_by_trade_date). None when
    absent: enrichment must describe the SAME session the quote came from,
    so there is deliberately NO nearest-prior fallback here."""
    out: dict | None = None
    for row in rows:
        if row_trade_date(row) == trade_date:
            out = row
    return out


def extract_open_interest(row: dict | None) -> dict:
    """``{"open_interest": int}`` when the row carries a finite value, else
    ``{}``. A vendor-served 0 is a real observation and kept; an ABSENT field
    yields no key -- the additive-schema rule (never fabricate a 0)."""
    if row is None:
        return {}
    try:
        number = float(row.get("open_interest"))
    except (TypeError, ValueError):
        return {}
    if not math.isfinite(number):
        return {}
    return {"open_interest": int(number)}


def _enrich_bar(
    client: OptionsClient,
    symbol: str,
    exp_str: str,
    strike: float,
    right: str,
    bar: dict,
    start: str,
    end: str,
) -> dict:
    """A COPY of the leg's decision bar carrying the additive ``open_interest``
    key when available.

    Order of truth: a field already ON the EOD bar (some terminal builds serve
    open_interest inline) wins without spending a request; otherwise the
    dedicated endpoint is queried and the row matching the bar's OWN trade date
    is extracted. Any failure degrades to key-absent -- enrichment must never
    sink a leg that already has a usable quote. (No greeks path exists: both
    greeks endpoints 404 on this tier -- Task 1 discovery -- so only OI is
    fetched here.)
    """
    enriched = dict(bar)
    for key in _ENRICHMENT_KEYS:  # only extractor-vetted values may carry these keys
        enriched.pop(key, None)
    trade_date = row_trade_date(bar)
    if trade_date is None:
        return enriched

    oi = extract_open_interest(bar)  # inline-on-EOD wins, zero extra requests
    if len(oi) == 0:
        try:
            rows = client.history_open_interest(symbol, exp_str, strike, right, start, end)
            oi = extract_open_interest(select_row_for_date(rows, trade_date))
        except Exception:  # noqa: BLE001 -- degrade to absent, never sink the leg
            log.warning(
                "open-interest fetch failed %s %s %s%s -- key absent",
                symbol, exp_str, strike, right,
            )
    enriched.update(oi)
    return enriched


# --- Cell assembly ---------------------------------------------------------


def _mid(bid, ask) -> float | None:
    """(bid+ask)/2, or None when either side is missing/non-finite."""
    if bid is None or ask is None:
        return None
    try:
        b, a = float(bid), float(ask)
    except (TypeError, ValueError):
        return None
    if not (math.isfinite(b) and math.isfinite(a)):
        return None
    return (b + a) / 2.0

# Cell roles, their option right, and the JSON "type" label.
_ROLE_IS_CALL = {"atm": True, "otm_put": False, "otm_call": True}
_ROLE_TYPE = {"atm": "call", "otm_put": "put", "otm_call": "call"}


def _contracts_and_skews(
    decision_date: date,
    spot: float,
    expiration: date,
    legs: dict[str, tuple[float, dict | None]],
    *,
    rate: float = RATE,
    div_yield: float = DIV_YIELD,
) -> tuple[list[dict], float | None, float | None]:
    """Contracts + (skew_put_atm, skew_put_call) for one expiration's legs.

    Factored out of build_cell so the far block (v2) reuses the identical
    leg->contract semantics: mid from bid/ask, IV from mid, collapsed-call
    drop, and the additive v2 enrichment keys copied only when present.
    """
    # days_to_expiry (hence t_years) is measured from the DECISION date for every
    # leg, even when a leg's quote fell back to a trade date up to 3 days prior.
    # That <=3-day approximation is deliberate and negligible HERE: a skew is a
    # DIFFERENCE of IVs at the same t, so a small common shift in t cancels. We do
    # NOT use per-leg trade dates -- that would give legs different t and corrupt
    # the skew's apples-to-apples comparison.
    days_to_expiry = (expiration - decision_date).days
    t_years = days_to_expiry / 365.0

    # Collapsed ladder: an OTM-call snapped onto the ATM strike is the same
    # contract, not a real risk-reversal leg. Drop it so we emit neither a
    # duplicate contract nor a degenerate skew_put_call (== skew_put_atm).
    collapsed_call = (
        "otm_call" in legs
        and "atm" in legs
        and float(legs["otm_call"][0]) == float(legs["atm"][0])
    )

    contracts: list[dict] = []
    iv_by_role: dict[str, float | None] = {}
    for role in ("atm", "otm_put", "otm_call"):
        if role not in legs:
            continue
        if role == "otm_call" and collapsed_call:
            continue  # degenerate leg -> omitted
        strike, bar = legs[role]
        if bar is None:
            continue  # leg not gathered -> omitted from the cell
        is_call = _ROLE_IS_CALL[role]
        bid, ask, close = bar.get("bid"), bar.get("ask"), bar.get("close")
        mid = _mid(bid, ask)
        iv = (
            implied_vol(mid, spot, strike, t_years, rate=rate, div_yield=div_yield, is_call=is_call)
            if mid is not None
            else None
        )
        iv_by_role[role] = iv
        contract = {
            "role": role,
            "type": _ROLE_TYPE[role],
            "strike": float(strike),
            "bid": bid,
            "ask": ask,
            "close": close,
            "mid": mid,
            "iv": iv,
            # Flow proxy: EOD contract volume + trade count on the decision
            # bar. Only these 3 contracts (ATM / OTM put / OTM call), NOT the
            # full chain, so it is a THIN put-vs-call demand proxy -- enough
            # to test "are people piling into the calls?" without a chain pull.
            "volume": bar.get("volume"),
            "count": bar.get("count"),
        }
        # v2 additive enrichment: keys ride only when the gather vetted a
        # served value onto the bar (absent field -> absent key, never 0).
        for key in _ENRICHMENT_KEYS:
            if key in bar:
                contract[key] = bar[key]
        contracts.append(contract)

    iv_atm = iv_by_role.get("atm")
    iv_put = iv_by_role.get("otm_put")
    iv_call = iv_by_role.get("otm_call")
    skew_put_atm = iv_put - iv_atm if (iv_put is not None and iv_atm is not None) else None
    skew_put_call = iv_put - iv_call if (iv_put is not None and iv_call is not None) else None
    return contracts, skew_put_atm, skew_put_call


def build_cell(
    symbol: str,
    decision_date: date,
    spot: float,
    expiration: date,
    legs: dict[str, tuple[float, dict | None]],
    *,
    rate: float = RATE,
    div_yield: float = DIV_YIELD,
) -> dict | None:
    """Assemble one samples.jsonl cell from the gathered legs.

    ``legs`` maps each role to ``(strike, bar_or_None)``. For every role that
    produced a bar we compute ``mid`` and invert it for ``iv`` (None when the
    quote is uninvertible -- ``implied_vol`` handles that). A role with no bar is
    omitted from ``contracts``. The cell is DROPPED (returns None) when the ATM
    or OTM-put leg is missing, because there is no usable primary skew signal
    without both. Skews are computed from the mid-IVs, None when a leg is absent
    or its IV is None. The cell's top-level shape is the v1 schema exactly;
    v2 enrichment (open interest) rides additively inside the contracts.
    """
    contracts, skew_put_atm, skew_put_call = _contracts_and_skews(
        decision_date, spot, expiration, legs, rate=rate, div_yield=div_yield
    )
    present = {c["role"] for c in contracts}
    if not {"atm", "otm_put"} <= present:
        return None  # no ATM or OTM-put leg -> no primary signal, skip the cell

    return {
        "symbol": symbol,
        "decision_date": decision_date.isoformat(),
        "spot_at_decision": spot,
        "target_expiration": expiration.isoformat(),
        "days_to_expiry": (expiration - decision_date).days,
        "contracts": contracts,
        "skew_put_atm": skew_put_atm,
        "skew_put_call": skew_put_call,
    }


def build_far_block(
    decision_date: date,
    spot: float,
    expiration: date,
    legs: dict[str, tuple[float, dict | None]],
    *,
    rate: float = RATE,
    div_yield: float = DIV_YIELD,
) -> dict | None:
    """The additive ``far`` block: the same contract schema as the near legs,
    at the far monthly. Same ATM+OTM-put requirement as the near cell -- a far
    block without its primary legs carries no usable term-structure reading,
    so it is dropped (the CELL stays valid; see gather_cell)."""
    contracts, skew_put_atm, skew_put_call = _contracts_and_skews(
        decision_date, spot, expiration, legs, rate=rate, div_yield=div_yield
    )
    present = {c["role"] for c in contracts}
    if not {"atm", "otm_put"} <= present:
        return None
    return {
        "target_expiration": expiration.isoformat(),
        "days_to_expiry": (expiration - decision_date).days,
        "contracts": contracts,
        "skew_put_atm": skew_put_atm,
        "skew_put_call": skew_put_call,
    }


# --- Per-cell orchestration ------------------------------------------------


def _resolve_legs(
    client: OptionsClient,
    symbol: str,
    exp_str: str,
    decision_date: date,
    spot: float,
    strikes: list[float],
    *,
    put_mult: float,
    call_mult: float,
    window_days: int,
    max_candidates: int,
) -> dict[str, tuple[float, dict]]:
    """The nearest-strike-with-data walk for ONE expiration (near or far).

    Identical semantics to the v1 gather_cell walk: each role walks outward
    from its nominal target to the nearest strike that actually returns a
    bar (up to ``max_candidates``), and every (strike, right) probe is cached
    so no contract is fetched twice within the cell. Resolved legs are then
    ENRICHED (open interest) exactly once per distinct (strike, right) --
    collapsed roles share the enriched bar.
    """
    start = (decision_date - timedelta(days=window_days)).isoformat()
    end = (decision_date + timedelta(days=window_days)).isoformat()
    bar_cache: dict[tuple[float, bool], dict | None] = {}

    def resolve_leg(target: float, is_call: bool) -> tuple[float, dict] | None:
        right = "C" if is_call else "P"
        for strike in rank_strikes_by_distance(strikes, target)[:max_candidates]:
            key = (strike, is_call)
            if key not in bar_cache:
                bars = client.history_eod(symbol, exp_str, strike, right, start, end)
                bar_cache[key] = select_bar_for_decision(
                    bars, decision_date, window_days=window_days
                )
            bar = bar_cache[key]
            if bar is not None:
                return strike, bar
        return None  # no data-bearing strike within the candidate cap

    role_targets = {
        "atm": (spot, _ROLE_IS_CALL["atm"]),
        "otm_put": (spot * put_mult, _ROLE_IS_CALL["otm_put"]),
        "otm_call": (spot * call_mult, _ROLE_IS_CALL["otm_call"]),
    }
    resolved: dict[str, tuple[float, dict]] = {}
    for role, (target, is_call) in role_targets.items():
        leg = resolve_leg(target, is_call)
        if leg is not None:
            resolved[role] = leg

    enriched_cache: dict[tuple[float, bool], dict] = {}
    legs: dict[str, tuple[float, dict]] = {}
    for role, (strike, bar) in resolved.items():
        is_call = _ROLE_IS_CALL[role]
        key = (strike, is_call)
        if key not in enriched_cache:
            right = "C" if is_call else "P"
            enriched_cache[key] = _enrich_bar(
                client, symbol, exp_str, strike, right, bar, start, end
            )
        legs[role] = (strike, enriched_cache[key])
    return legs


def _gather_far_block(
    client: OptionsClient,
    symbol: str,
    decision_date: date,
    spot: float,
    expirations: list[str],
    near_expiration: date,
    *,
    far_min_dte: int,
    far_max_dte: int,
    put_mult: float,
    call_mult: float,
    window_days: int,
    max_candidates: int,
    rate: float,
    div_yield: float,
) -> dict | None:
    """Resolve + build the far block; None (never raise upward from here for
    data-absence reasons) when the far monthly is missing, its ladder is
    empty, or its primary legs are dataless."""
    far_expiration = select_far_expiration(
        expirations, decision_date, near_expiration, min_dte=far_min_dte, max_dte=far_max_dte
    )
    if far_expiration is None:
        return None
    far_str = far_expiration.isoformat()
    strikes = client.list_strikes(symbol, far_str)
    if len(strikes) == 0:
        return None
    legs = _resolve_legs(
        client, symbol, far_str, decision_date, spot, strikes,
        put_mult=put_mult, call_mult=call_mult, window_days=window_days,
        max_candidates=max_candidates,
    )
    return build_far_block(
        decision_date, spot, far_expiration, legs, rate=rate, div_yield=div_yield
    )


def gather_cell(
    client: OptionsClient,
    symbol: str,
    decision_date: date,
    spot: float,
    *,
    target_dte: int = TARGET_DTE,
    min_dte: int = MIN_DTE,
    max_dte: int = MAX_DTE,
    put_mult: float = OTM_PUT_MULT,
    call_mult: float = OTM_CALL_MULT,
    window_days: int = QUOTE_WINDOW_DAYS,
    max_candidates: int = MAX_STRIKE_CANDIDATES,
    rate: float = RATE,
    div_yield: float = DIV_YIELD,
    far_min_dte: int = FAR_MIN_DTE,
    far_max_dte: int = FAR_MAX_DTE,
    include_far: bool = True,
) -> dict | None:
    """Gather one (symbol, decision-date) cell end to end.

    Selects the target expiration, resolves each role to the nearest strike
    that actually returns a bar (see _resolve_legs), enriches the resolved
    legs with open interest (additive; absent on any failure), and assembles
    the v1-shaped cell. A second (far) expiration block -- the next standard
    monthly after the near target, DTE 55..90 -- is gathered with the
    identical walk and attached additively as ``cell["far"]``. A
    missing/dataless/failing far expiration drops the far block ONLY; the
    near cell stays valid. Returns the cell dict, or None (with a logged
    reason) when the cell should be skipped.
    """
    expirations = client.list_expirations(symbol)
    expiration = select_expiration(
        expirations,
        decision_date,
        target_dte=target_dte,
        min_dte=min_dte,
        max_dte=max_dte,
    )
    if expiration is None:
        log.info("skip %s %s: no standard monthly expiration in window", symbol, decision_date)
        return None

    exp_str = expiration.isoformat()
    strikes = client.list_strikes(symbol, exp_str)
    if len(strikes) == 0:
        log.info("skip %s %s: empty strike ladder for %s", symbol, decision_date, exp_str)
        return None

    legs = _resolve_legs(
        client, symbol, exp_str, decision_date, spot, strikes,
        put_mult=put_mult, call_mult=call_mult, window_days=window_days,
        max_candidates=max_candidates,
    )
    cell = build_cell(
        symbol, decision_date, spot, expiration, legs, rate=rate, div_yield=div_yield
    )
    if cell is None:
        log.info("skip %s %s: ATM or OTM-put leg not gathered", symbol, decision_date)
        return None

    if include_far:
        far = None
        try:
            far = _gather_far_block(
                client, symbol, decision_date, spot, expirations, expiration,
                far_min_dte=far_min_dte, far_max_dte=far_max_dte,
                put_mult=put_mult, call_mult=call_mult, window_days=window_days,
                max_candidates=max_candidates, rate=rate, div_yield=div_yield,
            )
        except Exception:  # noqa: BLE001 -- far failure drops the BLOCK only
            log.warning(
                "far block failed for %s %s -- dropped, near cell kept",
                symbol, decision_date, exc_info=True,
            )
        if far is not None:
            cell["far"] = far
    return cell


# --- Universe selection ----------------------------------------------------


def build_universe(
    membership_csv: Path,
    cache_dir: Path,
    size: int,
    *,
    start_month: str = "2019-01",
    end_month: str = "2025-12",
    membership_window: tuple[str, str] = UNIVERSE_MEMBERSHIP_WINDOW,
    indices: tuple[str, ...] = ("sp500", "ndx"),
) -> list[str]:
    """Top ``size`` names by median daily dollar volume.

    Candidates are symbols that were EVER a member of one of ``indices`` (default
    the large-cap ``sp500``/``ndx``; pass ``("sp400",)`` for the S&P MidCap 400)
    with an interval overlapping ``membership_window`` (survivorship-safe:
    delisted names that are still cached count) AND have a Tiingo parquet in
    ``cache_dir``. Each is ranked by the median of ``close*volume`` over the
    decision span (``start_month``..``end_month``); ties break alphabetically so
    the ordering is deterministic.
    """
    df = pd.read_csv(membership_csv, comment="#", dtype=str).fillna("")
    win_lo = pd.Timestamp(membership_window[0])
    win_hi = pd.Timestamp(membership_window[1])

    candidates: set[str] = set()
    for row in df.itertuples(index=False):
        if row.index not in indices:
            continue
        start = pd.Timestamp(row.start)
        end = pd.Timestamp(row.end) if row.end else pd.Timestamp.max
        if start <= win_hi and end > win_lo:  # [start, end) overlaps the window
            candidates.add(row.symbol)

    lo = pd.Timestamp(pd.Period(start_month, freq="M").start_time, tz="UTC")
    hi = pd.Timestamp(pd.Period(end_month, freq="M").end_time, tz="UTC")

    ranked: list[tuple[float, str]] = []
    for symbol in candidates:
        path = cache_dir / f"{symbol}.parquet"
        if not path.exists():
            continue
        frame = pd.read_parquet(path, columns=["close", "volume"])
        window = frame.loc[(frame.index >= lo) & (frame.index <= hi)]
        if window.empty:
            continue
        dollar_vol = float((window["close"] * window["volume"]).median())
        if not math.isfinite(dollar_vol):
            continue
        ranked.append((dollar_vol, symbol))

    ranked.sort(key=lambda r: (-r[0], r[1]))
    return [symbol for _, symbol in ranked[:size]]


# --- Resume / persistence --------------------------------------------------


def _ensure_trailing_newline(out_path: Path) -> None:
    """Guard against torn-line concatenation on resume.

    If a previous run was SIGKILLed mid-flush the file can end with a partial
    line and no newline; appending the next cell would glue the two into one
    unparseable JSON line. If the last byte is not a newline we add one, so the
    torn fragment stands alone (and is skipped by the tolerant readers) instead
    of corrupting the next good cell.
    """
    if not out_path.exists() or out_path.stat().st_size == 0:
        return
    with out_path.open("rb") as fh:
        fh.seek(-1, 2)  # last byte
        last = fh.read(1)
    if last != b"\n":
        with out_path.open("a") as fh:
            fh.write("\n")


def backup_v1(out_path: Path) -> Path | None:
    """Move an existing samples file aside as ``<stem>.v1<suffix>`` before a
    fresh full re-gather (samples.jsonl -> samples.v1.jsonl). Returns the
    backup path, or None when there is nothing to back up. REFUSES to
    overwrite an existing backup: the v1 baseline must survive exactly as
    gathered -- it feeds the coverage report's IV-agreement check."""
    if not out_path.exists():
        return None
    backup = out_path.with_name(f"{out_path.stem}.v1{out_path.suffix}")
    if backup.exists():
        raise FileExistsError(
            f"refusing to clobber existing v1 backup: {backup} "
            "(move it aside manually if you truly mean to replace the baseline)"
        )
    out_path.rename(backup)
    return backup


def load_existing_keys(out_path: Path) -> set[tuple[str, str]]:
    """(symbol, decision_date) pairs already written, so a re-run resumes without
    duplicating. Malformed lines are ignored -- a torn final line from a killed
    run must not abort the resume."""
    keys: set[tuple[str, str]] = set()
    if not out_path.exists():
        return keys
    with out_path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                cell = json.loads(line)
                keys.add((cell["symbol"], cell["decision_date"]))
            except (ValueError, KeyError):
                continue
    return keys


# --- Top-level run ---------------------------------------------------------


def _load_underlying(cache_dir: Path, symbol: str) -> pd.DataFrame | None:
    path = cache_dir / f"{symbol}.parquet"
    if not path.exists():
        return None
    return pd.read_parquet(path, columns=["close"])


def build_work_items(
    symbols: list[str],
    cache_dir: Path,
    raw_dir: Path,
    start_month: str,
    end_month: str,
    existing: set[tuple[str, str]],
) -> list[tuple[str, date, float]]:
    """One (symbol, decision_date, spot) tuple per not-yet-gathered cell.

    The decision-date CALENDAR (first trading day per month) comes from the
    adjusted ``data/equities-tiingo`` cache -- split/dividend adjustment does not
    move trading dates, so it is the right source for the calendar. The SPOT,
    however, is the RAW unadjusted close from ``raw_dir`` (see ``load_raw_close``)
    because option strikes and the BS inversion live in raw price space. A symbol
    with no raw series -- or a decision date with no raw bar on/before it -- is
    skipped and logged; we never fall back to the adjusted (strike-mismatched)
    price.
    """
    work: list[tuple[str, date, float]] = []
    for symbol in symbols:
        underlying = _load_underlying(cache_dir, symbol)
        if underlying is None or underlying.empty:
            log.warning("no Tiingo cache for %s -- skipping", symbol)
            continue
        raw_close = load_raw_close(symbol, raw_dir)
        if raw_close is None or raw_close.empty:
            log.warning(
                "no raw underlying for %s -- skipping (run backfill_options_underlying_raw)",
                symbol,
            )
            continue
        for ts in first_trading_days(underlying.index, start_month, end_month):
            decision_date = ts.date()
            if (symbol, decision_date.isoformat()) in existing:
                continue
            spot = raw_spot_at(raw_close, ts)
            if spot is None or spot <= 0:
                log.info("skip %s %s: no raw spot on/before decision", symbol, decision_date)
                continue
            work.append((symbol, decision_date, spot))
    return work


def run_gather(
    client: OptionsClient,
    out_path: Path,
    *,
    symbols: list[str] | None = None,
    universe_size: int = 100,
    start_month: str = "2019-01",
    end_month: str = "2025-12",
    cache_dir: Path,
    raw_dir: Path,
    membership_csv: Path,
    indices: tuple[str, ...] = ("sp500", "ndx"),
    max_workers: int = 4,
    log_every: int = 50,
    **cell_kwargs,
) -> dict:
    """Gather every outstanding cell for ``symbols`` (or a freshly-built universe)
    and append them to ``out_path``.

    ``cache_dir`` (adjusted OHLCV) supplies the decision calendar and the
    dollar-volume ranking; ``raw_dir`` (unadjusted closes) supplies the spot that
    feeds strike-snapping and the stored ``spot_at_decision``.

    Idempotent: existing keys are read up front and skipped. Concurrency is one
    worker per cell, capped at ``max_workers`` (<=4 for the terminal), each doing
    its handful of requests sequentially so the in-flight count never exceeds the
    worker count. Writes are serialised behind a lock and flushed per cell so a
    restart loses nothing. Returns a summary dict (also logged).
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if symbols is None:
        symbols = build_universe(
            membership_csv,
            cache_dir,
            universe_size,
            start_month=start_month,
            end_month=end_month,
            indices=indices,
        )
        log.info("built universe of %d symbols", len(symbols))

    _ensure_trailing_newline(out_path)  # a torn final line must not glue onto the next cell
    existing = load_existing_keys(out_path)
    work = build_work_items(symbols, cache_dir, raw_dir, start_month, end_month, existing)
    total = len(work)
    log.info(
        "gather start: %d symbols, %d cells to do (%d already present)",
        len(symbols),
        total,
        len(existing),
    )

    caching = _CachingClient(client)
    counters = {"written": 0, "skipped": 0, "errors": 0, "done": 0}
    lock = threading.Lock()
    started = time.monotonic()

    with out_path.open("a") as fh:

        def process(item: tuple[str, date, float]) -> None:
            symbol, decision_date, spot = item
            cell: dict | None = None
            failed = False
            try:
                cell = gather_cell(caching, symbol, decision_date, spot, **cell_kwargs)
            except Exception:  # noqa: BLE001 -- one bad cell must not sink the run
                failed = True
                log.exception("error gathering %s %s", symbol, decision_date)
            with lock:
                counters["done"] += 1
                if failed:
                    counters["errors"] += 1
                elif cell is None:
                    counters["skipped"] += 1
                else:
                    fh.write(json.dumps(cell) + "\n")
                    fh.flush()
                    counters["written"] += 1
                if counters["done"] % log_every == 0 or counters["done"] == total:
                    elapsed = time.monotonic() - started
                    log.info(
                        "progress %d/%d (written=%d skipped=%d errors=%d) %.1fs",
                        counters["done"],
                        total,
                        counters["written"],
                        counters["skipped"],
                        counters["errors"],
                        elapsed,
                    )

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            list(pool.map(process, work))

    summary = {
        "symbols": len(symbols),
        "cells_attempted": total,
        "written": counters["written"],
        "skipped": counters["skipped"],
        "errors": counters["errors"],
        "already_present": len(existing),
        "elapsed_s": round(time.monotonic() - started, 1),
    }
    log.info("gather done: %s", summary)
    return summary
