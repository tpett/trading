# Options Gather v2 (OI, greeks, term structure) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enrich the ThetaData options gather with per-leg open interest + vendor greeks and a second (far) expiration block, additively, then fully re-gather both pools on the mini with backup, resumability, and a v1-vs-v2 coverage/agreement report.

**Architecture:** All changes live in `src/trading/research/options_gather.py` (client methods + enrichment + far block, behind the same injectable-client seam the tests already use), plus a small new coverage module and two CLI flags. The near cell keeps the exact v1 shape; every v2 field is an ADDITIVE key that is simply absent when the vendor has nothing. The re-gather itself is an ops runbook (Task 6) executed by the orchestrator on the mini — it is not code.

**Tech Stack:** Python 3.12, stdlib `urllib`/`json`, pandas 2.3.1, pytest 8.4.1 (warnings-as-errors), ruff 0.12.3, `uv` runner. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-07-09-options-gather-v2-design.md` (approved 2026-07-09, DATA-ONLY — no signal definitions here).

## Global Constraints

- **ADDITIVE ONLY:** every v1 cell must keep parsing everywhere (gather resume, `trading.alphasearch.panel`, `options_iv.skew_from_cell`); new keys appear only when the vendor served a value. Absent field → key absent — **never** 0, never null-fabricated. (A vendor-served literal 0 is a real observation and is kept.)
- **The four live-data caveats stay honored:** (1) HTTP 472 = "valid request, no data" = empty, not error; (2) listed strikes can be dataless — the nearest-strike-with-data walk with `MAX_STRIKE_CANDIDATES = 4` stays; (3) IV comes from the bid/ask MID, never `close` (EOD `close` is often 0.0); (4) ThetaData strikes are RAW dollars — spot is the unadjusted `close_raw` cache, never the adjusted Tiingo close.
- **DTE bands:** near 25..55 (TARGET_DTE 35, unchanged); far 55..90 (`FAR_MIN_DTE = 55`, `FAR_MAX_DTE = 90`).
- **Terminal limits:** max 4 concurrent requests; ONE terminal per account; v3 API at `localhost:25503`, `symbol` param (not `root`), dashed dates, history capped at ~1 month per request, strikes in raw dollars.
- **Tests:** deterministic, mocked client only — nothing in `tests/` may touch the network or a live terminal. `uv run pytest` runs warnings-as-errors (`filterwarnings = error`).
- **Repo lessons:** no truthiness checks on collections in new code (`len(x) == 0` / `is None`); `pd.Timedelta(N, unit="D")` (never the deprecated positional-string form); ruff clean (`line-length 100`, rules E/F/I/UP/B).
- **Commits:** granular, one logical change each, message explains why, tagged `[AI]`.
- **Nothing under `data/` is ever committed** (gitignored); verification numbers go in docs (Task 6).
- The gather itself runs ON THE MINI (`mac-m1`) — data acquisition is the Task 6 runbook, not a coding task.

## File Structure

- **Modify** `src/trading/research/options_gather.py` — client endpoints (Task 1), leg enrichment (Task 2), far block (Task 3), `backup_v1` (Task 5).
- **Modify** `tests/test_options_gather.py` — FakeClient extension + all gather tests (Tasks 1-5).
- **Create** `src/trading/research/options_coverage.py` — v2-vs-v1 coverage/agreement report (Task 5).
- **Create** `tests/test_options_coverage.py` (Task 5).
- **Create** `scripts/options_coverage_report.py` — CLI wrapper (Task 5).
- **Modify** `scripts/gather_options_iv.py` — `--fresh` backup flag, `--skip-greeks` (Task 5).
- **Modify** `tests/test_alphasearch_panel.py` — enriched-cell tolerance tests (Task 4).
- **Modify** `docs/options-data-vendors.md` — delisted-test verdict (Task 6, runbook — orchestrator edits it with live results).
- **Untouched:** `src/trading/alphasearch/panel.py` (`cell_metrics` must already tolerate enriched cells — Task 4 proves it with tests only; `OPTION_COLUMNS` gains NO new columns, this spec is data-only).

## Ops prerequisite (orchestrator, before Task 1 is implemented)

Task 1 Step 1 is a live endpoint discovery that needs the terminal running on the mini. Execute Runbook steps **R1–R3** (Task 6) first, run the discovery curls, and hand the implementer the verdict (which endpoint serves `open_interest`, which — if any — serves greeks, and the exact row field names). **STOP-and-report if no endpoint serves open interest** (see Task 1 Step 1).

---

### Task 1: OI/greeks endpoints on ThetaClient (+ row selection/extraction helpers)

**Files:**
- Modify: `src/trading/research/options_gather.py` (constants near line 90; `OptionsClient` protocol ~line 96; `ThetaClient` ~line 134; `_CachingClient` ~line 228; new module-level helpers after `select_bar_for_decision` ~line 438)
- Test: `tests/test_options_gather.py`

**Interfaces:**
- Consumes: existing `ThetaClient._get_json`, `_fmt_strike`, `_THETA_NO_DATA_STATUS`.
- Produces (used by Task 2):
  - `OI_PATH: str`, `GREEKS_PATH: str`, `GREEK_FIELDS: tuple[str, ...]`, `_ENRICHMENT_KEYS: tuple[str, ...]`
  - `ThetaClient.history_open_interest(symbol: str, expiration: str, strike: float, right: str, start_date: str, end_date: str) -> list[dict]`
  - `ThetaClient.history_greeks(...same signature...) -> list[dict]` (both also added to the `OptionsClient` protocol and passed through `_CachingClient`)
  - `row_trade_date(row: dict) -> date | None`
  - `select_row_for_date(rows: list[dict], trade_date: date) -> dict | None`
  - `extract_open_interest(row: dict | None) -> dict` (returns `{}` or `{"open_interest": int}`)
  - `extract_greeks(row: dict | None) -> dict` (subset of `{"delta","gamma","theta","vega"} -> float`)

- [ ] **Step 1: LIVE ENDPOINT DISCOVERY (on the mini, terminal already running per Runbook R1–R3)**

The paid Standard tier serves OI and (maybe) greeks, but the exact v3 route was never exercised by this codebase. Verify it now — do NOT trust this plan's guesses. Run each curl from the laptop; contract: AAPL, expiration 2019-04-18, strike 185 C, a within-one-month window (the API caps history at ~1 month/request).

```bash
# (a) Does the EOD tape itself already carry open_interest / greeks fields?
ssh mac-m1 'curl -s "http://127.0.0.1:25503/v3/option/history/eod?symbol=AAPL&expiration=2019-04-18&strike=185&right=C&start_date=2019-02-25&end_date=2019-03-05&format=json"' | python3 -m json.tool | head -60

# (b) Dedicated OI endpoint candidates (stop at the first that returns rows):
ssh mac-m1 'curl -s -w "\nHTTP %{http_code}\n" "http://127.0.0.1:25503/v3/option/history/open_interest?symbol=AAPL&expiration=2019-04-18&strike=185&right=C&start_date=2019-02-25&end_date=2019-03-05&format=json"' | head -40
ssh mac-m1 'curl -s -w "\nHTTP %{http_code}\n" "http://127.0.0.1:25503/v3/option/history/oi?symbol=AAPL&expiration=2019-04-18&strike=185&right=C&start_date=2019-02-25&end_date=2019-03-05&format=json"' | head -40

# (c) Greeks endpoint candidates:
ssh mac-m1 'curl -s -w "\nHTTP %{http_code}\n" "http://127.0.0.1:25503/v3/option/history/greeks?symbol=AAPL&expiration=2019-04-18&strike=185&right=C&start_date=2019-02-25&end_date=2019-03-05&format=json"' | head -40
ssh mac-m1 'curl -s -w "\nHTTP %{http_code}\n" "http://127.0.0.1:25503/v3/option/history/greeks/eod?symbol=AAPL&expiration=2019-04-18&strike=185&right=C&start_date=2019-02-25&end_date=2019-03-05&format=json"' | head -40
ssh mac-m1 'curl -s -w "\nHTTP %{http_code}\n" "http://127.0.0.1:25503/v3/option/history/implied_volatility?symbol=AAPL&expiration=2019-04-18&strike=185&right=C&start_date=2019-02-25&end_date=2019-03-05&format=json"' | head -40

# (d) If everything 404s, ask the terminal what routes exist (404 bodies often list them),
#     and cross-check https://docs.thetadata.us (v3 REST reference).
ssh mac-m1 'curl -s "http://127.0.0.1:25503/v3/nonexistent"' | head -40
```

Interpretation:
- **HTTP 200 with rows** → that's the endpoint. Record the exact path, the row **date field name** (expected `"date"`, dashed) and the OI/greeks field names.
- **HTTP 472** → route EXISTS (472 = valid request, no data in range); try a different window (e.g. `2019-03-25..2019-04-10`, close to expiry) before concluding anything.
- **HTTP 404** → route does not exist; try the next candidate.
- **If (a) shows `open_interest` on the EOD rows themselves**, the dedicated endpoint is optional — the Task 2 code reads the inline field first and never spends the extra request.
- **STOP-AND-REPORT** (do not proceed with Task 1) if NO source serves open interest — neither inline on EOD nor any dedicated endpoint. OI is the load-bearing field for the follow-on `oi_put_call`/`d_oi` signals; report back to the orchestrator so the spec can be amended before code is written against a fiction.
- Greeks missing everywhere is NOT a stop: per spec §2 greeks are "where the endpoint serves them" — proceed, and the gather will be launched with `--skip-greeks` (Task 5 flag) so no requests are wasted.
- Set `OI_PATH` / `GREEKS_PATH` in Step 4 to the VERIFIED paths, and adjust `row_trade_date`'s field list if the live rows date differently than `"date"`. Note the verdict in the Task 1 commit message.

- [ ] **Step 2: Write the failing tests**

Append to `tests/test_options_gather.py` (imports go at the top of the existing import block, alphabetized to keep ruff-I happy):

```python
from trading.research.options_gather import (  # add to the existing import list
    OI_PATH,
    extract_greeks,
    extract_open_interest,
    row_trade_date,
    select_row_for_date,
)
```

```python
# --- OI / greeks endpoints & extraction (v2) --------------------------------


class _FakeResp:
    """Minimal urlopen context-manager double serving one canned JSON body."""

    status = 200

    def __init__(self, payload: dict):
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self):
        return json.dumps(self._payload).encode()


def test_theta_client_history_open_interest_endpoint_and_parse(monkeypatch):
    """history_open_interest hits the verified v3 OI route with the canonical
    contract params and flattens the response envelope to rows."""
    captured = {}

    def fake_urlopen(req, timeout):
        captured["url"] = req.full_url
        return _FakeResp(
            {"response": [{"data": [{"date": "2019-03-01", "open_interest": 1523}]}]}
        )

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    client = ThetaClient("http://x")
    rows = client.history_open_interest(
        "AAPL", "2019-04-18", 185.0, "C", "2019-02-25", "2019-03-05"
    )
    assert rows == [{"date": "2019-03-01", "open_interest": 1523}]
    assert f"{OI_PATH}?" in captured["url"]  # tracks the discovery-verified route
    assert "symbol=AAPL" in captured["url"]
    assert "strike=185" in captured["url"]  # whole-dollar strike, canonical bare-int form
    assert "right=C" in captured["url"]
    assert "start_date=2019-02-25" in captured["url"]  # dashed dates


def test_theta_client_history_greeks_flat_rows_and_472_empty(monkeypatch):
    """Greeks parse tolerates a FLAT response (rows not nested under "data"),
    and a 472 is a normal empty -- one call, no retry, no exception."""
    calls = {"n": 0}

    def flat_then_472(req, timeout):
        calls["n"] += 1
        if calls["n"] == 1:
            return _FakeResp({"response": [{"date": "2019-03-01", "delta": 0.52}]})
        raise urllib.error.HTTPError(req.full_url, 472, "No data", {}, None)

    monkeypatch.setattr("urllib.request.urlopen", flat_then_472)
    client = ThetaClient("http://x", sleep=lambda _s: None)
    rows = client.history_greeks("AAPL", "2019-04-18", 185.0, "C", "2019-02-25", "2019-03-05")
    assert rows == [{"date": "2019-03-01", "delta": 0.52}]
    assert client.history_greeks(
        "AAPL", "2019-04-18", 185.0, "C", "2019-02-25", "2019-03-05"
    ) == []
    assert calls["n"] == 2  # the 472 was NOT retried


def test_row_trade_date_reads_date_then_last_trade():
    assert row_trade_date({"date": "2019-03-01"}) == date(2019, 3, 1)
    assert row_trade_date({"last_trade": "2019-03-01T15:59:56.204"}) == date(2019, 3, 1)
    assert row_trade_date({"date": "garbage"}) is None
    assert row_trade_date({}) is None


def test_select_row_for_date_exact_match_last_wins_no_fallback():
    rows = [
        {"date": "2019-03-01", "open_interest": 10},
        {"date": "2019-03-01", "open_interest": 11},  # tape double-print: last wins
        {"date": "2019-02-28", "open_interest": 9},
    ]
    assert select_row_for_date(rows, date(2019, 3, 1))["open_interest"] == 11
    # NO nearest-prior fallback: enrichment must ride the SAME date as the bar.
    assert select_row_for_date(rows, date(2019, 3, 4)) is None


def test_extract_open_interest_absent_or_junk_yields_no_key():
    assert extract_open_interest(None) == {}
    assert extract_open_interest({"date": "2019-03-01"}) == {}  # field absent -> no key
    assert extract_open_interest({"open_interest": None}) == {}
    assert extract_open_interest({"open_interest": "junk"}) == {}
    assert extract_open_interest({"open_interest": float("nan")}) == {}
    # A vendor-SERVED zero is a real observation, kept (never fabricated, never dropped).
    assert extract_open_interest({"open_interest": 0}) == {"open_interest": 0}
    assert extract_open_interest({"open_interest": 1523.0}) == {"open_interest": 1523}


def test_extract_greeks_partial_row_keeps_good_fields():
    row = {"delta": 0.52, "gamma": None, "theta": "junk", "vega": 0.11, "rho": 0.9}
    assert extract_greeks(row) == {"delta": 0.52, "vega": 0.11}  # rho not in GREEK_FIELDS
    assert extract_greeks(None) == {}
    assert extract_greeks({}) == {}
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_options_gather.py -q 2>&1 | tee /tmp/claude-gather-v2-t1.log`
Expected: ImportError — `cannot import name 'extract_greeks' from 'trading.research.options_gather'`.

- [ ] **Step 4: Implement**

In `src/trading/research/options_gather.py`:

(a) Constants, after the `UNIVERSE_MEMBERSHIP_WINDOW` block (~line 90) — **substitute the Step 1 verified paths**:

```python
# --- v2 enrichment (OI, vendor greeks, far expiration) ----------------------
# Far (term-structure) leg: the NEXT standard monthly after the near target.
# Near sits 25..55 DTE, so the following monthly lands ~55..85; 90 caps a
# straggler month. Spec: 2026-07-09-options-gather-v2-design.md section 2.
FAR_MIN_DTE = 55
FAR_MAX_DTE = 90
# Endpoint paths VERIFIED against the live Standard-tier terminal (plan Task 1
# discovery, 2026-07). If the terminal build changes these, re-run discovery.
OI_PATH = "/v3/option/history/open_interest"
GREEKS_PATH = "/v3/option/history/greeks"
GREEK_FIELDS = ("delta", "gamma", "theta", "vega")
# Contract keys the gather may ADD to a leg; only extractor-vetted values ever
# carry them (absent field -> absent key, never 0).
_ENRICHMENT_KEYS = ("open_interest", *GREEK_FIELDS)
```

(b) Extend the `OptionsClient` protocol (add below `history_eod` inside the class):

```python
    def history_open_interest(
        self,
        symbol: str,
        expiration: str,
        strike: float,
        right: str,
        start_date: str,
        end_date: str,
    ) -> list[dict]: ...
    def history_greeks(
        self,
        symbol: str,
        expiration: str,
        strike: float,
        right: str,
        start_date: str,
        end_date: str,
    ) -> list[dict]: ...
```

(c) Shared param/envelope helpers, placed just above `class ThetaClient`:

```python
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
    (the history/eod shape) or ARE the rows; tolerate both so the OI/greeks
    parsers survive either serialization."""
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
```

(d) `ThetaClient` methods — also refactor `history_eod` to use `_contract_params` (its own flatten stays as-is: EOD entries without `"data"` are contract metadata, not bars):

```python
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

    def history_greeks(
        self,
        symbol: str,
        expiration: str,
        strike: float,
        right: str,
        start_date: str,
        end_date: str,
    ) -> list[dict]:
        payload = self._get_json(
            GREEKS_PATH, _contract_params(symbol, expiration, strike, right, start_date, end_date)
        )
        return _flatten_rows(payload)
```

(e) `_CachingClient` passthroughs (history is unique per leg/date — no memoisation), below its `history_eod`:

```python
    def history_open_interest(self, *args, **kwargs) -> list[dict]:
        return self._client.history_open_interest(*args, **kwargs)

    def history_greeks(self, *args, **kwargs) -> list[dict]:
        return self._client.history_greeks(*args, **kwargs)
```

(f) Row selection/extraction helpers, placed after `select_bar_for_decision` (~line 438):

```python
# --- v2 enrichment row selection & extraction --------------------------------


def row_trade_date(row: dict) -> date | None:
    """Trade date of an enrichment/EOD row: ``date`` (the OI/greeks tapes),
    else the date part of ``last_trade`` (the EOD tape). None when neither
    parses -- an undated row can never be matched to a decision bar."""
    for key in ("date", "last_trade"):
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


def extract_greeks(row: dict | None) -> dict:
    """The finite vendor greeks among GREEK_FIELDS; absent/non-numeric fields
    are omitted, so a partially-served row keeps its good fields."""
    if row is None:
        return {}
    out: dict = {}
    for name in GREEK_FIELDS:
        try:
            number = float(row.get(name))
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            out[name] = number
    return out
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_options_gather.py -q`
Expected: all pass (new + all pre-existing).

- [ ] **Step 6: Lint and commit**

```bash
uv run ruff check src tests scripts
git add src/trading/research/options_gather.py tests/test_options_gather.py
git commit -m "Add verified OI/greeks v3 endpoints + additive extractors to options gather client [AI]"
```

(State the discovery verdict — endpoint paths, field names, greeks served or not — in the commit body.)

---

### Task 2: Per-leg OI + greeks enrichment in gather_cell

**Files:**
- Modify: `src/trading/research/options_gather.py` (`build_cell` ~line 461 refactor; `gather_cell` ~line 562 refactor; new `_enrich_bar` + `_resolve_legs`)
- Test: `tests/test_options_gather.py` (FakeClient upgrade + enrichment tests)

**Interfaces:**
- Consumes (Task 1): `extract_open_interest`, `extract_greeks`, `select_row_for_date`, `row_trade_date`, `_ENRICHMENT_KEYS`, client `history_open_interest` / `history_greeks`.
- Produces (used by Task 3):
  - `_resolve_legs(client, symbol: str, exp_str: str, decision_date: date, spot: float, strikes: list[float], *, put_mult: float, call_mult: float, window_days: int, max_candidates: int, include_greeks: bool) -> dict[str, tuple[float, dict]]` — the per-expiration nearest-strike-with-data walk, legs enriched.
  - `_contracts_and_skews(decision_date: date, spot: float, expiration: date, legs: dict[str, tuple[float, dict | None]], *, rate: float, div_yield: float) -> tuple[list[dict], float | None, float | None]` — contracts + (skew_put_atm, skew_put_call).
  - `gather_cell(..., include_greeks: bool = True)` keyword.
- The near cell's top-level shape and required keys are UNCHANGED; contracts may additionally carry `open_interest`, `delta`, `gamma`, `theta`, `vega`.

- [ ] **Step 1: Upgrade FakeClient (test double) for enrichment and (ahead of Task 3) per-expiration data**

Replace the whole `FakeClient` class in `tests/test_options_gather.py` with:

```python
class FakeClient:
    """Serves canned expirations / strikes / history from in-memory dicts.

    ``history`` (and ``open_interest`` / ``greeks``) are keyed by
    (strike, right) -> list-of-rows, or by (expiration, strike, right) when a
    test needs per-expiration data (the 3-tuple wins). ``strikes`` is a flat
    list served for every expiration, or a {expiration: [strikes]} dict.
    Unknown legs return an empty tape. Records every call so tests can assert
    request de-duplication.
    """

    def __init__(self, expirations=None, strikes=None, history=None,
                 open_interest=None, greeks=None):
        self._expirations = expirations or []
        self._strikes = strikes or []
        self._history = history or {}
        self._open_interest = open_interest or {}
        self._greeks = greeks or {}
        self.history_calls: list[tuple] = []
        self.oi_calls: list[tuple] = []
        self.greeks_calls: list[tuple] = []

    def list_expirations(self, symbol):
        return list(self._expirations)

    def list_strikes(self, symbol, expiration):
        if isinstance(self._strikes, dict):
            return list(self._strikes.get(expiration, []))
        return list(self._strikes)

    @staticmethod
    def _lookup(table, expiration, strike, right):
        key3 = (expiration, float(strike), right)
        if key3 in table:
            return list(table[key3])
        return list(table.get((float(strike), right), []))

    def history_eod(self, symbol, expiration, strike, right, start_date, end_date):
        self.history_calls.append((float(strike), right, start_date, end_date))
        return self._lookup(self._history, expiration, strike, right)

    def history_open_interest(self, symbol, expiration, strike, right, start_date, end_date):
        self.oi_calls.append((expiration, float(strike), right))
        return self._lookup(self._open_interest, expiration, strike, right)

    def history_greeks(self, symbol, expiration, strike, right, start_date, end_date):
        self.greeks_calls.append((expiration, float(strike), right))
        return self._lookup(self._greeks, expiration, strike, right)
```

(`history_calls` keeps its v1 4-tuple shape, so every existing assertion still holds.)

- [ ] **Step 2: Write the failing tests**

Append to `tests/test_options_gather.py`:

```python
# --- Per-leg OI + greeks enrichment (v2) -------------------------------------


def _enrichment_client() -> FakeClient:
    """The 90/100/110 cell with OI for both resolved legs and greeks for the
    ATM leg only."""
    dte = (date(2019, 4, 18) - date(2019, 3, 1)).days
    t = dte / 365
    price_c = bs_price(100.0, 100.0, t, RATE, DIV_YIELD, 0.30, True)
    price_p = bs_price(100.0, 90.0, t, RATE, DIV_YIELD, 0.35, False)
    price_oc = bs_price(100.0, 110.0, t, RATE, DIV_YIELD, 0.25, True)
    history = {
        (100.0, "C"): [_bar("2019-03-01", price_c - 0.05, price_c + 0.05, price_c, "c")],
        (90.0, "P"): [_bar("2019-03-01", price_p - 0.02, price_p + 0.02, price_p, "c")],
        (110.0, "C"): [_bar("2019-03-01", price_oc - 0.02, price_oc + 0.02, price_oc, "c")],
    }
    open_interest = {
        (100.0, "C"): [{"date": "2019-03-01", "open_interest": 1500}],
        (90.0, "P"): [{"date": "2019-03-01", "open_interest": 2200}],
        # (110.0, "C") has NO OI row -> key absent on that leg.
    }
    greeks = {
        (100.0, "C"): [
            {"date": "2019-03-01", "delta": 0.53, "gamma": 0.02, "theta": -0.03, "vega": 0.12}
        ],
    }
    return FakeClient(
        expirations=["2019-04-18"], strikes=[90.0, 100.0, 110.0],
        history=history, open_interest=open_interest, greeks=greeks,
    )


def test_gather_cell_attaches_oi_and_greeks_additively():
    cell = gather_cell(_enrichment_client(), "AAA", date(2019, 3, 1), 100.0)
    assert cell is not None
    by_role = {c["role"]: c for c in cell["contracts"]}
    assert by_role["atm"]["open_interest"] == 1500
    assert by_role["otm_put"]["open_interest"] == 2200
    assert "open_interest" not in by_role["otm_call"]  # no OI row -> key ABSENT, never 0
    assert by_role["atm"]["delta"] == 0.53
    assert by_role["atm"]["vega"] == 0.12
    assert "delta" not in by_role["otm_put"]  # greeks served for atm only
    # v1 fields intact on an enriched leg.
    assert by_role["atm"]["iv"] == pytest.approx(0.30, abs=2e-3)
    assert by_role["atm"]["volume"] is None  # _bar carries no volume; unchanged semantics


def test_gather_cell_oi_row_on_wrong_date_yields_no_key():
    client = _enrichment_client()
    client._open_interest = {
        (100.0, "C"): [{"date": "2019-02-25", "open_interest": 999}],  # not the bar's date
    }
    cell = gather_cell(client, "AAA", date(2019, 3, 1), 100.0)
    by_role = {c["role"]: c for c in cell["contracts"]}
    assert "open_interest" not in by_role["atm"]


def test_gather_cell_enrichment_failure_degrades_to_absent_keys():
    class ExplodingEnrichment(FakeClient):
        def history_open_interest(self, *args, **kwargs):
            raise OSError("boom")

        def history_greeks(self, *args, **kwargs):
            raise OSError("boom")

    base = _enrichment_client()
    client = ExplodingEnrichment(
        expirations=["2019-04-18"], strikes=[90.0, 100.0, 110.0], history=base._history
    )
    cell = gather_cell(client, "AAA", date(2019, 3, 1), 100.0)
    assert cell is not None  # enrichment failure must never sink a good cell
    for contract in cell["contracts"]:
        assert "open_interest" not in contract
        assert "delta" not in contract
        assert contract["iv"] is not None  # the quote side of the leg is untouched


def test_gather_cell_include_greeks_false_spends_no_greeks_requests():
    client = _enrichment_client()
    cell = gather_cell(client, "AAA", date(2019, 3, 1), 100.0, include_greeks=False)
    assert client.greeks_calls == []
    by_role = {c["role"]: c for c in cell["contracts"]}
    assert "delta" not in by_role["atm"]
    assert by_role["atm"]["open_interest"] == 1500  # OI still gathered


def test_gather_cell_enriches_collapsed_roles_once():
    """otm_call collapsed onto the ATM strike: one contract, one OI request."""
    dte = (date(2019, 4, 18) - date(2019, 3, 1)).days
    t = dte / 365
    price_c = bs_price(100.0, 100.0, t, RATE, DIV_YIELD, 0.30, True)
    price_p = bs_price(100.0, 90.0, t, RATE, DIV_YIELD, 0.35, False)
    client = FakeClient(
        expirations=["2019-04-18"],
        strikes=[90.0, 100.0],
        history={
            (100.0, "C"): [_bar("2019-03-01", price_c - 0.05, price_c + 0.05, price_c, "c")],
            (90.0, "P"): [_bar("2019-03-01", price_p - 0.02, price_p + 0.02, price_p, "c")],
        },
        open_interest={(100.0, "C"): [{"date": "2019-03-01", "open_interest": 7}]},
    )
    gather_cell(client, "AAA", date(2019, 3, 1), 100.0)
    oi_for_atm = [c for c in client.oi_calls if c[1] == 100.0 and c[2] == "C"]
    assert len(oi_for_atm) == 1
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_options_gather.py -q`
Expected: new tests FAIL (`KeyError: 'open_interest'` / `TypeError: gather_cell() got an unexpected keyword argument 'include_greeks'`); all v1 tests still PASS (the FakeClient upgrade is backwards-compatible).

- [ ] **Step 4: Implement**

In `src/trading/research/options_gather.py`:

(a) New `_enrich_bar`, placed after the extraction helpers from Task 1:

```python
def _enrich_bar(
    client: OptionsClient,
    symbol: str,
    exp_str: str,
    strike: float,
    right: str,
    bar: dict,
    start: str,
    end: str,
    *,
    include_greeks: bool,
) -> dict:
    """A COPY of the leg's decision bar carrying additive enrichment keys.

    Order of truth: a field already ON the EOD bar (some terminal builds serve
    open_interest inline) wins without spending a request; otherwise the
    dedicated endpoint is queried and the row matching the bar's OWN trade
    date is extracted. Any failure degrades to key-absent (spec section 6) --
    enrichment must never sink a leg that already has a usable quote.
    ``include_greeks=False`` skips the greeks endpoint entirely (set when
    discovery showed the tier does not serve it, so no requests are wasted).
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

    greeks = extract_greeks(bar)
    if len(greeks) == 0 and include_greeks:
        try:
            rows = client.history_greeks(symbol, exp_str, strike, right, start, end)
            greeks = extract_greeks(select_row_for_date(rows, trade_date))
        except Exception:  # noqa: BLE001
            log.warning(
                "greeks fetch failed %s %s %s%s -- keys absent",
                symbol, exp_str, strike, right,
            )
    enriched.update(greeks)
    return enriched
```

(b) New `_resolve_legs` — the walk lifted verbatim out of `gather_cell`, plus enrichment of each distinct resolved contract (place directly above `gather_cell`):

```python
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
    include_greeks: bool,
) -> dict[str, tuple[float, dict]]:
    """The nearest-strike-with-data walk for ONE expiration (near or far).

    Identical semantics to the v1 gather_cell walk: each role walks outward
    from its nominal target to the nearest strike that actually returns a
    bar (up to ``max_candidates``), and every (strike, right) probe is cached
    so no contract is fetched twice within the cell. Resolved legs are then
    ENRICHED (OI / greeks) exactly once per distinct (strike, right) --
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
                client, symbol, exp_str, strike, right, bar, start, end,
                include_greeks=include_greeks,
            )
        legs[role] = (strike, enriched_cache[key])
    return legs
```

(c) Refactor `build_cell`: extract its body into `_contracts_and_skews` (the enrichment-key copy is the ONLY new behavior; everything else moves verbatim, comments included):

```python
def _contracts_and_skews(
    decision_date: date,
    spot: float,
    expiration: date,
    legs: dict[str, tuple[float, dict | None]],
    *,
    rate: float,
    div_yield: float,
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
    v2 enrichment (OI / greeks) rides additively inside the contracts.
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
```

(d) Rewrite `gather_cell` to use `_resolve_legs` (far wiring lands in Task 3 — for now the tail is identical to v1):

```python
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
    include_greeks: bool = True,
) -> dict | None:
    """Gather one (symbol, decision-date) cell end to end.

    Selects the target expiration, resolves each role to the nearest strike
    that actually returns a bar (see _resolve_legs), enriches the resolved
    legs with OI / vendor greeks (additive; absent on any failure), and
    assembles the v1-shaped cell. Returns the cell dict, or None (with a
    logged reason) when the cell should be skipped.
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
        max_candidates=max_candidates, include_greeks=include_greeks,
    )
    cell = build_cell(
        symbol, decision_date, spot, expiration, legs, rate=rate, div_yield=div_yield
    )
    if cell is None:
        log.info("skip %s %s: ATM or OTM-put leg not gathered", symbol, decision_date)
    return cell
```

- [ ] **Step 5: Run the full gather test file**

Run: `uv run pytest tests/test_options_gather.py -q`
Expected: all pass — including every v1 test (`test_build_cell_schema_shape` proves un-enriched contracts keep the EXACT v1 key set; the walk tests prove probe caching is unchanged).

- [ ] **Step 6: Full suite + lint, then commit**

Run: `uv run pytest -q 2>&1 | tail -5` and `uv run ruff check src tests scripts`
Expected: all pass, ruff clean.

```bash
git add src/trading/research/options_gather.py tests/test_options_gather.py
git commit -m "Enrich gathered option legs with additive OI + vendor greeks [AI]"
```

---

### Task 3: Far (second-expiration) block

**Files:**
- Modify: `src/trading/research/options_gather.py` (new `select_far_expiration`, `build_far_block`, `_gather_far_block`; `gather_cell` tail)
- Test: `tests/test_options_gather.py`

**Interfaces:**
- Consumes (Tasks 1-2): `is_standard_monthly`, `_as_date`, `_resolve_legs`, `_contracts_and_skews`, `FAR_MIN_DTE`, `FAR_MAX_DTE`.
- Produces:
  - `select_far_expiration(expirations: list[str | date], decision_date: date, near_expiration: date, *, min_dte: int = FAR_MIN_DTE, max_dte: int = FAR_MAX_DTE) -> date | None`
  - `build_far_block(decision_date: date, spot: float, expiration: date, legs: dict[str, tuple[float, dict | None]], *, rate: float = RATE, div_yield: float = DIV_YIELD) -> dict | None`
  - `gather_cell(..., far_min_dte: int = FAR_MIN_DTE, far_max_dte: int = FAR_MAX_DTE, include_far: bool = True)` — cell gains an optional `"far"` key: `{"target_expiration": str, "days_to_expiry": int, "contracts": [...], "skew_put_atm": float|None, "skew_put_call": float|None}`.
- Cells WITHOUT a far block stay exactly v1-shaped; `far` failures/absence never invalidate the near cell (spec §6).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_options_gather.py` (add `select_far_expiration`, `FAR_MIN_DTE`, and `FAR_MAX_DTE` to the import list):

```python
# --- Far (second expiration) block (v2) --------------------------------------

V1_CELL_KEYS = {
    "symbol", "decision_date", "spot_at_decision", "target_expiration",
    "days_to_expiry", "contracts", "skew_put_atm", "skew_put_call",
}


def test_select_far_expiration_picks_next_monthly_in_band():
    decision = date(2019, 3, 1)
    near = date(2019, 4, 18)
    expirations = [
        "2019-04-18",  # the near itself -> excluded (not strictly after)
        "2019-05-10",  # a weekly in-band -> ignored (not a standard monthly)
        "2019-05-17",  # 77 DTE monthly -> pick (the NEXT monthly after near)
        "2019-06-21",  # 112 DTE -> above the far band
    ]
    assert select_far_expiration(expirations, decision, near) == date(2019, 5, 17)


def test_select_far_expiration_none_when_band_empty_or_only_near():
    decision = date(2019, 3, 1)
    near = date(2019, 4, 18)
    assert select_far_expiration(["2019-04-18"], decision, near) is None
    # A monthly in the DTE band that IS the near expiration must not be reused:
    # decision 2019-03-20 -> 2019-05-17 = 58 DTE (in 55..90) but == near.
    assert (
        select_far_expiration(["2019-05-17", "2019-06-21"], date(2019, 3, 20), date(2019, 5, 17))
        is None  # 2019-06-21 is 93 DTE, above the band
    )


def _near_far_client() -> FakeClient:
    """Near 2019-04-18 and far 2019-05-17 both with data, priced at their own
    tenors so the far ATM IV (0.28) is distinguishable from the near (0.30)."""
    decision = date(2019, 3, 1)
    t_near = (date(2019, 4, 18) - decision).days / 365
    t_far = (date(2019, 5, 17) - decision).days / 365
    near_c = bs_price(100.0, 100.0, t_near, RATE, DIV_YIELD, 0.30, True)
    near_p = bs_price(100.0, 90.0, t_near, RATE, DIV_YIELD, 0.35, False)
    far_c = bs_price(100.0, 100.0, t_far, RATE, DIV_YIELD, 0.28, True)
    far_p = bs_price(100.0, 90.0, t_far, RATE, DIV_YIELD, 0.33, False)
    history = {
        ("2019-04-18", 100.0, "C"): [_bar("2019-03-01", near_c - 0.05, near_c + 0.05, near_c, "c")],
        ("2019-04-18", 90.0, "P"): [_bar("2019-03-01", near_p - 0.02, near_p + 0.02, near_p, "c")],
        ("2019-05-17", 100.0, "C"): [_bar("2019-03-01", far_c - 0.05, far_c + 0.05, far_c, "c")],
        ("2019-05-17", 90.0, "P"): [_bar("2019-03-01", far_p - 0.02, far_p + 0.02, far_p, "c")],
    }
    open_interest = {
        ("2019-05-17", 100.0, "C"): [{"date": "2019-03-01", "open_interest": 640}],
    }
    return FakeClient(
        expirations=["2019-04-18", "2019-05-17"],
        strikes=[90.0, 100.0],
        history=history,
        open_interest=open_interest,
    )


def test_gather_cell_composes_near_plus_far_blocks():
    cell = gather_cell(_near_far_client(), "AAA", date(2019, 3, 1), 100.0)
    assert cell is not None
    assert set(cell) == V1_CELL_KEYS | {"far"}  # far is the ONLY new top-level key
    far = cell["far"]
    assert far["target_expiration"] == "2019-05-17"
    assert far["days_to_expiry"] == (date(2019, 5, 17) - date(2019, 3, 1)).days
    assert FAR_MIN_DTE <= far["days_to_expiry"] <= FAR_MAX_DTE
    far_by_role = {c["role"]: c for c in far["contracts"]}
    assert far_by_role["atm"]["iv"] == pytest.approx(0.28, abs=3e-3)  # far tenor, own IV
    assert far_by_role["atm"]["open_interest"] == 640  # far legs are enriched too
    # Near block untouched by the far gather.
    near_by_role = {c["role"]: c for c in cell["contracts"]}
    assert near_by_role["atm"]["iv"] == pytest.approx(0.30, abs=3e-3)
    assert cell["target_expiration"] == "2019-04-18"


def test_gather_cell_far_absent_when_no_far_monthly():
    client = _near_far_client()
    client._expirations = ["2019-04-18"]  # nothing after the near monthly
    cell = gather_cell(client, "AAA", date(2019, 3, 1), 100.0)
    assert cell is not None
    assert "far" not in cell  # absent, not null/empty
    assert set(cell) == V1_CELL_KEYS  # bit-for-bit v1 shape


def test_gather_cell_far_legs_dataless_drops_far_only():
    client = _near_far_client()
    client._history = {
        k: v for k, v in client._history.items() if k[0] == "2019-04-18"
    }  # far expiration listed but its whole tape is empty
    cell = gather_cell(client, "AAA", date(2019, 3, 1), 100.0)
    assert cell is not None
    assert "far" not in cell


def test_gather_cell_far_exception_drops_block_only():
    class FarExploding(FakeClient):
        def list_strikes(self, symbol, expiration):
            if expiration == "2019-05-17":
                raise OSError("terminal hiccup on the far ladder")
            return super().list_strikes(symbol, expiration)

    base = _near_far_client()
    client = FarExploding(
        expirations=["2019-04-18", "2019-05-17"],
        strikes=[90.0, 100.0],
        history=base._history,
    )
    cell = gather_cell(client, "AAA", date(2019, 3, 1), 100.0)
    assert cell is not None  # near cell survives (spec section 6)
    assert "far" not in cell


def test_gather_cell_include_far_false_spends_no_far_requests():
    client = _near_far_client()
    cell = gather_cell(client, "AAA", date(2019, 3, 1), 100.0, include_far=False)
    assert "far" not in cell
    # Only the near walk's probes happened: (100,C) + (90,P); the otm_call walk
    # reuses the cached (100,C). Far would have added two more history calls.
    assert len(client.history_calls) == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_options_gather.py -q`
Expected: ImportError on `select_far_expiration`, then (after a stub import fix attempt) failures — do not stub; go implement.

- [ ] **Step 3: Implement**

In `src/trading/research/options_gather.py`:

(a) `select_far_expiration`, directly below `select_expiration`:

```python
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
```

(b) `build_far_block` + `_gather_far_block`, directly below `build_cell`:

```python
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
    so it is dropped (the CELL stays valid; spec section 6)."""
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
    include_greeks: bool,
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
        max_candidates=max_candidates, include_greeks=include_greeks,
    )
    return build_far_block(
        decision_date, spot, far_expiration, legs, rate=rate, div_yield=div_yield
    )
```

(c) `gather_cell`: add the keywords `far_min_dte: int = FAR_MIN_DTE, far_max_dte: int = FAR_MAX_DTE, include_far: bool = True` (after `max_dte`), and replace the tail (after the `if cell is None:` log) with:

```python
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
                include_greeks=include_greeks,
            )
        except Exception:  # noqa: BLE001 -- far failure drops the BLOCK only (spec section 6)
            log.warning(
                "far block failed for %s %s -- dropped, near cell kept",
                symbol, decision_date, exc_info=True,
            )
        if far is not None:
            cell["far"] = far
    return cell
```

Also update `gather_cell`'s docstring final paragraph to:

```
    A second (far) expiration block -- the next standard monthly after the
    near target, DTE 55..90 -- is gathered with the identical walk and
    attached additively as ``cell["far"]``. A missing/dataless/failing far
    expiration drops the far block ONLY; the near cell stays valid.
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_options_gather.py -q`
Expected: all pass. Note two v1 tests now exercise far incidentally (`test_run_gather_resume_is_idempotent` / torn-line: their expiration lists contain a monthly in the far band and `_full_history()` serves every expiration via the 2-tuple fallback, so cells gain a valid `far` block) — resume keys are unchanged, so they must still pass unmodified. If either fails, that is a real regression; do not edit the v1 test to make it pass.

- [ ] **Step 5: Full suite + lint, then commit**

Run: `uv run pytest -q 2>&1 | tail -5` and `uv run ruff check src tests scripts`

```bash
git add src/trading/research/options_gather.py tests/test_options_gather.py
git commit -m "Gather a far (next-monthly, DTE 55-90) expiration block per cell, additively [AI]"
```

---

### Task 4: Downstream tolerance — panel reader + schema round-trip

**Files:**
- Modify: `tests/test_alphasearch_panel.py` (tests only — `src/trading/alphasearch/panel.py` must NOT change; this is the proof it already tolerates enriched cells)
- Modify: `tests/test_options_gather.py` (round-trip test)

**Interfaces:**
- Consumes: `cell_metrics`, `load_options`, `cells_have_volume`, `OPTION_COLUMNS` from `trading.alphasearch.panel`; `gather_cell`, `load_existing_keys` from the gather.
- Produces: nothing new — regression armor. `OPTION_COLUMNS` gains NO columns (spec is data-only; signal definitions freeze in the follow-on batch spec).

- [ ] **Step 1: Write the panel tolerance tests**

In `tests/test_alphasearch_panel.py`, add `cells_have_volume` to the existing `trading.alphasearch.panel` import list, then append:

```python
# --------------------------------------------------------------------------- #
# Options gather v2 tolerance: enriched cells must parse identically
# --------------------------------------------------------------------------- #


def _enriched_cell(symbol: str, date: str) -> dict:
    """A v2 cell: the v1 `_cell` plus per-leg OI/greeks and a far block."""
    cell = _cell(symbol, date)
    for contract in cell["contracts"]:
        contract["open_interest"] = 1500
        contract["delta"] = 0.5
        contract["gamma"] = 0.01
        contract["theta"] = -0.03
        contract["vega"] = 0.12
    cell["far"] = {
        "target_expiration": "2020-03-20",
        "days_to_expiry": 78,
        "contracts": [
            {"role": "atm", "iv": 0.31, "volume": 10, "open_interest": 5},
            {"role": "otm_put", "iv": 0.35, "volume": 4},
        ],
        "skew_put_atm": 0.04,
        "skew_put_call": None,
    }
    return cell


def test_cell_metrics_identical_on_v2_enriched_cell():
    """Enrichment keys and the far block are INVISIBLE to cell_metrics: every
    OPTION_COLUMNS value matches the un-enriched v1 cell exactly."""
    v1 = cell_metrics(_cell("AAA", "2020-01-02"))
    v2 = cell_metrics(_enriched_cell("AAA", "2020-01-02"))
    assert set(v2) == set(OPTION_COLUMNS)
    for key in OPTION_COLUMNS:
        if isinstance(v1[key], float) and math.isnan(v1[key]):
            assert math.isnan(v2[key])
        else:
            assert v2[key] == v1[key], key


def test_load_options_parses_mixed_v1_and_v2_lines(tmp_path):
    path = tmp_path / "samples.jsonl"
    lines = [
        json.dumps(_cell("AAA", "2020-01-02")),          # v1 cell
        json.dumps(_enriched_cell("AAA", "2020-02-03")),  # v2 cell, same symbol
        json.dumps(_enriched_cell("BBB", "2020-01-02")),
    ]
    path.write_text("\n".join(lines) + "\n")
    frames, corrupt, has_volume = load_options(path)
    assert corrupt == 0  # enrichment is NOT corruption
    assert set(frames) == {"AAA", "BBB"}
    assert len(frames["AAA"]) == 2  # both vintages land in one frame
    assert list(frames["AAA"].columns) == OPTION_COLUMNS
    assert frames["AAA"].dtypes.eq("float64").all()
    assert has_volume  # _cell legs carry volume


def test_cells_have_volume_reads_near_legs_only():
    """Volume living ONLY in the far block must not unlock the option-volume
    family: cp_vol/wing_vol/tot_vol are near-leg metrics, so the gate reads
    cell["contracts"] and nothing else."""
    cell = _enriched_cell("AAA", "2020-01-02")
    for contract in cell["contracts"]:
        del contract["volume"]  # near legs unmeasured; far still has volume
    assert cells_have_volume([cell]) is False
```

- [ ] **Step 2: Write the gather-side round-trip test**

Append to `tests/test_options_gather.py`:

```python
def test_enriched_cell_json_round_trip_and_resume_key(tmp_path: Path):
    """A gathered v2 cell survives JSON serialization exactly, and the resume
    reader extracts the same (symbol, decision_date) key it always did."""
    cell = gather_cell(_near_far_client(), "AAA", date(2019, 3, 1), 100.0)
    assert cell is not None and "far" in cell
    line = json.dumps(cell)
    assert json.loads(line) == cell  # nothing in the cell is non-JSON-native
    out = tmp_path / "samples.jsonl"
    out.write_text(line + "\n")
    assert load_existing_keys(out) == {("AAA", "2019-03-01")}
```

- [ ] **Step 3: Run tests to verify current behavior**

Run: `uv run pytest tests/test_alphasearch_panel.py tests/test_options_gather.py -q`
Expected: **all pass immediately** — `cell_metrics` reads roles/keys defensively (`d.get(role, {}).get(...)`, `cell.get("contracts", [])`) and ignores unknown keys. If ANY of these fail, `panel.py` has a real intolerance: STOP and report to the orchestrator before touching `panel.py` (a reader change is outside this task's intent and needs a decision).

- [ ] **Step 4: Full suite + lint, then commit**

Run: `uv run pytest -q 2>&1 | tail -5` and `uv run ruff check src tests scripts`

```bash
git add tests/test_alphasearch_panel.py tests/test_options_gather.py
git commit -m "Prove panel + resume readers tolerate v2-enriched option cells [AI]"
```

---

### Task 5: Fresh-gather backup, resume under the enriched schema, coverage report

**Files:**
- Modify: `src/trading/research/options_gather.py` (new `backup_v1`, placed next to `load_existing_keys`)
- Modify: `scripts/gather_options_iv.py` (`--fresh`, `--skip-greeks`)
- Create: `src/trading/research/options_coverage.py`
- Create: `tests/test_options_coverage.py`
- Create: `scripts/options_coverage_report.py`
- Test: `tests/test_options_gather.py` (backup + enriched-resume tests)

**Interfaces:**
- Consumes: `run_gather` (unchanged signature — `include_greeks` flows through the existing `**cell_kwargs`), `load_existing_keys`.
- Produces:
  - `backup_v1(out_path: Path) -> Path | None` — renames `samples.jsonl` → `samples.v1.jsonl` (stem + `.v1` + suffix); returns None when there is nothing to back up; raises `FileExistsError` rather than clobbering an existing backup.
  - `trading.research.options_coverage.load_cells(path: Path) -> list[dict]`
  - `trading.research.options_coverage.iv_deltas(v2_cells: list[dict], v1_cells: list[dict]) -> list[float]`
  - `trading.research.options_coverage.coverage_report(v2_cells: list[dict], v1_cells: list[dict]) -> dict` with keys `cells_v2, cells_v1, leg_volume_rate, oi_leg_rate, greeks_leg_rate, far_rate, iv_overlap_legs, iv_median_abs_delta, iv_red_flag`
  - `IV_DRIFT_RED_FLAG = 0.01` (median |iv_v2 − iv_v1| above one vol point on the SAME contract/date = investigate, per spec §5)
  - CLI: `uv run python scripts/options_coverage_report.py --v2 <path> --v1 <path>` prints the report as JSON.
  - CLI: `scripts/gather_options_iv.py --fresh` (backup then full re-gather) and `--skip-greeks`.

- [ ] **Step 1: Write the failing backup + enriched-resume tests**

Append to `tests/test_options_gather.py` (add `backup_v1` to the import list):

```python
# --- v2 fresh-gather backup + resume under the enriched schema ---------------


def test_backup_v1_renames_once_and_refuses_clobber(tmp_path: Path):
    out = tmp_path / "samples.jsonl"
    out.write_text('{"symbol":"AAA","decision_date":"2019-01-01"}\n')
    backup = backup_v1(out)
    assert backup == tmp_path / "samples.v1.jsonl"
    assert not out.exists() and backup.exists()
    # A second fresh run must NEVER destroy the v1 baseline.
    out.write_text("regathered\n")
    with pytest.raises(FileExistsError):
        backup_v1(out)
    assert backup.read_text().startswith('{"symbol"')  # baseline untouched
    # Nothing to back up -> None, no file created.
    assert backup_v1(tmp_path / "absent.jsonl") is None
    # The mid-cap file maps to its own backup name.
    mid = tmp_path / "samples-midcap.jsonl"
    mid.write_text("x\n")
    assert backup_v1(mid) == tmp_path / "samples-midcap.v1.jsonl"


def test_run_gather_resume_skips_enriched_cells(tmp_path: Path):
    """The completed-cell skip must hold when the existing file contains
    v2-enriched cells (OI/greeks/far): keys are (symbol, decision_date), and
    a second run adds nothing."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    raw_dir = tmp_path / "raw"
    _write_underlying(cache_dir, "AAA")
    _write_raw(raw_dir, "AAA", 100.0)
    out = tmp_path / "samples.jsonl"

    hist = _full_history()
    oi = {
        (100.0, "C"): [], (90.0, "P"): [], (110.0, "C"): [],
    }
    for key, bars in hist.items():
        oi[key] = [{"date": b["last_trade"][:10], "open_interest": 500} for b in bars]
    client = FakeClient(
        expirations=["2019-02-15", "2019-03-15", "2019-04-18"],
        strikes=[90.0, 100.0, 110.0],
        history=hist,
        open_interest=oi,
    )
    kwargs = dict(
        symbols=["AAA"], start_month="2019-01", end_month="2019-03",
        cache_dir=cache_dir, raw_dir=raw_dir, membership_csv=tmp_path / "unused.csv",
    )
    summary1 = run_gather(client, out, **kwargs)
    assert summary1["written"] > 0
    first = out.read_text().splitlines()
    cells = [json.loads(x) for x in first]
    assert all(
        any("open_interest" in c for c in cell["contracts"]) for cell in cells
    )  # the file really is enriched
    summary2 = run_gather(client, out, **kwargs)
    assert summary2["cells_attempted"] == 0
    assert out.read_text().splitlines() == first
```

- [ ] **Step 2: Run to verify failure, then implement `backup_v1` + script flags**

Run: `uv run pytest tests/test_options_gather.py -q` → ImportError on `backup_v1`.

(a) In `src/trading/research/options_gather.py`, directly below `load_existing_keys`:

```python
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
```

(b) In `scripts/gather_options_iv.py`: import `backup_v1` alongside `ThetaClient, run_gather`; add the arguments after `--indices`:

```python
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="back up an existing --out to <stem>.v1.jsonl, then re-gather from "
        "scratch (the v2 full re-gather; refuses to overwrite an existing backup)",
    )
    parser.add_argument(
        "--skip-greeks",
        action="store_true",
        help="skip the vendor-greeks endpoint (use when discovery showed the "
        "tier does not serve greeks, so no requests are wasted on 404 retries)",
    )
```

and between `logging.basicConfig(...)` and `client = ThetaClient(...)`:

```python
    if args.fresh:
        backup = backup_v1(args.out)
        if backup is not None:
            logging.getLogger("scripts.gather_options_iv").info(
                "backed up %s -> %s", args.out, backup
            )
```

then thread the flag through the call: add `include_greeks=not args.skip_greeks,` to the `run_gather(...)` kwargs (it flows into `gather_cell` via `**cell_kwargs`).

Run: `uv run pytest tests/test_options_gather.py -q` → all pass.

- [ ] **Step 3: Write the failing coverage tests**

Create `tests/test_options_coverage.py`:

```python
"""v2-vs-v1 gather coverage/agreement report (spec section 5). Pure fixtures,
no network, no real data files."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from trading.research.options_coverage import (
    IV_DRIFT_RED_FLAG,
    coverage_report,
    iv_deltas,
    load_cells,
)


def _leg(role: str, strike: float, iv: float | None, **extra) -> dict:
    return {"role": role, "type": "call" if role != "otm_put" else "put",
            "strike": strike, "bid": 1.0, "ask": 1.2, "close": 1.1, "mid": 1.1,
            "iv": iv, "volume": extra.pop("volume", None), "count": 3, **extra}


def _cell(symbol: str, day: str, *, iv: float = 0.30, volume=None, oi=False,
          far=False, expiration: str = "2019-04-18") -> dict:
    legs = [
        _leg("atm", 100.0, iv, volume=volume, **({"open_interest": 100} if oi else {})),
        _leg("otm_put", 90.0, iv + 0.05, volume=volume),
    ]
    cell = {
        "symbol": symbol, "decision_date": day, "spot_at_decision": 100.0,
        "target_expiration": expiration, "days_to_expiry": 48,
        "contracts": legs, "skew_put_atm": 0.05, "skew_put_call": None,
    }
    if far:
        cell["far"] = {"target_expiration": "2019-05-17", "days_to_expiry": 77,
                       "contracts": [_leg("atm", 100.0, iv - 0.02)],
                       "skew_put_atm": None, "skew_put_call": None}
    return cell


def test_load_cells_skips_torn_and_keyless_lines(tmp_path: Path):
    path = tmp_path / "samples.jsonl"
    path.write_text(
        json.dumps(_cell("AAA", "2019-03-01")) + "\n"
        + "{ torn line\n"
        + json.dumps({"decision_date": "2019-03-01"}) + "\n"  # no symbol
    )
    cells = load_cells(path)
    assert [c["symbol"] for c in cells] == ["AAA"]
    assert load_cells(tmp_path / "absent.jsonl") == []


def test_iv_deltas_matches_same_contract_only():
    v1 = [_cell("AAA", "2019-03-01", iv=0.300)]
    v2 = [
        _cell("AAA", "2019-03-01", iv=0.302),                        # same contract: |d|=0.002
        _cell("AAA", "2019-04-01", iv=0.500),                        # different date: no match
        _cell("AAA", "2019-03-01", iv=0.900, expiration="2019-05-17"),  # different contract
    ]
    deltas = iv_deltas(v2, v1)
    # atm AND otm_put both match on the 03-01 cell (put ivs 0.352 vs 0.350).
    assert sorted(round(d, 6) for d in deltas) == [0.002, 0.002]


def test_coverage_report_rates_and_agreement():
    v1 = [_cell("AAA", "2019-03-01", iv=0.300), _cell("BBB", "2019-03-01", iv=0.40)]
    v2 = [
        _cell("AAA", "2019-03-01", iv=0.302, volume=50, oi=True, far=True),
        _cell("BBB", "2019-03-01", iv=0.400),  # no volume/oi/far
    ]
    report = coverage_report(v2, v1)
    assert report["cells_v2"] == 2 and report["cells_v1"] == 2
    assert report["leg_volume_rate"] == 0.5   # 1 of 2 cells has a volume-bearing near leg
    assert report["oi_leg_rate"] == 0.25      # 1 of 4 near legs carries open_interest
    assert report["far_rate"] == 0.5
    assert report["iv_overlap_legs"] == 4
    assert report["iv_median_abs_delta"] == pytest.approx(0.001)  # {.002,.002,0,0} -> median .001
    assert report["iv_red_flag"] is False


def test_coverage_report_red_flags_large_iv_drift():
    v1 = [_cell("AAA", "2019-03-01", iv=0.30)]
    v2 = [_cell("AAA", "2019-03-01", iv=0.30 + 2 * IV_DRIFT_RED_FLAG)]
    report = coverage_report(v2, v1)
    assert report["iv_red_flag"] is True


def test_coverage_report_empty_inputs_yield_none_rates():
    report = coverage_report([], [])
    assert report["cells_v2"] == 0
    assert report["leg_volume_rate"] is None
    assert report["oi_leg_rate"] is None
    assert report["far_rate"] is None
    assert report["iv_median_abs_delta"] is None
    assert report["iv_red_flag"] is False
```

Run: `uv run pytest tests/test_options_coverage.py -q` → ModuleNotFoundError.

- [ ] **Step 4: Implement the coverage module + CLI**

Create `src/trading/research/options_coverage.py`:

```python
"""Coverage/agreement report: options gather v2 vs the v1 backup (spec
2026-07-09-options-gather-v2-design.md section 5).

Answers, per pool, before the regathered data is trusted: how many cells; what
fraction carry leg volume (large-cap must jump from 0% to ~mid-cap levels),
open interest, greeks, and a far block; and whether IV agrees with v1 on the
SAME contract/date (large drift is a red flag to investigate, not accept).
Pure functions over parsed cells -- no network, no pandas needed.
"""

from __future__ import annotations

import json
from pathlib import Path
from statistics import median

# Median |iv_v2 - iv_v1| on the SAME (symbol, date, role, strike, expiration)
# above one vol point means the two gathers priced the same contract
# differently -- investigate (quote-window drift, spot mismatch) before use.
IV_DRIFT_RED_FLAG = 0.01


def load_cells(path: Path) -> list[dict]:
    """Tolerant samples.jsonl reader (the load_existing_keys discipline):
    torn/keyless lines are skipped, an absent file is just empty."""
    cells: list[dict] = []
    if not path.exists():
        return cells
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if len(line) == 0:
                continue
            try:
                cell = json.loads(line)
            except ValueError:
                continue
            if isinstance(cell, dict) and cell.get("symbol") and cell.get("decision_date"):
                cells.append(cell)
    return cells


def _near_legs(cells: list[dict]) -> list[dict]:
    return [c for cell in cells for c in cell.get("contracts", [])]


def _leg_rate(cells: list[dict], key: str) -> float | None:
    """Fraction of near legs carrying ``key`` with a non-None value; None when
    there are no legs at all (empty pool -> no rate, not a fake 0)."""
    legs = _near_legs(cells)
    if len(legs) == 0:
        return None
    return sum(1 for c in legs if c.get(key) is not None) / len(legs)


def _cell_rate(cells: list[dict], predicate) -> float | None:
    if len(cells) == 0:
        return None
    return sum(1 for cell in cells if predicate(cell)) / len(cells)


def iv_deltas(v2_cells: list[dict], v1_cells: list[dict]) -> list[float]:
    """|iv_v2 - iv_v1| per overlapping near leg: the SAME
    (symbol, decision_date, role, strike, target_expiration) where both
    gathers inverted an IV. A leg the v2 walk resolved to a different strike
    or expiration is a different contract and deliberately does not match."""

    def index(cells: list[dict]) -> dict[tuple, float]:
        out: dict[tuple, float] = {}
        for cell in cells:
            for c in cell.get("contracts", []):
                iv = c.get("iv")
                if iv is None:
                    continue
                key = (
                    cell.get("symbol"),
                    cell.get("decision_date"),
                    c.get("role"),
                    c.get("strike"),
                    cell.get("target_expiration"),
                )
                out[key] = float(iv)
        return out

    v1 = index(v1_cells)
    v2 = index(v2_cells)
    return [abs(v2[key] - v1[key]) for key in v2.keys() & v1.keys()]


def coverage_report(v2_cells: list[dict], v1_cells: list[dict]) -> dict:
    """The spec-section-5 verification numbers, JSON-serializable."""
    deltas = iv_deltas(v2_cells, v1_cells)
    iv_median = median(deltas) if len(deltas) > 0 else None
    return {
        "cells_v2": len(v2_cells),
        "cells_v1": len(v1_cells),
        "leg_volume_rate": _cell_rate(
            v2_cells,
            lambda cell: any(
                c.get("volume") is not None for c in cell.get("contracts", [])
            ),
        ),
        "oi_leg_rate": _leg_rate(v2_cells, "open_interest"),
        "greeks_leg_rate": _leg_rate(v2_cells, "delta"),
        "far_rate": _cell_rate(v2_cells, lambda cell: "far" in cell),
        "iv_overlap_legs": len(deltas),
        "iv_median_abs_delta": iv_median,
        "iv_red_flag": iv_median is not None and iv_median > IV_DRIFT_RED_FLAG,
    }
```

Create `scripts/options_coverage_report.py`:

```python
"""CLI: print the options gather v2-vs-v1 coverage/agreement report as JSON.

Run once per pool after the re-gather (spec section 5):

    uv run python scripts/options_coverage_report.py \
        --v2 data/options-iv/samples.jsonl --v1 data/options-iv/samples.v1.jsonl
    uv run python scripts/options_coverage_report.py \
        --v2 data/options-iv/samples-midcap.jsonl --v1 data/options-iv/samples-midcap.v1.jsonl

iv_red_flag=true (median |iv_v2 - iv_v1| > 0.01 on the same contract/date)
means investigate before trusting the data -- do not proceed to signals.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from trading.research.options_coverage import coverage_report, load_cells


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v2", type=Path, required=True, help="regathered samples.jsonl")
    parser.add_argument("--v1", type=Path, required=True, help="the .v1 backup baseline")
    args = parser.parse_args(argv)
    report = coverage_report(load_cells(args.v2), load_cells(args.v1))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_options_coverage.py tests/test_options_gather.py -q`
Expected: all pass.

- [ ] **Step 6: Full suite + lint, then commit**

Run: `uv run pytest -q 2>&1 | tail -5` and `uv run ruff check src tests scripts`

```bash
git add src/trading/research/options_gather.py src/trading/research/options_coverage.py \
        scripts/gather_options_iv.py scripts/options_coverage_report.py \
        tests/test_options_gather.py tests/test_options_coverage.py
git commit -m "Add fresh-gather v1 backup, enriched-schema resume proof, and v2 coverage report [AI]"
```

---

### Task 6: Ops runbook — the re-gather on the mini (ORCHESTRATOR EXECUTES; NOT CODE)

No implementation subagent for this task: the orchestrator runs these commands. Prereqs: Tasks 1–5 merged to master AND on `origin/master` (pushes are Travis-manual due to the permission classifier — ask him to `git push origin master` if needed), terminal jar staged at `mini:~/thetadata/202607071.jar`, ThetaData key in mini `~/.config/trading/config.toml`.

- [ ] **R1: Deploy the code to the mini**

```bash
ssh mac-m1 'cd ~/trading && git pull && ~/.local/bin/uv sync'
```

- [ ] **R2: Port cleanup + terminal launch**

ONE terminal per account; a second gets "Invalid session ID". Clear both ports first:

```bash
ssh mac-m1 'lsof -ti :25503 :25520 | xargs kill 2>/dev/null; sleep 2; lsof -i :25503 -i :25520 || echo PORTS-CLEAR'
# Extract the key (41 chars WITH underscores -- never strip them):
KEY=$(ssh mac-m1 'grep theta_data_api_key ~/.config/trading/config.toml | cut -d'"'"'"' -f2')
# Launch from ~/thetadata to keep terminal writes writable (launching elsewhere tries to write /Users/config.toml and dies):
ssh mac-m1 'cd ~/thetadata && nohup /opt/homebrew/opt/openjdk/bin/java -jar 202607071.jar --api-key "'"$KEY"'" > ~/thetadata/terminal.log 2>&1 & sleep 20; tail -5 ~/thetadata/terminal.log'
```

- [ ] **R3: Readiness check**

```bash
ssh mac-m1 'curl -s "http://127.0.0.1:25503/v3/option/list/expirations?symbol=AAPL&format=json" | head -c 200'
```
Expected: a JSON `response` array of expirations. If connection refused, wait and re-tail the terminal log (Java startup + login can take ~30s).

- [ ] **R4: Endpoint discovery** — run Task 1 Step 1's curls now (if not already done) and hand the verdict to the Task 1 implementer. STOP-and-report if no OI source exists.

- [ ] **R5: The ThetaData delisted-equity test (spec §4 — 5 minutes while the terminal is up)**

```bash
# XLNX delisted 2022-02; probe a PRE-delisting month:
ssh mac-m1 'curl -s -w "\nHTTP %{http_code}\n" "http://127.0.0.1:25503/v3/stock/history/eod?symbol=XLNX&start_date=2021-11-01&end_date=2021-11-30&format=json"' | head -c 600
# Control 1: AAPL, SAME 2021 window (separates entitlement-depth from survivorship):
ssh mac-m1 'curl -s -w "\nHTTP %{http_code}\n" "http://127.0.0.1:25503/v3/stock/history/eod?symbol=AAPL&start_date=2021-11-01&end_date=2021-11-30&format=json"' | head -c 600
# Control 2: AAPL, RECENT window (proves the stock endpoint works at all):
ssh mac-m1 'curl -s -w "\nHTTP %{http_code}\n" "http://127.0.0.1:25503/v3/stock/history/eod?symbol=AAPL&start_date=2026-06-01&end_date=2026-06-30&format=json"' | head -c 600
```

Interpretation (record the verdict in `docs/options-data-vendors.md`, new subsection "ThetaData stock history — delisted-bar test (2026-07)" under the existing delisted-coverage section; note the stock entitlement on this account is the free universal ~1-yr EOD, NOT a paid stock tier):
- **XLNX 2021 returns bars** → delisted stock history IS served → the augment case strengthens (record: "delisted history served; stock feed usable as microstructure augment; Tiingo stays system of record").
- **XLNX empty/472 but AAPL 2021 returns bars** → live-universe-only / no delisted depth → record: "stock feed is live-universe-only at our entitlement; microstructure add-on at best".
- **Both 2021 queries fail but AAPL 2026-06 works** → the free stock window (~1yr) can't reach 2021 → the test is INCONCLUSIVE-BY-ENTITLEMENT; record exactly that (do not over-claim either way).
- Either way: Tiingo remains the equity system of record.

- [ ] **R6: Pin the pools, verify Tiingo cache, and verify the raw-spot cache**

The re-gather must hit the SAME pools (alphasearch trials are keyed by universe NAME). Re-running `build_universe` with a longer end-month could re-rank the top-N and shift membership — so pin symbols to exactly what v1 gathered. First, confirm the mini's ADJUSTED Tiingo cache extends through 2026-07 (it drives the decision calendar) — else 2026 months silently produce no cells:

```bash
ssh mac-m1 'cd ~/trading && python3 -c "from trading.research.factors import TIINGO_CACHE; import polars as pl; df=pl.read_parquet(TIINGO_CACHE); print(df[\"date\"].max())"'
```

Expected: 2026-07-xx or later. If earlier, backfill before continuing:

```bash
ssh mac-m1 'cd ~/trading && ~/.local/bin/uv run python scripts/backfill_tiingo_prices.py --end-date 2026-07-31'
```

Now pin symbols to exactly what v1 gathered:

```bash
ssh mac-m1 'cd ~/trading && python3 -c "
import json
for f in (\"data/options-iv/samples.jsonl\", \"data/options-iv/samples-midcap.jsonl\"):
    syms = sorted({json.loads(l)[\"symbol\"] for l in open(f) if l.strip()})
    print(f, len(syms))
    open(f.replace(\".jsonl\", \".symbols.txt\"), \"w\").write(\" \".join(syms))
"'
# Raw (unadjusted) spot cache must cover every pinned name:
ssh mac-m1 'cd ~/trading && ls data/options-iv/underlying_raw | wc -l'
# If any pinned symbol lacks a raw parquet, backfill it (incremental, skips existing):
ssh mac-m1 'cd ~/trading && ~/.local/bin/uv run python scripts/backfill_options_underlying_raw.py --limit-symbols $(cat data/options-iv/samples.symbols.txt) --end-date 2026-07-31'
ssh mac-m1 'cd ~/trading && ~/.local/bin/uv run python scripts/backfill_options_underlying_raw.py --limit-symbols $(cat data/options-iv/samples-midcap.symbols.txt) --end-date 2026-07-31'
```

- [ ] **R7: Launch the re-gather (both pools, SEQUENTIAL — 4-request terminal cap means never two gathers at once)**

`--end-month 2026-07` implements spec §3's "2019-01..latest".

```bash
ssh mac-m1 'cd ~/trading && mkdir -p state/options-iv && nohup sh -c "\
  ~/.local/bin/uv run python scripts/gather_options_iv.py --fresh \
    --end-month 2026-07 \
    --limit-symbols \$(cat data/options-iv/samples.symbols.txt) \
    --out data/options-iv/samples.jsonl && \
  ~/.local/bin/uv run python scripts/gather_options_iv.py --fresh \
    --end-month 2026-07 --indices sp400 \
    --limit-symbols \$(cat data/options-iv/samples-midcap.symbols.txt) \
    --out data/options-iv/samples-midcap.jsonl" \
  > state/options-iv/gather-v2.log 2>&1 & echo "launched: $!"'
```

Notes: `--fresh` renames each file to its `.v1.jsonl` backup first (and REFUSES if a backup already exists — if a crashed v2 run must resume, re-run WITHOUT `--fresh`: the resume skip does the rest). `--limit-symbols` bypasses universe ranking entirely, which is what pins the pools.

- [ ] **R8: Monitor + resume on crash**

```bash
ssh mac-m1 'tail -5 ~/trading/state/options-iv/gather-v2.log'
ssh mac-m1 'grep progress ~/trading/state/options-iv/gather-v2.log | tail -3'
```

The progress lines print `done/total ... elapsed`; cells/s = done ÷ elapsed. v1 ran ~3.4 cells/s; v2 adds the far block (~2× EOD requests) plus OI (and greeks if served) per resolved leg — expect roughly 0.8–1.5 cells/s, i.e. ~2–4h per pool; budget an overnight for both. Check the log within ~10 minutes of launch — if errors are climbing or throughput is well under 0.5 cells/s, kill the terminal, fix the issue, and resume.

**On crash:** A crash during pool 1 means pool 2 never backed up — a blanket resume WITHOUT --fresh would silently append 2026 months onto the v1 midcap file and never re-gather pool 2. Resume PER POOL with conditional logic:

```bash
ssh mac-m1 'pgrep -fl gather_options_iv'
# Must be empty; nohup survives ssh drops — a second launch double-writes.
# For each pool, resume WITHOUT --fresh iff that pool's .v1.jsonl exists (already backed up); otherwise WITH --fresh:
ssh mac-m1 'cd ~/trading && test -f data/options-iv/samples.v1.jsonl && \
  nohup ~/.local/bin/uv run python scripts/gather_options_iv.py --end-month 2026-07 \
    --limit-symbols $(cat data/options-iv/samples.symbols.txt) \
    --out data/options-iv/samples.jsonl >> state/options-iv/gather-v2.log 2>&1 & \
  echo "resumed pool 1 (backup exists)" || \
  nohup ~/.local/bin/uv run python scripts/gather_options_iv.py --fresh --end-month 2026-07 \
    --limit-symbols $(cat data/options-iv/samples.symbols.txt) \
    --out data/options-iv/samples.jsonl >> state/options-iv/gather-v2.log 2>&1 & \
  echo "relaunched pool 1 (no backup)"'
# Repeat for pool 2 (midcap) once pool 1 completes.
```

- [ ] **R9: Rsync back + verification**

```bash
# Pull v2 files AND the v1 backups (the coverage report needs both):
rsync -av 'mac-m1:~/trading/data/options-iv/samples*.jsonl' data/options-iv/
# Coverage report, once per pool:
uv run python scripts/options_coverage_report.py \
  --v2 data/options-iv/samples.jsonl --v1 data/options-iv/samples.v1.jsonl \
  2>&1 | tee /tmp/claude-coverage-largecap.log
uv run python scripts/options_coverage_report.py \
  --v2 data/options-iv/samples-midcap.jsonl --v1 data/options-iv/samples-midcap.v1.jsonl \
  2>&1 | tee /tmp/claude-coverage-midcap.log
```

Acceptance reading (spec §5). Hard preconditions: `cells_v1 > 0` AND `iv_overlap_legs` in the thousands (< 100 overlaps = broken matching). Additionally: `iv_red_flag=false WITH iv_overlap_legs=0` is a FAIL, never a pass (missing backup).

- `cells_v2` ≳ `cells_v1` per pool (v2 adds 2026 months; a large DROP means something broke).
- Large-cap `leg_volume_rate` must jump from ~0 to ≈ the mid-cap pool's rate (the whole point of re-gathering the large caps).
- `oi_leg_rate` substantially > 0 (else the OI endpoint verdict was wrong — investigate before anything downstream).
- `greeks_leg_rate` follows oi_leg_rate (greeks served when available).
- `far_rate` high but < 1.0 is expected (some months have no monthly in 55..90).
- `iv_overlap_legs` in the thousands (confirms both gathers priced overlapping contracts).
- `iv_median_abs_delta` typically < 0.001–0.005 (same contract should price identically).
- `iv_red_flag` must be false; if true, diff a few matched legs by hand before trusting ANYTHING (same contract + date should produce near-identical mids).
- Spot-checks: (1) one liquid name's ATM `open_interest` vs public data — order of magnitude (10³–10⁵ for AAPL near-ATM) and the build-then-drop shape across an expiry cycle:
  `python3 -c "import json; [print(c['decision_date'], k['strike'], k.get('open_interest')) for c in map(json.loads, open('data/options-iv/samples.jsonl')) if c['symbol']=='AAPL' for k in c['contracts'] if k['role']=='atm']" | head -24`
  (2) strikes still RAW dollars snapped near `spot_at_decision` (AAPL 2019 rows ~165, not ~40); (3) every `far.days_to_expiry` within 55..90:
  `python3 -c "import json; ds=[c['far']['days_to_expiry'] for c in map(json.loads, open('data/options-iv/samples.jsonl')) if 'far' in c]; print(min(ds), max(ds), len(ds))"`
- (4) Verify `volume_cell_rate` and `volume_leg_rate` reported (not old `leg_volume_rate` name) and that large-cap's `volume_cell_rate` jumped to mid-cap levels.

- [ ] **R10: Record + close out**

- Append the coverage numbers (both pools) and the delisted-test verdict to `docs/experiments.md` as a short data-note (verification numbers go in docs; nothing under `data/` is committed — spec §3).
- Commit the `docs/options-data-vendors.md` + `docs/experiments.md` edits: `git commit -m "Record options gather v2 verification + ThetaData delisted-bar verdict [AI]"`.
- Leave the terminal running only if more pulls are imminent; otherwise `ssh mac-m1 'lsof -ti :25503 :25520 | xargs kill'`.
- Downstream reminder (recorded, not actioned here): alphasearch universes are keyed by NAME over regenerated data — the Piece 3 cache-drift guard is EXPECTED to refuse stale-baseline battery re-rolls afterward; that is it working as designed.

---

### OPS CORRECTION (2026-07-09, learned live)

The mid-cap gather requires BOTH `--cache-dir data/equities-midcap-tiingo`
AND mid-cap raw closes present in `data/options-iv/underlying_raw/` (run
`scripts/backfill_options_underlying_raw.py --limit-symbols <midcap pool>
--cache-dir data/equities-midcap-tiingo --indices sp400` first). The original
run-book mid-cap command omitted both; the resulting 91-cell enumeration
(vs ~12.7k expected) was caught by the cell-count check, and the pristine
`samples-midcap.v1.jsonl` backup made the re-run free. Run-book lesson:
enumerate per-pool DATA prerequisites (adjusted cache, raw closes), not just
CLI flags.

## Self-Review (run after writing, fixed inline)

1. **Spec coverage:** §2 per-leg OI/greeks → Tasks 1–2; §2 far block → Task 3; §2 large-cap leg volume → re-gather (R7) + coverage assertion (R9); §3 ops/paths/backup/log → Task 5 + R1–R8 (pools pinned via `--limit-symbols`, same output paths, `.v1` backups, `state/options-iv/gather-v2.log`); §4 delisted test → R5 (with the entitlement-window caveat); §5 verification → Task 5 coverage module + R9 numbers + unit tests across Tasks 1–5; §6 error handling → degrade-to-absent (Task 2), far-drop-only (Task 3), resume under enriched schema (Task 5). Out-of-scope items (signal registration, sweeps) correctly absent.
2. **Placeholders:** none — every step carries complete code or exact commands. The one deliberately unfixed value pair (`OI_PATH`/`GREEKS_PATH`) is not a placeholder but a live-verification requirement with an explicit discovery procedure and STOP rule (Task 1 Step 1).
3. **Type consistency:** `_resolve_legs` returns `dict[str, tuple[float, dict]]`, consumed by `build_cell`/`build_far_block` which accept `dict[str, tuple[float, dict | None]]` (wider — fine); `_contracts_and_skews` returns `tuple[list[dict], float | None, float | None]` and both callers unpack three values; `backup_v1` naming/signature identical in Task 5 code, tests, and script import; FakeClient's `history_calls` keeps the v1 4-tuple so pre-existing assertions hold; `include_greeks` threads gather_cell → `_resolve_legs` → `_enrich_bar` and reaches `run_gather` via the existing `**cell_kwargs`.
