# R3 — Down-cap / Illiquid Universe Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a survivorship-free, point-in-time down-cap ($50M–$2B) equities universe with a frozen GO/NO-GO verification gate (Phase A), then sweep the pre-registered `momentum_v1` long-only hypothesis (plus the exploratory battery) against it under the R1 cost-charged-vs-SPY gate (Phase B), spending no holdout.

**Architecture:** Five new focused modules under `src/trading/venues/universes/` carry the build: a Tiingo `supported_tickers` roster fetcher/parser (A1), a raw-price PIT band selector reusing the R1 Corwin-Schultz estimator (A2), a non-index bar-backfill driver over a fresh cache dir (A3), a dynamic PIT band-membership builder that emits both the science artifact and an audit-diagnostics artifact (A4), and a frozen verification-gate computation + report (A5). Phase B threads **dynamic per-date band membership** into the existing panel core: because a name's band membership changes month to month (its cap and liquidity cross the bounds), and `UniverseSpec.symbols`/`build_panel` model only a *static* symbol set, we express the dynamic membership as a precomputed `(band, symbol, start, end)` interval CSV — exactly mirroring the existing `equities_membership.csv` `(index, symbol, start, end)` precedent that `EquitiesAdapter.universe`/`segments._window_members` already consume. The static `UniverseSpec.symbols` holds the *ever-in-band union* (so bars load for every name), and a new `PanelData.membership` field drives a per-`as_of` filter inside `PanelView.symbols`, so every registry signal (including `momentum_v1`) automatically ranks only the names in-band at each decision date. This is the least-surface correct hook: the filter lives in the one place the no-look-ahead guarantee is already structural (the view), defaults to empty (existing universes bit-unchanged), and needs no change to any signal.

**Tech Stack:** Python 3.12, pandas, pytest, uv. Run tests with `.venv/bin/python -m pytest`. Data: Tiingo daily bars (existing `EquitiesAdapter`/`OhlcvCache`), SEC companyfacts shares (existing `FundamentalsStore`). No new third-party dependencies.

## Global Constraints

Every task's requirements implicitly include this section. Values copied verbatim from the spec (`docs/superpowers/specs/2026-07-10-downcap-universe-design.md`) and are FROZEN — a degenerate value is a NO-GO finding recorded in `experiments.md` and resolved by a written amendment, never a silent re-tune.

- **Cap band:** `[$50M, $2B]` — `50_000_000.0 .. 2_000_000_000.0`.
- **Market cap basis:** RAW/unadjusted share price (`close_raw`), NEVER the split/dividend-adjusted `close`. Cap = `shares_outstanding(latest companyfacts row FILED ≤ D) × close_raw(≤ D)`.
- **Spread screen:** trailing-63-session Corwin-Schultz effective spread ≤ `2.0%` (`0.02`). (Reuses the R1 estimator `trailing_effective_spread`; the R1 `effective_spread` helper uses a 21-session window — this universe uses the 63-session window the spec names, clamped by the same `SPREAD_FLOOR`/`SPREAD_CAP`.)
- **Depth screen:** trailing-63-session **median** dollar-volume (`close × volume`) ≥ `$50,000/day` (`50_000.0`).
- **Missing-shares handling:** a candidate with NO companyfacts shares FILED ≤ D is **excluded at D** (fail-closed, never guess a cap). The exclusion is measured and gated, not hidden.
- **Structural roster filters (no performance-dependent selection):** `assetType == "Stock"`; `priceCurrency == "USD"`; `exchange ∈ {NYSE, NASDAQ, NYSE ARCA, NYSE MKT, AMEX}` — the exact kept/dropped exchange strings are determined at build time from the file's actual values and recorded in the roster build log. A name is a candidate as-of D iff `D ∈ [startDate, endDate]` (empty endDate = still listed); delisted names are first-class.
- **Discovery window:** `2019-01-01..2023-12-31` (must equal `trading.alphasearch.sweep.DISCOVERY_WINDOW`).
- **Holdout:** `2024+` RESERVED, NEVER spent in this plan.
- **Multiple-testing:** BH-FDR `q = 0.10` across the whole journal; DSR advisory. (Reused unchanged from the existing sweep.)
- **Three universes (frozen `UniverseSpec` names):** `downcap` = full band `$50M–$2B`; `downcap:small` = `$300M–$2B`; `downcap:micro` = `$50M–$300M`. The two sub-bands partition the full band by the same cap rule; the tradeability screens apply identically to all three.
- **Fresh cache:** bars into `data/equities-downcap-tiingo/` with its own `.source = tiingo` marker (never mixed with `data/equities-tiingo/`).
- **companyfacts throttle:** `0.11s`/request (SEC 10 req/s ceiling; delisted filers covered — they filed 10-Ks while public).
- **GO/NO-GO thresholds (frozen, decided BEFORE any sweep):**
  - Survivorship present: delisted names ≥ `15%` of in-band candidate-months.
  - Shares-coverage: ≥ `70%` of band-eligible (tradeable) candidate-months have PIT shares. Below `70%` → NO-GO on the market-cap band → automatic (developer pre-approved) dollar-volume-only fallback (tradeability screens without the cap bound), recorded as a written prospective amendment.
  - Spread realism: report CS effective-spread distribution (median, IQR, % ≤ 2%); the 2% screen must be a real filter, not a no-op or a near-total cull.
  - Breadth: ≥ `15` tradeable names in EVERY month of the discovery window, for each of the three universes; a universe with any sub-15 month is dropped from the sweep (recorded, not silently skipped). ≥ 50/month is the comfort target.
- **Sweep mechanics (reused unchanged):** quintiles (tercile < 50 names, skip < 15), equal weight, monthly rebalance, rank-exit; `momentum_v1` long-only top-quintile is the pre-registered PRIMARY; the full registry battery runs alongside (exploratory, BH-counted) to the extent stores support it; options and insider signals are refused (no such stores for the band). The R1 `--long-only` cost-charged-vs-SPY leaderboard is the read; the four-factor L/S alpha remains the mandatory diagnostic.

---

## File Structure

**New modules** (`src/trading/venues/universes/` — the existing home of roster/membership data like `equities_membership.csv`, `sic_map.csv`):
- `downcap_roster.py` — Tiingo `supported_tickers.zip` fetch + parse + structural roster filter (A1).
- `downcap_band.py` — raw-price PIT market cap + the three-screen band selector (A2).
- `downcap_backfill.py` — non-index bar-backfill driver over a roster (A3).
- `downcap_membership.py` — the dynamic PIT band-membership + diagnostics builder, the `(band,symbol,start,end)` loader, and the three `downcap_universes` registrations (A4, B2).
- `downcap_verify.py` — the frozen GO/NO-GO computation + human report + fallback amendment (A5).

**New scripts** (thin CLIs, mirror `scripts/backfill_bars.py`):
- `scripts/backfill_downcap_bars.py` (A3), `scripts/build_downcap_membership.py` (A4), `scripts/downcap_verify.py` (A5).

**Modified** (Phase B core plumbing — all changes default-off so existing universes are bit-unchanged):
- `src/trading/alphasearch/panel.py` — `PanelData.membership` field; `PanelView.symbols` per-`as_of` filter; `build_panel(..., membership=...)` param (B1).
- `src/trading/alphasearch/sweep.py` — `UniverseSpec.membership_intervals`/`.bands` fields; `build_universe_panel` threads them (B1).
- `src/trading/cli.py` — `--downcap` flag on `alphasearch sweep`/`leaderboard` (B2).

**New tests:** `tests/test_downcap_roster.py`, `tests/test_downcap_band.py`, `tests/test_downcap_backfill.py`, `tests/test_downcap_membership.py`, `tests/test_downcap_verify.py`, `tests/test_downcap_universes.py`; plus additions to `tests/test_alphasearch_panel.py`.

**New docs:** section in `docs/experiments.md`; entries in `docs/glossary.md` (Docs task).

---

## Task A1: Tiingo `supported_tickers` fetch + parse + structural roster

**Files:**
- Create: `src/trading/venues/universes/downcap_roster.py`
- Test: `tests/test_downcap_roster.py`

**Interfaces:**
- Consumes: nothing (leaf module; stdlib `zipfile`/`urllib` + pandas only).
- Produces:
  - `SUPPORTED_TICKERS_URL: str`
  - `FROZEN_EXCHANGES: frozenset[str]`
  - `def _download_zip(url: str, dest: Path) -> None` (network seam, monkeypatched in tests)
  - `def fetch_supported_tickers(dest: Path, *, url: str = SUPPORTED_TICKERS_URL) -> Path`
  - `def parse_supported_tickers(zip_path: Path) -> pd.DataFrame` (columns `ticker, exchange, assetType, priceCurrency, startDate, endDate`, all `str`, NaN→"")
  - `def structural_roster(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]` → `(roster, exchange_report)` where `exchange_report = {"kept": {exch: count}, "dropped": {exch: count}}`
  - `def candidates_at(roster: pd.DataFrame, d: datetime.date) -> set[str]`
  - `def delisted_symbols(roster: pd.DataFrame) -> set[str]` (endDate non-empty → eventually delisted)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_downcap_roster.py
import datetime
import io
import zipfile
from pathlib import Path

import pandas as pd
import pytest

from trading.venues.universes import downcap_roster as dr


def _zip_with(csv_text: str, dest: Path) -> Path:
    with zipfile.ZipFile(dest, "w") as zf:
        zf.writestr("supported_tickers.csv", csv_text)
    return dest


CSV = (
    "ticker,exchange,assetType,priceCurrency,startDate,endDate\n"
    "AAPL,NASDAQ,Stock,USD,1990-01-01,\n"                 # keep: live
    "OLDCO,NYSE,Stock,USD,2001-01-01,2015-06-30\n"        # keep: delisted
    "ARCAX,NYSE ARCA,Stock,USD,2010-01-01,\n"             # keep: NYSE ARCA
    "MKTCO,NYSE MKT,Stock,USD,2012-01-01,\n"              # keep: NYSE MKT
    "SPY,NYSE ARCA,ETF,USD,1993-01-01,\n"                 # drop: assetType
    "FOREIGN,NASDAQ,Stock,CAD,2000-01-01,\n"              # drop: currency
    "PINKY,OTC,Stock,USD,2005-01-01,\n"                   # drop: exchange
)


def test_parse_reads_all_columns_as_strings(tmp_path):
    zp = _zip_with(CSV, tmp_path / "st.zip")
    df = dr.parse_supported_tickers(zp)
    assert list(df.columns) == [
        "ticker", "exchange", "assetType", "priceCurrency", "startDate", "endDate"
    ]
    assert df.loc[df["ticker"] == "AAPL", "endDate"].iloc[0] == ""  # NaN normalized to ""


def test_structural_roster_applies_frozen_filters(tmp_path):
    zp = _zip_with(CSV, tmp_path / "st.zip")
    raw = dr.parse_supported_tickers(zp)
    roster, report = dr.structural_roster(raw)
    assert set(roster["ticker"]) == {"AAPL", "OLDCO", "ARCAX", "MKTCO"}
    assert report["dropped"] == {"OTC": 1}  # only the venue-dropped row lands here
    assert report["kept"]["NYSE ARCA"] == 1


def test_candidates_at_uses_listing_interval(tmp_path):
    zp = _zip_with(CSV, tmp_path / "st.zip")
    roster, _ = dr.structural_roster(dr.parse_supported_tickers(zp))
    # OLDCO delisted 2015-06-30: a candidate before, not after.
    assert "OLDCO" in dr.candidates_at(roster, datetime.date(2015, 1, 1))
    assert "OLDCO" not in dr.candidates_at(roster, datetime.date(2016, 1, 1))
    assert "AAPL" in dr.candidates_at(roster, datetime.date(2020, 1, 1))


def test_delisted_symbols_flags_nonempty_enddate(tmp_path):
    zp = _zip_with(CSV, tmp_path / "st.zip")
    roster, _ = dr.structural_roster(dr.parse_supported_tickers(zp))
    assert dr.delisted_symbols(roster) == {"OLDCO"}


def test_fetch_supported_tickers_uses_download_seam(tmp_path, monkeypatch):
    def fake_download(url, dest):
        _zip_with(CSV, dest)

    monkeypatch.setattr(dr, "_download_zip", fake_download)
    out = dr.fetch_supported_tickers(tmp_path / "st.zip")
    assert out.exists()
    df = dr.parse_supported_tickers(out)
    assert "AAPL" in set(df["ticker"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_downcap_roster.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'trading.venues.universes.downcap_roster'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/trading/venues/universes/downcap_roster.py
"""Tiingo supported_tickers roster: the survivorship-free US-common-stock
candidate list (R3 spec section 2). Fetched from Tiingo's downloadable ZIP
(delisted names included), parsed, and STRUCTURALLY filtered -- no
performance-dependent selection. Leaf module: stdlib + pandas only."""

from __future__ import annotations

import datetime
import io
import zipfile
from pathlib import Path

import pandas as pd

# The full historical ticker list (delisted included). NOT the /utilities/search
# endpoint -- that is an autocomplete API; this ZIP is the roster of record.
SUPPORTED_TICKERS_URL = "https://apimedia.tiingo.com/docs/tiingo/daily/supported_tickers.zip"

# The frozen major-US-common-stock venue set (spec section 2). OTC/pink venues
# are excluded (untradeable, fraud-prone, outside the account's reach). The
# EXACT kept/dropped strings observed in the real file are recorded in the
# build log by structural_roster's exchange_report.
FROZEN_EXCHANGES = frozenset({"NYSE", "NASDAQ", "NYSE ARCA", "NYSE MKT", "AMEX"})

_COLUMNS = ["ticker", "exchange", "assetType", "priceCurrency", "startDate", "endDate"]


def _download_zip(url: str, dest: Path) -> None:
    """Network touchpoint, isolated for monkeypatching (tests never hit the
    network). Streams the ZIP to `dest`."""
    import urllib.request

    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=120) as resp:
        dest.write_bytes(resp.read())


def fetch_supported_tickers(dest: Path, *, url: str = SUPPORTED_TICKERS_URL) -> Path:
    """Download the supported_tickers ZIP to `dest`; return `dest`."""
    _download_zip(url, dest)
    return dest


def parse_supported_tickers(zip_path: Path) -> pd.DataFrame:
    """Unzip and read supported_tickers.csv into a typed frame (all str,
    NaN normalized to "")."""
    with zipfile.ZipFile(zip_path) as zf:
        name = next(n for n in zf.namelist() if n.endswith(".csv"))
        raw = zf.read(name)
    df = pd.read_csv(io.BytesIO(raw), dtype=str).fillna("")
    # Tiingo ships these exact headers; select+reorder so downstream is stable.
    return df[_COLUMNS].copy()


def structural_roster(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Apply the frozen structural filters (spec section 2). Returns the
    filtered roster AND an exchange_report (kept/dropped exchange-string
    counts) for the build log -- so the actual venue strings in the file are
    recorded, never assumed."""
    typed = raw[
        (raw["assetType"] == "Stock") & (raw["priceCurrency"] == "USD")
    ]
    kept = typed[typed["exchange"].isin(FROZEN_EXCHANGES)]
    dropped = typed[~typed["exchange"].isin(FROZEN_EXCHANGES)]
    report = {
        "kept": kept["exchange"].value_counts().to_dict(),
        "dropped": dropped["exchange"].value_counts().to_dict(),
    }
    return kept.reset_index(drop=True), report


def candidates_at(roster: pd.DataFrame, d: datetime.date) -> set[str]:
    """Tickers listed as-of `d`: startDate <= d <= endDate (empty endDate =
    still listed)."""
    iso = d.isoformat()
    active = roster[
        (roster["startDate"] <= iso)
        & ((roster["endDate"] == "") | (iso <= roster["endDate"]))
    ]
    return set(active["ticker"])


def delisted_symbols(roster: pd.DataFrame) -> set[str]:
    """Tickers with a non-empty endDate -- they left an exchange, so their
    presence is what makes the roster survivorship-free (spec section 4's
    survivorship metric numerator)."""
    return set(roster.loc[roster["endDate"] != "", "ticker"])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_downcap_roster.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add src/trading/venues/universes/downcap_roster.py tests/test_downcap_roster.py
git commit -m "feat: Tiingo supported_tickers survivorship-free roster [AI]"
```

---

## Task A2: Raw-price PIT market cap + three-screen band selector

**Files:**
- Create: `src/trading/venues/universes/downcap_band.py`
- Test: `tests/test_downcap_band.py`

**Interfaces:**
- Consumes: `trading.alphasearch.costs.trailing_effective_spread`, `SPREAD_FLOOR`, `SPREAD_CAP`; bar frames with `BAR_COLUMNS` (`trading.alphasearch.panel.BAR_COLUMNS` — includes `close`, `close_raw`, `high`, `low`, `volume`).
- Produces:
  - Constants `BAND_LO=50_000_000.0`, `SMALL_LO=300_000_000.0`, `BAND_HI=2_000_000_000.0`, `SPREAD_CAP_PCT=0.02`, `DV_FLOOR=50_000.0`, `DOWNCAP_TRAILING_WINDOW=63`
  - `def market_cap_raw(shares: float, close_raw: float) -> float`
  - `def band_of(market_cap: float) -> str | None` (`"micro"` | `"small"` | `None`)
  - `def downcap_effective_spread(bars: pd.DataFrame) -> float`
  - `def median_dollar_volume(bars: pd.DataFrame, window: int = DOWNCAP_TRAILING_WINDOW) -> float`
  - `@dataclass(frozen=True) class BandEval` fields `band: str | None, has_shares: bool, tradeable: bool, market_cap: float, spread: float, dollar_volume: float`
  - `def evaluate_band(bars: pd.DataFrame, shares: float, *, spread_cap=SPREAD_CAP_PCT, dv_floor=DV_FLOOR) -> BandEval`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_downcap_band.py
import math

import numpy as np
import pandas as pd
import pytest

from trading.alphasearch.panel import BAR_COLUMNS
from trading.venues.universes import downcap_band as db


def _bars(n=80, close_raw=10.0, close=10.0, high=10.05, low=9.95, volume=100_000.0):
    idx = pd.date_range("2019-01-01", periods=n, freq="B", tz="UTC")
    frame = pd.DataFrame(
        {
            "open": close, "high": high, "low": low, "close": close,
            "volume": volume, "div_cash": 0.0, "split_factor": 1.0,
            "close_raw": close_raw,
        },
        index=idx,
    )
    return frame[BAR_COLUMNS]


def test_market_cap_uses_raw_price_hand_computed():
    # 20M shares * $10 raw = $200M
    assert db.market_cap_raw(20_000_000.0, 10.0) == pytest.approx(200_000_000.0)


def test_band_of_boundaries():
    assert db.band_of(49_999_999.0) is None          # below $50M
    assert db.band_of(50_000_000.0) == "micro"       # inclusive lower
    assert db.band_of(299_999_999.0) == "micro"
    assert db.band_of(300_000_000.0) == "small"      # micro/small boundary
    assert db.band_of(2_000_000_000.0) == "small"    # inclusive upper
    assert db.band_of(2_000_000_001.0) is None       # above $2B


def test_median_dollar_volume_hand_computed():
    bars = _bars(close=10.0, volume=100_000.0)  # 10*100k = 1_000_000 every day
    assert db.median_dollar_volume(bars) == pytest.approx(1_000_000.0)


def test_evaluate_band_in_band_micro():
    bars = _bars(close_raw=10.0, close=10.0, volume=100_000.0)
    ev = db.evaluate_band(bars, shares=20_000_000.0)  # $200M -> micro
    assert ev.band == "micro"
    assert ev.has_shares is True and ev.tradeable is True
    assert ev.market_cap == pytest.approx(200_000_000.0)


def test_missing_shares_excluded_fail_closed():
    bars = _bars()
    ev = db.evaluate_band(bars, shares=math.nan)
    assert ev.has_shares is False
    assert ev.band is None                 # never guess a cap
    assert math.isnan(ev.market_cap)
    assert ev.tradeable is True            # still measured for the audit denom


def test_spread_screen_rejects_wide_names():
    # Very wide high/low band -> CS spread well above 2%.
    bars = _bars(high=12.0, low=8.0)
    ev = db.evaluate_band(bars, shares=20_000_000.0)
    assert ev.tradeable is False
    assert ev.band is None
    assert ev.spread > db.SPREAD_CAP_PCT


def test_depth_screen_rejects_thin_names():
    bars = _bars(close=10.0, volume=1_000.0)  # 10*1000 = 10k < $50k
    ev = db.evaluate_band(bars, shares=20_000_000.0)
    assert ev.tradeable is False
    assert ev.band is None
    assert ev.dollar_volume < db.DV_FLOOR


def test_cap_ignores_future_split_lookahead_guard():
    """A future split retro-adjusts the ADJUSTED close but never close_raw.
    The cap MUST read close_raw, so a name's as-of-D cap is identical whether
    or not a later split has been applied to the adjusted column."""
    pre = _bars(close_raw=10.0, close=10.0, volume=100_000.0)
    # Simulate a vendor 2:1 retro-adjustment of the ADJUSTED close only.
    post = pre.copy()
    post["close"] = post["close"] / 2.0        # adjusted halved
    post["volume"] = post["volume"] * 2.0      # adjusted volume doubled
    ev_pre = db.evaluate_band(pre, shares=20_000_000.0)
    ev_post = db.evaluate_band(post, shares=20_000_000.0)
    assert ev_pre.market_cap == ev_post.market_cap == pytest.approx(200_000_000.0)
    # Sanity: a cap computed on the adjusted close WOULD have leaked the split.
    assert (20_000_000.0 * float(post["close"].iloc[-1])) != pytest.approx(200_000_000.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_downcap_band.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'trading.venues.universes.downcap_band'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/trading/venues/universes/downcap_band.py
"""Raw-price PIT market cap + the three-screen down-cap band selector (R3 spec
section 2). Dynamic per decision date D: a candidate is IN the band iff its
raw-price cap is in [$50M, $2B] AND its trailing-63 Corwin-Schultz spread is
<= 2% AND its trailing-63 median dollar-volume is >= $50k/day.

Correctness (frozen): the cap uses close_raw (the actual share price), NEVER
the split/dividend-adjusted close -- an adjusted close misstates the absolute
cap (the div_yield look-ahead class). The tradeability screens reuse the
existing adjusted OHLCV (split-consistent by construction), matching the R1
cost machinery and the spec.py liquidity signals."""

from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

from trading.alphasearch.costs import SPREAD_CAP, SPREAD_FLOOR, trailing_effective_spread

BAND_LO = 50_000_000.0
SMALL_LO = 300_000_000.0
BAND_HI = 2_000_000_000.0
SPREAD_CAP_PCT = 0.02          # spec: 2.0% CS effective-spread screen
DV_FLOOR = 50_000.0            # spec: $50k/day median dollar-volume floor
DOWNCAP_TRAILING_WINDOW = 63   # spec: trailing-63-session for BOTH tradeability screens


def market_cap_raw(shares: float, close_raw: float) -> float:
    """shares_outstanding x RAW close (spec section 2, item 1)."""
    return shares * close_raw


def band_of(market_cap: float) -> str | None:
    """The cap bucket: "micro" ($50M-$300M), "small" ($300M-$2B), or None
    (outside [$50M, $2B]). Boundaries: lower inclusive, $300M is the
    micro/small split (>= $300M is small), $2B inclusive upper."""
    if math.isnan(market_cap) or market_cap < BAND_LO or market_cap > BAND_HI:
        return None
    return "small" if market_cap >= SMALL_LO else "micro"


def downcap_effective_spread(bars: pd.DataFrame) -> float:
    """The trailing-63-session CS effective spread as-of the last row, floored
    at SPREAD_FLOOR and capped at SPREAD_CAP -- reuses the R1 estimator at the
    63-session window the R3 spec names (R1's own `effective_spread` helper is
    21-session). NaN when fewer than 2 bars (no two-day pair)."""
    if len(bars) < 2:
        return math.nan
    trailing = trailing_effective_spread(bars, window=DOWNCAP_TRAILING_WINDOW)
    value = float(trailing.iloc[-1])
    if math.isnan(value):
        return math.nan
    return min(max(value, SPREAD_FLOOR), SPREAD_CAP)


def median_dollar_volume(bars: pd.DataFrame, window: int = DOWNCAP_TRAILING_WINDOW) -> float:
    """Trailing-`window` median of close x volume (adjusted, split-consistent;
    matches spec.py's liquidity signals). NaN on an empty frame."""
    if bars.empty:
        return math.nan
    dollar = (bars["close"] * bars["volume"]).tail(window)
    return float(dollar.median())


@dataclass(frozen=True)
class BandEval:
    """One candidate's as-of-D band decision plus the audit fields the Phase-A
    gate consumes. `band` is non-None ONLY when has_shares AND tradeable AND
    the raw-price cap falls in [$50M, $2B]."""
    band: str | None
    has_shares: bool
    tradeable: bool
    market_cap: float
    spread: float
    dollar_volume: float


def evaluate_band(
    bars: pd.DataFrame,
    shares: float,
    *,
    spread_cap: float = SPREAD_CAP_PCT,
    dv_floor: float = DV_FLOOR,
) -> BandEval:
    """Evaluate a candidate at the as-of bar (last row of `bars`, already
    truncated to <= D by the caller). Fail-closed on missing shares: no cap,
    no band -- but the tradeability screens are still computed so the Phase-A
    shares-coverage denominator (tradeable candidate-months) is honest."""
    spread = downcap_effective_spread(bars)
    dollar_volume = median_dollar_volume(bars)
    tradeable = (
        not math.isnan(spread) and spread <= spread_cap
        and not math.isnan(dollar_volume) and dollar_volume >= dv_floor
    )
    has_shares = not math.isnan(shares) and shares > 0
    if not has_shares:
        return BandEval(None, False, tradeable, math.nan, spread, dollar_volume)
    close_raw = float(bars["close_raw"].iloc[-1])
    market_cap = market_cap_raw(shares, close_raw)
    cap_band = band_of(market_cap)
    band = cap_band if (tradeable and cap_band is not None) else None
    return BandEval(band, True, tradeable, market_cap, spread, dollar_volume)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_downcap_band.py -v`
Expected: PASS (8 tests)

- [ ] **Step 5: Commit**

```bash
git add src/trading/venues/universes/downcap_band.py tests/test_downcap_band.py
git commit -m "feat: raw-price PIT band selector with look-ahead guard [AI]"
```

---

## Task A3: Non-index bar-backfill driver + fresh cache dir

**Files:**
- Create: `src/trading/venues/universes/downcap_backfill.py`
- Create: `scripts/backfill_downcap_bars.py`
- Test: `tests/test_downcap_backfill.py`

**Interfaces:**
- Consumes: `trading.venues.universes.downcap_roster.candidates_at`; `trading.data.cache.OhlcvCache`; `trading.data.cache.CacheSourceError`; an adapter with `.fetch_ohlcv(symbol, start, end)`; `trading.venues.base.DataFetchError`, `RateLimitError`.
- Produces:
  - `@dataclass class BackfillReport` fields `fetched: int, missing: list[str], errors: list[str], total: int`, property `coverage: float`
  - `def roster_symbols(roster: pd.DataFrame, start: datetime.date, end: datetime.date) -> list[str]`
  - `def run_backfill(symbols, cache, adapter, start, end, *, throttle_s=0.0, rate_limit_wait_s=300.0, on_progress=None) -> BackfillReport`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_downcap_backfill.py
import datetime
from pathlib import Path

import pandas as pd
import pytest

from trading.data.cache import CacheSourceError, OhlcvCache
from trading.venues.base import DataFetchError
from trading.venues.universes import downcap_backfill as bf
from trading.venues.universes import downcap_roster as dr


def _roster():
    return pd.DataFrame(
        {
            "ticker": ["AAA", "BBB", "GONE"],
            "exchange": ["NYSE", "NASDAQ", "NYSE"],
            "assetType": ["Stock", "Stock", "Stock"],
            "priceCurrency": ["USD", "USD", "USD"],
            "startDate": ["2019-01-01", "2019-01-01", "2019-01-01"],
            "endDate": ["", "", ""],
        }
    )


class _Adapter:
    """AAA/BBB serve bars; GONE 404s (adapter raises DataFetchError)."""

    def fetch_ohlcv(self, symbol, start, end):
        if symbol == "GONE":
            raise DataFetchError(f"no equities data for {symbol}")
        idx = pd.date_range(start, periods=3, freq="B", tz="UTC")
        return pd.DataFrame({"close": [1.0, 1.0, 1.0]}, index=idx)


def test_roster_symbols_are_window_candidates():
    syms = bf.roster_symbols(_roster(), datetime.date(2019, 1, 1), datetime.date(2019, 6, 1))
    assert syms == ["AAA", "BBB", "GONE"]


def test_run_backfill_records_gaps_not_guesses(tmp_path):
    cache = OhlcvCache(tmp_path / "equities-downcap-tiingo", refetch_days=5, source="tiingo")
    report = bf.run_backfill(
        ["AAA", "BBB", "GONE"], cache, _Adapter(),
        datetime.date(2019, 1, 1), datetime.date(2019, 1, 10),
    )
    assert report.fetched == 2
    assert report.missing == ["GONE"]        # recorded, never imputed
    assert report.total == 3
    assert report.coverage == pytest.approx(2 / 3)


def test_cache_dir_gets_tiingo_source_marker(tmp_path):
    cache_dir = tmp_path / "equities-downcap-tiingo"
    OhlcvCache(cache_dir, refetch_days=5, source="tiingo")
    assert (cache_dir / ".source").read_text().strip() == "tiingo"


def test_refuses_source_mismatch(tmp_path):
    cache_dir = tmp_path / "equities-downcap-tiingo"
    OhlcvCache(cache_dir, refetch_days=5, source="tiingo")   # writes .source=tiingo
    with pytest.raises(CacheSourceError):
        OhlcvCache(cache_dir, refetch_days=5, source="yfinance")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_downcap_backfill.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'trading.venues.universes.downcap_backfill'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/trading/venues/universes/downcap_backfill.py
"""Warm the down-cap bar cache from a survivorship-free ROSTER (not index
membership) and REPORT coverage explicitly (R3 spec section 3). Mirrors
scripts/backfill_bars.py's cold-warm discipline: fetch every candidate up
front so a source gap is a RECORDED coverage hole, never a silent loss. The
cache dir is fresh with its own `.source = tiingo` marker (OhlcvCache guards
mixing sources)."""

from __future__ import annotations

import datetime
from collections.abc import Callable
from dataclasses import dataclass, field

import pandas as pd

from trading.venues.base import DataFetchError, RateLimitError
from trading.venues.universes.downcap_roster import candidates_at


def _sleep(seconds: float) -> None:
    """Sleep touchpoint, isolated for tests."""
    import time

    time.sleep(seconds)


@dataclass
class BackfillReport:
    fetched: int = 0
    missing: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    total: int = 0

    @property
    def coverage(self) -> float:
        return self.fetched / self.total if self.total else 0.0


def roster_symbols(
    roster: pd.DataFrame, start: datetime.date, end: datetime.date
) -> list[str]:
    """Every ticker that was a roster candidate on ANY month-start in
    [start, end] -- the survivorship-free set including mid-window delistings.
    Monthly sampling suffices: listing intervals are far longer than a month."""
    seen: set[str] = set()
    day = start
    while day <= end:
        seen.update(candidates_at(roster, day))
        year, month = (day.year + 1, 1) if day.month == 12 else (day.year, day.month + 1)
        day = datetime.date(year, month, 1)
    return sorted(seen)


def _fetch_waiting_on_rate_limit(cache, adapter, symbol, start, end, wait_s):
    """cache.fetch, but a rate-limit rejection waits and retries the SAME
    symbol (a metered plan slows, never punches coverage holes). A genuine
    miss (404 -> empty -> DataFetchError) propagates immediately."""
    while True:
        try:
            return cache.fetch(symbol, start, end, adapter.fetch_ohlcv)
        except RateLimitError:
            _sleep(wait_s)


def run_backfill(
    symbols: list[str],
    cache,
    adapter,
    start: datetime.date,
    end: datetime.date,
    *,
    throttle_s: float = 0.0,
    rate_limit_wait_s: float = 300.0,
    on_progress: Callable[[int, int, BackfillReport], None] | None = None,
) -> BackfillReport:
    """Fetch every symbol into `cache`, recording gaps. A DataFetchError (no
    such ticker / thin history) is a RECORDED miss, not an abort; any other
    exception is a recorded hard error (investigate before trusting the run)."""
    report = BackfillReport(total=len(symbols))
    for i, symbol in enumerate(symbols, 1):
        if throttle_s and i > 1:
            _sleep(throttle_s)
        try:
            df = _fetch_waiting_on_rate_limit(
                cache, adapter, symbol, start, end, rate_limit_wait_s
            )
            if df.empty:
                report.missing.append(symbol)
            else:
                report.fetched += 1
        except DataFetchError:
            report.missing.append(symbol)
        except Exception as exc:  # noqa: BLE001 - report, don't abort the pass
            report.errors.append(f"{symbol}: {type(exc).__name__}: {exc}")
        if on_progress is not None:
            on_progress(i, len(symbols), report)
    return report
```

```python
# scripts/backfill_downcap_bars.py
"""Warm data/equities-downcap-tiingo/ for every roster candidate in the
discovery span, from Tiingo. Run overnight on the always-on mini (Tiingo
rate-limited). Coverage gaps are printed, never imputed.

Prereq: fetch the roster ZIP once, e.g.
    uv run python -c "from pathlib import Path; from trading.venues.universes.downcap_roster import fetch_supported_tickers; fetch_supported_tickers(Path('data/tiingo_supported_tickers.zip'))"

    uv run python scripts/backfill_downcap_bars.py --throttle-s 0.5
"""

from __future__ import annotations

import argparse
import datetime
import sys
from pathlib import Path

from trading.config import load_venue_config
from trading.data.cache import OhlcvCache
from trading.venues import make_adapter
from trading.venues.universes.downcap_backfill import roster_symbols, run_backfill
from trading.venues.universes.downcap_roster import (
    parse_supported_tickers,
    structural_roster,
)

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", default="config/experiments/tiingo")
    parser.add_argument("--roster-zip", default="data/tiingo_supported_tickers.zip")
    parser.add_argument("--cache-dir", default="data/equities-downcap-tiingo")
    parser.add_argument("--start", default="2018-01-01")  # 1yr pre-roll for trailing windows
    parser.add_argument("--throttle-s", type=float, default=0.0)
    parser.add_argument("--rate-limit-wait-s", type=float, default=300.0)
    parser.add_argument("--min-coverage", type=float, default=0.60)
    args = parser.parse_args()

    config = load_venue_config("equities", Path(args.config_dir))
    adapter = make_adapter(config)
    cache = OhlcvCache(Path(args.cache_dir), config.data.refetch_days, source="tiingo")

    roster, report = structural_roster(parse_supported_tickers(Path(args.roster_zip)))
    print(f"roster: {len(roster)} candidates; exchange report:")
    print(f"  kept:    {report['kept']}")
    print(f"  dropped: {report['dropped']}")

    start = datetime.date.fromisoformat(args.start)
    end = datetime.date.today()
    symbols = roster_symbols(roster, start, end)
    print(f"backfilling {len(symbols)} symbols into {args.cache_dir} ({start}..{end})")

    def progress(i, n, rep):
        if i % 50 == 0:
            print(f"  {i}/{n} ({rep.fetched} ok, {len(rep.missing)} missing)", flush=True)

    result = run_backfill(
        symbols, cache, adapter, start, end,
        throttle_s=args.throttle_s, rate_limit_wait_s=args.rate_limit_wait_s,
        on_progress=progress,
    )
    print(f"\ncoverage: {result.fetched}/{result.total} = {result.coverage:.1%}")
    if result.missing:
        print(f"source lacks bars for {len(result.missing)} symbol(s) (recorded gaps)")
    if result.errors:
        print(f"\n{len(result.errors)} hard error(s):")
        for line in result.errors[:20]:
            print(f"  {line}")
    if result.coverage < args.min_coverage:
        print(f"\nWARN: coverage {result.coverage:.1%} < {args.min_coverage:.0%} floor")
    print("\nbackfill done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_downcap_backfill.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add src/trading/venues/universes/downcap_backfill.py scripts/backfill_downcap_bars.py tests/test_downcap_backfill.py
git commit -m "feat: non-index down-cap bar backfill driver + fresh cache dir [AI]"
```

---

## Task A4: Dynamic PIT band-membership + diagnostics builder

**Files:**
- Create: `src/trading/venues/universes/downcap_membership.py`
- Create: `scripts/build_downcap_membership.py`
- Test: `tests/test_downcap_membership.py`

**Interfaces:**
- Consumes: `downcap_roster.candidates_at`, `downcap_roster.delisted_symbols`; `downcap_band.evaluate_band`; `trading.alphasearch.panel.load_bars`; `trading.fundamentals.store.FundamentalsStore`; `trading.signals.quality.latest_filed_row`; `trading.alphasearch.sweep.DISCOVERY_WINDOW`.
- Produces:
  - `MEMBERSHIP_COLUMNS = ["band", "symbol", "start", "end"]`
  - `DIAGNOSTICS_COLUMNS = ["date", "symbol", "delisted", "tradeable", "has_shares", "band", "spread", "dollar_volume", "market_cap"]`
  - `def monthly_decision_dates(calendar, start, end) -> list[pd.Timestamp]`
  - `def load_calendar(bars: dict[str, pd.DataFrame]) -> list[pd.Timestamp]`
  - `@dataclass class MembershipBuild` fields `membership: pd.DataFrame, diagnostics: pd.DataFrame`
  - `def build_band_membership(roster, bars, store, *, discovery_window=DISCOVERY_WINDOW, require_cap_band=True) -> MembershipBuild`
  - `def write_membership(build: MembershipBuild, membership_path: Path, diagnostics_path: Path) -> None`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_downcap_membership.py
import numpy as np
import pandas as pd
import pytest

from trading.alphasearch.panel import BAR_COLUMNS
from trading.venues.universes import downcap_membership as dm


def _roster(tickers, delisted=()):
    n = len(tickers)
    return pd.DataFrame(
        {
            "ticker": list(tickers),
            "exchange": ["NYSE"] * n,
            "assetType": ["Stock"] * n,
            "priceCurrency": ["USD"] * n,
            "startDate": ["2018-01-01"] * n,
            "endDate": ["2023-06-30" if t in delisted else "" for t in tickers],
        }
    )


def _bars(close_raw, close=None, volume=100_000.0, start="2018-06-01", periods=420):
    idx = pd.date_range(start, periods=periods, freq="B", tz="UTC")
    close = close_raw if close is None else close
    frame = pd.DataFrame(
        {
            "open": close, "high": close * 1.002, "low": close * 0.998, "close": close,
            "volume": volume, "div_cash": 0.0, "split_factor": 1.0, "close_raw": close_raw,
        },
        index=idx,
    )
    return frame[BAR_COLUMNS]


class _Store:
    """Minimal FundamentalsStore stand-in: shares filed 2018-01-02, visible
    for all discovery dates."""

    def __init__(self, shares_by_symbol):
        self._shares = shares_by_symbol

    def read(self, symbol):
        sh = self._shares.get(symbol)
        idx = pd.DatetimeIndex(["2018-01-02"], tz="UTC")
        if sh is None:
            return pd.DataFrame({"shares_outstanding": []},
                                index=pd.DatetimeIndex([], tz="UTC"))
        return pd.DataFrame({"shares_outstanding": [sh]}, index=idx)


def test_monthly_decision_dates_first_session_per_month():
    cal = list(pd.date_range("2019-01-01", "2019-03-31", freq="B", tz="UTC"))
    dates = dm.monthly_decision_dates(cal, pd.Timestamp("2019-01-01", tz="UTC"),
                                      pd.Timestamp("2019-03-31", tz="UTC"))
    assert [d.strftime("%Y-%m") for d in dates] == ["2019-01", "2019-02", "2019-03"]
    assert dates[0] == pd.Timestamp("2019-01-01", tz="UTC")


def test_build_membership_bands_and_breadth():
    # MICRO: 20M sh * $10 = $200M ; SMALL: 20M sh * $50 = $1B
    roster = _roster(["MIC", "SML", "NOSH"])
    bars = {"MIC": _bars(10.0), "SML": _bars(50.0), "NOSH": _bars(20.0)}
    store = _Store({"MIC": 20_000_000.0, "SML": 20_000_000.0})  # NOSH has no shares
    build = dm.build_band_membership(roster, bars, store,
                                     discovery_window="2019-01-01..2019-03-31")
    mem = build.membership
    assert set(mem.loc[mem["symbol"] == "MIC", "band"]) == {"micro"}
    assert set(mem.loc[mem["symbol"] == "SML", "band"]) == {"small"}
    assert "NOSH" not in set(mem["symbol"])            # fail-closed, excluded
    diag = build.diagnostics
    nosh = diag[diag["symbol"] == "NOSH"]
    assert (nosh["has_shares"] == False).all()          # noqa: E712
    assert (nosh["tradeable"] == True).all()            # noqa: E712  counted in denom


def test_membership_intervals_coalesce_contiguous_months():
    roster = _roster(["MIC"])
    bars = {"MIC": _bars(10.0)}
    store = _Store({"MIC": 20_000_000.0})
    build = dm.build_band_membership(roster, bars, store,
                                     discovery_window="2019-01-01..2019-03-31")
    rows = build.membership[build.membership["symbol"] == "MIC"]
    assert len(rows) == 1                                # 3 contiguous months -> 1 interval
    assert rows["start"].iloc[0] == "2019-01-01"
    assert rows["end"].iloc[0] == ""                     # in-band through the last month


def test_delisted_flag_recorded_for_survivorship_metric():
    roster = _roster(["MIC"], delisted=["MIC"])
    bars = {"MIC": _bars(10.0)}
    store = _Store({"MIC": 20_000_000.0})
    build = dm.build_band_membership(roster, bars, store,
                                     discovery_window="2019-01-01..2019-03-31")
    assert (build.diagnostics["delisted"] == True).all()  # noqa: E712


def test_dollar_volume_only_fallback_ignores_cap():
    # $5B cap is OUTSIDE the band, but tradeable -> included only in fallback mode.
    roster = _roster(["BIG"])
    bars = {"BIG": _bars(250.0)}
    store = _Store({"BIG": 20_000_000.0})               # 20M * $250 = $5B
    capped = dm.build_band_membership(roster, bars, store,
                                      discovery_window="2019-01-01..2019-02-28")
    assert "BIG" not in set(capped.membership["symbol"])
    fallback = dm.build_band_membership(roster, bars, store,
                                        discovery_window="2019-01-01..2019-02-28",
                                        require_cap_band=False)
    rows = fallback.membership[fallback.membership["symbol"] == "BIG"]
    assert set(rows["band"]) == {"downcap"}             # single band, cap ignored
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_downcap_membership.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'trading.venues.universes.downcap_membership'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/trading/venues/universes/downcap_membership.py
"""The dynamic PIT band-membership builder (R3 spec section 2/4). Runs the
band selector at the first trading session of every discovery month for every
roster candidate, then:

  * MEMBERSHIP artifact -- (band, symbol, start, end) intervals (end
    EXCLUSIVE, "" = open), the science input the UniverseSpecs consume. Its
    (band, symbol, start, end) shape deliberately mirrors
    equities_membership.csv so the same interval-overlap logic loads it.
  * DIAGNOSTICS artifact -- one row per (date, symbol) with the raw screen
    outcomes, the audit input the Phase-A gate (A5) reads for survivorship,
    shares-coverage, spread distribution, and breadth.

`require_cap_band=False` is the developer-pre-approved dollar-volume-only
fallback (spec section 4): membership = tradeable regardless of cap, single
band "downcap"."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from trading.alphasearch.sweep import DISCOVERY_WINDOW
from trading.fundamentals.store import FundamentalsStore
from trading.signals.quality import latest_filed_row
from trading.venues.universes.downcap_band import evaluate_band
from trading.venues.universes.downcap_roster import candidates_at, delisted_symbols

MEMBERSHIP_COLUMNS = ["band", "symbol", "start", "end"]
DIAGNOSTICS_COLUMNS = [
    "date", "symbol", "delisted", "tradeable", "has_shares",
    "band", "spread", "dollar_volume", "market_cap",
]
FALLBACK_BAND = "downcap"


def load_calendar(bars: dict[str, pd.DataFrame]) -> list[pd.Timestamp]:
    """The union trading calendar: any date on which at least one symbol has a
    bar (matches PanelData.decision_dates' union-calendar rule)."""
    return sorted({d for frame in bars.values() for d in frame.index})


def monthly_decision_dates(
    calendar, start: pd.Timestamp, end: pd.Timestamp
) -> list[pd.Timestamp]:
    """First trading session of each month in [start, end] (matches
    PanelData.decision_dates offset=0)."""
    by_month: dict[str, list[pd.Timestamp]] = {}
    for date in sorted(calendar):
        if start <= date <= end:
            by_month.setdefault(date.strftime("%Y-%m"), []).append(date)
    return [by_month[m][0] for m in sorted(by_month)]


@dataclass
class MembershipBuild:
    membership: pd.DataFrame
    diagnostics: pd.DataFrame


def _coalesce(month_rows: list[tuple[str, str, str]]) -> list[tuple[str, str, str]]:
    """(band, symbol, month_iso) rows in date order -> coalesced
    (band, symbol, start, end) intervals. A break in contiguous months OR a
    band change starts a new interval. `end` is the month-start of the first
    month NO LONGER in that (band) -- EXCLUSIVE -- or "" when in-band through
    the final month processed."""
    intervals: list[list[str]] = []
    # index months so contiguity is "adjacent in the ordered month list"
    all_months = sorted({m for _, _, m in month_rows})
    pos = {m: i for i, m in enumerate(all_months)}
    per_symbol: dict[str, list[tuple[str, str]]] = {}
    for band, symbol, month in month_rows:
        per_symbol.setdefault(symbol, []).append((month, band))
    out: list[tuple[str, str, str, str]] = []
    for symbol, entries in per_symbol.items():
        entries.sort()
        cur_band = None
        cur_start = None
        prev_pos = None
        for month, band in entries:
            contiguous = prev_pos is not None and pos[month] == prev_pos + 1
            if cur_band == band and contiguous:
                prev_pos = pos[month]
                continue
            if cur_band is not None:
                out.append((cur_band, symbol, cur_start, month))  # exclusive end
            cur_band, cur_start, prev_pos = band, month, pos[month]
        if cur_band is not None:
            out.append((cur_band, symbol, cur_start, ""))  # open: in-band through the end
    return out


def build_band_membership(
    roster: pd.DataFrame,
    bars: dict[str, pd.DataFrame],
    store: FundamentalsStore,
    *,
    discovery_window: str = DISCOVERY_WINDOW,
    require_cap_band: bool = True,
) -> MembershipBuild:
    start_s, _, end_s = discovery_window.partition("..")
    start = pd.Timestamp(start_s, tz="UTC")
    end = pd.Timestamp(end_s, tz="UTC")
    dates = monthly_decision_dates(load_calendar(bars), start, end)
    delisted = delisted_symbols(roster)

    diag_rows: list[dict] = []
    month_rows: list[tuple[str, str, str]] = []  # (band, symbol, month_iso)
    for d in dates:
        month_iso = d.date().isoformat()
        d_date = d.date()
        for symbol in sorted(candidates_at(roster, d_date)):
            frame = bars.get(symbol)
            if frame is None:
                continue                       # recorded gap: no bars, no row
            truncated = frame.loc[:d]          # PIT: strictly <= decision date
            if truncated.empty:
                continue
            filed = store.read(symbol)
            row = latest_filed_row(filed, d)
            shares = (
                float(row["shares_outstanding"])
                if row is not None and "shares_outstanding" in row.index
                else float("nan")
            )
            ev = evaluate_band(truncated, shares)
            diag_rows.append(
                {
                    "date": month_iso, "symbol": symbol,
                    "delisted": symbol in delisted,
                    "tradeable": ev.tradeable, "has_shares": ev.has_shares,
                    "band": ev.band, "spread": ev.spread,
                    "dollar_volume": ev.dollar_volume, "market_cap": ev.market_cap,
                }
            )
            if require_cap_band:
                if ev.band is not None:
                    month_rows.append((ev.band, symbol, month_iso))
            elif ev.tradeable:
                month_rows.append((FALLBACK_BAND, symbol, month_iso))

    membership = pd.DataFrame(_coalesce(month_rows), columns=MEMBERSHIP_COLUMNS)
    membership = membership.sort_values(["band", "symbol", "start"]).reset_index(drop=True)
    diagnostics = pd.DataFrame(diag_rows, columns=DIAGNOSTICS_COLUMNS)
    return MembershipBuild(membership=membership, diagnostics=diagnostics)


def write_membership(
    build: MembershipBuild, membership_path: Path, diagnostics_path: Path
) -> None:
    membership_path.parent.mkdir(parents=True, exist_ok=True)
    build.membership.to_csv(membership_path, index=False)
    build.diagnostics.to_csv(diagnostics_path, index=False)
```

```python
# scripts/build_downcap_membership.py
"""Build the down-cap band-membership CSV (the science artifact the three
UniverseSpecs consume) and the diagnostics CSV (the Phase-A gate input) from
the backfilled cache + fundamentals store.

    uv run python scripts/build_downcap_membership.py
    # dollar-volume-only fallback (only after A5 records the amendment):
    uv run python scripts/build_downcap_membership.py --no-cap-band
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from trading.alphasearch.panel import load_bars
from trading.fundamentals.store import FundamentalsStore
from trading.venues.universes.downcap_membership import (
    build_band_membership,
    write_membership,
)
from trading.venues.universes.downcap_roster import parse_supported_tickers, structural_roster

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--roster-zip", default="data/tiingo_supported_tickers.zip")
    p.add_argument("--cache-dir", default="data/equities-downcap-tiingo")
    p.add_argument("--fundamentals-dir", default="data/fundamentals/equities")
    p.add_argument("--out-membership", default="data/equities-downcap-tiingo/band_membership.csv")
    p.add_argument("--out-diagnostics", default="data/equities-downcap-tiingo/diagnostics.csv")
    p.add_argument("--no-cap-band", action="store_true", help="dollar-volume-only fallback")
    args = p.parse_args()

    roster, report = structural_roster(parse_supported_tickers(Path(args.roster_zip)))
    print(f"roster: {len(roster)} candidates; kept exchanges {report['kept']}")
    bars = load_bars(Path(args.cache_dir), sorted(roster["ticker"]))
    print(f"loaded bars for {len(bars)} candidates from {args.cache_dir}")
    store = FundamentalsStore(Path(args.fundamentals_dir))
    build = build_band_membership(roster, bars, store, require_cap_band=not args.no_cap_band)
    write_membership(build, Path(args.out_membership), Path(args.out_diagnostics))
    print(
        f"wrote {len(build.membership)} intervals -> {args.out_membership}; "
        f"{len(build.diagnostics)} candidate-month rows -> {args.out_diagnostics}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_downcap_membership.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add src/trading/venues/universes/downcap_membership.py scripts/build_downcap_membership.py tests/test_downcap_membership.py
git commit -m "feat: dynamic PIT band-membership + diagnostics builder [AI]"
```

---

## RUN-BOOK (operational, between A4 and A5): Phase A data acquisition

These are one-time operational steps on the always-on mini — NOT TDD code tasks. They exercise the code from A1/A3/A4 against the real Tiingo/SEC sources. Coverage gaps are printed and recorded, never imputed.

1. **Fetch the roster ZIP** (once):
   `uv run python -c "from pathlib import Path; from trading.venues.universes.downcap_roster import fetch_supported_tickers; fetch_supported_tickers(Path('data/tiingo_supported_tickers.zip'))"`
   Then record the exchange kept/dropped report (printed by step 2) in the roster build log.
2. **Overnight bar backfill** (~6–8k names, Tiingo rate-limited):
   `uv run python scripts/backfill_downcap_bars.py --throttle-s 0.5 2>&1 | tee /tmp/downcap-backfill.log`
   Re-runnable: the warm cache resumes; gaps are printed.
3. **companyfacts shares backfill** — REUSE existing machinery (`scripts/backfill_downcap_cik_map` is NOT built here; extend the CIK map first): build/extend the roster→CIK map (`scripts/build_cik_map.py` for current tickers, `scripts/build_cik_map_historical.py` for delisted), then `uv run python scripts/backfill_fundamentals.py` (companyfacts, 0.11s throttle) into `data/fundamentals/equities`. Delisted-ticker CIK coverage is IMPERFECT by nature; names whose shares never resolve are dropped fail-closed and are exactly what A5's shares-coverage metric measures. **(Flagged gap: roster-wide CIK mapping for delisted names is a known partial-coverage risk feeding the §4 gate — see final report.)**
4. **Build the membership + diagnostics artifacts:**
   `uv run python scripts/build_downcap_membership.py 2>&1 | tee /tmp/downcap-membership.log`

Only once `data/equities-downcap-tiingo/band_membership.csv` and `diagnostics.csv` exist does A5 run.

---

## Task A5: Verification report + frozen GO/NO-GO + fallback trigger

**Files:**
- Create: `src/trading/venues/universes/downcap_verify.py`
- Create: `scripts/downcap_verify.py`
- Test: `tests/test_downcap_verify.py`

**Interfaces:**
- Consumes: the diagnostics DataFrame (`DIAGNOSTICS_COLUMNS`) from A4.
- Produces:
  - `SURVIVORSHIP_MIN=0.15`, `SHARES_COVERAGE_MIN=0.70`, `BREADTH_MIN=15`
  - `@dataclass(frozen=True) class UniverseBreadth` fields `name: str, min_month_count: int, ok: bool`
  - `@dataclass(frozen=True) class GateResult` fields `survivorship_pct, shares_coverage_pct, spread_median, spread_iqr, spread_pct_le_2, breadth: list[UniverseBreadth], survivorship_ok, shares_coverage_ok, breadth_ok, fallback_triggered, go` (booleans as named)
  - `def compute_gate(diagnostics: pd.DataFrame) -> GateResult`
  - `def render_report(gate: GateResult) -> str`
  - `def render_amendment(gate: GateResult) -> str`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_downcap_verify.py
import pandas as pd
import pytest

from trading.venues.universes import downcap_verify as dv
from trading.venues.universes.downcap_membership import DIAGNOSTICS_COLUMNS


def _diag(rows):
    return pd.DataFrame(rows, columns=DIAGNOSTICS_COLUMNS)


def _row(date, symbol, band, *, delisted=False, tradeable=True, has_shares=True,
         spread=0.01, dv=1_000_000.0, cap=200_000_000.0):
    return {
        "date": date, "symbol": symbol, "delisted": delisted, "tradeable": tradeable,
        "has_shares": has_shares, "band": band, "spread": spread,
        "dollar_volume": dv, "market_cap": cap,
    }


def test_go_when_all_thresholds_met():
    rows = []
    for m in ("2019-01-01", "2019-02-01"):
        for i in range(20):                       # 20 in-band names/month, >= 15
            rows.append(_row(m, f"MIC{i}", "micro", delisted=(i < 4)))  # 20% delisted
    gate = dv.compute_gate(_diag(rows))
    assert gate.survivorship_ok is True           # 20% >= 15%
    assert gate.shares_coverage_ok is True        # all tradeable have shares
    assert gate.breadth_ok is True
    assert gate.fallback_triggered is False
    assert gate.go is True


def test_low_shares_coverage_triggers_fallback():
    rows = []
    for m in ("2019-01-01", "2019-02-01"):
        for i in range(20):
            has = i < 12                          # 12/20 = 60% < 70%
            band = "micro" if has else None
            rows.append(_row(m, f"S{i}", band, has_shares=has, delisted=(i < 3)))
    gate = dv.compute_gate(_diag(rows))
    assert gate.shares_coverage_pct == pytest.approx(0.60)
    assert gate.shares_coverage_ok is False
    assert gate.fallback_triggered is True        # developer pre-approved dv-only path
    assert "dollar-volume-only" in dv.render_amendment(gate)


def test_sub_15_breadth_month_fails_universe():
    rows = []
    # January: 20 micro names (ok). February: only 10 micro (sub-15).
    for i in range(20):
        rows.append(_row("2019-01-01", f"MIC{i}", "micro"))
    for i in range(10):
        rows.append(_row("2019-02-01", f"MIC{i}", "micro"))
    gate = dv.compute_gate(_diag(rows))
    micro = next(b for b in gate.breadth if b.name == "downcap:micro")
    assert micro.min_month_count == 10
    assert micro.ok is False
    assert gate.breadth_ok is False               # a universe with a sub-15 month
    assert gate.go is False


def test_spread_distribution_reported():
    rows = [_row("2019-01-01", f"M{i}", "micro", spread=s)
            for i, s in enumerate([0.005, 0.01, 0.015, 0.02, 0.05])]
    # only 4/5 <= 2% (0.05 excluded from in-band would normally not appear, but
    # here we force it into the band rows to exercise the distribution math)
    gate = dv.compute_gate(_diag(rows))
    assert gate.spread_median == pytest.approx(0.015)
    assert 0.0 <= gate.spread_pct_le_2 <= 1.0


def test_render_report_states_go_and_metrics():
    rows = [_row("2019-01-01", f"MIC{i}", "micro", delisted=(i < 4)) for i in range(20)]
    text = dv.render_report(dv.compute_gate(_diag(rows)))
    assert "GO" in text or "NO-GO" in text
    assert "survivorship" in text.lower()
    assert "shares-coverage" in text.lower()
    assert "breadth" in text.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_downcap_verify.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'trading.venues.universes.downcap_verify'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/trading/venues/universes/downcap_verify.py
"""Phase-A frozen GO/NO-GO gate + human report (R3 spec section 4). Pure
computation over the A4 diagnostics artifact -- no re-fetching. Thresholds
are FROZEN before any sweep so the decision is not post-hoc. The
dollar-volume-only fallback (shares-coverage < 70%) is an AUTOMATIC,
developer-pre-approved path recorded as a written amendment, never a silent
re-tune."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

SURVIVORSHIP_MIN = 0.15
SHARES_COVERAGE_MIN = 0.70
BREADTH_MIN = 15

# The three universes and the band(s) each counts (spec section 2).
_UNIVERSE_BANDS = {
    "downcap": {"micro", "small"},
    "downcap:small": {"small"},
    "downcap:micro": {"micro"},
}


@dataclass(frozen=True)
class UniverseBreadth:
    name: str
    min_month_count: int
    ok: bool


@dataclass(frozen=True)
class GateResult:
    survivorship_pct: float
    shares_coverage_pct: float
    spread_median: float
    spread_iqr: float
    spread_pct_le_2: float
    breadth: list[UniverseBreadth]
    survivorship_ok: bool
    shares_coverage_ok: bool
    breadth_ok: bool
    fallback_triggered: bool
    go: bool


def _breadth_for(in_band: pd.DataFrame, name: str, bands: set[str]) -> UniverseBreadth:
    sub = in_band[in_band["band"].isin(bands)]
    if sub.empty:
        return UniverseBreadth(name, 0, False)
    per_month = sub.groupby("date")["symbol"].nunique()
    min_count = int(per_month.min())
    return UniverseBreadth(name, min_count, min_count >= BREADTH_MIN)


def compute_gate(diagnostics: pd.DataFrame) -> GateResult:
    in_band = diagnostics[diagnostics["band"].notna()]
    tradeable = diagnostics[diagnostics["tradeable"]]

    # Survivorship: delisted share of IN-BAND candidate-months.
    survivorship_pct = (
        float(in_band["delisted"].mean()) if len(in_band) else 0.0
    )
    # Shares-coverage: among TRADEABLE candidate-months, fraction with shares.
    shares_coverage_pct = (
        float(tradeable["has_shares"].mean()) if len(tradeable) else 0.0
    )
    # Spread realism: distribution across in-band rows.
    spreads = in_band["spread"].dropna()
    spread_median = float(spreads.median()) if len(spreads) else float("nan")
    spread_iqr = (
        float(spreads.quantile(0.75) - spreads.quantile(0.25)) if len(spreads) else float("nan")
    )
    spread_pct_le_2 = float((spreads <= 0.02).mean()) if len(spreads) else float("nan")
    # Breadth: per-universe minimum monthly in-band name count.
    breadth = [_breadth_for(in_band, name, bands) for name, bands in _UNIVERSE_BANDS.items()]

    survivorship_ok = survivorship_pct >= SURVIVORSHIP_MIN
    shares_coverage_ok = shares_coverage_pct >= SHARES_COVERAGE_MIN
    breadth_ok = all(b.ok for b in breadth)
    fallback_triggered = not shares_coverage_ok
    # GO on the market-cap band iff every frozen criterion holds. A fallback is
    # NOT a GO on the market-cap band -- it is a recorded amendment to the
    # dollar-volume-only construction, re-verified on its own rebuild.
    go = survivorship_ok and shares_coverage_ok and breadth_ok
    return GateResult(
        survivorship_pct=survivorship_pct,
        shares_coverage_pct=shares_coverage_pct,
        spread_median=spread_median,
        spread_iqr=spread_iqr,
        spread_pct_le_2=spread_pct_le_2,
        breadth=breadth,
        survivorship_ok=survivorship_ok,
        shares_coverage_ok=shares_coverage_ok,
        breadth_ok=breadth_ok,
        fallback_triggered=fallback_triggered,
        go=go,
    )


def render_report(gate: GateResult) -> str:
    lines = [
        "# R3 down-cap universe -- Phase A verification report",
        "",
        f"VERDICT: {'GO' if gate.go else 'NO-GO'}",
        "",
        f"- survivorship (delisted share of in-band months): {gate.survivorship_pct:.1%} "
        f"(>= {SURVIVORSHIP_MIN:.0%}? {'PASS' if gate.survivorship_ok else 'FAIL'})",
        f"- shares-coverage (tradeable months with PIT shares): {gate.shares_coverage_pct:.1%} "
        f"(>= {SHARES_COVERAGE_MIN:.0%}? {'PASS' if gate.shares_coverage_ok else 'FAIL'})",
        f"- spread: median {gate.spread_median:.4f}, IQR {gate.spread_iqr:.4f}, "
        f"% <= 2% {gate.spread_pct_le_2:.1%}",
        "- breadth (min tradeable names / month, each universe):",
    ]
    for b in gate.breadth:
        lines.append(
            f"    {b.name}: min {b.min_month_count}/month "
            f"(>= {BREADTH_MIN}? {'PASS' if b.ok else 'DROP'})"
        )
    if gate.fallback_triggered:
        lines += ["", "AMENDMENT TRIGGERED:", render_amendment(gate)]
    return "\n".join(lines)


def render_amendment(gate: GateResult) -> str:
    return (
        f"shares-coverage {gate.shares_coverage_pct:.1%} < {SHARES_COVERAGE_MIN:.0%}: "
        "NO-GO on the market-cap band. Developer-pre-approved fallback to a "
        "dollar-volume-only band (tradeability screens 2-3 without the cap "
        "bound). Rebuild membership with "
        "`scripts/build_downcap_membership.py --no-cap-band`, re-run this gate on "
        "the fallback diagnostics, and record the size/vintage skew of the "
        "shares-dropped names in experiments.md."
    )
```

```python
# scripts/downcap_verify.py
"""Compute the Phase-A GO/NO-GO gate from the diagnostics artifact and write
the human report. Returns exit 0 on GO, 2 on NO-GO (still writes the report
and, if triggered, the fallback amendment)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from trading.venues.universes.downcap_verify import compute_gate, render_report


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--diagnostics", default="data/equities-downcap-tiingo/diagnostics.csv")
    p.add_argument("--out", default="docs/reports/downcap_phase_a.md")
    args = p.parse_args()

    diagnostics = pd.read_csv(args.diagnostics)
    # CSV round-trips bools as strings/objects; coerce the boolean columns.
    for col in ("delisted", "tradeable", "has_shares"):
        diagnostics[col] = diagnostics[col].astype(str).str.lower().isin(("true", "1"))
    gate = compute_gate(diagnostics)
    report = render_report(gate)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(report + "\n")
    print(report)
    return 0 if gate.go else 2


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_downcap_verify.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add src/trading/venues/universes/downcap_verify.py scripts/downcap_verify.py tests/test_downcap_verify.py
git commit -m "feat: Phase-A frozen GO/NO-GO gate + fallback amendment [AI]"
```

---

## Task B1: Dynamic per-date band membership plumbing in the panel core

**Files:**
- Modify: `src/trading/alphasearch/panel.py` (`PanelData` :550 add field; `PanelView.symbols` :412 add filter; `build_panel` :612 add param)
- Modify: `src/trading/alphasearch/sweep.py` (`UniverseSpec` :219 add fields; `build_universe_panel` :277 thread them)
- Create: `src/trading/venues/universes/downcap_membership.py` — ADD `load_band_membership` (extends the A4 module)
- Test: `tests/test_alphasearch_panel.py` (add cases), `tests/test_downcap_universes.py` (new)

**Interfaces:**
- Consumes: A4's `MEMBERSHIP_COLUMNS`; existing `build_panel`, `PanelData`, `PanelView`, `UniverseSpec`, `build_universe_panel`.
- Produces:
  - `PanelData.membership: dict[str, tuple[tuple[str, str], ...]]` (field, default `{}`)
  - `build_panel(..., membership: dict[str, tuple[tuple[str, str], ...]] | None = None)`
  - `UniverseSpec.membership_intervals: Path | None = None`, `UniverseSpec.bands: tuple[str, ...] | None = None`
  - `def load_band_membership(path: Path, bands: frozenset[str]) -> dict[str, tuple[tuple[str, str], ...]]` (in `downcap_membership.py`)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_downcap_universes.py
from pathlib import Path

import pandas as pd

from trading.alphasearch.panel import BAR_COLUMNS, PanelData
from trading.venues.universes.downcap_membership import (
    MEMBERSHIP_COLUMNS,
    load_band_membership,
)


def _write_membership(path):
    rows = [
        ("micro", "MIC", "2019-01-01", "2019-03-01"),  # micro Jan-Feb (end exclusive)
        ("small", "SML", "2019-01-01", ""),            # small, open through end
        ("micro", "MIC", "2019-04-01", ""),            # re-enters micro from Apr
    ]
    pd.DataFrame(rows, columns=MEMBERSHIP_COLUMNS).to_csv(path, index=False)


def test_load_band_membership_filters_bands(tmp_path):
    path = tmp_path / "band_membership.csv"
    _write_membership(path)
    micro_only = load_band_membership(path, frozenset({"micro"}))
    assert set(micro_only) == {"MIC"}
    assert micro_only["MIC"] == (("2019-01-01", "2019-03-01"), ("2019-04-01", ""))
    both = load_band_membership(path, frozenset({"micro", "small"}))
    assert set(both) == {"MIC", "SML"}


def test_panelview_symbols_are_per_date_band_filtered():
    # Two names, both have bars for the whole span; membership makes MIC in-band
    # only Jan-Feb, SML in-band always.
    idx = pd.date_range("2019-01-01", "2019-06-30", freq="B", tz="UTC")
    bars = {
        s: pd.DataFrame(
            {c: (1.0 if c != "volume" else 100_000.0) for c in BAR_COLUMNS}, index=idx
        )[BAR_COLUMNS]
        for s in ("MIC", "SML")
    }
    panel = PanelData(
        closes={s: bars[s]["close"] for s in bars},
        symbols=("MIC", "SML"),
        bars=bars,
        membership={
            "MIC": (("2019-01-01", "2019-03-01"),),
            "SML": (("2019-01-01", ""),),
        },
    )
    jan = panel.view(pd.Timestamp("2019-01-15", tz="UTC"))
    assert set(jan.symbols) == {"MIC", "SML"}
    apr = panel.view(pd.Timestamp("2019-04-15", tz="UTC"))
    assert set(apr.symbols) == {"SML"}          # MIC left the band -> excluded at D


def test_empty_membership_leaves_symbols_unfiltered():
    idx = pd.date_range("2019-01-01", periods=5, freq="B", tz="UTC")
    bars = {"X": pd.DataFrame(
        {c: 1.0 for c in BAR_COLUMNS}, index=idx)[BAR_COLUMNS]}
    panel = PanelData(closes={"X": bars["X"]["close"]}, symbols=("X",), bars=bars)
    view = panel.view(pd.Timestamp("2019-01-03", tz="UTC"))
    assert set(view.symbols) == {"X"}           # default {} -> unchanged behavior
```

Add to `tests/test_alphasearch_panel.py`:

```python
def test_build_panel_threads_membership(tmp_path):
    import pandas as pd
    from trading.alphasearch.panel import BAR_COLUMNS, build_panel

    idx = pd.date_range("2019-01-01", periods=10, freq="B", tz="UTC")
    frame = pd.DataFrame({c: (1.0 if c != "volume" else 1e5) for c in BAR_COLUMNS},
                         index=idx)[BAR_COLUMNS]
    frame.to_parquet(tmp_path / "AAA.parquet")
    panel = build_panel(
        tmp_path, None, None,
        symbols=("AAA",),
        membership={"AAA": (("2019-01-01", ""),)},
    )
    assert panel.membership == {"AAA": (("2019-01-01", ""),)}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_downcap_universes.py tests/test_alphasearch_panel.py::test_build_panel_threads_membership -v`
Expected: FAIL — `load_band_membership` undefined and `PanelData` has no `membership` field.

- [ ] **Step 3: Write minimal implementation**

In `src/trading/alphasearch/panel.py`, add the `membership` field to `PanelData` (after the `sectors` field at :550-block):

```python
    # symbol -> tuple of (start_iso, end_iso) band-membership intervals (end
    # EXCLUSIVE, "" = open). Non-empty ONLY for dynamic-membership universes
    # (R3 down-cap): PanelView.symbols then restricts to names whose interval
    # contains as_of, the point-in-time-correct expression of a band that a
    # name enters/leaves as its cap and liquidity cross the bounds. Empty {}
    # (the default) preserves Piece 1/2 behavior exactly.
    membership: dict[str, tuple[tuple[str, str], ...]] = field(default_factory=dict)
```

Replace `PanelView.symbols` (:412) with the per-`as_of` band filter (empty membership → unchanged behavior):

```python
    @property
    def symbols(self) -> tuple[str, ...]:
        membership = self._panel.membership
        if not membership:
            return self._panel.symbols
        iso = self.as_of.date().isoformat()
        return tuple(
            s
            for s in self._panel.symbols
            if any(a <= iso and (b == "" or iso < b) for (a, b) in membership.get(s, ()))
        )
```

Add the `membership` parameter to `build_panel` (:612) — extend the signature and the `PanelData(...)` construction:

```python
def build_panel(
    cache_dir: Path,
    samples: Path | None,
    fundamentals_dir: Path | None,
    *,
    insider_dir: Path | None = None,
    symbols: tuple[str, ...] | None = None,
    factors: pd.DataFrame | None = None,
    sectors: dict[str, str] | None = None,
    membership: dict[str, tuple[tuple[str, str], ...]] | None = None,
) -> PanelData:
```

and in the returned `PanelData(...)` add (restricting to universe symbols like the other maps):

```python
        membership=(
            {} if membership is None else {s: membership[s] for s in universe if s in membership}
        ),
```

In `src/trading/alphasearch/sweep.py`, add two fields to `UniverseSpec` (after `sic_map_path` at :237):

```python
    # R3 down-cap: the (band, symbol, start, end) membership CSV and the band(s)
    # this universe counts. When set, build_universe_panel loads the per-symbol
    # intervals for `bands` and hands them to build_panel, so PanelView.symbols
    # is per-date band-filtered. None (the default) = Piece 1/2 static behavior.
    membership_intervals: Path | None = None
    bands: tuple[str, ...] | None = None
```

Thread them in `build_universe_panel` (:277):

```python
def build_universe_panel(
    spec: UniverseSpec, factors: pd.DataFrame | None = None
) -> PanelData:
    membership = None
    if spec.membership_intervals is not None and spec.bands is not None:
        # Lazy import: downcap_membership imports UniverseSpec from this module,
        # so a top-level import here would be a cycle (same pattern as
        # _universe_sectors' lazy segments import).
        from trading.venues.universes.downcap_membership import load_band_membership

        membership = load_band_membership(spec.membership_intervals, frozenset(spec.bands))
    return build_panel(
        spec.cache_dir, spec.samples, spec.fundamentals_dir,
        insider_dir=spec.insider_dir,
        symbols=spec.symbols, factors=factors,
        sectors=_universe_sectors(spec.sic_map_path),
        membership=membership,
    )
```

In `src/trading/venues/universes/downcap_membership.py`, add the loader:

```python
def load_band_membership(
    path: Path, bands: frozenset[str]
) -> dict[str, tuple[tuple[str, str], ...]]:
    """Read the (band, symbol, start, end) CSV, keep only rows whose band is in
    `bands`, and return symbol -> tuple of (start_iso, end_iso) intervals (end
    EXCLUSIVE, "" = open) -- the interval shape PanelView.symbols filters on,
    mirroring equities_membership.csv's overlap logic."""
    df = pd.read_csv(path, dtype=str).fillna("")
    df = df[df["band"].isin(bands)]
    out: dict[str, list[tuple[str, str]]] = {}
    for row in df.itertuples():
        out.setdefault(row.symbol, []).append((row.start, row.end))
    return {s: tuple(sorted(iv)) for s, iv in out.items()}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_downcap_universes.py tests/test_alphasearch_panel.py -v`
Expected: PASS (new cases pass; existing panel tests unchanged).

- [ ] **Step 5: Verify existing universes are bit-unchanged, then commit**

Run: `.venv/bin/python -m pytest tests/test_alphasearch_panel.py tests/test_alphasearch_segments.py tests/test_alphasearch_sweep.py tests/test_alphasearch_golden.py -q`
Expected: PASS (default `membership={}` preserves prior behavior).

```bash
git add src/trading/alphasearch/panel.py src/trading/alphasearch/sweep.py src/trading/venues/universes/downcap_membership.py tests/test_downcap_universes.py tests/test_alphasearch_panel.py
git commit -m "feat: dynamic per-date band membership in the panel core [AI]"
```

---

## Task B2: Register the three down-cap UniverseSpecs + wire the sweep

**Files:**
- Modify: `src/trading/venues/universes/downcap_membership.py` — ADD `downcap_universes`
- Modify: `src/trading/cli.py` (`_cmd_alphasearch` :860 — add `--downcap` flag, mirror `--segments`; argparse registration alongside the existing `--segments` argument)
- Test: `tests/test_downcap_universes.py` (add cases)

**Interfaces:**
- Consumes: A4's membership CSV; `trading.alphasearch.sweep.UniverseSpec`; `load_band_membership`.
- Produces:
  - `def downcap_universes(root: Path, *, membership_path: Path | None = None) -> dict[str, UniverseSpec]`

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_downcap_universes.py
import pandas as pd

from trading.venues.universes.downcap_membership import (
    MEMBERSHIP_COLUMNS,
    downcap_universes,
)


def _write_full_membership(path):
    rows = [
        ("micro", "MIC", "2019-01-01", ""),
        ("small", "SML", "2019-01-01", ""),
    ]
    pd.DataFrame(rows, columns=MEMBERSHIP_COLUMNS).to_csv(path, index=False)


def test_downcap_universes_registers_three_specs(tmp_path):
    path = tmp_path / "band_membership.csv"
    _write_full_membership(path)
    specs = downcap_universes(tmp_path, membership_path=path)
    assert set(specs) == {"downcap", "downcap:small", "downcap:micro"}
    # Full band = union of both names; sub-bands partition it.
    assert set(specs["downcap"].symbols) == {"MIC", "SML"}
    assert set(specs["downcap:small"].symbols) == {"SML"}
    assert set(specs["downcap:micro"].symbols) == {"MIC"}
    # Each carries its band filter + the membership CSV + fresh cache dir.
    assert specs["downcap:micro"].bands == ("micro",)
    assert specs["downcap"].bands == ("micro", "small")
    assert specs["downcap"].membership_intervals == path
    assert specs["downcap"].cache_dir == tmp_path / "data" / "equities-downcap-tiingo"
    assert specs["downcap"].samples is None            # options signals refused


def test_downcap_universes_absent_csv_returns_empty(tmp_path):
    # No membership CSV built yet -> no specs (the leaderboard/sweep then just
    # omits them, like segments do when their inputs are absent).
    assert downcap_universes(tmp_path, membership_path=tmp_path / "missing.csv") == {}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_downcap_universes.py -k downcap_universes -v`
Expected: FAIL with `ImportError: cannot import name 'downcap_universes'`

- [ ] **Step 3: Write minimal implementation**

Add to `src/trading/venues/universes/downcap_membership.py`:

```python
_DOWNCAP_UNIVERSES = {
    "downcap": ("micro", "small"),
    "downcap:small": ("small",),
    "downcap:micro": ("micro",),
}


def downcap_universes(
    root: Path, *, membership_path: Path | None = None
) -> dict:
    """The three frozen down-cap UniverseSpecs (spec section 2). Each shares
    the fresh bars cache and the band-membership CSV; identity is the NAME
    (like segments). `samples=None` -> options signals refused; fundamentals
    attach when the store exists so the fundamentals family sweeps the band to
    the extent companyfacts covers it. Returns {} when the membership CSV is
    absent (nothing built yet) -- callers then simply omit these universes,
    mirroring segment_universes' absent-inputs behavior."""
    from trading.alphasearch.sweep import UniverseSpec  # lazy: avoids import cycle

    cache_dir = root / "data" / "equities-downcap-tiingo"
    if membership_path is None:
        membership_path = cache_dir / "band_membership.csv"
    if not membership_path.exists():
        return {}
    fundamentals_dir = root / "data" / "fundamentals" / "equities"
    specs: dict[str, UniverseSpec] = {}
    for name, bands in _DOWNCAP_UNIVERSES.items():
        members = load_band_membership(membership_path, frozenset(bands))
        symbols = tuple(sorted(members))
        if not symbols:
            continue  # no names in this band's CSV yet -> omit (never empty spec)
        specs[name] = UniverseSpec(
            name=name,
            cache_dir=cache_dir,
            samples=None,                               # options signals refused
            fundamentals_dir=fundamentals_dir if fundamentals_dir.is_dir() else None,
            symbols=symbols,                            # ever-in-band union (bar loading)
            membership_intervals=membership_path,       # per-date filter
            bands=bands,
        )
    return specs
```

In `src/trading/cli.py`, register a `--downcap` flag next to `--segments` (argparse setup for the `alphasearch` subcommand), then wire it into both `leaderboard --long-only` and `sweep` branches of `_cmd_alphasearch`, mirroring the existing `segment_universes` merge:

```python
    # in the leaderboard --long-only branch, after the segments merge:
    if args.downcap:
        from trading.venues.universes.downcap_membership import downcap_universes

        universes = {**universes, **downcap_universes(Path("."))}
```

```python
    # in the sweep branch, after the segments merge:
    if args.downcap:
        from trading.venues.universes.downcap_membership import downcap_universes

        dc = downcap_universes(Path("."))
        if not dc:
            print(
                "ERROR: --downcap requested but no data/equities-downcap-tiingo/"
                "band_membership.csv found; build it with "
                "scripts/build_downcap_membership.py after Phase A backfill",
                file=sys.stderr,
            )
            return 1
        universes = {**universes, **dc}
```

(Add `parser_alphasearch.add_argument("--downcap", action="store_true", help="include the R3 down-cap universes (needs the band_membership.csv from Phase A)")` alongside the existing `--segments` argument definition.)

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_downcap_universes.py -v`
Expected: PASS (all cases).

- [ ] **Step 5: Verify the CLI wiring imports cleanly, then commit**

Run: `.venv/bin/python -m pytest tests/test_alphasearch_cli.py -q && .venv/bin/python -m trading.cli alphasearch --help`
Expected: PASS; `--downcap` appears in the help.

```bash
git add src/trading/venues/universes/downcap_membership.py src/trading/cli.py tests/test_downcap_universes.py
git commit -m "feat: register three down-cap UniverseSpecs + --downcap sweep flag [AI]"
```

---

## RUN-BOOK (operational, Phase B): sweep + R1 long-only read

Runs ONLY if Phase A (A5) returned GO (or the dollar-volume-only fallback amendment was recorded and its own gate re-verified). NOT a TDD code task — it spends no holdout and journals into the existing BH-FDR journal.

1. **Sweep** (momentum_v1 PRIMARY + full battery alongside, discovery window only):
   `uv run python -m trading.cli alphasearch sweep --downcap --universe all 2>&1 | tee /tmp/downcap-sweep.log`
   Options/insider signals are refused by `_check_universe_supports` (no such stores); the price/volume + fundamentals families run to the extent companyfacts covers the band. A universe dropped for a sub-15 breadth month (A5) is excluded here, recorded not silent.
2. **Read the R1 cost-charged long-only leaderboard** (vs SPY buy-and-hold, BH survivorship a reported property):
   `uv run python -m trading.cli alphasearch leaderboard --long-only --downcap 2>&1 | tee /tmp/downcap-long-only.log`
   The four-factor L/S alpha (SMB loading) remains the mandatory diagnostic to read whether a win is the size premium in disguise.
3. **Any gate-clearing candidate** goes to `run_battery` (full battery); the **2024+ holdout is spent only on explicit developer approval** — never in this plan.
4. Record the leaderboard read + verdict in `docs/experiments.md` (Docs task section).

---

## Task Docs: experiments.md section + glossary entries

**Files:**
- Modify: `docs/experiments.md` (new "R3 — down-cap universe" section stub)
- Modify: `docs/glossary.md` (new entries)

**Interfaces:** none (documentation).

- [ ] **Step 1: Add the experiments.md section stub**

Append to `docs/experiments.md`, under "Not yet run / candidate experiments" or as a new top-level section, an "R3 — down-cap / illiquid universe" entry recording: the pre-registered thesis (a simple momentum tilt BEATS SPY in the $50M–$2B band where a ~$1k account's smallness is a capacity edge); the frozen construction (survivorship-free Tiingo roster; raw-price PIT cap band; CS spread ≤ 2% and median dollar-volume ≥ $50k/day trailing-63 screens; three universes `downcap`/`downcap:small`/`downcap:micro`); the Phase-A GO/NO-GO gate and its four frozen criteria (survivorship ≥ 15%, shares-coverage ≥ 70% else dollar-volume-only fallback, spread realism, breadth ≥ 15/month); and placeholders for the Phase-A verdict, the shares-coverage figure + dropped-names size/vintage skew, and the Phase-B long-only-vs-SPY result. Note the holdout remains unspent.

- [ ] **Step 2: Add glossary entries**

Add to `docs/glossary.md`:
- **Down-cap universe (R3)** — the survivorship-free $50M–$2B band; dynamic PIT membership recomputed monthly from raw-price cap + the two tradeability screens; three frozen `UniverseSpec`s partitioned by cap.
- **Raw-price market cap** — `shares_outstanding(latest companyfacts row FILED ≤ D) × close_raw(≤ D)`; uses the unadjusted price, never the split/dividend-adjusted `close`, so a later corporate action cannot leak into a past cap.
- **Band-membership interval CSV** — the `(band, symbol, start, end)` artifact (end exclusive, "" = open) that expresses dynamic per-date membership; loaded into `PanelData.membership`, filtered per `as_of` in `PanelView.symbols`; mirrors `equities_membership.csv`'s overlap logic.
- **Shares-coverage gate + dollar-volume-only fallback** — the ≥ 70% (of tradeable candidate-months with PIT shares) frozen criterion; below it, the developer-pre-approved amendment drops the cap bound and screens on tradeability alone.
- **Phase-A GO/NO-GO** — the frozen, pre-sweep verification (survivorship ≥ 15%, shares-coverage ≥ 70%, spread realism, breadth ≥ 15/month) computed by `downcap_verify.compute_gate` from the diagnostics artifact.

- [ ] **Step 3: Commit**

```bash
git add docs/experiments.md docs/glossary.md
git commit -m "docs: R3 down-cap universe experiments section + glossary [AI]"
```

---

## Self-Review (completed by the plan author)

**1. Spec coverage** — every section mapped to a task:
- §2 survivorship-free roster + structural filters → A1. Raw-price PIT cap + three screens + missing-shares fail-closed → A2. Three sub-universes → A2 `band_of` + B2 registration.
- §3 data acquisition: bars backfill + fresh cache/source marker → A3; companyfacts shares + roster CIK mapping + overnight run → Phase-A run-book (reuse). Coverage gaps recorded → A3 `BackfillReport`.
- §4 verification gate (survivorship, shares-coverage + fallback, spread distribution, breadth) → A4 diagnostics + A5 `compute_gate`/`render_report`/`render_amendment`.
- §5 sweep + gate (momentum_v1 primary, full battery, holdout reserved, R1 long-only read) → B2 wiring + Phase-B run-book.
- §6 build vs reuse honored (reuse `UniverseSpec`/`build_universe_panel`/`_check_universe_supports`/costs/robustness/sweep unchanged).
- §7 out of scope respected (no holdout spend, long-only only, no paid vendor).
- §8 risks: shares-coverage leak gated (A5); Tiingo micro-cap coverage recorded not imputed (A3); spread-model stress reported (A5 distribution); down-cap factor exposure read via existing four-factor diagnostic (Phase-B run-book).

**2. Placeholder scan** — no "TBD"/"handle errors"/"similar to Task N". Every code step carries complete, self-contained code; the two spots that formerly showed a first-then-corrected form were collapsed to their final form.

**3. Type consistency** — `BandEval` fields consumed identically in A4; `DIAGNOSTICS_COLUMNS` produced in A4, consumed in A5 and its test; `MEMBERSHIP_COLUMNS` produced in A4, consumed by `load_band_membership` (B1) and `downcap_universes` (B2); `UniverseSpec.membership_intervals`/`.bands` defined in B1, set in B2; `PanelData.membership` defined B1, filtered in `PanelView.symbols` (B1), populated by `build_panel` (B1) and `build_universe_panel` (B1). `DISCOVERY_WINDOW` imported from `trading.alphasearch.sweep` (verified :43). `trailing_effective_spread`/`SPREAD_FLOOR`/`SPREAD_CAP` imported from `trading.alphasearch.costs` (verified). `latest_filed_row` from `trading.signals.quality` (verified :33). `load_bars`/`BAR_COLUMNS` from `trading.alphasearch.panel` (verified :283/:278).

**Flagged spec item needing operational care (not a clean single-task map):** §3's companyfacts shares backfill for the FULL roster depends on resolving CIKs for delisted tickers, which the existing `cik_map`/`cik_map_historical` machinery covers only partially. Rather than invent a fragile roster-wide CIK builder, the plan REUSES the existing scripts in the Phase-A run-book and lets imperfect delisted-CIK coverage flow into the shares-coverage metric (A5), which is precisely the frozen ≥ 70% gate designed to catch it (with the dollar-volume-only fallback as the pre-approved escape). This is the honest treatment the spec's §4/§8 mandate, but it is an operational reuse step, not a TDD code task — flagged here so the executor does not expect a green test to certify full shares coverage.
