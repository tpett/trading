# M4: Point-in-Time Fundamentals (Quality) Overlay Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A point-in-time gross-profitability overlay sourced from SEC EDGAR XBRL data (quarterly Financial Statement Data Set ZIPs for 2018q1→present backfill + the companyfacts API for current-quarter top-up), feeding a new opt-in `quality_momentum_v1` ranker: the six momentum_v1 features plus a 7th cross-sectional gross-profitability percentile, equal-weighted over 7 — with live defaults, the golden backtest, and the two-knob tunable surface all untouched.

**Architecture:** A new `trading.fundamentals` package with a pure parsing/compute core (quarterly-ZIP facts parser + companyfacts normalizer → one shared normalized facts table → one shared PIT series computation) and an append-only per-symbol parquet store keyed by FILING date (fundamentals history is immutable — no trailing refetch, unlike `OhlcvCache`). A committed CIK↔symbol point-in-time interval map attaches per-CIK series to ticker symbols. The ranker registry moves to a v2 contract (`RankerSpec`: a 4-arg callable + a `requires_fundamentals` flag); the fundamentals dict threads through `assemble_rankings` and backtest `prepare()` with the same as-of discipline as bars, and the live runner refreshes the store weekly (fail-open, journaled warning, earnings pattern) only when the configured ranker requires it.

**Tech Stack:** Python 3.12, uv, pandas + pyarrow (pinned), stdlib `urllib`/`zipfile`/`json` for SEC I/O — **no new runtime dependencies**. pytest (warnings-as-errors), ruff.

## Global Constraints

From the locked M4 decisions. Every task's requirements implicitly include this section.

- Repo root for all commands and relative paths: `/Users/travis/Source/personal/trading/worktrees/fundamentals` (git worktree, branch `tpett/ai/fundamentals`).
- Python 3.12, uv (`uv sync`, `uv run ...`). Before every commit: `uv run ruff check . && uv run ruff format .` and the affected tests. pytest runs with warnings-as-errors.
- Baseline before starting: 334 tests green (`uv run pytest -q`), ruff clean.
- **PIT discipline (non-negotiable):** a value becomes visible at its FILING date, never its fiscal-period date; per (cik, fiscal period) only the ORIGINAL (earliest-filed, tie-break lowest accession) filing's values are used — later amendments/restatements never rewrite history; provenance columns (`adsh`, tags used, `period`, `form`) on every stored row; regression coverage against restatements (synthetic unit test + a real-data invariant check in Task 11).
- **SEC access policy (mandatory):** User-Agent `trading-system travis@launchsupply.com` on every request; stay under the 10 req/s ceiling. Raw ZIPs cached under `data/edgar-raw/` (already gitignored via `/data/`).
- **Metric v1 (locked):** gross profitability = trailing-4-quarter (Revenue − COGS) / latest Assets. Tag fallback chains exactly: Revenue `RevenueFromContractWithCustomerExcludingAssessedTax` → `Revenues` → `RevenueFromContractWithCustomerIncludingAssessedTax`; COGS `CostOfGoodsAndServicesSold` → `CostOfRevenue` → `CostOfGoodsAndServiceExcludingDepreciationDepletionAndAmortization`; Assets `Assets`. Consolidated rows only (empty `segments`/`coreg`). Incomplete TTM → NaN.
- **Ranker (locked):** `quality_momentum_v1` = six momentum_v1 features + gross-profitability cross-sectional percentile (forward-filled step function as-of the session); NaN quality contributes a NEUTRAL 0.5 percentile; composite = equal weight over 7. **NO new tunable parameters** — the walk-forward tunable surface stays exactly `entry_score_threshold` × `stop_atr_multiple`.
- **Live defaults unchanged:** both venue TOMLs keep `ranker = "momentum_v1"`; `quality_momentum_v1` is experiment-config opt-in (the sp400 `--config-dir` pattern). The golden backtest fixture and its expected output are untouched.
- Storage: per-symbol parquet at `data/fundamentals/equities/SYMBOL.parquet`, UTC DatetimeIndex = FILED dates, metric + provenance columns, append-only semantics with atomic writes (tmp + `os.replace`).
- Pure modules stay pure: parsing/metrics/ranker code does no I/O and reads no clock. Network touchpoints are isolated single functions (monkeypatch pattern, like `_yf_download`).
- All timestamps UTC. Every new number lives in TOML or a module constant with a rationale comment — never a magic literal.
- Commit after every task, one logical change per commit, message tagged `[AI]`, with footer:

  ```
  Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01WuC8FbGLPEZUjYmdPSMMdv
  ```

## File Structure

```
src/trading/fundamentals/__init__.py       NEW: empty package marker
src/trading/fundamentals/edgar.py          NEW: quarterly-ZIP sub/num parser -> normalized facts table; tag chains; USER_AGENT
src/trading/fundamentals/metrics.py        NEW: PIT gross-profitability series (original-filing dedup, Q4 derivation, TTM)
src/trading/fundamentals/store.py          NEW: FundamentalsStore (append-only per-symbol parquet, refresh marker)
src/trading/fundamentals/cik_map.py        NEW: loader/lookup for the committed CIK<->symbol interval map
src/trading/fundamentals/cik_map.csv       NEW: committed artifact (generated by scripts/build_cik_map.py)
src/trading/fundamentals/backfill.py       NEW: quarter enumeration + ZIPs -> store orchestration (importable, testable)
src/trading/fundamentals/companyfacts.py   NEW: companyfacts JSON -> facts table; weekly refresh (fail-open)
scripts/build_cik_map.py                   NEW: builds cik_map.csv (company_tickers.json + RENAMES + membership CSV)
scripts/backfill_fundamentals.py           NEW: downloads 2018q1->present ZIPs to data/edgar-raw/, builds the store
scripts/verify_fundamentals.py             NEW: AAPL TTM spot-check + real-restatement PIT invariant (Task 11)
src/trading/signals/registry.py            MODIFY: v2 contract — RankerSpec(fn, requires_fundamentals)
src/trading/signals/quality.py             NEW: quality_momentum_v1 ranker (pure)
src/trading/config.py                      MODIFY: DataConfig.fundamentals_dir / fundamentals_refresh_days + load-time check
config/equities.toml                       MODIFY: [data] fundamentals keys
config/crypto.toml                         MODIFY: [data] fundamentals keys (disabled)
config/experiments/quality/equities.toml   NEW: opt-in experiment config (ranker = quality_momentum_v1)
src/trading/pipeline.py                    MODIFY: fundamentals param through assemble_rankings; store load in build_rankings
src/trading/backtest/engine.py             MODIFY: prepare() loads + as-of-slices fundamentals per session
src/trading/runner.py                      MODIFY: weekly fundamentals refresh, fail-open journaled warning
src/trading/venues/universes/sources/PROVENANCE.md  MODIFY: EDGAR + cik_map entries
README.md                                  MODIFY: fundamentals overlay section (Task 11)
tests/fundamentals_helpers.py              NEW: fixture-ZIP + facts-row builders shared by fundamentals tests
tests/test_fundamentals_edgar.py           NEW
tests/test_fundamentals_metrics.py         NEW
tests/test_fundamentals_store.py           NEW
tests/test_fundamentals_cik_map.py         NEW
tests/test_fundamentals_backfill.py        NEW
tests/test_fundamentals_companyfacts.py    NEW
tests/test_quality_ranker.py               NEW  (test_quality.py already exists for data quality — do not touch it)
tests/test_registry.py                     MODIFY: v2 contract
tests/test_config.py                       MODIFY: new keys + load-time validation
tests/test_pipeline.py                     MODIFY: fundamentals threading
tests/test_backtest_engine.py              MODIFY: per-session as-of fundamentals slicing
tests/test_runner.py                       MODIFY: weekly refresh gate + fail-open warning
```

Locked design decisions for this plan (referenced by tasks):

- **One normalized facts table.** Both sources (quarterly ZIP `sub.txt`+`num.txt`, companyfacts JSON) normalize to the same `FACT_COLUMNS` frame and flow through the same `compute_pit_series`, so backfill and top-up cannot diverge.
- **Registry contract v2.** `RANKERS` maps name → `RankerSpec(fn, requires_fundamentals)`. Every registered `fn` takes `(bars, as_of, config, fundamentals)` where `fundamentals: dict[str, pd.DataFrame] | None`. `compute_features` keeps its 3-arg signature untouched; momentum_v1 registers through a one-line named adapter. `get_ranker(name)` returns the spec; the flag is how pipeline/runner/backtest know whether to load/refresh fundamentals at all.
- **Original-filing dedup.** Forms accepted are exactly `10-K`/`10-Q` (amendments `10-K/A`/`10-Q/A` never parse); per (cik, fy, fp) the earliest-filed filing wins, tie-break lowest adsh. The scout's `prevrpt == 0` filter is deliberately NOT reused: `prevrpt` flags the superseded ORIGINAL, which is exactly the filing PIT must keep.
- **Quarterly values.** 10-Q → qtrs=1 fact at the filing's own period end (`ddate == sub.period`). 10-K reports the full year (qtrs=4); Q4 is derived as FY − (Q1+Q2+Q3 of the same `fy`), NaN if any is missing. Assets: qtrs=0 instant at the filing's own period end, from that filing only.
- **TTM window.** The 4 most recent known fiscal quarters as of each filing; NaN if fewer than 4, if any quarter value is NaN, or if the window is ragged (newest period end − oldest period end > 330 days; 4 consecutive quarter-ends span ~273 days, the slack absorbs 53-week fiscal years).
- **Rows are emitted even when the metric is NaN** (provenance shows a filing happened); the ranker reads the LAST row as-of the session — a step function on FILED dates, no interpolation, no dropna reach-back.
- **CIK map intervals.** `cik_map.csv` rows are `(symbol, cik, start, end)`, start inclusive / end exclusive / empty end = current, floored at 2017-01-01 (same pad as membership). A fundamentals row attaches to the symbol whose interval contains its FILED date, so FB gets pre-rename filings and META post-rename ones while sharing one CIK's TTM continuity.
- **Fail-open (earnings pattern).** Live refresh failures degrade to the last stored values with a journaled warning; they never crash or block a run. An empty/missing store yields NaN quality → all-neutral 0.5 → momentum-equivalent ordering.
- **2018q1 backfill start is locked.** Consequence (documented, not a bug): TTM needs 4 trailing quarters, so most symbols' metric is NaN (neutral) until the FY-2018 10-K wave lands in early 2019. `scripts/backfill_fundamentals.py` accepts `--from-quarter` (e.g. `2017q1`) if the operator later decides to fill the warm-up; that is a deliberate operator action, not this plan's default.

---

### Task 1: EDGAR quarterly-ZIP facts parser

**Files:**
- Create: `src/trading/fundamentals/__init__.py`
- Create: `src/trading/fundamentals/edgar.py`
- Create: `tests/fundamentals_helpers.py`
- Test: `tests/test_fundamentals_edgar.py`

**Interfaces:**
- Consumes: nothing from this milestone (stdlib + pandas only).
- Produces: `USER_AGENT: str`; `REVENUE_TAGS`, `COGS_TAGS`, `ASSETS_TAGS`, `TAG_PRIORITY: dict[str, list[str]]` (keys `"revenue" | "cogs" | "assets"`); `FACT_COLUMNS: list[str]`; `empty_facts() -> pd.DataFrame`; `load_quarter_facts(zip_path: Path, ciks: set[int] | None = None) -> pd.DataFrame`. Facts dtypes: `cik` int, `adsh/form/fy/fp/tag/concept` str, `period/filed` tz-naive `pd.Timestamp`, `qtrs` int, `value` float. Tasks 2, 5, 6 consume all of these.

- [ ] **Step 1: Write the shared fixture helpers**

Create `tests/fundamentals_helpers.py`:

```python
"""Builders for synthetic SEC Financial Statement Data Set fixtures.

write_quarter_zip fabricates a minimal quarterly ZIP (sub.txt + num.txt with
only the columns the parser reads via usecols); fact/filing_facts build
already-normalized facts rows for metrics-level tests that skip the ZIP layer.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import pandas as pd

from trading.fundamentals.edgar import FACT_COLUMNS

SUB_HEADER = "adsh\tcik\tname\tform\tperiod\tfy\tfp\tfiled"
NUM_HEADER = "adsh\ttag\tversion\tddate\tqtrs\tuom\tsegments\tcoreg\tvalue"


def sub_line(adsh: str, cik: int, form: str, period: str, fy: str, fp: str, filed: str) -> str:
    return f"{adsh}\t{cik}\tTEST CO\t{form}\t{period}\t{fy}\t{fp}\t{filed}"


def num_line(
    adsh: str,
    tag: str,
    ddate: str,
    qtrs: int,
    value: float,
    uom: str = "USD",
    segments: str = "",
    coreg: str = "",
) -> str:
    return f"{adsh}\t{tag}\tus-gaap/2023\t{ddate}\t{qtrs}\t{uom}\t{segments}\t{coreg}\t{value}"


def write_quarter_zip(path: Path, sub_lines: list[str], num_lines: list[str]) -> Path:
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("sub.txt", "\n".join([SUB_HEADER, *sub_lines]) + "\n")
        zf.writestr("num.txt", "\n".join([NUM_HEADER, *num_lines]) + "\n")
    return path


def fact(
    cik: int,
    adsh: str,
    form: str,
    fy: str,
    fp: str,
    period: str,
    filed: str,
    concept: str,
    tag: str,
    qtrs: int,
    value: float,
) -> dict:
    return {
        "cik": cik,
        "adsh": adsh,
        "form": form,
        "fy": fy,
        "fp": fp,
        "period": pd.Timestamp(period),
        "filed": pd.Timestamp(filed),
        "concept": concept,
        "tag": tag,
        "qtrs": qtrs,
        "value": value,
    }


def filing_facts(
    cik: int,
    adsh: str,
    form: str,
    fy: str,
    fp: str,
    period: str,
    filed: str,
    revenue: float | None = None,
    cogs: float | None = None,
    assets: float | None = None,
) -> list[dict]:
    """One filing's normalized facts: revenue/cogs at the form's duration
    (10-K = 4 quarters, 10-Q = 1), assets as an instant (qtrs=0)."""
    qtrs = 4 if form == "10-K" else 1
    rows = []
    if revenue is not None:
        rows.append(fact(cik, adsh, form, fy, fp, period, filed, "revenue", "Revenues", qtrs, revenue))
    if cogs is not None:
        rows.append(fact(cik, adsh, form, fy, fp, period, filed, "cogs", "CostOfRevenue", qtrs, cogs))
    if assets is not None:
        rows.append(fact(cik, adsh, form, fy, fp, period, filed, "assets", "Assets", 0, assets))
    return rows


def facts_frame(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=FACT_COLUMNS)
```

- [ ] **Step 2: Write the failing tests**

Create `tests/test_fundamentals_edgar.py`:

```python
from fundamentals_helpers import num_line, sub_line, write_quarter_zip
from trading.fundamentals.edgar import FACT_COLUMNS, empty_facts, load_quarter_facts

# One 10-Q (cik 100, Q1 2023) + one 10-K (cik 200, FY 2022).
SUBS = [
    sub_line("0001-23-000001", 100, "10-Q", "20230331", "2023", "Q1", "20230510"),
    sub_line("0002-23-000001", 200, "10-K", "20221231", "2022", "FY", "20230225"),
    sub_line("0003-23-000001", 300, "10-K/A", "20221231", "2022", "FY", "20230301"),
    sub_line("0004-23-000001", 400, "8-K", "20230331", "2023", "Q1", "20230410"),
]
NUMS = [
    # cik 100 10-Q: both revenue tags present -> priority tag must win.
    num_line("0001-23-000001", "RevenueFromContractWithCustomerExcludingAssessedTax", "20230331", 1, 100.0),
    num_line("0001-23-000001", "Revenues", "20230331", 1, 999.0),
    # segment breakout + co-registrant + non-USD + comparative period: all excluded.
    num_line("0001-23-000001", "Revenues", "20230331", 1, 555.0, segments="Region=US;"),
    num_line("0001-23-000001", "Revenues", "20230331", 1, 556.0, coreg="SubCo"),
    num_line("0001-23-000001", "Revenues", "20230331", 1, 557.0, uom="EUR"),
    num_line("0001-23-000001", "Revenues", "20220331", 1, 558.0),  # ddate != period
    num_line("0001-23-000001", "CostOfGoodsAndServicesSold", "20230331", 1, 40.0),
    num_line("0001-23-000001", "Assets", "20230331", 0, 1000.0),
    # 10-Q revenue at annual duration must NOT be picked for a 10-Q.
    num_line("0001-23-000001", "Revenues", "20230331", 4, 400.0),
    # cik 200 10-K: full-year (qtrs=4) facts; a qtrs=1 stray must be ignored.
    num_line("0002-23-000001", "Revenues", "20221231", 4, 480.0),
    num_line("0002-23-000001", "Revenues", "20221231", 1, 120.0),
    num_line("0002-23-000001", "CostOfRevenue", "20221231", 4, 200.0),
    num_line("0002-23-000001", "Assets", "20221231", 0, 2000.0),
    # amendment + wrong form: their facts must never appear.
    num_line("0003-23-000001", "Revenues", "20221231", 4, 111.0),
    num_line("0004-23-000001", "Revenues", "20230331", 1, 222.0),
]


def _facts(tmp_path, ciks=None):
    zip_path = write_quarter_zip(tmp_path / "2023q2.zip", SUBS, NUMS)
    return load_quarter_facts(zip_path, ciks=ciks)


def test_tag_priority_and_consolidated_only(tmp_path):
    facts = _facts(tmp_path)
    row = facts[(facts["cik"] == 100) & (facts["concept"] == "revenue")].iloc[0]
    assert row["tag"] == "RevenueFromContractWithCustomerExcludingAssessedTax"
    assert row["value"] == 100.0
    assert row["qtrs"] == 1


def test_form_duration_selection(tmp_path):
    facts = _facts(tmp_path)
    annual = facts[(facts["cik"] == 200) & (facts["concept"] == "revenue")].iloc[0]
    assert annual["qtrs"] == 4
    assert annual["value"] == 480.0
    assets = facts[(facts["cik"] == 200) & (facts["concept"] == "assets")].iloc[0]
    assert assets["qtrs"] == 0
    assert assets["value"] == 2000.0


def test_amendments_and_other_forms_never_parse(tmp_path):
    facts = _facts(tmp_path)
    assert set(facts["form"]) == {"10-Q", "10-K"}
    assert 300 not in set(facts["cik"])
    assert 400 not in set(facts["cik"])


def test_own_fiscal_period_only(tmp_path):
    facts = _facts(tmp_path)
    q = facts[facts["cik"] == 100]
    assert set(q["period"].dt.strftime("%Y%m%d")) == {"20230331"}
    assert 558.0 not in set(q["value"])


def test_cik_filter_and_dtypes(tmp_path):
    facts = _facts(tmp_path, ciks={100})
    assert set(facts["cik"]) == {100}
    assert list(facts.columns) == FACT_COLUMNS
    assert str(facts["filed"].iloc[0].date()) == "2023-05-10"


def test_no_matching_filings_returns_empty(tmp_path):
    facts = _facts(tmp_path, ciks={999})
    assert facts.empty
    assert list(empty_facts().columns) == FACT_COLUMNS
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_fundamentals_edgar.py -v 2>&1 | tee /tmp/claude-m4-t1.log`
Expected: FAIL — `ModuleNotFoundError: No module named 'trading.fundamentals'`.

- [ ] **Step 4: Implement the parser**

Create `src/trading/fundamentals/__init__.py` (empty file), then `src/trading/fundamentals/edgar.py`:

```python
"""SEC EDGAR Financial Statement Data Set parsing (spec: M4 fundamentals overlay).

Parses one quarterly ZIP (sub.txt + num.txt) into the normalized facts table
(FACT_COLUMNS) that trading.fundamentals.metrics consumes. Pure file parsing:
downloads live in scripts/backfill_fundamentals.py, and the companyfacts JSON
top-up (trading.fundamentals.companyfacts) normalizes to this SAME table so
backfill and top-up cannot diverge.

Tag fallback chains are LOCKED to the xbrl-scout findings (2023q1 census:
the three revenue tags cover 76.5% of 10-K/10-Q filers, the three COGS tags
~48.5%, Assets 99.5%). Consolidated rows only (empty segments + coreg).
Roughly half of filers (banks, insurers) report no COGS concept at all ->
their metric is NaN downstream and ranks neutral (0.5) by design.

Forms accepted are EXACTLY 10-K and 10-Q: amendments (10-K/A, 10-Q/A) never
enter the facts table, which is half of the PIT original-filing discipline
(the other half is metrics' earliest-filed dedup per (cik, fy, fp)).
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

USER_AGENT = "trading-system travis@launchsupply.com"  # mandatory on every SEC request

REVENUE_TAGS = [
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "Revenues",
    "RevenueFromContractWithCustomerIncludingAssessedTax",
]
COGS_TAGS = [
    "CostOfGoodsAndServicesSold",
    "CostOfRevenue",
    "CostOfGoodsAndServiceExcludingDepreciationDepletionAndAmortization",
]
ASSETS_TAGS = ["Assets"]
TAG_PRIORITY: dict[str, list[str]] = {
    "revenue": REVENUE_TAGS,
    "cogs": COGS_TAGS,
    "assets": ASSETS_TAGS,
}

FACT_COLUMNS = [
    "cik",
    "adsh",
    "form",
    "fy",
    "fp",
    "period",
    "filed",
    "concept",
    "tag",
    "qtrs",
    "value",
]

_SUB_COLS = ["adsh", "cik", "form", "period", "fy", "fp", "filed"]
_NUM_COLS = ["adsh", "tag", "ddate", "qtrs", "uom", "segments", "coreg", "value"]


def empty_facts() -> pd.DataFrame:
    return pd.DataFrame(columns=FACT_COLUMNS)


def load_quarter_facts(zip_path: Path, ciks: set[int] | None = None) -> pd.DataFrame:
    """One quarterly ZIP -> normalized facts: per (filing, concept) the single
    best fact by tag priority, consolidated only, the filing's OWN fiscal
    period only (ddate == sub.period), duration matched to the form (10-K
    qtrs=4 full year, 10-Q qtrs=1; balance-sheet Assets qtrs=0 instant)."""
    with zipfile.ZipFile(zip_path) as zf, zf.open("sub.txt") as fh:
        sub = pd.read_csv(fh, sep="\t", dtype=str, usecols=_SUB_COLS, encoding="latin-1")
    sub = sub[sub["form"].isin(["10-K", "10-Q"])]
    sub = sub.dropna(subset=["cik", "period", "fy", "fp", "filed"])
    sub = sub.copy()
    sub["cik"] = sub["cik"].astype(int)
    if ciks is not None:
        sub = sub[sub["cik"].isin(ciks)]
    if sub.empty:
        return empty_facts()

    all_tags = {tag for tags in TAG_PRIORITY.values() for tag in tags}
    wanted_adsh = set(sub["adsh"])
    chunks: list[pd.DataFrame] = []
    # num.txt is ~500MB / ~3.4M rows per quarter: stream in chunks and keep
    # only our filings + tags so the working set stays small.
    with zipfile.ZipFile(zip_path) as zf, zf.open("num.txt") as fh:
        for chunk in pd.read_csv(
            fh, sep="\t", dtype=str, usecols=_NUM_COLS, encoding="latin-1", chunksize=250_000
        ):
            keep = chunk[chunk["adsh"].isin(wanted_adsh) & chunk["tag"].isin(all_tags)]
            if not keep.empty:
                chunks.append(keep)
    if not chunks:
        return empty_facts()
    num = pd.concat(chunks, ignore_index=True)
    for col in ("segments", "coreg"):
        num[col] = num[col].fillna("")
    num = num[
        (num["segments"] == "") & (num["coreg"] == "") & (num["uom"] == "USD") & num["value"].notna()
    ].copy()
    if num.empty:
        return empty_facts()
    num["qtrs"] = num["qtrs"].astype(int)
    num["value"] = num["value"].astype(float)
    num = num.merge(sub, on="adsh", how="inner")
    num = num[num["ddate"] == num["period"]]  # the filing's own fiscal period only

    parts: list[pd.DataFrame] = []
    for concept, tags in TAG_PRIORITY.items():
        sel = num[num["tag"].isin(tags)].copy()
        if concept == "assets":
            sel = sel[sel["qtrs"] == 0]
        else:
            sel = sel[sel["qtrs"] == np.where(sel["form"] == "10-K", 4, 1)]
        if sel.empty:
            continue
        sel["_priority"] = sel["tag"].map({t: i for i, t in enumerate(tags)})
        sel = sel.sort_values(["adsh", "_priority"], kind="mergesort")
        sel = sel.drop_duplicates("adsh", keep="first")
        sel["concept"] = concept
        parts.append(sel)
    if not parts:
        return empty_facts()
    facts = pd.concat(parts, ignore_index=True)
    facts["period"] = pd.to_datetime(facts["period"], format="%Y%m%d")
    facts["filed"] = pd.to_datetime(facts["filed"], format="%Y%m%d")
    return facts[FACT_COLUMNS].reset_index(drop=True)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_fundamentals_edgar.py -v`
Expected: 6 PASS.

- [ ] **Step 6: Full suite + lint, then commit**

```bash
uv run ruff check . && uv run ruff format .
uv run pytest -q
git add src/trading/fundamentals tests/fundamentals_helpers.py tests/test_fundamentals_edgar.py
git commit -m "Add EDGAR quarterly-ZIP facts parser for the fundamentals overlay [AI]

Normalized facts table (locked tag fallback chains, consolidated rows only,
forms exactly 10-K/10-Q so amendments never parse), streamed num.txt reads.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01WuC8FbGLPEZUjYmdPSMMdv"
```

Expected: 334 + 6 = 340 tests green.

---

### Task 2: Point-in-time gross-profitability series

**Files:**
- Create: `src/trading/fundamentals/metrics.py`
- Test: `tests/test_fundamentals_metrics.py`

**Interfaces:**
- Consumes: `FACT_COLUMNS` facts frames (Task 1 helpers `filing_facts`/`facts_frame`).
- Produces: `SERIES_COLUMNS: list[str]` = `["gross_profitability", "revenue_ttm", "cogs_ttm", "assets", "adsh", "form", "fy", "fp", "period", "revenue_tag", "cogs_tag", "assets_tag"]`; `PROVENANCE_COLUMNS`; `MAX_TTM_SPAN_DAYS = 330`; `empty_series() -> pd.DataFrame`; `compute_pit_series(facts: pd.DataFrame) -> dict[int, pd.DataFrame]` (per-cik frame indexed by tz-aware UTC FILED dates, name `"filed"`, sorted, unique). Tasks 3, 5, 6 consume these.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_fundamentals_metrics.py`:

```python
import math

import pandas as pd
import pytest

from fundamentals_helpers import facts_frame, filing_facts
from trading.fundamentals.metrics import SERIES_COLUMNS, compute_pit_series, empty_series

CIK = 100


def _year_of_filings() -> list[dict]:
    """Q1-Q3 2023 10-Qs, the FY2023 10-K, then Q1 2024 — a full TTM ramp."""
    rows = []
    rows += filing_facts(CIK, "a-01", "10-Q", "2023", "Q1", "2023-03-31", "2023-05-10",
                         revenue=10.0, cogs=4.0, assets=90.0)
    rows += filing_facts(CIK, "a-02", "10-Q", "2023", "Q2", "2023-06-30", "2023-08-09",
                         revenue=12.0, cogs=5.0, assets=95.0)
    rows += filing_facts(CIK, "a-03", "10-Q", "2023", "Q3", "2023-09-30", "2023-11-08",
                         revenue=11.0, cogs=5.0, assets=98.0)
    rows += filing_facts(CIK, "a-04", "10-K", "2023", "FY", "2023-12-31", "2024-02-20",
                         revenue=48.0, cogs=20.0, assets=100.0)
    rows += filing_facts(CIK, "a-05", "10-Q", "2024", "Q1", "2024-03-31", "2024-05-09",
                         revenue=14.0, cogs=6.0, assets=110.0)
    return rows


def test_ttm_incomplete_is_nan_then_completes_at_the_10k():
    series = compute_pit_series(facts_frame(_year_of_filings()))[CIK]
    assert list(series.columns) == SERIES_COLUMNS
    assert series.index.tz is not None and series.index.name == "filed"
    # First three filings: fewer than 4 known quarters -> NaN metric, row still present.
    for filed in ("2023-05-10", "2023-08-09", "2023-11-08"):
        assert math.isnan(series.loc[pd.Timestamp(filed, tz="UTC"), "gross_profitability"])
    # 10-K: Q4 derived = FY - (Q1+Q2+Q3) -> rev 48-33=15, cogs 20-14=6; TTM = FY.
    at_10k = series.loc[pd.Timestamp("2024-02-20", tz="UTC")]
    assert at_10k["revenue_ttm"] == 48.0
    assert at_10k["cogs_ttm"] == 20.0
    assert at_10k["gross_profitability"] == pytest.approx((48.0 - 20.0) / 100.0)


def test_ttm_rolls_forward_at_the_next_quarter():
    series = compute_pit_series(facts_frame(_year_of_filings()))[CIK]
    # Q2'23 + Q3'23 + derived Q4'23 + Q1'24: rev 12+11+15+14, cogs 5+5+6+6.
    row = series.loc[pd.Timestamp("2024-05-09", tz="UTC")]
    assert row["revenue_ttm"] == pytest.approx(52.0)
    assert row["cogs_ttm"] == pytest.approx(22.0)
    assert row["gross_profitability"] == pytest.approx(30.0 / 110.0)


def test_restatement_never_rewrites_history():
    # Same (cik, fy, fp) filed twice: the ORIGINAL (earliest-filed) wins; the
    # later re-filing is discarded entirely -- no row at its filed date, and
    # every later TTM uses the original value.
    rows = _year_of_filings()
    rows += filing_facts(CIK, "a-99", "10-Q", "2023", "Q1", "2023-03-31", "2023-09-01",
                         revenue=999.0, cogs=999.0, assets=999.0)
    series = compute_pit_series(facts_frame(rows))[CIK]
    assert pd.Timestamp("2023-09-01", tz="UTC") not in series.index
    assert series.loc[pd.Timestamp("2024-02-20", tz="UTC"), "revenue_ttm"] == 48.0
    assert "a-99" not in set(series["adsh"])


def test_earliest_accession_breaks_a_same_day_tie():
    rows = filing_facts(CIK, "b-02", "10-Q", "2023", "Q1", "2023-03-31", "2023-05-10",
                        revenue=999.0, cogs=1.0, assets=50.0)
    rows += filing_facts(CIK, "b-01", "10-Q", "2023", "Q1", "2023-03-31", "2023-05-10",
                         revenue=10.0, cogs=4.0, assets=90.0)
    series = compute_pit_series(facts_frame(rows))[CIK]
    assert list(series["adsh"]) == ["b-01"]


def test_10k_without_all_three_prior_quarters_gives_nan_q4():
    rows = []
    rows += filing_facts(CIK, "c-01", "10-Q", "2023", "Q1", "2023-03-31", "2023-05-10",
                         revenue=10.0, cogs=4.0, assets=90.0)
    # Q2 missing entirely.
    rows += filing_facts(CIK, "c-03", "10-Q", "2023", "Q3", "2023-09-30", "2023-11-08",
                         revenue=11.0, cogs=5.0, assets=98.0)
    rows += filing_facts(CIK, "c-04", "10-K", "2023", "FY", "2023-12-31", "2024-02-20",
                         revenue=48.0, cogs=20.0, assets=100.0)
    series = compute_pit_series(facts_frame(rows))[CIK]
    assert math.isnan(series.loc[pd.Timestamp("2024-02-20", tz="UTC"), "gross_profitability"])


def test_ragged_quarter_window_is_nan():
    # Four known quarters but with a year gap inside: span > 330 days -> NaN.
    rows = []
    rows += filing_facts(CIK, "d-01", "10-Q", "2022", "Q3", "2022-09-30", "2022-11-08",
                         revenue=10.0, cogs=4.0, assets=90.0)
    rows += filing_facts(CIK, "d-02", "10-Q", "2023", "Q1", "2023-03-31", "2023-05-10",
                         revenue=10.0, cogs=4.0, assets=90.0)
    rows += filing_facts(CIK, "d-03", "10-Q", "2023", "Q2", "2023-06-30", "2023-08-09",
                         revenue=12.0, cogs=5.0, assets=95.0)
    rows += filing_facts(CIK, "d-04", "10-Q", "2023", "Q3", "2023-09-30", "2023-11-08",
                         revenue=11.0, cogs=5.0, assets=98.0)
    series = compute_pit_series(facts_frame(rows))[CIK]
    assert math.isnan(series.loc[pd.Timestamp("2023-11-08", tz="UTC"), "gross_profitability"])


def test_missing_cogs_gives_nan_metric_with_provenance():
    rows = _year_of_filings()
    facts = facts_frame(rows)
    facts = facts[~((facts["concept"] == "cogs") & (facts["adsh"] == "a-04"))]
    series = compute_pit_series(facts)[CIK]
    at_10k = series.loc[pd.Timestamp("2024-02-20", tz="UTC")]
    assert math.isnan(at_10k["gross_profitability"])
    assert at_10k["cogs_tag"] == ""
    assert at_10k["adsh"] == "a-04"
    assert at_10k["form"] == "10-K"
    assert at_10k["period"] == "2023-12-31"


def test_empty_facts_and_empty_series_shapes():
    assert compute_pit_series(facts_frame([])) == {}
    frame = empty_series()
    assert list(frame.columns) == SERIES_COLUMNS
    assert frame.index.tz is not None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_fundamentals_metrics.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'trading.fundamentals.metrics'`.

- [ ] **Step 3: Implement the metric**

Create `src/trading/fundamentals/metrics.py`:

```python
"""Point-in-time gross-profitability series (spec: M4 fundamentals overlay).

Metric v1: gross profitability = trailing-4-quarter (Revenue - COGS) / latest
Assets (Novy-Marx). PIT discipline (non-negotiable):

- A value becomes VISIBLE at its FILING date, never its fiscal-period date
  (2023q1 census: filing lag min 19d / median 44d / max 59d -- a period-date
  join would silently look ahead across that gap).
- Per (cik, fy, fp) only the ORIGINAL filing counts: earliest filed, tie-break
  lowest accession. Later re-filings/restatements never rewrite history
  (amendment forms never even reach this module -- edgar.py accepts exactly
  10-K / 10-Q).
- Incomplete trailing-4-quarter window -> NaN metric. The row is still
  emitted with provenance so "a filing happened, metric unknown" stays
  visible downstream; the ranker reads the LAST row as-of a session (a step
  function on FILED dates), so a NaN latest filing means neutral, never a
  silent reach-back to a stale value.
"""

from __future__ import annotations

import math

import pandas as pd

TTM_QUARTERS = 4
# Four consecutive fiscal quarter-ends span ~273 calendar days; the slack
# absorbs 53-week fiscal years and shifted quarter-ends. Beyond it the
# window has a hole (skipped quarter) and the TTM sum would be wrong.
MAX_TTM_SPAN_DAYS = 330

PROVENANCE_COLUMNS = [
    "adsh",
    "form",
    "fy",
    "fp",
    "period",
    "revenue_tag",
    "cogs_tag",
    "assets_tag",
]
SERIES_COLUMNS = ["gross_profitability", "revenue_ttm", "cogs_ttm", "assets", *PROVENANCE_COLUMNS]


def empty_series() -> pd.DataFrame:
    frame = pd.DataFrame(columns=SERIES_COLUMNS)
    frame.index = pd.DatetimeIndex([], tz="UTC", name="filed")
    return frame


def compute_pit_series(facts: pd.DataFrame) -> dict[int, pd.DataFrame]:
    """Normalized facts (edgar.FACT_COLUMNS) -> per-cik series indexed by
    tz-aware UTC FILED date. Safe to feed overlapping quarterly files: facts
    are deduped on (adsh, concept) first, then per (cik, fy, fp) only the
    original filing survives."""
    if facts.empty:
        return {}
    facts = facts.drop_duplicates(["adsh", "concept"], keep="first")
    facts = _original_filings(facts)
    out: dict[int, pd.DataFrame] = {}
    for cik, group in facts.groupby("cik"):
        series = _cik_series(group)
        if not series.empty:
            out[int(cik)] = series
    return out


def _original_filings(facts: pd.DataFrame) -> pd.DataFrame:
    """Earliest-filed filing per (cik, fy, fp) wins, tie-break lowest adsh:
    the ORIGINAL filing's values are frozen and later filings for the same
    fiscal period are discarded entirely (PIT: history never rewrites)."""
    filings = (
        facts[["cik", "adsh", "fy", "fp", "filed"]]
        .drop_duplicates("adsh")
        .sort_values(["filed", "adsh"], kind="mergesort")
    )
    keep = set(filings.drop_duplicates(["cik", "fy", "fp"], keep="first")["adsh"])
    return facts[facts["adsh"].isin(keep)]


def _derive_q4(fy_total: float, q123: list[dict | None], key: str) -> float:
    if math.isnan(fy_total) or any(q is None or math.isnan(q[key]) for q in q123):
        return math.nan
    return fy_total - sum(q[key] for q in q123)


def _ttm(quarters: dict[tuple[str, str], dict]) -> tuple[float, float]:
    known = sorted(quarters.values(), key=lambda q: q["period"])
    last4 = known[-TTM_QUARTERS:]
    if len(last4) < TTM_QUARTERS:
        return math.nan, math.nan
    if (last4[-1]["period"] - last4[0]["period"]).days > MAX_TTM_SPAN_DAYS:
        return math.nan, math.nan  # ragged window: a quarter is missing inside
    # NaN quarter values propagate through sum() -> NaN TTM by design.
    return sum(q["revenue"] for q in last4), sum(q["cogs"] for q in last4)


def _cik_series(group: pd.DataFrame) -> pd.DataFrame:
    filings: dict[str, dict] = {}
    for row in group.itertuples():
        filing = filings.setdefault(
            row.adsh,
            {
                "adsh": row.adsh,
                "form": row.form,
                "fy": row.fy,
                "fp": row.fp,
                "period": row.period,
                "filed": row.filed,
                "revenue": math.nan,
                "cogs": math.nan,
                "assets": math.nan,
                "revenue_tag": "",
                "cogs_tag": "",
                "assets_tag": "",
            },
        )
        filing[row.concept] = row.value
        filing[f"{row.concept}_tag"] = row.tag

    quarters: dict[tuple[str, str], dict] = {}  # (fy, fp) -> single-quarter values
    rows: list[dict] = []
    for f in sorted(filings.values(), key=lambda f: (f["filed"], f["adsh"])):
        if f["form"] == "10-Q":
            quarters[(f["fy"], f["fp"])] = {
                "period": f["period"],
                "revenue": f["revenue"],
                "cogs": f["cogs"],
            }
        else:  # 10-K reports the FULL year (qtrs=4): derive Q4 = FY - (Q1+Q2+Q3)
            q123 = [quarters.get((f["fy"], fp)) for fp in ("Q1", "Q2", "Q3")]
            quarters[(f["fy"], "Q4")] = {
                "period": f["period"],
                "revenue": _derive_q4(f["revenue"], q123, "revenue"),
                "cogs": _derive_q4(f["cogs"], q123, "cogs"),
            }
        revenue_ttm, cogs_ttm = _ttm(quarters)
        assets = f["assets"]
        gp = math.nan
        if not math.isnan(revenue_ttm) and not math.isnan(cogs_ttm) and assets and not math.isnan(assets):
            gp = (revenue_ttm - cogs_ttm) / assets
        rows.append(
            {
                "filed": f["filed"],
                "gross_profitability": gp,
                "revenue_ttm": revenue_ttm,
                "cogs_ttm": cogs_ttm,
                "assets": assets,
                "adsh": f["adsh"],
                "form": f["form"],
                "fy": f["fy"],
                "fp": f["fp"],
                "period": f["period"].date().isoformat(),
                "revenue_tag": f["revenue_tag"],
                "cogs_tag": f["cogs_tag"],
                "assets_tag": f["assets_tag"],
            }
        )
    frame = pd.DataFrame(rows).set_index("filed")
    frame.index = frame.index.tz_localize("UTC")
    frame.index.name = "filed"
    frame = frame.sort_index(kind="mergesort")
    # Two filings on one day (10-K + 10-Q together): the last row already
    # reflects BOTH filings' quarters; keep it, drop the interim duplicate.
    frame = frame[~frame.index.duplicated(keep="last")]
    return frame[SERIES_COLUMNS]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_fundamentals_metrics.py tests/test_fundamentals_edgar.py -v`
Expected: all PASS.

- [ ] **Step 5: Full suite + lint, then commit**

```bash
uv run ruff check . && uv run ruff format .
uv run pytest -q
git add src/trading/fundamentals/metrics.py tests/test_fundamentals_metrics.py
git commit -m "Add PIT gross-profitability TTM computation [AI]

Values visible at FILING date; original filing per (cik, fy, fp) frozen
(earliest filed, adsh tie-break); 10-K Q4 derived by subtraction; incomplete
or ragged TTM window -> NaN; provenance on every row.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01WuC8FbGLPEZUjYmdPSMMdv"
```

---

### Task 3: Append-only fundamentals store

**Files:**
- Create: `src/trading/fundamentals/store.py`
- Test: `tests/test_fundamentals_store.py`

**Interfaces:**
- Consumes: `empty_series`, `SERIES_COLUMNS` from `trading.fundamentals.metrics` (Task 2).
- Produces: `FundamentalsStore(root: Path)` with `path_for(symbol) -> Path`, `read(symbol) -> pd.DataFrame`, `append(symbol, rows: pd.DataFrame) -> int` (rows actually added), `load(symbols: Iterable[str]) -> dict[str, pd.DataFrame]` (non-empty frames only), `last_refresh() -> datetime.date | None`, `mark_refreshed(day: datetime.date) -> None`. Tasks 5, 6, 9, 10 consume these.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_fundamentals_store.py`:

```python
import datetime

import pandas as pd
import pytest

from trading.fundamentals.metrics import SERIES_COLUMNS
from trading.fundamentals.store import FundamentalsStore


def _rows(dated: dict[str, float]) -> pd.DataFrame:
    idx = pd.DatetimeIndex([pd.Timestamp(d, tz="UTC") for d in dated], name="filed")
    frame = pd.DataFrame({"gross_profitability": list(dated.values())}, index=idx)
    for col in SERIES_COLUMNS:
        if col not in frame.columns:
            frame[col] = "" if col in ("adsh", "form", "fy", "fp", "period",
                                       "revenue_tag", "cogs_tag", "assets_tag") else 0.0
    return frame[SERIES_COLUMNS]


def test_read_missing_symbol_is_empty_with_utc_index(tmp_path):
    store = FundamentalsStore(tmp_path)
    frame = store.read("AAPL")
    assert frame.empty
    assert list(frame.columns) == SERIES_COLUMNS
    assert frame.index.tz is not None


def test_append_then_read_round_trips(tmp_path):
    store = FundamentalsStore(tmp_path)
    added = store.append("AAPL", _rows({"2023-02-03": 0.48, "2023-05-05": 0.47}))
    assert added == 2
    frame = store.read("AAPL")
    assert list(frame["gross_profitability"]) == [0.48, 0.47]
    assert not store.path_for("AAPL").with_suffix(".parquet.tmp").exists()  # atomic write


def test_history_is_immutable_existing_filed_dates_never_rewrite(tmp_path):
    store = FundamentalsStore(tmp_path)
    store.append("AAPL", _rows({"2023-02-03": 0.48}))
    # Re-appending the same filed date with a DIFFERENT value must be a no-op:
    # fundamentals history is append-only by design (no OhlcvCache-style
    # trailing refetch).
    added = store.append("AAPL", _rows({"2023-02-03": 0.99, "2023-05-05": 0.47}))
    assert added == 1
    frame = store.read("AAPL")
    assert frame.loc[pd.Timestamp("2023-02-03", tz="UTC"), "gross_profitability"] == 0.48


def test_append_rejects_naive_index(tmp_path):
    store = FundamentalsStore(tmp_path)
    naive = _rows({"2023-02-03": 0.48})
    naive.index = naive.index.tz_localize(None)
    with pytest.raises(ValueError, match="tz-aware"):
        store.append("AAPL", naive)


def test_append_empty_is_noop(tmp_path):
    store = FundamentalsStore(tmp_path)
    assert store.append("AAPL", _rows({})) == 0
    assert not store.path_for("AAPL").exists()


def test_load_returns_only_symbols_with_rows(tmp_path):
    store = FundamentalsStore(tmp_path)
    store.append("AAPL", _rows({"2023-02-03": 0.48}))
    loaded = store.load(["AAPL", "MSFT"])
    assert set(loaded) == {"AAPL"}


def test_refresh_marker_round_trips(tmp_path):
    store = FundamentalsStore(tmp_path)
    assert store.last_refresh() is None
    store.mark_refreshed(datetime.date(2026, 7, 6))
    assert store.last_refresh() == datetime.date(2026, 7, 6)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_fundamentals_store.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'trading.fundamentals.store'`.

- [ ] **Step 3: Implement the store**

Create `src/trading/fundamentals/store.py`:

```python
"""Append-only per-symbol parquet store for fundamentals (spec: M4).

Layout: <root>/<SYMBOL>.parquet, tz-aware UTC DatetimeIndex = FILED dates,
metrics.SERIES_COLUMNS. Fundamentals history is IMMUTABLE: append() never
overwrites an existing filed-date row, so a later restated value can never
replace what was visible at the time. OhlcvCache's trailing-refetch model
deliberately does NOT apply here; its atomic tmp+os.replace write and
cache-through reads do.

<root>/.last_refresh (ISO date) records the last companyfacts top-up; the
runner's weekly refresh gate reads it (see trading.runner).
"""

from __future__ import annotations

import datetime
import os
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from trading.fundamentals.metrics import empty_series


class FundamentalsStore:
    def __init__(self, root: Path):
        self._root = root
        self._root.mkdir(parents=True, exist_ok=True)

    def path_for(self, symbol: str) -> Path:
        return self._root / f"{symbol.replace('/', '-')}.parquet"

    def read(self, symbol: str) -> pd.DataFrame:
        path = self.path_for(symbol)
        if not path.exists():
            return empty_series()
        return pd.read_parquet(path)

    def append(self, symbol: str, rows: pd.DataFrame) -> int:
        """Add rows whose FILED date is not already stored; existing rows are
        never touched. Returns the number of rows actually added."""
        if rows.empty:
            return 0
        if rows.index.tz is None:
            raise ValueError(f"{symbol}: fundamentals index must be tz-aware UTC filed dates")
        existing = self.read(symbol)
        fresh = rows[~rows.index.isin(existing.index)]
        if fresh.empty:
            return 0
        if existing.empty:
            # Never concat an empty frame (pandas 2.x FutureWarning, and the
            # suite runs warnings-as-errors).
            merged = fresh.sort_index(kind="mergesort")
        else:
            merged = pd.concat([existing, fresh]).sort_index(kind="mergesort")
        path = self.path_for(symbol)
        tmp = path.with_suffix(".parquet.tmp")
        merged.to_parquet(tmp)
        os.replace(tmp, path)  # atomic: never leave a torn store file
        return len(fresh)

    def load(self, symbols: Iterable[str]) -> dict[str, pd.DataFrame]:
        out: dict[str, pd.DataFrame] = {}
        for symbol in symbols:
            frame = self.read(symbol)
            if not frame.empty:
                out[symbol] = frame
        return out

    def last_refresh(self) -> datetime.date | None:
        path = self._root / ".last_refresh"
        if not path.exists():
            return None
        return datetime.date.fromisoformat(path.read_text().strip())

    def mark_refreshed(self, day: datetime.date) -> None:
        path = self._root / ".last_refresh"
        tmp = path.with_suffix(".tmp")
        tmp.write_text(day.isoformat())
        os.replace(tmp, path)
```

Note: `_rows({})` in the tests builds an empty frame whose index is a `DatetimeIndex` with tz — `append` returns 0 before the tz check matters, and `pd.DataFrame({...: []}, index=...)` keeps the SERIES_COLUMNS shape via the helper's column fill.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_fundamentals_store.py -v`
Expected: 7 PASS.

- [ ] **Step 5: Full suite + lint, then commit**

```bash
uv run ruff check . && uv run ruff format .
uv run pytest -q
git add src/trading/fundamentals/store.py tests/test_fundamentals_store.py
git commit -m "Add append-only per-symbol fundamentals parquet store [AI]

Immutable filed-date history (existing rows never rewritten), atomic writes,
cache-through reads, weekly-refresh marker.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01WuC8FbGLPEZUjYmdPSMMdv"
```

---

### Task 4: CIK↔symbol point-in-time interval map (committed artifact)

This task runs a network script once to generate a committed CSV (same pattern as `scripts/build_pit_membership.py` → `equities_membership.csv`). Tests run offline against the committed artifact.

**Files:**
- Create: `scripts/build_cik_map.py`
- Create: `src/trading/fundamentals/cik_map.py`
- Create: `src/trading/fundamentals/cik_map.csv` (generated by the script, then committed)
- Modify: `src/trading/venues/universes/sources/PROVENANCE.md` (append entry)
- Test: `tests/test_fundamentals_cik_map.py`

**Interfaces:**
- Consumes: `src/trading/venues/universes/equities_membership.csv` (committed, M3), `https://www.sec.gov/files/company_tickers.json` (network, once).
- Produces: `DEFAULT_CIK_MAP_CSV: Path`; `load_cik_map(path: Path | None = None) -> pd.DataFrame` (columns `symbol` str, `cik` int, `start` str, `end` str; empty end = current); `cik_for(cik_map: pd.DataFrame, symbol: str, as_of: datetime.date) -> int | None`; `interval_slice(frame: pd.DataFrame, start: str, end: str) -> pd.DataFrame`. Tasks 5, 6, 10, 11 consume these.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_fundamentals_cik_map.py`:

```python
import datetime

import pandas as pd

from trading.fundamentals.cik_map import cik_for, interval_slice, load_cik_map

MAP = load_cik_map()  # the committed artifact


def test_committed_map_shape():
    assert list(MAP.columns) == ["symbol", "cik", "start", "end"]
    assert MAP["cik"].dtype.kind == "i"
    assert len(MAP) > 400  # roughly the current sp500+ndx membership, plus history


def test_fb_meta_one_cik_across_the_rename():
    fb = MAP[MAP["symbol"] == "FB"].iloc[0]
    meta = MAP[MAP["symbol"] == "META"].iloc[0]
    assert fb["cik"] == meta["cik"] == 1326801
    assert fb["end"] == "2022-06-09"
    assert meta["start"] == "2022-06-09"
    # PIT lookup: before the rename FB resolves and META does not; after, vice versa.
    assert cik_for(MAP, "FB", datetime.date(2022, 6, 8)) == 1326801
    assert cik_for(MAP, "META", datetime.date(2022, 6, 8)) is None
    assert cik_for(MAP, "META", datetime.date(2022, 6, 9)) == 1326801
    assert cik_for(MAP, "FB", datetime.date(2022, 6, 9)) is None


def test_abc_cor_one_cik_across_the_rename():
    abc = MAP[MAP["symbol"] == "ABC"].iloc[0]
    cor = MAP[MAP["symbol"] == "COR"].iloc[0]
    assert abc["cik"] == cor["cik"] == 1140859


def test_unknown_symbol_resolves_to_none():
    assert cik_for(MAP, "NOSUCHTICKER", datetime.date(2024, 1, 1)) is None


def test_interval_slice_is_start_inclusive_end_exclusive():
    idx = pd.DatetimeIndex(
        [pd.Timestamp(d, tz="UTC") for d in ("2022-06-08", "2022-06-09", "2022-07-01")]
    )
    frame = pd.DataFrame({"gross_profitability": [1.0, 2.0, 3.0]}, index=idx)
    before = interval_slice(frame, "2017-01-01", "2022-06-09")
    after = interval_slice(frame, "2022-06-09", "")
    assert list(before["gross_profitability"]) == [1.0]
    assert list(after["gross_profitability"]) == [2.0, 3.0]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_fundamentals_cik_map.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'trading.fundamentals.cik_map'`.

- [ ] **Step 3: Implement the loader**

Create `src/trading/fundamentals/cik_map.py`:

```python
"""CIK <-> symbol point-in-time interval map (spec: M4 fundamentals overlay).

cik_map.csv is a COMMITTED artifact (symbol,cik,start,end; start inclusive,
end exclusive, empty end = current) generated by scripts/build_cik_map.py --
treat as frozen data, regenerate deliberately and review the diff (sources +
validation: src/trading/venues/universes/sources/PROVENANCE.md).

A fundamentals row attaches to the symbol whose interval contains the row's
FILED date: a renamed company's pre-rename filings land on the old symbol
(FB) and post-rename filings on the new one (META) while both share one
CIK's TTM continuity. Membership symbols with NO row here (mostly acquired/
delisted companies EDGAR no longer lists a ticker for) simply get no
fundamentals and rank neutral (0.5) -- fail-open by design.
"""

from __future__ import annotations

import datetime
from pathlib import Path

import pandas as pd

DEFAULT_CIK_MAP_CSV = Path(__file__).parent / "cik_map.csv"


def load_cik_map(path: Path | None = None) -> pd.DataFrame:
    df = pd.read_csv(path or DEFAULT_CIK_MAP_CSV, comment="#", dtype=str).fillna("")
    df["cik"] = df["cik"].astype(int)
    return df


def cik_for(cik_map: pd.DataFrame, symbol: str, as_of: datetime.date) -> int | None:
    iso = as_of.isoformat()
    rows = cik_map[
        (cik_map["symbol"] == symbol)
        & (cik_map["start"] <= iso)
        & ((cik_map["end"] == "") | (iso < cik_map["end"]))
    ]
    return int(rows.iloc[0]["cik"]) if len(rows) else None


def interval_slice(frame: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """Rows of a filed-indexed frame whose FILED date falls in [start, end)."""
    out = frame[frame.index >= pd.Timestamp(start, tz="UTC")]
    if end:
        out = out[out.index < pd.Timestamp(end, tz="UTC")]
    return out
```

- [ ] **Step 4: Implement the build script**

Create `scripts/build_cik_map.py`:

```python
"""Build the committed CIK<->symbol point-in-time interval map
(src/trading/fundamentals/cik_map.csv) for the M4 fundamentals overlay.

Sources: SEC company_tickers.json (current ticker -> CIK, fetched live with
the mandatory User-Agent) + the reviewed RENAMES table below (membership
symbols that changed ticker since 2017, cross-checked against the membership
CSV's remove/add dates) + src/trading/venues/universes/equities_membership.csv
(whose symbols define what needs mapping).

Primary-class selection is implicit: only symbols present in the membership
CSV are emitted, so a CIK's non-member share classes never appear.

Self-validating: aborts unless FB and META resolve to one CIK (1326801) with
the boundary at 2022-06-09, ABC and COR share CIK 1140859, and >= 95% of
CURRENT members map. Unmapped symbols (mostly acquired/delisted companies
EDGAR no longer lists a ticker for) are printed: they get no fundamentals and
rank neutral -- extend RENAMES deliberately if one matters. Update
src/trading/venues/universes/sources/PROVENANCE.md on every regeneration.

Usage: uv run python scripts/build_cik_map.py
"""

from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MEMBERSHIP = ROOT / "src" / "trading" / "venues" / "universes" / "equities_membership.csv"
OUTPUT = ROOT / "src" / "trading" / "fundamentals" / "cik_map.csv"
TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
USER_AGENT = "trading-system travis@launchsupply.com"
SINCE = "2017-01-01"  # same pad as the membership build

# (old_symbol, new_symbol, change_date). Reviewed ticker renames among 2017+
# membership symbols; the boundary date decides which symbol a filing filed
# near it attaches to, so day-exactness is low-stakes but should match the
# membership CSV's remove/add transition. Chains (A->B->C) are supported.
RENAMES = [
    ("DWDP", "DD", "2019-06-03"),
    ("HRS", "LHX", "2019-07-01"),
    ("UTX", "RTX", "2020-04-03"),
    ("MYL", "VTRS", "2020-11-16"),
    ("WLTW", "WTW", "2022-01-05"),
    ("FB", "META", "2022-06-09"),
    ("ANTM", "ELV", "2022-06-28"),
    ("PKI", "RVTY", "2023-05-16"),
    ("FISV", "FI", "2023-06-06"),
    ("RE", "EG", "2023-07-10"),
    ("ABC", "COR", "2023-08-30"),
]


def normalize(symbol: str) -> str:
    return str(symbol).strip().upper().replace(".", "-")


def fetch_company_tickers() -> dict[str, int]:
    req = urllib.request.Request(TICKERS_URL, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as resp:
        raw = json.loads(resp.read().decode("utf-8"))
    return {normalize(v["ticker"]): int(v["cik_str"]) for v in raw.values()}


def membership_symbols() -> tuple[set[str], set[str]]:
    """(all symbols ever in the membership CSV, symbols currently a member)."""
    df = pd.read_csv(MEMBERSHIP, comment="#", dtype=str).fillna("")
    current = set(df.loc[df["end"] == "", "symbol"])
    return set(df["symbol"]), current


def build_rows(
    symbols: set[str], current_tickers: dict[str, int]
) -> tuple[list[tuple[str, int, str, str]], list[str]]:
    new_by_old = {old: (new, date) for old, new, date in RENAMES}
    renamed_starts = {new: date for _, new, date in RENAMES}
    rows: list[tuple[str, int, str, str]] = []
    unmapped: list[str] = []
    for symbol in sorted(symbols):
        # Follow the rename chain forward until we hit a current EDGAR ticker.
        cursor, end, seen = symbol, "", set()
        while cursor not in current_tickers:
            if cursor in seen or cursor not in new_by_old:
                cursor = None
                break
            seen.add(cursor)
            new, date = new_by_old[cursor]
            if end == "":
                end = date  # this symbol stopped being the live ticker here
            cursor = new
        if cursor is None:
            unmapped.append(symbol)
            continue
        start = max(SINCE, renamed_starts.get(symbol, SINCE))
        rows.append((symbol, current_tickers[cursor], start, end))
    return rows, unmapped


def validate(rows: list[tuple[str, int, str, str]], current_members: set[str]) -> None:
    by_symbol = {s: (cik, start, end) for s, cik, start, end in rows}
    fb, meta = by_symbol.get("FB"), by_symbol.get("META")
    if (
        fb is None
        or meta is None
        or fb[0] != 1326801
        or meta[0] != 1326801
        or fb[2] != "2022-06-09"
        or meta[1] != "2022-06-09"
    ):
        sys.exit(f"FATAL: FB/META rename mapping wrong: FB={fb}, META={meta}")
    abc, cor = by_symbol.get("ABC"), by_symbol.get("COR")
    if abc is None or cor is None or abc[0] != 1140859 or cor[0] != 1140859:
        sys.exit(f"FATAL: ABC/COR rename mapping wrong: ABC={abc}, COR={cor}")
    mapped_current = current_members & set(by_symbol)
    ratio = len(mapped_current) / len(current_members)
    if ratio < 0.95:
        sys.exit(f"FATAL: only {ratio:.1%} of current members map to a CIK (need >= 95%)")
    print(f"validation OK: {ratio:.1%} of {len(current_members)} current members mapped")


def main() -> None:
    symbols, current_members = membership_symbols()
    rows, unmapped = build_rows(symbols, fetch_company_tickers())
    validate(rows, current_members)
    lines = [
        "# CIK<->symbol point-in-time intervals. GENERATED by scripts/build_cik_map.py",
        "# Sources + validation: see src/trading/venues/universes/sources/PROVENANCE.md.",
        "# start inclusive, end exclusive, empty end = current EDGAR ticker.",
        "symbol,cik,start,end",
    ]
    lines += [f"{s},{cik},{start},{end}" for s, cik, start, end in rows]
    OUTPUT.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUTPUT} ({len(rows)} intervals; {len(unmapped)} membership symbols unmapped)")
    print("unmapped (no fundamentals -> neutral quality): " + ", ".join(unmapped))


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run the script (network) and eyeball the output**

Run: `uv run python scripts/build_cik_map.py 2>&1 | tee /tmp/claude-m4-cikmap.log`
Expected: `validation OK: ...% of ~5xx current members mapped`, `wrote .../cik_map.csv (...)`, and an unmapped list consisting of acquired/delisted names (e.g. ATVI, TWTR-era symbols). If validation FATALs on FB/META or ABC/COR, fix the RENAMES row (dates/spellings) — do not weaken the check. If the mapped ratio is below 95%, inspect the unmapped list: current members missing from `company_tickers.json` usually mean a rename this table lacks — add the rename and rerun.

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run pytest tests/test_fundamentals_cik_map.py -v`
Expected: 5 PASS (they read the CSV written in Step 5).

- [ ] **Step 7: Append the PROVENANCE.md entry**

Append to `src/trading/venues/universes/sources/PROVENANCE.md`:

```markdown
## CIK <-> symbol point-in-time map (M4 fundamentals)
- Output: `src/trading/fundamentals/cik_map.csv` — regenerated only by
  `uv run python scripts/build_cik_map.py`; treat as frozen data, review diffs.
- Sources: SEC `company_tickers.json` (https://www.sec.gov/files/company_tickers.json,
  public domain, retrieved <RUN DATE> UTC) + the script's reviewed RENAMES
  table (ticker renames among membership symbols, boundary dates cross-checked
  against the membership CSV's remove/add rows) + `equities_membership.csv`
  symbols (which define what needs mapping).
- Validation (asserted by the script): FB and META share CIK 1326801 with the
  interval boundary at the 2022-06-09 rename; ABC and COR share CIK 1140859;
  >= 95% of current members map.
- Known limitation: acquired/delisted symbols absent from company_tickers.json
  and the RENAMES table are unmapped -> no fundamentals -> neutral (0.5)
  quality percentile. The build prints the list; extend RENAMES deliberately.
```

Replace `<RUN DATE>` with the actual run date and record the validation line the script printed.

- [ ] **Step 8: Full suite + lint, then commit**

```bash
uv run ruff check . && uv run ruff format .
uv run pytest -q
git add scripts/build_cik_map.py src/trading/fundamentals/cik_map.py \
        src/trading/fundamentals/cik_map.csv \
        src/trading/venues/universes/sources/PROVENANCE.md \
        tests/test_fundamentals_cik_map.py
git commit -m "Add committed CIK<->symbol PIT interval map + build script [AI]

company_tickers.json + reviewed rename chains; validated FB/META one CIK
across the rename and ABC->COR; unmapped symbols fail open to neutral.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01WuC8FbGLPEZUjYmdPSMMdv"
```

---

### Task 5: Backfill orchestration + download script

**Files:**
- Create: `src/trading/fundamentals/backfill.py`
- Create: `scripts/backfill_fundamentals.py`
- Test: `tests/test_fundamentals_backfill.py`

**Interfaces:**
- Consumes: `load_quarter_facts`, `empty_facts` (Task 1); `compute_pit_series` (Task 2); `FundamentalsStore` (Task 3); `load_cik_map`, `interval_slice` (Task 4).
- Produces: `quarter_range(start: str, end: str) -> list[str]`; `last_complete_quarter(today: datetime.date) -> str`; `backfill_quarters(zip_paths: list[Path], cik_map: pd.DataFrame, store: FundamentalsStore) -> dict[str, int]` (keys `"filers"`, `"symbols"`, `"rows"`). Task 11 runs the script for real.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_fundamentals_backfill.py`:

```python
import datetime

import pandas as pd

from fundamentals_helpers import num_line, sub_line, write_quarter_zip
from trading.fundamentals.backfill import backfill_quarters, last_complete_quarter, quarter_range
from trading.fundamentals.store import FundamentalsStore

# CIK 1326801 renamed FB -> NEWCO at 2023-08-01 in this fixture map: one CIK's
# series must split across the two symbols by FILED date.
CIK_MAP = pd.DataFrame(
    {
        "symbol": ["FB", "NEWCO", "OTHER"],
        "cik": [1326801, 1326801, 555],
        "start": ["2017-01-01", "2023-08-01", "2017-01-01"],
        "end": ["2023-08-01", "", ""],
    }
)


def _quarter_zip(tmp_path, name, adsh, fy, fp, period, filed, rev, cogs, assets, form="10-Q"):
    subs = [sub_line(adsh, 1326801, form, period, fy, fp, filed)]
    qtrs = 4 if form == "10-K" else 1
    nums = [
        num_line(adsh, "Revenues", period, qtrs, rev),
        num_line(adsh, "CostOfRevenue", period, qtrs, cogs),
        num_line(adsh, "Assets", period, 0, assets),
    ]
    return write_quarter_zip(tmp_path / name, subs, nums)


def _fixture_zips(tmp_path):
    return [
        _quarter_zip(tmp_path, "2023q2.zip", "f-01", "2023", "Q1", "20230331", "20230510",
                     10.0, 4.0, 90.0),
        _quarter_zip(tmp_path, "2023q3.zip", "f-02", "2023", "Q2", "20230630", "20230809",
                     12.0, 5.0, 95.0),
        _quarter_zip(tmp_path, "2023q4.zip", "f-03", "2023", "Q3", "20230930", "20231108",
                     11.0, 5.0, 98.0),
        _quarter_zip(tmp_path, "2024q1.zip", "f-04", "2023", "FY", "20231231", "20240220",
                     48.0, 20.0, 100.0, form="10-K"),
    ]


def test_backfill_splits_one_cik_series_across_rename_symbols(tmp_path):
    store = FundamentalsStore(tmp_path / "store")
    stats = backfill_quarters(_fixture_zips(tmp_path), CIK_MAP, store)
    assert stats == {"filers": 1, "symbols": 2, "rows": 4}
    fb = store.read("FB")
    newco = store.read("NEWCO")
    # Filed 2023-05-10 lands on FB (pre-rename); the rest on NEWCO.
    assert list(fb.index) == [pd.Timestamp("2023-05-10", tz="UTC")]
    assert len(newco) == 3
    # TTM continuity survives the split: the 10-K row completes 4 quarters.
    at_10k = newco.loc[pd.Timestamp("2024-02-20", tz="UTC")]
    assert at_10k["gross_profitability"] == (48.0 - 20.0) / 100.0
    assert store.read("OTHER").empty


def test_backfill_rerun_is_idempotent(tmp_path):
    store = FundamentalsStore(tmp_path / "store")
    zips = _fixture_zips(tmp_path)
    backfill_quarters(zips, CIK_MAP, store)
    stats = backfill_quarters(zips, CIK_MAP, store)
    assert stats["rows"] == 0  # append-only store: reprocessing adds nothing


def test_quarter_range_and_last_complete_quarter():
    assert quarter_range("2018q1", "2018q4") == ["2018q1", "2018q2", "2018q3", "2018q4"]
    assert quarter_range("2018q3", "2019q2") == ["2018q3", "2018q4", "2019q1", "2019q2"]
    assert last_complete_quarter(datetime.date(2026, 7, 6)) == "2026q2"
    assert last_complete_quarter(datetime.date(2026, 2, 1)) == "2025q4"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_fundamentals_backfill.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'trading.fundamentals.backfill'`.

- [ ] **Step 3: Implement the orchestration**

Create `src/trading/fundamentals/backfill.py`:

```python
"""Backfill orchestration: quarterly ZIPs -> facts -> PIT series -> store.

Pure composition of already-tested pieces (edgar/metrics/store/cik_map);
network + paths live in scripts/backfill_fundamentals.py. All quarters are
parsed together so TTM windows can span quarter boundaries; the append-only
store makes reruns idempotent (rows already visible are never rewritten).
"""

from __future__ import annotations

import datetime
from pathlib import Path

import pandas as pd

from trading.fundamentals.cik_map import interval_slice
from trading.fundamentals.edgar import empty_facts, load_quarter_facts
from trading.fundamentals.metrics import compute_pit_series
from trading.fundamentals.store import FundamentalsStore


def quarter_range(start: str, end: str) -> list[str]:
    """Inclusive "2018q1".."2019q2" -> every quarter label between."""
    year, quarter = int(start[:4]), int(start[5])
    end_year, end_quarter = int(end[:4]), int(end[5])
    out: list[str] = []
    while (year, quarter) <= (end_year, end_quarter):
        out.append(f"{year}q{quarter}")
        quarter += 1
        if quarter == 5:
            year, quarter = year + 1, 1
    return out


def last_complete_quarter(today: datetime.date) -> str:
    """SEC publishes a quarter's ZIP after the quarter ends; the in-progress
    quarter is served by the companyfacts top-up instead."""
    completed = (today.month - 1) // 3
    if completed == 0:
        return f"{today.year - 1}q4"
    return f"{today.year}q{completed}"


def backfill_quarters(
    zip_paths: list[Path], cik_map: pd.DataFrame, store: FundamentalsStore
) -> dict[str, int]:
    ciks = set(cik_map["cik"])
    # Drop empty per-quarter frames before concat (pandas 2.x warns on
    # empty-frame concatenation and the suite runs warnings-as-errors).
    parts = [f for path in zip_paths if not (f := load_quarter_facts(path, ciks)).empty]
    facts = pd.concat(parts, ignore_index=True) if parts else empty_facts()
    series_by_cik = compute_pit_series(facts)
    rows_appended = 0
    symbols_written: set[str] = set()
    for row in cik_map.itertuples():
        frame = series_by_cik.get(row.cik)
        if frame is None:
            continue
        window = interval_slice(frame, row.start, row.end)
        if window.empty:
            continue
        added = store.append(row.symbol, window)
        if added:
            symbols_written.add(row.symbol)
            rows_appended += added
    return {"filers": len(series_by_cik), "symbols": len(symbols_written), "rows": rows_appended}
```

- [ ] **Step 4: Implement the download script**

Create `scripts/backfill_fundamentals.py`:

```python
"""Download SEC Financial Statement Data Set quarterly ZIPs (default 2018q1 ->
the last complete quarter) into data/edgar-raw/ (gitignored via /data/) and
build the per-symbol fundamentals store at [data] fundamentals_dir.

Rerun-safe: ZIPs already on disk are not re-downloaded, and the store is
append-only so reprocessing appends 0 rows. The in-progress quarter has no
ZIP yet (404 -> warn + skip); the weekly companyfacts top-up covers it.

Locked default span is 2018q1 (see plan); TTM needs 4 trailing quarters, so
most metrics stay NaN/neutral until the FY-2018 10-K wave in early 2019.
Pass --from-quarter 2017q1 to fill that warm-up deliberately.

Usage: uv run python scripts/build_cik_map.py            (first, once)
       uv run python scripts/backfill_fundamentals.py [--from-quarter 2018q1]
"""

from __future__ import annotations

import argparse
import datetime
import os
import time
import tomllib
import urllib.error
import urllib.request
from pathlib import Path

from trading.fundamentals.backfill import backfill_quarters, last_complete_quarter, quarter_range
from trading.fundamentals.cik_map import load_cik_map
from trading.fundamentals.edgar import USER_AGENT
from trading.fundamentals.store import FundamentalsStore

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "edgar-raw"
ZIP_URL = "https://www.sec.gov/files/dera/data/financial-statement-data-sets/{quarter}.zip"
REQUEST_SPACING_S = 0.11  # SEC ceiling is 10 req/s; stay under it


def download(quarter: str) -> Path | None:
    dest = RAW_DIR / f"{quarter}.zip"
    if dest.exists():
        return dest
    url = ZIP_URL.format(quarter=quarter)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    print(f"downloading {url}")
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            data = resp.read()
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            print(f"WARNING: {quarter}.zip not published yet; skipping (top-up covers it)")
            return None
        raise
    tmp = dest.with_suffix(".zip.tmp")
    tmp.write_bytes(data)
    os.replace(tmp, dest)
    time.sleep(REQUEST_SPACING_S)
    return dest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-quarter", default="2018q1", help="first quarterly ZIP (YYYYqN)")
    args = parser.parse_args()

    data_cfg = tomllib.loads((ROOT / "config" / "equities.toml").read_text())["data"]
    store = FundamentalsStore(ROOT / data_cfg["fundamentals_dir"])
    cik_map = load_cik_map()
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    quarters = quarter_range(args.from_quarter, last_complete_quarter(datetime.date.today()))
    zips = [path for quarter in quarters if (path := download(quarter)) is not None]
    print(f"parsing {len(zips)} quarterly ZIPs for {len(set(cik_map['cik']))} CIKs ...")
    stats = backfill_quarters(zips, cik_map, store)
    print(
        f"done: {stats['filers']} filers -> {stats['symbols']} symbols, "
        f"{stats['rows']} rows appended"
    )


if __name__ == "__main__":
    main()
```

Note: the script reads `[data] fundamentals_dir` from `config/equities.toml`; that key lands in Task 7. Running the SCRIPT before Task 7 would `KeyError` — that is fine, it is not run until Task 11. The tests below do not touch the script.

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_fundamentals_backfill.py -v`
Expected: 3 PASS.

- [ ] **Step 6: Full suite + lint, then commit**

```bash
uv run ruff check . && uv run ruff format .
uv run pytest -q
git add src/trading/fundamentals/backfill.py scripts/backfill_fundamentals.py \
        tests/test_fundamentals_backfill.py
git commit -m "Add fundamentals backfill orchestration + EDGAR download script [AI]

All quarters parsed together for cross-quarter TTM; per-CIK series split
across rename symbols by FILED-date interval; idempotent via the append-only
store; ZIPs cached under data/edgar-raw/.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01WuC8FbGLPEZUjYmdPSMMdv"
```

---

### Task 6: companyfacts current-quarter top-up

**Files:**
- Create: `src/trading/fundamentals/companyfacts.py`
- Test: `tests/test_fundamentals_companyfacts.py`

**Interfaces:**
- Consumes: `TAG_PRIORITY`, `FACT_COLUMNS`, `empty_facts`, `USER_AGENT` (Task 1); `compute_pit_series`, `empty_series` (Task 2); `FundamentalsStore` (Task 3); `cik_for`, `interval_slice` (Task 4).
- Produces: `COMPANYFACTS_URL: str`; `facts_from_companyfacts(payload: dict, cik: int) -> pd.DataFrame`; `refresh_fundamentals(store, cik_map, symbols: Iterable[str], as_of: datetime.date, fetch_json: Callable[[str], dict] = _http_get_json) -> tuple[int, bool]` returning `(rows_appended, degraded)`. Task 10 (runner) consumes `refresh_fundamentals`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_fundamentals_companyfacts.py`:

```python
import datetime

import pandas as pd

from trading.fundamentals.companyfacts import facts_from_companyfacts, refresh_fundamentals
from trading.fundamentals.store import FundamentalsStore

CIK = 320193
CIK_MAP = pd.DataFrame(
    {"symbol": ["AAPL"], "cik": [CIK], "start": ["2017-01-01"], "end": [""]}
)


def _entry(end, val, accn, fy, fp, form, filed, start=None):
    entry = {"end": end, "val": val, "accn": accn, "fy": fy, "fp": fp,
             "form": form, "filed": filed}
    if start is not None:
        entry["start"] = start
    return entry


def _payload():
    """Four 10-Qs + one 10-K, plus traps: a comparative period re-reported in
    the 10-K, an amendment, and a year-to-date (qtrs=2) duration."""
    rev = [
        _entry("2023-03-31", 10.0, "a-01", 2023, "Q1", "10-Q", "2023-05-10", start="2023-01-01"),
        _entry("2023-06-30", 12.0, "a-02", 2023, "Q2", "10-Q", "2023-08-09", start="2023-04-01"),
        # YTD duration inside the same 10-Q: must be dropped (6 months != 1 quarter).
        _entry("2023-06-30", 22.0, "a-02", 2023, "Q2", "10-Q", "2023-08-09", start="2023-01-01"),
        _entry("2023-09-30", 11.0, "a-03", 2023, "Q3", "10-Q", "2023-11-08", start="2023-07-01"),
        _entry("2023-12-31", 48.0, "a-04", 2023, "FY", "10-K", "2024-02-20", start="2023-01-01"),
        # Comparative full-year 2022 re-reported inside the FY2023 10-K: the
        # own-period filter (max end per accession) must drop it.
        _entry("2022-12-31", 40.0, "a-04", 2023, "FY", "10-K", "2024-02-20", start="2022-01-01"),
        # Amendment: never parses.
        _entry("2023-03-31", 999.0, "a-05", 2023, "Q1", "10-Q/A", "2023-09-01", start="2023-01-01"),
    ]
    cogs = [
        _entry("2023-03-31", 4.0, "a-01", 2023, "Q1", "10-Q", "2023-05-10", start="2023-01-01"),
        _entry("2023-06-30", 5.0, "a-02", 2023, "Q2", "10-Q", "2023-08-09", start="2023-04-01"),
        _entry("2023-09-30", 5.0, "a-03", 2023, "Q3", "10-Q", "2023-11-08", start="2023-07-01"),
        _entry("2023-12-31", 20.0, "a-04", 2023, "FY", "10-K", "2024-02-20", start="2023-01-01"),
    ]
    assets = [
        _entry("2023-03-31", 90.0, "a-01", 2023, "Q1", "10-Q", "2023-05-10"),
        _entry("2023-06-30", 95.0, "a-02", 2023, "Q2", "10-Q", "2023-08-09"),
        _entry("2023-09-30", 98.0, "a-03", 2023, "Q3", "10-Q", "2023-11-08"),
        _entry("2023-12-31", 100.0, "a-04", 2023, "FY", "10-K", "2024-02-20"),
    ]
    return {
        "facts": {
            "us-gaap": {
                "Revenues": {"units": {"USD": rev}},
                "CostOfRevenue": {"units": {"USD": cogs}},
                "Assets": {"units": {"USD": assets}},
            }
        }
    }


def test_normalizes_to_own_period_facts_only():
    facts = facts_from_companyfacts(_payload(), CIK)
    assert set(facts["adsh"]) == {"a-01", "a-02", "a-03", "a-04"}
    fy = facts[(facts["adsh"] == "a-04") & (facts["concept"] == "revenue")]
    assert list(fy["value"]) == [48.0]
    assert list(fy["qtrs"]) == [4]
    q2 = facts[(facts["adsh"] == "a-02") & (facts["concept"] == "revenue")]
    assert list(q2["value"]) == [12.0]  # the qtrs=2 YTD duration was dropped


def test_refresh_appends_only_new_filed_dates(tmp_path):
    store = FundamentalsStore(tmp_path)
    calls: list[str] = []

    def fetch(url: str) -> dict:
        calls.append(url)
        return _payload()

    added, degraded = refresh_fundamentals(
        store, CIK_MAP, ["AAPL"], as_of=datetime.date(2024, 3, 1), fetch_json=fetch
    )
    assert (added, degraded) == (4, False)
    assert calls == [f"https://data.sec.gov/api/xbrl/companyfacts/CIK{CIK:010d}.json"]
    at_10k = store.read("AAPL").loc[pd.Timestamp("2024-02-20", tz="UTC")]
    assert at_10k["gross_profitability"] == (48.0 - 20.0) / 100.0

    added, degraded = refresh_fundamentals(
        store, CIK_MAP, ["AAPL"], as_of=datetime.date(2024, 3, 8), fetch_json=fetch
    )
    assert (added, degraded) == (0, False)  # append-only: nothing new to add


def test_refresh_is_fail_open_per_symbol(tmp_path):
    store = FundamentalsStore(tmp_path)
    cik_map = pd.DataFrame(
        {
            "symbol": ["AAPL", "BOOM", "UNMAPPED"],
            "cik": [CIK, 999, 0],
            "start": ["2017-01-01", "2017-01-01", "2017-01-01"],
            "end": ["", "", ""],
        }
    )
    cik_map = cik_map[cik_map["symbol"] != "UNMAPPED"]  # UNMAPPED has no row at all

    def fetch(url: str) -> dict:
        if "0000000999" in url:
            raise OSError("edgar down")
        return _payload()

    added, degraded = refresh_fundamentals(
        store, cik_map, ["AAPL", "BOOM", "UNMAPPED"],
        as_of=datetime.date(2024, 3, 1), fetch_json=fetch,
    )
    assert degraded is True   # BOOM failed -> degraded, run continues
    assert added == 4         # AAPL still refreshed; UNMAPPED silently skipped
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_fundamentals_companyfacts.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'trading.fundamentals.companyfacts'`.

- [ ] **Step 3: Implement the top-up**

Create `src/trading/fundamentals/companyfacts.py`:

```python
"""Current-quarter fundamentals top-up from data.sec.gov companyfacts (M4).

The quarterly ZIPs trail by a full quarter; companyfacts serves each filer's
complete XBRL history the day after it files. Entries are normalized to the
SAME facts table edgar.py produces and pushed through the SAME
compute_pit_series, so backfill and top-up cannot diverge -- and the
append-only store guarantees already-visible history is never rewritten by a
top-up (PIT: the earliest write for a filed date wins forever).

Fail-open (the earnings pattern): a per-symbol failure degrades -- that
symbol keeps its stored (possibly stale) values, the run continues, and the
caller journals the degraded flag as a warning. Never crash a run over
fundamentals.
"""

from __future__ import annotations

import datetime
import json
import time
import urllib.request
from collections.abc import Callable, Iterable

import pandas as pd

from trading.fundamentals.cik_map import cik_for, interval_slice
from trading.fundamentals.edgar import FACT_COLUMNS, TAG_PRIORITY, USER_AGENT, empty_facts
from trading.fundamentals.metrics import compute_pit_series, empty_series
from trading.fundamentals.store import FundamentalsStore

COMPANYFACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
_MIN_REQUEST_INTERVAL_S = 0.11  # SEC ceiling is 10 req/s; stay under it
# Duration windows for classifying a fact's period length from (start, end):
# a fiscal quarter is ~90-98 days, a fiscal year 357-371 (53-week years).
_QUARTER_DAYS = (80, 100)
_YEAR_DAYS = (350, 380)

_last_request_monotonic = 0.0


def _http_get_json(url: str) -> dict:
    """Network touchpoint, isolated for monkeypatching; throttled + UA per SEC policy."""
    global _last_request_monotonic
    wait = _MIN_REQUEST_INTERVAL_S - (time.monotonic() - _last_request_monotonic)
    if wait > 0:
        time.sleep(wait)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    _last_request_monotonic = time.monotonic()
    return payload


def facts_from_companyfacts(payload: dict, cik: int) -> pd.DataFrame:
    """companyfacts JSON -> the normalized facts table (edgar.FACT_COLUMNS)."""
    gaap = payload.get("facts", {}).get("us-gaap", {})
    records: list[dict] = []
    for concept, tags in TAG_PRIORITY.items():
        for priority, tag in enumerate(tags):
            for entry in gaap.get(tag, {}).get("units", {}).get("USD", []):
                if entry.get("form") not in ("10-K", "10-Q"):
                    continue  # amendments and other forms never parse (PIT)
                if entry.get("fy") is None or not entry.get("fp"):
                    continue
                end = pd.Timestamp(entry["end"])
                if concept == "assets":
                    qtrs = 0
                else:
                    if "start" not in entry:
                        continue
                    days = (end - pd.Timestamp(entry["start"])).days + 1
                    if _QUARTER_DAYS[0] <= days <= _QUARTER_DAYS[1]:
                        qtrs = 1
                    elif _YEAR_DAYS[0] <= days <= _YEAR_DAYS[1]:
                        qtrs = 4
                    else:
                        continue  # YTD and other durations are not single quarters/years
                    if qtrs != (4 if entry["form"] == "10-K" else 1):
                        continue
                records.append(
                    {
                        "cik": cik,
                        "adsh": entry["accn"],
                        "form": entry["form"],
                        "fy": str(entry["fy"]),
                        "fp": "FY" if entry["form"] == "10-K" else entry["fp"],
                        "period": end,
                        "filed": pd.Timestamp(entry["filed"]),
                        "concept": concept,
                        "tag": tag,
                        "qtrs": qtrs,
                        "value": float(entry["val"]),
                        "_priority": priority,
                    }
                )
    if not records:
        return empty_facts()
    facts = pd.DataFrame.from_records(records)
    # companyfacts carries COMPARATIVE periods (a 10-K re-reports prior
    # years); keep only each filing's OWN fiscal period -- the latest period
    # end that accession reports (the ZIP path's ddate == sub.period twin).
    own_period = facts.groupby("adsh")["period"].transform("max")
    facts = facts[facts["period"] == own_period]
    facts = facts.sort_values(["adsh", "concept", "_priority"], kind="mergesort")
    facts = facts.drop_duplicates(["adsh", "concept"], keep="first")
    return facts[FACT_COLUMNS].reset_index(drop=True)


def refresh_fundamentals(
    store: FundamentalsStore,
    cik_map: pd.DataFrame,
    symbols: Iterable[str],
    as_of: datetime.date,
    fetch_json: Callable[[str], dict] = _http_get_json,
) -> tuple[int, bool]:
    """Top up `symbols` from companyfacts. Returns (rows appended, degraded).
    Unmapped symbols are skipped silently (no EDGAR ticker -> neutral rank);
    per-symbol failures set degraded and never raise."""
    appended = 0
    degraded = False
    series_by_cik: dict[int, pd.DataFrame] = {}
    for symbol in symbols:
        cik = cik_for(cik_map, symbol, as_of)
        if cik is None:
            continue
        try:
            if cik not in series_by_cik:  # GOOG/GOOGL share one CIK: fetch once
                payload = fetch_json(COMPANYFACTS_URL.format(cik=cik))
                facts = facts_from_companyfacts(payload, cik)
                series_by_cik[cik] = compute_pit_series(facts).get(cik, empty_series())
            rows = cik_map[(cik_map["symbol"] == symbol) & (cik_map["cik"] == cik)]
            for row in rows.itertuples():
                appended += store.append(
                    symbol, interval_slice(series_by_cik[cik], row.start, row.end)
                )
        except Exception:
            degraded = True  # fail-open: stored values serve this run (earnings pattern)
    return appended, degraded
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_fundamentals_companyfacts.py -v`
Expected: 3 PASS.

- [ ] **Step 5: Full suite + lint, then commit**

```bash
uv run ruff check . && uv run ruff format .
uv run pytest -q
git add src/trading/fundamentals/companyfacts.py tests/test_fundamentals_companyfacts.py
git commit -m "Add companyfacts current-quarter fundamentals top-up [AI]

Normalizes to the same facts table + PIT computation as the ZIP backfill;
own-period filter drops comparatives; throttled + UA per SEC policy;
fail-open per symbol like the earnings fetch.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01WuC8FbGLPEZUjYmdPSMMdv"
```

---

### Task 7: Ranker registry v2 (`RankerSpec`) + fundamentals config keys

The registry contract gains an optional fundamentals input without touching `compute_features`: every REGISTERED callable takes `(bars, as_of, config, fundamentals)`, and the registry entry carries a `requires_fundamentals` flag so pipeline/backtest/runner know whether to load or refresh fundamentals at all. momentum_v1 registers through a one-line named adapter that ignores the 4th argument.

**Files:**
- Modify: `src/trading/signals/registry.py` (full rewrite below)
- Modify: `src/trading/pipeline.py:133-134` (call through the spec)
- Modify: `src/trading/config.py:92-105` (DataConfig fields), `:144-160` (load-time check)
- Modify: `config/equities.toml` ([data] keys), `config/crypto.toml` ([data] keys)
- Test: `tests/test_registry.py` (rewrite), `tests/test_config.py` (append)

**Interfaces:**
- Consumes: `compute_features` (unchanged, 3-arg).
- Produces: `RankerFn = Callable[[dict[str, pd.DataFrame], pd.Timestamp, SignalConfig, dict[str, pd.DataFrame] | None], pd.DataFrame]`; `RankerSpec(fn: RankerFn, requires_fundamentals: bool)` (frozen dataclass); `RANKERS: dict[str, RankerSpec]`; `get_ranker(name) -> RankerSpec`; `DataConfig.fundamentals_dir: str` ("" = venue has no fundamentals), `DataConfig.fundamentals_refresh_days: int`. Tasks 8–10 consume all of these.

- [ ] **Step 1: Rewrite `tests/test_registry.py` (failing against the current registry)**

Replace the whole file with:

```python
import numpy as np
import pandas as pd
import pytest

from trading.config import SignalConfig
from trading.signals.engine import compute_features
from trading.signals.registry import RANKERS, RankerSpec, get_ranker

CONFIG = SignalConfig(
    momentum_windows=(5, 10, 20),
    calendar_days=False,
    vol_window=5,
    volume_week=5,
    volume_baseline=20,
    breakout_windows=(10, 20),
    rsi_window=14,
    mean_window=20,
    raw_return_days=30,
    ranker="momentum_v1",
)


def _trending_bars(drift: float, periods: int = 80) -> pd.DataFrame:
    idx = pd.date_range("2025-11-03", periods=periods, freq="B", tz="UTC")
    jitter = np.where(np.arange(periods) % 2 == 0, 0.002, -0.002)
    close = 100 * np.cumprod(1 + drift + jitter)
    return pd.DataFrame(
        {
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": np.full(periods, 1e6),
        },
        index=idx,
    )


def test_get_ranker_unknown_name_raises_listing_known_names():
    with pytest.raises(ValueError, match="momentum_v1"):
        get_ranker("bogus")


def test_get_ranker_returns_registered_spec():
    spec = get_ranker("momentum_v1")
    assert isinstance(spec, RankerSpec)
    assert spec is RANKERS["momentum_v1"]
    assert spec.requires_fundamentals is False


def test_momentum_v1_ignores_fundamentals_and_matches_compute_features():
    bars = {
        "UP": _trending_bars(0.01),
        "FLAT": _trending_bars(0.0),
        "DOWN": _trending_bars(-0.01),
    }
    as_of = bars["UP"].index[-1]
    spec = get_ranker("momentum_v1")
    with_none = spec.fn(bars, as_of, CONFIG, None)
    with_junk = spec.fn(bars, as_of, CONFIG, {"UP": pd.DataFrame()})
    direct = compute_features(bars, as_of, CONFIG)
    pd.testing.assert_frame_equal(with_none, direct)
    pd.testing.assert_frame_equal(with_junk, direct)
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_registry.py -v`
Expected: FAIL — `ImportError: cannot import name 'RankerSpec'`.

- [ ] **Step 3: Rewrite the registry**

Replace `src/trading/signals/registry.py` with:

```python
"""Ranker registry (spec: pluggable ranking strategies), contract v2.

A registered ranker is a RankerSpec whose fn matches:

    fn(bars: dict[str, pd.DataFrame], as_of: pd.Timestamp,
       config: SignalConfig, fundamentals: dict[str, pd.DataFrame] | None
       ) -> pd.DataFrame

Input contract (what a ranker receives):

- `bars` maps symbol -> OHLCV DataFrame with columns exactly
  [open, high, low, close, volume], indexed by a sorted tz-aware UTC
  DatetimeIndex normalized to the bar's date (the shape enforced by
  trading.venues.base.validate_ohlcv). Frames may extend PAST as_of: the
  caller does not pre-cut them.
- `as_of` is a tz-aware UTC pd.Timestamp; momentum_v1 rejects a naive one.
- `fundamentals` maps symbol -> per-symbol fundamentals frame indexed by
  tz-aware UTC FILING dates (trading.fundamentals.store schema, at minimum a
  "gross_profitability" column). None (or a missing symbol) means "no
  fundamentals known" -- a ranker must treat that as neutral, never crash.
  Like bars, frames may extend past as_of; cutting to rows FILED at or
  before as_of is the RANKER's responsibility (same structural
  no-lookahead rule as bars).

Contract a registered ranker MUST guarantee (identical to the signal
engine's existing guarantees -- see trading.signals.engine):

- Purity: no I/O, no wall-clock reads; as_of is the only time input.
- Truncation to as_of is the ranker's job for BOTH bars and fundamentals.
- Column contract: the returned DataFrame is indexed by symbol and contains
  the ranker's feature-percentile columns plus "composite" and
  "raw_return_30d" (momentum_v1: trading.signals.engine.OUTPUT_COLUMNS;
  quality_momentum_v1 adds "quality"). Symbols without enough PRICE history
  are omitted; missing FUNDAMENTALS never drop a symbol (neutral instead).
- NaN semantics: an individual NaN feature yields a NaN composite for that
  symbol, which sorts last in rank().

requires_fundamentals tells the I/O layers what to do BEFORE the ranker
runs: pipeline/backtest load the fundamentals store and the live runner
refreshes it only when the configured ranker sets this flag -- momentum_v1
venues never touch fundamentals I/O at all.

The shared rank() sort is NOT part of a ranker's job: every ranker's output
feeds through trading.signals.engine.rank() afterward.

To add a new ranker: implement a RankerFn and register a RankerSpec here
under a new key. Select it per-venue via the `ranker` key in a venue's
[signals] TOML section; trading.config validates the name at load time.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd

from trading.config import SignalConfig
from trading.signals.engine import compute_features

RankerFn = Callable[
    [dict[str, pd.DataFrame], pd.Timestamp, SignalConfig, dict[str, pd.DataFrame] | None],
    pd.DataFrame,
]


@dataclass(frozen=True)
class RankerSpec:
    fn: RankerFn
    requires_fundamentals: bool


def _momentum_v1(
    bars: dict[str, pd.DataFrame],
    as_of: pd.Timestamp,
    config: SignalConfig,
    fundamentals: dict[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """v2-contract adapter: momentum_v1 has no fundamentals input by design;
    compute_features keeps its original 3-arg signature untouched."""
    return compute_features(bars, as_of, config)


RANKERS: dict[str, RankerSpec] = {
    "momentum_v1": RankerSpec(_momentum_v1, requires_fundamentals=False),
}


def get_ranker(name: str) -> RankerSpec:
    try:
        return RANKERS[name]
    except KeyError:
        known = ", ".join(sorted(RANKERS))
        raise ValueError(f"unknown ranker {name!r}; known rankers: {known}") from None
```

- [ ] **Step 4: Fix the pipeline call site**

In `src/trading/pipeline.py`, replace (currently lines 133–134):

```python
    ranker = get_ranker(config.signals.ranker)
    features = ranker(clean, as_of_ts, config.signals)
```

with:

```python
    spec = get_ranker(config.signals.ranker)
    features = spec.fn(clean, as_of_ts, config.signals, None)
```

(The `None` becomes the threaded fundamentals dict in Task 9.)

- [ ] **Step 5: Add the config fields and TOML keys**

In `src/trading/config.py`, add two fields at the end of `DataConfig` (after `seam_max_gap_days: int`):

```python
    # M4 fundamentals overlay. fundamentals_dir = "" means "this venue has no
    # fundamentals" (crypto); a ranker that requires fundamentals refuses to
    # load with it empty. refresh_days is the live top-up cadence -- data
    # plumbing, NOT a tunable hyperparameter (the walk-forward surface stays
    # entry_score_threshold x stop_atr_multiple only). Defaulted (like
    # membership_exit_buffer_days) so frozen test-venue TOMLs -- notably
    # tests/golden/golden.toml, which must stay byte-identical -- keep
    # loading; both real venue TOMLs still set them explicitly.
    fundamentals_dir: str = ""
    fundamentals_refresh_days: int = 0
```

In `load_venue_config`, the existing validation block currently reads:

```python
    from trading.signals.registry import get_ranker

    get_ranker(signals["ranker"])
```

Replace those two statements' second line with:

```python
    spec = get_ranker(signals["ranker"])
    if spec.requires_fundamentals and not raw["data"].get("fundamentals_dir"):
        raise ValueError(
            f"ranker {signals['ranker']!r} requires [data] fundamentals_dir to be set"
        )
```

In `config/equities.toml`, append to the `[data]` section (after `seam_max_gap_days = 0`):

```toml
# M4 fundamentals overlay (used only when [signals] ranker requires it --
# the default momentum_v1 never touches these). Weekly companyfacts top-up.
fundamentals_dir = "data/fundamentals/equities"
fundamentals_refresh_days = 7
```

In `config/crypto.toml`, append to the `[data]` section:

```toml
# No fundamentals concept for crypto; "" disables (a fundamentals-requiring
# ranker then refuses to load for this venue).
fundamentals_dir = ""
fundamentals_refresh_days = 0
```

- [ ] **Step 6: Append config tests**

Append to `tests/test_config.py` (it already imports `load_venue_config` and loads real configs; add if missing: `from pathlib import Path`):

```python
def test_fundamentals_data_keys_load_for_both_venues():
    eq = load_venue_config("equities", Path("config"))
    assert eq.data.fundamentals_dir == "data/fundamentals/equities"
    assert eq.data.fundamentals_refresh_days == 7
    cr = load_venue_config("crypto", Path("config"))
    assert cr.data.fundamentals_dir == ""  # no fundamentals concept for crypto
    assert cr.data.fundamentals_refresh_days == 0
```

- [ ] **Step 7: Run the affected tests, then the full suite**

Run: `uv run pytest tests/test_registry.py tests/test_config.py tests/test_pipeline.py tests/test_engine.py -v`
Expected: all PASS (missing-TOML-key TypeErrors here mean a venue TOML was missed in Step 5).
Run: `uv run pytest -q 2>&1 | tee /tmp/claude-m4-t7.log`
Expected: all green — in particular `tests/test_golden_backtest.py` (the golden path goes through `_momentum_v1`, which is bit-identical to `compute_features`).

- [ ] **Step 8: Lint + commit**

```bash
uv run ruff check . && uv run ruff format .
git add src/trading/signals/registry.py src/trading/pipeline.py src/trading/config.py \
        config/equities.toml config/crypto.toml tests/test_registry.py tests/test_config.py
git commit -m "Move ranker registry to v2 contract with fundamentals input [AI]

RankerSpec(fn, requires_fundamentals); registered callables take an optional
fundamentals dict; compute_features untouched (named adapter); new [data]
fundamentals keys in both venue TOMLs. Golden backtest unchanged.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01WuC8FbGLPEZUjYmdPSMMdv"
```

---

### Task 8: `quality_momentum_v1` ranker

**Files:**
- Create: `src/trading/signals/quality.py`
- Modify: `src/trading/signals/registry.py` (register the new spec)
- Create: `config/experiments/quality/equities.toml` (opt-in experiment config)
- Test: `tests/test_quality_ranker.py` (NEW — `tests/test_quality.py` is the data-quality module's, do not touch it), `tests/test_config.py` (append)

**Interfaces:**
- Consumes: `compute_features`, `FEATURE_COLUMNS` from `trading.signals.engine`; `RankerSpec` (Task 7).
- Produces: `quality_momentum_v1(bars, as_of, config, fundamentals) -> pd.DataFrame` with columns `[*FEATURE_COLUMNS, "quality", "composite", "raw_return_30d"]` (`OUTPUT_COLUMNS` in the module); `QUALITY_NEUTRAL = 0.5`; registry key `"quality_momentum_v1"` with `requires_fundamentals=True`. Tasks 9–11 consume these.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_quality_ranker.py`:

```python
import numpy as np
import pandas as pd
import pytest

from trading.config import SignalConfig
from trading.signals.engine import FEATURE_COLUMNS, compute_features
from trading.signals.quality import OUTPUT_COLUMNS, quality_momentum_v1
from trading.signals.registry import get_ranker

CONFIG = SignalConfig(
    momentum_windows=(5, 10, 20),
    calendar_days=False,
    vol_window=5,
    volume_week=5,
    volume_baseline=20,
    breakout_windows=(10, 20),
    rsi_window=14,
    mean_window=20,
    raw_return_days=30,
    ranker="quality_momentum_v1",
)


def _trending_bars(drift: float, periods: int = 80) -> pd.DataFrame:
    idx = pd.date_range("2025-11-03", periods=periods, freq="B", tz="UTC")
    jitter = np.where(np.arange(periods) % 2 == 0, 0.002, -0.002)
    close = 100 * np.cumprod(1 + drift + jitter)
    return pd.DataFrame(
        {
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": np.full(periods, 1e6),
        },
        index=idx,
    )


def _fund(dated: dict[str, float]) -> pd.DataFrame:
    idx = pd.DatetimeIndex([pd.Timestamp(d, tz="UTC") for d in dated], name="filed")
    return pd.DataFrame({"gross_profitability": list(dated.values())}, index=idx)


BARS = {
    "UP": _trending_bars(0.01),
    "FLAT": _trending_bars(0.0),
    "DOWN": _trending_bars(-0.01),
}
AS_OF = BARS["UP"].index[-1]


def test_registered_and_requires_fundamentals():
    spec = get_ranker("quality_momentum_v1")
    assert spec.fn is quality_momentum_v1
    assert spec.requires_fundamentals is True


def test_composite_is_equal_weight_over_seven():
    fundamentals = {
        "UP": _fund({"2025-11-10": 0.6}),
        "FLAT": _fund({"2025-11-10": 0.2}),
        "DOWN": _fund({"2025-11-10": 0.4}),
    }
    out = quality_momentum_v1(BARS, AS_OF, CONFIG, fundamentals)
    base = compute_features(BARS, AS_OF, CONFIG)
    assert list(out.columns) == OUTPUT_COLUMNS
    # quality = cross-sectional percentile of the latest filed value.
    assert out.loc["UP", "quality"] == pytest.approx(1.0)
    assert out.loc["DOWN", "quality"] == pytest.approx(2 / 3)
    assert out.loc["FLAT", "quality"] == pytest.approx(1 / 3)
    for symbol in BARS:
        expected = (base.loc[symbol, "composite"] * len(FEATURE_COLUMNS)
                    + out.loc[symbol, "quality"]) / (len(FEATURE_COLUMNS) + 1)
        assert out.loc[symbol, "composite"] == pytest.approx(expected)
        # The six price features and raw_return_30d pass through unchanged.
        for col in [*FEATURE_COLUMNS, "raw_return_30d"]:
            assert out.loc[symbol, col] == base.loc[symbol, col]


def test_missing_or_nan_fundamentals_are_neutral_half():
    fundamentals = {
        "UP": _fund({"2025-11-10": 0.6}),
        "FLAT": _fund({"2025-11-10": 0.2}),
        # DOWN absent entirely; and a NaN latest value must also be neutral.
    }
    out = quality_momentum_v1(BARS, AS_OF, CONFIG, fundamentals)
    assert out.loc["DOWN", "quality"] == 0.5

    fundamentals["DOWN"] = _fund({"2025-11-10": float("nan")})
    out = quality_momentum_v1(BARS, AS_OF, CONFIG, fundamentals)
    assert out.loc["DOWN", "quality"] == 0.5


def test_no_fundamentals_at_all_is_all_neutral_and_momentum_ordering():
    out = quality_momentum_v1(BARS, AS_OF, CONFIG, None)
    assert set(out["quality"]) == {0.5}
    base = compute_features(BARS, AS_OF, CONFIG)
    assert list(out.sort_values("composite").index) == list(base.sort_values("composite").index)


def test_step_function_uses_latest_value_filed_at_or_before_as_of():
    fundamentals = {
        "UP": _fund({"2025-11-10": 0.1, "2026-06-01": 0.9}),  # 2nd filing AFTER as_of
        "FLAT": _fund({"2025-11-10": 0.6}),
        "DOWN": _fund({"2025-11-10": 0.4}),
    }
    out = quality_momentum_v1(BARS, AS_OF, CONFIG, fundamentals)
    # The post-as_of filing is invisible: UP's visible value is 0.1 -> lowest percentile.
    assert out.loc["UP", "quality"] == pytest.approx(1 / 3)
    # A frame with only pre-listing/no rows as-of is neutral, not a crash.
    fundamentals["UP"] = _fund({"2026-06-01": 0.9})
    out = quality_momentum_v1(BARS, AS_OF, CONFIG, fundamentals)
    assert out.loc["UP", "quality"] == 0.5


def test_nan_latest_never_reaches_back_to_an_older_value():
    fundamentals = {
        "UP": _fund({"2025-11-10": 0.9, "2025-12-01": float("nan")}),
        "FLAT": _fund({"2025-11-10": 0.6}),
        "DOWN": _fund({"2025-11-10": 0.4}),
    }
    out = quality_momentum_v1(BARS, AS_OF, CONFIG, fundamentals)
    # UP's LATEST filing as-of is NaN -> neutral; the 0.9 is history, not current.
    assert out.loc["UP", "quality"] == 0.5


def test_empty_universe_returns_empty_frame_with_columns():
    out = quality_momentum_v1({}, AS_OF, CONFIG, None)
    assert out.empty
    assert list(out.columns) == OUTPUT_COLUMNS


def test_naive_as_of_rejected():
    with pytest.raises(ValueError, match="tz-aware"):
        quality_momentum_v1(BARS, pd.Timestamp("2026-02-27"), CONFIG, None)
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_quality_ranker.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'trading.signals.quality'`.

- [ ] **Step 3: Implement the ranker**

Create `src/trading/signals/quality.py`:

```python
"""quality_momentum_v1: the six momentum_v1 features + a 7th cross-sectional
gross-profitability percentile (spec: M4 fundamentals overlay).

Quality value per symbol = the LAST fundamentals row FILED at or before
as_of (a forward-filled step function on FILING dates -- never fiscal-period
dates, never interpolated). A NaN latest value or a missing symbol
contributes the NEUTRAL 0.5 percentile: ~half of filers (financials) have no
COGS concept, and punishing missing data would silently tilt the universe.
NaN never reaches back to an older non-NaN value -- the latest filing IS the
point-in-time state of knowledge.

Composite = equal weight over 7 = (6 * momentum_v1 composite + quality) / 7.
No new tunable parameters: the walk-forward surface stays exactly
entry_score_threshold x stop_atr_multiple.

Pure: no I/O, no clock. Fundamentals frames may extend past as_of; the cut
to <= as_of happens here (same structural no-lookahead rule as bars).
"""

from __future__ import annotations

import math

import pandas as pd

from trading.config import SignalConfig
from trading.signals.engine import FEATURE_COLUMNS, compute_features

QUALITY_NEUTRAL = 0.5
OUTPUT_COLUMNS = [*FEATURE_COLUMNS, "quality", "composite", "raw_return_30d"]


def quality_momentum_v1(
    bars: dict[str, pd.DataFrame],
    as_of: pd.Timestamp,
    config: SignalConfig,
    fundamentals: dict[str, pd.DataFrame] | None,
) -> pd.DataFrame:
    base = compute_features(bars, as_of, config)
    if base.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS, dtype="float64")
    known = fundamentals or {}
    raw: dict[str, float] = {}
    for symbol in base.index:
        frame = known.get(symbol)
        value = math.nan
        if frame is not None and not frame.empty:
            window = frame.loc[:as_of]  # structural no-lookahead cut (FILED dates)
            if not window.empty:
                value = float(window["gross_profitability"].iloc[-1])
        raw[symbol] = value
    quality = pd.Series(raw, dtype="float64").rank(pct=True).fillna(QUALITY_NEUTRAL)
    out = base.copy()
    out["quality"] = quality
    # Recompose: base composite already equal-weights the six price features
    # (with the overextension guard inverted inside compute_features).
    out["composite"] = (base["composite"] * len(FEATURE_COLUMNS) + quality) / (
        len(FEATURE_COLUMNS) + 1
    )
    return out[OUTPUT_COLUMNS]
```

Register it — in `src/trading/signals/registry.py` add the import and entry:

```python
from trading.signals.quality import quality_momentum_v1
```

```python
RANKERS: dict[str, RankerSpec] = {
    "momentum_v1": RankerSpec(_momentum_v1, requires_fundamentals=False),
    "quality_momentum_v1": RankerSpec(quality_momentum_v1, requires_fundamentals=True),
}
```

- [ ] **Step 4: Create the opt-in experiment config**

```bash
mkdir -p config/experiments/quality
cp config/equities.toml config/experiments/quality/equities.toml
```

Then edit `config/experiments/quality/equities.toml`:
- Change the `[signals]` ranker line to `ranker = "quality_momentum_v1"  # M4 experiment: momentum + gross-profitability quality percentile`.
- Add at the very top of the file:

```toml
# EXPERIMENT CONFIG (M4): identical to config/equities.toml except
# [signals] ranker = quality_momentum_v1. Use via:
#   trading rankings --venue equities --config-dir config/experiments/quality
#   trading backtest --venue equities --config-dir config/experiments/quality
# Never point the live/paper scheduler at this directory.
```

- [ ] **Step 5: Append the load-time validation test**

Append to `tests/test_config.py`:

```python
def test_fundamentals_requiring_ranker_with_empty_dir_fails_at_load(tmp_path):
    text = (Path("config") / "equities.toml").read_text()
    text = text.replace('ranker = "momentum_v1"', 'ranker = "quality_momentum_v1"')
    text = text.replace(
        'fundamentals_dir = "data/fundamentals/equities"', 'fundamentals_dir = ""'
    )
    (tmp_path / "equities.toml").write_text(text)
    with pytest.raises(ValueError, match="fundamentals_dir"):
        load_venue_config("equities", tmp_path)


def test_quality_experiment_config_loads():
    config = load_venue_config("equities", Path("config") / "experiments" / "quality")
    assert config.signals.ranker == "quality_momentum_v1"
    assert config.data.fundamentals_dir == "data/fundamentals/equities"
```

(`tests/test_config.py` already imports `pytest` and `load_venue_config`; add `import pytest` / `from pathlib import Path` only if missing.)

- [ ] **Step 6: Run the affected tests, then the full suite**

Run: `uv run pytest tests/test_quality_ranker.py tests/test_config.py tests/test_registry.py -v`
Expected: all PASS.
Run: `uv run pytest -q`
Expected: all green.

- [ ] **Step 7: Lint + commit**

```bash
uv run ruff check . && uv run ruff format .
git add src/trading/signals/quality.py src/trading/signals/registry.py \
        config/experiments/quality/equities.toml \
        tests/test_quality_ranker.py tests/test_config.py
git commit -m "Add quality_momentum_v1 ranker: momentum + gross-profitability percentile [AI]

Seventh feature = cross-sectional percentile of the latest FILED gross
profitability (step function, no reach-back); NaN/missing -> neutral 0.5;
composite equal-weight over 7. Opt-in experiment config; live defaults and
tunable surface unchanged.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01WuC8FbGLPEZUjYmdPSMMdv"
```

---

### Task 9: Thread fundamentals through the pipeline and the backtest

Same as-of discipline as bars: live `build_rankings` loads the store when (and only when) the configured ranker requires it; backtest `prepare()` loads once and slices each session to rows FILED ≤ that session (the ranker re-cuts as defense in depth).

**Files:**
- Modify: `src/trading/pipeline.py` (`assemble_rankings` param; store load in `build_rankings`)
- Modify: `src/trading/backtest/engine.py` (`prepare()` load + per-session slice)
- Test: `tests/test_pipeline.py` (append), `tests/test_backtest_engine.py` (append)

**Interfaces:**
- Consumes: `RankerSpec`/`get_ranker` (Task 7), `quality_momentum_v1` registration (Task 8), `FundamentalsStore` (Task 3).
- Produces: `assemble_rankings(config, infos, bars, benchmark_bars, as_of, fetch_failures=(), fundamentals=None)` — new keyword-only-by-position last param `fundamentals: dict[str, pd.DataFrame] | None`. `build_rankings` and `prepare` signatures unchanged. Task 10 and the CLI (no changes needed there) rely on this.

- [ ] **Step 1: Append the failing pipeline tests**

Append to `tests/test_pipeline.py` (the file already imports `dataclasses`, `datetime`, `Path`, `pd`, `build_rankings`, `OhlcvCache`, and defines `_make`, `CONFIG`, `AS_OF`):

```python
def _quality_config(tmp_path):
    return dataclasses.replace(
        CONFIG,
        signals=dataclasses.replace(CONFIG.signals, ranker="quality_momentum_v1"),
        data=dataclasses.replace(CONFIG.data, fundamentals_dir=str(tmp_path / "fundamentals")),
    )


def _fund_rows(dated: dict[str, float]) -> pd.DataFrame:
    idx = pd.DatetimeIndex([pd.Timestamp(d, tz="UTC") for d in dated], name="filed")
    return pd.DataFrame({"gross_profitability": list(dated.values())}, index=idx)


def test_build_rankings_loads_fundamentals_for_a_quality_ranker(tmp_path):
    from trading.fundamentals.store import FundamentalsStore

    adapter, cache, _ = _make(tmp_path)
    config = _quality_config(tmp_path)
    store = FundamentalsStore(Path(config.data.fundamentals_dir))
    store.append("S0", _fund_rows({"2026-06-01": 0.6}))
    store.append("S1", _fund_rows({"2026-06-01": 0.2}))
    result = build_rankings(config, adapter, cache, AS_OF)
    assert "quality" in result.table.columns
    assert result.table.loc["S0", "quality"] == 1.0
    assert result.table.loc["S1", "quality"] == 0.5
    assert result.table.loc["S2", "quality"] == 0.5  # no store file -> neutral


def test_momentum_v1_never_touches_the_fundamentals_store(tmp_path, monkeypatch):
    import trading.pipeline as pipeline_module

    def boom(*args, **kwargs):
        raise AssertionError("momentum_v1 must not construct a FundamentalsStore")

    monkeypatch.setattr(pipeline_module, "FundamentalsStore", boom)
    adapter, cache, _ = _make(tmp_path)
    result = build_rankings(CONFIG, adapter, cache, AS_OF)  # default momentum_v1
    assert "quality" not in result.table.columns


def test_assemble_rankings_threads_fundamentals_to_the_ranker(tmp_path):
    from trading.pipeline import assemble_rankings

    config = _quality_config(tmp_path)
    bars = {"AAA": frame(periods=300), "BBB": frame(periods=300)}
    infos = [SymbolInfo("AAA", "tradable"), SymbolInfo("BBB", "tradable")]
    fundamentals = {
        "AAA": _fund_rows({"2026-06-01": 0.7}),
        "BBB": _fund_rows({"2026-06-01": 0.1}),
    }
    result = assemble_rankings(
        config, infos, bars, frame(periods=300), AS_OF, fundamentals=fundamentals
    )
    assert result.table.loc["AAA", "quality"] == 1.0
    assert result.table.loc["BBB", "quality"] == 0.5
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_pipeline.py -v -k fundamentals`
Expected: FAIL — `assemble_rankings() got an unexpected keyword argument 'fundamentals'` / missing quality column / `AttributeError: ... has no attribute 'FundamentalsStore'`.

- [ ] **Step 3: Implement the pipeline threading**

In `src/trading/pipeline.py`:

Add imports (top of file, with the others):

```python
from pathlib import Path

from trading.fundamentals.store import FundamentalsStore
```

In `build_rankings`, right before the final `return assemble_rankings(...)`, add:

```python
    fundamentals: dict[str, pd.DataFrame] | None = None
    if get_ranker(config.signals.ranker).requires_fundamentals:
        # Read-only here: the live REFRESH (weekly companyfacts top-up) is the
        # runner's job, fail-open. An empty/missing store simply yields all-
        # neutral quality -- never an abort.
        store = FundamentalsStore(Path(config.data.fundamentals_dir))
        fundamentals = store.load([i.symbol for i in infos])
```

and pass it through:

```python
    return assemble_rankings(
        config,
        infos,
        bars,
        benchmark,
        as_of,
        fetch_failures=tuple(sorted(failures)),
        fundamentals=fundamentals,
    )
```

Change `assemble_rankings`'s signature:

```python
def assemble_rankings(
    config: VenueConfig,
    infos: list[SymbolInfo],
    bars: dict[str, pd.DataFrame],
    benchmark_bars: pd.DataFrame,
    as_of: datetime.date,
    fetch_failures: tuple[str, ...] = (),
    fundamentals: dict[str, pd.DataFrame] | None = None,
) -> RankingsResult:
```

and update its docstring's first line to mention fundamentals, plus the ranker call from Task 7:

```python
    spec = get_ranker(config.signals.ranker)
    features = spec.fn(clean, as_of_ts, config.signals, fundamentals)
```

- [ ] **Step 4: Run the pipeline tests**

Run: `uv run pytest tests/test_pipeline.py -v`
Expected: all PASS (including the pre-existing seam/purity tests).

- [ ] **Step 5: Append the failing backtest test**

Append to `tests/test_backtest_engine.py`. The file already imports `datetime`, `pd`, `prepare`, `OhlcvCache`, `noisy_frame`, `FakeBacktestAdapter`, `small_config`; ADD these two imports at the top (they are currently missing):

```python
from dataclasses import replace
from pathlib import Path
```

```python
def test_prepare_slices_fundamentals_to_each_sessions_filed_dates(tmp_path):
    from trading.fundamentals.store import FundamentalsStore

    frames = {
        "AAA": noisy_frame(seed=1, periods=120),
        "BBB": noisy_frame(seed=2, periods=120),
        "CCC": noisy_frame(seed=3, periods=120),
        "BENCH": noisy_frame(seed=9, periods=120),
    }
    adapter = FakeBacktestAdapter(frames, "BENCH")
    cache = OhlcvCache(tmp_path / "cache", 3)
    base = small_config()
    config = replace(
        base,
        benchmark="BENCH",
        signals=replace(base.signals, ranker="quality_momentum_v1"),
        data=replace(base.data, fundamentals_dir=str(tmp_path / "fund")),
    )
    store = FundamentalsStore(Path(config.data.fundamentals_dir))

    def fund_rows(dated):
        idx = pd.DatetimeIndex([pd.Timestamp(d, tz="UTC") for d in dated], name="filed")
        return pd.DataFrame({"gross_profitability": list(dated.values())}, index=idx)

    # AAA files 0.9 in March, then 0.05 in April; BBB has a steady 0.2.
    store.append("AAA", fund_rows({"2025-03-01": 0.9, "2025-04-01": 0.05}))
    store.append("BBB", fund_rows({"2025-03-01": 0.2}))

    prepared = prepare(
        config, adapter, cache, datetime.date(2025, 3, 10), datetime.date(2025, 4, 10)
    )
    by_date = {plan.ts.date().isoformat(): plan for plan in prepared.sessions}
    march = by_date["2025-03-15"].rankings.table
    april = by_date["2025-04-10"].rankings.table
    # March session: April's filing is INVISIBLE -> AAA (0.9) outranks BBB (0.2).
    assert march.loc["AAA", "quality"] == 1.0
    assert march.loc["BBB", "quality"] == 0.5
    # April session: the new filing is visible -> AAA (0.05) now below BBB (0.2).
    assert april.loc["AAA", "quality"] == 0.5
    assert april.loc["BBB", "quality"] == 1.0
    # CCC never filed anything: neutral both times.
    assert march.loc["CCC", "quality"] == 0.5
    assert april.loc["CCC", "quality"] == 0.5
```

- [ ] **Step 6: Run to verify failure**

Run: `uv run pytest tests/test_backtest_engine.py::test_prepare_slices_fundamentals_to_each_sessions_filed_dates -v`
Expected: FAIL — `KeyError: 'quality'` (prepare never loaded fundamentals, so the ranker saw None → neutral column exists actually only if ranker ran; the precise failure is `march.loc["AAA", "quality"] == 1.0` asserting 0.5 ≠ 1.0).

- [ ] **Step 7: Implement the prepare() threading**

In `src/trading/backtest/engine.py`:

Add imports:

```python
from pathlib import Path

from trading.fundamentals.store import FundamentalsStore
from trading.signals.registry import get_ranker
```

In `prepare()`, after the `bars`/`missing` loop completes (right before `sessions: list[SessionPlan] = []`), add:

```python
    spec = get_ranker(config.signals.ranker)
    fundamentals_all: dict[str, pd.DataFrame] = {}
    if spec.requires_fundamentals:
        # Full-span load once; sliced per session below. Same discipline as
        # bars: nothing FILED after a session may influence it.
        fundamentals_all = FundamentalsStore(Path(config.data.fundamentals_dir)).load(union)
```

In the per-session loop, replace:

```python
        sliced = {i.symbol: bars[i.symbol].loc[:ts] for i in available}
        try:
            rankings = assemble_rankings(config, available, sliced, benchmark.loc[:ts], ts.date())
```

with:

```python
        sliced = {i.symbol: bars[i.symbol].loc[:ts] for i in available}
        fundamentals = None
        if spec.requires_fundamentals:
            fundamentals = {
                symbol: window
                for symbol, frame in fundamentals_all.items()
                if not (window := frame.loc[:ts]).empty
            }
        try:
            rankings = assemble_rankings(
                config,
                available,
                sliced,
                benchmark.loc[:ts],
                ts.date(),
                fundamentals=fundamentals,
            )
```

- [ ] **Step 8: Run the affected tests, then the full suite**

Run: `uv run pytest tests/test_backtest_engine.py tests/test_pipeline.py -v`
Expected: all PASS.
Run: `uv run pytest -q 2>&1 | tee /tmp/claude-m4-t9.log`
Expected: all green — the golden backtest and every momentum_v1 path are untouched (`requires_fundamentals=False` short-circuits all new code).

- [ ] **Step 9: Lint + commit**

```bash
uv run ruff check . && uv run ruff format .
git add src/trading/pipeline.py src/trading/backtest/engine.py \
        tests/test_pipeline.py tests/test_backtest_engine.py
git commit -m "Thread fundamentals through assemble_rankings and backtest prepare [AI]

Loaded only when the configured ranker requires it; per-session slices cut
to rows FILED <= the session (same as-of discipline as bars). momentum_v1
paths and the golden backtest untouched.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01WuC8FbGLPEZUjYmdPSMMdv"
```

---

### Task 10: Live runner weekly refresh (fail-open, journaled)

The runner refreshes the fundamentals store from companyfacts BEFORE building rankings, only when the configured ranker requires it and at most every `fundamentals_refresh_days` (marker file). Failures degrade with a journaled warning — exactly the earnings pattern — and never block the run.

**Files:**
- Modify: `src/trading/runner.py`
- Test: `tests/test_runner.py` (append)

**Interfaces:**
- Consumes: `refresh_fundamentals` (Task 6), `FundamentalsStore` (Task 3), `load_cik_map` (Task 4), `get_ranker` (Task 7).
- Produces: no new public API — behavior only (journaled `warnings` entries `"fundamentals refresh degraded: ..."` / `"fundamentals refresh failed: ..."`).

- [ ] **Step 1: Append the failing runner tests**

Append to `tests/test_runner.py` (the file already defines `NOW`, `FakeAdapter`, `_run`, imports `EQ`, `Journal`, `run_venue`, `OhlcvCache`, `datetime`, `pytest`; add at the top if missing: `import dataclasses`):

```python
def _quality_eq(tmp_path):
    return dataclasses.replace(
        EQ,
        signals=dataclasses.replace(EQ.signals, ranker="quality_momentum_v1"),
        data=dataclasses.replace(EQ.data, fundamentals_dir=str(tmp_path / "fund")),
    )


def _run_quality(tmp_path, monkeypatch, refresh):
    monkeypatch.setattr("trading.runner.refresh_fundamentals", refresh)
    notes: list[tuple[str, str]] = []
    outcome = run_venue(
        _quality_eq(tmp_path),
        FakeAdapter(),
        OhlcvCache(tmp_path / "cache", EQ.data.refetch_days),
        now=NOW,
        state_root=tmp_path / "state",
        journal_root=tmp_path / "journal",
        notify=lambda title, message: notes.append((title, message)),
    )
    return outcome, notes


def _last_run_event(tmp_path):
    events = list(Journal(tmp_path / "journal" / "equities.jsonl").events())
    return [e for e in events if e["event"] == "run"][-1]


def test_momentum_v1_run_never_refreshes_fundamentals(tmp_path, monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("momentum_v1 must never trigger a fundamentals refresh")

    monkeypatch.setattr("trading.runner.refresh_fundamentals", boom)
    outcome, _ = _run(tmp_path)  # default EQ config: momentum_v1
    assert outcome.status == "ok"


def test_quality_run_refreshes_once_then_respects_the_weekly_gate(tmp_path, monkeypatch):
    calls: list[tuple] = []

    def refresh(store, cik_map, symbols, as_of, **kwargs):
        calls.append((sorted(symbols), as_of))
        return 3, False

    outcome, _ = _run_quality(tmp_path, monkeypatch, refresh)
    assert outcome.status == "ok"
    assert calls == [(sorted(SYMBOLS), NOW.date())]

    from trading.fundamentals.store import FundamentalsStore

    store = FundamentalsStore(tmp_path / "fund")
    assert store.last_refresh() == NOW.date()

    # Second run inside the 7-day window: the gate must skip the refresh.
    _run_quality(tmp_path, monkeypatch, refresh)
    assert len(calls) == 1


def test_degraded_refresh_journals_a_warning_and_run_proceeds(tmp_path, monkeypatch):
    outcome, _ = _run_quality(tmp_path, monkeypatch, lambda *a, **k: (0, True))
    assert outcome.status == "ok"
    warnings = _last_run_event(tmp_path)["warnings"]
    assert any(w.startswith("fundamentals refresh degraded") for w in warnings)


def test_failed_refresh_is_fail_open_with_journaled_warning(tmp_path, monkeypatch):
    def refresh(*args, **kwargs):
        raise OSError("edgar unreachable")

    outcome, notes = _run_quality(tmp_path, monkeypatch, refresh)
    assert outcome.status == "ok"  # rankings ran on the (empty) stored fundamentals
    warnings = _last_run_event(tmp_path)["warnings"]
    assert any(w.startswith("fundamentals refresh failed") for w in warnings)

    from trading.fundamentals.store import FundamentalsStore

    # Marker NOT written on total failure: the next run retries immediately.
    assert FundamentalsStore(tmp_path / "fund").last_refresh() is None
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_runner.py -v -k "refresh or fundamentals"`
Expected: FAIL — `AttributeError: <module 'trading.runner'> has no attribute 'refresh_fundamentals'` (monkeypatch target missing).

- [ ] **Step 3: Implement the runner refresh**

In `src/trading/runner.py`:

Add imports (with the existing ones):

```python
from trading.fundamentals.cik_map import load_cik_map
from trading.fundamentals.companyfacts import refresh_fundamentals
from trading.fundamentals.store import FundamentalsStore
from trading.signals.registry import get_ranker
```

In `run_venue`, immediately BEFORE the `try:` block that calls `build_rankings` (after the state-behind-journal check), insert:

```python
        # Fundamentals refresh (M4): only when the configured ranker needs it,
        # at most every fundamentals_refresh_days, BEFORE build_rankings reads
        # the store. Fail-open exactly like the earnings fetch: a refresh
        # failure degrades to the last stored values with a journaled warning
        # -- it must never crash or block the run. The marker is written even
        # when partially degraded (weekly cadence stands) but NOT on total
        # failure, so the next run retries immediately.
        extra_warnings: list[str] = []
        if get_ranker(config.signals.ranker).requires_fundamentals:
            store = FundamentalsStore(Path(config.data.fundamentals_dir))
            last = store.last_refresh()
            due = last is None or (now.date() - last).days >= config.data.fundamentals_refresh_days
            if due:
                try:
                    symbols = [i.symbol for i in adapter.universe(now.date())]
                    _, degraded = refresh_fundamentals(store, load_cik_map(), symbols, now.date())
                    store.mark_refreshed(now.date())
                    if degraded:
                        extra_warnings.append(
                            "fundamentals refresh degraded: some symbols kept stale values"
                        )
                except Exception as exc:
                    extra_warnings.append(
                        f"fundamentals refresh failed ({exc}); rankings use last stored values"
                    )
```

Then DELETE the later re-initialization (currently just above the earnings block):

```python
        extra_warnings: list[str] = []
```

(the earnings block keeps appending to the same list).

- [ ] **Step 4: Run the runner tests, then the full suite**

Run: `uv run pytest tests/test_runner.py -v`
Expected: all PASS (old tests too — the default momentum_v1 config skips the whole block).
Run: `uv run pytest -q`
Expected: all green.

- [ ] **Step 5: Lint + commit**

```bash
uv run ruff check . && uv run ruff format .
git add src/trading/runner.py tests/test_runner.py
git commit -m "Refresh fundamentals weekly in the live runner, fail-open [AI]

Runs only when the configured ranker requires fundamentals; marker-gated
weekly cadence; degraded/failed refreshes journal a warning and the run
proceeds on stored values (earnings pattern). momentum_v1 venues untouched.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01WuC8FbGLPEZUjYmdPSMMdv"
```

---

### Task 11: Full backfill run + live verification + docs

Network task, run on this machine. Everything before this task is offline-tested; this task proves the pipeline against real EDGAR data and journals the evidence in docs.

**Files:**
- Create: `scripts/verify_fundamentals.py`
- Modify: `README.md` (fundamentals section)
- Modify: `src/trading/venues/universes/sources/PROVENANCE.md` (EDGAR entry)

**Interfaces:**
- Consumes: everything from Tasks 1–10; `data/edgar-raw/*.zip` (downloaded here).
- Produces: a populated `data/fundamentals/equities/` store (local, gitignored), verification evidence recorded in PROVENANCE.md.

- [ ] **Step 1: Run the full 2018q1→present backfill (network, ~35 ZIPs / ~2 GB)**

```bash
uv run python scripts/build_cik_map.py            # refresh only if Task 4's CSV is stale
uv run python scripts/backfill_fundamentals.py 2>&1 | tee /tmp/claude-m4-backfill.log
```

Expected: one `downloading ...` line per quarter (the in-progress quarter may print the 404-skip warning), then a summary like `done: ~1500 filers -> ~800 symbols, ~40000 rows appended`. Rerun the script once and confirm it prints `0 rows appended` (idempotence on real data). If a mid-2020s quarter 404s unexpectedly, check the URL pattern against https://www.sec.gov/dera/data/financial-statement-data-sets — SEC has moved this page before; fix the constant, do not hand-download.

- [ ] **Step 2: Write the verification script**

Create `scripts/verify_fundamentals.py`:

```python
"""One-shot verification of the M4 fundamentals backfill. Run AFTER
scripts/backfill_fundamentals.py (re-parses the cached ZIPs; no network).

Check 1 -- AAPL PIT spot-check, TTM basis. The xbrl-scout reported AAPL's
2023-02-03 10-Q SINGLE-QUARTER gross profitability as 0.1452 =
(117,154 - 66,822) / 346,747 ($M). On the locked TTM basis the expectation
recomputes as (all $M, all from ORIGINAL filings):

    FY2022 10-K (filed 2022-10-28):  revenue 394,328  cogs 223,546
    Q1 FY22 10-Q (filed 2022-01-28): revenue 123,945  cogs  69,702
    Q2 FY22 10-Q (filed 2022-04-29): revenue  97,278  cogs  54,719
    Q3 FY22 10-Q (filed 2022-07-29): revenue  82,959  cogs  47,074
    derived Q4 FY22 = FY - (Q1+Q2+Q3): revenue 90,146  cogs 52,051
    Q1 FY23 10-Q (filed 2023-02-03): revenue 117,154  cogs  66,822; assets 346,747

    TTM revenue = 97,278 + 82,959 + 90,146 + 117,154 = 387,537
    TTM cogs    = 54,719 + 47,074 + 52,051 +  66,822 = 220,666
    gross profitability = (387,537 - 220,666) / 346,747 = 0.4813

Check 2 -- restatement regression against REAL data: every (cik, fy, fp)
filed more than once as a plain 10-K/10-Q across 2018+ must appear in the
store ONLY via its earliest accession; later re-filings never leak
(amendment forms are excluded structurally and cannot appear at all).

Usage: uv run python scripts/verify_fundamentals.py
"""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path

import pandas as pd

from trading.fundamentals.cik_map import load_cik_map
from trading.fundamentals.edgar import load_quarter_facts
from trading.fundamentals.store import FundamentalsStore

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "edgar-raw"

AAPL_EXPECTED_GP = (387_537e6 - 220_666e6) / 346_747e6  # 0.4813, derivation above
AAPL_TOLERANCE = 0.002


def check_aapl(store: FundamentalsStore) -> None:
    frame = store.read("AAPL")
    ts = pd.Timestamp("2023-02-03", tz="UTC")
    if ts not in frame.index:
        sys.exit("FATAL: AAPL has no row at filed date 2023-02-03; backfill incomplete?")
    row = frame.loc[ts]
    gp = float(row["gross_profitability"])
    print(
        f"AAPL @ 2023-02-03: gp={gp:.4f} (expected ~{AAPL_EXPECTED_GP:.4f}), "
        f"revenue_ttm={row['revenue_ttm']:.0f}, cogs_ttm={row['cogs_ttm']:.0f}, "
        f"assets={row['assets']:.0f}, adsh={row['adsh']}, tags="
        f"{row['revenue_tag']}/{row['cogs_tag']}/{row['assets_tag']}"
    )
    if abs(gp - AAPL_EXPECTED_GP) > AAPL_TOLERANCE:
        sys.exit(f"FATAL: AAPL TTM gross profitability {gp:.4f} != {AAPL_EXPECTED_GP:.4f}")


def check_restatements(store_root: Path, cik_map: pd.DataFrame) -> None:
    ciks = set(cik_map["cik"])
    parts = []
    for zip_path in sorted(RAW_DIR.glob("*.zip")):
        facts = load_quarter_facts(zip_path, ciks)
        if not facts.empty:
            parts.append(facts[["cik", "adsh", "fy", "fp", "filed"]].drop_duplicates("adsh"))
    filings = pd.concat(parts, ignore_index=True).drop_duplicates("adsh")
    later_adshes: set[str] = set()
    dup_groups = 0
    for _, group in filings.groupby(["cik", "fy", "fp"]):
        if group["adsh"].nunique() > 1:
            dup_groups += 1
            ordered = group.sort_values(["filed", "adsh"], kind="mergesort")
            later_adshes.update(ordered["adsh"].iloc[1:])
    if dup_groups == 0:
        sys.exit("FATAL: no re-filed (cik, fy, fp) found across 2018+; scan is broken")
    stored: set[str] = set()
    for path in sorted(store_root.glob("*.parquet")):
        stored.update(pd.read_parquet(path, columns=["adsh"])["adsh"])
    leaked = sorted(later_adshes & stored)
    if leaked:
        sys.exit(f"FATAL: PIT violation -- later re-filings leaked into the store: {leaked[:10]}")
    print(
        f"restatement invariant OK: {dup_groups} re-filed fiscal periods found in the raw "
        f"data; zero later accessions present in the store"
    )


def main() -> None:
    data_cfg = tomllib.loads((ROOT / "config" / "equities.toml").read_text())["data"]
    store_root = ROOT / data_cfg["fundamentals_dir"]
    store = FundamentalsStore(store_root)
    check_aapl(store)
    check_restatements(store_root, load_cik_map())
    n_files = len(list(store_root.glob("*.parquet")))
    print(f"store coverage: {n_files} symbols with fundamentals under {store_root}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run the verification**

Run: `uv run python scripts/verify_fundamentals.py 2>&1 | tee /tmp/claude-m4-verify.log`
Expected output (values close to):

```
AAPL @ 2023-02-03: gp=0.4813 (expected ~0.4813), revenue_ttm=387537000000, ...
restatement invariant OK: N re-filed fiscal periods found in the raw data; zero later accessions present in the store
store coverage: ~800 symbols with fundamentals under .../data/fundamentals/equities
```

If the AAPL check fails, debug with the provenance columns before touching code: read `data/fundamentals/equities/AAPL.parquet` rows around 2022-2023 and compare each quarter's revenue/cogs/tags against the docstring table — a wrong tag pick or a missing 10-Q will be visible per-row. Do not widen the tolerance.

- [ ] **Step 4: One real quality_momentum_v1 ranking**

```bash
uv run trading rankings --venue equities --config-dir config/experiments/quality --top 25 \
    2>&1 | tee /tmp/claude-m4-ranking.log
```

Expected: a ranked table containing the `quality` column; values in [0, 1]; financials (JPM-like COGS-less filers) showing 0.500; no crash, coverage line normal. Sanity: also run the default config (`uv run trading rankings --venue equities --top 5`) and confirm its table has NO quality column and momentum ordering unchanged from before this milestone.

- [ ] **Step 5: Document — README + PROVENANCE**

Append to `README.md` (new top-level section, after the existing backtest/data sections):

```markdown
## Fundamentals overlay (M4)

Point-in-time gross profitability (trailing-4-quarter (Revenue − COGS) / latest
Assets) from SEC EDGAR XBRL, feeding the opt-in `quality_momentum_v1` ranker
(six momentum features + a quality percentile, equal weight over 7; NaN
quality — e.g. financials without a COGS concept — is neutral 0.5). Live and
paper stay on `momentum_v1`; opt in per run with
`--config-dir config/experiments/quality`.

PIT discipline: a value becomes visible at its FILING date; per fiscal period
only the original (earliest-accession) filing counts — amendments and
restatements never rewrite history. Provenance (accession, tags, period, form)
rides on every stored row.

Data flow:

    uv run python scripts/build_cik_map.py          # committed CIK<->symbol map (rare)
    uv run python scripts/backfill_fundamentals.py  # 2018q1->present ZIPs -> data/fundamentals/
    uv run python scripts/verify_fundamentals.py    # AAPL spot-check + restatement invariant

The live runner tops up the store weekly from the companyfacts API — only when
the configured ranker requires fundamentals — failing open (journaled warning,
rankings proceed on stored values) exactly like the earnings fetch. Stores are
append-only: history, once visible, is immutable. Note: with the locked 2018q1
backfill start, TTM warms up through 2018 (mostly neutral quality) until the
FY-2018 10-K wave lands in early 2019.
```

Append to `src/trading/venues/universes/sources/PROVENANCE.md`:

```markdown
## SEC EDGAR fundamentals (M4)
- Backfill: SEC Financial Statement Data Sets quarterly ZIPs, 2018q1 -> present
  (https://www.sec.gov/dera/data/financial-statement-data-sets, US-government
  public domain), cached under data/edgar-raw/ (gitignored).
- Top-up: https://data.sec.gov/api/xbrl/companyfacts/ (same terms).
- Access policy: User-Agent "trading-system travis@launchsupply.com", requests
  spaced under the 10 req/s ceiling.
- Verification (run <DATE>): AAPL 2023-02-03 TTM gross profitability <VALUE>
  (expected 0.4813; single-quarter scout basis was 0.1452 — see
  scripts/verify_fundamentals.py for the recomputation arithmetic);
  restatement invariant passed with <N> re-filed fiscal periods, zero later
  accessions in the store; <M> symbols with fundamentals.
```

Fill `<DATE>/<VALUE>/<N>/<M>` from the Step 3/4 logs.

- [ ] **Step 6: Full suite + lint, then commit**

```bash
uv run ruff check . && uv run ruff format .
uv run pytest -q
git add scripts/verify_fundamentals.py README.md \
        src/trading/venues/universes/sources/PROVENANCE.md
git commit -m "Verify fundamentals backfill against real EDGAR data; document M4 [AI]

Full 2018q1->present backfill run; AAPL 2023-02-03 TTM gross profitability
spot-checked at 0.4813 (scout's 0.1452 was single-quarter basis); real
restatement invariant: no later accession ever enters the store.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01WuC8FbGLPEZUjYmdPSMMdv"
```

Milestone complete: `momentum_v1` behavior bit-identical everywhere (golden green), `quality_momentum_v1` available to `rankings`/`backtest`/`run` via the experiment config, tunable surface unchanged.

