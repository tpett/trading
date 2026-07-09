# Tier-1 Signal Batch — 21 pre-registered signals from on-disk data (design spec)

**Status:** approved design, 2026-07-09. Extends the alpha-search engine
(Piece 1, `2026-07-08-alpha-search-engine-design.md`) and segmented universes
(Piece 2, `2026-07-08-segmented-universes-design.md`).
**Scope:** register 21 new signals computable from data already on disk, the
minimal engine extensions they require, and ONE pre-registered full-grid
discovery sweep. Tier 2/3 data sources (ThetaData OI/greeks/term structure,
SEC Form 4 insider, 8-K earnings dates/PEAD, FINRA short interest, 13F,
Wikipedia attention) are explicitly queued OUT of this spec.

## 1. Purpose

The first sweep (238 trials) was an honest null on the 16 seed signals. This
batch widens the hypothesis space using data we already own — unused OHLCV
fields, stored fundamentals primitives, unexploited options-cell fields, and
the committed SIC map — with definitions grounded in the published anomaly
literature, frozen here before any sweep.

## 2. The frozen signal table (pre-registered; sign = higher score more attractive)

All rolling windows are trading days; all values computed strictly as-of the
decision date through PIT accessors.

### Price/volume family (9) — from bar columns `open, high, low, close, volume, div_cash`

| name | definition | sign rationale (literature) |
|---|---|---|
| `mom_12_2` | total return from t−252 to t−21 (skip most recent month) | + classic UMD; skip-month avoids reversal (Jegadeesh-Titman) |
| `overnight` | sum of overnight log-returns (prev close→open) over 63d | + overnight component carries persistence (Lou-Polk-Skouras) |
| `park_vol` | Parkinson range vol over 21d: sqrt(mean(ln(H/L)²/(4ln2)))·√252 | − low-vol anomaly, range estimator |
| `ivol` | std of daily FF3 residuals over 21d (regress excess ret on Mkt-RF, SMB, HML from the cached factor frame), annualized | − idiosyncratic-vol puzzle (Ang-Hodrick-Xing-Zhang) |
| `max5` | mean of the 5 largest daily returns in past 21d | − lottery demand (Bali-Cakici-Whitelaw) |
| `beta` | OLS slope of daily excess ret on Mkt-RF over 252d (min 126 obs) | − betting-against-beta (Frazzini-Pedersen) |
| `amihud` | mean over 252d of |daily ret| / (close·volume), min 126 obs | + illiquidity premium (Amihud) |
| `vol_trend` | mean dollar volume 21d / mean dollar volume 252d | + high-volume return premium (Gervais-Kaniel-Mingelgrin) |
| `div_yield` | sum of `div_cash` over trailing 252d / close | + income/value tilt |

### Fundamentals family (5) — from stored primitives (`assets, ttm_net_income, book_equity, shares_outstanding, revenue_ttm`), filing-date PIT

| name | definition | sign |
|---|---|---|
| `asset_growth` | assets(latest filing) / assets(latest filing ≥ 300 calendar days older) − 1 | − investment factor (Cooper-Gulen-Schill) |
| `net_issuance` | split-adjusted shares_outstanding YoY change (same 300-day rule) | − issuance anomaly (Pontiff-Woodgate) |
| `roa` | ttm_net_income / assets | + quality |
| `droa` | roa(latest) − roa(one-year-prior filing) | + fundamental momentum |
| `rev_growth` | revenue_ttm YoY (same rule) | + growth |

YoY convention: "one-year-prior filing" = the latest filing FILED at least
300 calendar days before the current filing's filed date; if none exists the
signal is NaN for that name (dropped, never imputed).

### Options family (5) — from existing `samples*.jsonl` cells

| name | definition | sign | universe constraint |
|---|---|---|---|
| `cp_vol` | log(1+call leg volume) − log(1+put leg volume) | + informed call demand (Pan-Poteshman) | requires option volume |
| `osv` | (sum of leg volumes·100·mid·contract) / stock dollar volume, 1 cell vs decision-day stock volume | − option/stock volume ratio (Johnson-So) | requires option volume |
| `otm_put_iv` | OTM put leg IV level | − smirk steepness predicts negative returns (Xing-Zhang-Zhao) |  |
| `iv_change` | atm_iv(current cell) − atm_iv(prior month's cell), NaN if prior missing/stale >45d | − rising vol = rising risk (An-Ang-Bali-Cakici innovations family) |  |
| `dskew` | skew_put_atm(current) − skew_put_atm(prior month), same staleness rule | − smirk steepening bearish (consistent with `hedge` sign) |  |

`requires option volume`: leg `volume` exists only in the mid-cap gather; a
new `SignalSpec.requires_option_volume` flag refuses these signals at sweep
assembly on universes whose cells lack leg volume (mirrors the existing
requires_options mechanism — a refusal, never silent NaN trials).

### Industry-relative family (2) — via committed `sic_map.csv` sectors (the 10 frozen SEGMENTS sectors as the industry partition; unmapped/unsegmented symbols get NaN)

| name | definition | sign |
|---|---|---|
| `ind_mom` | sector cross-sectional mean of mom_12_2, assigned to each member | + industry momentum (Moskowitz-Grinblatt) |
| `ind_rel_rev` | −(trail21 return − sector mean trail21) | + within-industry reversal (Da-Liu-Schaumburg; 21d horizon at monthly cadence) |

## 3. Engine extensions (minimal, all reviewed under the same discipline)

1. `panel.py`: `load_bars` (full OHLCV + div_cash, replacing close-only for
   these needs; `load_closes` remains for compatibility) and a `PanelView`
   as-of accessor for bar frames. `PanelData` gains the daily factor frame
   (`Mkt-RF, SMB, HML, RF` from `evaluate.load_factors`) served as-of — the
   factors are published history; PIT holds by truncation like everything
   else.
2. Heavy rolling features (`ivol`, `beta`) precompute once per symbol
   full-span (vectorized, the FeaturePanel pattern) and are gathered as-of;
   all other signals compute from truncated views. Every new signal is
   automatically covered by the registry-wide no-look-ahead perturbation
   test (it iterates `SIGNALS`).
3. `SignalSpec.requires_option_volume: bool = False` + assembly-check wiring
   (panel exposes whether its cells carry leg volume).
4. **Pre-registered amendment to Piece 2 §3.2:** deep-pool segments now carry
   `fundamentals_dir` when the store exists locally (it was None only because
   no store was backfilled at the time), so the fundamentals family sweeps
   segments as well as flat pools. Recorded as a written prospective
   amendment; no fundamentals segment trial has ever run, so nothing is
   spent.
5. Prior-month options-cell lookup for `iv_change`/`dskew`: PanelView gains
   `option_row_prior(symbol, as_of, max_age_days=45)` returning the
   most recent cell strictly OLDER than the current cell (and within the age
   cap), never a future one.

## 4. Pre-registered sweep (run once, after merge)

- Same locked mechanics: discovery 2019-01-01..2023-12-31, monthly first
  trading session, quintiles/tercile<50/skip<15, equal weight, 4F L/S alpha t
  gate, BH q=0.10 across the ENTIRE journal (238 existing + this batch),
  DSR advisory, holdout untouched.
- Grid: all 21 signals × every compatible universe (flat pools + all emitted
  segments; options family on options universes; option-volume family on
  mid-cap-side options universes only; fundamentals family everywhere a
  store is present post-amendment). Estimated ~450 new trials (exact count
  is whatever the journal records — the estimate binds nothing).
- One leaderboard read. BH survivors (if any) are reported to the developer;
  no holdout touch is spent autonomously.

## 5. Operational prerequisites

- Fundamentals store rsync'd from the mini (`ssh mac-m1`, repo `~/trading`,
  store at `data/fundamentals/equities`) to this machine before the sweep;
  the sweep refuses fundamentals signals if absent (existing actionable
  message).
- Factor cache already covers discovery (ends 2026-05; guard exists).

## 6. Error handling

Existing rules apply unchanged: symbols lacking a signal's inputs on a date
drop from that date's cross-section; all-NaN cross-sections → SortError →
journaled error trial; n≤k regressions fail loudly and count; universe
incompatibilities refuse at assembly. New: `ivol`/`beta` return NaN below
their minimum-observation floors (126d for 252d windows, 15d for 21d
windows); `iv_change`/`dskew` NaN when the prior cell is missing or stale.

## 7. Testing

Per-signal hand-computable unit fixtures (each formula verified on a tiny
constructed series, including sign conventions); no-look-ahead test covers
the registry automatically — plus explicit perturbation coverage for the new
bar/factor/prior-cell accessors; requires_option_volume refusal end-to-end;
YoY filing-selection rule fixtures (300-day boundary); golden sweep fixture
extended to include one signal from each new family; full suite
warnings-as-errors; ruff clean; subagent implementation + adversarial review
per charter §6 (look-ahead, sign-flip, and trial-honesty hunts).

## 8. Queued next (recorded, not in scope)

- **Tier 2:** ThetaData gather extension (open interest, greeks, full smile,
  second-expiration term structure, weekly cadence) + SEC Form 4 insider
  transactions data sets. ThetaData stock-feed research verdict (2026-07-09):
  **augment, don't switch** — delisted-stock coverage is undocumented (treat
  as unconfirmed-negative until an empirical free-tier test on a delisted
  ticker, e.g. XLNX), prices are raw-only (no pre-adjusted series), so Tiingo
  remains the survivorship-free system of record; ThetaData stock NBBO/1-min
  bars (likely already bundled in the $80/mo Standard tier — verify with
  support) are a candidate microstructure/intraday signal source for listed
  names only.
- **Tier 3:** 8-K Item 2.02 PIT earnings dates → PEAD; FINRA short interest
  (publication-date PIT); 13F crowding; Wikipedia pageview attention.
- Analyst revisions: no free PIT source — out until paid data is approved.
