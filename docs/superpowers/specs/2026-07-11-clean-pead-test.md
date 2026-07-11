# The Clean PEAD Test (pre-registered)

**Status:** pre-registered 2026-07-11, BEFORE the clean run. §20 found a
real-but-fragile positive-surprise drift using a NOISY earnings-day proxy
(companyfacts filed-dates + a gap detector). It cleared 3 of 4 pre-registered
conditions and failed the 4th (present-in-liquid-buckets), with three fragility
flags (outlier-carried mean, wrong-signed short leg, GME contamination). This
spec pins the decisive clean test that resolves whether the LOW/MID-liquidity
residual is a real tradeable edge or a detector-noise / regime artifact.

## 1. The data upgrade

- **Real earnings-announcement dates** = SEC 8-K **Item 2.02** ("Results of
  Operations and Financial Condition") filing dates, pulled from the EDGAR
  submissions API (`data.sec.gov/submissions/CIK##########.json`, the `items`
  field flags 2.02) for the ~1,100 index-name CIKs in `cik_map.csv`, 2019-2023.
  The 8-K is filed same-day/next-day with the earnings press release, so its
  filing date IS the announcement date (±1 day). This removes the detector
  noise that let GME-type non-earnings events and COVID-cluster false events
  contaminate §20.
- Bars: the survivorship-clean tiingo caches (delisted included). SPY for the
  market adjustment. Surprise remains the MODEL-FREE earnings-day reaction
  (close_t0/close_{t0−1}−1) — no consensus estimate is available, unchanged
  from §20 by necessity.

## 2. Construction (frozen)

Per real earnings event at date `t0`: enter at the **t0 CLOSE** (after the
announcement reaction — no look-ahead into the gap). Measure the stock's
forward return at **5, 21, 42, 63** trading days MINUS SPY over the identical
window (market-adjusted / active drift). Positive-surprise = positive
earnings-day reaction (the long, tradeable leg).

## 3. The four fixes over §20 (each targets a specific fragility it exposed)

1. **Real dates** (removes detector noise / GME-type contamination).
2. **Exclude the COVID regime:** drop events with `t0 ∈ [2020-02-15,
   2020-04-30]` (the crash/rebound window that wrong-signed §20's negative leg
   via un-stripped sector beta). Report WITH and WITHOUT for transparency.
3. **Median (typical-event) gate:** report and gate on the **median** drift,
   not just the mean — §20's midcap signal was an outlier-carried mean with a
   negative median.
4. **Momentum control:** bucket by (and/or regress out) each name's pre-earnings
   12-minus-1-month price momentum, to isolate PEAD from ordinary price
   momentum. Report the drift within momentum terciles.

## 4. The frozen decision rule — PEAD is a real tradeable edge here IFF ALL hold

- **(a)** positive-surprise market-adjusted drift is POSITIVE and MONOTONIC in
  surprise-magnitude tercile at the 42d/63d horizons;
- **(b)** CI-clear-of-zero (stationary bootstrap, 95%);
- **(c)** present in the **HIGH-liquidity (fillable)** bucket — not only
  LOW/MID (the §20 failure — the one that matters most for tradeability);
- **(d)** the **MEDIAN** drift is positive (typical event drifts, not just a
  right tail);
- **(e)** it **survives the momentum control** (drift remains within the
  neutral/low pre-earnings-momentum bucket — i.e. it is not just price
  momentum in disguise);
- **(f)** the **negative-surprise leg drifts DOWN** (economic corroboration
  that the effect is earnings-reaction, not a long-only/regime artifact).

Fail ANY of (c)/(d)/(e)/(f) → PEAD is dead in the tradeable band and the search
is over-determined-complete. Pass ALL → the program's first real edge; it then
earns a cost-charged backtest and, only on a developer decision, the reserved
holdout. The 2024+ holdout is NOT spent by this discovery-window study.

## 5. Build scope

- `scripts/fetch_earnings_dates.py` (or research script): EDGAR submissions →
  8-K/Item-2.02 filing dates per cik_map CIK → an `(symbol, earnings_date)`
  table, reusing the existing EDGAR throttle/User-Agent (`fundamentals/edgar.py`
  pattern) and `cik_map`. Report coverage (events/name, spot-check a few known
  earnings dates).
- A refined event-study script consuming those dates + bars + SPY, emitting the
  §4 decision-rule table (surprise × liquidity × momentum × horizon; mean AND
  median; CIs; the negative leg; with/without COVID). Data-only, no engine
  changes.
- Holdout untouched.
