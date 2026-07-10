# Glossary

A running reference for the quant/finance terms that come up while building this
system. Grounded in our own results where it helps. Organized by theme; new terms
get added to the relevant section as they come up — ask and it grows.

---

## Return, skill, and what you're actually being paid for

**Alpha** — return *above* what your factor exposures explain: the part attributable
to skill, not to riding a known reward. It's the intercept of a factor regression.
Only *statistically significant* positive alpha counts as a real edge. In our work
every strategy's four-factor alpha was indistinguishable from zero (OPT-1: +5.2%/yr
but t=0.88; mid-cap illiquidity: −1.6%/yr, t=−0.39) — i.e. no skill.

**Beta** — how much of a factor's return your strategy inherits (its *loading* on that
factor). "Beta" unqualified usually means **market beta**: 1.0 = you move one-for-one
with the market, 0.5 = half as much. A long-only book is mostly market beta by
construction — which is why "made money in a bull market" proves nothing. OPT-1's
market beta was 0.46 (dampened by its regime gate).

**Factor / risk premium** — a systematic, *rewarded* source of return shared by many
stocks (market, size, value, momentum…). Not skill — you're paid for bearing a common
risk, and anyone can buy the exposure via an index or factor ETF. The whole point of a
factor regression is to strip these out so only skill (alpha) remains.

**Excess return** — return above the risk-free rate (`return − RF`). Factor models run
on excess returns because the factors themselves are excess/spread returns.

**Risk-free rate (RF)** — the T-bill rate; what you earn taking no risk. The baseline
you must clear before any risk-taking is worthwhile.

**Loading / coefficient / exposure** — synonyms for the weight on a factor in a
regression (its slope). "How much of that factor am I?" Measured from *returns*.

**Characteristic vs loading** — a *stock* has a characteristic (it *is* small); a
*strategy* loads on a factor (it *behaves like* it holds the small-minus-big
portfolio). The regression measures loadings, not characteristics.

---

## The factors (each is itself a long/short portfolio return)

**Mkt-RF** ("Market minus Risk-Free"), the **market factor** — the whole US market's
return minus the T-bill rate; the reward for bearing market risk. Your loading on it is
your market beta.

**SMB** ("Small Minus Big"), the **size factor** — return of a portfolio long small-cap
stocks, short large-caps. Positive when small beats big. A positive loading means your
strategy behaves like it's tilted toward small companies. Our mid-cap illiquidity
strategy loaded +0.35 (t=13.1) — the formal proof it *is* the size factor.

**HML** ("High Minus Low"), the **value factor** — long "high" book-to-market (cheap /
value) stocks, short "low" (expensive / growth). Positive when value beats growth. A
positive loading = value tilt; negative = growth tilt.

**Mom** (also **UMD** "Up Minus Down", or **WML** "Winners Minus Losers"), the
**momentum factor** — long recent winners (up over ~12 months), short recent losers. A
positive loading = your strategy rides momentum. OPT-1 loaded +0.27 (t=10.9) — its
"skew" signal was quietly a momentum bet.

Note: every factor is a **long/short spread** between two groups sorted on one
characteristic. That construction cancels market beta, isolating the single dimension —
the same reason a long/short strategy isolates alpha (see *market-neutral*).

---

## Models that measure alpha

**CAPM** (Capital Asset Pricing Model) — the original one-factor model: return explained
by the **market** alone. "CAPM alpha" = return the market doesn't explain (a.k.a.
**Jensen's alpha**).

**Fama-French 3-factor** (1993) — market + size (SMB) + value (HML).

**Carhart 4-factor** (1997) — Fama-French 3-factor **plus momentum (Mom)**. This is our
default "is it real?" model. "Four-factor alpha" = return *none* of the four explain — a
stricter, more honest bar than CAPM.

**Fama-French 5-factor** — adds profitability and investment factors. We haven't needed
these.

**Jensen's alpha** — the CAPM intercept: alpha measured against the market only. The
classic single-factor skill measure.

**Why print CAPM and four-factor side by side** — the *gap* is diagnostic. A CAPM-alpha
that **vanishes** in the four-factor model was really a size/value/momentum tilt the
market-only model couldn't see. Our illiquidity strategy: CAPM alpha −3.5% → four-factor
−1.6%, as SMB absorbed the difference. That shrinkage *is* the size factor made visible.

---

## Statistics for judging a number

**t-statistic (t-stat)** — how many standard errors an estimate sits from zero, i.e.
how much to believe it isn't noise. Rule of thumb: **|t| > ~2 ≈ statistically
significant** (roughly <5% chance of a fluke). It's the difference between "the estimate
is +5%" and "we *believe* +5%." OPT-1's alpha (+5.2%/yr, t=0.88) is under one standard
error from zero → the truth could be −6% or +16% → we've learned nothing. SMB's t=13.1
is overwhelming → that exposure is unmistakable. *A big point estimate with a small t is
just noise wearing a nice number.*

**R² (R-squared)** — the fraction (0–1) of your return's variation the model explains.
OPT-1's 0.39 = the four factors explain 39% of its day-to-day movement; the rest is
stock-specific noise, not alpha. High R² is not the goal — high R² with zero alpha just
means "a well-diversified bundle of factors."

**Significant loading** — a factor beta with |t| > 2: you're genuinely exposed to it.

**Information coefficient (IC)** — for a *ranking* signal, the mean cross-sectional rank
correlation (Spearman) between the signal and the forward return, averaged over periods,
with a t-stat. Measures how well the signal sorts winners from losers. Our option-spread
signal had IC t=4.1 quarterly — a real sort, but it still didn't survive costs (see
`scripts/signal_scan.py`).

---

## Isolating alpha (removing beta)

**Benchmark / hurdle** (e.g. beating SPY's Sharpe) — a bar to clear, **not** beta
removal. A high-beta strategy can clear it in a bull market with zero alpha, so it's
necessary but not sufficient.

**Long-only** — a strategy that only buys. Mostly market beta by construction; a rising
market lifts the whole book regardless of the signal, hiding whether the signal added
anything.

**Long/short** — long the good names, short the bad *at the same time*. The market moves
cancel between the legs, leaving the signal's own contribution = the alpha.

**Market-neutral construction** — building the book so its net market exposure (beta) is
~zero, so realized return *is* alpha. Our cross-sectional **tercile spread** (top-third
minus bottom-third return) is this, and it's how we proved the skew signal had ~zero
alpha.

**Beta-hedging** — keep your long book but short an amount of index/futures equal to its
beta, neutralizing market exposure and leaving the alpha stream.

**Factor regression (Jensen / Carhart)** — the *statistical* route: regress the
strategy's excess returns on the factors and read the intercept (alpha) and loadings.
Our `scripts/factor_regression.py` does this. Complements the *structural* route
(long/short construction).

---

## Evaluation methods (how we measure a strategy here)

**Sharpe ratio** — annualized risk-adjusted return: `(return − risk-free) / volatility`.
Our system uses cash = 0%, so effectively `mean / stdev × √periods`. The go-live gate is
beating SPY's Sharpe with a positive return — but see *benchmark* (it's a hurdle, not
proof of skill).

**Full walk-forward backtest** — a day-by-day simulation of the actual strategy: real
slippage, the regime gate, ATR stops, position sizing, T+1 settlement, per-window
hyperparameter tuning. The authoritative, cost-and-risk-adjusted number. ~9–15 min to
run on a ~100–150 name universe.

**Cross-sectional screen** — a fast, *idealized* portfolio sort: each period rank the
names, form equal-weight buckets, look up forward returns. No costs, no stops, no risk
model. Runs in *seconds*. Great for triaging signal ideas, but it **overstates** what
survives real trading — never compare a screen number to a backtest number.

**Walk-forward validation** — rolling out-of-sample testing: tune the hyperparameters on
a training window (e.g. 30 months), test on the next (e.g. 3 months), roll forward, and
report only the *stitched out-of-sample* segments. Guards against overfitting by never
testing on data used to tune.

**Tercile spread** — mean forward return of the top third minus the bottom third by a
signal. A model-free, market-neutral read on whether the signal predicts returns (a
long/short proxy — see *market-neutral*).

**Slippage** — the cost of trading (paying the bid-ask spread + market impact), modeled
in basis points (bps). We use 5 bps for liquid large-caps, 15 bps for mid-caps. Load-
bearing for an illiquidity signal, which deliberately targets the widest-spread names.

**Survivorship bias** — inflating a backtest by testing only on names that survived
(excluding delisted/failed companies). Our data is survivorship-free (Tiingo, delisted-
inclusive); removing this bias cut earlier momentum results by ~0.14 Sharpe.

---

## Project-specific signals

**IV skew** (`skew_put_atm`) — implied vol of an out-of-the-money put minus the at-the-
money IV. A steep (large positive) skew = the market paying up for downside protection.
Our lead options hypothesis (steep skew → lower forward returns); refuted as tradeable
alpha.

**Risk-reversal / "excitement"** (`−skew_put_call`) — call-side IV richness (OTM call IV
minus OTM put IV). High = the options market pricing upside demand. Our reframed "buy the
excited names" signal.

**Hedge veto** — a *risk filter*: drop the most downside-hedged names (top third by
`skew_put_atm`) before selecting, on the logic that heavily-hedged names are priced for a
big move. Helped as risk control in mid-caps, but the residual edge was the size factor.

**Illiquidity premium / size premium** — the tendency of smaller, less-liquid, more-
neglected stocks to earn higher returns (the SMB factor). Real in our mid-cap data but a
known factor, cost-sensitive, and not an options edge.

---

## Multiple testing, and how the alpha-search engine stays honest

**Trial** — one evaluation of one (signal, universe, window, parameters)
combination. Every trial is journaled (`journal/alphasearch-trials.jsonl`)
whether it succeeds, fails, or errors — because the significance bar depends on
how many things were tried, an uncounted trial silently corrupts every later
p-value. Identical re-runs update in place (same `config_hash`); ANY parameter
change is a new trial.

**Multiple testing / data snooping** — run enough tests and "significant"
results appear by pure luck: at |t|>2, roughly 1 in 20 dead signals looks alive.
A sweep engine is a false-positive factory unless the bar rises with the number
of trials. This is the central risk the alpha-search engine is built around.

**False discovery rate (FDR)** — the expected fraction of your *accepted*
signals that are actually false. Controlling FDR at q=0.10 means: of the
candidates the gate passes, ~10% are expected to be flukes — a deliberate,
quantified tolerance, instead of the unquantified optimism of eyeballing top
rows.

**Benjamini-Hochberg (BH)** — the standard step-up procedure that controls FDR:
sort the n trial p-values ascending, find the largest k with
p_(k) <= (k/n)·q, accept the k smallest. The bar RISES as the journal grows —
running more trials makes every existing candidate harder to accept, which is
exactly the honesty we want. Our gate: q=0.10 across ALL journaled discovery
trials.

**Deflated Sharpe Ratio (DSR)** — Bailey & López de Prado's correction to
"best backtest of N": the probability the candidate's true Sharpe exceeds zero
after accounting for how many trials were run, the spread of Sharpes across
them, and fat tails/skew in the candidate's returns. Reported (advisory) for
BH survivors; DSR near 1 = likely real, near 0.5 = coin flip.

**Portfolio sort** — the cheap way to turn a signal into a return series
without a backtest: each month, rank the universe by the signal, go long the
top quantile and short the bottom (equal weight), hold to the next rebalance.
The daily long/short spread isolates the signal from the market; its
four-factor alpha t-stat is our gate statistic. Quintiles normally, terciles
under 50 names, skip (and journal) under 15.

**Discovery vs holdout window** — discovery (2019-01-01..2023-12-31) is where
the sweep is allowed to look; the holdout (2024-01-01..latest) is spent the
FIRST time a candidate reads it, enforced by the trial journal exactly like
the go-live holdout. Pass rule, pre-registered: same alpha sign AND >= 50% of
the discovery alpha magnitude retained.

**SIC code** — Standard Industrial Classification: the 4-digit industry code the SEC
records for each filer (e.g. 2836 = Biological Products, 6021/6022 = commercial banks,
7372 = Prepackaged Software). We read each filer's *current* code from
`data.sec.gov/submissions` into the committed `sic_map.csv` and apply it backward over
the whole discovery window — a disclosed caveat, since companies occasionally
reclassify.

**Segment universe** — a pre-registered slice of a cap pool by SIC range
(`trading.alphasearch.segments.SEGMENTS`, frozen before any segment sweep): ten coarse
sectors plus the fine industries biotech and banks, which deliberately overlap their
parents. Each segment is an ordinary sweep universe, so every (signal, segment) pair is
an honestly-counted extra trial — the BH bar spans flat + segment trials in the one
journal, meaning segmentation *raises* the significance bar; it can never lower it.

## The anomaly zoo (Tier-1 signal batch)

**Momentum 12−2** — total return from 12 months ago to 1 month ago, skipping the
most recent month because short-term reversal contaminates it (Jegadeesh-Titman).
The canonical UMD construction.

**Overnight return persistence** — split each day into overnight (prev close→open)
and intraday; the overnight component carries persistent, clientele-driven
momentum of its own (Lou-Polk-Skouras).

**Parkinson volatility** — a range-based vol estimator using ln(High/Low)²/(4·ln 2);
~5x more statistically efficient per day than close-to-close vol. Feeds the
low-vol anomaly.

**Idiosyncratic volatility (IVOL) puzzle** — the std of a stock's daily FF3
regression residuals; HIGH-ivol stocks anomalously UNDERPERFORM
(Ang-Hodrick-Xing-Zhang), the opposite of risk-reward intuition.

**MAX / lottery demand** — the mean of a stock's few biggest recent daily gains;
lottery-seekers overpay for jackpot-shaped names, which then underperform
(Bali-Cakici-Whitelaw).

**Betting against beta (BAB)** — leverage-constrained investors overpay for high-β
stocks, flattening the security market line; low-β outperforms per unit of risk
(Frazzini-Pedersen).

**Amihud illiquidity** — mean |return| / dollar volume: price impact per dollar
traded. Illiquid names earn a premium for being hard to trade.

**High-volume return premium** — unusually high recent trading volume (vs its own
baseline) attracts attention and predicts higher returns
(Gervais-Kaniel-Mingelgrin).

**Dividend yield (`div_yield`)** — trailing 12-month cash dividends per share
over price; an income/value tilt. Point-in-time subtlety (2026-07-09 fix):
divided by `close_raw` (the RAW, never-retroactively-adjusted close), not
Tiingo's `close`, because `close` is adjusted using the FULL downloaded
history and so bakes in FUTURE splits at any given as-of date — raw
div_cash / adjusted close would look-ahead-inflate the yield for a future
splitter (a 4:1 split after a $1 dividend reads 4x too high beforehand).
Each trailing payment is itself split-adjusted into as-of terms using only
the VISIBLE (<=as-of) split_factor history.

**Asset growth / investment factor** — firms that grow total assets fastest
subsequently underperform (Cooper-Gulen-Schill); empire-building is expensive.
The CMA factor's characteristic cousin.

**Net share issuance** — split-adjusted growth in shares outstanding; issuers
underperform, buyback firms outperform (Pontiff-Woodgate) — management times the
market with its own stock.

**Fundamental momentum (ΔROA)** — the year-over-year CHANGE in profitability;
improving firms keep outperforming beyond what the level explains.

**Option-to-stock volume ratio (O/S)** — option dollar volume relative to stock
dollar volume; high O/S marks informed (disproportionately bearish) positioning
ahead of returns (Johnson-So).

**Call-put volume skew (`cp_vol`)** — log(1 + ATM+call leg volume) − log(1 +
put leg volume) on the day's options cell; informed call demand predicts
positive returns (Pan-Poteshman). Requires per-leg volume (mid-cap gather
only — `SignalSpec.requires_option_volume`); a cell where NO leg carries a
"volume" key scores NaN rather than the fabricated log(1/1) = 0 a
key-absent-defaults-to-zero read would produce (2026-07-09 fix).

**Volatility smirk** — OTM-put implied vol above ATM implied vol; a steep or
steepening smirk means crash protection is being bid, predicting negative
returns (Xing-Zhang-Zhao). IV *innovations* (An-Ang-Bali-Cakici) time-difference
the level: rising implied vol = rising perceived risk.

**Industry momentum** — a sector's own trailing return, assigned to every member;
industries trend (Moskowitz-Grinblatt), and much of stock momentum is industry
momentum.

**Within-industry reversal** — a stock's SHORT-term return relative to its sector
mean reverses (Da-Liu-Schaumburg): sector-adjusted laggards bounce, leaders fade.

## Insider transactions (the Form 4 family)

**Form 4** — the SEC filing corporate insiders (officers, directors, >10%
owners) must submit within 2 business days of trading their own company's
stock. Our source is the SEC DERA "Insider Transactions Data Sets" quarterly
bulk files: official, free, as-reported forever (never restated), covering
delisted names — the cleanest PIT anomaly source we identified. The store
keeps only open-market purchases (`P`) and sales (`S`); awards, exercises,
gifts and plan transactions are excluded. 10b5-1 (pre-scheduled) trades are
NOT excluded — the flag is unreliable before 2023 (documented limitation).
Form 4/A amendments carry their OWN accessions and the frozen spec has no
form-type filter, so an amended transaction can appear twice (original +
amendment) — slight over-counting, a documented limitation. PIT discipline:
every signal keys the FILED date, never the transaction date (which precedes
filing and would be look-ahead).

**Net purchase ratio (NPR, `npr_90`)** — (buy dollars − sell dollars) /
(buy dollars + sell dollars) over the trailing 90 filed days
(Lakonishok-Lee). +1 = insiders only bought, −1 = only sold. Sales carry
less information than purchases (diversification, taxes, option vesting);
purchases are the deliberate act. NaN when the trailing window has no
priced P/S rows — a quiet window scores NaN, never a fabricated 0/0.

**Cluster buying (`cluster_buys_90`)** — the count of DISTINCT insiders with
at least one open-market purchase in the window. Several insiders buying
independently is far stronger evidence than one large buy. 0 is a real value
(a covered, quiet name); NaN is reserved for names with no Form 4 history at
all as of the date (never-covered ≠ quiet).

**Officer purchases (`officer_buy_90`)** — officer open-market purchase
dollars scaled by market cap on the RAW price basis (shares outstanding ×
raw unadjusted close — the div_yield lesson: adjusted closes bake in future
splits, and Form 4 dollars are raw dollars). Officers have the best
information per dollar traded. Requires both the insider store and
fundamentals (`shares_outstanding`), so it registers with both refusal
flags. NaN when `shares_outstanding` or `close_raw` is unavailable at the
as-of date (no market-cap denominator, no score).

## The robustness battery (Piece 3)

*R1 re-anchor note (2026-07-10):* the check descriptions below were written
for the original L/S-alpha anchor. Since the R1 amendment, checks 1-6 keep
their frozen THRESHOLDS but score the cost-charged LO-minus-SPY **active
return series** (see "the long-only gate" section) — read "alpha" in the
retention/sign rules below as "annualized active return", and check 1's t as
the active series' mean/se t.

- **Robustness battery** — the pre-registered, frozen set of seven checks a
  BH survivor must face before it may spend its once-only holdout touch
  (`trading alphasearch robustness <signal>:<universe>`). Pre-committed
  interrogation instead of ad-hoc survivor-poking: the checks and thresholds
  were written down before any survivor was examined, so passing them can't
  be the product of tweaking the exam after seeing the student. Caveat: its
  12 journaled re-evaluations are near-duplicates of ONE effect, not
  independent hypotheses, so a battery run can pseudo-replicate its way into
  lowering the BH bar for an UNRELATED marginal trial elsewhere on the
  leaderboard (experiments log, §11 caveats) — read any post-battery
  promotion as skeptically as a first-pass survivor.
- **Sub-period halves (check 1)** — re-run the discovery evaluation on each
  half of the window (2019-01..2021-06 / 2021-07..2023-12). A real effect
  shows the same sign in both halves with |t| ≥ 1; a one-regime wonder
  doesn't.
- **Universe-subset draws (check 2)** — five seeded random half-universes
  (seed 42+i, sorted draws). An alpha carried by the breadth of the
  cross-section survives ≥ 4 of 5; one carried by a few lucky names doesn't.
- **Parameter jitter (check 3)** — the same evaluation at quantiles {4, 6} ×
  min_names {10, 20}. Real effects don't care exactly where the bucket
  boundaries fall. Caveat: on a universe whose cross-section is below
  `tercile_below` (50 names), `portfolio_sort`'s tercile fallback collapses
  BOTH quantile settings to 3 buckets regardless of the requested 4 or 6, so
  check 3 there effectively only exercises the min_names dimension and
  journals pairwise-duplicate p-values (same q=3 sort at each min_names
  value, twice) — expected, disclosed, not a bug.
- **Decision-date offset (check 4)** — rebalance on the second trading
  session of each month instead of the first. Guards against calendar-turn
  artifacts; the alpha must keep its sign and half its magnitude.
- **Name concentration (check 5)** — recompute the L/S series with the top-3
  contributors to the long leg removed. A "three-name alpha" collapses; a
  broad one retains ≥ half its point estimate.
- **Month concentration (check 6)** — the top-3 calendar months' share of
  the cumulative L/S log return must be ≤ 60%; otherwise the "alpha" is a
  couple of episodes, not a process.
- **Factor-proxy flag (check 7)** — warning-only: any factor loading with
  |t| more than twice the alpha's |t| while regression R² > 0.5. The §9
  SMB-costume detector — the series is mostly a factor bet wearing an alpha
  costume.
- **Cost-adjusted alpha** — the L/S series re-regressed after charging
  parametric one-way costs (10/30/50 bps × turnover × both legs) at every
  rebalance. DIAGNOSTIC only since the R1 amendment (below) — no longer read
  by the eligibility gate, still printed as a fragility read.
- **Amihud λ (price impact)** — |daily return| / dollar volume, averaged
  over 252 days: the price move a dollar of trading buys. The amihud
  signal's own construction, reused as an impact price.
- **Capacity curve** — net alpha at book sizes $10k/$100k/$1M per side,
  charging each rebalanced name its own λ × (book / names-per-leg) on entry
  and exit. First-order model — an honest sketch of how fast paper alpha
  drowns in impact, not a fill simulator. Diagnostic only (see above).
- **Holdout-eligible** — the battery's verdict, journaled as one
  `kind="battery"` event per candidate. Since the R1 amendment (below):
  checks 1-6 pass AND the cost-charged long-only series beats SPY
  buy-and-hold (Sharpe and total return, both over discovery) — see "the
  long-only gate" section. The holdout command refuses candidates without
  it — a written prospective amendment to the Piece 1 holdout protocol.

---

## The long-only gate (R1 amendment, 2026-07-10)

The engine's original gate (four-factor L/S alpha) certifies returns in a
construction this account cannot trade — a long/short spread strips out the
four premia a long-only investor actually collects. "Beating the market"
from this seat means the long-only portfolio, after realistic costs,
outperforms SPY — that's what the amended gate measures. See
`docs/superpowers/specs/2026-07-10-longonly-gate-amendment.md`.

- **Long-only gate / promotion statistic** — the signal's existing `lo`
  series (equal-weight top quantile, monthly rebalance — Piece 1's
  tradability annotation, now promoted to the actual gate), charged
  spread-based costs (below), evaluated over the discovery window. A
  candidate is promotion-eligible when its annualized Sharpe ≥ SPY
  buy-and-hold's AND its total return exceeds SPY's, both over the
  identical window. The four-factor L/S regression is retained as a
  mandatory DIAGNOSTIC (alpha, loadings, R² printed with every candidate —
  know what you're being paid for) but is no longer the filter; BH-FDR
  keeps running on the L/S p-values unchanged and becomes a reported
  property, not the gate.
- **Corwin-Schultz (CS) effective-spread estimator** — a 2012 estimator of a
  stock's bid-ask spread from daily high/low prices alone (no quote data
  needed): two consecutive days' own high-low ranges (β) vs the TWO-DAY
  high-low range spanning both (γ) isolate the spread component from the
  volatility component (`trading.alphasearch.costs.trailing_effective_spread`).
  Implementation note: β and γ are averaged over the trailing 21-session
  window BEFORE the (nonlinear, two-nested-sqrt) alpha/spread transform is
  applied. This is a documented BIAS CORRECTION, ratified 2026-07-10 as a
  deviation from both the amendment spec §3's literal per-day text and the
  CS 2012 paper's own baseline (which computes a spread per two-day pair and
  averages those): at daily granularity the per-pair floor-at-zero-then-mean
  is a one-sided truncation of noise that inflates the result by roughly an
  order of magnitude — a Monte Carlo check (GBM ticks, a KNOWN zero spread)
  reads ~75bps of "spread" out of pure noise per-pair, vs ~12bps
  component-averaged (a residual, conservative window-level bias, mostly
  absorbed by the floor). Only the component-averaged form satisfies §3's
  own AAPL single-digit-bps acceptance criterion. Caveat: no overnight-gap
  adjustment (mildly anti-conservative for gappy names, immaterial under
  the floor). Floored at 2bps (large-cap reality), capped at 5% (data
  sanity).
- **Spread-based rebalance charge** — the R1 cost model: at each rebalance,
  every name ENTERING the long-only leg is charged half its effective
  spread at its new 1/n weight, every name EXITING the analogue at the OLD
  leg size (mirrors the battery's Amihud capacity-curve shape, spread
  instead of impact). Charged to the `lo` series on the first return day
  after the decision date (reuses `apply_rebalance_charges`). Book-size
  impact beyond the spread is NOT modeled at the account's $1k size
  (fractional shares, no commissions make it negligible) — documented,
  revisit if the account grows.
- **SPY buy-and-hold benchmark** — the frozen comparator: SPY's own daily
  closes (`data/equities-tiingo/SPY.parquet`) over the identical window,
  Sharpe and total return computed the same way as the candidate's charged
  `lo` series. Refuses loudly (no silent substitute) when the cache lacks a
  SPY parquet — the whole point is comparing against the ACTUAL benchmark
  an investor in this seat would have held.
- **Active return series (battery re-anchor)** — the daily cost-charged
  long-only return minus SPY's daily return (inner-joined calendars). The
  orchestrator-ratified statistic the battery's checks 1-6 score under R1:
  raw-LO retention across sub-periods/subsets would mostly test whether the
  MARKET regime repeated (beta), not whether the signal's edge over the
  benchmark did — the active series isolates exactly the edge the §2 gate
  certifies. Same frozen thresholds; check 1's t is the active series'
  mean/(sd/√n).
- **`--long-only` leaderboard view** — re-reads every journaled discovery
  trial and re-derives its cost-charged long-only series from CURRENT data
  (the journal keeps summary stats, not the raw daily series), ranking it
  against SPY over the trial's own window. A display, never a
  re-journaling: no trial is re-scored. Trials whose signal/universe no
  longer resolves show honestly as n/a.
