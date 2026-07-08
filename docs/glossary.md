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
