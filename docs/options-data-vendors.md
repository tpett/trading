# Historical options data vendors — evaluated for the IV-skew signal

Decision-ready comparison for **if** the free skew proof-of-concept (Robinhood
data + our own Black-Scholes inversion, see `scripts/option_skew_analysis.py`)
shows signal and we decide to buy a real dataset. Evaluated 2026-07 against our
use case: EOD implied volatility across the strike surface, ~500 US large-caps
(S&P 500 + NDX), history back to ~2018, **survivorship-free** (options on
underlyings that later delisted), precomputed IV a big plus (skip the BS
inversion at scale), macOS/Python delivery, retail budget (~$80–150/mo).

## Bottom line

**Buy ThetaData "Options Standard" at $80/mo.** It's the only vendor that fits
the budget, reaches back past 2018, and hands us **historical precomputed IV +
greeks** on a Mac-native, bulk-friendly interface (no per-call rate-limit fight
during an 8-year backfill). Tradeoff: it gives a per-contract IV grid, not a
ready-made 30-day constant-maturity surface — we interpolate ATM / OTM-put /
OTM-call IV to a fixed tenor ourselves (straightforward). **Verify before
committing:** that options on underlyings which *later delisted* are actually in
its historical tape (built on historical OPRA, so they should be — spot-check a
known delisting).

## Ranking

1. **ThetaData — Options Standard, $80/mo.** History to ~2016, historical IV +
   1st-order greeks precomputed, full US coverage incl. expired contracts, local
   Java terminal + Python SDK (macOS-native), bulk endpoints, no REST metering.
   (Options Pro at $160/mo extends to ~2012 + higher-order greeks — not needed
   for our window.)
2. **ORATS — Data API (Delayed), $99/mo.** Gives the *signal precomputed* — a
   smoothed IV surface with constant-maturity points, surface skewness/kurtosis,
   IV rank/percentile, back to 2007. Not first because the $99 tier is capped at
   **20k API req/mo** — far too tight for 500 names × ~2,000 days — and the clean
   bulk history file is ~**$2,000** (over budget). Best if we want turnkey skew
   metrics and will backfill going-forward or pay the one-time $2k.
3. **IVolatility.com — Data Download.** Best on the survivorship axis:
   *explicitly includes delisted names*, IV surface by moneyness + constant-
   maturity IVX, back to 2000, Mac-friendly flat files, ~70% retail discount.
   Third only because pricing is opaque (pay-per-use, quoted after order build);
   get a direct quote if survivorship is the top priority.

## Dealbreakers / traps

- **Polygon.io ("Massive" post-Oct-2025 rebrand) — the IV trap.** Its greeks and
  IV exist **only in the realtime snapshot; there is NO historical IV/greeks.**
  For a historical skew backtest you'd invert Black-Scholes yourself anyway, so
  it offers no precompute advantage over cheaper raw-price sources. (~$79–99/mo,
  unverified post-rebrand.)
- **OptionMetrics IvyDB / dxFeed** — the academic gold standard (survivorship-
  free, full surface, back to 1996) but institutional-only, $thousands/yr or
  WRDS-gated. Out of reach for an individual.
- **Intrinio** (~$1,000/mo for options) and **EODHD** (history only since Q4
  2023) — fail on price and 2018-depth respectively.
- **No Windows-only vendors** in this category — every serious contender is
  REST/flat-file/Java and runs on macOS. The Norgate problem does **not** recur.

## Cheaper / DIY fallback

**Databento OPRA** (pay-as-you-go, or $199/mo Standard): cleanest survivorship-
inclusive raw historical NBBO back to ~2010, excellent bulk download, Mac-native
— but **no IV/greeks**, so we own the full BS-inversion + surface-interpolation
pipeline. Only saves money if we value that modeling work at ~zero; ThetaData's
$80/mo effectively buys us out of the per-contract IV inversion for less than
Databento's subscription tier. (The Robinhood free-data POC is essentially a
tiny version of this DIY path — if it shows signal, ThetaData scales it
survivorship-free without the inversion burden.)

## Not verified (behind sales walls)
Concrete $/mo for IVolatility, Cboe DataShop Volatility Surfaces, dxFeed, and
OptionMetrics; exact post-rebrand Polygon/Massive options pricing; and explicit
delisted-*underlying* survivorship for ThetaData/ORATS (spot-check needed).

Comparison table and source URLs: see the vendor-evaluation agent transcript
(2026-07-07). Key sources: thetadata.net/pricing, orats.com/data-api,
ivolatility.com/historical-options-data, databento.com/datasets/OPRA.PILLAR.
