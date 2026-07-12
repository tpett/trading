# API-driven trading venues — capability survey & venue decision

Reference for the R6 "expanded capability" work (experiments.md §18–19).
Verified against current sources July 2026 (pricing/API features change fast —
re-verify before acting). This captures WHY the backtest was restructured to an
IBKR-class capability model and WHAT each platform unlocks beyond the live
Robinhood MCP.

## Why this exists

The live account trades via the **Robinhood MCP**, whose hard limits blocked the
interesting plays: **single-leg options only** (no verticals/spreads/straddles/
condors — no defined-risk vol structures), **no short-selling** (long-only
equity), no futures/FX, crypto unavailable to the agent, PFOF routing, single-
level quotes, seconds latency, whole-contract options. The recurring finding was
"the idea isn't the constraint, the platform is," so R6 restructured the backtest
to a more capable venue's capability set (multi-leg options + shorting) at
$12,500 to see if the previously-unreachable plays surface an edge. (They didn't
— §18–19 — because the binding constraint turned out to be market efficiency in
the liquid band, not the platform's features.)

## Two structural changes since older knowledge (verified 2026)

- **The Pattern Day Trader rule was retired (FINRA, effective 2026-06-04).** The
  $25k minimum is gone; only the pre-existing $2,000 margin-account minimum
  remains. A sub-$25k day-trading agent is no longer boxed out.
- **CFTC-regulated US crypto perpetual futures are now API-accessible to US
  retail** (Kraken via Bitnomial, 2026-06-15; Coinbase Financial Markets). The
  old "perps are geofenced from US persons" assumption is now true only offshore.

## Platform comparison (for an autonomous agent)

| Platform | Multi-leg options | Futures (index/VIX) | Short/margin | Streaming greeks | Headless auth | Paper API | Options cost |
|---|---|---|---|---|---|---|---|
| **Interactive Brokers** | Yes | **Yes incl. VIX + micros** | **cheapest borrow** | Yes | OAuth 1.0a self-serve (no gateway) | Yes | $0.15–0.65 |
| **tastytrade** | **Yes (native)** | **Yes (futures + futures options)** | Yes | **Yes (DXLink)** | OAuth2 long-lived refresh | cert/sandbox | **$1 open/$0 close, $10/leg cap** |
| **Alpaca** | Yes (≤4 legs) | **No** | Yes (≥$2k) | Yes ($99 OPRA tier) | **API key/secret (cleanest)** | **Yes (L3 paper)** | ~$0 + pass-through |
| **Tradier** | Yes (≤4 legs) | No | Yes | Yes | Bearer token | sandbox (delayed) | $0.35 or $10/mo flat |
| **TradeStation** | Yes | **Yes incl. VIX** | Yes | Yes | OAuth2 long-lived refresh | **Yes (SIM=Live)** | $0 |
| **Schwab** | Yes | No | Yes | Yes | OAuth2 **7-day manual refresh** | No | $0.65 |
| **E*TRADE** | Yes (spreads) | No | Yes | No streaming | OAuth 1.0a **daily re-login** | No | $0.50–0.65 |
| **Tradovate / AMP** | n/a | index (AMP adds VIX/CFE) | futures margin | Yes + DOM | self-serve key / gateway | Yes (demo) | — |
| **Coinbase / Kraken** | n/a | **US crypto perps** | perps/margin | Yes + full L2 WS | API key + HMAC | sandbox/testnet | — |

Also: **Public.com** (REST + Python SDK + a hosted MCP for LLM agents; single &
multi-leg + index options + crypto; no shorting), **Webull/Moomoo OpenAPI**,
**Lime** (true DMA/FIX + locates, ~$30k min), **DAS/Cobra/CenterPoint** (short
locates). Data-only: **Polygon/Massive** and **Theta Data** (cheap streaming
greeks/IV, absorb the ~$2k/mo OPRA fee); **Databento** (order-book depth).

## Capability → plays unlocked

- **Multi-leg (defined-risk) options — the #1 unlock.** Converts naked short-vol
  (uncapped, un-runnable small) into tail-capped credit spreads / iron condors.
  Crucially it **cuts per-position capital ~30×** (a $5-wide put spread risks
  ~$500 vs ~$5–15k of cash-secured-put collateral), dropping a diversified
  VRP-harvest book's minimum from ~$75–100k to ~$10–15k. Best: tastytrade (economics)
  or IBKR (breadth). *Tested §19 — no edge on liquid options anyway.*
- **Shorting** (IBKR cheapest) → market-neutral / pairs / stat-arb. *Tested §18 —
  amihud market-neutral is the illiquidity artifact; wash on liquid names.*
- **VIX / index micro futures** (IBKR/tastytrade/TradeStation) → direct vol carry;
  micros (VXM/MES) fit small size but carry blow-up risk (Feb-2018).
- **Crypto perps 24/7** (Kraken/Coinbase) → funding/basis carry (real, scales
  down); arb/market-making latency-dominated at retail.

## The venue decision (R6)

**Design target: Interactive Brokers-class** — the one venue covering everything
the search wanted to test (multi-leg options + equity shorting + VIX/micro
futures + cheapest borrow + headless OAuth). tastytrade wins if the scope ever
narrows to options+futures vol only (best economics + native multi-leg). Alpaca
is the best *paper/prototype* harness (API-key auth, L3 paper) but lacks futures.
Avoid Schwab/E*TRADE for autonomy (7-day / daily manual auth breaks unattended
operation). **No live API integration was built** — only the backtest's
capability model was expanded.

## Honest caveats (why several "unlocks" are traps even when available)

- Multi-leg still pays the retail spread on 2–4 legs with no options-book depth
  (OPRA is NBBO-only) — execution discipline, not geometry, decides P&L.
- Futures add leverage/blow-up risk that dwarfs the options tail you just capped.
- Shorting/stat-arb is capital-hungry and edge-thin at retail size (real as a
  hedge, a scale play as a standalone book).
- Crypto API access ≠ crypto edge; the defensible plays are the slow carry ones.
- DMA ≠ HFT-competitive; treat order-book depth as context, not a latency edge.

**Bottom line:** the biggest real unlock is multi-leg defined-risk options
(tastytrade/IBKR); the venue expansion was worth *testing* but did not change the
answer — see experiments.md §18–19 and the terminal conclusion.
