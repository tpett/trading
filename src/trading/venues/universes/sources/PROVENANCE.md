# Point-in-time universe data provenance

## S&P 500 membership
- Source: https://github.com/fja05680/sp500
- File: `S&P 500 Historical Components & Changes (Updated).csv` -> snapshotted as
  `sp500_history.csv`. Note: as of retrieval the upstream repo no longer names
  this file with an embedded date (earlier revisions used a
  `(MM-DD-YYYY)`-suffixed name); `(Updated)` is the current, actively
  maintained variant and is the one used here. A second, unsuffixed file in
  the same repo (`S&P 500 Historical Components & Changes.csv`) embeds
  delisting dates in the tickers themselves (e.g. `AAL-199702`) and was not
  used, since the build script expects plain ticker symbols per snapshot row.
- Pinned commit: `c403a121c2e766840f34837738cdd4725eeda818` (last commit to
  touch this file; repo HEAD at retrieval was `b792557e915703398ef9a67e4b583a37c6ec80d5`)
- Retrieved: 2026-07-04 (UTC)
- Licence: MIT

## Nasdaq-100 membership
- Source: https://en.wikipedia.org/wiki/Nasdaq-100 (current constituents +
  yearly change tables), retrieved by scripts/build_pit_membership.py
- Page revision: https://en.wikipedia.org/w/index.php?title=Nasdaq-100&oldid=1362572078
- Retrieved: 2026-07-04 (UTC)
- Licence: CC BY-SA 4.0 (Wikipedia text/data)

## S&P 400 (MidCap) membership
- Source: https://en.wikipedia.org/wiki/List_of_S%26P_400_companies (current
  constituents + changes table), retrieved by scripts/build_pit_membership.py
- Page revision: https://en.wikipedia.org/w/index.php?title=List_of_S%26P_400_companies&oldid=1362781477
- Retrieved: 2026-07-06 (UTC)
- Licence: CC BY-SA 4.0 (Wikipedia text/data) -- same terms already accepted
  for NDX above.
- Cell splitting: dual-class Wikipedia ticker cells ("UAA/UA") are split into
  one interval row per symbol; validate() hard-fails on any surviving
  non-ticker artifact (slashes, spaces, footnote markers).
- Coverage boundary: the changes table is only reliable back to 2019, so
  sp400 intervals never start before 2019-01-01 (SP400_SINCE in the build
  script) even when a symbol's real membership predates that. Backtests
  reaching further back than 2019 for sp400 simply see fewer sp400 members,
  same shape as the 2017-01-01 floor already applied to NDX/sp500.
- sp400 is opt-in: the live/paper venue config still defaults to sp500+ndx
  only (see config/equities.toml `[universe] indices`); sp400 participation
  is a backtest-only config choice.
- Build validation output (2026-07-06 run):
  `validation OK: drift vs M1 snapshot = 7 symbols; sp400 members today = 400;
  CDK removed 2022-07-06, MDP removed 2020-04-27`
  Spot-check anchors (scout-verified, asserted exactly by the build script):
  CDK Global (ticker CDK) removed 2022-07-06; Meredith Corp (ticker MDP)
  removed 2020-04-27.

## Output
- `../equities_membership.csv` — merged intervals, regenerated only by
  `uv run python scripts/build_pit_membership.py`; treat as frozen data, review diffs.

## Known limitations (annotated on backtest results)
- Ticker renames (FB->META, ANTM->ELV, ...) appear as remove+add: the backtest
  force-exits on the rename date. Conservative, and rare enough to accept.
- Residual survivorship: delisted tickers absent from yfinance are counted per
  session and reported as the coverage ratio on every equities result.
