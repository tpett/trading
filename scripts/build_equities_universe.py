"""One-off builder for src/trading/venues/universes/equities.csv.

Provenance (spec open item): current S&P 500 and Nasdaq-100 constituent
tables from en.wikipedia.org, fetched on the run date. This is a static M1
snapshot; point-in-time membership history is Milestone 3.

Run from the repo root:
    uv run --with lxml python scripts/build_equities_universe.py
"""

import io
import urllib.request
from pathlib import Path

import pandas as pd

SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
NDX_URL = "https://en.wikipedia.org/wiki/Nasdaq-100"
OUT = Path("src/trading/venues/universes/equities.csv")

# Wikipedia rejects urllib's default User-Agent (Python-urllib/x.x) with a 403;
# a descriptive UA is enough to get a normal 200 response.
_HEADERS = {"User-Agent": "momentum-system-universe-builder/1.0 (contact: travis@launchsupply.com)"}


def _fetch_tables(url: str) -> list[pd.DataFrame]:
    req = urllib.request.Request(url, headers=_HEADERS)
    with urllib.request.urlopen(req, timeout=30) as resp:
        html = resp.read().decode("utf-8")
    return pd.read_html(io.StringIO(html))


def main() -> None:
    sp500 = _fetch_tables(SP500_URL)[0]["Symbol"].tolist()
    ndx_tables = _fetch_tables(NDX_URL)
    ndx = next(t for t in ndx_tables if "Ticker" in t.columns)["Ticker"].tolist()
    # yfinance uses '-' for share classes (BRK.B -> BRK-B).
    symbols = sorted({str(s).strip().replace(".", "-") for s in sp500 + ndx})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("symbol\n" + "\n".join(symbols) + "\n")
    print(f"wrote {len(symbols)} symbols to {OUT}")


if __name__ == "__main__":
    main()
