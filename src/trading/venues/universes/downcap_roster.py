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
