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


# Tiingo's `endDate` is the LAST-DATA-DATE, populated for ACTIVE names too
# (verified against the live supported_tickers file: AAPL/MSFT/NVDA all carry
# an `endDate` equal to the file's fetch/build date -- NOT empty). So a
# non-empty `endDate` is NOT a delisted flag by itself. The reference date
# ("today" as of the file) isn't carried anywhere else in the roster, so it's
# inferred as the MAX endDate present (clamped to wall-clock as_of; see
# delisted_symbols) -- that's what an active name's endDate converges to.
# This buffer absorbs active names whose last trade lands a few calendar days
# before that inferred reference (thin volume, a holiday, a data lag) so they
# aren't misclassified as delisted.
DELISTED_BUFFER_DAYS = 7


def delisted_symbols(roster: pd.DataFrame, *, as_of: datetime.date | None = None) -> set[str]:
    """Tickers that are genuinely DELISTED as of the roster's own data-as-of
    date -- the survivorship metric numerator (spec section 4).

    `endDate` is Tiingo's LAST-DATA-DATE, set for active names too (see
    DELISTED_BUFFER_DAYS above), so "non-empty" is not "delisted": a name
    counts as delisted only when its endDate parses AND falls strictly more
    than DELISTED_BUFFER_DAYS before the roster's inferred data-as-of date.
    An empty or unparseable endDate is an anomaly under live data -- which
    always populates it for both active and delisted names -- so it is
    treated as NOT delisted rather than guessed at. With this fix,
    `survivorship_pct` (downcap_verify.compute_gate) is a genuine
    measurement, not a lower bound via over-counting.

    The data-as-of reference is the max endDate present CLAMPED to not exceed
    the real wall-clock `as_of` (default today). Without the clamp a single
    corrupt future-dated endDate row (e.g. a `2099-01-01` typo, plausible in
    a 15k-row live file) would poison `end.max()` and push the cutoff so far
    forward that genuinely active names (endDate == today) get misclassified
    as delisted -- re-breaking the survivorship metric via a spurious ~100%.
    `as_of` is a keyword so tests can pin the reference deterministically;
    production callers pass none and get today's date."""
    as_of_ts = pd.Timestamp(as_of or datetime.date.today())
    end = pd.to_datetime(roster["endDate"], format="%Y-%m-%d", errors="coerce")
    if end.notna().sum() == 0:
        return set()
    # Clamp the file's max endDate to the wall-clock as_of so a garbage
    # future endDate cannot inflate the reference. If EVERY endDate is in the
    # future (degenerate), min() falls back to as_of.
    reference_date = min(end.max(), as_of_ts)
    # pd.Timedelta(days=...) triggers a spurious numpy-generic-unit
    # DeprecationWarning on this pandas version; datetime.timedelta avoids it
    # and subtracts from a Timestamp just as well.
    cutoff = reference_date - datetime.timedelta(days=DELISTED_BUFFER_DAYS)
    is_delisted = end.notna() & (end < cutoff)
    return set(roster.loc[is_delisted, "ticker"])
