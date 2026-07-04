from __future__ import annotations

from trading.config import VenueConfig
from trading.venues.base import VenueAdapter


def make_adapter(config: VenueConfig) -> VenueAdapter:
    if config.name == "equities":
        from trading.venues.equities import EquitiesAdapter

        return EquitiesAdapter(config)
    if config.name == "crypto":
        from trading.venues.crypto import CryptoAdapter

        return CryptoAdapter(config)
    raise ValueError(f"unknown venue {config.name!r}")
