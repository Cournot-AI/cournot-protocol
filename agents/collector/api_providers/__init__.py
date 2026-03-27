"""
API Provider Registry.

Maintains a list of ``BaseAPIProvider`` instances and exposes helpers for
matching data requirements to the best-fit provider.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .base import APIMatchResult, APIProviderConfig, BaseAPIProvider

if TYPE_CHECKING:
    pass

# ------------------------------------------------------------------
# Internal registry
# ------------------------------------------------------------------

_PROVIDERS: list[BaseAPIProvider] = []


def _ensure_defaults() -> None:
    """Lazily initialise default providers on first access."""
    if _PROVIDERS:
        return
    from .binance import BinanceProvider
    from .coingecko import CoinGeckoProvider
    from .coinmarketcap import CoinMarketCapProvider
    from .open_meteo import OpenMeteoProvider

    _PROVIDERS.extend([
        OpenMeteoProvider(),
        CoinMarketCapProvider(),
        CoinGeckoProvider(),
        BinanceProvider(),
    ])


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------


def register_provider(provider: BaseAPIProvider) -> None:
    """Add a new provider to the registry."""
    _ensure_defaults()
    # Avoid duplicates
    if any(p.provider_id == provider.provider_id for p in _PROVIDERS):
        return
    _PROVIDERS.append(provider)


def get_provider_by_id(provider_id: str) -> BaseAPIProvider | None:
    """Look up a provider by its unique id."""
    _ensure_defaults()
    for p in _PROVIDERS:
        if p.provider_id == provider_id:
            return p
    return None


def get_provider_registry() -> list[BaseAPIProvider]:
    """Return the full list of registered providers."""
    _ensure_defaults()
    return list(_PROVIDERS)


def match_providers(
    description: str,
    configs: dict[str, APIProviderConfig],
    min_confidence: float = 0.1,
) -> list[APIMatchResult]:
    """Match *description* against all available providers.

    Returns results sorted by confidence descending.  Providers that are
    disabled or missing required keys are skipped.
    """
    _ensure_defaults()
    results: list[APIMatchResult] = []
    for provider in _PROVIDERS:
        cfg = configs.get(provider.provider_id, APIProviderConfig())
        if not provider.is_available(cfg):
            continue
        match = provider.match_requirement(description)
        if match is not None and match.confidence >= min_confidence:
            results.append(match)
    results.sort(key=lambda m: m.confidence, reverse=True)
    return results
