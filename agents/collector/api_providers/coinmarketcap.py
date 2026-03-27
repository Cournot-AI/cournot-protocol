"""
CoinMarketCap cryptocurrency API provider.

Requires a ``CMC_PRO_API_KEY`` to function.
https://coinmarketcap.com/api/documentation/v1/
"""

from __future__ import annotations

import re
from typing import Any

from .base import APIMatchResult, APIProviderConfig, BaseAPIProvider

# Common name → symbol mapping.
_NAME_TO_SYMBOL: dict[str, str] = {
    "bitcoin": "BTC",
    "ethereum": "ETH",
    "ether": "ETH",
    "solana": "SOL",
    "cardano": "ADA",
    "ripple": "XRP",
    "dogecoin": "DOGE",
    "polkadot": "DOT",
    "avalanche": "AVAX",
    "chainlink": "LINK",
    "polygon": "MATIC",
    "litecoin": "LTC",
    "uniswap": "UNI",
    "cosmos": "ATOM",
    "tether": "USDT",
    "bnb": "BNB",
    "binance coin": "BNB",
    "tron": "TRX",
    "stellar": "XLM",
    "monero": "XMR",
    "near": "NEAR",
}

# Regex: 2-5 uppercase letter tokens that look like ticker symbols
_SYMBOL_PATTERN = re.compile(r"\b([A-Z]{2,5})\b")

_KEYWORDS = [
    "crypto",
    "cryptocurrency",
    "bitcoin",
    "btc",
    "ethereum",
    "eth",
    "solana",
    "sol",
    "price",
    "market cap",
    "token",
    "coin",
    "altcoin",
    "defi",
    "blockchain",
    "trading",
    "crypto price",
    "digital currency",
]

_DEFAULT_BASE_URL = "https://pro-api.coinmarketcap.com"


class CoinMarketCapProvider(BaseAPIProvider):
    """CoinMarketCap cryptocurrency quotes provider."""

    @property
    def provider_id(self) -> str:
        return "coinmarketcap"

    @property
    def display_name(self) -> str:
        return "CoinMarketCap"

    @property
    def domains(self) -> list[str]:
        return ["crypto", "finance"]

    @property
    def keywords(self) -> list[str]:
        return _KEYWORDS

    @property
    def requires_api_key(self) -> bool:
        return True

    # ------------------------------------------------------------------
    # Matching
    # ------------------------------------------------------------------

    def match_requirement(
        self, description: str, **context: Any
    ) -> APIMatchResult | None:
        base = super().match_requirement(description, **context)
        if base is None:
            return None

        symbols = self._extract_symbols(description)
        if symbols:
            base.confidence = min(base.confidence + 0.3, 1.0)
            base.suggested_params["symbols"] = symbols
        return base

    # ------------------------------------------------------------------
    # Request building
    # ------------------------------------------------------------------

    def build_request(
        self, description: str, params: dict[str, Any]
    ) -> tuple[str, dict[str, str], dict[str, Any]]:
        symbols: list[str] = params.get("symbols", ["BTC"])
        api_key: str = params.get("api_key", "")
        base_url: str = params.get("base_url", _DEFAULT_BASE_URL)

        url = f"{base_url}/v1/cryptocurrency/quotes/latest"
        headers = {
            "X-CMC_PRO_API_KEY": api_key,
            "Accept": "application/json",
        }
        query: dict[str, Any] = {
            "symbol": ",".join(symbols),
            "convert": "USD",
        }
        return url, headers, query

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    def parse_response(self, data: dict[str, Any]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        coins = data.get("data", {})
        for symbol, info in coins.items():
            quote = info.get("quote", {}).get("USD", {})
            result[symbol] = {
                "price_usd": quote.get("price"),
                "market_cap_usd": quote.get("market_cap"),
                "volume_24h_usd": quote.get("volume_24h"),
                "percent_change_24h": quote.get("percent_change_24h"),
            }
        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_symbols(description: str) -> list[str]:
        """Extract crypto symbols from description text."""
        symbols: list[str] = []

        # 1. Named crypto → symbol
        desc_lower = description.lower()
        for name, sym in _NAME_TO_SYMBOL.items():
            if name in desc_lower and sym not in symbols:
                symbols.append(sym)

        # 2. Uppercase ticker-like tokens (BTC, ETH, etc.)
        for m in _SYMBOL_PATTERN.finditer(description):
            tok = m.group(1)
            # Only accept if it's a known symbol or mentioned alongside crypto keywords
            if tok in _NAME_TO_SYMBOL.values() and tok not in symbols:
                symbols.append(tok)

        return symbols
