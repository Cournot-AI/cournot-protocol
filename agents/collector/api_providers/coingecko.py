"""
CoinGecko cryptocurrency API provider.

Free, no API key required (rate-limited).  Supports both current prices
(``/simple/price``) and historical prices (``/coins/{id}/history``).
https://docs.coingecko.com/reference/introduction
"""

from __future__ import annotations

import re
from typing import Any

from .base import APIMatchResult, BaseAPIProvider

# Common name / ticker → CoinGecko slug mapping.
_NAME_TO_ID: dict[str, str] = {
    "bitcoin": "bitcoin",
    "btc": "bitcoin",
    "ethereum": "ethereum",
    "ether": "ethereum",
    "eth": "ethereum",
    "solana": "solana",
    "sol": "solana",
    "cardano": "cardano",
    "ada": "cardano",
    "ripple": "ripple",
    "xrp": "ripple",
    "dogecoin": "dogecoin",
    "doge": "dogecoin",
    "polkadot": "polkadot",
    "dot": "polkadot",
    "avalanche": "avalanche-2",
    "avax": "avalanche-2",
    "chainlink": "chainlink",
    "link": "chainlink",
    "litecoin": "litecoin",
    "ltc": "litecoin",
    "uniswap": "uniswap",
    "uni": "uniswap",
    "bnb": "binancecoin",
    "binance coin": "binancecoin",
    "tron": "tron",
    "trx": "tron",
    "stellar": "stellar",
    "xlm": "stellar",
    "monero": "monero",
    "xmr": "monero",
    "near": "near",
    "cosmos": "cosmos",
    "atom": "cosmos",
    "tether": "tether",
    "usdt": "tether",
}

# CoinGecko slug → display symbol (for output).
_ID_TO_SYMBOL: dict[str, str] = {
    "bitcoin": "BTC",
    "ethereum": "ETH",
    "solana": "SOL",
    "cardano": "ADA",
    "ripple": "XRP",
    "dogecoin": "DOGE",
    "polkadot": "DOT",
    "avalanche-2": "AVAX",
    "chainlink": "LINK",
    "litecoin": "LTC",
    "uniswap": "UNI",
    "binancecoin": "BNB",
    "tron": "TRX",
    "stellar": "XLM",
    "monero": "XMR",
    "near": "NEAR",
    "cosmos": "ATOM",
    "tether": "USDT",
}

# Uppercase ticker patterns
_SYMBOL_PATTERN = re.compile(r"\b([A-Z]{2,5})\b")

_KNOWN_TICKERS = set(_NAME_TO_ID.keys()) | {v.upper() for v in _ID_TO_SYMBOL.values()}

_KEYWORDS = [
    "crypto",
    "cryptocurrency",
    "bitcoin",
    "btc",
    "ethereum",
    "eth",
    "solana",
    "sol",
    "xrp",
    "price",
    "market cap",
    "token",
    "coin",
    "altcoin",
    "defi",
    "blockchain",
    "crypto price",
    "digital currency",
]


class CoinGeckoProvider(BaseAPIProvider):
    """CoinGecko cryptocurrency provider (free, no key).

    Supports current prices via ``/simple/price`` and historical prices
    via ``/coins/{id}/history``.
    """

    @property
    def provider_id(self) -> str:
        return "coingecko"

    @property
    def display_name(self) -> str:
        return "CoinGecko"

    @property
    def domains(self) -> list[str]:
        return ["crypto", "finance"]

    @property
    def keywords(self) -> list[str]:
        return _KEYWORDS

    @property
    def requires_api_key(self) -> bool:
        return False

    # ------------------------------------------------------------------
    # Schema for LLM-driven parameter extraction
    # ------------------------------------------------------------------

    @property
    def param_schema(self) -> dict[str, Any]:
        return {
            "coin_id": {
                "type": "string",
                "required": True,
                "description": "CoinGecko coin ID (e.g. 'bitcoin', 'ethereum', 'solana', 'ripple')",
            },
            "mode": {
                "type": "string",
                "required": False,
                "default": "current",
                "description": "'current' for live price or 'historical' for a past date",
            },
            "date": {
                "type": "string",
                "required": False,
                "description": "Date in dd-mm-yyyy format (required if mode is 'historical')",
            },
            "vs_currency": {
                "type": "string",
                "required": False,
                "default": "usd",
                "description": "Target currency for price conversion (default: 'usd')",
            },
        }

    @property
    def param_extraction_hint(self) -> str:
        coins = ", ".join(
            f"{name} → {cg_id}"
            for name, cg_id in sorted(set(
                (n, i) for n, i in _NAME_TO_ID.items()
            ))
        )
        return (
            "Extract cryptocurrency query parameters.\n\n"
            "CRITICAL — Date handling (check this FIRST):\n"
            "- Compare any date mentioned in the requirement against today's date.\n"
            "- If the mentioned date is BEFORE today, set mode to 'historical' and "
            "provide date in dd-mm-yyyy format.\n"
            "- If no date is mentioned or the date is today/future, set mode to 'current'.\n"
            "- Example: 'March 10, 2026' → date='10-03-2026'.\n\n"
            "Coin ID mapping:\n"
            f"{coins}\n\n"
            "Return ONLY a JSON object with the extracted parameters.\n"
        )

    # ------------------------------------------------------------------
    # Matching
    # ------------------------------------------------------------------

    def match_requirement(
        self, description: str, **context: Any
    ) -> APIMatchResult | None:
        base = super().match_requirement(description, **context)
        if base is None:
            return None

        # Try to extract coin ID as a hint
        coin_id = self._extract_coin_id(description)
        if coin_id:
            base.confidence = min(base.confidence + 0.3, 1.0)
            base.suggested_params["coin_id"] = coin_id

        return base

    # ------------------------------------------------------------------
    # Parameter validation
    # ------------------------------------------------------------------

    def validate_params(self, params: dict[str, Any]) -> tuple[bool, str]:
        if "coin_id" not in params:
            return False, "Could not determine which cryptocurrency to query"
        mode = params.get("mode", "current")
        if mode == "historical" and "date" not in params:
            return False, "Historical mode requires a date parameter (dd-mm-yyyy)"
        return True, ""

    # ------------------------------------------------------------------
    # Request building
    # ------------------------------------------------------------------

    def build_request(
        self, description: str, params: dict[str, Any]
    ) -> tuple[str, dict[str, str], dict[str, Any]]:
        coin_id = params.get("coin_id", "bitcoin")
        mode = params.get("mode", "current")
        vs_currency = params.get("vs_currency", "usd")

        if mode == "historical":
            url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/history"
            query: dict[str, Any] = {
                "date": params["date"],
                "localization": "false",
            }
        else:
            url = "https://api.coingecko.com/api/v3/simple/price"
            query = {
                "ids": coin_id,
                "vs_currencies": vs_currency,
                "include_24hr_change": "true",
                "include_market_cap": "true",
                "include_24hr_vol": "true",
            }
        return url, {}, query

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    def parse_response(self, data: dict[str, Any]) -> dict[str, Any]:
        # Historical response: {"id": "bitcoin", "market_data": {"current_price": {"usd": 68459}}}
        if "market_data" in data:
            md = data["market_data"]
            coin_id = data.get("id", "unknown")
            symbol = _ID_TO_SYMBOL.get(coin_id, coin_id.upper())
            prices = md.get("current_price", {})
            caps = md.get("market_cap", {})
            vols = md.get("total_volume", {})
            return {
                symbol: {
                    "price_usd": prices.get("usd"),
                    "market_cap_usd": caps.get("usd"),
                    "volume_24h_usd": vols.get("usd"),
                },
                "coin_id": coin_id,
                "symbol": symbol,
                "price_usd": prices.get("usd"),
            }

        # Current price response: {"bitcoin": {"usd": 68459, "usd_24h_change": 2.5}}
        result: dict[str, Any] = {}
        for coin_id, info in data.items():
            if not isinstance(info, dict):
                continue
            symbol = _ID_TO_SYMBOL.get(coin_id, coin_id.upper())
            result[symbol] = {
                "price_usd": info.get("usd"),
                "market_cap_usd": info.get("usd_market_cap"),
                "volume_24h_usd": info.get("usd_24h_vol"),
                "percent_change_24h": info.get("usd_24h_change"),
            }
            result["coin_id"] = coin_id
            result["symbol"] = symbol
            result["price_usd"] = info.get("usd")
        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_coin_id(description: str) -> str | None:
        """Extract CoinGecko coin ID from description text."""
        desc_lower = description.lower()

        # 1. Named crypto → CoinGecko ID
        for name, cg_id in sorted(
            _NAME_TO_ID.items(), key=lambda x: len(x[0]), reverse=True
        ):
            if name in desc_lower:
                return cg_id

        # 2. Uppercase ticker symbols
        for m in _SYMBOL_PATTERN.finditer(description):
            tok = m.group(1).lower()
            if tok in _NAME_TO_ID:
                return _NAME_TO_ID[tok]

        return None
