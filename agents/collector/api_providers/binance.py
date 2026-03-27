"""
Binance cryptocurrency API provider.

Free by default (no API key required), but optionally supports
``BINANCE_API_KEY`` for higher rate limits.  Provides spot prices
(``/api/v3/ticker/24hr``) and historical kline/candlestick data
(``/api/v3/klines``).
https://binance-docs.github.io/apidocs/spot/en/
"""

from __future__ import annotations

import re
from typing import Any

from .base import APIMatchResult, BaseAPIProvider

# Common name / ticker -> Binance trading pair (USDT).
_NAME_TO_PAIR: dict[str, str] = {
    "bitcoin": "BTCUSDT",
    "btc": "BTCUSDT",
    "ethereum": "ETHUSDT",
    "ether": "ETHUSDT",
    "eth": "ETHUSDT",
    "solana": "SOLUSDT",
    "sol": "SOLUSDT",
    "cardano": "ADAUSDT",
    "ada": "ADAUSDT",
    "ripple": "XRPUSDT",
    "xrp": "XRPUSDT",
    "dogecoin": "DOGEUSDT",
    "doge": "DOGEUSDT",
    "polkadot": "DOTUSDT",
    "dot": "DOTUSDT",
    "avalanche": "AVAXUSDT",
    "avax": "AVAXUSDT",
    "chainlink": "LINKUSDT",
    "link": "LINKUSDT",
    "litecoin": "LTCUSDT",
    "ltc": "LTCUSDT",
    "uniswap": "UNIUSDT",
    "uni": "UNIUSDT",
    "bnb": "BNBUSDT",
    "binance coin": "BNBUSDT",
    "tron": "TRXUSDT",
    "trx": "TRXUSDT",
    "stellar": "XLMUSDT",
    "xlm": "XLMUSDT",
    "monero": "XMRUSDT",
    "xmr": "XMRUSDT",
    "near": "NEARUSDT",
    "cosmos": "ATOMUSDT",
    "atom": "ATOMUSDT",
    "tether": "USDTUSD",
    "usdt": "USDTUSD",
}

# Uppercase ticker patterns
_SYMBOL_PATTERN = re.compile(r"\b([A-Z]{2,5})\b")

_KNOWN_TICKERS = set(_NAME_TO_PAIR.keys()) | {
    v.replace("USDT", "").replace("USD", "") for v in _NAME_TO_PAIR.values()
}

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
    "binance",
    "trading pair",
    "spot",
    "kline",
    "candlestick",
    "ohlcv",
]


class BinanceProvider(BaseAPIProvider):
    """Binance cryptocurrency provider (free, optional key for rate limits).

    Supports spot prices via ``/api/v3/ticker/24hr`` and historical
    kline/candlestick data via ``/api/v3/klines``.
    """

    @property
    def provider_id(self) -> str:
        return "binance"

    @property
    def display_name(self) -> str:
        return "Binance"

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
            "symbol": {
                "type": "string",
                "required": True,
                "description": (
                    "Binance trading pair e.g. 'BTCUSDT', 'ETHUSDT'"
                ),
            },
            "mode": {
                "type": "string",
                "required": False,
                "default": "current",
                "description": (
                    "'current' for live 24hr ticker or 'historical' for kline data"
                ),
            },
            "interval": {
                "type": "string",
                "required": False,
                "default": "1d",
                "description": (
                    "Kline interval (only for historical mode). "
                    "e.g. '1m', '5m', '1h', '4h', '1d', '1w'"
                ),
            },
            "start_date": {
                "type": "string",
                "required": False,
                "description": (
                    "Start date in ISO format YYYY-MM-DD (required for historical mode)"
                ),
            },
            "end_date": {
                "type": "string",
                "required": False,
                "description": (
                    "End date in ISO format YYYY-MM-DD (optional for historical mode)"
                ),
            },
        }

    @property
    def param_extraction_hint(self) -> str:
        pairs = ", ".join(
            f"{name} -> {pair}"
            for name, pair in sorted(
                set((n, p) for n, p in _NAME_TO_PAIR.items())
            )
        )
        return (
            "Extract Binance API parameters.\n\n"
            "CRITICAL - Date handling (check this FIRST):\n"
            "- Compare any date mentioned in the requirement against today's date.\n"
            "- If the mentioned date is BEFORE today, set mode to 'historical' and "
            "provide start_date in YYYY-MM-DD format.\n"
            "- If no date is mentioned or the date is today/future, set mode to 'current'.\n"
            "- Example: 'March 10, 2026' -> start_date='2026-03-10'.\n\n"
            "Symbol mapping (name -> Binance pair):\n"
            f"{pairs}\n\n"
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

        # Try to extract symbol as a hint
        symbol = self._extract_symbol(description)
        if symbol:
            base.confidence = min(base.confidence + 0.3, 1.0)
            base.suggested_params["symbol"] = symbol

        return base

    # ------------------------------------------------------------------
    # Parameter validation
    # ------------------------------------------------------------------

    def validate_params(self, params: dict[str, Any]) -> tuple[bool, str]:
        if "symbol" not in params:
            return False, "Could not determine which trading pair to query"
        mode = params.get("mode", "current")
        if mode == "historical" and "start_date" not in params:
            return False, "Historical mode requires a start_date parameter (YYYY-MM-DD)"
        return True, ""

    # ------------------------------------------------------------------
    # Request building
    # ------------------------------------------------------------------

    def build_request(
        self, description: str, params: dict[str, Any]
    ) -> tuple[str, dict[str, str], dict[str, Any]]:
        symbol = params.get("symbol", "BTCUSDT")
        mode = params.get("mode", "current")
        api_key = params.get("api_key")

        headers: dict[str, str] = {}
        if api_key:
            headers["X-MBX-APIKEY"] = api_key

        if mode == "historical":
            url = "https://api.binance.com/api/v3/klines"
            interval = params.get("interval", "1d")
            query: dict[str, Any] = {
                "symbol": symbol,
                "interval": interval,
            }
            start_date = params.get("start_date")
            if start_date:
                query["startTime"] = self._date_to_ms(start_date)
            end_date = params.get("end_date")
            if end_date:
                query["endTime"] = self._date_to_ms(end_date)
        else:
            url = "https://api.binance.com/api/v3/ticker/24hr"
            query = {"symbol": symbol}

        return url, headers, query

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    def parse_response(self, data: dict[str, Any] | list) -> dict[str, Any]:
        # Klines response: list of arrays
        # [[open_time, open, high, low, close, volume, close_time, ...], ...]
        if isinstance(data, list):
            candles: list[dict[str, Any]] = []
            for entry in data:
                if isinstance(entry, list) and len(entry) >= 6:
                    candles.append({
                        "open_time": entry[0],
                        "open": entry[1],
                        "high": entry[2],
                        "low": entry[3],
                        "close": entry[4],
                        "volume": entry[5],
                    })
            return {
                "klines": candles,
                "candle_count": len(candles),
            }

        # 24hr ticker response
        return {
            "symbol": data.get("symbol"),
            "last_price": data.get("lastPrice"),
            "price_change_percent": data.get("priceChangePercent"),
            "volume": data.get("volume"),
            "quote_volume": data.get("quoteVolume"),
            "high_price": data.get("highPrice"),
            "low_price": data.get("lowPrice"),
            "open_price": data.get("openPrice"),
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_symbol(description: str) -> str | None:
        """Extract Binance trading pair from description text."""
        desc_lower = description.lower()

        # 1. Named crypto -> Binance pair
        for name, pair in sorted(
            _NAME_TO_PAIR.items(), key=lambda x: len(x[0]), reverse=True
        ):
            if name in desc_lower:
                return pair

        # 2. Uppercase ticker symbols
        for m in _SYMBOL_PATTERN.finditer(description):
            tok = m.group(1).lower()
            if tok in _NAME_TO_PAIR:
                return _NAME_TO_PAIR[tok]

        return None

    @staticmethod
    def _date_to_ms(date_str: str) -> int:
        """Convert YYYY-MM-DD to milliseconds since epoch."""
        from datetime import datetime, timezone

        dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        return int(dt.timestamp() * 1000)
