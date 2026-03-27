"""
Open-Meteo weather API provider.

Free, no API key required.  Supports both current weather (forecast endpoint)
and historical weather (archive endpoint).
https://open-meteo.com/en/docs
"""

from __future__ import annotations

import re
from typing import Any

from .base import APIMatchResult, APIProviderConfig, BaseAPIProvider

# Hardcoded geocode cache for common cities (lat, lon).
_GEOCODE_CACHE: dict[str, tuple[float, float]] = {
    "new york": (40.7128, -74.0060),
    "london": (51.5074, -0.1278),
    "paris": (48.8566, 2.3522),
    "tokyo": (35.6762, 139.6503),
    "sydney": (-33.8688, 151.2093),
    "los angeles": (34.0522, -118.2437),
    "chicago": (41.8781, -87.6298),
    "berlin": (52.5200, 13.4050),
    "mumbai": (19.0760, 72.8777),
    "beijing": (39.9042, 116.4074),
    "dubai": (25.2048, 55.2708),
    "singapore": (1.3521, 103.8198),
    "hong kong": (22.3193, 114.1694),
    "toronto": (43.6532, -79.3832),
    "san francisco": (37.7749, -122.4194),
    "seattle": (47.6062, -122.3321),
    "miami": (25.7617, -80.1918),
    "madrid": (40.4168, -3.7038),
    "rome": (41.9028, 12.4964),
    "moscow": (55.7558, 37.6173),
    "seoul": (37.5665, 126.9780),
    "bangkok": (13.7563, 100.5018),
    "cairo": (30.0444, 31.2357),
    "rio de janeiro": (-22.9068, -43.1729),
    "buenos aires": (-34.6037, -58.3816),
    "istanbul": (41.0082, 28.9784),
    "lagos": (6.5244, 3.3792),
    "nairobi": (-1.2921, 36.8219),
    "amsterdam": (52.3676, 4.9041),
    "mexico city": (19.4326, -99.1332),
}

# ICAO airport codes → (lat, lon)
_ICAO_CODES: dict[str, tuple[float, float]] = {
    "SAEZ": (-34.8222, -58.5358),   # Ezeiza / Minister Pistarini, Buenos Aires
    "KJFK": (40.6413, -73.7781),    # JFK, New York
    "KLAX": (33.9425, -118.4081),   # LAX, Los Angeles
    "KORD": (41.9742, -87.9073),    # O'Hare, Chicago
    "EGLL": (51.4700, -0.4543),     # Heathrow, London
    "LFPG": (49.0097, 2.5479),      # Charles de Gaulle, Paris
    "RJTT": (35.5494, 139.7798),    # Haneda, Tokyo
    "EDDB": (52.3667, 13.5033),     # Berlin Brandenburg
    "OMDB": (25.2528, 55.3644),     # Dubai International
    "WSSS": (1.3644, 103.9915),     # Changi, Singapore
    "VHHH": (22.3080, 113.9185),    # Hong Kong International
    "CYYZ": (43.6777, -79.6248),    # Toronto Pearson
    "SBGL": (-22.8099, -43.2506),   # Galeão, Rio de Janeiro
    "YSSY": (-33.9461, 151.1772),   # Sydney Kingsford Smith
    "VIDP": (28.5562, 77.1000),     # Delhi Indira Gandhi
}

# Airport/landmark name aliases → (lat, lon)
_AIRPORT_ALIASES: dict[str, tuple[float, float]] = {
    "minister pistarini": (-34.8222, -58.5358),
    "pistarini": (-34.8222, -58.5358),
    "ezeiza": (-34.8222, -58.5358),
    "jfk": (40.6413, -73.7781),
    "john f kennedy": (40.6413, -73.7781),
    "lax": (33.9425, -118.4081),
    "heathrow": (51.4700, -0.4543),
    "charles de gaulle": (49.0097, 2.5479),
    "cdg": (49.0097, 2.5479),
    "haneda": (35.5494, 139.7798),
    "narita": (35.7647, 140.3864),
    "changi": (1.3644, 103.9915),
    "dubai international": (25.2528, 55.3644),
    "o'hare": (41.9742, -87.9073),
    "ohare": (41.9742, -87.9073),
    "pearson": (43.6777, -79.6248),
    "schiphol": (52.3105, 4.7683),
}

# Regex for explicit lat/lon in text, e.g. "lat 40.71 lon -74.00"
_LATLON_PATTERN = re.compile(
    r"lat(?:itude)?\s*[=:]?\s*(-?\d+\.?\d*)\s*[,;/&]?\s*"
    r"lon(?:gitude)?\s*[=:]?\s*(-?\d+\.?\d*)",
    re.IGNORECASE,
)

_KEYWORDS = [
    "weather",
    "temperature",
    "forecast",
    "wind",
    "rain",
    "humidity",
    "precipitation",
    "climate",
    "celsius",
    "fahrenheit",
    "sunny",
    "cloudy",
    "storm",
    "snow",
    "heatwave",
    "cold front",
    "wind speed",
    "wind direction",
]


class OpenMeteoProvider(BaseAPIProvider):
    """Open-Meteo weather provider (free, no key).

    Supports current weather via the forecast endpoint and historical
    weather via the archive endpoint.
    """

    @property
    def provider_id(self) -> str:
        return "open_meteo"

    @property
    def display_name(self) -> str:
        return "Open-Meteo Weather"

    @property
    def domains(self) -> list[str]:
        return ["weather", "climate"]

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
            "latitude": {
                "type": "float",
                "required": True,
                "description": "Latitude of the location (-90 to 90)",
            },
            "longitude": {
                "type": "float",
                "required": True,
                "description": "Longitude of the location (-180 to 180)",
            },
            "mode": {
                "type": "string",
                "required": False,
                "default": "current",
                "description": "'current' for live weather or 'historical' for past dates",
            },
            "start_date": {
                "type": "string",
                "required": False,
                "description": "Start date in YYYY-MM-DD format (required if mode is 'historical')",
            },
            "end_date": {
                "type": "string",
                "required": False,
                "description": "End date in YYYY-MM-DD format (required if mode is 'historical')",
            },
        }

    @property
    def param_extraction_hint(self) -> str:
        # Build reference lists from the geocode caches
        cities = ", ".join(
            f"{name} ({lat}, {lon})"
            for name, (lat, lon) in sorted(_GEOCODE_CACHE.items())
        )
        airports = ", ".join(
            f"{name} ({lat}, {lon})"
            for name, (lat, lon) in sorted(_AIRPORT_ALIASES.items())
        )
        icao = ", ".join(
            f"{code} ({lat}, {lon})"
            for code, (lat, lon) in sorted(_ICAO_CODES.items())
        )
        return (
            "Extract weather query parameters from the requirement description.\n\n"
            "CRITICAL — Date handling (check this FIRST):\n"
            "- Compare any date mentioned in the requirement against today's date.\n"
            "- If the mentioned date is BEFORE today, set mode to 'historical' and "
            "provide start_date and end_date in YYYY-MM-DD format.\n"
            "- For a single past date, set start_date and end_date to the same value.\n"
            "- If no date is mentioned or the date is today/future, set mode to 'current'.\n"
            "- Dates like 'March 10, 2026' or '10 Mar 26' must be converted to "
            "YYYY-MM-DD format (e.g. '2026-03-10').\n\n"
            "Location handling:\n"
            "- If the location is not in the reference lists below, use your "
            "knowledge to provide the latitude and longitude.\n"
            "- For airport weather stations, use the airport coordinates, not "
            "the city center.\n\n"
            f"Known cities: {cities}\n\n"
            f"Known airports/landmarks: {airports}\n\n"
            f"Known ICAO codes: {icao}\n\n"
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

        # Try to extract location params as hints for LLM extraction
        loc_params = self._extract_location(description)
        if loc_params:
            base.confidence = min(base.confidence + 0.3, 1.0)
            base.suggested_params.update(loc_params)

        return base

    # ------------------------------------------------------------------
    # Parameter validation
    # ------------------------------------------------------------------

    def validate_params(self, params: dict[str, Any]) -> tuple[bool, str]:
        if "latitude" not in params or "longitude" not in params:
            return False, "Could not extract location from requirement"
        mode = params.get("mode", "current")
        if mode == "historical":
            if "start_date" not in params or "end_date" not in params:
                return False, "Historical mode requires start_date and end_date"
        return True, ""

    # ------------------------------------------------------------------
    # Request building
    # ------------------------------------------------------------------

    def build_request(
        self, description: str, params: dict[str, Any]
    ) -> tuple[str, dict[str, str], dict[str, Any]]:
        lat = params.get("latitude", 40.7128)
        lon = params.get("longitude", -74.0060)
        mode = params.get("mode", "current")

        if mode == "historical":
            url = "https://archive-api.open-meteo.com/v1/archive"
            query: dict[str, Any] = {
                "latitude": lat,
                "longitude": lon,
                "start_date": params["start_date"],
                "end_date": params["end_date"],
                "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum",
                "timezone": "auto",
            }
        else:
            url = "https://api.open-meteo.com/v1/forecast"
            query = {
                "latitude": lat,
                "longitude": lon,
                "current_weather": "true",
            }
        return url, {}, query

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    def parse_response(self, data: dict[str, Any]) -> dict[str, Any]:
        # Historical response has a "daily" key
        if "daily" in data:
            daily = data["daily"]
            result: dict[str, Any] = {}
            if "time" in daily and daily["time"]:
                result["date"] = daily["time"][0]
            if "temperature_2m_max" in daily and daily["temperature_2m_max"]:
                result["temperature_2m_max"] = daily["temperature_2m_max"][0]
            if "temperature_2m_min" in daily and daily["temperature_2m_min"]:
                result["temperature_2m_min"] = daily["temperature_2m_min"][0]
            if "precipitation_sum" in daily and daily["precipitation_sum"]:
                result["precipitation_sum"] = daily["precipitation_sum"][0]
            return result

        # Current weather response
        cw = data.get("current_weather", {})
        return {
            "temperature_c": cw.get("temperature"),
            "windspeed_kmh": cw.get("windspeed"),
            "winddirection_deg": cw.get("winddirection"),
            "weathercode": cw.get("weathercode"),
            "observation_time": cw.get("time"),
        }

    # ------------------------------------------------------------------
    # Location extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_location(description: str) -> dict[str, float]:
        """Try to resolve lat/lon from the description text.

        Checks airport/landmark aliases first, then ICAO codes, then city
        names, then explicit lat/lon patterns.
        """
        desc_lower = description.lower()

        # 1. Airport/landmark aliases (longest match first to avoid partial)
        for alias, (lat, lon) in sorted(
            _AIRPORT_ALIASES.items(), key=lambda x: len(x[0]), reverse=True
        ):
            if alias in desc_lower:
                return {"latitude": lat, "longitude": lon}

        # 2. ICAO codes (case-insensitive scan of original text)
        desc_upper = description.upper()
        for code, (lat, lon) in _ICAO_CODES.items():
            if code in desc_upper:
                return {"latitude": lat, "longitude": lon}

        # 3. City name lookup
        for city, (lat, lon) in _GEOCODE_CACHE.items():
            if city in desc_lower:
                return {"latitude": lat, "longitude": lon}

        # 4. Explicit lat/lon pattern
        m = _LATLON_PATTERN.search(description)
        if m:
            return {"latitude": float(m.group(1)), "longitude": float(m.group(2))}

        return {}
