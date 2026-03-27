"""
Base abstractions for structured API providers.

Each provider encapsulates a single external API (weather, crypto, etc.)
and exposes a uniform interface for matching requirements, building requests,
and parsing responses.
"""

from __future__ import annotations

import hashlib
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

import httpx

logger = logging.getLogger(__name__)


@dataclass
class APIProviderConfig:
    """Per-provider runtime configuration."""

    enabled: bool = True
    api_key: str | None = None
    base_url: str | None = None
    timeout: float = 30.0
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class APIMatchResult:
    """Result of matching a data requirement to a provider."""

    provider_id: str
    confidence: float  # 0.0 – 1.0
    matched_keywords: list[str] = field(default_factory=list)
    suggested_params: dict[str, Any] = field(default_factory=dict)


@dataclass
class APICallResult:
    """Result of calling an API provider."""

    success: bool
    data: dict[str, Any] = field(default_factory=dict)
    raw_response: str | None = None
    status_code: int | None = None
    error: str | None = None
    source_uri: str = ""
    extracted_fields: dict[str, Any] = field(default_factory=dict)


class BaseAPIProvider(ABC):
    """Abstract base for structured API providers."""

    @property
    @abstractmethod
    def provider_id(self) -> str:
        """Unique identifier, e.g. ``'open_meteo'``."""

    @property
    @abstractmethod
    def display_name(self) -> str:
        """Human-readable name shown in logs and evidence provenance."""

    @property
    @abstractmethod
    def domains(self) -> list[str]:
        """Topical domains this provider covers, e.g. ``['weather']``."""

    @property
    @abstractmethod
    def keywords(self) -> list[str]:
        """Keywords used for requirement matching."""

    @property
    @abstractmethod
    def requires_api_key(self) -> bool:
        """Whether the provider needs an API key to function."""

    # ------------------------------------------------------------------
    # Schema for LLM-driven parameter extraction
    # ------------------------------------------------------------------

    @property
    def param_schema(self) -> dict[str, Any]:
        """JSON-like description of parameters this provider needs."""
        return {}

    @property
    def param_extraction_hint(self) -> str:
        """Natural language instructions for LLM param extraction."""
        return ""

    # ------------------------------------------------------------------
    # Matching
    # ------------------------------------------------------------------

    def match_requirement(
        self, description: str, **context: Any
    ) -> APIMatchResult | None:
        """Return a match result if this provider can serve *description*.

        Default implementation uses keyword overlap.  Subclasses may add
        pattern extraction (e.g. lat/lon, coin symbols).
        """
        desc_lower = description.lower()
        matched = [kw for kw in self.keywords if kw in desc_lower]
        if not matched:
            return None
        confidence = min(len(matched) / 3.0, 1.0)
        return APIMatchResult(
            provider_id=self.provider_id,
            confidence=confidence,
            matched_keywords=matched,
        )

    # ------------------------------------------------------------------
    # Request / response
    # ------------------------------------------------------------------

    @abstractmethod
    def build_request(
        self, description: str, params: dict[str, Any]
    ) -> tuple[str, dict[str, str], dict[str, Any]]:
        """Return ``(url, headers, query_params)`` for the API call."""

    @abstractmethod
    def parse_response(self, data: dict[str, Any]) -> dict[str, Any]:
        """Extract domain-specific fields from the raw JSON response."""

    # ------------------------------------------------------------------
    # Parameter validation
    # ------------------------------------------------------------------

    def validate_params(self, params: dict[str, Any]) -> tuple[bool, str]:
        """Check if extracted params are sufficient for a valid API call.

        Returns ``(ok, reason)``.  Override in subclasses to enforce
        provider-specific requirements (e.g. location must be present).
        """
        return True, ""

    # ------------------------------------------------------------------
    # Availability
    # ------------------------------------------------------------------

    def is_available(self, config: APIProviderConfig) -> bool:
        """Check whether the provider is enabled and has required keys."""
        if not config.enabled:
            return False
        if self.requires_api_key and not config.api_key:
            return False
        return True

    # ------------------------------------------------------------------
    # Template call method
    # ------------------------------------------------------------------

    def call(
        self,
        config: APIProviderConfig,
        description: str,
        params: dict[str, Any],
    ) -> APICallResult:
        """Build request → fetch → parse.  Synchronous via ``httpx``."""
        url, headers, query_params = self.build_request(description, params)
        source_uri = url
        if query_params:
            qs = "&".join(f"{k}={v}" for k, v in query_params.items())
            source_uri = f"{url}?{qs}"

        timeout = config.timeout
        try:
            resp = httpx.get(
                url,
                headers=headers,
                params=query_params,
                timeout=timeout,
            )
            status_code = resp.status_code
            raw_text = resp.text

            if status_code >= 400:
                return APICallResult(
                    success=False,
                    status_code=status_code,
                    raw_response=raw_text,
                    error=f"HTTP {status_code}: {raw_text[:500]}",
                    source_uri=source_uri,
                )

            json_data = resp.json()
            extracted = self.parse_response(json_data)

            return APICallResult(
                success=True,
                data=json_data,
                raw_response=raw_text,
                status_code=status_code,
                source_uri=source_uri,
                extracted_fields=extracted,
            )
        except httpx.TimeoutException:
            return APICallResult(
                success=False,
                error=f"Timeout after {timeout}s",
                source_uri=source_uri,
            )
        except Exception as exc:  # noqa: BLE001
            return APICallResult(
                success=False,
                error=str(exc),
                source_uri=source_uri,
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _content_hash(text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()[:16]
