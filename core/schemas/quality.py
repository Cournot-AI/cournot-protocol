"""
Module 01 - Schemas & Canonicalization
File: quality.py

Purpose: Quality check schemas for the collect ↔ quality_check feedback loop.
Defines the EvidenceQualityScorecard returned by /step/quality_check,
including actionable recommendations and machine-readable retry_hints
that callers can pass back to /step/collect.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


QualityLevel = Literal["HIGH", "MEDIUM", "LOW"]


class EvidenceQualityScorecard(BaseModel):
    """Quality assessment of collected evidence against a prompt spec."""

    # --- Core quality signals ---
    source_match: Literal["FULL", "PARTIAL", "NONE"] = Field(
        ...,
        description=(
            "Whether evidence sources match the required data-source domains. "
            "FULL = all required domains covered, PARTIAL = some covered, NONE = none covered."
        ),
    )
    data_type_match: bool = Field(
        ...,
        description="Whether evidence data types match what the market requires.",
    )
    collector_agreement: Literal["AGREE", "DISAGREE", "SINGLE"] = Field(
        ...,
        description=(
            "Whether multiple collectors agree on the outcome. "
            "SINGLE if only one collector was used."
        ),
    )
    requirements_coverage: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Fraction of data requirements that were fulfilled (0.0-1.0).",
    )

    # --- Aggregate ---
    quality_level: QualityLevel = Field(
        ...,
        description="Overall quality level: HIGH, MEDIUM, or LOW.",
    )
    quality_flags: list[str] = Field(
        default_factory=list,
        description="List of quality issue flags (e.g. 'source_mismatch', 'requirements_gap').",
    )

    # --- Actionable outputs (the key additions for the feedback loop) ---
    meets_threshold: bool = Field(
        ...,
        description=(
            "Quick pass/fail: True means quality is good enough to proceed to audit. "
            "True when quality_level is HIGH or MEDIUM."
        ),
    )
    recommendations: list[str] = Field(
        default_factory=list,
        description="Human-readable action items for improving evidence quality.",
    )
    retry_hints: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Machine-readable hints for the next /step/collect call. "
            "May contain: search_queries, required_domains, skip_domains, "
            "data_type_hint, focus_requirements, collector_guidance."
        ),
    )
