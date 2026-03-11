"""
Quality scorecard computation.

Pure-Python logic that evaluates collected evidence against a PromptSpec
and produces an EvidenceQualityScorecard with actionable recommendations
and machine-readable retry_hints for the collect ↔ quality_check loop.
"""

from __future__ import annotations

from typing import Any
from urllib.parse import urlparse

from core.schemas.evidence import EvidenceBundle
from core.schemas.prompts import PromptSpec
from core.schemas.quality import EvidenceQualityScorecard, QualityLevel


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_scorecard(
    prompt_spec: PromptSpec,
    evidence_bundles: list[EvidenceBundle],
) -> EvidenceQualityScorecard:
    """Compute a quality scorecard for collected evidence.

    Args:
        prompt_spec: The compiled prompt specification with data requirements.
        evidence_bundles: Evidence bundles from one or more collectors.

    Returns:
        EvidenceQualityScorecard with quality signals, recommendations,
        and retry_hints for the next /step/collect call.
    """
    source_match = _assess_source_match(prompt_spec, evidence_bundles)
    data_type_match = _assess_data_type_match(prompt_spec, evidence_bundles)
    collector_agreement = _assess_collector_agreement(evidence_bundles)
    requirements_coverage = _assess_requirements_coverage(prompt_spec, evidence_bundles)

    quality_flags = _build_quality_flags(
        source_match, data_type_match, collector_agreement, requirements_coverage,
    )
    quality_level = _compute_quality_level(quality_flags, source_match, requirements_coverage)
    meets_threshold = quality_level in ("HIGH", "MEDIUM")

    recommendations = _build_recommendations(
        prompt_spec, evidence_bundles, quality_flags,
    )
    retry_hints = _build_retry_hints(
        prompt_spec, evidence_bundles, quality_flags,
    )

    return EvidenceQualityScorecard(
        source_match=source_match,
        data_type_match=data_type_match,
        collector_agreement=collector_agreement,
        requirements_coverage=requirements_coverage,
        quality_level=quality_level,
        quality_flags=quality_flags,
        meets_threshold=meets_threshold,
        recommendations=recommendations,
        retry_hints=retry_hints,
    )


# ---------------------------------------------------------------------------
# Source match assessment
# ---------------------------------------------------------------------------

def _extract_required_domains(prompt_spec: PromptSpec) -> set[str]:
    """Extract all required domains from all data requirements."""
    domains: set[str] = set()
    for req in prompt_spec.data_requirements:
        for target in req.source_targets:
            parsed = urlparse(target.uri)
            domain = parsed.netloc or parsed.path
            domain = domain.lower().lstrip("www.")
            if domain:
                domains.add(domain)
    return domains


def _extract_evidence_domains(evidence_bundles: list[EvidenceBundle]) -> set[str]:
    """Extract all domains present in evidence sources across bundles."""
    domains: set[str] = set()
    for bundle in evidence_bundles:
        for item in bundle.items:
            if not item.success:
                continue
            # Check provenance source_uri
            parsed = urlparse(item.provenance.source_uri)
            host = (parsed.netloc or "").lower().lstrip("www.")
            if host:
                domains.add(host)
            # Check evidence_sources in extracted_fields
            for src in item.extracted_fields.get("evidence_sources", []):
                url = src.get("url", "")
                parsed = urlparse(url)
                host = (parsed.netloc or "").lower().lstrip("www.")
                if host:
                    domains.add(host)
    return domains


def _domain_matches(actual: str, required: str) -> bool:
    """Check if an actual domain matches a required domain (substring match)."""
    return required in actual or actual in required


def _has_deferred_requirements(prompt_spec: PromptSpec) -> bool:
    """Check if any data requirement uses deferred source discovery."""
    return any(req.deferred_source_discovery for req in prompt_spec.data_requirements)


def _assess_source_match(
    prompt_spec: PromptSpec,
    evidence_bundles: list[EvidenceBundle],
) -> str:
    required = _extract_required_domains(prompt_spec)
    if not required:
        return "FULL"  # No domains required → trivially satisfied

    actual = _extract_evidence_domains(evidence_bundles)

    matched = 0
    for req_domain in required:
        if any(_domain_matches(act, req_domain) for act in actual):
            matched += 1

    if matched == len(required):
        return "FULL"
    elif matched > 0:
        return "PARTIAL"

    # When deferred requirements coexist with specific-domain requirements,
    # the specific domains are preferred but not mandatory.  Evidence from
    # non-required domains still satisfies the deferred requirement, so
    # having *any* successful evidence should be PARTIAL, not NONE.
    if _has_deferred_requirements(prompt_spec) and actual:
        return "PARTIAL"

    return "NONE"


# ---------------------------------------------------------------------------
# Data type match assessment
# ---------------------------------------------------------------------------

def _assess_data_type_match(
    prompt_spec: PromptSpec,
    evidence_bundles: list[EvidenceBundle],
) -> bool:
    """Check if evidence data types match market requirements.

    Currently a simple heuristic: if all successful evidence items have
    a resolution_status of RESOLVED or UNRESOLVED, the data type matches.
    Returns False only when evidence contains no successful items at all.
    """
    has_successful = False
    for bundle in evidence_bundles:
        for item in bundle.items:
            if item.success:
                has_successful = True
                break
        if has_successful:
            break
    return has_successful


# ---------------------------------------------------------------------------
# Collector agreement
# ---------------------------------------------------------------------------

def _assess_collector_agreement(
    evidence_bundles: list[EvidenceBundle],
) -> str:
    if len(evidence_bundles) <= 1:
        return "SINGLE"

    outcomes: list[str] = []
    for bundle in evidence_bundles:
        for item in bundle.items:
            if not item.success:
                continue
            outcome = item.extracted_fields.get("outcome", "")
            if outcome:
                outcomes.append(outcome.lower())

    if not outcomes:
        return "SINGLE"

    unique = set(outcomes)
    # Filter out "unresolved" — disagreement only counts between yes/no
    definitive = {o for o in unique if o in ("yes", "no")}
    if len(definitive) <= 1:
        return "AGREE"
    return "DISAGREE"


# ---------------------------------------------------------------------------
# Requirements coverage
# ---------------------------------------------------------------------------

def _assess_requirements_coverage(
    prompt_spec: PromptSpec,
    evidence_bundles: list[EvidenceBundle],
) -> float:
    total_reqs = len(prompt_spec.data_requirements)
    if total_reqs == 0:
        return 1.0

    fulfilled: set[str] = set()
    for bundle in evidence_bundles:
        fulfilled.update(bundle.requirements_fulfilled)

    all_req_ids = {req.requirement_id for req in prompt_spec.data_requirements}
    covered = len(fulfilled & all_req_ids)
    return covered / total_reqs


# ---------------------------------------------------------------------------
# Quality flags and level
# ---------------------------------------------------------------------------

def _build_quality_flags(
    source_match: str,
    data_type_match: bool,
    collector_agreement: str,
    requirements_coverage: float,
) -> list[str]:
    flags: list[str] = []
    if source_match == "NONE":
        flags.append("source_mismatch")
    elif source_match == "PARTIAL":
        flags.append("source_partial")
    if not data_type_match:
        flags.append("data_type_mismatch")
    if collector_agreement == "DISAGREE":
        flags.append("collector_disagreement")
    if requirements_coverage < 1.0:
        flags.append("requirements_gap")
    return flags


def _compute_quality_level(
    quality_flags: list[str],
    source_match: str,
    requirements_coverage: float,
) -> QualityLevel:
    if not quality_flags:
        return "HIGH"

    # Any hard failure → LOW
    hard_failures = {"source_mismatch", "data_type_mismatch"}
    if hard_failures & set(quality_flags):
        return "LOW"

    # Partial source + low coverage → LOW
    if source_match == "PARTIAL" and requirements_coverage < 0.5:
        return "LOW"

    return "MEDIUM"


# ---------------------------------------------------------------------------
# Recommendations (human-readable)
# ---------------------------------------------------------------------------

def _build_recommendations(
    prompt_spec: PromptSpec,
    evidence_bundles: list[EvidenceBundle],
    quality_flags: list[str],
) -> list[str]:
    recs: list[str] = []

    if "source_mismatch" in quality_flags or "source_partial" in quality_flags:
        required = _extract_required_domains(prompt_spec)
        if required:
            recs.append(
                f"Search required domains: {', '.join(sorted(required))}"
            )

    if "data_type_mismatch" in quality_flags:
        recs.append(
            "Market requires specific data types not found in evidence. "
            "Use an appropriate collector for the required data type."
        )

    if "collector_disagreement" in quality_flags:
        recs.append(
            "Collectors disagree on the outcome. Add more collectors "
            "or use site-pinned collection for authoritative sources."
        )

    if "requirements_gap" in quality_flags:
        unfulfilled = _get_unfulfilled_requirements(prompt_spec, evidence_bundles)
        if unfulfilled:
            recs.append(
                f"Unfulfilled requirements: {', '.join(sorted(unfulfilled))}"
            )

    return recs


# ---------------------------------------------------------------------------
# Retry hints (machine-readable)
# ---------------------------------------------------------------------------

def _build_retry_hints(
    prompt_spec: PromptSpec,
    evidence_bundles: list[EvidenceBundle],
    quality_flags: list[str],
) -> dict[str, Any]:
    if not quality_flags:
        return {}

    hints: dict[str, Any] = {}

    required_domains = _extract_required_domains(prompt_spec)

    if "source_mismatch" in quality_flags or "source_partial" in quality_flags:
        hints["required_domains"] = sorted(required_domains)
        # Build site-scoped search queries from source_targets
        search_queries: list[str] = []
        for req in prompt_spec.data_requirements:
            for target in req.source_targets:
                if target.search_query:
                    search_queries.append(target.search_query)
                else:
                    parsed = urlparse(target.uri)
                    domain = parsed.netloc or parsed.path
                    domain = domain.lower().lstrip("www.")
                    if domain:
                        question = prompt_spec.market.question or ""
                        search_queries.append(f"site:{domain} {question}")
        if search_queries:
            hints["search_queries"] = search_queries

        # Domains already tried (skip list)
        tried = _extract_evidence_domains(evidence_bundles)
        skip = sorted(tried - required_domains)
        if skip:
            hints["skip_domains"] = skip

    if "data_type_mismatch" in quality_flags:
        hints["data_type_hint"] = "api_data"

    if "requirements_gap" in quality_flags:
        unfulfilled = _get_unfulfilled_requirements(prompt_spec, evidence_bundles)
        if unfulfilled:
            hints["focus_requirements"] = sorted(unfulfilled)

    # Build collector_guidance summary
    guidance_parts: list[str] = []
    if "source_mismatch" in quality_flags:
        guidance_parts.append(
            f"Previous attempt did not find evidence from required domains "
            f"({', '.join(sorted(required_domains))}). "
            f"Search specifically for these domains."
        )
    if "requirements_gap" in quality_flags:
        unfulfilled = _get_unfulfilled_requirements(prompt_spec, evidence_bundles)
        guidance_parts.append(
            f"Requirements {', '.join(sorted(unfulfilled))} were not fulfilled."
        )
    if "collector_disagreement" in quality_flags:
        guidance_parts.append(
            "Collectors disagreed — seek authoritative sources to break the tie."
        )
    if guidance_parts:
        hints["collector_guidance"] = " ".join(guidance_parts)

    return hints


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_unfulfilled_requirements(
    prompt_spec: PromptSpec,
    evidence_bundles: list[EvidenceBundle],
) -> set[str]:
    all_req_ids = {req.requirement_id for req in prompt_spec.data_requirements}
    fulfilled: set[str] = set()
    for bundle in evidence_bundles:
        fulfilled.update(bundle.requirements_fulfilled)
    return all_req_ids - fulfilled
