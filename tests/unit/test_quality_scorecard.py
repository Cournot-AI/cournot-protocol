"""
Tests for agents.quality.scorecard — compute_scorecard, recommendations, retry_hints.
"""

from datetime import datetime, timezone

import pytest

from core.schemas.evidence import EvidenceBundle, EvidenceItem, Provenance
from core.schemas.market import DisputePolicy, ResolutionRule, ResolutionRules, ResolutionWindow
from core.schemas.prompts import (
    DataRequirement,
    MarketSpec,
    PredictionSemantics,
    PromptSpec,
    SelectionPolicy,
    SourceTarget,
)
from core.schemas.quality import EvidenceQualityScorecard
from agents.quality.scorecard import compute_scorecard


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

def _make_prompt_spec(
    *,
    domains: list[str] | None = None,
    requirement_ids: list[str] | None = None,
) -> PromptSpec:
    """Build a minimal PromptSpec with configurable required domains."""
    req_ids = requirement_ids or ["req_001"]
    reqs = []
    for req_id in req_ids:
        targets = []
        if domains:
            for domain in domains:
                targets.append(SourceTarget(
                    source_id=domain,
                    uri=f"https://{domain}/data",
                    method="GET",
                    expected_content_type="json",
                ))
        reqs.append(DataRequirement(
            requirement_id=req_id,
            description=f"Get data for {req_id}",
            source_targets=targets,
            deferred_source_discovery=len(targets) == 0,
            selection_policy=SelectionPolicy(
                strategy="single_best",
                min_sources=1,
                max_sources=1,
                quorum=1,
            ),
        ))

    return PromptSpec(
        schema_version="1.0.0",
        market=MarketSpec(
            market_id="mk_test",
            question="Will X happen?",
            event_definition="X happens",
            resolution_deadline=datetime(2026, 12, 31, tzinfo=timezone.utc),
            resolution_window=ResolutionWindow(
                start=datetime(2026, 1, 1, tzinfo=timezone.utc),
                end=datetime(2026, 12, 31, tzinfo=timezone.utc),
            ),
            resolution_rules=ResolutionRules(rules=[
                ResolutionRule(rule_id="R1", description="Check event", priority=100),
            ]),
            dispute_policy=DisputePolicy(dispute_window_seconds=3600),
        ),
        prediction_semantics=PredictionSemantics(
            target_entity="X",
            predicate="happens",
        ),
        data_requirements=reqs,
    )


def _make_bundle(
    *,
    bundle_id: str = "bundle_1",
    collector_name: str = "CollectorOpenSearch",
    requirement_id: str = "req_001",
    outcome: str = "Yes",
    success: bool = True,
    evidence_domains: list[str] | None = None,
    fulfilled: list[str] | None = None,
) -> EvidenceBundle:
    """Build a minimal EvidenceBundle."""
    evidence_sources = []
    if evidence_domains:
        for domain in evidence_domains:
            evidence_sources.append({
                "url": f"https://{domain}/page",
                "source_id": domain,
                "credibility_tier": 1,
                "key_fact": "some fact",
                "supports": "YES",
            })

    item = EvidenceItem(
        evidence_id=f"ev_{bundle_id}",
        requirement_id=requirement_id,
        provenance=Provenance(
            source_id="test",
            source_uri=f"https://{(evidence_domains or ['example.com'])[0]}/data",
            tier=1,
        ),
        success=success,
        extracted_fields={
            "outcome": outcome,
            "reason": "Test reason",
            "evidence_sources": evidence_sources,
        },
    )

    bundle = EvidenceBundle(
        bundle_id=bundle_id,
        market_id="mk_test",
        plan_id="plan_test",
        collector_name=collector_name,
    )
    bundle.add_item(item)
    # Override auto-populated fulfillment if caller specified explicitly
    if fulfilled is not None:
        bundle.requirements_fulfilled = fulfilled
        bundle.requirements_unfulfilled = []
    return bundle


# ---------------------------------------------------------------------------
# Tests: basic scorecard computation
# ---------------------------------------------------------------------------

class TestComputeScorecard:

    def test_high_quality_all_good(self):
        """Full source match, single collector, all requirements met → HIGH."""
        ps = _make_prompt_spec(domains=["data.example.com"])
        bundle = _make_bundle(
            evidence_domains=["data.example.com"],
            fulfilled=["req_001"],
        )

        sc = compute_scorecard(ps, [bundle])

        assert sc.quality_level == "HIGH"
        assert sc.meets_threshold is True
        assert sc.source_match == "FULL"
        assert sc.data_type_match is True
        assert sc.requirements_coverage == 1.0
        assert sc.quality_flags == []
        assert sc.recommendations == []
        assert sc.retry_hints == {}

    def test_low_quality_source_mismatch(self):
        """Required domains not in evidence → source_mismatch → LOW."""
        ps = _make_prompt_spec(domains=["data.giss.nasa.gov"])
        bundle = _make_bundle(
            evidence_domains=["climate.gov", "weather.com"],
            fulfilled=["req_001"],
        )

        sc = compute_scorecard(ps, [bundle])

        assert sc.quality_level == "LOW"
        assert sc.meets_threshold is False
        assert sc.source_match == "NONE"
        assert "source_mismatch" in sc.quality_flags

    def test_medium_quality_partial_source(self):
        """Some required domains matched → source_partial → MEDIUM."""
        ps = _make_prompt_spec(
            domains=["data.example.com", "api.other.com"],
            requirement_ids=["req_001", "req_002"],
        )
        bundle = _make_bundle(
            evidence_domains=["data.example.com"],
            fulfilled=["req_001", "req_002"],
        )

        sc = compute_scorecard(ps, [bundle])

        assert sc.quality_level == "MEDIUM"
        assert sc.meets_threshold is True
        assert sc.source_match == "PARTIAL"
        assert "source_partial" in sc.quality_flags

    def test_requirements_gap(self):
        """Unfulfilled requirements → requirements_gap flag."""
        ps = _make_prompt_spec(
            requirement_ids=["req_001", "req_002"],
        )
        bundle = _make_bundle(fulfilled=["req_001"])

        sc = compute_scorecard(ps, [bundle])

        assert sc.requirements_coverage == 0.5
        assert "requirements_gap" in sc.quality_flags

    def test_no_domains_required_is_full(self):
        """When prompt spec has no source_targets, source_match should be FULL."""
        ps = _make_prompt_spec(domains=None)
        bundle = _make_bundle(fulfilled=["req_001"])

        sc = compute_scorecard(ps, [bundle])

        assert sc.source_match == "FULL"

    def test_no_successful_evidence(self):
        """All evidence items failed → data_type_mismatch."""
        ps = _make_prompt_spec()
        bundle = _make_bundle(success=False, fulfilled=[])

        sc = compute_scorecard(ps, [bundle])

        assert sc.data_type_match is False
        assert "data_type_mismatch" in sc.quality_flags


# ---------------------------------------------------------------------------
# Tests: collector agreement
# ---------------------------------------------------------------------------

class TestCollectorAgreement:

    def test_single_collector(self):
        """Single bundle → SINGLE."""
        ps = _make_prompt_spec()
        bundle = _make_bundle(fulfilled=["req_001"])

        sc = compute_scorecard(ps, [bundle])
        assert sc.collector_agreement == "SINGLE"

    def test_agree(self):
        """Two collectors with same outcome → AGREE."""
        ps = _make_prompt_spec()
        b1 = _make_bundle(bundle_id="b1", outcome="Yes", fulfilled=["req_001"])
        b2 = _make_bundle(bundle_id="b2", outcome="Yes", fulfilled=["req_001"])

        sc = compute_scorecard(ps, [b1, b2])
        assert sc.collector_agreement == "AGREE"

    def test_disagree(self):
        """Two collectors with different outcomes → DISAGREE."""
        ps = _make_prompt_spec()
        b1 = _make_bundle(bundle_id="b1", outcome="Yes", fulfilled=["req_001"])
        b2 = _make_bundle(bundle_id="b2", outcome="No", fulfilled=["req_001"])

        sc = compute_scorecard(ps, [b1, b2])
        assert sc.collector_agreement == "DISAGREE"
        assert "collector_disagreement" in sc.quality_flags


# ---------------------------------------------------------------------------
# Tests: recommendations
# ---------------------------------------------------------------------------

class TestRecommendations:

    def test_source_mismatch_recommends_domains(self):
        """source_mismatch flag → recommendation mentions required domains."""
        ps = _make_prompt_spec(domains=["data.giss.nasa.gov"])
        bundle = _make_bundle(
            evidence_domains=["weather.com"],
            fulfilled=["req_001"],
        )

        sc = compute_scorecard(ps, [bundle])

        assert len(sc.recommendations) >= 1
        assert "data.giss.nasa.gov" in sc.recommendations[0]

    def test_requirements_gap_lists_unfulfilled(self):
        """requirements_gap flag → recommendation lists unfulfilled IDs."""
        ps = _make_prompt_spec(requirement_ids=["req_001", "req_002"])
        bundle = _make_bundle(fulfilled=["req_001"])

        sc = compute_scorecard(ps, [bundle])

        recs_text = " ".join(sc.recommendations)
        assert "req_002" in recs_text


# ---------------------------------------------------------------------------
# Tests: retry_hints
# ---------------------------------------------------------------------------

class TestRetryHints:

    def test_source_mismatch_produces_search_queries(self):
        """source_mismatch → retry_hints has search_queries and required_domains."""
        ps = _make_prompt_spec(domains=["data.giss.nasa.gov"])
        bundle = _make_bundle(
            evidence_domains=["weather.com"],
            fulfilled=["req_001"],
        )

        sc = compute_scorecard(ps, [bundle])

        assert "search_queries" in sc.retry_hints
        assert "required_domains" in sc.retry_hints
        assert "data.giss.nasa.gov" in sc.retry_hints["required_domains"]
        assert "collector_guidance" in sc.retry_hints

    def test_requirements_gap_produces_focus_requirements(self):
        """requirements_gap → retry_hints has focus_requirements."""
        ps = _make_prompt_spec(requirement_ids=["req_001", "req_002"])
        bundle = _make_bundle(fulfilled=["req_001"])

        sc = compute_scorecard(ps, [bundle])

        assert "focus_requirements" in sc.retry_hints
        assert "req_002" in sc.retry_hints["focus_requirements"]

    def test_high_quality_empty_retry_hints(self):
        """HIGH quality → retry_hints is empty dict."""
        ps = _make_prompt_spec(domains=["data.example.com"])
        bundle = _make_bundle(
            evidence_domains=["data.example.com"],
            fulfilled=["req_001"],
        )

        sc = compute_scorecard(ps, [bundle])

        assert sc.retry_hints == {}

    def test_collector_disagreement_guidance(self):
        """Disagreement → retry_hints has collector_guidance about tie-breaking."""
        ps = _make_prompt_spec()
        b1 = _make_bundle(bundle_id="b1", outcome="Yes", fulfilled=["req_001"])
        b2 = _make_bundle(bundle_id="b2", outcome="No", fulfilled=["req_001"])

        sc = compute_scorecard(ps, [b1, b2])

        assert "collector_guidance" in sc.retry_hints
        assert "disagree" in sc.retry_hints["collector_guidance"].lower()


# ---------------------------------------------------------------------------
# Tests: meets_threshold
# ---------------------------------------------------------------------------

class TestMeetsThreshold:

    def test_high_meets(self):
        ps = _make_prompt_spec()
        bundle = _make_bundle(fulfilled=["req_001"])
        sc = compute_scorecard(ps, [bundle])
        assert sc.meets_threshold is True

    def test_low_does_not_meet(self):
        ps = _make_prompt_spec(domains=["required-source.com"])
        bundle = _make_bundle(
            evidence_domains=["wrong-source.com"],
            fulfilled=["req_001"],
        )
        sc = compute_scorecard(ps, [bundle])
        assert sc.meets_threshold is False
