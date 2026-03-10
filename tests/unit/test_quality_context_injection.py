"""
Tests for quality_context injection into auditor and judge prompts.

Verifies that when quality_scorecard data is present in ctx.extra["quality_context"],
the auditor (LLMReasoner) and judge (JudgeLLM) inject an EVIDENCE QUALITY ADVISORY
block into their LLM messages.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from agents.context import AgentContext, FrozenClock, NoOpCache
from agents.auditor.llm_reasoner import LLMReasoner
from agents.judge.agent import JudgeLLM
from core.schemas.evidence import EvidenceBundle, EvidenceItem, Provenance
from core.schemas.market import (
    DisputePolicy,
    ResolutionRule,
    ResolutionRules,
    ResolutionWindow,
)
from core.schemas.prompts import (
    DataRequirement,
    MarketSpec,
    PredictionSemantics,
    PromptSpec,
    SelectionPolicy,
    SourceTarget,
)
from core.schemas.reasoning import (
    ReasoningStep,
    ReasoningTrace,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_prompt_spec() -> PromptSpec:
    """Minimal PromptSpec for testing."""
    return PromptSpec(
        schema_version="1.0.0",
        market=MarketSpec(
            market_id="mk_test",
            question="Will BTC hit 100k?",
            event_definition="BTC price >= 100000 USD",
            resolution_deadline=datetime(2026, 12, 31, tzinfo=timezone.utc),
            resolution_window=ResolutionWindow(
                start=datetime(2026, 1, 1, tzinfo=timezone.utc),
                end=datetime(2026, 12, 31, tzinfo=timezone.utc),
            ),
            resolution_rules=ResolutionRules(rules=[
                ResolutionRule(rule_id="R1", description="Check price", priority=100),
            ]),
            dispute_policy=DisputePolicy(dispute_window_seconds=3600),
        ),
        prediction_semantics=PredictionSemantics(
            target_entity="BTC",
            predicate="price >= 100000",
        ),
        data_requirements=[
            DataRequirement(
                requirement_id="req_001",
                description="Get BTC price",
                source_targets=[
                    SourceTarget(
                        source_id="coinbase.com",
                        uri="https://coinbase.com/api/price",
                        method="GET",
                        expected_content_type="json",
                    ),
                ],
                selection_policy=SelectionPolicy(
                    strategy="single_best",
                    min_sources=1,
                    max_sources=1,
                    quorum=1,
                ),
            ),
        ],
    )


def _make_evidence_bundle() -> EvidenceBundle:
    """Minimal EvidenceBundle for testing."""
    item = EvidenceItem(
        evidence_id="ev_001",
        requirement_id="req_001",
        provenance=Provenance(
            source_id="coinbase.com",
            source_uri="https://coinbase.com/api/price",
            tier=1,
        ),
        success=True,
        extracted_fields={
            "outcome": "Yes",
            "reason": "BTC is at 105000",
        },
    )
    bundle = EvidenceBundle(
        bundle_id="bundle_001",
        market_id="mk_test",
        plan_id="plan_test",
        collector_name="CollectorOpenSearch",
    )
    bundle.add_item(item)
    bundle.requirements_fulfilled = ["req_001"]
    return bundle


def _make_reasoning_trace() -> ReasoningTrace:
    """Minimal ReasoningTrace for testing."""
    return ReasoningTrace(
        trace_id="trace_001",
        market_id="mk_test",
        bundle_id="bundle_001",
        steps=[
            ReasoningStep(
                step_id="step_001",
                step_type="inference",
                description="BTC price check",
                evidence_refs=[],
            ),
        ],
        conflicts=[],
        reasoning_summary="BTC is above 100k based on evidence.",
        preliminary_outcome="YES",
        preliminary_confidence=0.85,
        recommended_rule_id="R1",
    )


SAMPLE_QUALITY_CONTEXT = {
    "quality_level": "LOW",
    "meets_threshold": False,
    "quality_flags": ["source_mismatch", "data_type_mismatch"],
    "source_match": "NONE",
    "data_type_match": False,
    "collector_agreement": "SINGLE",
    "requirements_coverage": 1.0,
    "recommendations": [
        "Evidence not from required domains: coinbase.com",
        "No successful evidence items with extracted data",
    ],
    "retry_hints": {
        "search_queries": ["BTC price coinbase.com"],
        "required_domains": ["coinbase.com"],
    },
}


SAMPLE_QUALITY_CONTEXT_HIGH = {
    "quality_level": "HIGH",
    "meets_threshold": True,
    "quality_flags": [],
    "source_match": "FULL",
    "data_type_match": True,
    "collector_agreement": "SINGLE",
    "requirements_coverage": 1.0,
    "recommendations": [],
    "retry_hints": {},
}


def _make_mock_ctx(
    *,
    quality_context: dict[str, Any] | None = None,
    llm_response: str = "{}",
) -> AgentContext:
    """Create a mock AgentContext with optional quality_context and a mock LLM."""
    from core.receipts import ReceiptRecorder

    recorder = ReceiptRecorder()

    # Build a mock LLM that captures .chat() calls
    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.content = llm_response
    mock_llm.chat.return_value = mock_response

    ctx = AgentContext(
        llm=mock_llm,
        recorder=recorder,
        clock=FrozenClock(),
        cache=NoOpCache(),
    )
    if quality_context is not None:
        ctx.extra["quality_context"] = quality_context

    return ctx


# ---------------------------------------------------------------------------
# Tests: _build_quality_context_prompt (static method)
# ---------------------------------------------------------------------------

class TestBuildQualityContextPrompt:
    """Test the prompt builder functions directly."""

    def test_auditor_prompt_contains_key_fields(self):
        prompt = LLMReasoner._build_quality_context_prompt(SAMPLE_QUALITY_CONTEXT)

        assert "EVIDENCE QUALITY ADVISORY" in prompt
        assert "Quality level: LOW" in prompt
        assert "Meets threshold: False" in prompt
        assert "source_mismatch" in prompt
        assert "data_type_mismatch" in prompt
        assert "Source match: NONE" in prompt
        assert "coinbase.com" in prompt
        assert "INSTRUCTIONS" in prompt

    def test_auditor_prompt_no_flags(self):
        prompt = LLMReasoner._build_quality_context_prompt(SAMPLE_QUALITY_CONTEXT_HIGH)

        assert "EVIDENCE QUALITY ADVISORY" in prompt
        assert "Quality level: HIGH" in prompt
        assert "Meets threshold: True" in prompt
        # No flags or recommendations should appear
        assert "Quality flags:" not in prompt
        assert "Issues identified:" not in prompt

    def test_judge_prompt_contains_key_fields(self):
        prompt = JudgeLLM._build_quality_context_prompt(SAMPLE_QUALITY_CONTEXT)

        assert "EVIDENCE QUALITY ADVISORY" in prompt
        assert "Quality level: LOW" in prompt
        assert "Meets threshold: False" in prompt
        assert "source_mismatch" in prompt
        assert "Source match: NONE" in prompt
        assert "INVALID" in prompt  # judge prompt instructs to return INVALID

    def test_judge_prompt_no_flags(self):
        prompt = JudgeLLM._build_quality_context_prompt(SAMPLE_QUALITY_CONTEXT_HIGH)

        assert "EVIDENCE QUALITY ADVISORY" in prompt
        assert "Quality level: HIGH" in prompt
        assert "Quality flags:" not in prompt


# ---------------------------------------------------------------------------
# Tests: LLMReasoner quality_context injection
# ---------------------------------------------------------------------------

class TestAuditorQualityContextInjection:
    """Verify LLMReasoner.reason() injects quality_context into LLM messages."""

    def _make_valid_llm_response(self) -> str:
        return json.dumps({
            "trace_id": "trace_test",
            "steps": [
                {
                    "step_id": "step_001",
                    "step_type": "inference",
                    "description": "Checked evidence",
                    "evidence_refs": [],
                }
            ],
            "conflicts": [],
            "evidence_summary": "Evidence checked.",
            "reasoning_summary": "Evidence is from wrong source.",
            "preliminary_outcome": "INVALID",
            "preliminary_confidence": 0.3,
            "recommended_rule_id": "R1",
        })

    def test_quality_context_injected_into_messages(self):
        """When quality_context is in ctx.extra, the LLM should receive it."""
        ctx = _make_mock_ctx(
            quality_context=SAMPLE_QUALITY_CONTEXT,
            llm_response=self._make_valid_llm_response(),
        )
        reasoner = LLMReasoner(strict_mode=True)
        prompt_spec = _make_prompt_spec()
        bundles = [_make_evidence_bundle()]

        reasoner.reason(ctx, prompt_spec, bundles)

        # Verify LLM was called and messages contain quality advisory
        call_args = ctx.llm.chat.call_args
        messages = call_args[0][0]  # first positional arg

        quality_messages = [
            m for m in messages
            if "EVIDENCE QUALITY ADVISORY" in m.get("content", "")
        ]
        assert len(quality_messages) == 1
        assert "Quality level: LOW" in quality_messages[0]["content"]
        assert "Source match: NONE" in quality_messages[0]["content"]

    def test_no_quality_context_no_injection(self):
        """When quality_context is absent, no advisory should be injected."""
        ctx = _make_mock_ctx(
            quality_context=None,
            llm_response=self._make_valid_llm_response(),
        )
        reasoner = LLMReasoner(strict_mode=True)
        prompt_spec = _make_prompt_spec()
        bundles = [_make_evidence_bundle()]

        reasoner.reason(ctx, prompt_spec, bundles)

        call_args = ctx.llm.chat.call_args
        messages = call_args[0][0]

        quality_messages = [
            m for m in messages
            if "EVIDENCE QUALITY ADVISORY" in m.get("content", "")
        ]
        assert len(quality_messages) == 0

    def test_quality_context_after_dispute_context(self):
        """Quality context should be injected after dispute context."""
        ctx = _make_mock_ctx(
            quality_context=SAMPLE_QUALITY_CONTEXT,
            llm_response=self._make_valid_llm_response(),
        )
        ctx.extra["dispute_context"] = {
            "reason_code": "WRONG_SOURCE",
            "message": "Evidence is from the wrong source",
        }
        reasoner = LLMReasoner(strict_mode=True)
        prompt_spec = _make_prompt_spec()
        bundles = [_make_evidence_bundle()]

        reasoner.reason(ctx, prompt_spec, bundles)

        call_args = ctx.llm.chat.call_args
        messages = call_args[0][0]

        # Find indices of dispute and quality messages
        dispute_idx = None
        quality_idx = None
        for i, m in enumerate(messages):
            content = m.get("content", "")
            if "DISPUTE CONTEXT" in content:
                dispute_idx = i
            if "EVIDENCE QUALITY ADVISORY" in content:
                quality_idx = i

        assert dispute_idx is not None
        assert quality_idx is not None
        assert quality_idx > dispute_idx


# ---------------------------------------------------------------------------
# Tests: JudgeLLM quality_context injection
# ---------------------------------------------------------------------------

class TestJudgeQualityContextInjection:
    """Verify JudgeLLM._get_llm_review() injects quality_context into LLM messages."""

    def _make_valid_llm_response(self) -> str:
        return json.dumps({
            "outcome": "INVALID",
            "confidence": 0.3,
            "resolution_rule_id": "R1",
            "reasoning_valid": True,
            "reasoning_issues": [],
            "final_justification": "Evidence from wrong source, returning INVALID.",
        })

    def test_quality_context_injected_into_messages(self):
        """When quality_context is in ctx.extra, the judge LLM should receive it."""
        ctx = _make_mock_ctx(
            quality_context=SAMPLE_QUALITY_CONTEXT,
            llm_response=self._make_valid_llm_response(),
        )
        judge = JudgeLLM(strict_mode=True)
        prompt_spec = _make_prompt_spec()
        bundles = [_make_evidence_bundle()]
        trace = _make_reasoning_trace()

        # Call _get_llm_review directly to inspect messages
        judge._get_llm_review(ctx, prompt_spec, bundles, trace)

        call_args = ctx.llm.chat.call_args
        messages = call_args[0][0]

        quality_messages = [
            m for m in messages
            if "EVIDENCE QUALITY ADVISORY" in m.get("content", "")
        ]
        assert len(quality_messages) == 1
        assert "Quality level: LOW" in quality_messages[0]["content"]
        assert "INVALID" in quality_messages[0]["content"]

    def test_no_quality_context_no_injection(self):
        """When quality_context is absent, no advisory should be injected."""
        ctx = _make_mock_ctx(
            quality_context=None,
            llm_response=self._make_valid_llm_response(),
        )
        judge = JudgeLLM(strict_mode=True)
        prompt_spec = _make_prompt_spec()
        bundles = [_make_evidence_bundle()]
        trace = _make_reasoning_trace()

        judge._get_llm_review(ctx, prompt_spec, bundles, trace)

        call_args = ctx.llm.chat.call_args
        messages = call_args[0][0]

        quality_messages = [
            m for m in messages
            if "EVIDENCE QUALITY ADVISORY" in m.get("content", "")
        ]
        assert len(quality_messages) == 0

    def test_quality_context_after_dispute_context(self):
        """Quality context should be injected after dispute context in judge."""
        ctx = _make_mock_ctx(
            quality_context=SAMPLE_QUALITY_CONTEXT,
            llm_response=self._make_valid_llm_response(),
        )
        ctx.extra["dispute_context"] = {
            "reason_code": "WRONG_SOURCE",
            "message": "Evidence is from the wrong source",
        }
        judge = JudgeLLM(strict_mode=True)
        prompt_spec = _make_prompt_spec()
        bundles = [_make_evidence_bundle()]
        trace = _make_reasoning_trace()

        judge._get_llm_review(ctx, prompt_spec, bundles, trace)

        call_args = ctx.llm.chat.call_args
        messages = call_args[0][0]

        dispute_idx = None
        quality_idx = None
        for i, m in enumerate(messages):
            content = m.get("content", "")
            if "DISPUTE CONTEXT" in content:
                dispute_idx = i
            if "EVIDENCE QUALITY ADVISORY" in content:
                quality_idx = i

        assert dispute_idx is not None
        assert quality_idx is not None
        assert quality_idx > dispute_idx

    def test_full_judge_run_with_quality_context(self):
        """End-to-end: JudgeLLM.run() should work with quality_context."""
        ctx = _make_mock_ctx(
            quality_context=SAMPLE_QUALITY_CONTEXT,
            llm_response=self._make_valid_llm_response(),
        )
        judge = JudgeLLM(strict_mode=True)
        prompt_spec = _make_prompt_spec()
        bundles = [_make_evidence_bundle()]
        trace = _make_reasoning_trace()

        result = judge.run(ctx, prompt_spec, bundles, trace)

        assert result.success
        verdict = result.output
        assert verdict.outcome == "INVALID"
