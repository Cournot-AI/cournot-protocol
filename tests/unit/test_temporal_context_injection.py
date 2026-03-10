"""
Tests for temporal_context injection into auditor and judge prompts.

Verifies that when temporal_constraint data is present in ctx.extra["temporal_context"],
the auditor (LLMReasoner) and judge (JudgeLLM) inject a TEMPORAL ADVISORY
block into their LLM messages, computing the temporal status (FUTURE/ACTIVE/PAST)
at resolution time from event_time vs current_time.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock

import pytest

from agents.context import AgentContext, FrozenClock, NoOpCache
from agents.auditor.llm_reasoner import LLMReasoner, _resolve_temporal_status
from agents.judge.agent import JudgeLLM
from agents.judge.agent import _resolve_temporal_status as _judge_resolve_temporal_status
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
# Constants — FrozenClock defaults to 2026-01-01T00:00:00Z
# ---------------------------------------------------------------------------

FROZEN_NOW = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_prompt_spec() -> PromptSpec:
    """Minimal PromptSpec for testing."""
    return PromptSpec(
        schema_version="1.0.0",
        market=MarketSpec(
            market_id="mk_test",
            question="Will Team A win on Sep 23 2026?",
            event_definition="Team A wins the match on 2026-09-23",
            resolution_deadline=datetime(2026, 12, 31, tzinfo=timezone.utc),
            resolution_window=ResolutionWindow(
                start=datetime(2026, 9, 23, tzinfo=timezone.utc),
                end=datetime(2026, 9, 24, tzinfo=timezone.utc),
            ),
            resolution_rules=ResolutionRules(rules=[
                ResolutionRule(rule_id="R1", description="Check result", priority=100),
            ]),
            dispute_policy=DisputePolicy(dispute_window_seconds=3600),
        ),
        prediction_semantics=PredictionSemantics(
            target_entity="Team A",
            predicate="wins the match",
            timeframe="2026-09-23",
        ),
        data_requirements=[
            DataRequirement(
                requirement_id="req_001",
                description="Get match result",
                source_targets=[
                    SourceTarget(
                        source_id="espn.com",
                        uri="https://espn.com/api/match",
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
            source_id="espn.com",
            source_uri="https://espn.com/api/match",
            tier=1,
        ),
        success=True,
        extracted_fields={
            "outcome": "Yes",
            "reason": "Team A won 3-1",
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
                description="Match result check",
                evidence_refs=[],
            ),
        ],
        conflicts=[],
        reasoning_summary="Team A won based on evidence.",
        preliminary_outcome="YES",
        preliminary_confidence=0.85,
        recommended_rule_id="R1",
    )


# temporal_constraint as stored in prompt_spec.extra — NO status field.
# Status is computed at resolution time from event_time vs current_time.
SAMPLE_TEMPORAL_FUTURE = {
    "enabled": True,
    "event_time": "2026-09-23T00:00:00Z",
    "reason": "Match is scheduled for Sep 23 2026",
}

SAMPLE_TEMPORAL_ACTIVE = {
    "enabled": True,
    "event_time": "2026-01-01T06:00:00Z",  # 6 hours after FROZEN_NOW → within 24h → ACTIVE
    "reason": "Event recently started",
}

SAMPLE_TEMPORAL_PAST = {
    "enabled": True,
    "event_time": "2025-06-15T00:00:00Z",  # well before 2026-01-01 → PAST
    "reason": "Event concluded in June 2025",
}


def _make_mock_ctx(
    *,
    temporal_context: dict[str, Any] | None = None,
    quality_context: dict[str, Any] | None = None,
    dispute_context: dict[str, Any] | None = None,
    llm_response: str = "{}",
) -> AgentContext:
    """Create a mock AgentContext with optional contexts and a mock LLM."""
    from core.receipts import ReceiptRecorder

    recorder = ReceiptRecorder()

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
    if temporal_context is not None:
        ctx.extra["temporal_context"] = temporal_context
    if quality_context is not None:
        ctx.extra["quality_context"] = quality_context
    if dispute_context is not None:
        ctx.extra["dispute_context"] = dispute_context

    return ctx


# ---------------------------------------------------------------------------
# Tests: _resolve_temporal_status helper
# ---------------------------------------------------------------------------

class TestResolveTemporalStatus:
    """Test the status computation from event_time vs current_time."""

    def test_future_event(self):
        status = _resolve_temporal_status("2026-09-23T00:00:00Z", FROZEN_NOW)
        assert status == "FUTURE"

    def test_active_event_within_24h(self):
        # event_time is 6 hours AFTER frozen now — still FUTURE
        # but let's test with event_time slightly before now
        past_6h = datetime(2025, 12, 31, 18, 0, 0, tzinfo=timezone.utc)
        status = _resolve_temporal_status(past_6h.isoformat(), FROZEN_NOW)
        assert status == "ACTIVE"

    def test_past_event(self):
        status = _resolve_temporal_status("2025-06-15T00:00:00Z", FROZEN_NOW)
        assert status == "PAST"

    def test_invalid_event_time(self):
        status = _resolve_temporal_status("not-a-date", FROZEN_NOW)
        assert status == "UNKNOWN"

    def test_empty_event_time(self):
        status = _resolve_temporal_status("", FROZEN_NOW)
        assert status == "UNKNOWN"

    def test_judge_resolve_same_as_auditor(self):
        """Both modules should produce the same result."""
        assert _resolve_temporal_status("2026-09-23T00:00:00Z", FROZEN_NOW) == \
               _judge_resolve_temporal_status("2026-09-23T00:00:00Z", FROZEN_NOW)


# ---------------------------------------------------------------------------
# Tests: _build_temporal_context_prompt (static method)
# ---------------------------------------------------------------------------

class TestBuildTemporalContextPrompt:
    """Test the prompt builder functions directly."""

    def test_auditor_prompt_future(self):
        prompt = LLMReasoner._build_temporal_context_prompt(SAMPLE_TEMPORAL_FUTURE, FROZEN_NOW)

        assert "TEMPORAL ADVISORY" in prompt
        assert "Temporal status: FUTURE" in prompt
        assert "2026-09-23T00:00:00Z" in prompt
        assert "MUST return preliminary_outcome = INVALID" in prompt

    def test_auditor_prompt_active(self):
        # Use a current_time that makes SAMPLE_TEMPORAL_ACTIVE's event_time in the past 24h
        now = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        prompt = LLMReasoner._build_temporal_context_prompt(SAMPLE_TEMPORAL_ACTIVE, now)

        assert "TEMPORAL ADVISORY" in prompt
        assert "Temporal status: ACTIVE" in prompt
        assert "CONCLUDED outcome" in prompt
        assert "INVALID, not NO" in prompt

    def test_auditor_prompt_past(self):
        prompt = LLMReasoner._build_temporal_context_prompt(SAMPLE_TEMPORAL_PAST, FROZEN_NOW)

        assert "TEMPORAL ADVISORY" in prompt
        assert "Temporal status: PAST" in prompt
        assert "Evaluate evidence normally" in prompt

    def test_judge_prompt_future(self):
        prompt = JudgeLLM._build_temporal_context_prompt(SAMPLE_TEMPORAL_FUTURE, FROZEN_NOW)

        assert "TEMPORAL ADVISORY" in prompt
        assert "Temporal status: FUTURE" in prompt
        assert "MUST return outcome = INVALID" in prompt

    def test_judge_prompt_active(self):
        now = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        prompt = JudgeLLM._build_temporal_context_prompt(SAMPLE_TEMPORAL_ACTIVE, now)

        assert "TEMPORAL ADVISORY" in prompt
        assert "Temporal status: ACTIVE" in prompt
        assert "override to INVALID" in prompt

    def test_judge_prompt_past(self):
        prompt = JudgeLLM._build_temporal_context_prompt(SAMPLE_TEMPORAL_PAST, FROZEN_NOW)

        assert "TEMPORAL ADVISORY" in prompt
        assert "Temporal status: PAST" in prompt
        assert "normally" in prompt

    def test_prompt_includes_current_time(self):
        """The advisory should show current_time so the LLM sees both timestamps."""
        prompt = LLMReasoner._build_temporal_context_prompt(SAMPLE_TEMPORAL_FUTURE, FROZEN_NOW)
        assert "Current time: 2026-01-01T00:00:00+00:00" in prompt


# ---------------------------------------------------------------------------
# Tests: LLMReasoner temporal_context injection
# ---------------------------------------------------------------------------

class TestAuditorTemporalContextInjection:
    """Verify LLMReasoner.reason() injects temporal_context into LLM messages."""

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
            "reasoning_summary": "Event is in the future, returning INVALID.",
            "preliminary_outcome": "INVALID",
            "preliminary_confidence": 0.25,
            "recommended_rule_id": "R1",
        })

    def test_temporal_context_injected_into_messages(self):
        """When temporal_context is in ctx.extra, the LLM should receive it."""
        ctx = _make_mock_ctx(
            temporal_context=SAMPLE_TEMPORAL_FUTURE,
            llm_response=self._make_valid_llm_response(),
        )
        reasoner = LLMReasoner(strict_mode=True)
        prompt_spec = _make_prompt_spec()
        bundles = [_make_evidence_bundle()]

        reasoner.reason(ctx, prompt_spec, bundles)

        call_args = ctx.llm.chat.call_args
        messages = call_args[0][0]

        temporal_messages = [
            m for m in messages
            if m.get("role") == "user" and "## TEMPORAL ADVISORY" in m.get("content", "")
        ]
        assert len(temporal_messages) == 1
        assert "Temporal status: FUTURE" in temporal_messages[0]["content"]
        assert "2026-09-23T00:00:00Z" in temporal_messages[0]["content"]

    def test_no_temporal_context_no_injection(self):
        """When temporal_context is absent, no advisory should be injected."""
        ctx = _make_mock_ctx(
            temporal_context=None,
            llm_response=self._make_valid_llm_response(),
        )
        reasoner = LLMReasoner(strict_mode=True)
        prompt_spec = _make_prompt_spec()
        bundles = [_make_evidence_bundle()]

        reasoner.reason(ctx, prompt_spec, bundles)

        call_args = ctx.llm.chat.call_args
        messages = call_args[0][0]

        temporal_messages = [
            m for m in messages
            if m.get("role") == "user" and "## TEMPORAL ADVISORY" in m.get("content", "")
        ]
        assert len(temporal_messages) == 0

    def test_temporal_after_quality_after_dispute(self):
        """Temporal context should be injected after quality, which is after dispute."""
        ctx = _make_mock_ctx(
            temporal_context=SAMPLE_TEMPORAL_FUTURE,
            quality_context={
                "quality_level": "HIGH",
                "meets_threshold": True,
                "quality_flags": [],
                "recommendations": [],
            },
            dispute_context={
                "reason_code": "WRONG_SOURCE",
                "message": "Evidence is from the wrong source",
            },
            llm_response=self._make_valid_llm_response(),
        )
        reasoner = LLMReasoner(strict_mode=True)
        prompt_spec = _make_prompt_spec()
        bundles = [_make_evidence_bundle()]

        reasoner.reason(ctx, prompt_spec, bundles)

        call_args = ctx.llm.chat.call_args
        messages = call_args[0][0]

        dispute_idx = None
        quality_idx = None
        temporal_idx = None
        for i, m in enumerate(messages):
            content = m.get("content", "")
            if "DISPUTE CONTEXT" in content:
                dispute_idx = i
            if "EVIDENCE QUALITY ADVISORY" in content:
                quality_idx = i
            if "TEMPORAL ADVISORY" in content:
                temporal_idx = i

        assert dispute_idx is not None
        assert quality_idx is not None
        assert temporal_idx is not None
        assert dispute_idx < quality_idx < temporal_idx

    def test_past_event_still_injected(self):
        """Even for past events, temporal advisory is injected (says 'evaluate normally')."""
        ctx = _make_mock_ctx(
            temporal_context=SAMPLE_TEMPORAL_PAST,
            llm_response=self._make_valid_llm_response(),
        )
        reasoner = LLMReasoner(strict_mode=True)
        prompt_spec = _make_prompt_spec()
        bundles = [_make_evidence_bundle()]

        reasoner.reason(ctx, prompt_spec, bundles)

        call_args = ctx.llm.chat.call_args
        messages = call_args[0][0]

        temporal_messages = [
            m for m in messages
            if m.get("role") == "user" and "## TEMPORAL ADVISORY" in m.get("content", "")
        ]
        assert len(temporal_messages) == 1
        assert "Temporal status: PAST" in temporal_messages[0]["content"]


# ---------------------------------------------------------------------------
# Tests: JudgeLLM temporal_context injection
# ---------------------------------------------------------------------------

class TestJudgeTemporalContextInjection:
    """Verify JudgeLLM._get_llm_review() injects temporal_context into LLM messages."""

    def _make_valid_llm_response(self) -> str:
        return json.dumps({
            "outcome": "INVALID",
            "confidence": 0.25,
            "resolution_rule_id": "R1",
            "reasoning_valid": True,
            "reasoning_issues": [],
            "final_justification": "Event is in the future, returning INVALID.",
        })

    def test_temporal_context_injected_into_messages(self):
        """When temporal_context is in ctx.extra, the judge LLM should receive it."""
        ctx = _make_mock_ctx(
            temporal_context=SAMPLE_TEMPORAL_FUTURE,
            llm_response=self._make_valid_llm_response(),
        )
        judge = JudgeLLM(strict_mode=True)
        prompt_spec = _make_prompt_spec()
        bundles = [_make_evidence_bundle()]
        trace = _make_reasoning_trace()

        judge._get_llm_review(ctx, prompt_spec, bundles, trace)

        call_args = ctx.llm.chat.call_args
        messages = call_args[0][0]

        temporal_messages = [
            m for m in messages
            if m.get("role") == "user" and "## TEMPORAL ADVISORY" in m.get("content", "")
        ]
        assert len(temporal_messages) == 1
        assert "Temporal status: FUTURE" in temporal_messages[0]["content"]
        assert "INVALID" in temporal_messages[0]["content"]

    def test_no_temporal_context_no_injection(self):
        """When temporal_context is absent, no advisory should be injected."""
        ctx = _make_mock_ctx(
            temporal_context=None,
            llm_response=self._make_valid_llm_response(),
        )
        judge = JudgeLLM(strict_mode=True)
        prompt_spec = _make_prompt_spec()
        bundles = [_make_evidence_bundle()]
        trace = _make_reasoning_trace()

        judge._get_llm_review(ctx, prompt_spec, bundles, trace)

        call_args = ctx.llm.chat.call_args
        messages = call_args[0][0]

        temporal_messages = [
            m for m in messages
            if m.get("role") == "user" and "## TEMPORAL ADVISORY" in m.get("content", "")
        ]
        assert len(temporal_messages) == 0

    def test_temporal_after_quality_after_dispute(self):
        """Temporal context should be injected after quality, which is after dispute."""
        ctx = _make_mock_ctx(
            temporal_context=SAMPLE_TEMPORAL_FUTURE,
            quality_context={
                "quality_level": "HIGH",
                "meets_threshold": True,
                "quality_flags": [],
                "recommendations": [],
            },
            dispute_context={
                "reason_code": "WRONG_SOURCE",
                "message": "Evidence is from the wrong source",
            },
            llm_response=self._make_valid_llm_response(),
        )
        judge = JudgeLLM(strict_mode=True)
        prompt_spec = _make_prompt_spec()
        bundles = [_make_evidence_bundle()]
        trace = _make_reasoning_trace()

        judge._get_llm_review(ctx, prompt_spec, bundles, trace)

        call_args = ctx.llm.chat.call_args
        messages = call_args[0][0]

        dispute_idx = None
        quality_idx = None
        temporal_idx = None
        for i, m in enumerate(messages):
            content = m.get("content", "")
            if "DISPUTE CONTEXT" in content:
                dispute_idx = i
            if "EVIDENCE QUALITY ADVISORY" in content:
                quality_idx = i
            if "TEMPORAL ADVISORY" in content:
                temporal_idx = i

        assert dispute_idx is not None
        assert quality_idx is not None
        assert temporal_idx is not None
        assert dispute_idx < quality_idx < temporal_idx

    def test_full_judge_run_with_temporal_context(self):
        """End-to-end: JudgeLLM.run() should work with temporal_context."""
        ctx = _make_mock_ctx(
            temporal_context=SAMPLE_TEMPORAL_FUTURE,
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
