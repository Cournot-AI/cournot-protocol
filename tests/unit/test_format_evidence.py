"""Tests for POST /step/format_evidence endpoint."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any

import pytest

from api.routes.format_evidence import (
    ExternalEvidenceItem,
    FormatEvidenceRequest,
    FormatEvidenceResponse,
    run_format_evidence,
)
from core.schemas.evidence import EvidenceBundle


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_NOW = datetime(2026, 3, 1, 12, 0, 0, tzinfo=timezone.utc)


def _prompt_spec_dict(
    market_id: str = "mk_test",
    data_requirements: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Minimal valid prompt_spec dict."""
    reqs = data_requirements if data_requirements is not None else []
    return {
        "market": {
            "market_id": market_id,
            "question": "Will BTC be above $100k?",
            "event_definition": "price(BTC_USD) > 100000",
            "resolution_deadline": _NOW.isoformat(),
            "resolution_window": {
                "start": _NOW.isoformat(),
                "end": _NOW.isoformat(),
            },
            "resolution_rules": {
                "rules": [
                    {
                        "rule_id": "R_THRESHOLD",
                        "description": "Compare to threshold",
                        "priority": 100,
                    }
                ],
            },
            "dispute_policy": {"dispute_window_seconds": 86400},
        },
        "prediction_semantics": {
            "target_entity": "bitcoin",
            "predicate": "price above threshold",
        },
        "data_requirements": reqs,
    }


def _data_req(req_id: str, desc: str = "test requirement") -> dict[str, Any]:
    """Minimal DataRequirement dict with deferred discovery."""
    return {
        "requirement_id": req_id,
        "description": desc,
        "source_targets": [],
        "deferred_source_discovery": True,
        "selection_policy": {
            "strategy": "single_best",
            "min_sources": 1,
            "max_sources": 1,
            "quorum": 1,
        },
    }


def _tool_plan_dict(plan_id: str = "plan_abc") -> dict[str, Any]:
    return {"plan_id": plan_id, "requirements": []}


def _make_request(**overrides: Any) -> FormatEvidenceRequest:
    """Build a FormatEvidenceRequest with sensible defaults, applying overrides."""
    defaults: dict[str, Any] = {
        "prompt_spec": _prompt_spec_dict(),
        "items": [
            {
                "source_url": "https://example.com/data",
                "content": "default content",
            },
        ],
    }
    defaults.update(overrides)
    return FormatEvidenceRequest(**defaults)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_happy_path_multiple_items():
    """Multiple items with all optional fields should produce a valid bundle."""
    req = FormatEvidenceRequest(
        prompt_spec=_prompt_spec_dict(
            data_requirements=[_data_req("req_a"), _data_req("req_b")],
        ),
        tool_plan=_tool_plan_dict("plan_123"),
        items=[
            ExternalEvidenceItem(
                source_url="https://api.example.com/data",
                content='{"price": 105000}',
                parsed_value=105000,
                extracted_fields={"price": 105000},
                tier=2,
                requirement_id="req_a",
                source_id="example_api",
                evidence_id="ev_custom_1",
            ),
            ExternalEvidenceItem(
                source_url="https://api.other.com/quote",
                content='{"price": 104500}',
                parsed_value=104500,
                extracted_fields={"price": 104500},
                tier=1,
                requirement_id="req_b",
            ),
        ],
        collector_name="MyCronJob",
        bundle_id="eb_custom_bundle",
        include_raw_content=True,
    )

    resp = await run_format_evidence(req)
    assert resp.ok is True
    assert resp.collectors_used == ["MyCronJob"]
    assert len(resp.evidence_bundles) == 1
    assert len(resp.execution_logs) == 1

    bundle = resp.evidence_bundles[0]
    assert bundle["bundle_id"] == "eb_custom_bundle"
    assert bundle["market_id"] == "mk_test"
    assert bundle["plan_id"] == "plan_123"
    assert bundle["collector_name"] == "MyCronJob"
    assert len(bundle["items"]) == 2

    # First item: custom IDs preserved
    item0 = bundle["items"][0]
    assert item0["evidence_id"] == "ev_custom_1"
    assert item0["requirement_id"] == "req_a"
    assert item0["provenance"]["source_id"] == "example_api"
    assert item0["provenance"]["tier"] == 2
    assert item0["raw_content"] == '{"price": 105000}'
    assert item0["parsed_value"] == 105000
    assert item0["extracted_fields"] == {"price": 105000}

    # Second item: auto-generated source_id from domain
    item1 = bundle["items"][1]
    assert item1["provenance"]["source_id"] == "api.other.com"

    # Requirements cross-reference
    assert sorted(bundle["requirements_fulfilled"]) == ["req_a", "req_b"]
    assert bundle["requirements_unfulfilled"] == []


@pytest.mark.asyncio
async def test_minimal_input():
    """Only source_url and content, no tool_plan — should auto-generate IDs."""
    req = _make_request()

    resp = await run_format_evidence(req)
    assert resp.ok is True

    bundle = resp.evidence_bundles[0]

    # Auto-generated plan_id should start with ext_
    assert bundle["plan_id"].startswith("ext_")

    # Auto-generated bundle_id should start with eb_
    assert bundle["bundle_id"].startswith("eb_")

    # Single item with auto-generated IDs
    item = bundle["items"][0]
    assert item["evidence_id"].startswith("ev_")
    assert item["requirement_id"] == "external_0"
    assert item["provenance"]["source_id"] == "example.com"
    assert item["provenance"]["tier"] == 1  # default

    # Default: raw_content/parsed_value should be stripped
    assert item["raw_content"] is None
    assert item["parsed_value"] is None

    # Default collector name
    assert resp.collectors_used == ["ExternalEvidence"]


@pytest.mark.asyncio
async def test_invalid_prompt_spec():
    """Invalid prompt_spec should return ok=False with error."""
    req = FormatEvidenceRequest(
        prompt_spec={"invalid": "data"},
        items=[
            ExternalEvidenceItem(
                source_url="https://example.com",
                content="test",
            ),
        ],
    )

    resp = await run_format_evidence(req)
    assert resp.ok is False
    assert len(resp.errors) > 0
    assert "Invalid prompt_spec" in resp.errors[0]


@pytest.mark.asyncio
async def test_custom_ids_passthrough():
    """Custom evidence_id, source_id, requirement_id should pass through unchanged."""
    req = _make_request(
        items=[
            {
                "source_url": "https://example.com/api",
                "content": "data",
                "evidence_id": "ev_my_id",
                "source_id": "my_source",
                "requirement_id": "my_req",
            },
        ],
    )

    resp = await run_format_evidence(req)
    assert resp.ok is True

    item = resp.evidence_bundles[0]["items"][0]
    assert item["evidence_id"] == "ev_my_id"
    assert item["requirement_id"] == "my_req"
    assert item["provenance"]["source_id"] == "my_source"


@pytest.mark.asyncio
async def test_include_raw_content_true():
    """include_raw_content=True should preserve raw_content and parsed_value."""
    req = _make_request(
        items=[
            {
                "source_url": "https://example.com",
                "content": "the raw text",
                "parsed_value": {"key": "val"},
            },
        ],
        include_raw_content=True,
    )

    resp = await run_format_evidence(req)
    item = resp.evidence_bundles[0]["items"][0]
    assert item["raw_content"] == "the raw text"
    assert item["parsed_value"] == {"key": "val"}


@pytest.mark.asyncio
async def test_include_raw_content_false_strips():
    """include_raw_content=False (default) should strip raw_content and parsed_value."""
    req = _make_request(
        items=[
            {
                "source_url": "https://example.com",
                "content": "the raw text",
                "parsed_value": 42,
            },
        ],
        include_raw_content=False,
    )

    resp = await run_format_evidence(req)
    item = resp.evidence_bundles[0]["items"][0]
    assert item["raw_content"] is None
    assert item["parsed_value"] is None


@pytest.mark.asyncio
async def test_requirements_cross_reference():
    """Fulfilled vs unfulfilled requirements should be tracked correctly."""
    req = FormatEvidenceRequest(
        prompt_spec=_prompt_spec_dict(
            data_requirements=[
                _data_req("req_fulfilled"),
                _data_req("req_missing"),
                _data_req("req_also_fulfilled"),
            ],
        ),
        items=[
            ExternalEvidenceItem(
                source_url="https://example.com/1",
                content="evidence 1",
                requirement_id="req_fulfilled",
            ),
            ExternalEvidenceItem(
                source_url="https://example.com/2",
                content="evidence 2",
                requirement_id="req_also_fulfilled",
            ),
        ],
    )

    resp = await run_format_evidence(req)
    bundle = resp.evidence_bundles[0]

    assert sorted(bundle["requirements_fulfilled"]) == [
        "req_also_fulfilled",
        "req_fulfilled",
    ]
    assert bundle["requirements_unfulfilled"] == ["req_missing"]


@pytest.mark.asyncio
async def test_schema_roundtrip():
    """Response evidence_bundles[0] should deserialize as a valid EvidenceBundle."""
    req = _make_request(
        tool_plan=_tool_plan_dict(),
        items=[
            {
                "source_url": "https://example.com/data",
                "content": "roundtrip test",
            },
        ],
        include_raw_content=True,
    )

    resp = await run_format_evidence(req)
    assert resp.ok is True

    # Should deserialize without error
    bundle = EvidenceBundle(**resp.evidence_bundles[0])
    assert bundle.market_id == "mk_test"
    assert len(bundle.items) == 1
    assert bundle.items[0].provenance.source_id == "example.com"


@pytest.mark.asyncio
async def test_content_hash_format():
    """content_hash should follow the 0x + sha256 convention."""
    content = "deterministic content"
    expected_hash = "0x" + hashlib.sha256(content.encode()).hexdigest()

    req = _make_request(
        items=[
            {
                "source_url": "https://example.com",
                "content": content,
            },
        ],
        include_raw_content=True,
    )

    resp = await run_format_evidence(req)
    item = resp.evidence_bundles[0]["items"][0]
    assert item["provenance"]["content_hash"] == expected_hash


@pytest.mark.asyncio
async def test_tool_plan_missing_plan_id():
    """tool_plan provided but missing plan_id should return ok=False."""
    req = _make_request(
        tool_plan={"requirements": []},
    )

    resp = await run_format_evidence(req)
    assert resp.ok is False
    assert "plan_id" in resp.errors[0]


@pytest.mark.asyncio
async def test_execution_log_shape():
    """Execution log should contain plan_id and empty calls list."""
    req = _make_request(
        tool_plan=_tool_plan_dict("plan_xyz"),
    )

    resp = await run_format_evidence(req)
    log = resp.execution_logs[0]
    assert log["plan_id"] == "plan_xyz"
    assert log["calls"] == []


@pytest.mark.asyncio
async def test_deterministic_ids():
    """Same input should produce the same auto-generated IDs."""
    req = _make_request()

    resp1 = await run_format_evidence(req)
    resp2 = await run_format_evidence(req)

    b1 = resp1.evidence_bundles[0]
    b2 = resp2.evidence_bundles[0]

    assert b1["bundle_id"] == b2["bundle_id"]
    assert b1["plan_id"] == b2["plan_id"]
    assert b1["items"][0]["evidence_id"] == b2["items"][0]["evidence_id"]
