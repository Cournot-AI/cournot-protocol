"""Format Evidence Route

POST /format_evidence — Format externally-collected evidence into an EvidenceBundle.

Accepts pre-collected evidence items (e.g., from a cron job monitoring sports APIs)
and packages them into the same CollectResponse shape that /step/collect returns,
so the result can be passed directly to /step/audit.

No LLM calls, HTTP fetches, or agent context required.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any, Literal
from urllib.parse import urlparse

from fastapi import APIRouter
from pydantic import BaseModel, Field

from api.errors import InternalError

logger = logging.getLogger(__name__)

router = APIRouter(tags=["format_evidence"])


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class ExternalEvidenceItem(BaseModel):
    """A single evidence item supplied by an external caller."""

    source_url: str = Field(..., description="URL/URI where evidence was obtained", min_length=1)
    content: str = Field(..., description="Evidence text/JSON", min_length=1)
    parsed_value: Any = Field(default=None, description="Structured value extracted from content")
    extracted_fields: dict[str, Any] = Field(
        default_factory=dict, description="Key-value extracted data"
    )
    tier: Literal[0, 1, 2, 3, 4] = Field(default=1, description="Provenance tier (0-4)")
    requirement_id: str | None = Field(
        default=None, description="DataRequirement ID (auto-generated as external_<idx> if omitted)"
    )
    source_id: str | None = Field(
        default=None, description="Source name (auto-generated from URL domain if omitted)"
    )
    evidence_id: str | None = Field(
        default=None, description="Evidence ID (auto-generated from hash if omitted)"
    )


class FormatEvidenceRequest(BaseModel):
    """Request to format externally-collected evidence into an EvidenceBundle."""

    prompt_spec: dict[str, Any] = Field(
        ..., description="Compiled prompt specification (from /step/prompt output)"
    )
    tool_plan: dict[str, Any] | None = Field(
        default=None,
        description="Tool execution plan (from /step/prompt output); synthetic if omitted",
    )
    items: list[ExternalEvidenceItem] = Field(
        ..., description="Evidence items to format", min_length=1
    )
    collector_name: str = Field(
        default="ExternalEvidence", description="Collector name tag"
    )
    bundle_id: str | None = Field(
        default=None, description="Explicit bundle ID (auto-generated if omitted)"
    )
    include_raw_content: bool = Field(
        default=False,
        description="Keep raw_content/parsed_value in output (default strips them)",
    )


class FormatEvidenceResponse(BaseModel):
    """Response from format evidence endpoint."""

    ok: bool = Field(..., description="Whether formatting succeeded")
    collectors_used: list[str] = Field(default_factory=list, description="Names of collectors used")
    evidence_bundles: list[dict[str, Any]] = Field(
        default_factory=list, description="Evidence bundles from each collector"
    )
    execution_logs: list[dict[str, Any]] = Field(
        default_factory=list, description="Execution logs from each collector"
    )
    errors: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------


@router.post("/format_evidence", response_model=FormatEvidenceResponse)
async def run_format_evidence(request: FormatEvidenceRequest) -> FormatEvidenceResponse:
    """
    Format externally-collected evidence into an EvidenceBundle.

    Accepts pre-collected evidence items and packages them into the same
    shape that /step/collect returns, so the result can be passed directly
    to /step/audit.

    No LLM calls, HTTP fetches, or agent context required.
    """
    try:
        from core.schemas.prompts import PromptSpec
        from core.schemas.evidence import EvidenceBundle, EvidenceItem, Provenance
        from core.schemas.transport import ToolExecutionLog

        # 1. Parse prompt_spec -> extract market_id
        try:
            prompt_spec = PromptSpec(**request.prompt_spec)
        except Exception as e:
            return FormatEvidenceResponse(ok=False, errors=[f"Invalid prompt_spec: {e}"])

        market_id = prompt_spec.market_id

        # 2. Parse tool_plan -> extract plan_id, or generate synthetic
        if request.tool_plan is not None:
            plan_id = request.tool_plan.get("plan_id")
            if not plan_id:
                return FormatEvidenceResponse(
                    ok=False, errors=["tool_plan provided but missing plan_id"]
                )
        else:
            plan_id = "ext_" + hashlib.sha256(market_id.encode()).hexdigest()[:12]

        # 3. Generate bundle_id if not provided
        if request.bundle_id:
            bundle_id = request.bundle_id
        else:
            bundle_seed = f"{market_id}:{plan_id}:{request.collector_name}"
            bundle_id = "eb_" + hashlib.sha256(bundle_seed.encode()).hexdigest()[:16]

        # 4. Build EvidenceItems
        evidence_items: list[EvidenceItem] = []
        for idx, ext_item in enumerate(request.items):
            # Derive source_id from URL domain if missing
            source_id = ext_item.source_id
            if not source_id:
                try:
                    source_id = urlparse(ext_item.source_url).netloc or ext_item.source_url
                except Exception:
                    source_id = ext_item.source_url

            # Derive requirement_id if missing
            requirement_id = ext_item.requirement_id or f"external_{idx}"

            # Derive evidence_id from deterministic hash if missing
            evidence_id = ext_item.evidence_id
            if not evidence_id:
                eid_seed = f"{bundle_id}:{idx}:{ext_item.source_url}"
                evidence_id = "ev_" + hashlib.sha256(eid_seed.encode()).hexdigest()[:16]

            # Compute content_hash (0x + sha256, matching adapter convention)
            content_hash = "0x" + hashlib.sha256(ext_item.content.encode()).hexdigest()

            provenance = Provenance(
                source_id=source_id,
                source_uri=ext_item.source_url,
                tier=ext_item.tier,
                content_hash=content_hash,
            )

            evidence_items.append(
                EvidenceItem(
                    evidence_id=evidence_id,
                    requirement_id=requirement_id,
                    provenance=provenance,
                    raw_content=ext_item.content,
                    parsed_value=ext_item.parsed_value,
                    extracted_fields=ext_item.extracted_fields,
                    success=True,
                )
            )

        # 5. Cross-reference requirement_ids against prompt_spec.data_requirements
        all_req_ids = {r.requirement_id for r in prompt_spec.data_requirements}
        fulfilled_ids = {item.requirement_id for item in evidence_items}
        requirements_fulfilled = sorted(all_req_ids & fulfilled_ids)
        requirements_unfulfilled = sorted(all_req_ids - fulfilled_ids)

        # 6. Build EvidenceBundle and validate schema compliance
        evidence_bundle = EvidenceBundle(
            bundle_id=bundle_id,
            market_id=market_id,
            plan_id=plan_id,
            collector_name=request.collector_name,
            items=evidence_items,
            total_sources_attempted=len(request.items),
            total_sources_succeeded=len(evidence_items),
            requirements_fulfilled=requirements_fulfilled,
            requirements_unfulfilled=requirements_unfulfilled,
        )
        evidence_bundle = EvidenceBundle.model_validate(
            evidence_bundle.model_dump(mode="json")
        )

        # 7. Strip raw_content / parsed_value if include_raw_content=False
        eb_dict = evidence_bundle.model_dump(mode="json")
        if not request.include_raw_content:
            for item in eb_dict.get("items", []):
                item["raw_content"] = None
                item["parsed_value"] = None

        # 8. Build minimal ToolExecutionLog
        exec_log = ToolExecutionLog(plan_id=plan_id, calls=[])

        return FormatEvidenceResponse(
            ok=True,
            collectors_used=[request.collector_name],
            evidence_bundles=[eb_dict],
            execution_logs=[exec_log.model_dump(mode="json")],
        )

    except Exception as e:
        logger.exception("Format evidence step failed")
        raise InternalError(f"Format evidence step failed: {str(e)}")
