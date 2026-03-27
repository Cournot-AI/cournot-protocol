"""
Shared Temporal Constraint Logic

Resolves temporal status and builds temporal advisory prompts for both
the auditor and judge agents. Supports two modes:

- **event**: single `event_time` deadline (e.g. "Will Liverpool beat Brighton?")
- **range**: `[start_time, end_time]` window (e.g. "Will Iran close Strait of Hormuz in Q2 2026?")
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from core.schemas import PromptSpec


# ---------------------------------------------------------------------------
# Status resolution
# ---------------------------------------------------------------------------

def resolve_temporal_status(temporal_ctx: dict[str, Any], current_time: datetime) -> str:
    """Dispatch to event or range resolver based on ``mode`` field.

    Missing ``mode`` defaults to ``"event"`` for backwards compatibility.
    """
    mode = temporal_ctx.get("mode", "event")
    if mode == "range":
        return _resolve_range_status(temporal_ctx, current_time)
    return _resolve_event_status(temporal_ctx.get("event_time", ""), current_time)


def _resolve_event_status(event_time_raw: str, current_time: datetime) -> str:
    """Compute DEADLINE_OPEN/DEADLINE_RECENT/DEADLINE_PASSED from event_time vs current_time.

    event_time represents the **deadline** by which the event must occur, not
    necessarily when the event happens.  The event can occur at any time before
    the deadline.

    - DEADLINE_OPEN: deadline is still in the future — the event may have
      already happened (allow YES) but cannot be ruled out (block NO).
    - DEADLINE_RECENT: deadline passed within the last 24 hours — evidence of
      the final state may still be emerging.
    - DEADLINE_PASSED: deadline passed more than 24 hours ago — resolve normally.
    """
    try:
        event_dt = datetime.fromisoformat(event_time_raw.replace("Z", "+00:00"))
        if event_dt.tzinfo is None:
            event_dt = event_dt.replace(tzinfo=timezone.utc)
    except (ValueError, AttributeError):
        return "UNKNOWN"

    if current_time.tzinfo is None:
        current_time = current_time.replace(tzinfo=timezone.utc)

    if event_dt > current_time:
        return "DEADLINE_OPEN"
    elif current_time - event_dt < timedelta(hours=24):
        return "DEADLINE_RECENT"
    else:
        return "DEADLINE_PASSED"


def _resolve_range_status(temporal_ctx: dict[str, Any], current_time: datetime) -> str:
    """Compute range-mode status from start_time/end_time vs current_time.

    - BEFORE_WINDOW: now < start_time — too early, INVALID.
    - WINDOW_OPEN: start_time <= now <= end_time — YES if evidenced,
      NO blocked, default INVALID.
    - WINDOW_CLOSING: now - end_time < 24h — YES/NO if clear, else INVALID.
    - WINDOW_CLOSED: now - end_time >= 24h — evaluate normally.
    """
    start_raw = temporal_ctx.get("start_time", "")
    end_raw = temporal_ctx.get("end_time", "")

    try:
        start_dt = datetime.fromisoformat(start_raw.replace("Z", "+00:00"))
        if start_dt.tzinfo is None:
            start_dt = start_dt.replace(tzinfo=timezone.utc)
        end_dt = datetime.fromisoformat(end_raw.replace("Z", "+00:00"))
        if end_dt.tzinfo is None:
            end_dt = end_dt.replace(tzinfo=timezone.utc)
    except (ValueError, AttributeError):
        return "UNKNOWN"

    if current_time.tzinfo is None:
        current_time = current_time.replace(tzinfo=timezone.utc)

    if current_time < start_dt:
        return "BEFORE_WINDOW"
    elif current_time <= end_dt:
        return "WINDOW_OPEN"
    elif current_time - end_dt < timedelta(hours=24):
        return "WINDOW_CLOSING"
    else:
        return "WINDOW_CLOSED"


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def build_temporal_context_prompt(
    temporal_ctx: dict[str, Any],
    current_time: datetime,
    prompt_spec: "PromptSpec | None" = None,
    *,
    role: str = "auditor",
) -> str:
    """Build a TEMPORAL ADVISORY block for LLM messages.

    Dispatches on ``mode`` (default ``"event"``).  The ``role`` parameter
    controls minor wording differences between auditor and judge advisories.
    """
    mode = temporal_ctx.get("mode", "event")
    if mode == "range":
        return _build_range_prompt(temporal_ctx, current_time, prompt_spec, role=role)
    return _build_event_prompt(temporal_ctx, current_time, prompt_spec, role=role)


def _build_event_prompt(
    temporal_ctx: dict[str, Any],
    current_time: datetime,
    prompt_spec: "PromptSpec | None" = None,
    *,
    role: str = "auditor",
) -> str:
    """Build the event-mode temporal advisory (existing logic)."""
    event_time_raw = temporal_ctx.get("event_time", "")
    reason = temporal_ctx.get("reason", "No reason provided")

    status = _resolve_event_status(event_time_raw, current_time)

    is_multi = prompt_spec is not None and prompt_spec.is_multi_choice
    if is_multi:
        outcomes_label = ", ".join(prompt_spec.possible_outcomes)
        affirm = f"the matching outcome (one of: {outcomes_label})"
        negative = "rule out other outcomes"
    else:
        affirm = "YES"
        negative = "NO"

    parts: list[str] = [
        "## TEMPORAL ADVISORY",
        "",
        f"Temporal status: {status}",
        f"Resolution deadline: {event_time_raw}",
        f"Current time: {current_time.isoformat()}",
        f"Reason: {reason}",
        "",
    ]

    if status == "DEADLINE_OPEN":
        parts.extend([
            "INSTRUCTIONS:",
            "The resolution deadline has NOT passed yet. The event may occur at",
            "any time before the deadline — it does NOT have to happen at the",
            "exact deadline time.",
            "",
            "1) If evidence shows the event ALREADY HAPPENED before the deadline,",
            f"   you CAN return {affirm} with appropriate confidence.",
            f"2) You MUST NOT return {negative} — the event could still happen before the",
            "   deadline. Absence of evidence is not evidence of absence.",
            "3) If no evidence confirms the event has occurred yet, return INVALID",
            "   with low confidence (0.2-0.3) and note the deadline has not passed.",
        ])
        if role == "judge":
            parts.append(
                f"4) If the Auditor returned {negative}, override to INVALID — we cannot rule"
            )
            parts.append(
                "   out the event occurring before the deadline."
            )
        else:
            parts.append(
                f"4) Summary: {affirm} is allowed if evidenced. {negative} is blocked. INVALID is the"
            )
            parts.append(
                "   default when uncertain."
            )
    elif status == "DEADLINE_RECENT":
        parts.extend([
            "INSTRUCTIONS:",
            "The resolution deadline passed very recently (within 24 hours).",
            "Evidence of the final state may still be emerging.",
            "",
            "1) If evidence confirms the event occurred before the deadline,",
            f"   return {affirm}.",
            f"2) If evidence confirms the event did NOT occur and the deadline has",
            f"   now passed, you may return {negative}.",
            "3) If evidence is still unclear or emerging, return INVALID.",
        ])
    else:
        if role == "judge":
            parts.extend([
                "INSTRUCTIONS:",
                "1) The deadline has passed. Evaluate the Auditor's reasoning normally.",
            ])
        else:
            parts.extend([
                "INSTRUCTIONS:",
                "1) The deadline has passed. Evaluate evidence normally.",
            ])

    return "\n".join(parts)


def _build_range_prompt(
    temporal_ctx: dict[str, Any],
    current_time: datetime,
    prompt_spec: "PromptSpec | None" = None,
    *,
    role: str = "auditor",
) -> str:
    """Build the range-mode temporal advisory."""
    start_time_raw = temporal_ctx.get("start_time", "")
    end_time_raw = temporal_ctx.get("end_time", "")
    reason = temporal_ctx.get("reason", "No reason provided")

    status = _resolve_range_status(temporal_ctx, current_time)

    is_multi = prompt_spec is not None and prompt_spec.is_multi_choice
    if is_multi:
        outcomes_label = ", ".join(prompt_spec.possible_outcomes)
        affirm = f"the matching outcome (one of: {outcomes_label})"
        negative = "rule out other outcomes"
    else:
        affirm = "YES"
        negative = "NO"

    parts: list[str] = [
        "## TEMPORAL ADVISORY",
        "",
        f"Temporal mode: range",
        f"Temporal status: {status}",
        f"Window start: {start_time_raw}",
        f"Window end: {end_time_raw}",
        f"Current time: {current_time.isoformat()}",
        f"Reason: {reason}",
        "",
    ]

    if status == "BEFORE_WINDOW":
        parts.extend([
            "INSTRUCTIONS:",
            "The observation window has NOT started yet. It is too early to",
            "evaluate this market.",
            "",
            "1) Return INVALID — the window has not opened so no evidence can",
            "   be relevant yet.",
            "2) Do NOT return YES or NO.",
        ])
    elif status == "WINDOW_OPEN":
        parts.extend([
            "INSTRUCTIONS:",
            "The observation window is currently OPEN. The event may occur at",
            "any time before the window closes.",
            "",
            "1) If evidence shows the event ALREADY HAPPENED within the window,",
            f"   you CAN return {affirm} with appropriate confidence.",
            f"2) You MUST NOT return {negative} — the event could still happen before the",
            "   window closes. Absence of evidence is not evidence of absence.",
            "3) If no evidence confirms the event has occurred yet, return INVALID",
            "   with low confidence (0.2-0.3) and note the window is still open.",
        ])
        if role == "judge":
            parts.append(
                f"4) If the Auditor returned {negative}, override to INVALID — we cannot rule"
            )
            parts.append(
                "   out the event occurring before the window closes."
            )
        else:
            parts.append(
                f"4) Summary: {affirm} is allowed if evidenced. {negative} is blocked. INVALID is the"
            )
            parts.append(
                "   default when uncertain."
            )
    elif status == "WINDOW_CLOSING":
        parts.extend([
            "INSTRUCTIONS:",
            "The observation window closed very recently (within 24 hours).",
            "Evidence of the final state may still be emerging.",
            "",
            "1) If evidence confirms the event occurred within the window,",
            f"   return {affirm}.",
            f"2) If evidence confirms the event did NOT occur during the window,",
            f"   you may return {negative}.",
            "3) If evidence is still unclear or emerging, return INVALID.",
        ])
    else:
        # WINDOW_CLOSED
        if role == "judge":
            parts.extend([
                "INSTRUCTIONS:",
                "1) The observation window has closed. Evaluate the Auditor's reasoning normally.",
            ])
        else:
            parts.extend([
                "INSTRUCTIONS:",
                "1) The observation window has closed. Evaluate evidence normally.",
            ])

    return "\n".join(parts)
