"""
Streaming adapter — converts agent event streams into UI-consumable formats.

Provides:
  - async generator that yields formatted event dicts for Gradio
  - timeline formatter that converts events to human-readable log entries
  - code snapshot extractor for live code display
  - learning log extractor for live lesson display

This layer is stateless — it transforms event streams without buffering state.
The UI layer owns all display state.
"""

import asyncio
import json
import logging
from typing import Any, AsyncGenerator

from agent.events import (
    CODE_GENERATED,
    FAILURE,
    LEARNING_UPDATE,
    SUCCESS,
    STEP,
    DIAGNOSIS,
    TESTS_GENERATED,
    ITERATION_START,
    TIMEOUT,
)

logger = logging.getLogger(__name__)

# Events that are shown in the public timeline (others are internal)
# Public set — importable by demo and test modules
PUBLIC_EVENT_TYPES = {
    STEP,
    CODE_GENERATED,
    FAILURE,
    LEARNING_UPDATE,
    SUCCESS,
    DIAGNOSIS,
    TESTS_GENERATED,
}


def format_event_for_timeline(event: dict[str, Any]) -> str:
    """
    Convert a single event dict to a human-readable timeline entry.

    Does NOT expose internal reasoning or prompts.
    Only structured decisions and observable outcomes are shown.
    """
    event_type = event.get("type", "unknown")
    message = event.get("message", "")
    iteration = event.get("iteration", 0)
    payload = event.get("payload", {})

    prefix = f"[Iteration {iteration}]"

    if event_type == STEP:
        return f"{prefix} {message}"

    if event_type == CODE_GENERATED:
        explanation = payload.get("explanation", "")
        code_preview = payload.get("code", "")[:80].replace("\n", " ")
        line = f"{prefix} Code generated."
        if explanation:
            line += f" Approach: {explanation}"
        return line

    if event_type == TESTS_GENERATED:
        count = payload.get("test_count", "?")
        return f"{prefix} {count} adversarial tests generated."

    if event_type == FAILURE:
        assertions = payload.get("failed_assertions", [])
        if assertions:
            first = assertions[0][:120]
            return f"{prefix} FAIL — {first}"
        summary = payload.get("summary", "")[:120]
        return f"{prefix} FAIL — {summary}"

    if event_type == DIAGNOSIS:
        category = payload.get("failure_category", "unknown")
        root_cause = payload.get("root_cause", "")[:120]
        return f"{prefix} Diagnosis: [{category}] {root_cause}"

    if event_type == LEARNING_UPDATE:
        count = len(payload.get("lessons", []))
        return f"{prefix} Learning log updated ({count} lessons retained)."

    if event_type == SUCCESS:
        return f"{prefix} SUCCESS — all tests passed."

    return f"{prefix} {message}"


async def stream_events_for_ui(
    event_stream: AsyncGenerator[dict[str, Any], None],
    include_internal: bool = False,
) -> AsyncGenerator[dict[str, Any], None]:
    """
    Filter and enrich events for UI consumption.

    Strips events that are not meant for public display.
    Adds a formatted 'display_text' field for simple rendering.
    """
    async for event in event_stream:
        if event is None:
            break

        event_type = event.get("type", "")
        if not include_internal and event_type not in PUBLIC_EVENT_TYPES:
            continue

        enriched = {
            **event,
            "display_text": format_event_for_timeline(event),
        }
        yield enriched


def extract_latest_code(events: list[dict[str, Any]]) -> str:
    """Return the most recently generated code from a list of events."""
    for event in reversed(events):
        if event.get("type") == CODE_GENERATED:
            return event.get("payload", {}).get("code", "")
    return ""


def extract_learning_log(events: list[dict[str, Any]]) -> list[str]:
    """Return the most recent set of lessons from a list of events."""
    for event in reversed(events):
        if event.get("type") == LEARNING_UPDATE:
            return event.get("payload", {}).get("lessons", [])
    return []


def build_timeline_text(events: list[dict[str, Any]]) -> str:
    """Convert a full event list to a multi-line timeline string for display."""
    lines = []
    for event in events:
        if event.get("type") in PUBLIC_EVENT_TYPES:
            lines.append(format_event_for_timeline(event))
    return "\n".join(lines)
