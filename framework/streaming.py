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
    SPEC_TESTS_GENERATED,
    REPAIR_REVIEW,
    TOOL_USE,
    CRITIC_REVIEW,
    PARALLEL_REPAIR,
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
    SPEC_TESTS_GENERATED,
    REPAIR_REVIEW,
    TOOL_USE,
    CRITIC_REVIEW,
    PARALLEL_REPAIR,
}


def _trunc(s: str, n: int) -> str:
    """Return s truncated to n chars with ellipsis if needed."""
    return s if len(s) <= n else s[:n - 1] + "\u2026"


def format_event_for_timeline(event: dict[str, Any]) -> str:
    """
    Convert a single event dict to a concise human-readable timeline entry.

    Returns "" for event types that should be suppressed (STEP, unknown).
    All lines are capped at 100 characters.
    """
    event_type = event.get("type", "unknown")
    iteration = event.get("iteration", 0)
    payload = event.get("payload", {})
    if not isinstance(payload, dict):
        payload = {}

    n = iteration  # short alias for f-strings

    if event_type == STEP:
        return ""  # too noisy — suppress

    if event_type == CODE_GENERATED:
        code = payload.get("code", "")
        return _trunc(f"[iter {n}] \u2713 Code generated ({len(code)} chars)", 100)

    if event_type == TESTS_GENERATED:
        count = payload.get("test_count", "?")
        return _trunc(f"[iter {n}] \u2713 {count} adversarial tests ready", 100)

    if event_type == SPEC_TESTS_GENERATED:
        count = payload.get("test_count", "?")
        return _trunc(f"[iter {n}] \u2713 {count} spec tests ready", 100)

    if event_type == FAILURE:
        assertions = payload.get("failed_assertions", [])
        summary = assertions[0] if assertions else payload.get("summary", "")
        return _trunc(f"[iter {n}] \u2717 {_trunc(summary, 80)}", 100)

    if event_type == DIAGNOSIS:
        category = payload.get("failure_category", "unknown")
        root_cause = payload.get("root_cause", "")
        return _trunc(f"[iter {n}] \u2192 [{category}] {_trunc(root_cause, 80)}", 100)

    if event_type == LEARNING_UPDATE:
        count = len(payload.get("lessons", []))
        return _trunc(f"[iter {n}] \U0001f4dd {count} lesson(s) logged", 100)

    if event_type == SUCCESS:
        return f"[iter {n}] \u2713 All tests passed"

    if event_type == TOOL_USE:
        tool_name = payload.get("tool_name", "unknown")
        result = payload.get("result", "").replace("\n", " ")
        return _trunc(f"[iter {n}] \U0001f527 {tool_name}: {_trunc(result, 60)}", 100)

    if event_type == CRITIC_REVIEW:
        verdict = payload.get("verdict", "unknown").upper()
        confidence = payload.get("confidence", 0.0)
        issues = payload.get("issues", [])
        line = f"[iter {n}] \U0001f50d Critic: {verdict} ({confidence:.0%})"
        if issues:
            line += f" \u2014 {_trunc(issues[0], 50)}"
        return _trunc(line, 100)

    if event_type == PARALLEL_REPAIR:
        strategy = payload.get("strategy_name", "unknown")
        spec = payload.get("spec_passed", False)
        adv = payload.get("adv_passed", False)
        return _trunc(f"[iter {n}] \u26a1 [{strategy}] spec={spec} adv={adv}", 100)

    if event_type == REPAIR_REVIEW:
        category = payload.get("failure_category", "?")
        confidence = payload.get("confidence", 0.0)
        return _trunc(
            f"[iter {n}] \u23f8  Human review \u2014 [{category}] confidence {confidence:.0%}",
            100,
        )

    return ""  # suppress unknown event types


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
    lines = [
        format_event_for_timeline(e)
        for e in events
        if e.get("type") in PUBLIC_EVENT_TYPES
    ]
    return "\n".join(line for line in lines if line)
