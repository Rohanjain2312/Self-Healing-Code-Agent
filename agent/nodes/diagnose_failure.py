"""
Debugger node — structured root cause analysis.

The debugger sees the failing code, test output, and iteration history.
It returns a structured diagnosis: root_cause, failure_category, and
a concrete repair_strategy that guides the generator's next attempt.

The debugger does NOT write code. It only reasons about failures.
"""

import logging
from typing import Any

from agent.state import AgentState
from agent.events import step_event, diagnosis_event
from llm.router import LLMRouter

logger = logging.getLogger(__name__)

_MAX_HISTORY_ENTRIES = 3  # limit how much history the debugger sees


async def diagnose_failure(
    state: AgentState,
    router: LLMRouter,
) -> dict[str, Any]:
    """LangGraph node: analyze test failures and prescribe a repair strategy."""
    iteration = state.get("iteration", 0)
    events = list(state.get("events", []))

    events.append(step_event(
        "Analyzing failure root cause...",
        iteration=iteration,
    ).to_dict())

    # Format iteration history for context (limit to avoid token overflow)
    history = state.get("iteration_history", [])
    relevant_history = history[-_MAX_HISTORY_ENTRIES:] if history else []
    iteration_history_text = _format_history(relevant_history)

    result = await router.call(
        role="debugger",
        template_key="diagnose",
        variables={
            "task_description": state["task_description"],
            "code": state["current_code"],
            "test_results": state.get("last_failure_summary", "No failure details."),
            "iteration_history": iteration_history_text,
        },
        max_new_tokens=768,
    )

    root_cause = result["root_cause"]
    failure_category = result["failure_category"]
    repair_strategy = result["repair_strategy"]
    confidence = result.get("confidence", 0.5)

    logger.info(
        "Diagnosis: category=%s confidence=%.2f (iteration=%d)",
        failure_category,
        confidence,
        iteration,
    )

    events.append(diagnosis_event(
        root_cause=root_cause,
        category=failure_category,
        strategy=repair_strategy,
        iteration=iteration,
    ).to_dict())

    return {
        "root_cause": root_cause,
        "failure_category": failure_category,
        "repair_strategy": repair_strategy,
        "events": events,
    }


def _format_history(history: list[dict]) -> str:
    if not history:
        return "No prior iteration history."
    lines = []
    for record in history:
        lines.append(
            f"Iteration {record['iteration']}: "
            f"passed={record['passed']} | "
            f"category={record.get('failure_category', 'unknown')} | "
            f"summary={record.get('failure_summary', '')[:200]}"
        )
    return "\n".join(lines)
