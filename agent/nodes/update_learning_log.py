"""
Rolling memory summarizer node.

Compresses the iteration's diagnosis + outcome into at most 5 bullet lessons.
Lessons are generalizable patterns, not iteration-specific facts.
This prevents the context from growing unboundedly across repair cycles.

The node runs AFTER diagnosis but BEFORE the repair, so lessons from the
current failure are incorporated before the generator is called again.
"""

import logging
from typing import Any

from agent.state import AgentState
from agent.events import step_event, learning_update_event
from llm.router import LLMRouter

logger = logging.getLogger(__name__)


async def update_learning_log(
    state: AgentState,
    router: LLMRouter,
) -> dict[str, Any]:
    """LangGraph node: compress iteration evidence into rolling learning log."""
    iteration = state.get("iteration", 0)
    events = list(state.get("events", []))

    events.append(step_event(
        "Updating rolling learning log...",
        iteration=iteration,
    ).to_dict())

    prior_lessons = state.get("learning_log", [])

    # Only call LLM if we have meaningful failure context to learn from
    root_cause = state.get("root_cause", "")
    if not root_cause:
        return {"events": events}

    # Determine the outcome of this iteration for the summarizer
    last_passed = state.get("last_execution_passed", False)
    outcome = "Test passed after repair." if last_passed else "Test still failing after repair attempt."

    result = await router.call(
        role="memory_summarizer",
        template_key="summarize",
        variables={
            "prior_lessons": _format_lessons(prior_lessons),
            "root_cause": root_cause,
            "failure_category": state.get("failure_category", "unknown"),
            "repair_strategy": state.get("repair_strategy", "None"),
            "outcome": outcome,
        },
        max_new_tokens=256,
    )

    lessons = result.get("lessons", prior_lessons)
    # Hard cap at 5 lessons — schema validates this but enforce defensively
    lessons = lessons[:5]

    logger.info("Learning log updated: %d lessons", len(lessons))

    events.append(learning_update_event(
        lessons=lessons,
        iteration=iteration,
    ).to_dict())

    return {
        "learning_log": lessons,
        "events": events,
    }


def _format_lessons(lessons: list[str]) -> str:
    if not lessons:
        return "No prior lessons."
    return "\n".join(f"- {l}" for l in lessons)
