"""
Generator node — produces initial and repaired code solutions.

Uses template 'initial' on first iteration, 'repair' on subsequent ones.
Diagnosis fields are injected on repair cycles so the model targets
the specific root cause rather than rewriting from scratch.
"""

import logging
from typing import Any

from agent.state import AgentState
from agent.events import (
    step_event,
    code_generated_event,
)
from llm.router import LLMRouter

logger = logging.getLogger(__name__)


async def generate_solution(
    state: AgentState,
    router: LLMRouter,
) -> dict[str, Any]:
    """
    LangGraph node: generate or repair code.

    On iteration 0: calls 'initial' template with task + learning log.
    On iteration N>0: calls 'repair' template with full diagnosis context.
    """
    iteration = state.get("iteration", 0)
    events = list(state.get("events", []))

    events.append(step_event(
        f"{'Generating initial solution' if iteration == 0 else 'Applying repair'}...",
        iteration=iteration,
    ).to_dict())

    is_repair = (
        iteration > 0
        and state.get("current_code", "")
        and state.get("root_cause", "")
    )

    learning_log = _format_learning_log(state.get("learning_log", []))

    if is_repair:
        template_key = "repair"
        variables = {
            "task_description": state["task_description"],
            "current_code": state["current_code"],
            "test_results": state.get("last_failure_summary", "No failure details."),
            "root_cause": state.get("root_cause", "Unknown"),
            "repair_strategy": state.get("repair_strategy", "No strategy available."),
            "learning_log": learning_log,
        }
    else:
        template_key = "initial"
        variables = {
            "task_description": state["task_description"],
            "learning_log": learning_log,
        }

    result = await router.call(
        role="generator",
        template_key=template_key,
        variables=variables,
        max_new_tokens=2048,
    )

    code = result["code"]
    explanation = result.get("explanation", "")

    logger.info("Generator produced %d chars of code (iteration=%d)", len(code), iteration)

    events.append(code_generated_event(
        code=code,
        iteration=iteration,
        explanation=explanation,
    ).to_dict())

    return {
        "current_code": code,
        "events": events,
    }


def _format_learning_log(lessons: list[str]) -> str:
    if not lessons:
        return "No prior lessons recorded."
    return "\n".join(f"- {lesson}" for lesson in lessons)
