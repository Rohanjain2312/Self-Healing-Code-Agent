"""
QA Adversarial node — generates test cases designed to break the solution.

The QA agent acts as a hostile reviewer: it sees the task description and
the generated code, and produces adversarial assert-based tests.

Tests are returned as a string of Python source (not executed here).
Execution happens in the execute_solution node.
"""

import logging
from typing import Any

from agent.state import AgentState
from agent.events import step_event, AgentEvent, TESTS_GENERATED
from llm.router import LLMRouter

logger = logging.getLogger(__name__)


async def create_adversarial_tests(
    state: AgentState,
    router: LLMRouter,
) -> dict[str, Any]:
    """LangGraph node: generate adversarial test cases for current code."""
    iteration = state.get("iteration", 0)
    events = list(state.get("events", []))

    events.append(step_event(
        "Generating adversarial test suite...",
        iteration=iteration,
    ).to_dict())

    result = await router.call(
        role="qa_adversarial",
        template_key="generate",
        variables={
            "task_description": state["task_description"],
            "code": state["current_code"],
        },
        max_new_tokens=768,
    )

    test_code = result["test_code"]
    descriptions = result.get("test_cases_description", [])

    logger.info(
        "QA generated %d test cases (iteration=%d)",
        len(descriptions),
        iteration,
    )

    events.append(AgentEvent(
        type=TESTS_GENERATED,
        message=f"Generated {len(descriptions)} adversarial tests",
        iteration=iteration,
        payload={
            "test_cases_description": descriptions,
            "test_count": len(descriptions),
        },
    ).to_dict())

    return {
        "current_test_code": test_code,
        "events": events,
    }
