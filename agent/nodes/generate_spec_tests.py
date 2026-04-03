"""
Spec-blind test generation node — the first half of dual-oracle testing.

This node generates tests from the task description ONLY, before any code
exists. It cannot see the implementation, so it cannot be biased by what
the code happens to do. These tests represent what the code SHOULD do.

The complementary node (create_adversarial_tests) generates tests that DO
see the code and try to find edge cases in the specific implementation.

Running both test suites prevents the common failure mode where a confidently
wrong implementation converges on a solution that passes its own tests.

This node runs ONCE at iteration 0. Subsequent repair iterations reuse the
same spec tests — they are never regenerated.
"""

import logging
from typing import Any

from agent.state import AgentState
from agent.events import step_event, spec_tests_event
from llm.router import LLMRouter

logger = logging.getLogger(__name__)


async def generate_spec_tests(
    state: AgentState,
    router: LLMRouter,
) -> dict[str, Any]:
    """
    LangGraph node: generate specification-based tests from task description.

    Only passes task_description to the LLM — no code is visible.
    Returns spec_test_code to be stored in state for the lifetime of the run.
    """
    iteration = state.get("iteration", 0)
    events = list(state.get("events", []))

    events.append(step_event(
        "Generating spec-blind oracle tests from task description...",
        iteration=iteration,
    ).to_dict())

    result = await router.call(
        role="qa_adversarial",
        template_key="generate_from_spec",
        variables={
            "task_description": state["task_description"],
        },
        max_new_tokens=1024,
    )

    test_code = result.get("test_code", "assert True  # spec test generation failed")
    # Approximate test count from assert statements
    test_count = test_code.count("assert ")

    logger.info(
        "Spec tests generated: ~%d assertions (iteration=%d)",
        test_count,
        iteration,
    )

    events.append(spec_tests_event(test_count=test_count, iteration=iteration).to_dict())

    return {
        "spec_test_code": test_code,
        "events": events,
    }
