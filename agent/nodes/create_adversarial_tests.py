"""
QA Adversarial node — generates hostile test cases designed to break the solution.

ROLE IN THE GRAPH
-----------------
create_adversarial_tests runs after the fan-in join of generate_spec_tests and
generate_solution. It receives the COMPLETED solution code (current_code) and
writes an adversarial test suite (current_test_code) that tries to break it.

WHY ADVERSARIAL TESTS?
-----------------------
A good QA agent doesn't write tests that the code is likely to pass — it writes
tests designed to expose weaknesses. Specifically it targets:
  - Empty / None inputs
  - Boundary values (0, 1, -1, MAX_INT, empty strings, single-element lists)
  - Type mismatches (passing a string where int expected)
  - Ordering/sorting edge cases
  - Duplicate values
  - Very large or very small numbers

The test code produced here is a Python string of ``assert`` statements. It is
NOT executed in this node — execution happens in execute_solution, which runs
both this test suite and spec_test_code in the sandbox.

DIFFERENCE FROM SPEC TESTS
---------------------------
  - spec_test_code (generate_spec_tests.py): written BEFORE code exists,
    purely from the task description. Oracle tests — the generator never
    sees them.
  - current_test_code (this node): written AFTER seeing the code. Can target
    specific implementation choices the generator made. Regenerated on every
    repair iteration (because the code changes).

Both suites must pass for execute_solution to set last_execution_passed=True.
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
    """LangGraph node: generate adversarial tests for the current solution.

    Args:
        state:  Current AgentState. Reads: iteration, task_description,
                current_code (the solution to attack).
        router: LLMRouter bound at build_graph time. Uses the ``qa_adversarial``
                role → Claude (Haiku 4.5).

    Returns:
        Partial state: {"current_test_code": ..., "events": ...}
    """
    iteration = state.get("iteration", 0)
    events = list(state.get("events", []))

    events.append(step_event(
        "Generating adversarial test suite...",
        iteration=iteration,
    ).to_dict())

    # The QA prompt sees both the task description (to understand intent) and
    # the actual code (to target its specific weaknesses). This is the key
    # difference from spec tests: here the QA has full visibility into how
    # the code was implemented and can exploit it.
    result = await router.call(
        role="qa_adversarial",
        template_key="generate",
        variables={
            "task_description": state["task_description"],
            "code": state["current_code"],
        },
        max_new_tokens=1024,  # test suites are shorter than solution code
    )

    test_code = result["test_code"]
    # test_cases_description is a human-readable list of what each test does;
    # used for logging and for the timeline display.
    descriptions = result.get("test_cases_description", [])

    logger.info(
        "QA generated %d test cases (iteration=%d)",
        len(descriptions),
        iteration,
    )

    # Emit a TESTS_GENERATED event — the Gradio timeline shows what kinds of
    # edge cases the QA agent decided to test.
    events.append(AgentEvent(
        type=TESTS_GENERATED,
        message=f"Generated {len(descriptions)} adversarial tests",
        iteration=iteration,
        payload={
            "test_cases_description": descriptions,
            "test_count": len(descriptions),
        },
    ).to_dict())

    # Only update current_test_code — spec_test_code is untouched (it was set
    # once by generate_spec_tests and never changes after that).
    return {
        "current_test_code": test_code,
        "events": events,
    }
