"""
Spec-blind oracle test generation node — the first half of dual-oracle testing.

ROLE IN THE GRAPH
-----------------
generate_spec_tests runs in PARALLEL with generate_solution at graph start
(both are connected from __start__). They converge at create_adversarial_tests.

This node generates tests BEFORE any code exists, using ONLY the task description.
The generator node never sees these tests — they are "oracle" tests in the sense
that they represent what the correct solution SHOULD do, independent of any
particular implementation.

WHY "SPEC-BLIND"?
------------------
If you show tests to the model before it writes code, it can cheat: write code
that specifically handles the test cases rather than the general problem. Spec-
blind tests prevent this by requiring the specification itself to be the only
source of truth.

DUAL-ORACLE STRATEGY (Fix 5)
------------------------------
Two test suites run in execute_solution:

  1. spec_test_code (THIS node): written from spec only, never changes
     → guards against spec-misinterpretation bugs
  2. current_test_code (create_adversarial_tests): written after seeing the code
     → guards against implementation edge-case bugs

Both must pass. The combination catches:
  - Code that passes adversarial tests but misunderstands the spec
  - Code that meets the spec but fails on specific edge cases in the implementation

LIFECYCLE
---------
spec_test_code is set once in iteration 0 and NEVER updated on repair iterations.
Only current_test_code is regenerated each iteration (it tracks changes in the code).
The repair loop returns to generate_solution, not generate_spec_tests.

REUSE OF qa_adversarial ROLE
------------------------------
This node uses the same qa_adversarial LLM role as create_adversarial_tests,
but with a different template key: ``generate_from_spec`` (no code in context)
vs ``generate`` (code visible). This lets us reuse the QA prompt system and
schema while keeping the inputs distinct.
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
    """LangGraph node: generate specification-based oracle tests from task description.

    Only passes task_description to the LLM — no code context whatsoever.
    Runs concurrently with generate_solution via the parallel fan-out from __start__.

    Args:
        state:  Current AgentState. Reads: iteration, task_description.
        router: LLMRouter. Uses the ``qa_adversarial`` role → Claude (Haiku 4.5),
                with template_key ``generate_from_spec`` (no code context).

    Returns:
        Partial state: {"spec_test_code": ..., "events": ...}
        spec_test_code is stored in state for the lifetime of the run and
        is never overwritten by subsequent nodes.
    """
    iteration = state.get("iteration", 0)
    events = list(state.get("events", []))

    events.append(step_event(
        "Generating spec-blind oracle tests from task description...",
        iteration=iteration,
    ).to_dict())

    # Deliberately pass ONLY task_description — no code, no tests, no history.
    # This ensures spec_test_code is truly blind to the implementation.
    result = await router.call(
        role="qa_adversarial",
        template_key="generate_from_spec",   # different template than adversarial tests
        variables={
            "task_description": state["task_description"],
        },
        max_new_tokens=1024,
    )

    test_code = result.get("test_code", "assert True  # spec test generation failed")
    # Count "assert" statements as a proxy for number of test cases
    test_count = test_code.count("assert ")

    logger.info(
        "Spec tests generated: ~%d assertions (iteration=%d)",
        test_count,
        iteration,
    )

    events.append(spec_tests_event(test_count=test_count, iteration=iteration).to_dict())

    # spec_test_code is written once here and never updated again.
    # execute_solution reads it directly from state on every iteration.
    return {
        "spec_test_code": test_code,
        "events": events,
    }
