"""
Critic node — agent self-reflection after tests pass (Fix 15).

When all tests pass, the agent typically declares success immediately.
This creates a blind spot: the QA-generated tests may not cover all
edge cases specified in the task, and the code can be algorithmically
wrong while passing the generated tests.

The critic examines the passing code against the original specification
and flags issues the test suite missed. If confidence > threshold AND
verdict is "reject", the critic re-opens the repair loop by:
  - Setting last_execution_passed = False
  - Injecting critic feedback as root_cause and repair_strategy

Design decision: use call_with_fallback() rather than call() so that
critic failures never block success. On LLM failure, the fallback
conservatively approves (avoids infinite repair loops from bad critique).
Reject decisions require confidence > config.critic_confidence_threshold
to prevent low-confidence rejections from triggering unnecessary repairs.
"""

import logging
from typing import Any

from agent.state import AgentState
from agent.events import step_event, critic_event
from agent.config import AgentConfig
from llm.router import LLMRouter

logger = logging.getLogger(__name__)


async def critic_review(
    state: AgentState,
    router: LLMRouter,
    agent_config: AgentConfig,
) -> dict[str, Any]:
    """
    LangGraph node: final correctness review of passing solutions.

    Returns empty dict (no state change) on approval.
    On rejection with sufficient confidence, re-opens the repair loop.
    """
    iteration = state.get("iteration", 0)
    events = list(state.get("events", []))

    events.append(step_event(
        "Critic reviewing solution for correctness...",
        iteration=iteration,
    ).to_dict())

    current_code = state.get("current_code", "")
    spec_test_code = state.get("spec_test_code", "")
    adversarial_test_code = state.get("current_test_code", "")

    # Build a summary of what tests were run and passed
    test_summary_parts = []
    if spec_test_code.strip():
        test_summary_parts.append("Spec-blind oracle tests: PASSED")
    if adversarial_test_code.strip():
        test_summary_parts.append("Adversarial edge-case tests: PASSED")
    if not test_summary_parts:
        test_summary_parts.append("No formal tests were run (code executed without errors).")
    test_summary = "\n".join(test_summary_parts)

    result, used_fallback = await router.call_with_fallback(
        role="critic",
        template_key="review",
        variables={
            "task_description": state["task_description"],
            "code": current_code,
            "test_summary": test_summary,
        },
        max_new_tokens=512,
    )

    verdict = result.get("verdict", "approve")
    issues = result.get("issues", [])
    confidence = float(result.get("confidence", 0.5))

    logger.info(
        "Critic verdict=%s confidence=%.2f issues=%d fallback=%s (iteration=%d)",
        verdict, confidence, len(issues), used_fallback, iteration,
    )

    events.append(critic_event(
        verdict=verdict,
        issues=issues,
        confidence=confidence,
        iteration=iteration,
    ).to_dict())

    threshold = agent_config.critic_confidence_threshold
    should_reject = (
        verdict == "reject"
        and confidence > threshold
        and issues
        and not used_fallback
    )

    if should_reject:
        logger.info(
            "Critic rejected solution (confidence=%.2f > threshold=%.2f) — re-entering repair loop",
            confidence, threshold,
        )
        issue_text = " | ".join(issues[:3])
        return {
            "last_execution_passed": False,
            "last_failure_summary": f"Critic review failed:\n{issue_text}",
            "root_cause": f"Critic identified correctness issues: {issue_text}",
            "failure_category": "logic_error",
            "repair_strategy": (
                f"Address the critic's concerns: {issue_text}. "
                "Ensure all edge cases from the specification are handled."
            ),
            "status": "running",
            "events": events,
        }

    # Approved (or low-confidence rejection) — no state change needed
    return {"events": events}
