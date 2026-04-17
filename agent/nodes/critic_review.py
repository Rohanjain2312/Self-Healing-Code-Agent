"""
Critic node — agent self-reflection after all tests pass (Fix 15).

ROLE IN THE GRAPH
-----------------
critic_review runs AFTER execute_solution when last_execution_passed=True.
It acts as a second gate before declaring success — it checks whether the
solution is actually correct or just "tests-pass-but-wrong" correct.

THE PROBLEM IT SOLVES
----------------------
Generated tests can be wrong or incomplete. A generator model and a QA model
can both agree on a wrong interpretation of the spec and write mutually-consistent
but incorrect code and tests. The critic has INDEPENDENT perspective — it reads
the original task description and the code without any awareness of what tests
were generated, and asks: "does this code actually solve the problem correctly?"

Examples of what the critic catches:
  - A function that silently returns None for invalid inputs instead of raising
    the specified exception.
  - Off-by-one errors that the edge-case tests happened not to cover.
  - Sorting a list in-place when the spec says "return a sorted copy".
  - Returning the wrong data type (e.g. string instead of int).

THE CONFIDENCE THRESHOLD
------------------------
Rejecting a passing solution is expensive (wastes an iteration). Low-confidence
rejections are false positives — the critic wasn't sure but flagged it anyway.
We only re-open the repair loop when:
  - verdict == "reject"
  - confidence > config.critic_confidence_threshold (default: 0.6)
  - At least one specific issue was listed
  - The LLM response did NOT use the fallback path

This means the critic errs on the side of approval when uncertain, which is the
right default for an autonomous agent.

WHY call_with_fallback() NOT call()
-----------------------------------
Critic failures must never BLOCK a successful run. If the LLM fails to produce
a valid review, the fallback conservatively approves the solution. Using call()
(raises on failure) would turn a passing run into a crash just because the critic
had a bad day.
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
    """LangGraph node: review a passing solution for correctness issues.

    Args:
        state:        Current AgentState. Reads: iteration, task_description,
                      current_code, spec_test_code, current_test_code.
        router:       LLMRouter. Uses the ``critic`` role → Claude (Haiku 4.5).
        agent_config: AgentConfig (bound via functools.partial). Used for
                      critic_confidence_threshold.

    Returns:
        Empty dict (approved, no state change) if verdict == "approve".
        State override dict (re-opens repair loop) if verdict == "reject"
        with high confidence.
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

    # Build a human-readable summary of which tests passed — the critic uses
    # this context to understand what was already verified so it doesn't
    # re-flag things the tests cover well.
    test_summary_parts = []
    if spec_test_code.strip():
        test_summary_parts.append("Spec-blind oracle tests: PASSED")
    if adversarial_test_code.strip():
        test_summary_parts.append("Adversarial edge-case tests: PASSED")
    if not test_summary_parts:
        test_summary_parts.append("No formal tests were run (code executed without errors).")
    test_summary = "\n".join(test_summary_parts)

    # call_with_fallback — returns (result, used_fallback). On fallback:
    # result is {"verdict": "approve", "issues": [], "confidence": 0.5}
    # so a broken critic always approves conservatively.
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

    # Decide whether to re-open the repair loop. All four conditions must hold:
    threshold = agent_config.critic_confidence_threshold
    should_reject = (
        verdict == "reject"
        and confidence > threshold    # high confidence — not a borderline call
        and issues                    # at least one specific issue identified
        and not used_fallback         # don't trust the fallback's output as a rejection
    )

    if should_reject:
        logger.info(
            "Critic rejected solution (confidence=%.2f > threshold=%.2f) — re-entering repair loop",
            confidence, threshold,
        )
        issue_text = " | ".join(issues[:3])  # include top 3 issues in the repair context
        # Inject the critic's feedback as the root_cause and repair_strategy so
        # the generator in the next iteration knows exactly what to fix.
        # Setting last_execution_passed=False causes _route_after_critic to
        # return "diagnose_failure", re-entering the repair loop.
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

    # Critic approved (or low-confidence rejection — treat as approval).
    # Return only events — no other state changes needed. The routing function
    # will see last_execution_passed=True and route to END.
    return {"events": events}
