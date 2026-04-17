"""
Rolling memory summarizer node — distills lessons from each repair iteration.

ROLE IN THE GRAPH
-----------------
update_learning_log runs after diagnose_failure (when the diagnosis had high
enough confidence) and before increment_iteration (or review_repair in HITL mode).
It acts as the agent's short-term memory consolidation step.

WHY A SUMMARIZER (NOT JUST APPEND)?
------------------------------------
Simply appending the root_cause from every iteration would cause the learning_log
to grow without bound and become noisy (duplicate patterns, contradictory advice).
Instead, the memory_summarizer LLM:
  1. Reads the current learning_log (up to 5 bullets)
  2. Reads the latest diagnosis (root_cause, failure_category, repair_strategy)
  3. Produces a NEW compressed list of ≤5 bullet lessons that integrate the
     new information while removing redundancies

This keeps the prompt injection (in generate_solution) concise and useful across
many iterations. The lessons are generalizable patterns, not iteration-specific
facts — e.g. "handle empty input" not "iteration 2 had a KeyError on line 5".

CROSS-SESSION MEMORY
--------------------
After the run completes, agent/graph.py persists the learning_log to ChromaDB
(if enable_cross_session_memory=True). Future runs on similar tasks will
retrieve the most relevant lessons at iteration 0 via embedding similarity.

WHY call() NOT call_with_fallback()?
-------------------------------------
We use router.call() (strict) so that if memory summarization fails, we notice
it and don't silently return stale lessons from the fallback. The fallback for
memory_summarizer just returns the prior_lessons unchanged — if we hit the
fallback path we'd miss the new diagnosis entirely. A hard failure is more
informative here.

Actually we DO use router.call() here but the node gracefully handles the case
where root_cause is empty (skips LLM call and returns early).
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
    """LangGraph node: compress iteration evidence into a rolling lesson log.

    Args:
        state:  Current AgentState. Reads: iteration, learning_log,
                root_cause, failure_category, repair_strategy,
                last_execution_passed.
        router: LLMRouter. Uses ``memory_summarizer`` role → Claude (Haiku 4.5).

    Returns:
        Partial state: {"learning_log": [...], "events": [...]}
        Returns just events (no learning_log update) if root_cause is empty.
    """
    iteration = state.get("iteration", 0)
    events = list(state.get("events", []))

    events.append(step_event(
        "Updating rolling learning log...",
        iteration=iteration,
    ).to_dict())

    prior_lessons = state.get("learning_log", [])

    # Guard: only call the LLM if we have actual diagnosis context.
    # root_cause is empty on iteration 0 (no failure yet to learn from) and
    # when coming from the blind-retry path (confidence < 0.3 skips this node).
    root_cause = state.get("root_cause", "")
    if not root_cause:
        # Nothing to learn from — return early with just the step event
        return {"events": events}

    # Determine the outcome label for the summarizer prompt.
    # "passed" means the repair in the PREVIOUS iteration worked; "still failing"
    # means we're about to attempt another repair. This context helps the
    # summarizer write more accurate lessons.
    last_passed = state.get("last_execution_passed", False)
    outcome = "Test passed after repair." if last_passed else "Test still failing after repair attempt."

    # Call memory_summarizer: given prior lessons + current diagnosis, produce
    # an updated lesson list of ≤5 bullet points. The schema enforces this cap.
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
        max_new_tokens=256,   # lessons are short bullet points, 256 is plenty
    )

    lessons = result.get("lessons", prior_lessons)
    # Defensive cap — the schema should enforce ≤5, but truncate here as a
    # safety net in case schema coercion fails.
    lessons = lessons[:5]

    logger.info("Learning log updated: %d lessons", len(lessons))

    # Emit a LEARNING_UPDATE event — the Gradio UI intercepts this to refresh
    # the "Learning Log" panel in the right column.
    events.append(learning_update_event(
        lessons=lessons,
        iteration=iteration,
    ).to_dict())

    return {
        "learning_log": lessons,
        "events": events,
    }


def _format_lessons(lessons: list[str]) -> str:
    """Render lesson list as a bullet-point string for the prompt template."""
    if not lessons:
        return "No prior lessons."
    return "\n".join(f"- {lesson}" for lesson in lessons)
