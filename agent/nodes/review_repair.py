"""
Human-in-the-loop review node — uses LangGraph interrupt() to pause execution.

When autonomy_level is not "full_auto", this node fires after diagnosis and
before the next repair iteration. It presents the diagnosis to a human operator
who can approve, edit, or abort.

interrupt() suspends the graph at this node. The caller (demo or CLI) must
resume by calling graph.invoke(Command(resume=decision), thread_config).

Decision payload:
  {"action": "approve"}
      → continue with the diagnosed repair strategy unchanged

  {"action": "edit", "edited_strategy": "...", "edited_root_cause": "..."}
      → override the repair strategy and/or root cause before regenerating

  {"action": "abort"}
      → end the run immediately via Command(goto="__end__")

Design note: LangGraph requires a checkpointer to be set on the compiled graph
for interrupt() to work. build_graph() always attaches at least InMemorySaver
when autonomy_level != "full_auto".
"""

import logging
from typing import Any

from langgraph.types import interrupt, Command

from agent.state import AgentState
from agent.events import repair_review_event

logger = logging.getLogger(__name__)


async def review_repair(state: AgentState) -> dict[str, Any] | Command:
    """
    LangGraph node: pause for human review of the proposed repair strategy.

    Emits a REPAIR_REVIEW event for the UI, then calls interrupt() to suspend
    the graph. Resumes when the caller provides a Command(resume=decision).
    """
    iteration = state.get("iteration", 0)
    events = list(state.get("events", []))
    root_cause = state.get("root_cause", "")
    failure_category = state.get("failure_category", "")
    repair_strategy = state.get("repair_strategy", "")
    confidence = state.get("diagnosis_confidence", 0.5)

    events.append(repair_review_event(
        root_cause=root_cause,
        failure_category=failure_category,
        repair_strategy=repair_strategy,
        confidence=confidence,
        iteration=iteration,
    ).to_dict())

    # Suspend execution — caller receives this payload and must resume
    decision = interrupt({
        "type": "repair_review",
        "iteration": iteration,
        "root_cause": root_cause,
        "failure_category": failure_category,
        "repair_strategy": repair_strategy,
        "confidence": confidence,
        "current_code_preview": state.get("current_code", "")[:500],
        "question": "Approve this repair strategy, edit it, or abort?",
    })

    action = decision.get("action", "approve") if isinstance(decision, dict) else "approve"
    logger.info("HITL review decision: action=%s (iteration=%d)", action, iteration)

    if action == "abort":
        return Command(goto="__end__")

    if action == "edit":
        return {
            "repair_strategy": decision.get("edited_strategy", repair_strategy),
            "root_cause": decision.get("edited_root_cause", root_cause),
            "events": events,
        }

    # "approve" — continue unchanged
    return {"events": events}
