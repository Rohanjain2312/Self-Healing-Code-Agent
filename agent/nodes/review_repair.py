"""
Human-in-the-loop review node — pauses the graph for human repair approval.

ROLE IN THE GRAPH
-----------------
review_repair is inserted between update_learning_log and increment_iteration
when ``config.autonomy_level != "full_auto"``. It is the only node that uses
LangGraph's ``interrupt()`` mechanism.

HOW LangGraph interrupt() WORKS
---------------------------------
When ``interrupt(payload)`` is called:
  1. LangGraph serializes the current AgentState to the checkpointer
     (InMemorySaver or SqliteSaver).
  2. The graph execution is suspended. The ``astream()`` call yields one last
     update and then the async generator STOPS.
  3. The caller (demo/demo_runner.py) detects the pause by calling
     ``app.get_state(thread_config)`` and checking ``current.next`` — if it's
     non-empty, the graph is paused.
  4. The caller presents the interrupt payload (diagnosis info) to the user
     in the Gradio UI's repair review panel.
  5. When the user clicks Approve/Edit/Abort, demo_runner.py calls
     ``app.astream(Command(resume=decision), thread_config)`` to resume.
  6. LangGraph restores the serialized state, resumes this node AFTER the
     interrupt() call, and ``decision`` is the return value of interrupt().

THE THREE DECISIONS
--------------------
  - ``approve``:  Continue unchanged — the debugger's root_cause and
                  repair_strategy flow directly into the next generate_solution.
  - ``edit``:     User overrides the repair_strategy (and optionally root_cause).
                  New values are injected into state before increment_iteration.
  - ``abort``:    End the run immediately via Command(goto="__end__").

WHY THIS IS DISABLED ON HF SPACES
------------------------------------
The Gradio streaming generator yields events via ``yield`` inside
``_run_streaming()``. On HF Spaces the interrupt/resume flow can desync:
the generator pauses mid-stream, but the Gradio UI may not reliably catch
the pause and present the HITL panel. For this reason, app.py explicitly
sets ``autonomy_level="full_auto"`` so this node is never added to the graph
on the deployed Space. HITL works best in the local dev setup.

CHECKPOINTER REQUIREMENT
--------------------------
``interrupt()`` will raise ``RuntimeError: "checkpointer is required"`` if the
graph was compiled without one. ``build_graph()`` always attaches at least
``InMemorySaver`` when ``autonomy_level != "full_auto"``, so this is guaranteed.
"""

import logging
from typing import Any

from langgraph.types import interrupt, Command

from agent.state import AgentState
from agent.events import repair_review_event

logger = logging.getLogger(__name__)


async def review_repair(state: AgentState) -> dict[str, Any] | Command:
    """LangGraph node: emit a REPAIR_REVIEW event, then pause for human decision.

    Emits the review event BEFORE calling interrupt() so the Gradio timeline
    shows the diagnosis panel immediately (before the pause point).

    Args:
        state: Current AgentState. Reads: iteration, root_cause,
               failure_category, repair_strategy, diagnosis_confidence,
               current_code (for the code preview shown in the UI).

    Returns:
        - Command(goto="__end__") if the user chose to abort.
        - State update dict with overridden repair fields if the user edited.
        - {"events": events} if the user approved unchanged.
    """
    iteration = state.get("iteration", 0)
    events = list(state.get("events", []))
    root_cause = state.get("root_cause", "")
    failure_category = state.get("failure_category", "")
    repair_strategy = state.get("repair_strategy", "")
    confidence = state.get("diagnosis_confidence", 0.5)

    # Emit the REPAIR_REVIEW event first — the Gradio UI intercepts this to
    # render the "Approve / Edit / Abort" panel and populate it with the
    # debugger's diagnosis.
    events.append(repair_review_event(
        root_cause=root_cause,
        failure_category=failure_category,
        repair_strategy=repair_strategy,
        confidence=confidence,
        iteration=iteration,
    ).to_dict())

    # interrupt() suspends the graph here. The payload is available to the
    # caller via app.get_state(thread_config).tasks[0].interrupts[0].value.
    # LangGraph returns this payload to the caller and waits for a resume.
    decision = interrupt({
        "type": "repair_review",
        "iteration": iteration,
        "root_cause": root_cause,
        "failure_category": failure_category,
        "repair_strategy": repair_strategy,
        "confidence": confidence,
        # Only send a preview to keep the payload small — the full code is in state
        "current_code_preview": state.get("current_code", "")[:500],
        "question": "Approve this repair strategy, edit it, or abort?",
    })

    # Execution resumes here when Command(resume=decision) is received.
    action = decision.get("action", "approve") if isinstance(decision, dict) else "approve"
    logger.info("HITL review decision: action=%s (iteration=%d)", action, iteration)

    if action == "abort":
        # User wants to stop the run — jump directly to END
        return Command(goto="__end__")

    if action == "edit":
        # User overrode the repair context — inject the new values into state
        # so the next generate_solution uses them instead of the debugger's originals.
        return {
            "repair_strategy": decision.get("edited_strategy", repair_strategy),
            "root_cause": decision.get("edited_root_cause", root_cause),
            "events": events,
        }

    # "approve" — let the repair proceed as-is
    return {"events": events}
