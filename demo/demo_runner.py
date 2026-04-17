"""
Demo runner — async bridge between the LangGraph agent and the Gradio UI.

WHAT THIS MODULE DOES
---------------------
Gradio's streaming interface works with Python generators: a button click
triggers a generator function that ``yield``s (timeline_text, code_text,
lesson_text) tuples, and Gradio updates the three output components on each
yield. This module adapts the agent's async event stream to exactly that shape.

THE STREAMING CONTRACT
-----------------------
run_demo_async() is an async generator that yields different things at
different points in the run:

  1. First yield: ``{"type": "session", "app": ..., "thread_config": ...}``
     Why? Gradio needs to store the live graph reference BEFORE the first
     event arrives so that if the graph later hits an interrupt(), we can
     call resume_demo_async() on the same graph. Gradio's gr.State holds this.

  2. Subsequent yields: ``(timeline_text, code_text, lessons_text)`` tuples
     One per event — Gradio updates the three panels atomically.

  3. On HITL pause: ``{"type": "repair_review", "payload": {...}, ...}``
     The generator STOPS (returns). The UI shows the Approve/Edit/Abort panel.
     After human decision, resume_demo_async() picks up where we left off.

EVENTS LIST DEDUPLICATION
---------------------------
Each LangGraph node returns a CUMULATIVE events list (because of the
operator.add reducer — see agent/state.py). If node A emits events [e1, e2]
and node B emits events [e3], the state after B contains [e1, e2, e3] — not
just [e3]. Without deduplication we'd replay e1 and e2 again when B completes.

The fix: track total_seen (a running index). On each node completion, slice
state["events"][total_seen:] to get only new events, then advance total_seen.

HITL PAUSE AND RESUME
----------------------
When autonomy_level != "full_auto", the review_repair node calls interrupt().
LangGraph serializes the graph state to InMemorySaver and raises internally.
From the astream() caller's perspective, the stream just ends.

After the stream ends we check app.get_state(thread_config).next — if it's
non-empty, the graph is paused (not finished). We extract the interrupt
payload and yield it as a repair_review dict. The Gradio event handler
stores the AgentSession in gr.State and renders the review panel.

When the user clicks Approve/Edit/Abort, demo/app.py calls
resume_demo_async(session, decision) which sends Command(resume=decision)
to the graph and continues streaming from where it paused.

WHY NO TIMEOUT ON INTERRUPT CHECK?
------------------------------------
There is no explicit timeout on the interrupt detection loop. If the graph
hangs (e.g. a deadlock in LangGraph internals), the generator will hang
indefinitely. In practice this hasn't been an issue because LangGraph's
checkpointer always serializes state before interrupt() returns.
"""

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Generator

from agent.config import AgentConfig
from agent.graph import build_graph, _make_initial_state
from framework.streaming import (
    format_event_for_timeline,
    PUBLIC_EVENT_TYPES,
)
from agent.events import SUCCESS, CODE_GENERATED, LEARNING_UPDATE
from llm.router import LLMRouter, build_router_with_generator_override

logger = logging.getLogger(__name__)

# Hard cap on iterations for the demo — prevents runaway inference on HF Spaces
# free tier where each LLM call can take 30-90s.
_MAX_DEMO_ITERATIONS = 4

# Example tasks shown as quick-start buttons in the Gradio UI.
# Each task is chosen to reliably trigger the repair loop on the 3B generator:
# the comments explain exactly which edge case the 3B model typically gets wrong.
EXAMPLE_TASKS = [
    # Merge overlapping intervals.
    # Common 3b failure: uses > instead of >= so touching intervals [1,3],[3,5] don't merge.
    """Write a Python function `merge_intervals(intervals: list[list[int]]) -> list[list[int]]`
that merges all overlapping intervals and returns a sorted list of non-overlapping intervals.

Requirements:
- Input: list of [start, end] integer pairs. May be unsorted.
- Output: merged, sorted list of [start, end] pairs.
- Empty input returns [].
- Single interval returned unchanged.
- Intervals are inclusive: [1,3] and [3,5] must merge to [1,5].
- Do not modify the input list.

Examples:
merge_intervals([]) == []
merge_intervals([[1,3],[2,6],[8,10]]) == [[1,6],[8,10]]
merge_intervals([[1,3],[3,5]]) == [[1,5]]
merge_intervals([[3,4],[1,2]]) == [[1,2],[3,4]]""",

    # Recursively flatten nested lists and tuples.
    # Common 3b failure: iterates into string characters instead of treating strings as scalars.
    """Write a Python function `flatten(nested)` that recursively flattens nested lists
and tuples into a flat list.

Requirements:
- Flatten arbitrarily deep nesting of lists and tuples.
- Strings are scalars — do NOT iterate their characters.
- None is a scalar — preserve it in output.
- Empty lists/tuples contribute nothing.
- A non-container scalar input returns [scalar].

Examples:
flatten([1,[2,[3]],4]) == [1,2,3,4]
flatten(["hello",["world"]]) == ["hello","world"]
flatten([1,None,2]) == [1,None,2]
flatten((1,(2,3))) == [1,2,3]
flatten(42) == [42]""",

    # Deduplicate strings preserving insertion order.
    # Common 3b failure: uses set() which destroys order.
    """Write a Python function `deduplicate(items: list[str]) -> list[str]`
that removes duplicate strings while preserving the order of first occurrence.

Requirements:
- Comparison is case-sensitive.
- Strip trailing whitespace before comparing and before storing.
- Return empty list for empty input.
- Do not modify the input list.

Examples:
deduplicate([]) == []
deduplicate(["a","b","a"]) == ["a","b"]
deduplicate(["a  ","a"]) == ["a"]
deduplicate(["z","a","z","b"]) == ["z","a","b"]""",

    # Safe dict get with optional type casting.
    # Common 3b failure: doesn't handle None values or casting exceptions.
    """Write a Python function `safe_get(d: dict, key: str, default, cast_type=None)`.

Behaviour:
- Return d[key] if key exists and value is not None.
- Return default if key is missing or value is None.
- If cast_type is provided, attempt cast_type(value). On any exception return default.
- Never raise any exception regardless of inputs.

Examples:
safe_get({"a": 1}, "a", 0) == 1
safe_get({"a": None}, "a", 0) == 0
safe_get({}, "a", 0) == 0
safe_get({"a": "42"}, "a", 0, int) == 42
safe_get({"a": "bad"}, "a", 0, int) == 0
safe_get({"a": 1}, "a", 0, str) == "1"
safe_get(None, "a", 0) == 0""",
]


@dataclass
class AgentSession:
    """Holds the live compiled graph between a HITL pause and its resume.

    When the graph hits interrupt(), run_demo_async() stops and yields this
    session as a dict. Gradio's gr.State stores it. When the user decides,
    resume_demo_async(session, decision) uses app and thread_config to resume
    the EXACT same graph instance from the EXACT same checkpoint. Without this,
    the resume would try to talk to a different graph instance that doesn't
    know about the paused state.

    events_seen tracks how many events we've already processed so the resume
    path can skip ahead and only yield truly new events.
    """
    app: Any              # compiled LangGraph graph (CompiledGraph)
    thread_config: dict   # {"configurable": {"thread_id": "<uuid>"}}
    events_seen: int = field(default=0)  # cursor into the cumulative events list


class DemoUIState:
    """Local accumulator for the three Gradio output panel contents.

    Each call to apply_event() updates the relevant panel text. The getter
    methods (timeline_text, code_text, lessons_text) produce the final strings
    that get yielded to Gradio.

    This class is local to each run/resume — it does not survive across HTTP
    requests. The persistent state is in AgentState (in the checkpointer).
    """

    def __init__(self) -> None:
        self.timeline_lines: list[str] = []   # lines for the Execution Timeline panel
        self.current_code: str = ""            # content for the Code Snapshot panel
        self.learning_lessons: list[str] = []  # bullets for the Learning Log panel
        self.is_complete: bool = False         # set on SUCCESS event
        self.final_status: str = ""
        self._last_iteration: int = 0          # for display in timeline labels

    def apply_event(self, event: dict) -> None:
        """Update local UI state based on one event dict from the agent's event stream.

        Events arrive from state["events"][total_seen:] slices. Each event dict
        has a ``type`` key and a ``payload`` dict. apply_event handles:
          - PUBLIC_EVENT_TYPES: formatted as a timeline line
          - CODE_GENERATED: updates the code panel
          - LEARNING_UPDATE: updates the learning log panel
          - SUCCESS: marks the run as complete
          - REPAIR_REVIEW: adds a pause indicator to the timeline
        """
        event_type = event.get("type", "")

        # Track the iteration number for display in timeline labels.
        # Most events include "iteration" in the payload.
        if "iteration" in event:
            self._last_iteration = event["iteration"]

        # Format this event as a timeline line if it's a public event type
        # (internal events like STEP are filtered to keep the timeline readable).
        if event_type in PUBLIC_EVENT_TYPES:
            line = format_event_for_timeline(event)
            if line:
                self.timeline_lines.append(line)

        # CODE_GENERATED: live-update the code panel with the latest generated code
        if event_type == CODE_GENERATED:
            self.current_code = event.get("payload", {}).get("code", self.current_code)

        # LEARNING_UPDATE: refresh the learning log with the updated lesson list
        if event_type == LEARNING_UPDATE:
            self.learning_lessons = event.get("payload", {}).get("lessons", self.learning_lessons)

        # SUCCESS: the agent finished — mark for the final status message
        if event_type == SUCCESS:
            self.is_complete = True
            self.final_status = "success"

        # REPAIR_REVIEW is already formatted by format_event_for_timeline above.
        # No duplicate append needed here.

    def timeline_text(self) -> str:
        return "\n".join(self.timeline_lines) if self.timeline_lines else "Waiting for agent..."

    def code_text(self) -> str:
        return self.current_code if self.current_code else "# Waiting for code generation..."

    def lessons_text(self) -> str:
        if not self.learning_lessons:
            return "No lessons recorded yet."
        # Bullet format for the Learning Log panel
        return "\n".join(f"• {lesson}" for lesson in self.learning_lessons)


async def run_demo_async(
    task_description: str,
    router: LLMRouter | None = None,
    config: AgentConfig | None = None,
) -> AsyncGenerator[Any, None]:
    """Async generator: run the agent and stream (timeline, code, log) tuples to Gradio.

    YIELD SEQUENCE:
      1. {"type": "session", ...}         ← Gradio stores this in gr.State
      2. ("Agent starting...", ...)        ← initial UI state
      3+. (timeline, code, log) tuples    ← one per new event
      EITHER:
        4a. Normal end: final tuple + return
        4b. HITL pause: {"type": "repair_review", ...} + return (wait for human)

    Args:
        task_description: User's problem statement.
        router:           Optional pre-built LLMRouter. If None, auto-detected.
        config:           AgentConfig. Defaults to AgentConfig() if None.
    """
    if not task_description or not task_description.strip():
        yield ("No task provided.", "# No task.", "")
        return

    # Build router and config if not provided by the caller (demo/app.py always passes both)
    if router is None:
        router = build_router_with_generator_override()
    if config is None:
        config = AgentConfig()

    # Build the LessonStore for cross-session memory retrieval at iteration 0
    lesson_store = None
    if config.enable_cross_session_memory:
        from agent.memory_store import LessonStore
        lesson_store = LessonStore(config.memory_persist_dir)

    # Build and compile the graph. The compiled app is STATEFUL — it holds
    # checkpoint data when interrupt() fires. We must keep the SAME app
    # instance throughout the run and resume.
    app = build_graph(router=router, config=config, lesson_store=lesson_store)
    initial_state = _make_initial_state(task_description, _MAX_DEMO_ITERATIONS)
    # Unique thread_id per run so concurrent runs don't share checkpoint state
    thread_config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    # ── Yield 1: session dict ────────────────────────────────────────────────
    # Gradio's _run_streaming() intercepts this dict (not a UI tuple) and
    # stores it in agent_session_state. Without this first yield, the session
    # reference would be lost if the graph pauses before the stream ends.
    yield {"type": "session", "app": app, "thread_config": thread_config}

    state = DemoUIState()
    yield ("Agent starting...", "# Initializing...", "No lessons yet.")

    # total_seen: cursor into the cumulative state["events"] list.
    # See module docstring — events are cumulative (operator.add reducer), so
    # we track how many we've already processed and only yield new ones.
    total_seen = 0

    try:
        # astream yields {node_name: partial_state} as each node completes.
        async for state_update in app.astream(initial_state, thread_config):
            for node_name, node_state in state_update.items():
                if not isinstance(node_state, dict):
                    continue  # skip non-dict updates (metadata, __interrupt__ tags)

                # Extract only new events using our cursor
                events = node_state.get("events", [])
                new_events = events[total_seen:]
                total_seen = len(events)   # advance cursor for next iteration

                for event in new_events:
                    if isinstance(event, dict):
                        state.apply_event(event)
                        # Yield on every event so Gradio updates in real time.
                        # This produces fine-grained streaming updates — the user
                        # sees the timeline grow line by line, not in big batches.
                        yield (state.timeline_text(), state.code_text(), state.lessons_text())

            # After processing each node's output, check whether the graph paused.
            # graph.get_state().next is non-empty when interrupt() was called and
            # the graph is waiting for a Command(resume=...).
            current = app.get_state(thread_config)
            if current.next:
                # Extract the interrupt payload — contains the HITL review info
                try:
                    payload = current.tasks[0].interrupts[0].value
                except (IndexError, AttributeError):
                    payload = {}  # unexpected structure — use empty payload

                # Yield the repair_review dict and STOP this generator.
                # The caller (demo/app.py._run_streaming) stores the session and
                # renders the HITL panel. The graph stays alive in InMemorySaver.
                yield {
                    "type": "repair_review",
                    "payload": payload,
                    "iteration": state._last_iteration,
                    "events_seen": total_seen,  # pass cursor so resume can skip old events
                }
                return  # generator stops here; resume_demo_async() continues

    except Exception as exc:
        # Surface errors in the Gradio timeline instead of crashing the server.
        # The user sees the error in the timeline panel and can try a new task.
        logger.error("Demo runner error: %s", exc, exc_info=True)
        state.timeline_lines.append(f"[ERROR] {type(exc).__name__}: {exc}")
        yield (state.timeline_text(), state.code_text(), state.lessons_text())
        return

    # Normal completion — add a summary line and yield the final state
    if state.is_complete:
        state.timeline_lines.append("Agent completed successfully.")
    else:
        state.timeline_lines.append("Agent reached maximum iterations.")

    yield (state.timeline_text(), state.code_text(), state.lessons_text())


async def resume_demo_async(
    session: AgentSession,
    decision: dict,
    router: LLMRouter | None = None,
    config: AgentConfig | None = None,
) -> AsyncGenerator[tuple[str, str, str], None]:
    """Resume a paused agent graph after a HITL decision.

    Called by demo/app.py when the user clicks Approve/Edit/Abort in the
    repair review panel. Sends the decision to the graph and continues streaming.

    If another interrupt() fires (another repair iteration), this generator
    yields a partial update and stops again — the cycle repeats until either
    the run completes or the user aborts.

    Args:
        session:  AgentSession from the first yield of run_demo_async().
                  Must hold the same compiled app instance that was paused.
        decision: Human decision dict:
                  {"action": "approve"}
                  {"action": "edit", "edited_strategy": "...", "edited_root_cause": "..."}
                  {"action": "abort"}
        router:   Unused (reserved for future use).
        config:   Unused (reserved for future use).
    """
    from langgraph.types import Command

    app = session.app
    thread_config = session.thread_config
    state = DemoUIState()
    # Start the cursor where the previous run left off — skip all events we
    # already yielded before the pause.
    total_seen = session.events_seen

    try:
        # Command(resume=decision) restores the pickled graph state from the
        # checkpointer, injects decision as the return value of interrupt(),
        # and continues executing from the review_repair node.
        async for state_update in app.astream(Command(resume=decision), thread_config):
            for node_name, node_state in state_update.items():
                if not isinstance(node_state, dict):
                    continue

                events = node_state.get("events", [])
                new_events = events[total_seen:]
                total_seen = len(events)

                for event in new_events:
                    if isinstance(event, dict):
                        state.apply_event(event)
                        yield (state.timeline_text(), state.code_text(), state.lessons_text())

            # Check for another interrupt (another repair iteration needing review)
            current = app.get_state(thread_config)
            if current.next:
                try:
                    payload = current.tasks[0].interrupts[0].value
                except (IndexError, AttributeError):
                    payload = {}
                # Emit a timeline update showing the pause, then stop and wait
                interrupt_event = {
                    "type": "repair_review",
                    "payload": payload,
                    "iteration": state._last_iteration,
                    "events_seen": total_seen,
                }
                state.apply_event(interrupt_event)
                yield (state.timeline_text(), state.code_text(), state.lessons_text())
                return  # caller must call resume_demo_async() again

    except Exception as exc:
        logger.error("Resume error: %s", exc, exc_info=True)
        state.timeline_lines.append(f"[ERROR] {type(exc).__name__}: {exc}")
        yield (state.timeline_text(), state.code_text(), state.lessons_text())
        return

    # Normal completion after this resume
    if state.is_complete:
        state.timeline_lines.append("Agent completed successfully.")
    else:
        state.timeline_lines.append("Agent reached maximum iterations.")

    yield (state.timeline_text(), state.code_text(), state.lessons_text())


def run_demo_sync(
    task_description: str,
    router: LLMRouter | None = None,
    config: AgentConfig | None = None,
) -> Generator[tuple[str, str, str], None, None]:
    """Synchronous generator wrapper for contexts that can't use async generators.

    IMPORTANT: Only works with autonomy_level='full_auto'. If review_repairs
    or review_all is set, this function raises NotImplementedError because
    interrupt() requires async resume and asyncio.run() can't be nested to
    handle it.

    WHY asyncio.run() HERE?
    Gradio's AnyIO worker threads don't provide a current event loop via
    asyncio.get_event_loop(). asyncio.run() creates its own isolated loop,
    runs the coroutine to completion, and tears it down — the only safe
    pattern in this context.

    Args:
        task_description: User's problem statement.
        router:           Optional pre-built LLMRouter.
        config:           AgentConfig. Must have autonomy_level='full_auto'.
    """
    if config and config.autonomy_level != "full_auto":
        raise NotImplementedError(
            "run_demo_sync() only supports autonomy_level='full_auto'. "
            "Use run_demo_async() with the async Gradio path for HITL."
        )

    async def _collect():
        results = []
        async for update in run_demo_async(task_description, router=router, config=config):
            # Filter out session/repair_review dicts — this sync wrapper only
            # yields the (timeline, code, log) UI tuples.
            if isinstance(update, tuple):
                results.append(update)
        return results

    # asyncio.run() always creates and tears down its own event loop.
    results = asyncio.run(_collect())

    yield from results
