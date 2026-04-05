"""
Demo runner — async bridge between the agent and the Gradio UI.

Converts the agent's async event stream into incremental UI state updates
that Gradio can consume via its generator-based streaming API.

The runner maintains UI state locally and yields (timeline, code, log) tuples
after each event so Gradio can update all three components atomically.

HITL support:
  When autonomy_level != "full_auto", the graph pauses at interrupt() for human
  review. run_demo_async() yields a session dict on startup (so the UI can hold
  the live graph reference) and a repair_review dict on interrupt (so the UI
  can show the review panel). resume_demo_async() resumes the graph with the
  human decision and continues streaming.

HF Spaces constraints respected:
  - max_iterations capped at 4 to limit inference time
  - No benchmark execution in demo mode
  - Streaming via Python async generators
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
from agent.events import SUCCESS, CODE_GENERATED, LEARNING_UPDATE, REPAIR_REVIEW
from llm.router import LLMRouter, build_router_with_generator_override

logger = logging.getLogger(__name__)

# Cap for HF Spaces free tier — prevents runaway inference costs
_MAX_DEMO_ITERATIONS = 4

# Example tasks shown in the demo UI
EXAMPLE_TASKS = [
    # Task 1 (from benchmark): Merge overlapping intervals
    # Common 3b failure: uses > instead of >= so touching intervals [1,3],[3,5] don't merge
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

    # Task 2 (from benchmark): Flatten nested lists
    # Common 3b failure: iterates into string characters instead of treating strings as scalars
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

    # Task 3 (fresh): Deduplicate preserving insertion order
    # Common 3b failure: uses set() which destroys order
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

    # Task 4 (fresh): Safe dict get with optional type casting
    # Common 3b failure: doesn't handle None values or casting exceptions
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
    """Holds the live graph and thread config between HITL pause and resume."""
    app: Any          # compiled LangGraph graph
    thread_config: dict
    events_seen: int = field(default=0)  # total events seen so far for de-duplication


class DemoUIState:
    """Accumulated UI state built from the event stream."""

    def __init__(self) -> None:
        self.timeline_lines: list[str] = []
        self.current_code: str = ""
        self.learning_lessons: list[str] = []
        self.is_complete: bool = False
        self.final_status: str = ""
        self._last_iteration: int = 0

    def apply_event(self, event: dict) -> None:
        event_type = event.get("type", "")

        # Track current iteration from any event that carries it
        if "iteration" in event:
            self._last_iteration = event["iteration"]

        if event_type in PUBLIC_EVENT_TYPES:
            line = format_event_for_timeline(event)
            if line:
                self.timeline_lines.append(line)

        if event_type == CODE_GENERATED:
            self.current_code = event.get("payload", {}).get("code", self.current_code)

        if event_type == LEARNING_UPDATE:
            self.learning_lessons = event.get("payload", {}).get("lessons", self.learning_lessons)

        if event_type == SUCCESS:
            self.is_complete = True
            self.final_status = "success"

        if event_type == REPAIR_REVIEW:
            payload = event.get("payload", {})
            category = payload.get("failure_category", "?")
            confidence = payload.get("confidence", 0)
            line = (
                f"[iter {self._last_iteration}] ⏸  Human review — "
                f"[{category}] confidence {confidence:.0%}"
            )
            self.timeline_lines.append(line)

    def timeline_text(self) -> str:
        return "\n".join(self.timeline_lines) if self.timeline_lines else "Waiting for agent..."

    def code_text(self) -> str:
        return self.current_code if self.current_code else "# Waiting for code generation..."

    def lessons_text(self) -> str:
        if not self.learning_lessons:
            return "No lessons recorded yet."
        return "\n".join(f"• {lesson}" for lesson in self.learning_lessons)


async def run_demo_async(
    task_description: str,
    router: LLMRouter | None = None,
    config: AgentConfig | None = None,
) -> AsyncGenerator[Any, None]:
    """
    Async generator that yields:
      - {"type": "session", "app": ..., "thread_config": ...} first (so UI can hold session)
      - (timeline, code, learning_log) tuples on each event
      - {"type": "repair_review", "payload": ..., "iteration": ...} on HITL pause (then stops)

    The generator stops (does NOT close the graph) when an interrupt is encountered.
    Call resume_demo_async() with the AgentSession to continue.

    Args:
        task_description: The user's task string.
        router: Optional pre-constructed LLMRouter.
        config: AgentConfig controlling which features are active.
    """
    if not task_description or not task_description.strip():
        yield ("No task provided.", "# No task.", "")
        return

    if router is None:
        router = build_router_with_generator_override()
    if config is None:
        config = AgentConfig()

    lesson_store = None
    if config.enable_cross_session_memory:
        from agent.memory_store import LessonStore
        lesson_store = LessonStore(config.memory_persist_dir)

    app = build_graph(router=router, config=config, lesson_store=lesson_store)
    initial_state = _make_initial_state(task_description, _MAX_DEMO_ITERATIONS)
    thread_config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    # Yield session event first so the UI can store the session for HITL resume
    yield {"type": "session", "app": app, "thread_config": thread_config}

    state = DemoUIState()
    yield ("Agent starting...", "# Initializing...", "No lessons yet.")

    # total_seen tracks position in the cumulative events list returned by each node.
    # Nodes read state["events"] and append new items, returning the full list.
    # We use total_seen to extract only the genuinely new events each time.
    total_seen = 0

    try:
        async for state_update in app.astream(initial_state, thread_config):
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

            # Check for interrupt after processing each graph step
            current = app.get_state(thread_config)
            if current.next:
                try:
                    payload = current.tasks[0].interrupts[0].value
                except (IndexError, AttributeError):
                    payload = {}
                yield {
                    "type": "repair_review",
                    "payload": payload,
                    "iteration": state._last_iteration,
                    "events_seen": total_seen,
                }
                return  # Stop; wait for human decision via resume_demo_async()

    except Exception as exc:
        logger.error("Demo runner error: %s", exc, exc_info=True)
        state.timeline_lines.append(f"[ERROR] {type(exc).__name__}: {exc}")
        yield (state.timeline_text(), state.code_text(), state.lessons_text())
        return

    # Final update on normal completion
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
    """
    Resume a paused agent graph after a HITL decision and continue streaming.

    Args:
        session: AgentSession holding the live graph and thread config.
        decision: Human decision dict, e.g. {"action": "approve"},
                  {"action": "edit", "edited_strategy": "..."} or {"action": "abort"}.
        router: Optional LLMRouter (unused here, reserved for future use).
        config: Optional AgentConfig (unused here, reserved for future use).
    """
    from langgraph.types import Command

    app = session.app
    thread_config = session.thread_config
    state = DemoUIState()
    total_seen = session.events_seen

    try:
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

            # Check for another interrupt
            current = app.get_state(thread_config)
            if current.next:
                try:
                    payload = current.tasks[0].interrupts[0].value
                except (IndexError, AttributeError):
                    payload = {}
                interrupt_event = {
                    "type": "repair_review",
                    "payload": payload,
                    "iteration": state._last_iteration,
                    "events_seen": total_seen,
                }
                state.apply_event(interrupt_event)
                yield (state.timeline_text(), state.code_text(), state.lessons_text())
                return  # Stop and wait for next human decision

    except Exception as exc:
        logger.error("Resume error: %s", exc, exc_info=True)
        state.timeline_lines.append(f"[ERROR] {type(exc).__name__}: {exc}")
        yield (state.timeline_text(), state.code_text(), state.lessons_text())
        return

    # Final update on normal completion
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
    """
    Synchronous generator wrapper for Gradio's streaming interface.

    Only supports autonomy_level='full_auto'. For HITL (review_repairs / review_all),
    use the async Gradio path with run_demo_async() and resume_demo_async().

    Args:
        task_description: The user's task string.
        router: Optional pre-constructed LLMRouter.
        config: AgentConfig preset to use. Defaults to AgentConfig().
    """
    if config and config.autonomy_level != "full_auto":
        raise NotImplementedError(
            "run_demo_sync() only supports autonomy_level='full_auto'. "
            "Use run_demo_async() with the async Gradio path for HITL."
        )

    async def _collect():
        results = []
        async for update in run_demo_async(task_description, router=router, config=config):
            # Skip session/interrupt dict events — only collect UI tuples
            if isinstance(update, tuple):
                results.append(update)
        return results

    # asyncio.run() always creates and tears down its own event loop.
    # This is required inside Gradio's AnyIO worker threads, where
    # asyncio.get_event_loop() raises RuntimeError (no current loop).
    results = asyncio.run(_collect())

    yield from results
