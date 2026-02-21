"""
Demo runner — async bridge between the agent and the Gradio UI.

Converts the agent's async event stream into incremental UI state updates
that Gradio can consume via its generator-based streaming API.

The runner maintains UI state locally and yields (timeline, code, log) tuples
after each event so Gradio can update all three components atomically.

HF Spaces constraints respected:
  - max_iterations capped at 4 to limit inference time
  - No benchmark execution in demo mode
  - Streaming via Python async generators
"""

import asyncio
import logging
from typing import AsyncGenerator, Generator

from agent.graph import stream_agent
from framework.streaming import (
    format_event_for_timeline,
    extract_learning_log,
    extract_latest_code,
    PUBLIC_EVENT_TYPES,
)
from agent.events import SUCCESS, FAILURE, CODE_GENERATED, LEARNING_UPDATE
from llm.router import LLMRouter

logger = logging.getLogger(__name__)

# Cap for HF Spaces free tier — prevents runaway inference costs
_MAX_DEMO_ITERATIONS = 4

# Example tasks shown in the demo UI
EXAMPLE_TASKS = [
    "Write a Python function `merge_intervals(intervals: list[list[int]]) -> list[list[int]]` that merges overlapping intervals. Handle empty input, single intervals, and touching intervals (e.g. [1,3] and [3,5] should merge to [1,5]).",
    "Write a Python function `flatten(nested) -> list` that recursively flattens nested lists and tuples. Strings are scalars (do not iterate their characters). None values are preserved.",
    "Write a Python function `deduplicate_logs(logs: list[str]) -> list[str]` that removes duplicate log lines, preserving first-occurrence order. Strip trailing whitespace before comparing.",
    "Write a Python function `compress_ranges(numbers: list[int]) -> list[str]` that compresses a list of integers into range strings. E.g. [1,2,3,5,7,8,9] → ['1-3', '5', '7-9']. Handle duplicates and single elements.",
    "Write a Python function `safe_divide(numerator: float, denominator: float) -> float | None` that returns None for division by zero, infinity, or NaN arguments. No exceptions should propagate.",
]


class DemoUIState:
    """Accumulated UI state built from the event stream."""

    def __init__(self) -> None:
        self.timeline_lines: list[str] = []
        self.current_code: str = ""
        self.learning_lessons: list[str] = []
        self.is_complete: bool = False
        self.final_status: str = ""

    def apply_event(self, event: dict) -> None:
        event_type = event.get("type", "")

        if event_type in PUBLIC_EVENT_TYPES:
            line = format_event_for_timeline(event)
            self.timeline_lines.append(line)

        if event_type == CODE_GENERATED:
            self.current_code = event.get("payload", {}).get("code", self.current_code)

        if event_type == LEARNING_UPDATE:
            self.learning_lessons = event.get("payload", {}).get("lessons", self.learning_lessons)

        if event_type == SUCCESS:
            self.is_complete = True
            self.final_status = "success"

    def timeline_text(self) -> str:
        return "\n".join(self.timeline_lines) if self.timeline_lines else "Waiting for agent..."

    def code_text(self) -> str:
        return self.current_code if self.current_code else "# Waiting for code generation..."

    def lessons_text(self) -> str:
        if not self.learning_lessons:
            return "No lessons recorded yet."
        return "\n".join(f"• {l}" for l in self.learning_lessons)


async def run_demo_async(
    task_description: str,
    router: LLMRouter | None = None,
) -> AsyncGenerator[tuple[str, str, str], None]:
    """
    Async generator that yields (timeline, code, learning_log) tuples.

    Each yield updates all three Gradio components simultaneously.
    """
    if not task_description or not task_description.strip():
        yield ("No task provided.", "# No task.", ""), False
        return

    state = DemoUIState()
    yield (
        "Agent starting...",
        "# Initializing...",
        "No lessons yet.",
    )

    try:
        async for event in stream_agent(
            task_description=task_description.strip(),
            max_iterations=_MAX_DEMO_ITERATIONS,
            router=router,
        ):
            if event is None:
                break
            state.apply_event(event)
            yield (
                state.timeline_text(),
                state.code_text(),
                state.lessons_text(),
            )

    except Exception as exc:
        logger.error("Demo runner error: %s", exc, exc_info=True)
        state.timeline_lines.append(f"[ERROR] Agent encountered an error: {exc}")
        yield (
            state.timeline_text(),
            state.code_text(),
            state.lessons_text(),
        )
        return

    # Final update
    if state.is_complete:
        state.timeline_lines.append("Agent completed successfully.")
    else:
        state.timeline_lines.append("Agent reached maximum iterations.")

    yield (
        state.timeline_text(),
        state.code_text(),
        state.lessons_text(),
    )


def run_demo_sync(
    task_description: str,
    router: LLMRouter | None = None,
) -> Generator[tuple[str, str, str], None, None]:
    """
    Synchronous generator wrapper for Gradio's streaming interface.

    Gradio's gr.Interface with streaming expects a regular generator.
    This bridges the async generator to the synchronous Gradio API.
    """
    async def _collect():
        results = []
        async for update in run_demo_async(task_description, router=router):
            results.append(update)
        return results

    # asyncio.run() always creates and tears down its own event loop.
    # This is required inside Gradio's AnyIO worker threads, where
    # asyncio.get_event_loop() raises RuntimeError (no current loop).
    results = asyncio.run(_collect())

    yield from results
