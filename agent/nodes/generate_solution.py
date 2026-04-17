"""
Generator node — produces initial and repaired Python code solutions.

ROLE IN THE GRAPH
-----------------
generate_solution runs TWICE in the graph lifecycle:
  1. At graph start (iteration=0): generates the INITIAL solution from scratch,
     using only the task description and any cross-session lessons.
  2. In the repair loop (iteration>0): generates a TARGETED REPAIR guided by
     the debugger's root_cause + repair_strategy from the previous iteration.

This node uses the ``generator`` role, which is intentionally routed to a WEAK
model (Llama-3.2-3B via HuggingFace or Ollama). A weak generator is what makes
the self-healing loop interesting — it fails on edge cases reliably, triggering
diagnose → repair cycles that showcase the agent's autonomous correction ability.
All other roles (QA, debugger, critic) use Claude (strong model).

TEMPLATE SELECTION
------------------
Two prompt templates live in ``prompts/generator.yaml``:
  - ``initial``: "Write a Python function for: {task_description}"
     + optional cross-session lessons as context.
  - ``repair``:  "Fix this code: {current_code} — failure: {test_results},
     root cause: {root_cause}, strategy: {repair_strategy}"

WHY CROSS-SESSION LESSONS ON ITERATION 0?
------------------------------------------
If lesson_store is provided and iteration==0, we query it for the top-3
lessons most similar to the current task (via embedding similarity). These
lessons come from PREVIOUS runs on similar tasks — e.g. "always handle empty
list input" or "remember to return None instead of raising on missing key".
Prepending them to the learning_log gives the generator a head-start so it
avoids common mistakes it (or the system) has seen before.
"""

import logging
from typing import TYPE_CHECKING, Any

from agent.state import AgentState
from agent.events import (
    step_event,
    code_generated_event,
)
from llm.router import LLMRouter

if TYPE_CHECKING:
    from agent.memory_store import LessonStore

logger = logging.getLogger(__name__)


async def generate_solution(
    state: AgentState,
    router: LLMRouter,
    lesson_store: "LessonStore | None" = None,
) -> dict[str, Any]:
    """LangGraph node: generate or repair a Python code solution.

    On iteration 0 → calls ``initial`` template: task description + cross-session lessons.
    On iteration N>0 → calls ``repair`` template: task + diagnosis context + lessons.

    Args:
        state:        Current AgentState. Reads: iteration, task_description,
                      current_code, root_cause, repair_strategy, learning_log.
        router:       LLMRouter (bound via functools.partial in build_graph).
                      The generator role uses HF/Ollama; other roles use Claude.
        lesson_store: Optional ChromaDB lesson store for cross-session retrieval.
                      Only consulted on iteration=0.

    Returns:
        Partial state update: {"current_code": ..., "events": ...}
    """
    iteration = state.get("iteration", 0)
    # Copy the events list — never mutate state directly. LangGraph's reducer
    # concatenates these with any other concurrent node's events.
    events = list(state.get("events", []))

    # Emit a STEP event so the Gradio timeline shows progress immediately,
    # before the LLM call starts (which can take 30-90s on CPU).
    events.append(step_event(
        f"{'Generating initial solution' if iteration == 0 else 'Applying repair'}...",
        iteration=iteration,
    ).to_dict())

    # Decide template: we only use "repair" when we have both existing code AND
    # a root cause diagnosis to guide the repair. On iteration 0 both fields are
    # empty strings, so is_repair is False even if something weird set iteration>0.
    is_repair = (
        iteration > 0
        and state.get("current_code", "")    # code must exist to repair
        and state.get("root_cause", "")      # diagnosis must exist to guide repair
    )

    # ── Cross-session memory: retrieve relevant lessons from past runs ────────
    # Lesson retrieval only runs at iteration 0 (first generation of a new task).
    # On repair iterations the in-session learning_log already has lessons from
    # this run, so we don't need to query the vector store again.
    current_lessons = list(state.get("learning_log", []))
    if iteration == 0 and lesson_store is not None:
        retrieved = lesson_store.retrieve_relevant_lessons(
            query=state["task_description"],
            n_results=3,  # top-3 most similar lessons from past runs
        )
        if retrieved:
            logger.info(
                "Cross-session memory: prepending %d retrieved lessons", len(retrieved)
            )
            # Prepend retrieved lessons so they appear FIRST in the prompt —
            # they're more important (past experience) than in-run lessons.
            current_lessons = retrieved + current_lessons

    # Format the lesson list for prompt injection
    learning_log = _format_learning_log(current_lessons)

    # ── Build template variables based on repair vs initial ──────────────────
    if is_repair:
        # Repair prompt: give the model the failing code, WHY it failed
        # (root_cause), and WHAT to do about it (repair_strategy). This
        # targeted context is what makes repair better than just asking
        # the model to "try again".
        template_key = "repair"
        variables = {
            "task_description": state["task_description"],
            "current_code": state["current_code"],
            "test_results": state.get("last_failure_summary", "No failure details."),
            "root_cause": state.get("root_cause", "Unknown"),
            "repair_strategy": state.get("repair_strategy", "No strategy available."),
            "learning_log": learning_log,
        }
    else:
        # Initial prompt: just the task and whatever lessons we have
        template_key = "initial"
        variables = {
            "task_description": state["task_description"],
            "learning_log": learning_log,
        }

    # ── Call the generator LLM ────────────────────────────────────────────────
    # router.call() handles: system prompt load → context building → inference
    # → schema validation → retry on bad output. The "generator" role is
    # routed to the WEAK model (HF 3B or Ollama) so it fails on edge cases.
    result = await router.call(
        role="generator",
        template_key=template_key,
        variables=variables,
        max_new_tokens=2048,   # code solutions can be long
    )

    code = result["code"]
    explanation = result.get("explanation", "")

    logger.info("Generator produced %d chars of code (iteration=%d)", len(code), iteration)

    # Emit a CODE_GENERATED event — the Gradio UI intercepts this to update
    # the live code viewer panel.
    events.append(code_generated_event(
        code=code,
        iteration=iteration,
        explanation=explanation,
    ).to_dict())

    # Only update current_code and events — all other fields remain unchanged.
    # The next node (create_adversarial_tests) will read current_code.
    return {
        "current_code": code,
        "events": events,
    }


def _format_learning_log(lessons: list[str]) -> str:
    """Convert a list of lesson strings into a bullet-list for prompt injection.

    Returns a placeholder string if empty, so the prompt template always has
    something to render at the {learning_log} variable slot.
    """
    if not lessons:
        return "No prior lessons recorded."
    return "\n".join(f"- {lesson}" for lesson in lessons)
