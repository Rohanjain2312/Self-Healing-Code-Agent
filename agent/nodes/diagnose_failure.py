"""
Debugger node — ReAct-style root-cause analysis with tool use.

ROLE IN THE GRAPH
-----------------
diagnose_failure is the most complex node. It runs AFTER execute_solution when
tests fail. Its job is to figure out WHY the code failed and WHAT specific
repair to apply — producing root_cause, failure_category, repair_strategy, and
diagnosis_confidence.

The diagnosis is critical: it's injected into the generator's repair prompt,
so a good diagnosis leads to a targeted fix, while a bad diagnosis wastes an
iteration on a wrong repair direction.

REACT LOOP ARCHITECTURE
-----------------------
Unlike other nodes that make a single LLM call, the debugger runs a
think → act → observe loop:

  1. Call the LLM with current context (code, failures, history)
  2. LLM returns either:
     a. {"action": "use_tool", "tool": "...", "args": {...}} — wants to investigate
     b. {"action": "final_diagnosis", "root_cause": "...", ...} — has conclusion
  3. If (a): run the tool, append observation, loop back to step 1
  4. If (b) or max steps reached: extract the diagnosis and exit

Three tools are available:
  - run_snippet:      execute a small Python snippet to test a hypothesis
                      e.g. "does this specific edge case reproduce the bug?"
  - inspect_function: parse the function's AST to check its signature/docs
  - diff_iterations:  compare code from two different iterations to see what changed

WHY AN INLINE LOOP (NOT A SUB-GRAPH)?
--------------------------------------
We implement the ReAct loop as a Python for-loop inside the node rather than
as a LangGraph sub-graph because:
  1. Ollama/Llama3 doesn't support native function calling — we simulate tool
     use via JSON-in-text.
  2. An inline loop is simpler to unit-test and easier to inspect in logs.
  3. We don't need checkpointing at each tool step — only at the node boundary.

FALLBACK
--------
If the ReAct loop fails to produce a final_diagnosis (all retries exhausted,
or JSON parsing fails repeatedly), we fall back to a single LLM call with
the standard validated schema via router.call_with_fallback(). This ensures
the node always produces SOME diagnosis rather than crashing.

LOW-CONFIDENCE ROUTING
-----------------------
If confidence < 0.3, graph routing (_route_after_diagnosis) skips the
targeted repair and jumps directly to generate_solution for a blind retry.
This prevents a confidently-wrong diagnosis from wasting iterations.
"""

import json
import logging
from typing import Any

from agent.state import AgentState
from agent.events import step_event, diagnosis_event, tool_use_event
from agent.tools import call_tool
from llm.router import LLMRouter

logger = logging.getLogger(__name__)

# Limit history to last 3 iterations to keep prompt size manageable.
# The debugger can still see patterns across recent repairs.
_MAX_HISTORY_ENTRIES = 3

# Maximum number of tool calls before we force a final_diagnosis. Without this
# cap, the loop could run indefinitely if the LLM keeps requesting tool calls.
_MAX_TOOL_STEPS = 3


async def diagnose_failure(
    state: AgentState,
    router: LLMRouter,
) -> dict[str, Any]:
    """LangGraph node: run a ReAct tool-use loop to diagnose the test failure.

    Args:
        state:  Current AgentState. Reads: iteration, task_description,
                current_code, last_failure_summary, iteration_history,
                degraded_nodes.
        router: LLMRouter. Uses the ``debugger`` role → Claude (strong model).
                The debugger ALWAYS uses Claude (not the weak generator) because
                high-quality root-cause analysis is where the smart model earns
                its keep.

    Returns:
        Partial state: root_cause, failure_category, repair_strategy,
                       diagnosis_confidence, degraded_nodes (if fallback used),
                       events.
    """
    iteration = state.get("iteration", 0)
    events = list(state.get("events", []))

    events.append(step_event(
        "Analyzing failure root cause...",
        iteration=iteration,
    ).to_dict())

    # Limit history to recent entries — too much history causes context overflow
    # and confuses the LLM with irrelevant old failures.
    history = state.get("iteration_history", [])
    relevant_history = history[-_MAX_HISTORY_ENTRIES:] if history else []
    iteration_history_text = _format_history(relevant_history)

    # Base context passed to the LLM on every ReAct step.
    # tool_observations is appended and updated in the loop below.
    base_variables = {
        "task_description": state["task_description"],
        "code": state["current_code"],
        "test_results": state.get("last_failure_summary", "No failure details."),
        "iteration_history": iteration_history_text,
    }

    # ── ReAct loop ────────────────────────────────────────────────────────────
    tool_observations: list[dict[str, Any]] = []  # accumulated tool results
    final_result: dict[str, Any] | None = None    # set when LLM issues final_diagnosis

    for step in range(_MAX_TOOL_STEPS + 1):
        # Inject accumulated tool observations so the LLM can reason over them
        tool_context_text = _format_tool_observations(tool_observations)
        react_variables = {**base_variables, "tool_observations": tool_context_text}

        try:
            # Use get_raw_response (not call) because the debugger's output can
            # be either a tool-use JSON OR a final-diagnosis JSON — we need to
            # inspect it ourselves before deciding which schema to apply.
            raw_text = await router.get_raw_response(
                role="debugger",
                template_key="react_diagnose",
                variables=react_variables,
                max_new_tokens=768,
            )
            parsed = _parse_react_response(raw_text)
        except Exception as exc:
            logger.warning("ReAct step %d: failed to get/parse response: %s", step, exc)
            break  # exit loop, fall through to single-call fallback

        action = parsed.get("action", "final_diagnosis")

        if action == "use_tool" and step < _MAX_TOOL_STEPS:
            # LLM wants to use a tool — execute it and record the observation
            tool_name = parsed.get("tool", "")
            args = parsed.get("args", {})

            logger.info("ReAct tool call: tool=%s args=%s (step=%d)", tool_name, args, step)
            # call_tool is async and safe — it runs in a subprocess if needed
            observation = await call_tool(tool_name, args)

            # Emit a tool-use event for the Gradio timeline — users can see
            # which tools the debugger used and what they found.
            events.append(tool_use_event(
                tool_name=tool_name,
                args=args,
                result=observation,
                iteration=iteration,
            ).to_dict())

            # Append to accumulated observations for the next ReAct iteration
            tool_observations.append({
                "tool": tool_name,
                "args": args,
                "result": observation,
            })
            # Loop continues — LLM gets the new observation on next step

        else:
            # Either "final_diagnosis" action or we hit max tool steps (forced exit).
            # Extract the result if it looks like a diagnosis dict.
            if "root_cause" in parsed:
                final_result = parsed
            break

    # ── Extract diagnosis or fall back to single-call path ────────────────────
    if final_result and "root_cause" in final_result:
        # ReAct loop succeeded — extract fields from the structured response
        root_cause = final_result["root_cause"]
        failure_category = _validate_category(final_result.get("failure_category", "other"))
        repair_strategy = final_result.get("repair_strategy", "Regenerate the solution.")
        confidence = float(final_result.get("confidence", 0.5))
        used_fallback = False
        logger.info(
            "ReAct diagnosis: category=%s confidence=%.2f tool_steps=%d (iteration=%d)",
            failure_category, confidence, len(tool_observations), iteration,
        )
    else:
        # ReAct loop failed to produce a diagnosis — fall back to a single
        # validated LLM call (schema is strictly enforced, returns structured JSON).
        logger.info(
            "ReAct loop produced no final diagnosis — falling back to single-call path."
        )
        fallback_result, used_fallback = await router.call_with_fallback(
            role="debugger",
            template_key="diagnose",
            variables=base_variables,
            max_new_tokens=768,
        )
        root_cause = fallback_result["root_cause"]
        failure_category = _validate_category(fallback_result["failure_category"])
        repair_strategy = fallback_result["repair_strategy"]
        confidence = float(fallback_result.get("confidence", 0.5))

    logger.info(
        "Diagnosis: category=%s confidence=%.2f fallback=%s (iteration=%d)",
        failure_category, confidence, used_fallback, iteration,
    )

    events.append(diagnosis_event(
        root_cause=root_cause,
        category=failure_category,
        strategy=repair_strategy,
        iteration=iteration,
    ).to_dict())

    # Track degraded nodes (fallback used) so operators can audit run quality
    degraded_nodes = list(state.get("degraded_nodes", []))
    if used_fallback:
        degraded_nodes.append("diagnose_failure")

    return {
        "root_cause": root_cause,
        "failure_category": failure_category,
        "repair_strategy": repair_strategy,
        "diagnosis_confidence": confidence,
        "degraded_nodes": degraded_nodes,
        "events": events,
    }


def _parse_react_response(raw_text: str) -> dict[str, Any]:
    """Parse raw LLM output into a structured dict for the ReAct dispatcher.

    Handles three formats:
    1. Clean JSON: ``{"action": "use_tool", ...}``
    2. Legacy / final diagnosis: ``{"root_cause": "...", ...}`` (no action key)
    3. Markdown-fenced JSON: `` ```json { ... } ``` ``
    """
    import re
    text = raw_text.strip()

    # Strip markdown code fences — small models often wrap JSON in ``` blocks
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fence_match:
        text = fence_match.group(1).strip()

    # Find the start of the JSON object (skip any preamble text the LLM added)
    brace_start = text.find("{")
    if brace_start == -1:
        raise ValueError("No JSON object found in response")

    return json.loads(text[brace_start:])


def _format_tool_observations(observations: list[dict[str, Any]]) -> str:
    """Format tool results for injection into the react_diagnose template.

    Truncates each result to 400 chars to keep the prompt size bounded.
    Returns a placeholder if no observations yet.
    """
    if not observations:
        return "None yet."
    lines: list[str] = []
    for i, obs in enumerate(observations, 1):
        lines.append(
            f"Step {i}: {obs['tool']}({obs['args']}) →\n{obs['result'][:400]}"
        )
    return "\n\n".join(lines)


# Canonical set of failure categories the LLM can assign. Anything outside
# this set is coerced to "other" to keep the category field consistent.
_VALID_CATEGORIES = frozenset({
    "logic_error", "boundary_condition", "type_error",
    "missing_case", "off_by_one", "algorithm_error", "performance", "other",
})


def _validate_category(category: str) -> str:
    """Coerce unknown category values to 'other' to keep values canonical."""
    return category if category in _VALID_CATEGORIES else "other"


def _format_history(history: list[dict]) -> str:
    """Summarize recent iteration history for the debugger prompt.

    Provides the LLM with a compact view of what's been tried and whether
    each attempt improved things. The debugger uses this to avoid repeating
    the same diagnosis and to detect oscillation patterns.
    """
    if not history:
        return "No prior iteration history."
    lines = []
    for record in history:
        lines.append(
            f"Iteration {record['iteration']}: "
            f"passed={record['passed']} | "
            f"category={record.get('failure_category', 'unknown')} | "
            f"summary={record.get('failure_summary', '')[:200]}"
        )
    return "\n".join(lines)
