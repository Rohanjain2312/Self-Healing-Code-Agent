"""
Debugger node — ReAct-style root cause analysis with tool use.

The debugger runs a think → act → observe loop before issuing its final
diagnosis. It can use three tools:

  - run_snippet: execute a short Python snippet to verify an edge-case hypothesis
  - inspect_function: parse AST to check function signatures and docstrings
  - diff_iterations: compare code across repair attempts to track progress

When the LLM's raw response contains {"action": "use_tool", ...}, the tool is
invoked and the observation is appended to the tool context. When it outputs
{"action": "final_diagnosis", ...}, the loop ends and the diagnosis is returned.

Design decision: option (b) from the spec — a simple inline loop rather than a
LangGraph sub-graph. This avoids provider dependency on native function calling
(Ollama/llama3 doesn't support it) and keeps the loop entirely within the node.

The loop falls back to the standard single-call path if:
  - The LLM returns the legacy format (root_cause key without action key)
  - JSON parsing fails for any reason
  - Max tool steps are exhausted without a final_diagnosis
"""

import json
import logging
from typing import Any

from agent.state import AgentState
from agent.events import step_event, diagnosis_event, tool_use_event
from agent.tools import call_tool
from llm.router import LLMRouter

logger = logging.getLogger(__name__)

_MAX_HISTORY_ENTRIES = 3  # limit how much history the debugger sees
_MAX_TOOL_STEPS = 3       # max tool calls before forcing final diagnosis


async def diagnose_failure(
    state: AgentState,
    router: LLMRouter,
) -> dict[str, Any]:
    """LangGraph node: ReAct-loop root cause analysis with optional tool use."""
    iteration = state.get("iteration", 0)
    events = list(state.get("events", []))

    events.append(step_event(
        "Analyzing failure root cause...",
        iteration=iteration,
    ).to_dict())

    history = state.get("iteration_history", [])
    relevant_history = history[-_MAX_HISTORY_ENTRIES:] if history else []
    iteration_history_text = _format_history(relevant_history)

    base_variables = {
        "task_description": state["task_description"],
        "code": state["current_code"],
        "test_results": state.get("last_failure_summary", "No failure details."),
        "iteration_history": iteration_history_text,
    }

    # --- ReAct loop ---
    tool_observations: list[dict[str, Any]] = []
    final_result: dict[str, Any] | None = None

    for step in range(_MAX_TOOL_STEPS + 1):
        tool_context_text = _format_tool_observations(tool_observations)
        react_variables = {**base_variables, "tool_observations": tool_context_text}

        try:
            raw_text = await router.get_raw_response(
                role="debugger",
                template_key="react_diagnose",
                variables=react_variables,
                max_new_tokens=768,
            )
            parsed = _parse_react_response(raw_text)
        except Exception as exc:
            logger.warning("ReAct step %d: failed to get/parse response: %s", step, exc)
            break

        action = parsed.get("action", "final_diagnosis")

        if action == "use_tool" and step < _MAX_TOOL_STEPS:
            tool_name = parsed.get("tool", "")
            args = parsed.get("args", {})

            logger.info("ReAct tool call: tool=%s args=%s (step=%d)", tool_name, args, step)
            observation = await call_tool(tool_name, args)

            events.append(tool_use_event(
                tool_name=tool_name,
                args=args,
                result=observation,
                iteration=iteration,
            ).to_dict())

            tool_observations.append({
                "tool": tool_name,
                "args": args,
                "result": observation,
            })
        else:
            # "final_diagnosis" or legacy format (root_cause key without action)
            if "root_cause" in parsed:
                final_result = parsed
            break

    # --- Extract diagnosis from ReAct result or fall back to single-call path ---
    if final_result and "root_cause" in final_result:
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
        # Fallback: single LLM call with standard validated schema
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
    """
    Parse raw LLM output into a structured dict.

    Handles:
    - Clean JSON: {"action": "use_tool", ...} or {"action": "final_diagnosis", ...}
    - Legacy format: {"root_cause": "...", ...} (no action key — treated as final_diagnosis)
    - Markdown-fenced JSON (```json ... ```)
    """
    import re
    text = raw_text.strip()

    # Strip markdown fences if present
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fence_match:
        text = fence_match.group(1).strip()

    # Find first JSON object
    brace_start = text.find("{")
    if brace_start == -1:
        raise ValueError("No JSON object found in response")

    # Try to parse from brace_start onwards
    return json.loads(text[brace_start:])


def _format_tool_observations(observations: list[dict[str, Any]]) -> str:
    """Format tool observations for injection into the react_diagnose template."""
    if not observations:
        return "None yet."
    lines: list[str] = []
    for i, obs in enumerate(observations, 1):
        lines.append(
            f"Step {i}: {obs['tool']}({obs['args']}) →\n{obs['result'][:400]}"
        )
    return "\n\n".join(lines)


_VALID_CATEGORIES = frozenset({
    "logic_error", "boundary_condition", "type_error",
    "missing_case", "off_by_one", "algorithm_error", "performance", "other",
})


def _validate_category(category: str) -> str:
    """Coerce unknown category values to 'other'."""
    return category if category in _VALID_CATEGORIES else "other"


def _format_history(history: list[dict]) -> str:
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
