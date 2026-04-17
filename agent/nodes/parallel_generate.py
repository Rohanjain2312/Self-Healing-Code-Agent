"""
Parallel repair strategies with tournament selection (Fix 14).

WHAT THIS MODULE DOES
---------------------
When ``config.parallel_strategies=True``, instead of applying one repair
strategy per iteration serially, this module fans out THREE strategies
simultaneously and picks the best result via tournament selection.

The three strategies are:
  1. minimal_fix   — change only the failing line(s); minimal diff
  2. restructure   — rewrite the core algorithm from scratch; same interface
  3. add_guards    — add input validation and edge-case guards

WHY THREE STRATEGIES?
----------------------
Different failure modes respond better to different repair approaches:
  - A logic error in one branch → minimal_fix is best (surgical change)
  - A fundamentally wrong algorithm → restructure is best (start fresh)
  - An edge case (empty input, None, boundary) → add_guards is best (defensive code)

Running all three in parallel means we almost always have at least one
strategy that's appropriate, without needing the debugger to correctly
predict which one to use.

LANGGRAPH SEND() FAN-OUT
------------------------
fan_out_repairs() returns a list of ``Send("parallel_generate", {...})``
objects. LangGraph interprets a list of Send objects as a concurrent fan-out:
it launches all of them simultaneously, collects their results in
state["parallel_repairs"] (merged via operator.add reducer), then proceeds
to select_best_repair when ALL branches complete.

TOURNAMENT SELECTION
--------------------
select_best_repair ranks candidates by:
  (both_pass, spec_pass, -total_failures)

This prioritizes full correctness (both suites pass) > spec correctness >
fewest failures. The winner's code becomes current_code for the next iteration.

WHEN IS THIS DISABLED?
-----------------------
config.parallel_strategies defaults to False. It triples LLM cost per repair
iteration and adds complexity to the graph topology. Only enable it if you
have budget and the serial repair loop is getting stuck.
"""

import logging
from typing import Any

from langgraph.types import Send

from agent.state import AgentState
from agent.events import step_event, code_generated_event, parallel_repair_event
from sandbox.python_executor import execute
from llm.router import LLMRouter

logger = logging.getLogger(__name__)

# The three strategies injected as instruction constraints into the repair prompt.
# Each strategy constrains HOW the generator should repair the code — the base
# diagnosis (root_cause, repair_strategy from diagnose_failure) is also included
# so each branch still has the debugger's guidance.
_STRATEGIES = [
    {
        "name": "minimal_fix",
        "instruction": "Change ONLY the failing line(s). Do not restructure or rename. Minimal diff.",
    },
    {
        "name": "restructure",
        "instruction": "Rewrite the core algorithm from scratch keeping the exact same function signature and interface.",
    },
    {
        "name": "add_guards",
        "instruction": "Add input validation and edge-case guards at the start of the function without changing core logic.",
    },
]


def fan_out_repairs(state: AgentState) -> list[Send]:
    """LangGraph conditional edge: create parallel repair branches via Send().

    Called by _route_after_increment_parallel when iteration < max_iterations.
    Returns a list of Send objects — LangGraph runs them concurrently.

    Each Send carries a copy of the current state with two overrides:
      - repair_strategy: original diagnosis + strategy-specific constraint
      - strategy_name:   which strategy this branch is running

    Args:
        state: Current AgentState. Used to build the per-branch state dicts.

    Returns:
        List of Send("parallel_generate", modified_state) objects, one per strategy.
    """
    logger.info(
        "Fanning out %d parallel repair strategies (iteration=%d)",
        len(_STRATEGIES),
        state.get("iteration", 0),
    )
    return [
        Send(
            "parallel_generate",
            {
                **state,
                # Augment the base repair_strategy with the strategy-specific
                # instruction. The generator sees BOTH the debugger's diagnosis
                # AND the strategy constraint in its prompt.
                "repair_strategy": (
                    f"{state.get('repair_strategy', '')}\n\n"
                    f"Approach constraint: {s['instruction']}"
                ),
                "strategy_name": s["name"],
                # Reset parallel_repairs in each branch's state copy so each
                # branch starts with an empty list. The operator.add reducer
                # then merges all branches' [candidate] lists into one.
                "parallel_repairs": [],
            },
        )
        for s in _STRATEGIES
    ]


async def parallel_generate(
    state: AgentState,
    router: LLMRouter,
) -> dict[str, Any]:
    """LangGraph node: generate one repair candidate and immediately test it.

    Runs as a parallel branch, launched by fan_out_repairs()'s Send() objects.
    Each branch generates repair code and tests it, then appends one candidate
    dict to parallel_repairs (via operator.add — no coordination needed).

    Args:
        state:  Branch's state dict (copy of AgentState with strategy overrides).
        router: LLMRouter. Uses ``generator`` role → weak model (HF/Ollama).

    Returns:
        Partial state: {"parallel_repairs": [candidate_dict], "events": [...]}.
        The single-element list is merged with other branches via operator.add.
    """
    iteration = state.get("iteration", 0)
    strategy_name = state.get("strategy_name", "unknown")
    events = list(state.get("events", []))

    events.append(step_event(
        f"Parallel strategy '{strategy_name}': generating repair...",
        iteration=iteration,
    ).to_dict())

    # Build repair variables — same as generate_solution's repair template,
    # but repair_strategy has the strategy-specific instruction appended.
    learning_log = _format_learning_log(state.get("learning_log", []))
    variables = {
        "task_description": state["task_description"],
        "current_code": state["current_code"],
        "test_results": state.get("last_failure_summary", "No failure details."),
        "root_cause": state.get("root_cause", "Unknown"),
        "repair_strategy": state.get("repair_strategy", ""),
        "learning_log": learning_log,
    }

    try:
        result = await router.call(
            role="generator",
            template_key="repair",
            variables=variables,
            max_new_tokens=2048,
        )
        candidate_code = result["code"]
    except Exception as exc:
        # If this branch fails to generate code (schema error, timeout, etc.),
        # fall back to the current code so we still have a candidate to score.
        logger.warning(
            "parallel_generate '%s' failed at generation: %s", strategy_name, exc
        )
        candidate_code = state.get("current_code", "")

    events.append(code_generated_event(
        code=candidate_code,
        iteration=iteration,
        explanation=f"Strategy: {strategy_name}",
    ).to_dict())

    # ── Test the candidate immediately in this branch ─────────────────────────
    # Each branch runs its own test suite in the sandbox so select_best_repair
    # has real test results to compare, not just the code itself.
    spec_test_code = state.get("spec_test_code", "")
    adversarial_test_code = state.get("current_test_code", "")

    spec_passed = True
    spec_failures = 0
    if spec_test_code.strip():
        spec_result = await execute(solution_code=candidate_code, test_code=spec_test_code)
        spec_passed = spec_result.passed
        spec_failures = len(spec_result.failed_assertions)
        logger.info(
            "Parallel[%s] spec: passed=%s (iteration=%d)",
            strategy_name, spec_passed, iteration,
        )

    adv_result = await execute(solution_code=candidate_code, test_code=adversarial_test_code)
    adv_passed = adv_result.passed
    adv_failures = len(adv_result.failed_assertions)
    logger.info(
        "Parallel[%s] adv: passed=%s (iteration=%d)",
        strategy_name, adv_passed, iteration,
    )

    events.append(parallel_repair_event(
        strategy_name=strategy_name,
        spec_passed=spec_passed,
        adv_passed=adv_passed,
        iteration=iteration,
    ).to_dict())

    # Build the candidate record that select_best_repair will score
    candidate = {
        "strategy_name": strategy_name,
        "code": candidate_code,
        "spec_passed": spec_passed,
        "adv_passed": adv_passed,
        "spec_failures": spec_failures,
        "adv_failures": adv_failures,
    }

    # Return a single-element list. The operator.add reducer in AgentState
    # concatenates all branches' lists: [[c1], [c2], [c3]] → [c1, c2, c3].
    return {
        "parallel_repairs": [candidate],
        "events": events,
    }


def select_best_repair(state: AgentState) -> dict[str, Any]:
    """LangGraph node: tournament selection across parallel repair candidates.

    Runs after ALL parallel_generate branches complete (LangGraph waits for
    all incoming edges before running a fan-in node).

    Scoring (lexicographic, higher = better):
      1. both_pass: both spec AND adversarial tests pass (1 > 0)
      2. spec_pass: at least spec tests pass (1 > 0)
      3. -total_failures: fewer failures is better

    Args:
        state: Current AgentState. Reads parallel_repairs (merged list).

    Returns:
        Partial state: current_code (winner), last_execution_passed,
                       strategy_name (winner's), parallel_repairs (cleared).
    """
    candidates = state.get("parallel_repairs", [])

    if not candidates:
        # Should not happen — all 3 branches always produce a candidate —
        # but handle gracefully just in case.
        logger.warning("select_best_repair: no candidates — keeping current code")
        return {"parallel_repairs": []}

    def _score(c: dict) -> tuple:
        both = c.get("spec_passed", False) and c.get("adv_passed", False)
        spec = c.get("spec_passed", False)
        # Negative total failures — fewer failures = higher (less negative) score
        failures = -(c.get("spec_failures", 0) + c.get("adv_failures", 0))
        return (both, spec, failures)

    candidates_sorted = sorted(candidates, key=_score, reverse=True)
    best = candidates_sorted[0]  # highest-scoring candidate

    logger.info(
        "Tournament selection: winner='%s' spec=%s adv=%s (from %d candidates)",
        best["strategy_name"],
        best["spec_passed"],
        best["adv_passed"],
        len(candidates),
    )

    overall_passed = best["spec_passed"] and best["adv_passed"]

    return {
        "current_code": best["code"],
        "last_execution_passed": overall_passed,
        "strategy_name": best["strategy_name"],
        "parallel_repairs": [],  # reset for the next iteration
    }


def _format_learning_log(lessons: list[str]) -> str:
    """Format lesson list as bullet points for prompt injection."""
    if not lessons:
        return "No prior lessons recorded."
    return "\n".join(f"- {lesson}" for lesson in lessons)
