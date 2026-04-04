"""
Parallel repair strategies with tournament selection (Fix 14).

Instead of trying one repair strategy per iteration serially, this module
fans out three strategies simultaneously and selects the best result:

  - minimal_fix: change only the failing line(s)
  - restructure: rewrite the core algorithm keeping the same interface
  - add_guards: add input validation and edge-case guards

Each parallel branch (fan_out_repairs → Send → parallel_generate) generates
a repair candidate and immediately executes it against both test suites.
The select_best_repair node picks the winner using tournament selection:

  Score = (both_pass, spec_pass, -(spec_failures + adv_failures))

Design decision: fan-out is implemented with LangGraph Send() so branches
run concurrently. The parallel_repairs field uses Annotated[list, operator.add]
so each branch appends its result without coordination. select_best_repair
reads the merged list and picks the winner.

Falls back gracefully when parallel_strategies=False — the graph wires
generate_solution directly as before.
"""

import logging
from typing import Any

from langgraph.types import Send

from agent.state import AgentState
from agent.events import step_event, code_generated_event, parallel_repair_event
from sandbox.python_executor import execute
from llm.router import LLMRouter

logger = logging.getLogger(__name__)

# The three repair strategies offered to the LLM in parallel
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
    """
    LangGraph conditional edge: fan out to parallel_generate with different strategies.

    Returns a list of Send() objects — one per strategy. LangGraph executes
    them concurrently; results accumulate via the parallel_repairs reducer.
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
                "repair_strategy": (
                    f"{state.get('repair_strategy', '')}\n\n"
                    f"Approach constraint: {s['instruction']}"
                ),
                "strategy_name": s["name"],
                # Reset parallel_repairs so each branch starts clean
                "parallel_repairs": [],
            },
        )
        for s in _STRATEGIES
    ]


async def parallel_generate(
    state: AgentState,
    router: LLMRouter,
) -> dict[str, Any]:
    """
    LangGraph node: generate one repair candidate and execute it.

    Runs as a parallel branch launched by fan_out_repairs → Send().
    Appends its result to parallel_repairs for select_best_repair to evaluate.
    """
    iteration = state.get("iteration", 0)
    strategy_name = state.get("strategy_name", "unknown")
    events = list(state.get("events", []))

    events.append(step_event(
        f"Parallel strategy '{strategy_name}': generating repair...",
        iteration=iteration,
    ).to_dict())

    # Generate repair using the strategy-augmented prompt
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
        logger.warning(
            "parallel_generate '%s' failed at generation: %s", strategy_name, exc
        )
        candidate_code = state.get("current_code", "")

    events.append(code_generated_event(
        code=candidate_code,
        iteration=iteration,
        explanation=f"Strategy: {strategy_name}",
    ).to_dict())

    # Execute against both test suites
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

    candidate = {
        "strategy_name": strategy_name,
        "code": candidate_code,
        "spec_passed": spec_passed,
        "adv_passed": adv_passed,
        "spec_failures": spec_failures,
        "adv_failures": adv_failures,
    }

    return {
        "parallel_repairs": [candidate],  # appended via operator.add reducer
        "events": events,
    }


def select_best_repair(state: AgentState) -> dict[str, Any]:
    """
    LangGraph node: tournament selection across parallel repair candidates.

    Scoring (higher = better):
      1. Both spec and adversarial tests pass
      2. Spec tests pass (correctness first)
      3. Fewest total failures

    The winning candidate's code replaces current_code.
    parallel_repairs is reset to [] so the next iteration starts fresh.
    """
    candidates = state.get("parallel_repairs", [])

    if not candidates:
        logger.warning("select_best_repair: no candidates — keeping current code")
        return {"parallel_repairs": []}

    def _score(c: dict) -> tuple:
        both = c.get("spec_passed", False) and c.get("adv_passed", False)
        spec = c.get("spec_passed", False)
        failures = -(c.get("spec_failures", 0) + c.get("adv_failures", 0))
        return (both, spec, failures)

    candidates_sorted = sorted(candidates, key=_score, reverse=True)
    best = candidates_sorted[0]

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
        # Reset for next iteration
        "parallel_repairs": [],
    }


def _format_learning_log(lessons: list[str]) -> str:
    if not lessons:
        return "No prior lessons recorded."
    return "\n".join(f"- {lesson}" for lesson in lessons)
