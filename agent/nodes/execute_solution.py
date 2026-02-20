"""
Execution node — runs solution code against adversarial tests in the sandbox.

This is the only node that executes code. It delegates to the sandbox module
which runs everything in a separate subprocess with a timeout.

The result determines whether the graph routes to success or the debug loop.
"""

import logging
from typing import Any

from agent.state import AgentState, IterationRecord
from agent.events import step_event, failure_event, success_event
from sandbox.python_executor import execute, format_failure_summary

logger = logging.getLogger(__name__)


async def execute_solution(state: AgentState) -> dict[str, Any]:
    """LangGraph node: execute solution + adversarial tests in sandbox."""
    iteration = state.get("iteration", 0)
    events = list(state.get("events", []))

    events.append(step_event(
        "Executing solution against adversarial tests...",
        iteration=iteration,
    ).to_dict())

    result = await execute(
        solution_code=state["current_code"],
        test_code=state["current_test_code"],
    )

    failure_summary = format_failure_summary(result)

    logger.info(
        "Execution complete: passed=%s elapsed=%.2fs (iteration=%d)",
        result.passed,
        result.elapsed_seconds,
        iteration,
    )

    if result.passed:
        events.append(success_event(
            code=state["current_code"],
            iteration=iteration,
        ).to_dict())
        new_status = "success"
    else:
        events.append(failure_event(
            summary=failure_summary,
            iteration=iteration,
            failed_assertions=result.failed_assertions,
        ).to_dict())
        new_status = "running"

    # Record this iteration for the debugger's history context
    iteration_record: IterationRecord = {
        "iteration": iteration,
        "code": state["current_code"],
        "test_code": state["current_test_code"],
        "passed": result.passed,
        "failure_summary": failure_summary,
        "root_cause": state.get("root_cause", ""),
        "failure_category": state.get("failure_category", ""),
        "repair_strategy": state.get("repair_strategy", ""),
    }
    history = list(state.get("iteration_history", []))
    history.append(iteration_record)

    return {
        "last_execution_passed": result.passed,
        "last_failure_summary": failure_summary,
        "iteration_history": history,
        "status": new_status,
        "events": events,
    }
