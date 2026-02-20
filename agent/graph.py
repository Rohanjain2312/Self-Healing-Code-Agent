"""
LangGraph state machine definition for the Self-Healing Code Agent.

Graph topology:
  generate_solution
       ↓
  create_adversarial_tests
       ↓
  execute_solution
       ↓ (pass) ────────────────────────→ END
       ↓ (fail, iterations remaining)
  diagnose_failure
       ↓
  update_learning_log
       ↓
  increment_iteration
       ↓
  generate_solution  (repair cycle)

Node functions accept state + router to allow dependency injection in tests.
The router is bound via functools.partial before nodes are added to the graph.
"""

import functools
import logging
from typing import Any, Literal, AsyncGenerator

from langgraph.graph import StateGraph, END

from agent.state import AgentState
from agent.nodes.generate_solution import generate_solution
from agent.nodes.create_adversarial_tests import create_adversarial_tests
from agent.nodes.execute_solution import execute_solution
from agent.nodes.diagnose_failure import diagnose_failure
from agent.nodes.update_learning_log import update_learning_log
from llm.router import LLMRouter

logger = logging.getLogger(__name__)


def _increment_iteration(state: AgentState) -> dict[str, Any]:
    """
    Increment iteration counter before each repair cycle.
    Also checks if max_iterations is reached and sets terminal status.
    """
    new_iteration = state.get("iteration", 0) + 1
    max_iter = state.get("max_iterations", 4)

    if new_iteration >= max_iter:
        logger.warning(
            "Max iterations (%d) reached. Terminating repair loop.",
            max_iter,
        )
        return {"iteration": new_iteration, "status": "max_iterations_reached"}

    return {"iteration": new_iteration, "status": "running"}


def _route_after_execution(
    state: AgentState,
) -> Literal["diagnose_failure", "__end__", "max_iterations"]:
    """
    Conditional edge: determine next node after test execution.

    - Pass → END
    - Fail + iterations remaining → diagnose_failure
    - Fail + max iterations → END (with failed status)
    """
    if state.get("last_execution_passed"):
        return "__end__"

    if state.get("status") == "max_iterations_reached":
        return "max_iterations"

    iteration = state.get("iteration", 0)
    max_iter = state.get("max_iterations", 4)
    if iteration >= max_iter:
        return "max_iterations"

    return "diagnose_failure"


def _route_after_increment(
    state: AgentState,
) -> Literal["generate_solution", "__end__"]:
    """Route after iteration increment — stops if max reached."""
    if state.get("status") == "max_iterations_reached":
        return "__end__"
    return "generate_solution"


def build_graph(router: LLMRouter | None = None) -> StateGraph:
    """
    Construct and compile the agent state graph.

    Args:
        router: Optional pre-constructed LLMRouter. If None, auto-resolved.

    Returns:
        Compiled LangGraph StateGraph ready for invocation.
    """
    if router is None:
        router = LLMRouter()

    # Bind router to all nodes that require it (partial application)
    _generate = functools.partial(generate_solution, router=router)
    _qa = functools.partial(create_adversarial_tests, router=router)
    _diagnose = functools.partial(diagnose_failure, router=router)
    _memory = functools.partial(update_learning_log, router=router)

    graph = StateGraph(AgentState)

    # Register nodes
    graph.add_node("generate_solution", _generate)
    graph.add_node("create_adversarial_tests", _qa)
    graph.add_node("execute_solution", execute_solution)
    graph.add_node("diagnose_failure", _diagnose)
    graph.add_node("update_learning_log", _memory)
    graph.add_node("increment_iteration", _increment_iteration)
    # Absorbing terminal node for max_iterations path
    graph.add_node("max_iterations", lambda s: {"status": "max_iterations_reached"})

    # Linear flow: generate → qa → execute
    graph.set_entry_point("generate_solution")
    graph.add_edge("generate_solution", "create_adversarial_tests")
    graph.add_edge("create_adversarial_tests", "execute_solution")

    # Conditional routing after execution
    graph.add_conditional_edges(
        "execute_solution",
        _route_after_execution,
        {
            "__end__": END,
            "diagnose_failure": "diagnose_failure",
            "max_iterations": "max_iterations",
        },
    )

    # Repair loop: diagnose → memory → increment → generate
    graph.add_edge("diagnose_failure", "update_learning_log")
    graph.add_edge("update_learning_log", "increment_iteration")

    graph.add_conditional_edges(
        "increment_iteration",
        _route_after_increment,
        {
            "generate_solution": "generate_solution",
            "__end__": END,
        },
    )

    graph.add_edge("max_iterations", END)

    return graph.compile()


def _make_initial_state(
    task_description: str,
    max_iterations: int = 4,
) -> AgentState:
    """Construct a clean initial state for a new task."""
    return AgentState(
        task_description=task_description,
        max_iterations=max_iterations,
        current_code="",
        current_test_code="",
        last_execution_passed=False,
        last_failure_summary="",
        root_cause="",
        failure_category="",
        repair_strategy="",
        learning_log=[],
        iteration=0,
        iteration_history=[],
        status="running",
        events=[],
    )


async def run_agent(
    task_description: str,
    max_iterations: int = 4,
    router: LLMRouter | None = None,
) -> AgentState:
    """
    High-level entry point: run the agent to completion and return final state.

    For streaming use cases, use stream_agent() instead.
    """
    app = build_graph(router=router)
    initial_state = _make_initial_state(task_description, max_iterations)
    final_state = await app.ainvoke(initial_state)
    return final_state


async def stream_agent(
    task_description: str,
    max_iterations: int = 4,
    router: LLMRouter | None = None,
) -> AsyncGenerator[dict[str, Any], None]:
    """
    Stream agent events as they are produced by each node.

    Yields event dicts from AgentState.events. The caller receives events
    in real-time rather than waiting for the full run to complete.
    This is consumed by the Gradio demo for live UI updates.
    """
    app = build_graph(router=router)
    initial_state = _make_initial_state(task_description, max_iterations)

    # Track the total number of events seen across all node updates.
    # Each node returns the FULL accumulated events list in its state slice,
    # so we track the global count and yield only genuinely new events.
    total_seen = 0

    async for state_update in app.astream(initial_state):
        # astream yields {node_name: partial_state} for each completed node
        for node_name, node_state in state_update.items():
            if not isinstance(node_state, dict):
                continue
            events = node_state.get("events", [])
            if not events:
                continue
            # events list grows cumulatively; yield only the tail we haven't seen
            new_events = events[total_seen:]
            total_seen = len(events)
            for event in new_events:
                yield event
