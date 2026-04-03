"""
LangGraph state machine definition for the Self-Healing Code Agent.

Target graph topology (with all features enabled via AgentConfig):

  [generate_spec_tests]  ← once at iteration 0 if enable_spec_tests
          ↓
  generate_solution
          ↓
  create_adversarial_tests
          ↓
  execute_solution  (runs BOTH spec + adversarial tests)
          ↓ (pass) ──────────────────────────────────→ END
          ↓ (fail, iterations remaining)
  diagnose_failure  ← uses call_with_fallback for graceful degradation
          ↓
  update_learning_log
          ↓
  [review_repair]  ← HITL interrupt() when autonomy_level != full_auto
          ↓
  increment_iteration
          ↓ (max reached) ──────────────────────────→ END
          ↓
  generate_solution  (repair cycle)

When AgentConfig.development() is used (or no config provided), the graph
collapses back to the original linear topology with no HITL, no spec tests,
no checkpointing — exactly matching the pre-upgrade behavior.

Node functions that require the LLM router are bound via functools.partial
before registration, keeping nodes pure functions for easy unit testing.
"""

import functools
import logging
import uuid
from typing import Any, Literal, AsyncGenerator

from langgraph.graph import StateGraph, END

from agent.config import AgentConfig
from agent.state import AgentState
from agent.nodes.generate_solution import generate_solution
from agent.nodes.create_adversarial_tests import create_adversarial_tests
from agent.nodes.execute_solution import execute_solution
from agent.nodes.diagnose_failure import diagnose_failure
from agent.nodes.update_learning_log import update_learning_log
from agent.nodes.generate_spec_tests import generate_spec_tests
from agent.nodes.review_repair import review_repair
from llm.router import LLMRouter

logger = logging.getLogger(__name__)

_LOW_CONFIDENCE_THRESHOLD = 0.3  # below this → blind retry instead of targeted repair


def _increment_iteration(state: AgentState) -> dict[str, Any]:
    """
    Increment iteration counter before each repair cycle.
    Also sets terminal status when max_iterations is reached.
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
    - Fail + max iterations exhausted → terminal END
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


def _route_after_diagnosis(
    state: AgentState,
) -> Literal["update_learning_log", "generate_solution"]:
    """
    Confidence-aware routing after diagnosis (Fix 7).

    If the debugger's confidence is below the threshold, skip targeted repair
    and regenerate from scratch — a blind retry is better than acting on a
    low-quality diagnosis.
    """
    confidence = state.get("diagnosis_confidence", 1.0)
    if confidence < _LOW_CONFIDENCE_THRESHOLD:
        logger.info(
            "Low diagnosis confidence (%.2f < %.2f) — routing to blind retry.",
            confidence,
            _LOW_CONFIDENCE_THRESHOLD,
        )
        return "generate_solution"
    return "update_learning_log"


def _route_after_increment(
    state: AgentState,
) -> Literal["generate_solution", "__end__"]:
    """Route after iteration increment — stops if max reached."""
    if state.get("status") == "max_iterations_reached":
        return "__end__"
    return "generate_solution"


def build_graph(
    router: LLMRouter | None = None,
    config: AgentConfig | None = None,
) -> Any:
    """
    Construct and compile the agent state graph.

    Args:
        router: Optional pre-constructed LLMRouter. If None, auto-resolved.
        config: Optional AgentConfig controlling feature flags.
                Defaults to AgentConfig.development() (original behavior).

    Returns:
        Compiled LangGraph StateGraph (CompiledGraph) ready for ainvoke/astream.
    """
    if router is None:
        router = LLMRouter()
    if config is None:
        config = AgentConfig.development()

    # Bind router to all nodes that require it (partial application)
    _generate = functools.partial(generate_solution, router=router)
    _qa = functools.partial(create_adversarial_tests, router=router)
    _diagnose = functools.partial(diagnose_failure, router=router)
    _memory = functools.partial(update_learning_log, router=router)
    _spec_tests = functools.partial(generate_spec_tests, router=router)

    graph = StateGraph(AgentState)

    # ── Core nodes (always present) ──────────────────────────────────────────
    graph.add_node("generate_solution", _generate)
    graph.add_node("create_adversarial_tests", _qa)
    graph.add_node("execute_solution", execute_solution)
    graph.add_node("diagnose_failure", _diagnose)
    graph.add_node("update_learning_log", _memory)
    graph.add_node("increment_iteration", _increment_iteration)
    graph.add_node("max_iterations", lambda s: {"status": "max_iterations_reached"})

    # ── Optional: spec test node (Fix 5) ─────────────────────────────────────
    if config.enable_spec_tests:
        graph.add_node("generate_spec_tests", _spec_tests)
        graph.set_entry_point("generate_spec_tests")
        graph.add_edge("generate_spec_tests", "generate_solution")
    else:
        graph.set_entry_point("generate_solution")

    # ── Linear flow: generate → qa → execute ─────────────────────────────────
    graph.add_edge("generate_solution", "create_adversarial_tests")
    graph.add_edge("create_adversarial_tests", "execute_solution")

    # ── Conditional routing after execution ──────────────────────────────────
    graph.add_conditional_edges(
        "execute_solution",
        _route_after_execution,
        {
            "__end__": END,
            "diagnose_failure": "diagnose_failure",
            "max_iterations": "max_iterations",
        },
    )

    # ── Confidence-aware routing after diagnosis (Fix 7) ─────────────────────
    graph.add_conditional_edges(
        "diagnose_failure",
        _route_after_diagnosis,
        {
            "update_learning_log": "update_learning_log",
            "generate_solution": "generate_solution",  # blind retry path
        },
    )

    # ── Optional: HITL review node (Fix 12) ──────────────────────────────────
    if config.autonomy_level != "full_auto":
        graph.add_node("review_repair", review_repair)
        graph.add_edge("update_learning_log", "review_repair")
        graph.add_edge("review_repair", "increment_iteration")
    else:
        graph.add_edge("update_learning_log", "increment_iteration")

    # ── Repair loop completion ────────────────────────────────────────────────
    graph.add_conditional_edges(
        "increment_iteration",
        _route_after_increment,
        {
            "generate_solution": "generate_solution",
            "__end__": END,
        },
    )
    graph.add_edge("max_iterations", END)

    # ── Checkpointer (Fix 13) ─────────────────────────────────────────────────
    # A checkpointer is required for interrupt() to work. Always attach one
    # when HITL is enabled; also attach in full_auto if enable_checkpointing.
    checkpointer = None
    needs_checkpointer = (
        config.autonomy_level != "full_auto" or config.enable_checkpointing
    )
    if needs_checkpointer:
        if config.persist_checkpoints:
            try:
                from langgraph.checkpoint.sqlite import SqliteSaver
                checkpointer = SqliteSaver.from_conn_string(".agent_checkpoints.db")
                logger.info("Using SqliteSaver checkpointer (.agent_checkpoints.db)")
            except (ImportError, Exception) as exc:
                logger.warning(
                    "SqliteSaver unavailable (%s) — falling back to InMemorySaver.", exc
                )
                from langgraph.checkpoint.memory import InMemorySaver
                checkpointer = InMemorySaver()
        else:
            from langgraph.checkpoint.memory import InMemorySaver
            checkpointer = InMemorySaver()
            logger.info("Using InMemorySaver checkpointer")

    return graph.compile(checkpointer=checkpointer)


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
        spec_test_code="",
        last_execution_passed=False,
        last_failure_summary="",
        root_cause="",
        failure_category="",
        repair_strategy="",
        diagnosis_confidence=0.5,
        learning_log=[],
        iteration=0,
        iteration_history=[],
        status="running",
        degraded_nodes=[],
        events=[],
    )


async def run_agent(
    task_description: str,
    max_iterations: int = 4,
    router: LLMRouter | None = None,
    config: AgentConfig | None = None,
) -> AgentState:
    """
    High-level entry point: run the agent to completion and return final state.

    config defaults to AgentConfig.development() — identical to pre-upgrade
    behavior — so all existing call sites continue to work without changes.

    For streaming use cases, use stream_agent() instead.
    """
    app = build_graph(router=router, config=config)
    initial_state = _make_initial_state(task_description, max_iterations)
    thread_config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    final_state = await app.ainvoke(initial_state, thread_config)
    return final_state


async def stream_agent(
    task_description: str,
    max_iterations: int = 4,
    router: LLMRouter | None = None,
    config: AgentConfig | None = None,
) -> AsyncGenerator[dict[str, Any], None]:
    """
    Stream agent events as they are produced by each node.

    Yields event dicts from AgentState.events. The caller receives events
    in real-time rather than waiting for the full run to complete.
    Consumed by the Gradio demo for live UI updates.
    """
    app = build_graph(router=router, config=config)
    initial_state = _make_initial_state(task_description, max_iterations)
    thread_config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    # Track total events seen to yield only genuinely new ones per node update
    total_seen = 0

    async for state_update in app.astream(initial_state, thread_config):
        for node_name, node_state in state_update.items():
            if not isinstance(node_state, dict):
                continue
            events = node_state.get("events", [])
            if not events:
                continue
            new_events = events[total_seen:]
            total_seen = len(events)
            for event in new_events:
                yield event


async def run_agent_with_history(
    task_description: str,
    config: AgentConfig,
    router: LLMRouter | None = None,
) -> tuple[AgentState, list[Any]]:
    """
    Run the agent and return both the final state and full checkpoint history.

    The checkpoint history can be used for time-travel debugging: inspect any
    intermediate state, modify it, and fork_from_iteration() to replay forward.

    Requires enable_checkpointing=True in config (or HITL enabled).
    """
    app = build_graph(router=router, config=config)
    thread_config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    initial_state = _make_initial_state(task_description, config.max_iterations)
    final_state = await app.ainvoke(initial_state, thread_config)
    history = list(app.get_state_history(thread_config))
    return final_state, history


async def fork_from_iteration(
    app: Any,
    thread_config: dict[str, Any],
    target_iteration: int,
    modified_state: dict[str, Any],
) -> AgentState:
    """
    Rewind to a previous checkpoint and replay with modified state.

    Time-travel debugging: find the checkpoint where iteration == target_iteration,
    apply modified_state overrides, then resume execution from diagnose_failure.

    Args:
        app: Compiled graph returned by build_graph() (must have a checkpointer).
        thread_config: Thread config dict used for the original run.
        target_iteration: Iteration number to rewind to.
        modified_state: Partial state overrides to apply at the rewound checkpoint.

    Returns:
        Final state after replaying from the fork point.

    Raises:
        ValueError: If no checkpoint exists for the given iteration.
    """
    history = list(app.get_state_history(thread_config))
    target = next(
        (s for s in history if s.values.get("iteration") == target_iteration),
        None,
    )
    if target is None:
        raise ValueError(
            f"No checkpoint found for iteration={target_iteration}. "
            f"Available iterations: {sorted({s.values.get('iteration') for s in history})}"
        )
    fork_config = app.update_state(
        target.config, values=modified_state, as_node="diagnose_failure"
    )
    return await app.ainvoke(None, fork_config)
