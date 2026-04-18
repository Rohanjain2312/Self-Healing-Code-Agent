"""
LangGraph state machine definition for the Self-Healing Code Agent.

HOW THE GRAPH IS BUILT
----------------------
LangGraph works like a directed graph where each node is an async function
that receives the current AgentState and returns a partial dict of updates.
Edges define which node runs next; conditional edges run a router function
to decide at runtime.

``build_graph(config, router)`` constructs and COMPILES the graph. The
compiled object (``app``) has two key methods:
  - ``app.ainvoke(initial_state, thread_config)``  → runs to completion
  - ``app.astream(initial_state, thread_config)``  → streams state after each node

FULL GRAPH TOPOLOGY (all features enabled)
-------------------------------------------

  __start__ ──┬──→ generate_spec_tests ──────────────────────┐
              │                                               ↓
              └──→ generate_solution ──────────────────→ create_adversarial_tests
                                                              ↓
                                                       execute_solution
                                                         │         │
                                                    (pass)         (fail / max-iter)
                                                       ↓                 ↓
                                               critic_review      diagnose_failure
                                               │         │               │
                                           (approve)  (reject)    (confidence ≥ 0.3)
                                               ↓         ↓               ↓
                                              END  diagnose_failure  update_learning_log
                                                                          ↓
                                                                    [review_repair]  ← HITL if not full_auto
                                                                          ↓
                                                                  increment_iteration
                                                                    │           │
                                                               (max-iter)   (< max)
                                                                  ↓             ↓
                                                                 END    generate_solution (repair)

KEY DESIGN DECISIONS
--------------------
- ``functools.partial`` binds the router and lesson_store to every node at
  construction time. Nodes are pure functions — no global state.
- Conditional edge factories (``_make_route_after_execution``) return closures
  so the edge logic can capture ``config`` without embedding it in node state.
- The HITL review_repair node is only added when ``autonomy_level != "full_auto"``.
  On HF Spaces the entry point (app.py) forces full_auto.
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
from agent.nodes.critic_review import critic_review
from agent.nodes.parallel_generate import fan_out_repairs, parallel_generate, select_best_repair
from llm.router import LLMRouter

logger = logging.getLogger(__name__)

# Confidence threshold for the blind-retry routing path. If the debugger
# returns confidence < 0.3 we don't trust its diagnosis enough to apply a
# targeted repair — we discard the diagnosis and regenerate from scratch.
# This prevents confidently-wrong diagnoses from locking the agent into a
# bad repair direction across multiple iterations.
_LOW_CONFIDENCE_THRESHOLD = 0.3


def _increment_iteration(state: AgentState) -> dict[str, Any]:
    """Pure helper node — increment iteration counter and set terminal status.

    This is registered as a node (not an edge) so it appears in the graph
    diagram and in streaming events. It does no LLM work — just arithmetic
    and a status flag update.

    Called after: update_learning_log (or review_repair when HITL enabled).
    Routes to: generate_solution (if iterations remain) or END (if max hit).
    """
    new_iteration = state.get("iteration", 0) + 1
    max_iter = state.get("max_iterations", 4)

    if new_iteration >= max_iter:
        # The routing function _route_after_increment reads this status flag
        # and returns "__end__" → graph terminates on the next edge resolution.
        logger.warning(
            "Max iterations (%d) reached. Terminating repair loop.",
            max_iter,
        )
        return {"iteration": new_iteration, "status": "max_iterations_reached"}

    # Still within budget — signal the routing function to continue to
    # generate_solution for another repair attempt.
    return {"iteration": new_iteration, "status": "running"}


def _make_route_after_execution(enable_critic: bool):
    """Factory that produces the conditional edge function for execute_solution.

    WHY A FACTORY?
    LangGraph's conditional edge functions must accept only ``state`` (no extra
    args). We need to know whether the critic is enabled at routing time, but
    we don't want to embed that in AgentState. The factory captures ``enable_critic``
    in a closure so the returned function satisfies LangGraph's signature.
    """
    def _route_after_execution(
        state: AgentState,
    ) -> Literal["critic_review", "diagnose_failure", "__end__", "max_iterations"]:
        """Decide what to do after execute_solution completes.

        Decision tree:
          1. All tests passed + critic enabled  → critic_review (extra sanity check)
          2. All tests passed + no critic       → END immediately
          3. Tests failed + max iterations hit  → END via max_iterations terminal node
          4. Tests failed + iterations remain   → diagnose_failure to start ReAct loop
        """
        if state.get("last_execution_passed"):
            # Tests passed — either run the critic or end immediately
            return "critic_review" if enable_critic else "__end__"

        # Tests failed — check if we've exhausted our iteration budget.
        # The status may have been set to "max_iterations_reached" by a previous
        # call to _increment_iteration, or we might still be at iteration 0
        # (first attempt). Check both the status flag and the raw counter.
        if state.get("status") == "max_iterations_reached":
            return "max_iterations"

        iteration = state.get("iteration", 0)
        max_iter = state.get("max_iterations", 4)
        if iteration >= max_iter:
            return "max_iterations"

        return "diagnose_failure"

    return _route_after_execution


def _route_after_critic(
    state: AgentState,
) -> Literal["diagnose_failure", "__end__"]:
    """Conditional edge after critic_review.

    The critic node sets last_execution_passed=False if it finds issues the
    tests missed. We route to diagnose_failure in that case so the repair loop
    re-engages on the critic's feedback, even though the automated tests passed.

    This catches "technically correct but semantically wrong" solutions — e.g.
    a function that never raises exceptions but silently returns wrong values on
    edge inputs that the tests didn't cover.
    """
    if state.get("last_execution_passed", True):
        # Critic approved (or last_execution_passed was never set to False)
        return "__end__"
    # Critic rejected — re-enter diagnosis with critic's feedback as the failure
    return "diagnose_failure"


def _route_after_diagnosis(
    state: AgentState,
) -> Literal["update_learning_log", "increment_iteration"]:
    """Confidence-aware routing after the debugger's ReAct loop (Fix 7).

    WHY THIS MATTERS:
    The debugger runs a multi-step ReAct loop and produces a diagnosis with a
    confidence score. When confidence is very low (< 0.3), the root cause is
    probably wrong — applying it as a repair target could send the agent in the
    wrong direction for multiple iterations. It's better to regenerate from
    scratch (blind retry) than to confidently do the wrong repair.

    High confidence (≥ 0.3) → update learning log → increment → repair with the diagnosis.
    Low confidence (< 0.3)  → skip learning log → increment → regenerate from scratch.

    IMPORTANT: Both paths must go through increment_iteration. Previously the low-
    confidence path jumped directly to "generate_solution", bypassing increment_iteration
    entirely. This left state["iteration"] at 0 forever, causing the max_iterations check
    in _route_after_execution to never trigger — infinite loop on persistent failures.
    """
    confidence = state.get("diagnosis_confidence", 1.0)
    if confidence < _LOW_CONFIDENCE_THRESHOLD:
        logger.info(
            "Low diagnosis confidence (%.2f < %.2f) — routing to blind retry via increment.",
            confidence,
            _LOW_CONFIDENCE_THRESHOLD,
        )
        # Skip update_learning_log (a wrong diagnosis produces no useful lessons),
        # but MUST go through increment_iteration to advance the counter and
        # allow _route_after_increment to terminate at max_iterations.
        return "increment_iteration"
    return "update_learning_log"


def _route_after_increment(
    state: AgentState,
) -> Literal["generate_solution", "__end__"]:
    """Route after iteration increment.

    Simple status check: max_iterations_reached → stop, else continue.
    This is used in the default (non-parallel) repair path.
    """
    if state.get("status") == "max_iterations_reached":
        return "__end__"
    return "generate_solution"


def _build_role_providers(config: AgentConfig) -> dict:
    """Build per-role provider overrides from config.model_overrides (Fix 17).

    Allows running different Ollama models for different agent roles without
    changing the default provider. The primary use case is running a smaller
    model for memory_summarizer (cheap, repetitive task) and a larger model
    for debugger (complex reasoning needed).

    Currently only OllamaProvider honors model overrides. HuggingFace and
    Mock providers ignore the model_name parameter.

    Example config.model_overrides: {"memory_summarizer": "llama3.2:1b"}
    """
    if not config.model_overrides:
        return {}  # fast path — most production runs use no overrides

    role_providers: dict = {}
    import os
    provider_env = os.environ.get("LLM_PROVIDER", "").lower()

    for role, model_name in config.model_overrides.items():
        try:
            if provider_env in ("ollama", ""):
                # Only create Ollama overrides; other providers can't swap models
                from llm.providers.ollama_provider import OllamaProvider
                role_providers[role] = OllamaProvider(model=model_name)
                logger.info("Role '%s' overridden to model '%s' (Ollama)", role, model_name)
        except Exception as exc:
            logger.warning(
                "Failed to create role override for role=%s model=%s: %s",
                role, model_name, exc,
            )

    return role_providers


def build_graph(
    router: LLMRouter | None = None,
    config: AgentConfig | None = None,
    lesson_store: Any = None,
) -> Any:
    """Construct and compile the agent state graph.

    This is the central wiring function. It:
      1. Binds the router to every LLM-calling node via functools.partial
      2. Adds all nodes to the StateGraph
      3. Adds edges (static) and conditional edges (dynamic routing)
      4. Optionally adds HITL review_repair node if not full_auto
      5. Optionally adds parallel repair fan-out if parallel_strategies=True
      6. Attaches a checkpointer (required for interrupt() to work)
      7. Compiles and returns the runnable graph

    Args:
        router:       Optional pre-built LLMRouter. Pass this when you want to
                      share a single router (and its cached model weights) across
                      multiple graph invocations (e.g. the Gradio demo).
                      If None, a new router is auto-resolved.
        config:       AgentConfig controlling topology and feature flags.
                      Defaults to AgentConfig() (production defaults).
        lesson_store: Optional LessonStore for cross-session memory retrieval
                      at iteration 0. If None, lessons are not loaded.

    Returns:
        A compiled LangGraph graph (CompiledGraph) ready for ainvoke/astream.
    """
    if config is None:
        config = AgentConfig()

    # Build role-specific provider overrides if configured, then create router.
    # The router is constructed once here and shared across ALL nodes via
    # functools.partial — nodes never construct their own providers.
    if router is None:
        role_providers = _build_role_providers(config)
        router = LLMRouter(role_providers=role_providers)

    # Bind the router (and lesson_store for generate_solution) to each node.
    # functools.partial turns the 2-arg functions (state, *, router) into
    # single-arg (state) functions that LangGraph can call. This is the
    # dependency-injection pattern used throughout the codebase.
    _generate = functools.partial(generate_solution, router=router, lesson_store=lesson_store)
    _qa = functools.partial(create_adversarial_tests, router=router)
    _diagnose = functools.partial(diagnose_failure, router=router)
    _memory = functools.partial(update_learning_log, router=router)
    _spec_tests = functools.partial(generate_spec_tests, router=router)
    _critic = functools.partial(critic_review, router=router, agent_config=config)
    _parallel_gen = functools.partial(parallel_generate, router=router)

    # Create the StateGraph — LangGraph's builder object. We pass AgentState
    # so it knows the schema and can validate partial updates from each node.
    graph = StateGraph(AgentState)

    # ── Core nodes (always present regardless of config) ─────────────────────
    # These 7 nodes are in every graph topology.
    graph.add_node("generate_solution", _generate)           # writes current_code
    graph.add_node("create_adversarial_tests", _qa)          # writes current_test_code
    graph.add_node("execute_solution", execute_solution)     # runs sandbox, sets last_execution_passed
    graph.add_node("diagnose_failure", _diagnose)            # ReAct loop, sets root_cause/repair_strategy
    graph.add_node("update_learning_log", _memory)           # compresses lessons, updates learning_log
    graph.add_node("increment_iteration", _increment_iteration)  # bumps iteration counter
    # Terminal node for the max-iterations path — sets final status and is
    # connected directly to END so the graph terminates cleanly.
    graph.add_node("max_iterations", lambda s: {"status": "max_iterations_reached"})

    # ── Parallel entry: spec tests + solution generation run concurrently ────
    # Both generate_spec_tests and generate_solution are connected from __start__
    # with a static edge. LangGraph sees two outgoing edges from __start__ and
    # runs both branches concurrently (fan-out). They rejoin at
    # create_adversarial_tests (fan-in) — LangGraph's fan-in waits for BOTH
    # upstream nodes to complete before proceeding.
    #
    # Why generate spec tests in parallel with solution generation?
    # Spec tests must be written BEFORE the solution to be oracle tests (no
    # peeking at the code). But we don't need to wait for spec tests before
    # starting code generation — they're independent until they both converge
    # at create_adversarial_tests. Running them in parallel saves time.
    graph.add_node("generate_spec_tests", _spec_tests)
    graph.add_edge("__start__", "generate_spec_tests")
    graph.add_edge("__start__", "generate_solution")
    # Both branches converge at create_adversarial_tests (fan-in)
    graph.add_edge("generate_spec_tests", "create_adversarial_tests")
    graph.add_edge("generate_solution", "create_adversarial_tests")
    graph.add_edge("create_adversarial_tests", "execute_solution")

    # ── Critic node — always present ─────────────────────────────────────────
    # Conditional edge after execute_solution checks three things:
    #   (a) did tests pass? (b) is critic enabled? (c) max iterations?
    # We always create the critic node (it's always registered in the graph)
    # even though the routing function might never route to it. This avoids
    # a KeyError if the conditional edge map ever resolves to "critic_review"
    # unexpectedly. The factory captures enable_critic=True.
    _route_exec = _make_route_after_execution(enable_critic=True)
    graph.add_node("critic_review", _critic)
    graph.add_conditional_edges(
        "execute_solution",
        _route_exec,
        {
            "critic_review": "critic_review",         # tests passed, critic review
            "diagnose_failure": "diagnose_failure",   # tests failed, debug
            "max_iterations": "max_iterations",       # max hit, terminal
            "__end__": END,                           # tests passed, no critic, done
        },
    )
    # After critic runs, it either approves (END) or rejects (back to diagnose)
    graph.add_conditional_edges(
        "critic_review",
        _route_after_critic,
        {
            "__end__": END,
            "diagnose_failure": "diagnose_failure",
        },
    )

    # ── Confidence-aware routing after diagnosis (Fix 7) ─────────────────────
    # The debugger's confidence score determines whether we do a targeted repair
    # (via update_learning_log → increment → generate_solution with repair context)
    # or a blind retry (skip directly to generate_solution with no diagnosis).
    graph.add_conditional_edges(
        "diagnose_failure",
        _route_after_diagnosis,
        {
            "update_learning_log": "update_learning_log",  # confident → use diagnosis
            "increment_iteration": "increment_iteration",  # not confident → retry via increment
        },
    )

    # ── Optional: HITL review node (Fix 12) ──────────────────────────────────
    # review_repair uses LangGraph's interrupt() to pause the graph and yield
    # control back to the caller (demo_runner.py). The caller can then resume
    # with a Command(resume=decision) to either approve, reject, or edit the
    # repair plan before the next iteration starts.
    #
    # This is only wired in when autonomy_level != "full_auto". On HF Spaces
    # (app.py) we always use full_auto to avoid the interrupt/resume desync.
    if config.autonomy_level != "full_auto":
        graph.add_node("review_repair", review_repair)
        graph.add_edge("update_learning_log", "review_repair")  # pause point AFTER logging
        graph.add_edge("review_repair", "increment_iteration")  # resume leads to increment
    else:
        # full_auto: skip review_repair entirely, go straight to increment
        graph.add_edge("update_learning_log", "increment_iteration")

    # ── Repair loop completion: serial or parallel strategies ─────────────────
    if config.parallel_strategies:
        # Fix 14: fan out to 3 parallel repair branches, tournament-select winner.
        # fan_out_repairs() returns a list of Send("parallel_generate", {...})
        # objects — LangGraph's mechanism for dynamic fan-out. Each Send runs
        # parallel_generate with a different strategy_name in its payload.
        graph.add_node("parallel_generate", _parallel_gen)
        graph.add_node("select_best_repair", select_best_repair)

        def _route_after_increment_parallel(state: AgentState):
            if state.get("status") == "max_iterations_reached":
                return END
            # fan_out_repairs returns [Send("parallel_generate", {...}), ...]
            # LangGraph runs all of them concurrently, collecting results in
            # state["parallel_repairs"] (operator.add reducer merges them).
            return fan_out_repairs(state)

        graph.add_conditional_edges(
            "increment_iteration",
            _route_after_increment_parallel,
            ["parallel_generate", END],  # declare possible destinations
        )
        graph.add_edge("parallel_generate", "select_best_repair")
        # After tournament selection picks the best code, re-run QA and
        # execution on it — the winning strategy might not have been tested yet.
        graph.add_edge("select_best_repair", "create_adversarial_tests")
    else:
        # Default (parallel_strategies=False): simple serial repair loop.
        # increment_iteration → generate_solution (or END)
        graph.add_conditional_edges(
            "increment_iteration",
            _route_after_increment,
            {
                "generate_solution": "generate_solution",
                "__end__": END,
            },
        )

    # Terminal edge — max_iterations node always goes directly to END
    graph.add_edge("max_iterations", END)

    # ── Checkpointer (Fix 13) ─────────────────────────────────────────────────
    # LangGraph's interrupt() works by saving the graph state to a checkpointer
    # at the pause point and reloading it when Command(resume=...) is received.
    # Without a checkpointer, calling interrupt() raises a RuntimeError.
    #
    # We also attach a checkpointer in full_auto mode if enable_checkpointing
    # is True — this enables time-travel debugging (fork_from_iteration) even
    # for autonomous runs.
    checkpointer = None
    needs_checkpointer = (
        config.autonomy_level != "full_auto" or config.enable_checkpointing
    )
    if needs_checkpointer:
        if config.persist_checkpoints:
            # SQLite gives durable checkpoints across process restarts.
            # NOT safe on HF Spaces (ephemeral filesystem) — SqliteSaver may
            # fail there, so we fall back to InMemorySaver.
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
            # Default: in-memory only — fastest option, lost when process ends.
            # This is sufficient for HF Spaces where sessions don't persist.
            from langgraph.checkpoint.memory import InMemorySaver
            checkpointer = InMemorySaver()
            logger.info("Using InMemorySaver checkpointer")

    # compile() locks the graph topology — no more nodes or edges can be added.
    # The returned object (CompiledGraph) has ainvoke, astream, get_state, etc.
    return graph.compile(checkpointer=checkpointer)


def _make_initial_state(
    task_description: str,
    max_iterations: int = 4,
) -> AgentState:
    """Construct a clean AgentState for a brand-new task.

    All string fields start empty; booleans start as their safe default;
    lists start empty. iteration starts at 0, status starts at 'running'.
    This is the only place where an AgentState is constructed from scratch —
    nodes always return partial dicts, never full AgentState objects.
    """
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
        diagnosis_confidence=0.5,     # neutral default — doesn't trigger blind retry
        learning_log=[],
        iteration=0,
        iteration_history=[],
        status="running",
        degraded_nodes=[],
        parallel_repairs=[],
        strategy_name="",
        events=[],
    )


async def run_agent(
    task_description: str,
    max_iterations: int = 4,
    router: LLMRouter | None = None,
    config: AgentConfig | None = None,
) -> AgentState:
    """Run the agent to completion and return the final state.

    This is the simplest high-level entry point. It blocks until the agent
    either succeeds, exhausts iterations, or errors.

    For streaming / live UI updates, use stream_agent() instead.

    Cross-session memory: if config.enable_cross_session_memory is True, a
    LessonStore is created, passed to build_graph (for retrieval at iteration 0),
    and post-run lessons are persisted back to disk + optionally synced to HF Hub.
    """
    if config is None:
        config = AgentConfig()

    # Build the LessonStore once here so we can use the same instance for both
    # the retrieval (inside the graph via lesson_store arg) and the post-run
    # persistence below. Don't create it inside the graph — that would create
    # two separate ChromaDB connections.
    lesson_store = None
    if config.enable_cross_session_memory:
        from agent.memory_store import LessonStore
        lesson_store = LessonStore(config.memory_persist_dir)

    app = build_graph(router=router, config=config, lesson_store=lesson_store)
    initial_state = _make_initial_state(task_description, max_iterations)
    # Each run gets a unique thread_id so LangGraph's checkpointer can
    # distinguish between concurrent runs in the same process.
    thread_config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    final_state = await app.ainvoke(initial_state, thread_config)

    # Persist lessons learned during this run so future runs on similar tasks
    # start with the benefit of this experience.
    if lesson_store is not None:
        from datetime import datetime, timezone
        from agent.hf_memory_sync import compute_fingerprint

        final_log = final_state.get("learning_log", [])
        failure_category = final_state.get("failure_category", "")
        timestamp = datetime.now(timezone.utc).isoformat()
        lesson_dicts: list[dict] = []

        for lesson in final_log:
            lesson_store.store_lesson(
                lesson,
                failure_category=failure_category,
                task_id=task_description[:100],
            )
            lesson_dicts.append({
                "lesson": lesson,
                "task_id": task_description[:100],
                "failure_category": failure_category,
                "timestamp": timestamp,
                "fingerprint": compute_fingerprint(lesson),
            })

        logger.info("Cross-session memory: persisted %d lessons locally.", len(lesson_dicts))

        if lesson_dicts:
            # Sync to HuggingFace Dataset (no-ops if HF_TOKEN is not set)
            await lesson_store.sync_to_hf(lesson_dicts, router=router)

    return final_state


async def stream_agent(
    task_description: str,
    max_iterations: int = 4,
    router: LLMRouter | None = None,
    config: AgentConfig | None = None,
) -> AsyncGenerator[dict[str, Any], None]:
    """Stream agent events as they are produced by each node.

    Yields individual event dicts from state["events"] as nodes complete.
    Events arrive in the order nodes run — faster than waiting for ainvoke
    to return the full final state.

    The caller is responsible for consuming the entire generator (so that
    post-run cross-session memory persistence runs).

    Typical event shapes:
        {"type": "CODE_GENERATED", "data": {"code": "...", "iteration": 0}}
        {"type": "FAILURE", "data": {"summary": "...", "tests": "..."}}
        {"type": "DIAGNOSIS", "data": {"root_cause": "...", "confidence": 0.8}}
        {"type": "SUCCESS", "data": {"code": "...", "iterations": 2}}
    """
    if config is None:
        config = AgentConfig()

    lesson_store = None
    if config.enable_cross_session_memory:
        from agent.memory_store import LessonStore
        lesson_store = LessonStore(config.memory_persist_dir)

    app = build_graph(router=router, config=config, lesson_store=lesson_store)
    initial_state = _make_initial_state(task_description, max_iterations)
    thread_config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    # We track the total events seen so far. On each astream update, we slice
    # state["events"][total_seen:] to get only the NEW events from the latest
    # node. Without this, we'd re-yield all previous events on every node update
    # (because events is cumulative — the operator.add reducer never discards).
    total_seen = 0
    final_learning_log: list[str] = []

    async for state_update in app.astream(initial_state, thread_config):
        # state_update is a dict of {node_name: partial_state_dict}
        for node_name, node_state in state_update.items():
            if not isinstance(node_state, dict):
                continue  # skip non-dict updates (e.g. __interrupt__ metadata)
            # Capture the latest learning_log for post-stream persistence
            if "learning_log" in node_state:
                final_learning_log = node_state["learning_log"]
            events = node_state.get("events", [])
            if not events:
                continue
            # Slice only the events this node added (events list is cumulative)
            new_events = events[total_seen:]
            total_seen = len(events)  # advance the cursor for the next iteration
            for event in new_events:
                if isinstance(event, dict):
                    yield event

    # Post-stream cross-session memory persistence
    if lesson_store is not None and final_learning_log:
        from datetime import datetime, timezone
        from agent.hf_memory_sync import compute_fingerprint

        timestamp = datetime.now(timezone.utc).isoformat()
        lesson_dicts: list[dict] = []

        for lesson in final_learning_log:
            lesson_store.store_lesson(lesson, task_id=task_description[:100])
            lesson_dicts.append({
                "lesson": lesson,
                "task_id": task_description[:100],
                "failure_category": "",
                "timestamp": timestamp,
                "fingerprint": compute_fingerprint(lesson),
            })

        if lesson_dicts:
            await lesson_store.sync_to_hf(lesson_dicts, router=router)


async def run_agent_with_history(
    task_description: str,
    config: AgentConfig,
    router: LLMRouter | None = None,
) -> tuple[AgentState, list[Any]]:
    """Run the agent and also return the full checkpoint history.

    Used for time-travel debugging: you can inspect any intermediate state
    and use fork_from_iteration() to rewind and replay with modifications.

    Requires config.enable_checkpointing=True (or HITL enabled) — otherwise
    the checkpointer is None and get_state_history() returns an empty list.
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
    """Rewind to a prior checkpoint and replay with modified state.

    Time-travel debugging: find the checkpoint for target_iteration, apply
    modified_state overrides (e.g. a manually-corrected repair_strategy),
    and resume execution from diagnose_failure with the new values.

    This is how you investigate counterfactuals: "what if the debugger had
    given a different diagnosis on iteration 2?"

    Args:
        app:              Compiled graph (must have a checkpointer attached).
        thread_config:    Same thread config dict used for the original run.
        target_iteration: Which iteration checkpoint to rewind to.
        modified_state:   Partial dict of state overrides to inject at fork.

    Returns:
        Final AgentState after replaying from the fork point.

    Raises:
        ValueError: If no checkpoint found for the given iteration number.
    """
    history = list(app.get_state_history(thread_config))
    # Search the checkpoint history for the iteration we want to revisit
    target = next(
        (s for s in history if s.values.get("iteration") == target_iteration),
        None,
    )
    if target is None:
        raise ValueError(
            f"No checkpoint found for iteration={target_iteration}. "
            f"Available iterations: {sorted({s.values.get('iteration') for s in history})}"
        )
    # update_state injects modified_state into the checkpoint and marks the
    # next node to run as "diagnose_failure" (so we re-run from that point
    # with the new values, not from generate_solution).
    fork_config = app.update_state(
        target.config, values=modified_state, as_node="diagnose_failure"
    )
    return await app.ainvoke(None, fork_config)


def get_graph_mermaid(
    config: AgentConfig | None = None,
    router: LLMRouter | None = None,
) -> str:
    """Generate a Mermaid diagram string for the current graph topology.

    Builds the graph with the given config and calls LangGraph's built-in
    draw_mermaid() method. The topology changes based on config (HITL node,
    parallel strategies, critic) so this always reflects the live config.

    Falls back to a minimal static diagram if LangGraph can't generate it
    (e.g. if a node import fails during graph construction).
    """
    try:
        app = build_graph(router=router, config=config)
        return app.get_graph().draw_mermaid()
    except Exception as exc:
        logger.warning("Could not generate Mermaid diagram: %s", exc)
        return (
            "graph TD\n"
            "    A[generate_solution] --> B[create_adversarial_tests]\n"
            "    B --> C[execute_solution]\n"
            "    C -->|pass| D([END])\n"
            "    C -->|fail| E[diagnose_failure]\n"
            "    E --> F[update_learning_log]\n"
            "    F --> G[increment_iteration]\n"
            "    G --> A\n"
        )
