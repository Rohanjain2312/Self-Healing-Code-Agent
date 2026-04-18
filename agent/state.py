"""
LangGraph state definition for the Self-Healing Code Agent.

WHY TYPEDDICT?
--------------
LangGraph requires state to be a TypedDict (or Pydantic BaseModel). TypedDict
was chosen because it is lighter — no validation overhead on every state merge,
and it interoperates naturally with the ``Annotated`` type hint syntax that
LangGraph uses for custom reducers.

HOW NODES WRITE TO STATE
------------------------
Nodes never receive the full state as a mutable object. Instead, they return
a PARTIAL dict of only the fields they want to update::

    async def my_node(state: AgentState) -> dict[str, Any]:
        return {"current_code": new_code, "status": "running"}

LangGraph merges that partial dict into the master state. Fields not mentioned
in the return are left unchanged. For fields with a custom reducer (Annotated
with ``operator.add``), LangGraph calls the reducer instead of replacing — so
two concurrent nodes can both append to ``events`` without either overwriting
the other.

PARALLEL SAFETY
---------------
Two nodes run concurrently at graph start: ``generate_spec_tests`` and
``generate_solution``. They write to independent fields (``spec_test_code`` vs
``current_code``) so there is no merge conflict. Both also emit events, and
because ``events`` uses ``operator.add`` as reducer, both appends are merged
correctly without a race condition.

STATE LIFETIME
--------------
One AgentState is created per run in ``agent.graph._make_initial_state``. It
accumulates updates from every node. At the end of the run LangGraph returns
the final merged state dict as an AgentState.
"""

import operator
from typing import Annotated, Any, TypedDict


def _merge_events(existing: list, new: list) -> list:
    """Dedup-aware reducer for the events list.

    Each node reads ``state["events"]``, appends its new events, and returns
    the full list. With a plain ``operator.add`` reducer this DOUBLES the list
    on every serial node (existing + returned = existing + existing + new).
    This reducer only appends events that aren't already present, identified by
    the (type, timestamp) pair which is unique per event construction.

    The parallel fan-out (generate_spec_tests + generate_solution) is still
    handled correctly: both start from events=[], so their contributions are
    disjoint and both are added without duplicates.
    """
    if not existing:
        return list(new)
    existing_keys = {
        (e.get("type"), e.get("timestamp"))
        for e in existing
        if isinstance(e, dict)
    }
    result = list(existing)
    for ev in new:
        if isinstance(ev, dict):
            key = (ev.get("type"), ev.get("timestamp"))
            if key not in existing_keys:
                result.append(ev)
                existing_keys.add(key)
    return result


class IterationRecord(TypedDict):
    """Snapshot of a single generate → test → diagnose → repair cycle.

    Stored in ``AgentState.iteration_history`` and passed to the debugger node
    so it can compare the current failure against previous ones (detecting
    oscillation / convergence patterns). Also used by the ``diff_iterations``
    debugger tool.
    """
    iteration: int             # which iteration number (0-indexed)
    code: str                  # the code that was tested
    test_code: str             # the adversarial tests that ran against it
    passed: bool               # True if all tests passed
    failure_summary: str       # textual description of failures (empty if passed)
    root_cause: str            # debugger's root-cause conclusion
    failure_category: str      # e.g. "edge_case", "type_error", "logic_error"
    repair_strategy: str       # repair plan the debugger issued


class AgentState(TypedDict):
    """Master state dict threaded through every node in the agent graph.

    Fields are grouped by their lifecycle:
      - **Task context** — set once at entry, never mutated
      - **Generated artifacts** — overwritten each iteration by nodes
      - **Execution outcome** — set by execute_solution after each test run
      - **Debugger diagnosis** — set by diagnose_failure after each failure
      - **Memory** — accumulates across iterations, compressed by memory_summarizer
      - **Tracking** — counters and status flags
      - **Streaming** — events list consumed by the Gradio UI in real time
    """

    # ── Task context ─────────────────────────────────────────────────────────
    # These two fields are written once by _make_initial_state() and never
    # changed again. All nodes treat them as read-only constants.

    task_description: str
    # The user's natural-language problem statement. Passed to every prompt
    # as context so the LLM understands what it's trying to build.

    max_iterations: int
    # Hard cap on generate→test→diagnose→repair cycles. When the iteration
    # counter reaches this value, graph routing immediately goes to END.
    # Default: 4 (set in AgentConfig.max_iterations).

    # ── Generated artifacts ───────────────────────────────────────────────────
    # Overwritten by nodes each iteration. Previous values are captured in
    # iteration_history before being replaced so the debugger can compare them.

    current_code: str
    # The latest Python solution generated by the generator node. This is the
    # code that execute_solution will run against the tests.

    current_test_code: str
    # Adversarial tests generated by create_adversarial_tests. These are
    # written AFTER looking at the code, with the goal of breaking it.
    # Regenerated every iteration because the code changes each time.

    spec_test_code: str
    # Spec-blind oracle tests generated by generate_spec_tests. Unlike
    # current_test_code, these are produced from the task description BEFORE
    # any code exists, so the generator never sees them. They stay constant
    # across all repair iterations — only the adversarial tests change.

    # ── Execution outcome ─────────────────────────────────────────────────────
    # Written by execute_solution after running BOTH test suites in the sandbox.

    last_execution_passed: bool
    # True if all tests passed (both spec_test_code and current_test_code).
    # Used by _route_after_execution to decide: critic_review or diagnose_failure.
    # Also checked by _route_after_critic — the critic can flip this to False
    # if it detects correctness issues the test suite missed.

    last_failure_summary: str
    # Human-readable summary of test failures: which tests failed, what
    # error/assertion was raised, and the stderr output from the subprocess.
    # Fed verbatim into the debugger prompt as evidence for root-cause analysis.

    # ── Debugger diagnosis ────────────────────────────────────────────────────
    # Written by diagnose_failure after its ReAct tool-use loop concludes.

    root_cause: str
    # The debugger's conclusion: what specific bug in current_code caused the
    # observed test failures. Passed to the generator node on the next repair
    # iteration as targeted guidance.

    failure_category: str
    # Categorical label: "edge_case" | "type_error" | "logic_error" |
    # "boundary_condition" | "off_by_one" | etc. Used by memory_summarizer
    # to group lessons by failure type for cross-session retrieval.

    repair_strategy: str
    # Actionable instruction for the generator: "validate input type",
    # "handle empty list edge case", "fix off-by-one in slice". More specific
    # than root_cause — this is the "what to do next" field.

    diagnosis_confidence: float
    # 0.0–1.0 confidence score from the debugger. Below 0.3 triggers the
    # low-confidence routing path: instead of acting on a possibly-wrong
    # diagnosis, the graph routes directly back to generate_solution for a
    # blind retry from scratch. See _route_after_diagnosis in agent/graph.py.

    # ── Rolling memory ────────────────────────────────────────────────────────

    learning_log: list[str]
    # Rolling list of bullet-point lessons from past iterations. Capped at
    # 5 items by the memory_summarizer node (it compresses + deduplicates on
    # each update). Lessons are prepended to the generator prompt on repair
    # iterations so the LLM avoids repeating the same mistake.

    # ── Iteration tracking ────────────────────────────────────────────────────

    iteration: int
    # Current iteration count (0-indexed). Starts at 0, incremented by
    # increment_iteration before each repair. Checked in routing to stop at
    # max_iterations.

    iteration_history: list[IterationRecord]
    # Chronological list of IterationRecord snapshots. Passed to diagnose_failure
    # so the debugger can see whether past repairs made progress or oscillated.
    # Also used by the diff_iterations tool.

    # ── Terminal status ───────────────────────────────────────────────────────

    status: str
    # 'running'                 — agent is in-flight, not yet done
    # 'success'                 — all tests passed and critic approved
    # 'failed'                  — all iterations exhausted, still failing
    # 'max_iterations_reached'  — hit the max_iterations cap (set by increment_iteration)

    # ── Degradation tracking ──────────────────────────────────────────────────

    degraded_nodes: list[str]
    # Names of nodes that used their fallback response because the LLM output
    # failed schema validation on all retries. Allows operators to audit which
    # parts of a run degraded. E.g. ["diagnose_failure", "memory_summarizer"]

    # ── Parallel repair strategies (Fix 14) ───────────────────────────────────

    parallel_repairs: Annotated[list, operator.add]
    # When config.parallel_strategies=True, the graph fans out via LangGraph's
    # Send() to run multiple parallel_generate branches. Each branch appends
    # its result here. The operator.add reducer is critical: both branches
    # write to parallel_repairs concurrently, and operator.add merges the two
    # lists rather than overwriting. Without the reducer, one branch would
    # silently overwrite the other.

    strategy_name: str
    # The strategy identifier for this parallel branch (e.g. "targeted_fix",
    # "alternative_algorithm", "defensive_coding"). Set by fan_out_repairs()
    # when it creates the Send() objects. Empty string in non-parallel mode.

    # ── Event stream ──────────────────────────────────────────────────────────

    events: Annotated[list[dict[str, Any]], _merge_events]
    # Chronological list of event dicts emitted by every node. The Gradio UI
    # (demo/demo_runner.py) reads this list and converts each event to a
    # timeline entry, a code snapshot, or a learning-log update.
    #
    # WHY operator.add?
    # generate_spec_tests and generate_solution run in parallel at graph start.
    # Both emit their own events. If events used simple replacement (no reducer),
    # whichever branch completed last would overwrite the other's events.
    # operator.add concatenates both branches' lists so no events are lost.
    #
    # IMPORTANT GOTCHA: Because the reducer concatenates, the stored events
    # list is CUMULATIVE across all nodes. demo_runner.py tracks total_seen to
    # slice only the NEW events from the latest state update — otherwise it
    # would reprocess all previous events on every node completion.
    #
    # Event dict schema (fields vary by event type):
    #   type: str          — e.g. "CODE_GENERATED", "TESTS_CREATED", "FAILURE",
    #                          "DIAGNOSIS", "REPAIR_REVIEW", "LEARNING_UPDATE",
    #                          "SUCCESS", "ERROR"
    #   timestamp: float   — time.time() at event creation
    #   data: dict         — event-specific payload (code, error, lesson, etc.)
