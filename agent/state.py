"""
LangGraph state definition for the Self-Healing Code Agent.

State is a TypedDict — LangGraph passes it between nodes immutably.
Each node returns a partial dict that is merged into state.

Design decisions:
  - learning_log is a list[str] of bullet lessons (max 5)
  - iteration_history is a list of dicts preserving per-iteration context
    for the debugger; cleared or summarized after success to save memory
  - max_iterations is set at graph construction time and checked in routing
"""

import operator
from typing import Annotated, Any, TypedDict


class IterationRecord(TypedDict):
    """Record of a single repair iteration."""
    iteration: int
    code: str
    test_code: str
    passed: bool
    failure_summary: str
    root_cause: str
    failure_category: str
    repair_strategy: str


class AgentState(TypedDict):
    # --- Task context (set once at graph entry, never mutated) ---
    task_description: str
    max_iterations: int

    # --- Generated artifacts (updated each iteration) ---
    current_code: str
    current_test_code: str  # adversarial tests (sees the code, generated per iteration)
    spec_test_code: str     # spec-blind tests (generated once from task description only)

    # --- Execution outcome ---
    last_execution_passed: bool
    last_failure_summary: str

    # --- Debugger diagnosis (populated after failure) ---
    root_cause: str
    failure_category: str
    repair_strategy: str
    diagnosis_confidence: float  # 0.0–1.0; below 0.3 triggers blind retry routing

    # --- Rolling memory (compressed by memory_summarizer node) ---
    learning_log: list[str]

    # --- Iteration tracking ---
    iteration: int
    iteration_history: list[IterationRecord]

    # --- Terminal status ---
    # 'running' | 'success' | 'failed' | 'max_iterations_reached'
    status: str

    # --- Degradation tracking ---
    # Nodes that used fallback behavior due to LLM schema validation failure
    degraded_nodes: list[str]

    # --- Parallel repair strategies (Fix 14) ---
    # Each parallel branch appends its result; reducer merges them
    parallel_repairs: Annotated[list, operator.add]
    # Which strategy this branch is executing (set by fan_out_repairs Send())
    strategy_name: str

    # --- Event stream (appended by each node for UI streaming) ---
    # Annotated with operator.add so parallel nodes can both append events
    # without a merge conflict (both generate_spec_tests and generate_solution
    # run concurrently and each emit their own events).
    events: Annotated[list[dict[str, Any]], operator.add]
