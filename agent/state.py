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

from typing import Any, TypedDict


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
    current_test_code: str

    # --- Execution outcome ---
    last_execution_passed: bool
    last_failure_summary: str

    # --- Debugger diagnosis (populated after failure) ---
    root_cause: str
    failure_category: str
    repair_strategy: str

    # --- Rolling memory (compressed by memory_summarizer node) ---
    learning_log: list[str]

    # --- Iteration tracking ---
    iteration: int
    iteration_history: list[IterationRecord]

    # --- Terminal status ---
    # 'running' | 'success' | 'failed' | 'max_iterations_reached'
    status: str

    # --- Event stream (appended by each node for UI streaming) ---
    events: list[dict[str, Any]]
