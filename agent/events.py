"""
Event definitions for the Self-Healing Code Agent.

Events are appended to AgentState.events and emitted via the event bus.
The UI consumes events from the async generator exposed by the streaming layer.

Event types are string constants to keep them JSON-serializable.
"""

from dataclasses import dataclass, field, asdict
from typing import Any
import time


# --- Event type constants ---
STEP = "step"
CODE_GENERATED = "code_generated"
FAILURE = "failure"
LEARNING_UPDATE = "learning_update"
SUCCESS = "success"
ITERATION_START = "iteration_start"
TESTS_GENERATED = "tests_generated"
DIAGNOSIS = "diagnosis"
REPAIR_START = "repair_start"
TIMEOUT = "timeout"


@dataclass
class AgentEvent:
    """
    Base event structure emitted by every node.

    Keeping payload optional allows lightweight step events that carry
    only a message, while richer events carry structured data for the UI.
    """
    type: str
    message: str
    payload: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    iteration: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# --- Constructor helpers — one per event type for type safety ---

def step_event(message: str, iteration: int = 0, **payload) -> AgentEvent:
    return AgentEvent(type=STEP, message=message, iteration=iteration, payload=payload)


def code_generated_event(code: str, iteration: int = 0, explanation: str = "") -> AgentEvent:
    return AgentEvent(
        type=CODE_GENERATED,
        message=f"Code generated (iteration {iteration})",
        iteration=iteration,
        payload={"code": code, "explanation": explanation},
    )


def failure_event(summary: str, iteration: int = 0, failed_assertions: list = None) -> AgentEvent:
    return AgentEvent(
        type=FAILURE,
        message=f"Test failure detected (iteration {iteration})",
        iteration=iteration,
        payload={
            "summary": summary,
            "failed_assertions": failed_assertions or [],
        },
    )


def learning_update_event(lessons: list[str], iteration: int = 0) -> AgentEvent:
    return AgentEvent(
        type=LEARNING_UPDATE,
        message="Learning log updated",
        iteration=iteration,
        payload={"lessons": lessons},
    )


def success_event(code: str, iteration: int = 0) -> AgentEvent:
    return AgentEvent(
        type=SUCCESS,
        message=f"All tests passed on iteration {iteration}",
        iteration=iteration,
        payload={"code": code, "iterations_required": iteration},
    )


def diagnosis_event(root_cause: str, category: str, strategy: str, iteration: int = 0) -> AgentEvent:
    return AgentEvent(
        type=DIAGNOSIS,
        message=f"Root cause identified: {category}",
        iteration=iteration,
        payload={
            "root_cause": root_cause,
            "failure_category": category,
            "repair_strategy": strategy,
        },
    )
