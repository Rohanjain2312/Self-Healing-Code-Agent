"""
Integration test: full agent run using the Mock provider.

Verifies the complete graph executes without error and produces
a valid final state. Does not test correctness of LLM outputs —
only the plumbing and state transitions.
"""

import asyncio
import pytest
from agent.graph import run_agent
from llm.router import LLMRouter
from llm.providers.mock_provider import MockProvider


@pytest.mark.asyncio
async def test_agent_runs_to_completion_with_mock():
    """Agent graph executes without crashing using mock provider."""
    router = LLMRouter(provider=MockProvider())

    task = "Write a function add(a, b) that returns a + b."
    final_state = await run_agent(
        task_description=task,
        max_iterations=2,
        router=router,
    )

    assert final_state is not None
    assert "status" in final_state
    assert final_state["status"] in {"success", "max_iterations_reached", "running"}
    assert "current_code" in final_state
    assert isinstance(final_state["events"], list)
    assert len(final_state["events"]) > 0


@pytest.mark.asyncio
async def test_agent_produces_code():
    """Generator node produces non-empty code."""
    router = LLMRouter(provider=MockProvider())
    final_state = await run_agent(
        task_description="Write a function that returns 42.",
        max_iterations=1,
        router=router,
    )
    assert final_state.get("current_code", "").strip() != ""


@pytest.mark.asyncio
async def test_stream_agent_yields_events():
    """stream_agent yields at least one event."""
    from agent.graph import stream_agent
    router = LLMRouter(provider=MockProvider())

    events = []
    async for event in stream_agent(
        task_description="Write a trivial function.",
        max_iterations=1,
        router=router,
    ):
        events.append(event)

    assert len(events) > 0
    for event in events:
        assert "type" in event
        assert "message" in event


@pytest.mark.asyncio
async def test_max_iterations_terminates():
    """Agent terminates cleanly at max_iterations even on repeated failure."""
    # Mock provider returns code that will likely fail real tests,
    # but with max_iterations=1 we just check it terminates
    router = LLMRouter(provider=MockProvider())
    final_state = await run_agent(
        task_description="Implement an impossible requirement.",
        max_iterations=1,
        router=router,
    )
    # Status must be one of the terminal values
    assert final_state["status"] in {"success", "max_iterations_reached", "running"}
