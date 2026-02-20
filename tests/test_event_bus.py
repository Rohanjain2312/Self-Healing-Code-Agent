"""
Tests for the async event bus.
"""

import asyncio
import pytest
from framework.event_bus import EventBus


@pytest.mark.asyncio
async def test_emit_received_by_subscriber():
    bus = EventBus()
    received = []

    async def consume():
        async with bus.subscribe() as queue:
            event = await asyncio.wait_for(queue.get(), timeout=1.0)
            received.append(event)

    task = asyncio.create_task(consume())
    await asyncio.sleep(0.01)  # let consumer register

    await bus.emit({"type": "step", "message": "hello"})
    await task

    assert len(received) == 1
    assert received[0]["message"] == "hello"


@pytest.mark.asyncio
async def test_multiple_subscribers():
    bus = EventBus()
    results_a = []
    results_b = []

    async def consumer(results):
        async with bus.subscribe() as queue:
            event = await asyncio.wait_for(queue.get(), timeout=1.0)
            results.append(event)

    task_a = asyncio.create_task(consumer(results_a))
    task_b = asyncio.create_task(consumer(results_b))
    await asyncio.sleep(0.01)

    await bus.emit({"type": "test"})
    await asyncio.gather(task_a, task_b)

    assert len(results_a) == 1
    assert len(results_b) == 1


@pytest.mark.asyncio
async def test_close_sends_sentinel():
    bus = EventBus()

    async def consume():
        async with bus.subscribe() as queue:
            sentinel = await asyncio.wait_for(queue.get(), timeout=1.0)
            return sentinel

    task = asyncio.create_task(consume())
    await asyncio.sleep(0.01)
    await bus.close()
    result = await task
    assert result is None  # sentinel value


@pytest.mark.asyncio
async def test_subscriber_count():
    bus = EventBus()
    assert bus.subscriber_count == 0

    async with bus.subscribe():
        assert bus.subscriber_count == 1

    assert bus.subscriber_count == 0
