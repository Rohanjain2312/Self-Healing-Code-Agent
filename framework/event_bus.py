"""
Async event bus — publish/subscribe infrastructure for agent events.

Nodes publish events via emit(). Subscribers receive events via async queues.
The bus is designed to be lightweight and reusable across agent types.

Design decisions:
  - Each subscriber gets its own asyncio.Queue to decouple producers and consumers
  - Subscribers are cleaned up automatically when they close their context
  - The bus is not a singleton — callers construct one per agent run
    to avoid state leaking between runs
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator

logger = logging.getLogger(__name__)

_QUEUE_MAXSIZE = 256  # prevents unbounded memory growth on slow consumers


class EventBus:
    """
    Lightweight async publish-subscribe bus.

    Usage:
        bus = EventBus()

        # Producer side (inside agent nodes via emit helper):
        await bus.emit({"type": "step", "message": "..."})

        # Consumer side (Gradio UI, logging, etc.):
        async with bus.subscribe() as queue:
            async for event in queue:
                handle(event)
    """

    def __init__(self) -> None:
        self._subscribers: list[asyncio.Queue] = []
        self._lock = asyncio.Lock()
        self._closed = False

    async def emit(self, event: dict[str, Any]) -> None:
        """Publish an event to all active subscribers."""
        if self._closed:
            return
        async with self._lock:
            dead = []
            for queue in self._subscribers:
                try:
                    queue.put_nowait(event)
                except asyncio.QueueFull:
                    logger.warning("EventBus: subscriber queue full, dropping event")
                except Exception as exc:
                    logger.error("EventBus: error publishing to subscriber: %s", exc)
                    dead.append(queue)
            for queue in dead:
                self._subscribers.remove(queue)

    @asynccontextmanager
    async def subscribe(self) -> AsyncGenerator[asyncio.Queue, None]:
        """
        Context manager that yields a Queue of events.

        Automatically registers and deregisters the subscriber.
        """
        queue: asyncio.Queue = asyncio.Queue(maxsize=_QUEUE_MAXSIZE)
        async with self._lock:
            self._subscribers.append(queue)
        try:
            yield queue
        finally:
            async with self._lock:
                try:
                    self._subscribers.remove(queue)
                except ValueError:
                    pass

    async def close(self) -> None:
        """Signal all subscribers that no more events will arrive."""
        self._closed = True
        async with self._lock:
            for queue in self._subscribers:
                try:
                    queue.put_nowait(None)  # sentinel
                except asyncio.QueueFull:
                    pass

    @property
    def subscriber_count(self) -> int:
        return len(self._subscribers)


# Module-level convenience: emit to a bus from outside async context
def emit_sync(bus: EventBus, event: dict[str, Any]) -> None:
    """Synchronous emit for contexts where async is not available."""
    try:
        loop = asyncio.get_running_loop()
        # Already inside a running event loop — schedule as a task
        loop.create_task(bus.emit(event))
    except RuntimeError:
        # No running loop — start one just for this call
        try:
            asyncio.run(bus.emit(event))
        except Exception:
            pass
