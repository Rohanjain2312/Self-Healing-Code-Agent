"""
Demo runner — async bridge between the agent and the Gradio UI.

Converts the agent's async event stream into incremental UI state updates
that Gradio can consume via its generator-based streaming API.

The runner maintains UI state locally and yields (timeline, code, log) tuples
after each event so Gradio can update all three components atomically.

HF Spaces constraints respected:
  - max_iterations capped at 4 to limit inference time
  - No benchmark execution in demo mode
  - Streaming via Python async generators
"""

import asyncio
import logging
from typing import AsyncGenerator, Generator

from agent.config import AgentConfig
from agent.graph import stream_agent
from framework.streaming import (
    format_event_for_timeline,
    PUBLIC_EVENT_TYPES,
)
from agent.events import SUCCESS, CODE_GENERATED, LEARNING_UPDATE, REPAIR_REVIEW
from llm.router import LLMRouter

logger = logging.getLogger(__name__)

# Cap for HF Spaces free tier — prevents runaway inference costs
_MAX_DEMO_ITERATIONS = 4

# Example tasks shown in the demo UI
EXAMPLE_TASKS = [
    # Problem 1: LRU Cache with frequency-based tie-breaking
    # Hard because: model knows LRU but not this specific tie-breaking rule.
    # Fails on: eviction when multiple keys have same frequency — wrong key evicted.
    """Implement a class `LFUCache` with `get(key)` and `put(key, value)` methods.
It must have O(1) average time complexity for both operations.

Eviction rules (strictly in this order):
1. Evict the key with the LOWEST access frequency (get + put count both increment frequency).
2. If multiple keys share the lowest frequency, evict the one that was LEAST RECENTLY USED among them.
3. put() on an existing key updates its value AND increments its frequency.
4. put() on a new key when at capacity must evict before inserting. The new key starts with frequency 1.

Example:
cache = LFUCache(2)
cache.put(1, 1)   # cache: {1: freq=1}
cache.put(2, 2)   # cache: {1: freq=1, 2: freq=1}
cache.get(1)      # returns 1, cache: {1: freq=2, 2: freq=1}
cache.put(3, 3)   # evicts key 2 (lowest freq=1), cache: {1: freq=2, 3: freq=1}
cache.get(2)      # returns -1 (not found)
cache.get(3)      # returns 3
cache.put(4, 4)   # evicts key 3 (lowest freq=1, LRU among freq=1 keys), cache: {1: freq=2, 4: freq=1}
cache.get(3)      # returns -1
cache.get(4)      # returns 4
cache.get(1)      # returns 1

Additional constraints:
- get() on a missing key returns -1 and does NOT affect any frequencies.
- A cache with capacity 0 always returns -1 on get() and silently ignores put().
- All keys and values are non-negative integers.""",

    # Problem 2: Expression evaluator with operator precedence and left-associativity
    # Hard because: model typically uses a stack but gets precedence or left-associativity wrong.
    # Fails on: chained same-precedence operators (8/2/2 should be 2 not 8), unary minus edge cases.
    """Write a function `evaluate(expression: str) -> float` that evaluates a mathematical
expression string containing:
- Non-negative integers and decimals (e.g. 3, 3.14)
- Operators: +, -, *, / (true division, not floor)
- Unary minus (e.g. -3 * 2, -(4+5))
- Parentheses for grouping (arbitrarily nested)
- Spaces anywhere between tokens (ignore all whitespace)

Operator precedence (high to low): *, / then +, -
All operators are LEFT-ASSOCIATIVE:
  8 / 2 / 2  →  (8 / 2) / 2  →  2.0   (NOT 8 / (2/2) = 8.0)
  10 - 3 - 2  →  (10 - 3) - 2  →  5.0

Edge cases your implementation must handle:
- Unary minus before parentheses: -(3 + 4) → -7.0
- Unary minus before a number: -3 + 5 → 2.0
- Chained unary: --3 → 3.0
- Expression starting with unary minus: -3 * -2 → 6.0
- Division always returns float: 7 / 2 → 3.5
- Single number: "42" → 42.0, "-42" → -42.0

Do NOT use Python's eval() or ast.literal_eval().
Do NOT import any external libraries.

Examples:
evaluate("3 + 4 * 2")       → 11.0
evaluate("(3 + 4) * 2")     → 14.0
evaluate("8 / 2 / 2")       → 2.0
evaluate("-3 * -2")         → 6.0
evaluate("-(3 + 4)")        → -7.0
evaluate("10 - 3 - 2")      → 5.0
evaluate("3.5 * 2")         → 7.0""",

    # Problem 3: Alien dictionary topological sort with specific error handling
    # Hard because: model gets basic topo sort right but fails on cycle detection,
    # missing characters, or single-word edge cases.
    # Fails on: words where a prefix word appears AFTER the longer word (invalid ordering).
    """Write a function `alien_order(words: list[str]) -> str` that determines the
character ordering of an alien alphabet given a sorted list of words in that alien language.

Rules:
- Return a string of all unique characters in the alien alphabet in a valid topological order.
- If multiple valid orderings exist, return the LEXICOGRAPHICALLY SMALLEST valid ordering.
- If the ordering is CONTRADICTORY (contains a cycle), return "" (empty string).
- If a longer word appears before its prefix (e.g. ["abc", "ab"]), this is INVALID — return "".
- Characters that appear in the word list but have NO ordering constraints relative to others
  must still be included in the output.
- If the input has only one word, return its unique characters in lexicographically sorted order.

Examples:
alien_order(["wrt", "wrf", "er", "ett", "rftt"])  → "wertf"
alien_order(["z", "x"])                            → "zx"
alien_order(["z", "x", "z"])                       → ""  (cycle: z→x→z)
alien_order(["abc", "ab"])                          → ""  (prefix after longer word)
alien_order(["abc"])                                → "abc"
alien_order(["z", "z"])                             → "z"  (duplicate, no info)
alien_order(["ac", "ab", "zc", "zb"])              → "azbc" or valid topo with lex smallest first

Implementation requirements:
- Use a min-heap or equivalent to always pick the lexicographically smallest available character.
- Do not use any topological sort library — implement from scratch using adjacency list + in-degree tracking.""",

    # Problem 4: Run-length encoding with a very specific output format
    # Hard because: model knows RLE but gets the format wrong for runs of length 1,
    # or fails on unicode/mixed characters, or gets the decode round-trip wrong.
    # Fails on: single characters (should not emit "1x"), consecutive same chars across boundaries.
    """Write two functions:

1. `rle_encode(s: str) -> str`
   Run-length encode a string using this EXACT format:
   - A run of N identical characters is encoded as: the character followed by N (e.g. "a3")
   - EXCEPTION: a run of exactly 1 character is encoded as just the character with NO number (e.g. "a" not "a1")
   - The encoded string must be strictly shorter than or equal to the original for it to be worth encoding.
     If encoding would make it longer, return the original string unchanged prefixed with "RAW:" (e.g. "RAW:abc")
   - Empty string encodes to empty string.

2. `rle_decode(s: str) -> str`
   Decode a string produced by rle_encode():
   - If the string starts with "RAW:", strip the prefix and return the rest unchanged.
   - Otherwise parse character-then-optional-digits pairs and expand them.
   - Must be a perfect inverse of rle_encode() for all inputs.

Examples:
rle_encode("aabbbcccc")    → "a2b3c4"
rle_encode("abcd")         → "RAW:abcd"   (encoding "a1b1c1d1" would be longer... wait no:
                                            "abcd" encodes to "abcd" since each run=1, no digits,
                                            so encoded = "abcd" same length → return "RAW:abcd"
                                            because NOT strictly shorter)
rle_encode("aaabcd")       → "RAW:aaabcd" (encodes to "a3bcd" which IS shorter → return "a3bcd")
rle_encode("")             → ""
rle_encode("aaaa")         → "a4"
rle_encode("aabb")         → "RAW:aabb"   (encodes to "a2b2" same length → not shorter → RAW)

rle_decode("a2b3c4")       → "aabbbcccc"
rle_decode("RAW:abcd")     → "abcd"
rle_decode("a3bcd")        → "aaabcd"
rle_decode("a4")           → "aaaa"
rle_decode("")             → ""

Additional constraints:
- Input strings may contain any printable ASCII character including digits, spaces, punctuation.
- rle_decode(rle_encode(s)) == s must hold for ALL strings s.
- rle_encode(rle_decode(t)) == t must hold for all valid encoded strings t.""",
]


class DemoUIState:
    """Accumulated UI state built from the event stream."""

    def __init__(self) -> None:
        self.timeline_lines: list[str] = []
        self.current_code: str = ""
        self.learning_lessons: list[str] = []
        self.is_complete: bool = False
        self.final_status: str = ""

    def apply_event(self, event: dict) -> None:
        event_type = event.get("type", "")

        if event_type in PUBLIC_EVENT_TYPES:
            line = format_event_for_timeline(event)
            self.timeline_lines.append(line)

        if event_type == CODE_GENERATED:
            self.current_code = event.get("payload", {}).get("code", self.current_code)

        if event_type == LEARNING_UPDATE:
            self.learning_lessons = event.get("payload", {}).get("lessons", self.learning_lessons)

        if event_type == SUCCESS:
            self.is_complete = True
            self.final_status = "success"

        if event_type == REPAIR_REVIEW:
            # HITL pause — surface the interrupt payload in the timeline.
            # Full interrupt/resume flow is handled by the async demo when
            # autonomy_level != "full_auto".
            payload = event.get("payload", {})
            self.timeline_lines.append(
                f"[HITL] Repair review requested — "
                f"category={payload.get('failure_category', '?')} "
                f"confidence={payload.get('confidence', 0):.0%}"
            )

    def timeline_text(self) -> str:
        return "\n".join(self.timeline_lines) if self.timeline_lines else "Waiting for agent..."

    def code_text(self) -> str:
        return self.current_code if self.current_code else "# Waiting for code generation..."

    def lessons_text(self) -> str:
        if not self.learning_lessons:
            return "No lessons recorded yet."
        return "\n".join(f"• {lesson}" for lesson in self.learning_lessons)


async def run_demo_async(
    task_description: str,
    router: LLMRouter | None = None,
    config: AgentConfig | None = None,
) -> AsyncGenerator[tuple[str, str, str], None]:
    """
    Async generator that yields (timeline, code, learning_log) tuples.

    Each yield updates all three Gradio components simultaneously.

    Args:
        task_description: The user's task string.
        router: Optional pre-constructed LLMRouter.
        config: AgentConfig controlling which features are active (Fix 19).
                Defaults to AgentConfig.development().
    """
    if not task_description or not task_description.strip():
        yield ("No task provided.", "# No task.", ""), False
        return

    state = DemoUIState()
    yield (
        "Agent starting...",
        "# Initializing...",
        "No lessons yet.",
    )

    try:
        async for event in stream_agent(
            task_description=task_description.strip(),
            max_iterations=_MAX_DEMO_ITERATIONS,
            router=router,
            config=config,
        ):
            if event is None:
                break
            state.apply_event(event)
            yield (
                state.timeline_text(),
                state.code_text(),
                state.lessons_text(),
            )

    except Exception as exc:
        logger.error("Demo runner error: %s", exc, exc_info=True)
        state.timeline_lines.append(f"[ERROR] Agent encountered an error: {exc}")
        yield (
            state.timeline_text(),
            state.code_text(),
            state.lessons_text(),
        )
        return

    # Final update
    if state.is_complete:
        state.timeline_lines.append("Agent completed successfully.")
    else:
        state.timeline_lines.append("Agent reached maximum iterations.")

    yield (
        state.timeline_text(),
        state.code_text(),
        state.lessons_text(),
    )


def run_demo_sync(
    task_description: str,
    router: LLMRouter | None = None,
    config: AgentConfig | None = None,
) -> Generator[tuple[str, str, str], None, None]:
    """
    Synchronous generator wrapper for Gradio's streaming interface.

    Gradio's gr.Interface with streaming expects a regular generator.
    This bridges the async generator to the synchronous Gradio API.

    Args:
        task_description: The user's task string.
        router: Optional pre-constructed LLMRouter.
        config: AgentConfig preset to use (Fix 19). Defaults to development().
    """
    async def _collect():
        results = []
        async for update in run_demo_async(task_description, router=router, config=config):
            results.append(update)
        return results

    # asyncio.run() always creates and tears down its own event loop.
    # This is required inside Gradio's AnyIO worker threads, where
    # asyncio.get_event_loop() raises RuntimeError (no current loop).
    results = asyncio.run(_collect())

    yield from results
