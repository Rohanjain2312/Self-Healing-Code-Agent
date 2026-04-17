"""
HuggingFace Spaces entry point for the Self-Healing Code Agent.

WHY THIS FILE EXISTS
--------------------
HuggingFace Spaces follows a strict convention: it expects a file named ``app.py``
at the *repository root* and runs it with ``python app.py`` when the Space starts.
Everything that should run on Space startup lives here:

    1. Pre-warming the local generator model so the first user request is fast.
    2. Constructing the AgentConfig and LLMRouter that Gradio will share across
       every request.
    3. Building the Gradio Blocks app and launching the HTTP server on
       ``0.0.0.0:$PORT`` (HF Spaces expects port 7860 by default).

WHY THE ROUTING IS HYBRID (HF + CLAUDE)
---------------------------------------
The agent is *intentionally* designed so the **generator** role (initial code
writing) uses a small, weak model (Llama-3.2-3B). A weak generator is what makes
the self-healing loop interesting — strong models like Claude one-shot most
tasks, so you never see the diagnose → repair cycle that the project is meant to
showcase. The other roles (QA adversarial tester, debugger, critic, memory
summarizer) all use Claude (Anthropic API) for higher reasoning quality.

This split is wired in :func:`llm.router.build_router_with_generator_override`
and the conditions there determine which provider gets bound to the
``generator`` role. To enable the small-model generator on HF Spaces you MUST
set ``LLM_PROVIDER=huggingface`` as a *Variable* in the Space settings;
otherwise the generator silently falls back to Claude and the repair loop
almost never triggers.

WHY autonomy_level IS OVERRIDDEN TO full_auto
---------------------------------------------
:class:`AgentConfig` defaults to ``autonomy_level="review_repairs"`` which uses
LangGraph's ``interrupt()`` to pause the graph after every diagnosis so a human
can approve each repair. That is the right default for local development but it
is *not* reliable inside the Gradio UI on HF Spaces (the resume-from-pause flow
can desync with the streaming generator), so we explicitly force ``full_auto``
here for the deployed Space.

REQUIRED ENV VARS ON THE SPACE
------------------------------
- ``ANTHROPIC_API_KEY``        — Secret. Required for QA / debugger / critic /
                                  memory-summarizer roles.
- ``LLM_PROVIDER=huggingface`` — Variable. Required to enable the local 3B
                                  generator (otherwise generator → Claude).

OPTIONAL ENV VARS
-----------------
- ``HF_MODEL``                 — Override the generator model id (default:
                                  ``meta-llama/Llama-3.2-3B-Instruct``).
- ``HF_TOKEN``                 — Required only if you want cross-session memory
                                  to persist to a HuggingFace Dataset.
- ``PORT``                     — Override the Gradio HTTP port (default 7860).
"""

import asyncio
import logging
import os
import sys
from pathlib import Path

# HF Spaces imports app.py with the project root as cwd, but Python's import
# system also needs the project root on sys.path for ``from agent...`` to work.
# Insert it at index 0 so our local packages take precedence over any
# similarly-named installed packages.
sys.path.insert(0, str(Path(__file__).parent))

# Configure root logger early so every subsequent import that uses ``logging``
# inherits this format. INFO level is verbose enough to see provider routing
# decisions and per-LLM-call latencies in the Space logs.
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Module-level state populated by :func:`_prewarm`. If the pre-warm fails we
# still want to launch Gradio (the Space should be usable for the Performance
# tab even when the generator is broken), but we surface the error in the logs
# at ERROR level so it's obvious from the HF Space "Logs" tab.
_PREWARM_ERROR: str | None = None


def _prewarm() -> None:
    """Load the generator model into memory before Gradio starts accepting requests.

    Pre-warming is only meaningful for the HuggingFace provider — Claude and
    Ollama don't load model weights into our process so there's nothing to
    pre-load. We deliberately call ``asyncio.run()`` here because this runs at
    import time, before any event loop is started by Gradio. Doing it now
    avoids a 30-90s "first request takes forever" surprise for the first user.

    Pre-warm failures are caught and logged but never raised — a broken model
    download should not prevent the Space from booting (the Performance tab and
    the rest of the UI still work).
    """
    global _PREWARM_ERROR

    # Read the env var lazily here (rather than at module import) so tests can
    # monkeypatch os.environ without importing this module twice.
    llm_provider = os.environ.get("LLM_PROVIDER", "").lower()

    if llm_provider != "huggingface":
        # Most common path for local dev: ANTHROPIC_API_KEY is set, the router
        # picks Claude for everything, and we have nothing to pre-warm.
        logger.info(
            "LLM_PROVIDER=%r — not 'huggingface', skipping generator pre-warm.",
            llm_provider or "(unset)",
        )
        return

    try:
        # Imported here (not at module top) so we don't pay the import cost
        # when pre-warming is skipped. transformers/torch are heavy.
        from llm.router import build_router_with_generator_override

        router = build_router_with_generator_override()

        # The override router stores the generator-specific provider in a
        # private dict. If for some reason no override was registered (e.g. HF
        # provider import failed), fall back to the default provider so we
        # don't crash on KeyError.
        generator_provider = router._role_providers.get("generator") or router.provider

        # Only HuggingFaceProvider exposes ``_ensure_loaded`` — Claude and
        # Ollama don't need it because their "model" is a remote API.
        if hasattr(generator_provider, "_ensure_loaded"):
            logger.info(
                "Pre-warming generator model %s (this can take 30-90s on free CPU)...",
                generator_provider.model_name,
            )
            asyncio.run(generator_provider._ensure_loaded())
            logger.info("Pre-warm complete: %s", generator_provider.model_name)
        else:
            logger.info(
                "Generator provider %r does not require pre-warming.",
                generator_provider.provider_name,
            )
    except Exception as exc:
        # Surface the failure prominently so it's visible in Space logs without
        # having to scroll through INFO-level chatter. We also stash the
        # message in a module-level global so the Gradio UI could display it
        # later if we ever wire it up.
        _PREWARM_ERROR = f"{type(exc).__name__}: {exc}"
        logger.error("Pre-warm failed (non-fatal, server still starting): %s", _PREWARM_ERROR)


# Pre-warm runs synchronously at import — Gradio's event loop has not started
# yet so it's safe to call asyncio.run(). After this returns the model weights
# are resident in memory (or pre-warm was skipped) and Gradio can launch.
_prewarm()


# Imports below are intentionally placed AFTER _prewarm() so the heavy
# transformers / torch imports happen inside _prewarm() (timed, logged, error-
# handled) rather than implicitly at top of module. The ``noqa: E402`` markers
# silence the "import not at top of file" lint warnings that this triggers.
from agent.config import AgentConfig                            # noqa: E402
from demo.app import build_app                                  # noqa: E402
from llm.router import build_router_with_generator_override     # noqa: E402


# ── Build the AgentConfig the entire Space will share ──────────────────────
#
# IMPORTANT: We override autonomy_level to "full_auto" here because the
# default ("review_repairs") uses LangGraph's interrupt() to pause for human
# approval, and that pause-and-resume flow is not reliable inside Gradio on
# HF Spaces. For local development the default is fine; for the Space we want
# the agent to run end-to-end without manual intervention.
_config = AgentConfig(autonomy_level="full_auto")
logger.info("Agent config: %s", _config)


# Build the router ONCE at startup so every request reuses the same provider
# instances (and any cached model weights). The router itself is stateless so
# it's safe to share across concurrent Gradio requests.
_router = build_router_with_generator_override()

# Log explicitly which providers got bound. This makes debugging the Space far
# easier — you can scroll the logs and immediately see whether the generator
# is using HF (intended behavior) or Claude (silent fallback when LLM_PROVIDER
# is not set as a Variable).
_generator_provider = _router._role_providers.get("generator") or _router.provider
logger.info(
    "Provider routing: generator=%s/%s | other roles=%s/%s",
    _generator_provider.provider_name,
    _generator_provider.model_name,
    _router.provider.provider_name,
    _router.provider.model_name,
)
if _PREWARM_ERROR:
    logger.error(
        "STARTUP WARNING — generator pre-warm failed earlier: %s. The first generator "
        "call will retry the load lazily; expect a long delay or failure on first request.",
        _PREWARM_ERROR,
    )


# Build the Gradio Blocks app, threading the config and router through so the
# UI uses our pre-built instances instead of constructing fresh ones on each
# request (which would defeat pre-warming).
demo = build_app(config=_config, router=_router)

# .queue() is REQUIRED in Gradio 5+ before .launch() — it initializes the
# pending-message lock and event-handler scheduler. Without it, async
# generators (which is how we stream the agent timeline) raise an exception
# on the first yield.
demo.queue()

# server_name="0.0.0.0" exposes the server on all interfaces (HF Spaces routes
# external traffic to it). PORT is read from the env so HF Spaces' dynamic
# port assignment works; we default to 7860 for local-dev parity.
demo.launch(
    server_name="0.0.0.0",
    server_port=int(os.environ.get("PORT", 7860)),
)
