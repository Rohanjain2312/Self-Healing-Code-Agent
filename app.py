"""
HuggingFace Spaces entry point.

HF Spaces expects app.py in the repository root.
This file pre-warms the LLM synchronously before launching Gradio, so the
model is fully loaded and ready the moment the first user request arrives.
"""

import asyncio
import logging
import os
import sys
from pathlib import Path

# Ensure the project root is importable
sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _prewarm() -> None:
    """Load generator model weights into memory before Gradio starts accepting requests.

    Only pre-warm when LLM_PROVIDER=huggingface (generator role). Claude (Anthropic)
    and Ollama do not load local model weights — skip pre-warming for them.
    """
    llm_provider = os.environ.get("LLM_PROVIDER", "").lower()
    if llm_provider != "huggingface":
        logger.info("Generator provider is not HuggingFace — skipping pre-warm.")
        return

    try:
        from llm.router import build_router_with_generator_override
        router = build_router_with_generator_override()
        # Only pre-warm the generator provider (HuggingFace); Claude never needs it.
        generator_provider = router._role_providers.get("generator") or router.provider
        if hasattr(generator_provider, "_ensure_loaded"):
            logger.info("Pre-warming generator model: %s ...", generator_provider.model_name)
            asyncio.run(generator_provider._ensure_loaded())
            logger.info("Model pre-warm complete: %s", generator_provider.model_name)
        else:
            logger.info(
                "Generator provider '%s' does not need pre-warming.",
                generator_provider.provider_name,
            )
    except Exception as exc:  # pre-warm is best-effort; never crash the server
        logger.warning("Pre-warm failed (non-fatal): %s", exc)


# Pre-warm synchronously — blocks until model is loaded, then launch Gradio
_prewarm()

from agent.config import AgentConfig                    # noqa: E402
from demo.app import build_app                          # noqa: E402 — import after path setup
from llm.router import build_router_with_generator_override  # noqa: E402

_config = AgentConfig()
logger.info("Agent config: %s", _config)

_router = build_router_with_generator_override()

demo = build_app(config=_config, router=_router)
demo.queue()  # required in Gradio 5 — initializes pending_message_lock before launch
demo.launch(
    server_name="0.0.0.0",
    server_port=int(os.environ.get("PORT", 7860)),
)
