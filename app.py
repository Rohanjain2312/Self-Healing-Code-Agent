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
    """Load model weights into memory before Gradio starts accepting requests.

    Runs synchronously so the model is fully loaded before launch().
    API providers (Anthropic) and Ollama do not need pre-warming — skip them.
    Only pre-warm HuggingFace Transformers (local model load takes minutes).

    If ANTHROPIC_API_KEY is set, set LLM_PROVIDER=anthropic explicitly in
    your HF Space variables to skip HuggingFace model loading entirely.
    """
    # API providers don't load model weights locally — skip pre-warm entirely
    llm_provider = os.environ.get("LLM_PROVIDER", "").lower()
    if llm_provider == "anthropic" or os.environ.get("ANTHROPIC_API_KEY"):
        logger.info("Anthropic API provider detected — skipping pre-warm.")
        return

    try:
        from llm.router import LLMRouter
        router = LLMRouter()
        if hasattr(router.provider, "_ensure_loaded"):
            logger.info("Pre-warming model: %s ...", router.provider.model_name)
            asyncio.run(router.provider._ensure_loaded())
            logger.info("Model pre-warm complete: %s", router.provider.model_name)
        else:
            logger.info(
                "Provider '%s' does not need pre-warming.", router.provider.provider_name
            )
    except Exception as exc:  # pre-warm is best-effort; never crash the server
        logger.warning("Pre-warm failed (non-fatal): %s", exc)


# Pre-warm synchronously — blocks until model is loaded, then launch Gradio
_prewarm()

from agent.config import AgentConfig  # noqa: E402
from demo.app import build_app       # noqa: E402 — import after path setup

# All defaults are production-grade — no preset selection needed.
# If ANTHROPIC_API_KEY is set, the router auto-selects Claude API.
# Set LLM_PROVIDER=anthropic explicitly to skip HuggingFace model loading.
_config = AgentConfig()
logger.info("Agent config: %s", _config)

demo = build_app(config=_config)
demo.queue()  # required in Gradio 5 — initializes pending_message_lock before launch
demo.launch(
    server_name="0.0.0.0",
    server_port=int(os.environ.get("PORT", 7860)),
)
