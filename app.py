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
    If the provider does not support pre-warming (e.g. Ollama, Mock),
    the function exits silently.
    """
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

from demo.app import build_app  # noqa: E402 — import after path setup

demo = build_app()
demo.queue()  # required in Gradio 5 — initializes pending_message_lock before launch
demo.launch(
    server_name="0.0.0.0",
    server_port=int(os.environ.get("PORT", 7860)),
)
