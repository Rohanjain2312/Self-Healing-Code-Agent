"""
HuggingFace Spaces entry point.

HF Spaces expects app.py in the repository root.
This file delegates to demo/app.py and pre-warms the LLM in the background
so the model is loaded before the first user request arrives.
"""

import asyncio
import logging
import sys
import threading
from pathlib import Path

# Ensure the project root is importable
sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _prewarm() -> None:
    """Load model weights into memory before the first user request.

    Runs in a daemon thread so the Gradio server starts immediately.
    If the provider does not support pre-warming (e.g. Ollama, Mock),
    the function exits silently.
    """
    try:
        from llm.router import LLMRouter
        router = LLMRouter()
        if hasattr(router.provider, "_ensure_loaded"):
            asyncio.run(router.provider._ensure_loaded())
            logger.info("Model pre-warm complete: %s", router.provider.model_name)
        else:
            logger.info(
                "Provider '%s' does not need pre-warming.", router.provider.provider_name
            )
    except Exception as exc:  # pre-warm is best-effort; never crash the server
        logger.warning("Pre-warm failed (non-fatal): %s", exc)


# Start pre-warm in background so Gradio server starts immediately
threading.Thread(target=_prewarm, daemon=True, name="model-prewarm").start()

from demo.app import build_app  # noqa: E402 — import after path setup

demo = build_app()
demo.launch(theme="soft")
