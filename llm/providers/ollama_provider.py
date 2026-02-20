"""
Ollama provider — local CPU inference via the Ollama REST API.

Ollama runs as a local daemon and serves models via HTTP.
Default model: llama3 (8B) — small enough for CPU-only MacBook development.

The provider expects Ollama to be running at http://localhost:11434.
Environment variable OLLAMA_BASE_URL overrides the default host.
"""

import json
import os
import asyncio
import httpx
from ..base import BaseLLMProvider, InferenceRequest, InferenceResponse

_DEFAULT_BASE_URL = "http://localhost:11434"
_DEFAULT_MODEL = "llama3.1:8b"
# CPU inference on an 8B model can take several minutes per request.
# 600s (10 min) gives ample headroom; override with OLLAMA_TIMEOUT env var.
_TIMEOUT_SECONDS = float(os.environ.get("OLLAMA_TIMEOUT", "600"))


class OllamaProvider(BaseLLMProvider):
    """
    Calls Ollama's /api/generate endpoint.

    Uses httpx async client. The combined system + user prompt is sent
    as a single request with system field to guide instruction-tuned models.
    """

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self._model = model or os.environ.get("OLLAMA_MODEL", _DEFAULT_MODEL)
        self._base_url = base_url or os.environ.get("OLLAMA_BASE_URL", _DEFAULT_BASE_URL)

    @property
    def provider_name(self) -> str:
        return "ollama"

    @property
    def model_name(self) -> str:
        return self._model

    async def infer(self, request: InferenceRequest) -> InferenceResponse:
        payload = {
            "model": self._model,
            "system": request.system_prompt,
            "prompt": request.user_prompt,
            "stream": False,
            "options": {
                "temperature": request.temperature,
                "num_predict": request.max_new_tokens,
            },
        }

        # Use separate connect vs read timeouts: connect must be fast,
        # but read can be very long on CPU inference (minutes for 8B models).
        timeout_config = httpx.Timeout(connect=10.0, read=_TIMEOUT_SECONDS, write=30.0, pool=10.0)
        async with httpx.AsyncClient(timeout=timeout_config) as client:
            try:
                response = await client.post(
                    f"{self._base_url}/api/generate",
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()
            except httpx.ConnectError as exc:
                raise RuntimeError(
                    f"Ollama not reachable at {self._base_url}. "
                    "Ensure `ollama serve` is running."
                ) from exc
            except httpx.ReadTimeout as exc:
                raise RuntimeError(
                    f"Ollama read timeout after {_TIMEOUT_SECONDS}s for model '{self._model}'. "
                    "The model may be under load. Set OLLAMA_TIMEOUT env var to increase the limit "
                    "or switch to a smaller model via OLLAMA_MODEL."
                ) from exc

        return InferenceResponse(
            text=data.get("response", ""),
            # Ollama reports eval_count (output tokens) and prompt_eval_count
            input_tokens=data.get("prompt_eval_count", -1),
            output_tokens=data.get("eval_count", -1),
            provider=self.provider_name,
            model=self._model,
        )

    def is_available_sync(self) -> bool:
        """
        Synchronous health-check using httpx's sync client.

        Used by _resolve_provider() at router construction time, which runs
        before any async event loop is active. Avoids the
        asyncio.get_event_loop().run_until_complete() pattern that raises
        RuntimeError when called from within an already-running loop.
        """
        try:
            r = httpx.get(f"{self._base_url}/api/tags", timeout=5.0)
            return r.status_code == 200
        except Exception:
            return False

    async def is_available(self) -> bool:
        """Async health-check: True if Ollama daemon is reachable."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                r = await client.get(f"{self._base_url}/api/tags")
                return r.status_code == 200
        except Exception:
            return False

    async def ensure_model_pulled(self) -> None:
        """
        Pull model if not already present. Called lazily on first use.
        This can take several minutes on first run.
        """
        async with httpx.AsyncClient(timeout=600.0) as client:
            await client.post(
                f"{self._base_url}/api/pull",
                json={"name": self._model, "stream": False},
            )
