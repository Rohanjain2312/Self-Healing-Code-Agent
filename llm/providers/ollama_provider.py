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
_DEFAULT_MODEL = "llama3"
# Ollama /api/generate timeout — models can be slow on first load
_TIMEOUT_SECONDS = 180.0


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

        async with httpx.AsyncClient(timeout=_TIMEOUT_SECONDS) as client:
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

        return InferenceResponse(
            text=data.get("response", ""),
            # Ollama reports eval_count (output tokens) and prompt_eval_count
            input_tokens=data.get("prompt_eval_count", -1),
            output_tokens=data.get("eval_count", -1),
            provider=self.provider_name,
            model=self._model,
        )

    async def is_available(self) -> bool:
        """Health-check: True if Ollama daemon is reachable."""
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
