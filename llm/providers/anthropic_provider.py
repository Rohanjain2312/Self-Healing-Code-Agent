"""
Anthropic Claude API provider.

Uses the official anthropic Python SDK for async inference.
Default model: Haiku 4.5 ($1/$5 per MTok) — fast, cheap, good enough for
LeetCode-level code generation. Override with ANTHROPIC_MODEL env var.
"""

import logging
import os

from ..base import BaseLLMProvider, InferenceRequest, InferenceResponse

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "claude-haiku-4-5-20251001"


class AnthropicProvider(BaseLLMProvider):
    """
    Anthropic Claude API provider.

    Uses the official anthropic Python SDK for async inference.
    Default model: Haiku 4.5 ($1/$5 per MTok) — fast, cheap, good enough for
    LeetCode-level code generation. Override with ANTHROPIC_MODEL env var.
    """

    def __init__(self, model: str | None = None, api_key: str | None = None) -> None:
        self._model = model or os.environ.get("ANTHROPIC_MODEL", _DEFAULT_MODEL)
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        if not self._api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY environment variable is required. "
                "Get your key at https://console.anthropic.com/api-keys"
            )

    @property
    def provider_name(self) -> str:
        return "anthropic"

    @property
    def model_name(self) -> str:
        return self._model

    async def infer(self, request: InferenceRequest) -> InferenceResponse:
        """
        Execute inference via the Anthropic Messages API.

        Uses the async client with a 60-second timeout — API calls are fast
        compared to local inference. Raises RuntimeError on auth or API errors
        with clear diagnostic messages.
        """
        import anthropic

        client = anthropic.AsyncAnthropic(
            api_key=self._api_key,
            timeout=60.0,
        )

        try:
            response = await client.messages.create(
                model=self._model,
                max_tokens=request.max_new_tokens,
                temperature=request.temperature,
                system=request.system_prompt,
                messages=[{"role": "user", "content": request.user_prompt}],
            )
        except anthropic.AuthenticationError as exc:
            raise RuntimeError(
                "Anthropic API authentication failed. Check your ANTHROPIC_API_KEY."
            ) from exc
        except anthropic.APIError as exc:
            raise RuntimeError(f"Anthropic API error: {exc}") from exc

        text = response.content[0].text if response.content else ""

        return InferenceResponse(
            text=text,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            provider=self.provider_name,
            model=self._model,
        )
