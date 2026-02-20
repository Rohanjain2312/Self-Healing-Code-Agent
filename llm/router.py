"""
LLM Router — the single point of contact between agent nodes and model inference.

Nodes NEVER call providers directly. This ensures:
  - Provider selection logic lives in one place
  - Schema validation is always applied
  - Retry-on-failure is handled uniformly
  - Context building is applied consistently

Provider selection priority:
  1. Explicit provider passed to Router constructor (test injection)
  2. Environment variable: LLM_PROVIDER = ollama | huggingface | mock
  3. Auto-detection: Ollama health check → HuggingFace → Mock fallback
"""

import asyncio
import logging
import os
from typing import Any

from .base import BaseLLMProvider, InferenceRequest
from .prompt_loader import get_system_prompt, get_schema, render_template
from .context_builder import build_context
from .schema_validator import parse_and_validate, StructuredOutputError

logger = logging.getLogger(__name__)

_MAX_RETRIES = 3
_RETRY_TEMPERATURE_INCREMENT = 0.1  # raise temp on retry to escape degenerate outputs


def _resolve_provider() -> BaseLLMProvider:
    """
    Select provider based on environment signals.

    Resolution order ensures the system works identically across:
      - MacBook (Ollama local)
      - Colab / HF Spaces (HuggingFace)
      - CI / unit tests (Mock)
    """
    env_provider = os.environ.get("LLM_PROVIDER", "").lower()

    if env_provider == "mock":
        from .providers.mock_provider import MockProvider
        return MockProvider()

    if env_provider == "huggingface":
        from .providers.hf_provider import HuggingFaceProvider
        return HuggingFaceProvider()

    if env_provider == "ollama":
        from .providers.ollama_provider import OllamaProvider
        return OllamaProvider()

    # Auto-detect: try Ollama first using the synchronous health-check.
    # is_available_sync() uses httpx's sync client so it never touches the
    # event loop — safe to call at router construction time regardless of
    # whether an async loop is already running.
    try:
        from .providers.ollama_provider import OllamaProvider
        provider = OllamaProvider()
        if provider.is_available_sync():
            logger.info("Auto-selected Ollama provider at %s", provider._base_url)
            return provider
    except Exception:
        pass

    # Check if transformers is installed (Colab / HF Spaces environment)
    try:
        import transformers  # noqa: F401
        from .providers.hf_provider import HuggingFaceProvider
        logger.info("Auto-selected HuggingFace Transformers provider")
        return HuggingFaceProvider()
    except ImportError:
        pass

    # Final fallback — no models available
    logger.warning(
        "No LLM provider available. Using Mock provider. "
        "Set LLM_PROVIDER=ollama or LLM_PROVIDER=huggingface to use real models."
    )
    from .providers.mock_provider import MockProvider
    return MockProvider()


class LLMRouter:
    """
    Stateless orchestrator that combines prompt loading, context building,
    inference, and schema validation into a single async call per node.
    """

    def __init__(self, provider: BaseLLMProvider | None = None) -> None:
        # Allow explicit injection for testing; otherwise auto-resolve
        self._provider = provider or _resolve_provider()
        logger.info(
            "LLMRouter initialized with provider=%s model=%s",
            self._provider.provider_name,
            self._provider.model_name,
        )

    async def call(
        self,
        role: str,
        template_key: str,
        variables: dict[str, Any],
        max_new_tokens: int = 1024,
        base_temperature: float = 0.2,
    ) -> dict[str, Any]:
        """
        Full pipeline: load prompt → build context → infer → validate → return.

        Retries up to _MAX_RETRIES times on StructuredOutputError.
        Each retry slightly increases temperature to escape stuck outputs.

        Args:
            role: Agent role matching a YAML file in prompts/
            template_key: Named template within that YAML (e.g. 'initial', 'repair')
            variables: Template substitution variables
            max_new_tokens: Generation length cap
            base_temperature: Starting temperature; incremented on retries

        Returns:
            Validated dict matching the role's JSON schema

        Raises:
            StructuredOutputError: If all retries exhausted with invalid output
            RuntimeError: If provider is unreachable
        """
        system_prompt = get_system_prompt(role)
        schema = get_schema(role)
        rendered = render_template(role, template_key, variables)
        user_prompt = build_context(rendered, variables)

        last_error: StructuredOutputError | None = None

        for attempt in range(_MAX_RETRIES):
            temperature = base_temperature + (attempt * _RETRY_TEMPERATURE_INCREMENT)

            request = InferenceRequest(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_new_tokens=max_new_tokens,
                temperature=min(temperature, 1.0),
                metadata={"role": role, "attempt": attempt},
            )

            try:
                response = await self._provider.infer(request)
                logger.debug(
                    "role=%s attempt=%d input_tokens=%d output_tokens=%d",
                    role,
                    attempt,
                    response.input_tokens,
                    response.output_tokens,
                )
                result = parse_and_validate(response.text, schema)
                return result

            except StructuredOutputError as exc:
                last_error = exc
                logger.warning(
                    "role=%s attempt=%d/%d schema validation failed: %s",
                    role,
                    attempt + 1,
                    _MAX_RETRIES,
                    exc,
                )
                if attempt < _MAX_RETRIES - 1:
                    # Brief backoff before retry
                    await asyncio.sleep(0.5 * (attempt + 1))

        raise last_error or StructuredOutputError(
            f"All {_MAX_RETRIES} retries exhausted for role={role}"
        )

    @property
    def provider(self) -> BaseLLMProvider:
        return self._provider
