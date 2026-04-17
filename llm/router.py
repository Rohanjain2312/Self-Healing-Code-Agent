"""
LLM Router — the single point of contact between agent nodes and model inference.

Nodes NEVER call providers directly. This ensures:
  - Provider selection logic lives in one place
  - Schema validation is always applied
  - Retry-on-failure is handled uniformly
  - Context building is applied consistently

Provider selection priority:
  1. Explicit provider passed to Router constructor (test injection)
  2. Environment variable: LLM_PROVIDER = mock | huggingface | ollama | anthropic
  3. Auto-detection: ANTHROPIC_API_KEY present → AnthropicProvider
  4. Auto-detection: Ollama health check → HuggingFace → Mock fallback
"""

import asyncio
import logging
import os
import time
from typing import Any

from .base import BaseLLMProvider, InferenceRequest
from .prompt_loader import get_system_prompt, get_schema, render_template, get_raw_template
from .context_builder import build_context
from .schema_validator import parse_and_validate, StructuredOutputError

logger = logging.getLogger(__name__)

# Enable LangSmith tracing when LANGCHAIN_TRACING_V2 is set in the environment.
# All other LangSmith configuration (API key, endpoint) is also read from env vars.
_LANGSMITH_ENABLED = bool(os.environ.get("LANGCHAIN_TRACING_V2"))
if _LANGSMITH_ENABLED:
    os.environ.setdefault("LANGCHAIN_PROJECT", "self-healing-agent")
    logger.info("LangSmith tracing enabled — project=%s", os.environ["LANGCHAIN_PROJECT"])

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

    if env_provider == "anthropic":
        from .providers.anthropic_provider import AnthropicProvider
        return AnthropicProvider()

    # Auto-detect: Anthropic API key present → use Claude (fast, no model download)
    if os.environ.get("ANTHROPIC_API_KEY"):
        try:
            from .providers.anthropic_provider import AnthropicProvider
            logger.info("Auto-selected Anthropic provider (ANTHROPIC_API_KEY found)")
            return AnthropicProvider()
        except Exception as exc:
            logger.warning("Anthropic provider init failed: %s", exc)

    # Auto-detect: try Ollama next using the synchronous health-check.
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

    def __init__(
        self,
        provider: BaseLLMProvider | None = None,
        role_providers: dict[str, BaseLLMProvider] | None = None,
    ) -> None:
        # Allow explicit injection for testing; otherwise auto-resolve
        self._provider = provider or _resolve_provider()
        # Per-role provider overrides (Fix 17): roles not in this dict fall back to _provider
        self._role_providers: dict[str, BaseLLMProvider] = role_providers or {}
        logger.info(
            "LLMRouter initialized with provider=%s model=%s role_overrides=%s",
            self._provider.provider_name,
            self._provider.model_name,
            list(self._role_providers),
        )

    def _get_provider(self, role: str) -> BaseLLMProvider:
        """
        Return the provider for a given role.

        Checks role_providers first (Fix 17 — per-role model routing).
        Falls back to the default provider if no override is registered.
        Logs which model handled each call for observability.
        """
        if role in self._role_providers:
            p = self._role_providers[role]
            logger.debug(
                "Role '%s' routed to override provider=%s model=%s",
                role, p.provider_name, p.model_name,
            )
            return p
        return self._provider

    async def call(
        self,
        role: str,
        template_key: str,
        variables: dict[str, Any],
        max_new_tokens: int = 2048,
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
        raw_template = get_raw_template(role, template_key)
        rendered = render_template(role, template_key, variables)
        user_prompt = build_context(rendered, variables, template_str=raw_template)

        last_error: StructuredOutputError | None = None

        for attempt in range(_MAX_RETRIES):
            temperature = base_temperature + (attempt * _RETRY_TEMPERATURE_INCREMENT)

            request = InferenceRequest(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_new_tokens=max_new_tokens,
                temperature=min(temperature, 1.0),
                metadata={"role": role, "template_key": template_key, "attempt": attempt},
            )

            try:
                start = time.monotonic()
                response = await self._get_provider(role).infer(request)
                elapsed = time.monotonic() - start
                # Token counts: Ollama=real, HuggingFace=-1, Mock=word approx.
                # Log as-is; consumers must treat -1 as "unknown".
                logger.info(
                    "LLM_CALL role=%s tokens_in=%d tokens_out=%d latency=%.2fs attempt=%d",
                    role,
                    response.input_tokens,
                    response.output_tokens,
                    elapsed,
                    attempt,
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

    async def get_raw_response(
        self,
        role: str,
        template_key: str,
        variables: dict[str, Any],
        max_new_tokens: int = 2048,
        base_temperature: float = 0.2,
    ) -> str:
        """
        Return the raw LLM response text without schema validation.

        Used by the ReAct debugger loop to receive either a tool-use JSON
        or a final-diagnosis JSON before deciding which schema to apply.

        Args:
            role: Agent role matching a YAML file in prompts/
            template_key: Named template within that YAML
            variables: Template substitution variables
            max_new_tokens: Generation length cap
            base_temperature: Temperature for sampling

        Returns:
            Raw text string from the LLM provider.

        Raises:
            RuntimeError: If provider is unreachable.
        """
        system_prompt = get_system_prompt(role)
        raw_template = get_raw_template(role, template_key)
        rendered = render_template(role, template_key, variables)
        user_prompt = build_context(rendered, variables, template_str=raw_template)

        request = InferenceRequest(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_new_tokens=max_new_tokens,
            temperature=base_temperature,
            metadata={"role": role, "template_key": template_key, "attempt": 0},
        )
        response = await self._get_provider(role).infer(request)
        logger.info(
            "RAW_CALL role=%s template=%s tokens_in=%d tokens_out=%d",
            role, template_key, response.input_tokens, response.output_tokens,
        )
        return response.text

    def _get_fallback(self, role: str, raw_text: str = "") -> dict[str, Any]:
        """
        Return a safe default response when all retries are exhausted.

        Fallbacks are role-specific and designed to keep the agent running
        rather than crashing. Nodes that use fallbacks record themselves in
        state["degraded_nodes"] so operators can audit degraded runs.
        """
        if role == "debugger":
            return {
                "root_cause": "Unable to diagnose — LLM output was malformed.",
                "failure_category": "unknown",
                "repair_strategy": "Regenerate from scratch with a fresh approach.",
                "confidence": 0.1,
            }
        if role == "qa_adversarial":
            return {
                "test_code": "assert True  # fallback: QA agent failed to generate tests",
                "test_cases_description": ["fallback smoke test"],
            }
        if role == "generator":
            # Try to extract code from markdown fences in the raw LLM output
            import re
            match = re.search(r"```(?:python)?\n(.*?)```", raw_text, re.DOTALL)
            code = match.group(1).strip() if match else "# generation failed\ndef placeholder(): pass"
            return {
                "code": code,
                "explanation": "Extracted from raw output (schema validation failed).",
            }
        if role == "memory_summarizer":
            return {
                "updated_lessons": [],
                "summary": "Memory summarization failed — no lessons extracted.",
            }
        if role == "critic":
            # On failure, approve conservatively — don't force a re-repair cycle
            return {
                "verdict": "approve",
                "issues": [],
                "confidence": 0.5,
            }
        # Generic fallback for any other role
        return {"error": f"Fallback for role={role}: LLM output was invalid."}

    async def call_with_fallback(
        self,
        role: str,
        template_key: str,
        variables: dict[str, Any],
        max_new_tokens: int = 2048,
        base_temperature: float = 0.2,
    ) -> tuple[dict[str, Any], bool]:
        """
        Like call(), but returns a (result, used_fallback) tuple instead of raising.

        When used_fallback is True, the caller should append its node name to
        state["degraded_nodes"] so operators can audit which nodes degraded.

        Use this for non-critical roles (debugger, memory_summarizer) where
        partial / approximate results are acceptable. Generator and QA should
        use call() directly since their failure is unrecoverable.
        """
        # Keep the last raw response text for fallback code extraction
        last_raw: str = ""

        system_prompt = get_system_prompt(role)
        schema = get_schema(role)
        raw_template = get_raw_template(role, template_key)
        rendered = render_template(role, template_key, variables)
        user_prompt = build_context(rendered, variables, template_str=raw_template)

        for attempt in range(_MAX_RETRIES):
            temperature = base_temperature + (attempt * _RETRY_TEMPERATURE_INCREMENT)
            request = InferenceRequest(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_new_tokens=max_new_tokens,
                temperature=min(temperature, 1.0),
                metadata={"role": role, "template_key": template_key, "attempt": attempt},
            )
            try:
                start = time.monotonic()
                response = await self._get_provider(role).infer(request)
                elapsed = time.monotonic() - start
                last_raw = response.text
                logger.info(
                    "LLM_CALL role=%s tokens_in=%d tokens_out=%d latency=%.2fs attempt=%d",
                    role, response.input_tokens, response.output_tokens, elapsed, attempt,
                )
                result = parse_and_validate(response.text, schema)
                return result, False  # success — no fallback used
            except StructuredOutputError as exc:
                logger.warning(
                    "role=%s attempt=%d/%d schema validation failed (call_with_fallback): %s",
                    role, attempt + 1, _MAX_RETRIES, exc,
                )
                if attempt < _MAX_RETRIES - 1:
                    await asyncio.sleep(0.5 * (attempt + 1))

        logger.error(
            "role=%s all retries exhausted — using fallback response", role
        )
        return self._get_fallback(role, last_raw), True

    @property
    def provider(self) -> BaseLLMProvider:
        return self._provider


def build_router_with_generator_override() -> "LLMRouter":
    """Build a router where the generator role uses a local weak model and
    all other roles use Claude.

    This is the *intended* production routing for the project: the generator
    is intentionally weak (Llama-3.2-3B via HF or Llama-3.2-3B via Ollama) so
    the diagnose → repair loop is exercised meaningfully on most tasks. Claude
    handles the smarter roles where a strong model materially improves output:
    QA (adversarial test design), debugger (root-cause analysis), critic
    (correctness review), and memory_summarizer (lesson distillation).

    Detection order for the generator override:
      1. ``LLM_PROVIDER=huggingface`` env var → use HuggingFaceProvider
         (this is the path used on HF Spaces — MUST be set as a Variable on
         the Space or the generator silently falls through to Claude).
      2. Ollama reachable on localhost → use OllamaProvider with the model
         in ``OLLAMA_GENERATOR_MODEL`` (default ``llama3.2:3b``).
      3. Neither available → fall through to a plain LLMRouter with no role
         overrides; ALL roles (including generator) use Claude.

    The third case is the silent-fallback that causes the "I deployed to HF
    Spaces and the agent always succeeds on iteration 0" symptom — Claude is
    too good at the example tasks to trigger the repair loop. Always check
    the log line below to confirm which branch was taken.
    """
    import os

    # Default provider for the smart roles. Falls back to the auto-resolved
    # provider (which may be Mock) if the Anthropic SDK / API key is missing.
    try:
        from .providers.anthropic_provider import AnthropicProvider
        default_provider = AnthropicProvider()
    except Exception as exc:
        logger.warning("AnthropicProvider unavailable, falling back to auto-resolve: %s", exc)
        default_provider = _resolve_provider()

    env_provider = os.environ.get("LLM_PROVIDER", "").lower()

    # Make the routing decision visible in logs — when debugging an HF Space
    # this is the first line to grep for. Print the env signals we considered,
    # so you can immediately see whether LLM_PROVIDER was set at all.
    logger.info(
        "build_router_with_generator_override: env LLM_PROVIDER=%r, "
        "ANTHROPIC_API_KEY=%s, default_provider=%s/%s",
        env_provider or "(unset)",
        "set" if os.environ.get("ANTHROPIC_API_KEY") else "missing",
        default_provider.provider_name,
        default_provider.model_name,
    )

    # ── Branch 1: HF Spaces ──────────────────────────────────────────────
    # The Space's "Variables" panel must include LLM_PROVIDER=huggingface
    # for this branch to be taken. transformers + torch are imported lazily
    # inside HuggingFaceProvider so this is cheap when the env var is unset.
    if env_provider == "huggingface":
        try:
            from .providers.hf_provider import HuggingFaceProvider
            generator_provider = HuggingFaceProvider()
            logger.info(
                "Generator -> HuggingFaceProvider(%s) | Other roles -> %s(%s)",
                generator_provider.model_name,
                default_provider.provider_name, default_provider.model_name,
            )
            return LLMRouter(
                provider=default_provider,
                role_providers={"generator": generator_provider},
            )
        except Exception as exc:
            # Don't crash the Space if the HF model can't load — fall through
            # to the all-Claude router so the rest of the agent still works.
            logger.warning("HuggingFaceProvider failed (%s), generator uses default.", exc)
            return LLMRouter(provider=default_provider)

    # ── Branch 2: Local MacBook with Ollama ─────────────────────────────
    # The sync health-check uses httpx's sync client so this is safe to call
    # at router construction time even if an event loop is already running.
    try:
        from .providers.ollama_provider import OllamaProvider
        generator_model = os.environ.get("OLLAMA_GENERATOR_MODEL", "llama3.2:3b")
        ollama = OllamaProvider(model=generator_model)
        if ollama.is_available_sync():
            logger.info(
                "Generator -> OllamaProvider(%s) | Other roles -> %s(%s)",
                generator_model,
                default_provider.provider_name, default_provider.model_name,
            )
            return LLMRouter(
                provider=default_provider,
                role_providers={"generator": ollama},
            )
        logger.info("Ollama not reachable — generator uses default provider.")
    except Exception as exc:
        logger.warning("Ollama check failed (%s), generator uses default.", exc)

    # ── Branch 3: Silent fallback — all roles use Claude ────────────────
    # If you hit this branch on HF Spaces it means LLM_PROVIDER was not set
    # as a Variable. The agent will still work but every role uses Claude
    # so the repair loop will rarely trigger.
    logger.warning(
        "No generator override applied — ALL roles will use %s. "
        "Set LLM_PROVIDER=huggingface (HF Spaces) or run Ollama locally to "
        "enable the small-model generator and the self-healing repair loop.",
        default_provider.provider_name,
    )
    return LLMRouter(provider=default_provider)
