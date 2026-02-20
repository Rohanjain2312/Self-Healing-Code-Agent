"""
Abstract base class for all LLM providers.

All providers must implement async inference so the event loop never blocks.
Token counting is best-effort — providers that cannot count exactly return -1.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class InferenceRequest:
    """Normalized request passed to any provider."""
    system_prompt: str
    user_prompt: str
    # Maximum tokens to generate; providers clip to their own limits
    max_new_tokens: int = 1024
    temperature: float = 0.2
    # Caller-supplied metadata — not forwarded to models
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class InferenceResponse:
    """Normalized response returned from any provider."""
    text: str
    # Approximate input tokens used; -1 if provider cannot report
    input_tokens: int = -1
    # Approximate output tokens generated; -1 if provider cannot report
    output_tokens: int = -1
    provider: str = ""
    model: str = ""


class BaseLLMProvider(ABC):
    """
    Providers are stateless wrappers around model backends.
    They handle authentication, batching, and retry at the HTTP level.
    Schema validation and structured output parsing happen in the Router.
    """

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Identifier used in logs and InferenceResponse.provider."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Active model identifier."""

    @abstractmethod
    async def infer(self, request: InferenceRequest) -> InferenceResponse:
        """
        Execute inference asynchronously.

        Must not block the event loop. Implementations using synchronous
        libraries (e.g. transformers pipeline) must run in a thread executor.
        """
