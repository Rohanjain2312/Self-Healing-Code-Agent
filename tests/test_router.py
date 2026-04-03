"""Tests for LLM Router retry and fallback behavior."""
import pytest
from llm.router import LLMRouter
from llm.base import BaseLLMProvider, InferenceRequest, InferenceResponse
from llm.schema_validator import StructuredOutputError


class BadThenGoodProvider(BaseLLMProvider):
    """Returns garbage for the first two calls, then valid JSON on the third."""

    def __init__(self) -> None:
        self.call_count = 0

    @property
    def provider_name(self) -> str:
        return "test"

    @property
    def model_name(self) -> str:
        return "test"

    async def infer(self, request: InferenceRequest) -> InferenceResponse:
        self.call_count += 1
        if self.call_count < 3:
            return InferenceResponse(text="not json", provider="test", model="test")
        return InferenceResponse(
            text='{"code":"def f(): pass","explanation":"ok"}',
            provider="test",
            model="test",
        )


class AlwaysBadProvider(BaseLLMProvider):
    """Always returns unparseable output."""

    @property
    def provider_name(self) -> str:
        return "test"

    @property
    def model_name(self) -> str:
        return "test"

    async def infer(self, request: InferenceRequest) -> InferenceResponse:
        return InferenceResponse(text="garbage", provider="test", model="test")


@pytest.mark.asyncio
async def test_router_retries_on_schema_failure():
    """Router retries up to 3 times and returns valid result on third attempt."""
    provider = BadThenGoodProvider()
    router = LLMRouter(provider=provider)
    result = await router.call(
        role="generator",
        template_key="initial",
        variables={"task_description": "test", "learning_log": ""},
    )
    assert provider.call_count == 3
    assert "def f" in result["code"]


@pytest.mark.asyncio
async def test_router_raises_after_max_retries():
    """Router raises StructuredOutputError after all retries are exhausted."""
    router = LLMRouter(provider=AlwaysBadProvider())
    with pytest.raises(StructuredOutputError):
        await router.call(
            role="generator",
            template_key="initial",
            variables={"task_description": "test", "learning_log": ""},
        )
