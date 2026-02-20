"""
Mock LLM provider for deterministic unit testing.

Returns pre-registered JSON responses keyed by role.
Falls back to a generic valid response when no fixture is registered.
Never makes network calls — safe for offline CI environments.
"""

import json
from ..base import BaseLLMProvider, InferenceRequest, InferenceResponse

# Default responses per agent role — structurally valid per each schema
_DEFAULT_FIXTURES: dict[str, dict] = {
    "generator": {
        "code": (
            "def solve(data):\n"
            "    # Minimal placeholder implementation\n"
            "    if not data:\n"
            "        return []\n"
            "    return sorted(data)\n"
        ),
        "explanation": "Placeholder implementation returning sorted data.",
    },
    "qa_adversarial": {
        "test_code": (
            "result = solve([])\n"
            "assert result == [], 'Empty input should return empty list'\n"
            "result = solve([3, 1, 2])\n"
            "assert result == [1, 2, 3], 'Should return sorted list'\n"
        ),
        "test_cases_description": [
            "Empty input returns empty list",
            "Standard list is sorted correctly",
        ],
    },
    "debugger": {
        "root_cause": "Placeholder: no actual failure detected in mock mode.",
        "failure_category": "other",
        "repair_strategy": "No repair needed in mock mode.",
        "confidence": 0.5,
    },
    "memory_summarizer": {
        "lessons": [
            "Always validate empty inputs before processing.",
        ],
    },
}


class MockProvider(BaseLLMProvider):
    """
    Deterministic provider for testing without model dependencies.

    Callers may inject custom fixtures to test specific behaviour.
    Role is extracted from InferenceRequest.metadata['role'].
    """

    def __init__(self, fixtures: dict[str, dict] | None = None) -> None:
        # Allow tests to override per-role responses
        self._fixtures = {**_DEFAULT_FIXTURES, **(fixtures or {})}

    @property
    def provider_name(self) -> str:
        return "mock"

    @property
    def model_name(self) -> str:
        return "mock-v1"

    async def infer(self, request: InferenceRequest) -> InferenceResponse:
        role = request.metadata.get("role", "generator")
        payload = self._fixtures.get(role, self._fixtures["generator"])
        text = json.dumps(payload)
        return InferenceResponse(
            text=text,
            input_tokens=len(request.user_prompt.split()),
            output_tokens=len(text.split()),
            provider=self.provider_name,
            model=self.model_name,
        )

    def register_fixture(self, role: str, response: dict) -> None:
        """Register a custom fixture for a given role during testing."""
        self._fixtures[role] = response
