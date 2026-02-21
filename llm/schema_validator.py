"""
JSON schema validator for structured LLM outputs.

LLMs frequently return malformed JSON or structurally invalid responses.
This module:
  1. Strips markdown code fences before parsing
  2. Validates against the role's JSON schema
  3. Raises typed exceptions so the router can retry cleanly
"""

import json
import re
from typing import Any

import jsonschema
from jsonschema import ValidationError as JsonSchemaValidationError


class StructuredOutputError(Exception):
    """Raised when LLM output cannot be parsed or validated."""

    def __init__(self, message: str, raw_text: str = "") -> None:
        super().__init__(message)
        self.raw_text = raw_text


def _strip_markdown_fences(text: str) -> str:
    """
    Remove ```json ... ``` or ``` ... ``` wrappers that models often add
    despite being instructed not to.
    """
    # Match optional language tag after opening fence
    fence_pattern = re.compile(r"```(?:json)?\s*([\s\S]*?)\s*```", re.IGNORECASE)
    match = fence_pattern.search(text)
    if match:
        return match.group(1).strip()
    return text.strip()


def _extract_json_object(text: str) -> str:
    """
    Extract the first complete JSON object or array from text.
    Models sometimes prepend explanation text before the JSON.
    """
    # Find the first { or [ and attempt to extract from there
    for start_char, end_char in [('{', '}'), ('[', ']')]:
        idx = text.find(start_char)
        if idx == -1:
            continue
        # Walk forward counting braces to find matching close
        depth = 0
        in_string = False
        escape_next = False
        for i, ch in enumerate(text[idx:], start=idx):
            if escape_next:
                escape_next = False
                continue
            if ch == '\\' and in_string:
                escape_next = True
                continue
            if ch == '"' and not escape_next:
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == start_char:
                depth += 1
            elif ch == end_char:
                depth -= 1
                if depth == 0:
                    return text[idx:i + 1]
    return text


def _coerce_parsed(parsed: dict[str, Any], schema: dict) -> dict[str, Any]:
    """
    Best-effort coercion for common model output mistakes before schema validation.

    Handles the case where the model puts structured JSON under a field that
    should be a plain string (e.g. "code": {"functions": [...]} instead of
    "code": "def f(): ..."). Converts those nested values back to strings.
    """
    required_string_fields = [
        field
        for field, defn in schema.get("properties", {}).items()
        if defn.get("type") == "string"
        and field in schema.get("required", [])
    ]
    for field in required_string_fields:
        value = parsed.get(field)
        # Only coerce container types (dict/list) — the model returned structured
        # JSON where it should have put Python source text. Primitives (int, bool,
        # None) are genuine type errors and should fail validation normally.
        if isinstance(value, (dict, list)):
            parsed[field] = json.dumps(value)
    return parsed


def parse_and_validate(raw_text: str, schema: dict) -> dict[str, Any]:
    """
    Parse raw LLM text into a validated dict.

    Raises StructuredOutputError if:
      - text cannot be parsed as JSON
      - parsed object fails schema validation after coercion attempts
    """
    cleaned = _strip_markdown_fences(raw_text)
    extracted = _extract_json_object(cleaned)

    try:
        # strict=False accepts literal control characters (raw tabs/newlines)
        # inside JSON string values — a common LLM mistake when writing code.
        parsed = json.loads(extracted, strict=False)
    except json.JSONDecodeError as exc:
        raise StructuredOutputError(
            f"JSON parse failed: {exc}",
            raw_text=raw_text,
        ) from exc

    if schema:
        # Attempt coercion before validation so type mismatches don't burn retries
        parsed = _coerce_parsed(parsed, schema)
        try:
            jsonschema.validate(instance=parsed, schema=schema)
        except JsonSchemaValidationError as exc:
            raise StructuredOutputError(
                f"Schema validation failed: {exc.message}",
                raw_text=raw_text,
            ) from exc

    return parsed
