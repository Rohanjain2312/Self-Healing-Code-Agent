"""
Tests for JSON schema validation and extraction from LLM raw output.
"""

import pytest
from llm.schema_validator import parse_and_validate, StructuredOutputError

_SIMPLE_SCHEMA = {
    "type": "object",
    "required": ["code", "explanation"],
    "properties": {
        "code": {"type": "string"},
        "explanation": {"type": "string"},
    },
}


def test_valid_json():
    raw = '{"code": "def f(): pass", "explanation": "simple"}'
    result = parse_and_validate(raw, _SIMPLE_SCHEMA)
    assert result["code"] == "def f(): pass"


def test_json_with_markdown_fence():
    raw = '```json\n{"code": "def f(): pass", "explanation": "simple"}\n```'
    result = parse_and_validate(raw, _SIMPLE_SCHEMA)
    assert result["code"] == "def f(): pass"


def test_json_with_prose_prefix():
    raw = 'Here is the code:\n{"code": "def f(): pass", "explanation": "simple"}'
    result = parse_and_validate(raw, _SIMPLE_SCHEMA)
    assert result["code"] == "def f(): pass"


def test_invalid_json_raises():
    raw = "this is not json"
    with pytest.raises(StructuredOutputError):
        parse_and_validate(raw, _SIMPLE_SCHEMA)


def test_schema_violation_raises():
    raw = '{"code": 123}'  # code should be string, missing explanation
    with pytest.raises(StructuredOutputError):
        parse_and_validate(raw, _SIMPLE_SCHEMA)


def test_no_schema_skips_validation():
    raw = '{"anything": true}'
    result = parse_and_validate(raw, {})
    assert result == {"anything": True}


def test_nested_json_extraction():
    raw = 'Response: {"code": "x=1", "explanation": "assigns x"} done.'
    result = parse_and_validate(raw, _SIMPLE_SCHEMA)
    assert result["code"] == "x=1"
