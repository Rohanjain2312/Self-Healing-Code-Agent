"""
Tests for JSON schema validation and extraction from LLM raw output.
"""

import pytest
from llm.schema_validator import parse_and_validate, StructuredOutputError

_SIMPLE_SCHEMA = {
    "type": "object",
    "required": ["code"],
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
    # code is a number — coercion only handles dict/list → string, not int → string
    raw = '{"code": 123}'
    with pytest.raises(StructuredOutputError):
        parse_and_validate(raw, _SIMPLE_SCHEMA)


def test_nested_dict_in_required_string_field_is_coerced():
    # Model returns a nested dict under "code" instead of a Python source string.
    # _coerce_parsed should convert it to a JSON string so validation passes.
    raw = '{"code": {"functions": [{"name": "f"}]}, "explanation": "test"}'
    result = parse_and_validate(raw, _SIMPLE_SCHEMA)
    assert isinstance(result["code"], str)
    assert "functions" in result["code"]  # JSON-serialised representation


def test_no_schema_skips_validation():
    raw = '{"anything": true}'
    result = parse_and_validate(raw, {})
    assert result == {"anything": True}


def test_nested_json_extraction():
    raw = 'Response: {"code": "x=1", "explanation": "assigns x"} done.'
    result = parse_and_validate(raw, _SIMPLE_SCHEMA)
    assert result["code"] == "x=1"


def test_literal_newlines_in_string_value_parse():
    # Model embeds real (literal) newlines inside a JSON string value instead
    # of \n escape sequences — strict JSON rejects this; strict=False accepts it.
    _QA_SCHEMA = {
        "type": "object",
        "required": ["test_code"],
        "properties": {"test_code": {"type": "string"}},
    }
    # Build a string that contains a literal newline character inside the JSON value
    raw_literal = '{"test_code": "assert f([]) == []\n assert f([1]) == [1]\n"}'
    result = parse_and_validate(raw_literal, _QA_SCHEMA)
    assert isinstance(result["test_code"], str)
    assert "assert" in result["test_code"]

