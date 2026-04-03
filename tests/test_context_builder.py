"""Tests for context builder — verifies truncation actually applies."""
import pytest
from llm.context_builder import build_context


def test_truncation_actually_applies():
    """Truncation must produce a shorter result when over budget."""
    huge_var = "x" * 50000
    template = "Code: {code}"
    rendered = template.format(code=huge_var)
    result = build_context(rendered, {"code": huge_var}, template_str=template, max_context_tokens=100)
    assert len(result) < len(rendered)
    assert "TRUNCATED" in result


def test_no_truncation_when_within_budget():
    """Content within token budget is returned unchanged."""
    small_var = "def f(): pass"
    template = "Code: {code}"
    rendered = template.format(code=small_var)
    result = build_context(rendered, {"code": small_var}, template_str=template, max_context_tokens=5000)
    assert result == rendered


def test_truncation_without_template_str_fallback():
    """Fallback string-replacement path also truncates correctly."""
    huge_var = "x" * 50000
    rendered = f"Code: {huge_var}"
    result = build_context(rendered, {"code": huge_var}, max_context_tokens=100)
    assert len(result) < len(rendered)
