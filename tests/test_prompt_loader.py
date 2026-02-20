"""
Tests for the prompt loader and template rendering.
"""

import pytest
from llm.prompt_loader import (
    get_system_prompt,
    get_schema,
    render_template,
    list_available_roles,
    invalidate_cache,
)


def setup_function():
    invalidate_cache()  # ensure fresh state per test


def test_list_available_roles():
    roles = list_available_roles()
    assert "generator" in roles
    assert "qa_adversarial" in roles
    assert "debugger" in roles
    assert "memory_summarizer" in roles


def test_get_system_prompt_generator():
    prompt = get_system_prompt("generator")
    assert len(prompt) > 10
    assert "JSON" in prompt or "json" in prompt.lower()


def test_get_schema_generator():
    schema = get_schema("generator")
    assert schema.get("type") == "object"
    assert "code" in schema.get("properties", {})


def test_render_template_initial():
    rendered = render_template(
        "generator",
        "initial",
        {
            "task_description": "Write a sort function.",
            "learning_log": "No lessons.",
        },
    )
    assert "Write a sort function." in rendered
    assert "No lessons." in rendered


def test_render_template_missing_variable_shows_marker():
    rendered = render_template(
        "generator",
        "initial",
        {
            "task_description": "Test task.",
            # learning_log intentionally omitted
        },
    )
    assert "<MISSING:learning_log>" in rendered


def test_render_template_invalid_key():
    with pytest.raises(KeyError):
        render_template("generator", "nonexistent_template", {})


def test_unknown_role_raises():
    with pytest.raises(FileNotFoundError):
        get_system_prompt("nonexistent_role_xyz")
