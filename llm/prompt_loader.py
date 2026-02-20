"""
YAML prompt loader.

Loads prompt definitions from the prompts/ directory.
Caches parsed YAML in memory after first load to avoid repeated disk I/O.
Supports template variable substitution via Python str.format_map.
"""

import os
import re
from pathlib import Path
from typing import Any

import yaml

# Prompts directory relative to project root
_PROMPTS_DIR = Path(__file__).parent.parent / "prompts"
_cache: dict[str, dict] = {}


def _load_yaml(role: str) -> dict:
    """Load and cache YAML file for a given role."""
    if role in _cache:
        return _cache[role]

    path = _PROMPTS_DIR / f"{role}.yaml"
    if not path.exists():
        raise FileNotFoundError(
            f"Prompt file not found: {path}. "
            f"Available roles: {list_available_roles()}"
        )

    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)

    _cache[role] = data
    return data


def get_system_prompt(role: str) -> str:
    """Return the system prompt string for the given role."""
    data = _load_yaml(role)
    return data.get("system", "").strip()


def get_schema(role: str) -> dict:
    """Return the JSON schema dict for the given role's expected output."""
    data = _load_yaml(role)
    return data.get("schema", {})


def render_template(role: str, template_key: str, variables: dict[str, Any]) -> str:
    """
    Render a named template with the given variables.

    Missing variables are replaced with '<MISSING:variable_name>' to
    make incomplete contexts visible in logs rather than crashing silently.
    """
    data = _load_yaml(role)
    templates = data.get("templates", {})

    if template_key not in templates:
        available = list(templates.keys())
        raise KeyError(
            f"Template '{template_key}' not found for role '{role}'. "
            f"Available: {available}"
        )

    template = templates[template_key]

    # Safe substitution — surfaces missing keys rather than raising KeyError
    class _SafeMap(dict):
        def __missing__(self, key: str) -> str:
            return f"<MISSING:{key}>"

    return template.format_map(_SafeMap(variables))


def list_available_roles() -> list[str]:
    """Return all roles with prompt YAML files in the prompts directory."""
    return [p.stem for p in _PROMPTS_DIR.glob("*.yaml")]


def invalidate_cache(role: str | None = None) -> None:
    """Clear cached prompts. Used in tests to reload modified YAML."""
    if role is None:
        _cache.clear()
    else:
        _cache.pop(role, None)
