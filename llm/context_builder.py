"""
Token-aware context builder.

Assembles the user prompt from template variables while enforcing a
token budget. Truncates the largest variable fields when the combined
context would exceed the model's limit.

Token counting is approximate (character-based) since we do not want
to import a tokenizer as a hard dependency for every environment.
The approximation: 1 token ≈ 4 characters (conservative for English code).
"""

from typing import Any

_CHARS_PER_TOKEN = 4  # conservative approximation
_DEFAULT_MAX_TOKENS = 3072  # leave headroom for system prompt + output


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // _CHARS_PER_TOKEN)


def _truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Hard-truncate a string to approximately max_tokens tokens."""
    max_chars = max_tokens * _CHARS_PER_TOKEN
    if len(text) <= max_chars:
        return text
    # Truncate and append a note so the LLM knows content was cut
    truncated = text[:max_chars]
    return truncated + "\n...[TRUNCATED FOR CONTEXT BUDGET]"


def build_context(
    rendered_template: str,
    variables: dict[str, Any],
    template_str: str = "",
    max_context_tokens: int = _DEFAULT_MAX_TOKENS,
) -> str:
    """
    Return the rendered template, truncating fields if needed to fit token budget.

    The rendered_template is already assembled; this function checks if it
    fits within budget and truncates the most expensive variable fields, then
    RE-RENDERS the template with the truncated values so the truncation
    actually takes effect.

    Args:
        rendered_template: Already-substituted prompt string (used as fallback).
        variables: The original substitution dict used to render the template.
        template_str: Raw template before substitution. When provided, truncated
            variables are re-substituted into the template for a clean result.
            If omitted, falls back to string-replacement on the rendered output.
        max_context_tokens: Hard token budget (approximate, char-based).

    Truncation priority (highest cost fields truncated first):
      1. test_results
      2. iteration_history
      3. current_code / code
      4. learning_log / prior_lessons
    """
    total_tokens = _estimate_tokens(rendered_template)

    if total_tokens <= max_context_tokens:
        return rendered_template

    # Fields to attempt truncation in order of expendability
    truncation_candidates = [
        "test_results",
        "iteration_history",
        "current_code",
        "code",
        "learning_log",
        "prior_lessons",
    ]

    # Rebuild template with progressively shorter fields
    trimmed_vars = dict(variables)
    for field in truncation_candidates:
        if field not in trimmed_vars:
            continue
        original = str(trimmed_vars[field])
        # Allow this field to consume at most half the remaining budget
        field_budget = max_context_tokens // 2
        trimmed_vars[field] = _truncate_to_tokens(original, field_budget)

        # Re-estimate (rough); if within budget, stop
        new_estimate = _estimate_tokens(rendered_template) - _estimate_tokens(
            original
        ) + _estimate_tokens(str(trimmed_vars[field]))
        if new_estimate <= max_context_tokens:
            break

    # ACTUALLY RE-RENDER with truncated variables so the truncation takes effect.
    # Previously this function returned the original rendered_template — that was
    # a bug where trimmed_vars was computed but never used.
    if template_str:
        # Clean re-render: substitute truncated values into the raw template.
        class _SafeMap(dict):
            def __missing__(self, key: str) -> str:
                return f"<MISSING:{key}>"

        return template_str.format_map(_SafeMap(trimmed_vars))
    else:
        # Fallback: replace each changed field value in the already-rendered string.
        re_rendered = rendered_template
        for field, trimmed_value in trimmed_vars.items():
            original_value = str(variables.get(field, ""))
            if original_value != str(trimmed_value):
                re_rendered = re_rendered.replace(original_value, str(trimmed_value))
        return re_rendered
