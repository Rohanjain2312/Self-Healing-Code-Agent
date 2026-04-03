"""
Debugger tools for the ReAct investigation loop.

These tools are used by the diagnose_failure node to verify hypotheses
about bugs before issuing a final diagnosis. They allow the debugger to:

  - run_snippet: execute short Python snippets to test edge cases
  - inspect_function: parse AST to extract function signature and docstring
  - diff_iterations: compare code across repair attempts via unified diff

Design decision: tools are plain async functions rather than LangGraph ToolNode
wrappers because the underlying LLM (Ollama/llama3) does not support native
function-calling. The ReAct loop in diagnose_failure invokes these directly
by name based on the LLM's JSON tool-use response.

langchain-core is already in requirements.txt — no new dependency needed.
"""

import ast
import difflib
import logging
from typing import Any

logger = logging.getLogger(__name__)


async def run_snippet(code: str) -> str:
    """
    Execute a short Python snippet to test a hypothesis about a bug.

    Uses the sandbox executor with a 5-second timeout. Returns stdout +
    stderr combined so the debugger sees both output and error messages.

    Args:
        code: Short Python snippet (not a full function — can use simple asserts).

    Returns:
        Combined stdout/stderr string, or an error message if execution fails.
    """
    try:
        from sandbox.python_executor import execute
        result = await execute(solution_code="", test_code=code, timeout=5.0)
        output_parts: list[str] = []
        if result.stdout.strip():
            output_parts.append(f"stdout:\n{result.stdout.strip()}")
        if result.stderr.strip():
            # Strip internal sandbox markers before returning
            cleaned_stderr = "\n".join(
                line for line in result.stderr.splitlines()
                if not line.startswith("SANDBOX_RESULT:")
            )
            if cleaned_stderr.strip():
                output_parts.append(f"stderr:\n{cleaned_stderr.strip()}")
        if result.passed:
            output_parts.append("Status: PASS")
        else:
            output_parts.append("Status: FAIL")
        return "\n".join(output_parts) if output_parts else "(no output)"
    except Exception as exc:
        logger.warning("run_snippet failed: %s", exc)
        return f"Tool error: {exc}"


def inspect_function(code: str, function_name: str) -> str:
    """
    Parse AST to extract function signature, arguments, and docstring.

    Helps the debugger verify what a function actually accepts vs what the
    tests call it with — catching argument mismatch bugs without execution.

    Args:
        code: Python source containing the function definition.
        function_name: Name of the function to inspect.

    Returns:
        Structured string with signature, args, defaults, and docstring.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return f"SyntaxError while parsing code: {exc}"

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name != function_name:
                continue

            # Extract argument names
            args = [arg.arg for arg in node.args.args]
            # Extract defaults (right-aligned with positional args)
            defaults = [ast.unparse(d) for d in node.args.defaults]
            n_no_default = len(args) - len(defaults)
            arg_parts: list[str] = []
            for i, name in enumerate(args):
                offset = i - n_no_default
                if offset >= 0:
                    arg_parts.append(f"{name}={defaults[offset]}")
                else:
                    arg_parts.append(name)

            if node.args.vararg:
                arg_parts.append(f"*{node.args.vararg.arg}")
            if node.args.kwarg:
                arg_parts.append(f"**{node.args.kwarg.arg}")

            signature = f"def {function_name}({', '.join(arg_parts)}):"

            # Extract docstring if present
            docstring = ""
            if (
                node.body
                and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, ast.Constant)
                and isinstance(node.body[0].value.value, str)
            ):
                docstring = node.body[0].value.value.strip()

            return_annotation = ""
            if node.returns:
                return_annotation = f" -> {ast.unparse(node.returns)}"

            lines = [
                f"Signature: {signature}{return_annotation}",
                f"Arguments: {args}",
                f"Defaults: {dict(zip(args[n_no_default:], defaults)) if defaults else 'none'}",
            ]
            if docstring:
                lines.append(f"Docstring: {docstring[:200]}")
            return "\n".join(lines)

    return f"Function '{function_name}' not found in provided code."


def diff_iterations(code_v1: str, code_v2: str) -> str:
    """
    Show unified diff between two code versions.

    Helps the debugger understand what changed across repair attempts —
    whether the generator is making progress or cycling in the same mistakes.

    Args:
        code_v1: Older code version (e.g. from a prior iteration).
        code_v2: Newer code version (e.g. current_code).

    Returns:
        Unified diff output, or a message if the two versions are identical.
    """
    lines_v1 = code_v1.splitlines(keepends=True)
    lines_v2 = code_v2.splitlines(keepends=True)
    diff = list(difflib.unified_diff(
        lines_v1,
        lines_v2,
        fromfile="iteration_N-1",
        tofile="iteration_N",
        lineterm="",
    ))
    if not diff:
        return "No differences — code is identical between iterations."
    return "".join(diff)


# Registry for the ReAct loop to invoke tools by name
TOOL_REGISTRY: dict[str, Any] = {
    "run_snippet": run_snippet,
    "inspect_function": inspect_function,
    "diff_iterations": diff_iterations,
}


async def call_tool(tool_name: str, args: dict[str, Any]) -> str:
    """
    Dispatch a tool call by name from the registry.

    Args:
        tool_name: One of 'run_snippet', 'inspect_function', 'diff_iterations'.
        args: Keyword arguments to pass to the tool.

    Returns:
        Tool output as a string, or an error message for unknown tools.
    """
    if tool_name not in TOOL_REGISTRY:
        return f"Unknown tool '{tool_name}'. Available: {list(TOOL_REGISTRY)}"

    tool_fn = TOOL_REGISTRY[tool_name]
    try:
        import asyncio
        if asyncio.iscoroutinefunction(tool_fn):
            return await tool_fn(**args)
        return tool_fn(**args)
    except TypeError as exc:
        return f"Tool call error (bad args for '{tool_name}'): {exc}"
    except Exception as exc:
        logger.warning("Tool '%s' raised unexpected error: %s", tool_name, exc)
        return f"Tool error: {exc}"
