"""
Debugger tools for the ReAct investigation loop in diagnose_failure.

WHAT THESE TOOLS DO
-------------------
When the debugger node runs its ReAct loop, it can call these tools to
gather evidence before issuing its final diagnosis. Each tool is invoked
when the LLM outputs ``{"action": "use_tool", "tool": "<name>", "args": {...}}``.

The three tools:
  run_snippet:      Execute a short Python snippet in the sandbox to test
                    a specific edge-case hypothesis. E.g. "what does my_func([])
                    return?" allows verifying whether empty-input handling is wrong
                    without reading the code mentally.

  inspect_function: Parse the function's AST to extract signature, argument
                    names, defaults, and docstring. Catches argument mismatch
                    bugs (tests call f(x, y) but the function only takes f(x)).

  diff_iterations:  Show a unified diff between two code versions. Helps the
                    debugger understand whether recent repairs made progress or
                    whether the generator is cycling through the same mistakes.

WHY NOT LANGGRAPH ToolNode?
----------------------------
LangGraph's native ToolNode assumes the LLM supports function-calling via
the OpenAI tools API. The small generator model (Llama-3.2-3B via Ollama or
HuggingFace) does NOT support this. All other roles use Claude, which does
support it, but we implement tool dispatch as plain JSON-in-text for provider
uniformity. The dispatcher (call_tool) is a simple dict lookup.

HOW THEY ARE INVOKED
---------------------
diagnose_failure.py calls call_tool(tool_name, args) which looks up the tool
in TOOL_REGISTRY and calls it with ``**args``. The result (a plain string)
is appended to the tool_observations list and included in the next ReAct
prompt iteration.
"""

import ast
import difflib
import logging
from typing import Any

logger = logging.getLogger(__name__)


async def run_snippet(code: str) -> str:
    """Execute a short Python snippet to test a bug hypothesis.

    Uses the sandbox executor with a tight 5-second timeout (not the full
    15-second timeout used for solution tests). This is for quick verification
    of hypotheses like "does the function return None on empty input?" or
    "does this edge case reproduce the crash?".

    The sandbox's isolation means the snippet runs in a fresh subprocess —
    no risk of affecting the agent's own state.

    Args:
        code: Short Python snippet. Can be a simple expression, a print
              statement, or a mini test (``assert f([]) == []``).
              Does NOT need to be a full function — just runnable code.

    Returns:
        Combined stdout + stderr + pass/fail status as a single string.
        Always returns a string; never raises.
    """
    try:
        from sandbox.python_executor import execute
        # Run the snippet as the "test code" with an empty solution (no imports).
        # The snippet must be self-contained or import what it needs itself.
        result = await execute(solution_code="", test_code=code, timeout=5.0)
        output_parts: list[str] = []
        if result.stdout.strip():
            output_parts.append(f"stdout:\n{result.stdout.strip()}")
        if result.stderr.strip():
            # Strip internal SANDBOX_RESULT: markers — those are for our parser,
            # not for the LLM to read.
            cleaned_stderr = "\n".join(
                line for line in result.stderr.splitlines()
                if not line.startswith("SANDBOX_RESULT:")
            )
            if cleaned_stderr.strip():
                output_parts.append(f"stderr:\n{cleaned_stderr.strip()}")
        # Append pass/fail status so the LLM has a clear yes/no to reason about
        output_parts.append("Status: PASS" if result.passed else "Status: FAIL")
        return "\n".join(output_parts) if output_parts else "(no output)"
    except Exception as exc:
        logger.warning("run_snippet failed: %s", exc)
        return f"Tool error: {exc}"


def inspect_function(code: str, function_name: str) -> str:
    """Parse a function's AST to extract its signature, arguments, and docstring.

    WHY AST INSTEAD OF EXEC?
    Exec-ing the code just to inspect the function signature is risky and slow.
    AST parsing is safe (never runs the code) and fast. We extract everything
    the debugger needs to check argument mismatches.

    Typical use case: the tests call ``my_func(arr, key=True)`` but the function
    signature is ``def my_func(arr)``. This tool surfaces that mismatch without
    needing to run the code.

    Args:
        code:          Python source code containing the function definition.
        function_name: Name of the function to inspect.

    Returns:
        Multi-line string with:
          - Signature (with defaults)
          - Argument names list
          - Defaults dict
          - Docstring (first 200 chars)
        Or an error message if the function is not found or the code has syntax errors.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return f"SyntaxError while parsing code: {exc}"

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name != function_name:
                continue

            # Reconstruct the argument list with defaults inline, e.g.
            # ``(items, key=None, reverse=False)`` instead of raw AST nodes.
            args = [arg.arg for arg in node.args.args]
            defaults = [ast.unparse(d) for d in node.args.defaults]
            # Defaults are right-aligned: the last N args have the N defaults
            n_no_default = len(args) - len(defaults)
            arg_parts: list[str] = []
            for i, name in enumerate(args):
                offset = i - n_no_default
                if offset >= 0:
                    arg_parts.append(f"{name}={defaults[offset]}")
                else:
                    arg_parts.append(name)

            # Include *args and **kwargs if present
            if node.args.vararg:
                arg_parts.append(f"*{node.args.vararg.arg}")
            if node.args.kwarg:
                arg_parts.append(f"**{node.args.kwarg.arg}")

            signature = f"def {function_name}({', '.join(arg_parts)}):"

            # Extract docstring: it's the first statement if it's a string literal
            docstring = ""
            if (
                node.body
                and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, ast.Constant)
                and isinstance(node.body[0].value.value, str)
            ):
                docstring = node.body[0].value.value.strip()

            # Include return type annotation if present
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
    """Show a unified diff between two code versions to track repair progress.

    Helps the debugger answer: "are repairs making progress or cycling in the
    same pattern?" If the diff shows only whitespace changes across 3 iterations,
    the agent is stuck and a blind retry is more useful than another targeted repair.

    Also useful for detecting accidentally introduced regressions: a repair
    that "fixes" one line might silently break another.

    Args:
        code_v1: Older code version (e.g. state["iteration_history"][-2]["code"])
        code_v2: Newer code version (e.g. state["current_code"])

    Returns:
        Unified diff output, or "No differences" if the versions are identical.
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


# TOOL_REGISTRY maps the string names the LLM uses in its JSON tool-use output
# to the actual Python functions. The ReAct loop in diagnose_failure.py calls
# call_tool(name, args) which looks up the function here.
TOOL_REGISTRY: dict[str, Any] = {
    "run_snippet": run_snippet,
    "inspect_function": inspect_function,
    "diff_iterations": diff_iterations,
}


async def call_tool(tool_name: str, args: dict[str, Any]) -> str:
    """Dispatch a tool call by name to the corresponding function.

    Args:
        tool_name: Tool to invoke. Must be a key in TOOL_REGISTRY.
        args:      Keyword arguments forwarded to the tool function.

    Returns:
        Tool output as a string. Never raises — errors are returned as
        descriptive strings so the LLM can read them and adjust its reasoning.
    """
    if tool_name not in TOOL_REGISTRY:
        return f"Unknown tool '{tool_name}'. Available tools: {list(TOOL_REGISTRY)}"

    tool_fn = TOOL_REGISTRY[tool_name]
    try:
        import asyncio
        # Support both sync functions (inspect_function, diff_iterations) and
        # async functions (run_snippet). await only if it's a coroutine.
        if asyncio.iscoroutinefunction(tool_fn):
            return await tool_fn(**args)
        return tool_fn(**args)
    except TypeError as exc:
        # Wrong arguments passed by the LLM — give a clear error message
        return f"Tool call error (bad args for '{tool_name}'): {exc}"
    except Exception as exc:
        logger.warning("Tool '%s' raised unexpected error: %s", tool_name, exc)
        return f"Tool error: {exc}"
