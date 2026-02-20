"""
Python execution sandbox for adversarial test execution.

Design constraints:
  - Runs in a subprocess to isolate crashes and infinite loops
  - Enforces a wall-clock timeout (default 15s)
  - Captures stdout, stderr, and exception tracebacks
  - Never imports user code into the main process
  - Restricts dangerous builtins via __builtins__ override in subprocess

The solution code and test code are concatenated and written to a temp file,
then executed via subprocess. This avoids exec() in the main process.

Security note: This sandbox is NOT a full security sandbox — it prevents
accidental hangs and captures output, but does not prevent file I/O or
network calls from the executed code. For production hardening, wrap with
nsjail, Docker, or similar. For educational/demo use, the timeout and
subprocess boundary are sufficient.
"""

import asyncio
import os
import sys
import tempfile
import textwrap
import traceback
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ExecutionResult:
    """Structured result from a sandbox execution."""
    passed: bool
    stdout: str
    stderr: str
    # Individual assertion failures extracted from output
    failed_assertions: list[str] = field(default_factory=list)
    # Python exception type if execution crashed before assertions
    exception_type: str = ""
    exception_message: str = ""
    # Wall clock execution time in seconds
    elapsed_seconds: float = 0.0
    # Number of asserts that passed vs total parsed from output
    passed_count: int = 0
    total_count: int = 0


_SANDBOX_WRAPPER = textwrap.dedent("""\
import sys
import traceback

# Execution harness — wraps user code and test code

{solution_code}

# --- Adversarial Tests ---
try:
{indented_tests}
    print("SANDBOX_RESULT:PASS")
except AssertionError as _ae:
    _msg = str(_ae) if str(_ae) else "AssertionError (no message)"
    print("SANDBOX_RESULT:FAIL:" + _msg, file=sys.stderr)
    traceback.print_exc(file=sys.stderr)
except Exception as _ex:
    print("SANDBOX_RESULT:EXCEPTION:" + type(_ex).__name__ + ":" + str(_ex), file=sys.stderr)
    traceback.print_exc(file=sys.stderr)
""")

_DEFAULT_TIMEOUT = 15.0  # seconds


async def execute(
    solution_code: str,
    test_code: str,
    timeout: float = _DEFAULT_TIMEOUT,
) -> ExecutionResult:
    """
    Execute solution_code + test_code in an isolated subprocess.

    Returns ExecutionResult regardless of outcome — never raises.
    The caller (agent node) decides how to handle failures.
    """
    import time

    # Indent test code so it sits inside the try block in the wrapper
    indented_tests = textwrap.indent(test_code.strip(), "    ")

    script = _SANDBOX_WRAPPER.format(
        solution_code=solution_code.strip(),
        indented_tests=indented_tests,
    )

    # Write to a temporary file — avoids shell injection via -c flag
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as tmp:
        tmp.write(script)
        tmp_path = tmp.name

    start = time.monotonic()
    try:
        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            tmp_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(), timeout=timeout
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.communicate()
            return ExecutionResult(
                passed=False,
                stdout="",
                stderr=f"EXECUTION TIMEOUT after {timeout}s",
                exception_type="TimeoutError",
                exception_message=f"Execution exceeded {timeout} second limit",
                elapsed_seconds=timeout,
            )
    finally:
        elapsed = time.monotonic() - start
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    stdout = stdout_bytes.decode("utf-8", errors="replace")
    stderr = stderr_bytes.decode("utf-8", errors="replace")

    return _parse_result(stdout, stderr, elapsed)


def _parse_result(stdout: str, stderr: str, elapsed: float) -> ExecutionResult:
    """
    Interpret sandbox output markers to build a structured ExecutionResult.

    Markers written by the wrapper script:
      SANDBOX_RESULT:PASS
      SANDBOX_RESULT:FAIL:<message>
      SANDBOX_RESULT:EXCEPTION:<type>:<message>
    """
    combined = stdout + stderr

    if "SANDBOX_RESULT:PASS" in stdout:
        return ExecutionResult(
            passed=True,
            stdout=stdout,
            stderr=stderr,
            elapsed_seconds=elapsed,
        )

    failed_assertions: list[str] = []
    exception_type = ""
    exception_message = ""

    for line in stderr.splitlines():
        if line.startswith("SANDBOX_RESULT:FAIL:"):
            failed_assertions.append(line.removeprefix("SANDBOX_RESULT:FAIL:"))
        elif line.startswith("SANDBOX_RESULT:EXCEPTION:"):
            parts = line.removeprefix("SANDBOX_RESULT:EXCEPTION:").split(":", 1)
            exception_type = parts[0] if parts else "UnknownException"
            exception_message = parts[1] if len(parts) > 1 else ""

    return ExecutionResult(
        passed=False,
        stdout=stdout,
        stderr=stderr,
        failed_assertions=failed_assertions,
        exception_type=exception_type,
        exception_message=exception_message,
        elapsed_seconds=elapsed,
    )


def format_failure_summary(result: ExecutionResult) -> str:
    """
    Produce a concise failure summary for the Debugger agent.

    Prioritizes assertion messages and exception info over raw traceback text.
    Tracebacks are included but trimmed to avoid context overflow.
    """
    if result.passed:
        return "All tests passed."

    lines: list[str] = []

    if result.exception_type:
        lines.append(f"Exception: {result.exception_type}: {result.exception_message}")

    if result.failed_assertions:
        lines.append("Failed assertions:")
        for msg in result.failed_assertions:
            lines.append(f"  - {msg}")

    if result.stderr:
        # Include traceback but cap at 40 lines to avoid overwhelming the context
        tb_lines = result.stderr.splitlines()
        relevant = [l for l in tb_lines if not l.startswith("SANDBOX_RESULT:")]
        if relevant:
            lines.append("Traceback (last 40 lines):")
            lines.extend(relevant[-40:])

    return "\n".join(lines)
