"""
Python execution sandbox for adversarial test execution.

Security layering (innermost → outermost):
  Layer 1: subprocess isolation — user code never imports into the main process.
  Layer 2: rlimit resource limits — cap memory (256 MB), CPU time (10 s),
           and subprocess creation (0 child processes). Applied via preexec_fn
           on POSIX systems; gracefully skipped on Windows/macOS when unavailable.
  Layer 3: importlib namespace separation — solution code is loaded as an
           isolated module; only public names (no leading _) are injected into
           the test namespace. This prevents tests from coupling to internal
           helper functions or module-level state.
  Layer 4 (production hardening): nsjail, Docker, or seccomp BPF. Not included
           here — this sandbox is appropriate for educational/demo use.

Usage:
    result = await execute(solution_code="def f(): ...", test_code="assert f() == ...")
    if not result.passed:
        print(format_failure_summary(result))
"""

import asyncio
import os
import sys
import tempfile
import textwrap
from dataclasses import dataclass, field


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


# ---------------------------------------------------------------------------
# Sandbox wrapper script — written to a temp file and executed in subprocess.
#
# The solution is loaded as a separate module via importlib so tests cannot
# accidentally reference implementation internals (leading-_ names are excluded).
# Public names are injected into the test namespace via globals().
# ---------------------------------------------------------------------------
_SANDBOX_WRAPPER = textwrap.dedent("""\
import sys
import importlib.util
import traceback

# --- Load solution as an isolated module ---
_solution_spec = importlib.util.spec_from_file_location("_solution", {solution_path!r})
_solution_mod = importlib.util.module_from_spec(_solution_spec)
try:
    _solution_spec.loader.exec_module(_solution_mod)
except Exception as _load_ex:
    print(
        "SANDBOX_RESULT:EXCEPTION:" + type(_load_ex).__name__ + ":" + str(_load_ex),
        file=sys.stderr,
    )
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)

# Inject only public names into global namespace
for _name in dir(_solution_mod):
    if not _name.startswith("_"):
        globals()[_name] = getattr(_solution_mod, _name)

# --- Execute tests ---
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
_RESOURCE_LIMITS_AVAILABLE = sys.platform != "win32"


def _apply_resource_limits() -> None:
    """
    Apply POSIX rlimits to cap resource usage in the sandbox subprocess.

    Called as preexec_fn — runs inside the child process after fork().
    Silently skipped if the resource module is unavailable (e.g. Windows).
    """
    try:
        import resource
        # 256 MB virtual address space
        resource.setrlimit(resource.RLIMIT_AS, (256 * 1024 * 1024, 256 * 1024 * 1024))
        # 10 seconds CPU time
        resource.setrlimit(resource.RLIMIT_CPU, (10, 10))
        # No child processes (prevents fork bombs)
        resource.setrlimit(resource.RLIMIT_NPROC, (0, 0))
    except (ImportError, AttributeError, ValueError):
        # resource module absent (Windows) or limit not supported on this OS
        pass


async def execute(
    solution_code: str,
    test_code: str,
    timeout: float = _DEFAULT_TIMEOUT,
) -> ExecutionResult:
    """
    Execute solution_code + test_code in an isolated subprocess.

    The solution and tests are written to separate temp files. The solution is
    loaded as an isolated module via importlib so tests are namespace-separated.

    Returns ExecutionResult regardless of outcome — never raises.
    The caller (agent node) decides how to handle failures.
    """
    import time

    solution_tmp: str | None = None
    wrapper_tmp: str | None = None

    try:
        # Write solution to its own temp file (loaded as isolated module)
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        ) as sol_f:
            sol_f.write(solution_code.strip() if solution_code.strip() else "# empty solution")
            solution_tmp = sol_f.name

        # Indent test code for the try block in the wrapper
        indented_tests = textwrap.indent(test_code.strip() if test_code.strip() else "pass", "    ")

        wrapper_script = _SANDBOX_WRAPPER.format(
            solution_path=solution_tmp,
            indented_tests=indented_tests,
        )

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        ) as wrap_f:
            wrap_f.write(wrapper_script)
            wrapper_tmp = wrap_f.name

        preexec = _apply_resource_limits if _RESOURCE_LIMITS_AVAILABLE else None

        start = time.monotonic()
        try:
            proc = await asyncio.create_subprocess_exec(
                sys.executable,
                wrapper_tmp,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                preexec_fn=preexec,
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

    finally:
        for path in [solution_tmp, wrapper_tmp]:
            if path:
                try:
                    os.unlink(path)
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
        relevant = [ln for ln in tb_lines if not ln.startswith("SANDBOX_RESULT:")]
        if relevant:
            lines.append("Traceback (last 40 lines):")
            lines.extend(relevant[-40:])

    return "\n".join(lines)
