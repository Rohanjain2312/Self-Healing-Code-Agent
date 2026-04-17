"""
Python execution sandbox for safe, isolated code testing.

WHY A SUBPROCESS SANDBOX?
--------------------------
The agent generates code from untrusted LLM output — it could contain infinite
loops, memory bombs, file-system writes, or simply crash-inducing code. Running
it in the main process would either hang the agent or corrupt its state.

The sandbox runs generated code in a completely separate subprocess with:
  - Process isolation: the agent process is never touched by the generated code
  - rlimit resource limits: hard caps on memory (256 MB), CPU time (10 s),
    and subprocess creation (0 child processes = no fork bombs)
  - Timeout: kills the subprocess if it runs longer than 15 seconds
  - Namespace isolation: the solution is loaded as a separate module; only
    public names (no leading ``_``) are exported to the test namespace

SECURITY NOTE
-------------
This sandbox is appropriate for educational/demo use. Production deployments
should add nsjail, seccomp BPF, or Docker container isolation as Layer 4.
The current multi-layer setup stops common accidents but is not hardened
against a determined attacker.

HOW THE SANDBOX WORKS (STEP BY STEP)
--------------------------------------
1. Write solution_code to a temporary .py file (e.g. /tmp/sol_xyz.py)
2. Inject solution_path + test_code into the _SANDBOX_WRAPPER template
3. Write the rendered wrapper to a second temp file (e.g. /tmp/wrap_xyz.py)
4. Launch: ``python /tmp/wrap_xyz.py`` via asyncio.create_subprocess_exec
5. Apply rlimits via preexec_fn (runs in child after fork, before exec)
6. Wait for the subprocess with a timeout; kill if exceeded
7. Parse stdout/stderr for SANDBOX_RESULT: markers
8. Delete both temp files
9. Return ExecutionResult

OUTPUT PROTOCOL
---------------
The wrapper script writes tagged lines to stdout/stderr so we can parse the
outcome without relying on exit codes (which can be unreliable across platforms):
  stdout: SANDBOX_RESULT:PASS
  stderr: SANDBOX_RESULT:FAIL:<assertion message>
  stderr: SANDBOX_RESULT:EXCEPTION:<ExceptionType>:<message>
"""

import asyncio
import os
import sys
import tempfile
import textwrap
from dataclasses import dataclass, field


@dataclass
class ExecutionResult:
    """Structured result from one sandbox execution.

    All fields are populated whether the run passed or failed — callers should
    check ``passed`` first, then read ``failed_assertions`` / ``exception_type``
    for failure details.
    """
    passed: bool               # True only if SANDBOX_RESULT:PASS was found
    stdout: str                # full subprocess stdout
    stderr: str                # full subprocess stderr (includes SANDBOX_RESULT markers)
    failed_assertions: list[str] = field(default_factory=list)
    # Set when the code raised an unhandled exception before/during the test block
    exception_type: str = ""
    exception_message: str = ""
    # Wall-clock time from subprocess launch to completion (or timeout)
    elapsed_seconds: float = 0.0
    # Populated by _parse_result only if output contains structured counts
    passed_count: int = 0
    total_count: int = 0


# ── Sandbox wrapper script ───────────────────────────────────────────────────
# This template is filled in with solution_path and indented_tests, then
# written to a temp file and executed as ``python <temp_file>``.
#
# WHY importlib?
# We could just exec() the solution code directly, but that would put it in
# the same namespace as the test code. Using importlib loads it as an isolated
# module so the test code only sees what the solution explicitly defines (public
# names only — nothing starting with _).
#
# The try/except structure in the wrapper captures both AssertionErrors (test
# failures) and other exceptions (solution crashes) as tagged stderr lines.
_SANDBOX_WRAPPER = textwrap.dedent("""\
import sys
import importlib.util
import traceback

# Load the solution file as a self-contained module (namespace isolation).
# The module's global namespace does not bleed into the test namespace.
_solution_spec = importlib.util.spec_from_file_location("_solution", {solution_path!r})
_solution_mod = importlib.util.module_from_spec(_solution_spec)
try:
    _solution_spec.loader.exec_module(_solution_mod)
except Exception as _load_ex:
    # Solution itself raised an exception on import — report it and exit.
    print(
        "SANDBOX_RESULT:EXCEPTION:" + type(_load_ex).__name__ + ":" + str(_load_ex),
        file=sys.stderr,
    )
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)

# Export only public names from the solution module into the test namespace.
# Names starting with _ are private implementation details that tests should
# not depend on.
for _name in dir(_solution_mod):
    if not _name.startswith("_"):
        globals()[_name] = getattr(_solution_mod, _name)

# Run the test code. Each assert failure is caught individually so we report
# the specific assertion message. Non-assertion exceptions are also caught so
# the sandbox always exits cleanly (not via uncaught exception).
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

# Default timeout in seconds before the subprocess is killed.
# 15s is generous for most algorithmic tasks but fast enough not to block
# the agent if the code has an infinite loop.
_DEFAULT_TIMEOUT = 15.0

# rlimits are only available on POSIX (Linux/macOS). Skip silently on Windows.
_RESOURCE_LIMITS_AVAILABLE = sys.platform != "win32"


def _apply_resource_limits() -> None:
    """Apply POSIX rlimits to the sandbox subprocess.

    WHY preexec_fn?
    This function is passed as ``preexec_fn`` to asyncio.create_subprocess_exec.
    It runs in the child process AFTER fork() but BEFORE exec() — so the limits
    are set on the exact process that will run our code, not the parent.

    The three limits protect against:
      - RLIMIT_AS:    memory bomb (allocating huge arrays)
      - RLIMIT_CPU:   CPU-bound infinite loops (separate from wall-clock timeout)
      - RLIMIT_NPROC: fork bombs (code that spawns many subprocesses)

    Silently no-ops if resource module is unavailable (Windows, some HF Spaces configs).
    """
    try:
        import resource
        # Hard limit: 256 MB virtual address space. The 3B model weights are NOT
        # in this process — they're in the main agent process. This limit only
        # applies to the tiny sandbox subprocess.
        resource.setrlimit(resource.RLIMIT_AS, (256 * 1024 * 1024, 256 * 1024 * 1024))
        # Hard limit: 10 seconds CPU time. Wall-clock timeout (asyncio.wait_for)
        # handles the case where the process sleeps/blocks; RLIMIT_CPU handles
        # pure CPU-bound infinite loops.
        resource.setrlimit(resource.RLIMIT_CPU, (10, 10))
        # Prevent the sandbox from spawning child processes (fork bomb prevention)
        resource.setrlimit(resource.RLIMIT_NPROC, (0, 0))
    except (ImportError, AttributeError, ValueError):
        # resource module absent (Windows) or limit not supported — skip
        pass


async def execute(
    solution_code: str,
    test_code: str,
    timeout: float = _DEFAULT_TIMEOUT,
) -> ExecutionResult:
    """Execute solution_code + test_code in an isolated subprocess.

    The solution and tests are written to separate temp files. The solution is
    loaded as an isolated module so the test namespace is clean.

    Args:
        solution_code: The Python function/class to test.
        test_code:     Python assert statements to run against the solution.
        timeout:       Wall-clock seconds before the subprocess is killed.

    Returns:
        ExecutionResult — always returns (never raises), even on timeout or crash.
    """
    import time

    solution_tmp: str | None = None
    wrapper_tmp: str | None = None

    try:
        # Write the solution to its own temp file. importlib loads it from disk,
        # so it needs an actual file path. delete=False because we clean up manually
        # in the finally block below.
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        ) as sol_f:
            # Handle empty solution gracefully (e.g. first iteration generation failed)
            sol_f.write(solution_code.strip() if solution_code.strip() else "# empty solution")
            solution_tmp = sol_f.name

        # Indent test code by 4 spaces so it fits inside the try block in the wrapper.
        # A bare ``pass`` handles empty test suites (no assertions = trivially passes).
        indented_tests = textwrap.indent(test_code.strip() if test_code.strip() else "pass", "    ")

        # Fill in the template and write to a second temp file
        wrapper_script = _SANDBOX_WRAPPER.format(
            solution_path=solution_tmp,
            indented_tests=indented_tests,
        )
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        ) as wrap_f:
            wrap_f.write(wrapper_script)
            wrapper_tmp = wrap_f.name

        # Use rlimits as preexec_fn on POSIX; None on Windows (no-op).
        preexec = _apply_resource_limits if _RESOURCE_LIMITS_AVAILABLE else None

        start = time.monotonic()
        try:
            # sys.executable ensures we use the same Python interpreter as the
            # agent process (important for virtual environments on HF Spaces).
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
                # Kill the process and drain pipes before returning.
                # Without the second communicate(), the subprocess may become
                # a zombie that holds the pipe open.
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
        # Always clean up temp files, even if subprocess raised
        for path in [solution_tmp, wrapper_tmp]:
            if path:
                try:
                    os.unlink(path)
                except OSError:
                    pass  # file already deleted or never created

    stdout = stdout_bytes.decode("utf-8", errors="replace")
    stderr = stderr_bytes.decode("utf-8", errors="replace")
    return _parse_result(stdout, stderr, elapsed)


def _parse_result(stdout: str, stderr: str, elapsed: float) -> ExecutionResult:
    """Parse SANDBOX_RESULT: markers from subprocess output into ExecutionResult.

    The wrapper writes exactly one of these per run:
      - ``SANDBOX_RESULT:PASS`` to stdout on success
      - ``SANDBOX_RESULT:FAIL:<msg>`` to stderr on AssertionError
      - ``SANDBOX_RESULT:EXCEPTION:<type>:<msg>`` to stderr on any other exception
    """
    # Success: the PASS marker in stdout is definitive
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
            # Extract the assertion failure message (everything after the third colon)
            failed_assertions.append(line.removeprefix("SANDBOX_RESULT:FAIL:"))
        elif line.startswith("SANDBOX_RESULT:EXCEPTION:"):
            # Format: SANDBOX_RESULT:EXCEPTION:<ExceptionType>:<message>
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
    """Produce a concise failure summary for the debugger agent's prompt.

    Prioritizes specific assertion messages and exception info — more useful
    to the LLM debugger than raw tracebacks. Tracebacks are included but
    capped at 40 lines to avoid context overflow.

    Args:
        result: ExecutionResult from execute(). Must not be passed=True.

    Returns:
        Human-readable failure summary string.
    """
    if result.passed:
        return "All tests passed."

    lines: list[str] = []

    if result.exception_type:
        # An unhandled exception before/during the test block — the most
        # informative error type (tells the debugger exactly what broke).
        lines.append(f"Exception: {result.exception_type}: {result.exception_message}")

    if result.failed_assertions:
        # Specific assertion messages — which test case failed and how.
        lines.append("Failed assertions:")
        for msg in result.failed_assertions:
            lines.append(f"  - {msg}")

    if result.stderr:
        # Include the traceback for line-number context, but cap at 40 lines.
        # Strip our internal SANDBOX_RESULT: markers before including the traceback.
        tb_lines = result.stderr.splitlines()
        relevant = [ln for ln in tb_lines if not ln.startswith("SANDBOX_RESULT:")]
        if relevant:
            lines.append("Traceback (last 40 lines):")
            lines.extend(relevant[-40:])

    return "\n".join(lines)
