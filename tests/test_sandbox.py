"""
Tests for the Python execution sandbox.

These tests verify the sandbox correctly:
  - Captures passing tests
  - Captures assertion failures
  - Handles timeouts
  - Handles syntax errors in generated code
  - Does not raise from the main process on failure
"""

import asyncio
import pytest
from sandbox.python_executor import execute, format_failure_summary


@pytest.mark.asyncio
async def test_passing_code():
    solution = "def add(a, b): return a + b"
    tests = "assert add(1, 2) == 3, 'basic addition'"
    result = await execute(solution, tests)
    assert result.passed is True
    assert result.exception_type == ""


@pytest.mark.asyncio
async def test_failing_assertion():
    solution = "def add(a, b): return a - b"  # wrong implementation
    tests = "assert add(1, 2) == 3, 'should return 3'"
    result = await execute(solution, tests)
    assert result.passed is False
    assert result.failed_assertions or result.exception_type


@pytest.mark.asyncio
async def test_syntax_error_in_solution():
    solution = "def add(a, b return a + b"  # syntax error
    tests = "assert add(1, 2) == 3"
    result = await execute(solution, tests)
    assert result.passed is False
    assert result.exception_type or result.stderr


@pytest.mark.asyncio
async def test_timeout_enforcement():
    solution = "def spin(): pass"
    # Infinite loop that should be killed by timeout
    tests = "while True: pass"
    result = await execute(solution, tests, timeout=2.0)
    assert result.passed is False
    assert "timeout" in result.exception_type.lower() or "timeout" in result.stderr.lower()


@pytest.mark.asyncio
async def test_empty_tests():
    solution = "def noop(): pass"
    tests = "pass  # no assertions"
    result = await execute(solution, tests)
    assert result.passed is True


@pytest.mark.asyncio
async def test_exception_in_solution():
    solution = "def divide(a, b): return a / b"
    tests = "assert divide(1, 0) == 0, 'division by zero'"
    result = await execute(solution, tests)
    assert result.passed is False
    assert result.exception_type  # ZeroDivisionError


def test_format_failure_summary_on_pass():
    from sandbox.python_executor import ExecutionResult
    result = ExecutionResult(passed=True, stdout="", stderr="")
    assert format_failure_summary(result) == "All tests passed."


def test_format_failure_summary_with_assertion():
    from sandbox.python_executor import ExecutionResult
    result = ExecutionResult(
        passed=False,
        stdout="",
        stderr="",
        failed_assertions=["expected 3, got 5"],
    )
    summary = format_failure_summary(result)
    assert "expected 3, got 5" in summary
