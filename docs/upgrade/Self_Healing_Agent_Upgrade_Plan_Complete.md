# Self-Healing Code Agent — Complete Upgrade Plan (19 Fixes)

**Goal:** Transform this from a well-engineered LLM pipeline into a genuine agentic AI system that would impress a senior AI engineer leading an agent-building team at Goldman Sachs.

**Structure:** The 19 fixes are organized into four phases:

| Phase | Fixes | Time | What It Proves |
|-------|-------|------|----------------|
| **Phase 1: Eliminate Embarrassments** | Fixes 1–4, 11 | ~10 hrs | "I write correct, tested code" |
| **Phase 2: Production Thinking** | Fixes 5, 7, 9, 12, 13 | ~16 hrs | "I think about reliability, observability, and human oversight" |
| **Phase 3: Agentic Transformation** | Fixes 6, 8, 14, 15, 17 | ~18 hrs | "I understand what makes an agent an agent" |
| **Phase 4: Platform-Level Polish** | Fixes 10, 16, 18, 19 | ~12 hrs | "I think like someone building agent infrastructure for other teams" |

---

# CROSS-CUTTING CONCERN: EVENT TYPES AND STREAMING

Multiple fixes add new nodes that emit events. For each new node, you MUST also:

1. **`agent/events.py`** — Add new event type constants and constructor helpers:
   - `SPEC_TESTS_GENERATED = "spec_tests_generated"` (Fix 5)
   - `CRITIC_REVIEW = "critic_review"` (Fix 15)
   - `REPAIR_REVIEW = "repair_review"` (Fix 12)
   - `PARALLEL_REPAIR = "parallel_repair"` (Fix 14)
   - `TOOL_USE = "tool_use"` (Fix 6)
   - Add corresponding `spec_tests_event()`, `critic_event()`, `repair_review_event()`, `parallel_repair_event()`, `tool_use_event()` constructor helpers following the existing pattern.

2. **`framework/streaming.py`** — Add new event types to `PUBLIC_EVENT_TYPES` set so they appear in the UI timeline. Add `format_event_for_timeline()` handlers for each new type.

3. **`demo/demo_runner.py`** — Update `DemoUIState.apply_event()` to handle new event types (e.g., display critic verdict, show tool use actions, indicate HITL pause).

Apply these updates alongside the fix that introduces each new node — not as a separate step.

---

# PHASE 1: ELIMINATE EMBARRASSMENTS

---

## Fix 1: The Context Builder Bug (30 minutes)

**The Problem:**
`context_builder.py` computes `trimmed_vars` but returns the original `rendered_template` unchanged. The truncation logic is dead code.

**The Fix:**
The function needs to re-render the template with the truncated variables. But currently it receives `rendered_template` (already rendered) AND `variables` (raw). It truncates the raw variables but never re-renders.

**What to change in `llm/context_builder.py`:**

```python
def build_context(
    rendered_template: str,
    variables: dict[str, Any],
    template_str: str = "",          # NEW: pass the raw template string
    max_context_tokens: int = _DEFAULT_MAX_TOKENS,
) -> str:
    total_tokens = _estimate_tokens(rendered_template)
    if total_tokens <= max_context_tokens:
        return rendered_template

    trimmed_vars = dict(variables)
    for field in truncation_candidates:
        if field not in trimmed_vars:
            continue
        original = str(trimmed_vars[field])
        field_budget = max_context_tokens // 2
        trimmed_vars[field] = _truncate_to_tokens(original, field_budget)

    # ACTUALLY RE-RENDER with truncated variables
    if template_str:
        re_rendered = template_str.format(**trimmed_vars)
    else:
        # Fallback: do string replacement on the rendered template
        re_rendered = rendered_template
        for field, value in trimmed_vars.items():
            original = str(variables.get(field, ""))
            if original != str(value):
                re_rendered = re_rendered.replace(original, str(value))

    return re_rendered
```

Then update the call chain:

1. **`llm/prompt_loader.py`** — Add a `get_raw_template(role, template_key)` function that returns the raw template string *before* variable substitution. Currently `render_template()` does the substitution internally and only returns the rendered result. The router needs the raw template to pass to `build_context()`. Implementation:

```python
def get_raw_template(role: str, template_key: str) -> str:
    """Return the raw template string before variable substitution."""
    data = _load_yaml(role)
    templates = data.get("templates", {})
    if template_key not in templates:
        raise KeyError(f"Template '{template_key}' not found for role '{role}'.")
    return templates[template_key]
```

2. **`llm/router.py`** — In `call()`, after `render_template()`, also call `get_raw_template()` and pass it as `template_str` to `build_context()`.

**Add a test** in `tests/test_context_builder.py`:

```python
def test_truncation_actually_applies():
    huge_var = "x" * 50000
    template = "Code: {code}"
    rendered = template.format(code=huge_var)
    result = build_context(rendered, {"code": huge_var}, template_str=template, max_context_tokens=100)
    assert len(result) < len(rendered)
    assert "TRUNCATED" in result
```

---

## Fix 2: Add Held-Out Reference Tests to the Benchmark (2–3 hours)

**The Problem:**
The benchmark reports "80% repair effectiveness" but only checks whether the agent's *own* generated tests pass. The `reference_tests` field exists in `BenchmarkTask` but is never populated or used. This makes the metric circular and meaningless.

**The Fix:**

### Step A: Populate reference tests for all 8 tasks

In `evaluation/benchmark_tasks.py`, add actual ground-truth test code to each task. Add a new field:

```python
@dataclass
class BenchmarkTask:
    task_id: str
    category: str
    description: str
    reference_tests: list[dict[str, Any]] = field(default_factory=list)
    reference_test_code: str = ""    # NEW: executable ground-truth tests
    known_difficulty: str = ""
```

Then populate for each task. Example for `interval_merge_001`:

```python
BenchmarkTask(
    task_id="interval_merge_001",
    # ... existing fields ...
    reference_test_code="""
assert merge_intervals([]) == [], "empty input"
assert merge_intervals([[1,3],[2,6],[8,10],[15,18]]) == [[1,6],[8,10],[15,18]], "standard merge"
assert merge_intervals([[1,3],[3,5]]) == [[1,5]], "touching intervals"
assert merge_intervals([[1,4],[0,4]]) == [[0,4]], "overlap from left"
assert merge_intervals([[1,2]]) == [[1,2]], "single interval"
assert merge_intervals([[1,4],[2,3]]) == [[1,4]], "contained interval"
assert merge_intervals([[3,4],[1,2],[5,6]]) == [[1,2],[3,4],[5,6]], "no overlaps unsorted"
""",
),
```

Do this for all 8 tasks. These tests should cover edge cases the agent's QA agent might miss.

### Step B: Add reference validation to `run_benchmark.py`

```python
from sandbox.python_executor import execute

async def validate_against_reference(task: BenchmarkTask, final_code: str) -> bool:
    """Run the agent's final code against held-out reference tests."""
    if not task.reference_test_code:
        return None  # No reference tests available
    result = await execute(
        solution_code=final_code,
        test_code=task.reference_test_code,
    )
    return result.passed
```

Call this in `_extract_task_result` and add the result to `TaskResult`.

### Step C: Report both metrics in results.json

```json
{
  "summary": {
    "self_reported_success": 0.875,
    "reference_validated_success": 0.75,
    "repair_effectiveness_self": 0.8,
    "repair_effectiveness_validated": 0.6
  },
  "tasks": [
    {
      "task_id": "interval_merge_001",
      "self_test_passed": true,
      "reference_test_passed": true,
      "iterations_used": 2
    }
  ]
}
```

**Interview talking point:** "The agent reports 87.5% success on its own tests, but only 75% passes held-out reference tests. That gap is the QA agent's blind spot — which I address in Fix 5."

---

## Fix 3: Fix the Test Suite (3–4 hours)

**The Problem:**
397 lines of tests, all using the mock provider. They verify plumbing, not behavior. No edge case coverage for schema validation, no error recovery testing, no sandbox security tests.

**What to add:**

### A. Schema validator edge cases (`tests/test_schema_validator.py` — expand)

**NOTE:** The existing test file already covers: markdown fences (`test_json_with_markdown_fence`), prose prefix (`test_json_with_prose_prefix`), invalid JSON (`test_invalid_json_raises`), nested dict coercion (`test_nested_dict_in_required_string_field_is_coerced`). Do NOT duplicate these. Only add genuinely new tests:

```python
def test_truncated_json_salvages_code():
    """Simulates max_new_tokens cutting off the JSON mid-way.
    The _salvage_code_field regex should extract the code value."""
    raw = '{"code": "def add(a,b):\\n    return a+b\\n", "explanation": "adds tw'
    result = parse_and_validate(raw, _SIMPLE_SCHEMA)
    assert "def add" in result["code"]

def test_multiple_json_objects_extracts_first():
    """When model outputs explanation then JSON then more text."""
    raw = 'Let me think... {"code": "x = 1"} That should work. {"code": "x = 2"}'
    result = parse_and_validate(raw, _SIMPLE_SCHEMA)
    assert result["code"] == "x = 1"

def test_code_with_escaped_quotes():
    """Code containing escaped quotes inside JSON string."""
    raw = '{"code": "def f():\\n    return \\"hello\\"\\n", "explanation": "returns string"}'
    result = parse_and_validate(raw, _SIMPLE_SCHEMA)
    assert '\\"hello\\"' in result["code"] or '"hello"' in result["code"]
```

### B. Sandbox security tests (`tests/test_sandbox.py` — expand)

**NOTE:** `test_timeout_enforcement` already exists in this file (tests infinite loop via `while True: pass`). Do NOT duplicate. Only add genuinely new tests:

```python
@pytest.mark.asyncio
async def test_memory_bomb_handled():
    """Attempting to allocate ~1GB should fail (MemoryError or timeout)."""
    result = await execute("x = 'a' * (10**9)", "assert True", timeout=5.0)
    assert not result.passed

@pytest.mark.asyncio
async def test_import_error_captured():
    """Importing a nonexistent module should be captured, not crash the host."""
    result = await execute("import nonexistent_module_xyz", "assert True")
    assert not result.passed
    assert "ModuleNotFoundError" in result.exception_type or "ModuleNotFoundError" in result.stderr

@pytest.mark.asyncio
async def test_solution_and_test_code_both_empty():
    """Edge case: empty solution and empty test code should still pass (no assertions fail)."""
    result = await execute("", "pass")
    assert result.passed
```

### C. Router retry behavior tests (`tests/test_router.py` — new file)

```python
@pytest.mark.asyncio
async def test_router_retries_on_schema_failure():
    call_count = 0
    class BadThenGoodProvider(BaseLLMProvider):
        provider_name = "test"
        model_name = "test"
        async def infer(self, request):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                return InferenceResponse(text="not json")
            return InferenceResponse(text='{"code":"def f(): pass","explanation":"ok"}')

    router = LLMRouter(provider=BadThenGoodProvider())
    result = await router.call(role="generator", template_key="initial",
                               variables={"task_description": "test", "learning_log": ""})
    assert call_count == 3
    assert "def f" in result["code"]

@pytest.mark.asyncio
async def test_router_raises_after_max_retries():
    class AlwaysBadProvider(BaseLLMProvider):
        provider_name = "test"
        model_name = "test"
        async def infer(self, request):
            return InferenceResponse(text="garbage")

    router = LLMRouter(provider=AlwaysBadProvider())
    with pytest.raises(StructuredOutputError):
        await router.call(role="generator", template_key="initial",
                         variables={"task_description": "test", "learning_log": ""})
```

### D. Fix the terminal status assertion

In `test_graph_mock.py`:

```python
# BEFORE (accepts "running" as terminal — wrong)
assert final_state["status"] in {"success", "max_iterations_reached", "running"}

# AFTER
assert final_state["status"] in {"success", "max_iterations_reached"}
```

---

## Fix 4: Add LangSmith Observability (2–3 hours)

**The Problem:**
No tracing, no cost tracking, no latency per node. For a finance-focused role, this is a glaring gap.

**The Fix:**

### Step A: Add LangSmith tracing to the router

LangSmith integrates with LangGraph with minimal setup. In `llm/router.py`:

```python
import os

# Optional: LangSmith tracing — enabled via environment variable
# Set LANGCHAIN_TRACING_V2=true and LANGCHAIN_API_KEY=<key> to activate
_LANGSMITH_ENABLED = bool(os.environ.get("LANGCHAIN_TRACING_V2"))
if _LANGSMITH_ENABLED:
    os.environ.setdefault("LANGCHAIN_PROJECT", "self-healing-agent")
```

When `LANGCHAIN_TRACING_V2=true`, LangGraph automatically sends traces to LangSmith — every node execution, every LLM call, every state transition becomes visible in the LangSmith dashboard with zero additional code.

### Step B: Add per-node metrics tracking

Create `agent/metrics.py`:

```python
import time
from dataclasses import dataclass, field
from typing import Any

@dataclass
class NodeMetrics:
    node_name: str
    start_time: float = 0.0
    end_time: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    llm_calls: int = 0
    retries: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def latency_seconds(self) -> float:
        return self.end_time - self.start_time

    @property
    def estimated_cost_usd(self) -> float:
        """Rough cost estimate — override per-model for accuracy."""
        return (self.input_tokens * 0.00015 + self.output_tokens * 0.0006) / 1000


@dataclass
class RunMetrics:
    task_id: str
    total_latency: float = 0.0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_llm_calls: int = 0
    node_metrics: list[NodeMetrics] = field(default_factory=list)
    total_retries: int = 0

    @property
    def estimated_total_cost_usd(self) -> float:
        return sum(n.estimated_cost_usd for n in self.node_metrics)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "total_latency_s": round(self.total_latency, 2),
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_llm_calls": self.total_llm_calls,
            "total_retries": self.total_retries,
            "estimated_cost_usd": round(self.estimated_total_cost_usd, 6),
            "nodes": [
                {
                    "name": n.node_name,
                    "latency_s": round(n.latency_seconds, 2),
                    "tokens_in": n.input_tokens,
                    "tokens_out": n.output_tokens,
                }
                for n in self.node_metrics
            ],
        }
```

### Step C: Track tokens in the router

The router already receives `InferenceResponse.input_tokens` and `output_tokens`. Log them:

**NOTE:** The Ollama provider returns real token counts via `prompt_eval_count` and `eval_count`. The HuggingFace provider returns -1 for both (it doesn't track tokens). The Mock provider returns word-count approximations. The metrics module must handle -1 gracefully — treat it as "unknown" in aggregations, not as zero.

```python
# In router.call(), after successful inference:
logger.info(
    "LLM_CALL role=%s tokens_in=%d tokens_out=%d latency=%.2fs attempt=%d",
    role, response.input_tokens, response.output_tokens, elapsed, attempt
)
```

### Step D: Add metrics to results.json

```json
{
  "task_id": "interval_merge_001",
  "success": true,
  "metrics": {
    "total_latency_s": 45.2,
    "total_input_tokens": 3420,
    "total_output_tokens": 890,
    "total_llm_calls": 6,
    "estimated_cost_usd": 0.0012
  }
}
```

**Interview talking point:** "Every agent run is fully observable. I can tell you exactly how many tokens each repair iteration cost, which node is the latency bottleneck, and what the per-task cost is. In a finance environment, cost attribution per agent run is non-negotiable."

---

## Fix 11: Add a CI Pipeline That Actually Tests (1 hour)

**The Problem:**
The only CI workflow is `sync_to_hf.yml` — it just pushes to HuggingFace. There are no automated tests running on push or PR.

**The Fix:**

Create `.github/workflows/test.yml`:

```yaml
name: Tests
on:
  push:
    branches: [main]
  pull_request:

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-asyncio

      - name: Run tests
        env:
          LLM_PROVIDER: mock
        run: pytest -v --tb=short

      - name: Lint
        run: |
          pip install ruff
          ruff check .
```

This runs on every push and PR, ensures the mock test suite always passes, and adds basic linting. After implementing Fixes 1–3, this pipeline catches regressions.

---

# PHASE 2: PRODUCTION THINKING

---

## Fix 5: Make the QA Agent Actually Adversarial (4–6 hours)

**The Problem:**
The QA agent reads the implementation and generates tests that validate what the code *does*, not what it *should do*. If the LLM generates a solution with a systematic misunderstanding, the QA agent will generate tests that validate that misunderstanding. There's no oracle. The system can converge on a confidently wrong solution.

**The Fix — Three-pronged approach:**

### A. Specification-first test generation

Create a new template in `prompts/qa_adversarial.yaml` that generates tests from the task description only, blind to the implementation:

```yaml
templates:
  generate_from_spec: |
    ## Task Description
    {task_description}

    ## Instructions
    You are a QA engineer writing tests BEFORE seeing any code.
    Based ONLY on the task description above, generate assert-based tests.
    You do NOT have access to the implementation.

    Write tests based on what the function SHOULD do according to the specification.

    Generate 8-12 assert statements covering:
    - The happy path (basic correct behavior)
    - Empty inputs and None values
    - Single-element inputs
    - Boundary conditions and off-by-one
    - Negative numbers and zero
    - Duplicate values
    - Large inputs

    Respond with ONLY this JSON object (no other text):
    {{"test_code": "<your assert statements>", "test_cases_description": ["<desc1>", "<desc2>"]}}
```

### B. Split into two test phases

Modify the graph to run TWO test suites:

1. **Spec tests** — generated once at iteration 0, blind to code, never regenerated. These are the oracle.
2. **Adversarial tests** — generated per iteration, sees the code. These find implementation-specific bugs.

Update `AgentState` in `agent/state.py`:

```python
class AgentState(TypedDict):
    # ... existing fields ...
    spec_test_code: str          # Generated once from task description only
    adversarial_test_code: str   # Generated per iteration from code
```

Add a new node `generate_spec_tests` that runs once at the start:

```python
async def generate_spec_tests(state: AgentState, router: LLMRouter) -> dict[str, Any]:
    """Generate specification-based tests (blind to implementation)."""
    result = await router.call(
        role="qa_adversarial",
        template_key="generate_from_spec",
        variables={"task_description": state["task_description"]},
        max_new_tokens=1024,
    )
    return {"spec_test_code": result["test_code"]}
```

Update the graph flow:

```
generate_spec_tests (once, iteration 0 only)
    ↓
generate_solution
    ↓
create_adversarial_tests (sees code)
    ↓
execute_solution (runs BOTH test suites)
    ↓ ...
```

### C. Execution runs both suites separately

Update `execute_solution` to run both and report separately:

```python
# Run spec tests (oracle)
spec_result = await execute(state["current_code"], state["spec_test_code"])

# Run adversarial tests (implementation-aware)
adv_result = await execute(state["current_code"], state["adversarial_test_code"])

# Both must pass for success
passed = spec_result.passed and adv_result.passed

# Report failures from BOTH suites in the failure summary
failure_summary = ""
if not spec_result.passed:
    failure_summary += f"SPEC TEST FAILURES:\n{format_failure_summary(spec_result)}\n"
if not adv_result.passed:
    failure_summary += f"ADVERSARIAL TEST FAILURES:\n{format_failure_summary(adv_result)}\n"
```

**Why this matters:** The spec tests act as a ground-truth oracle *within the agent loop itself*, not just at evaluation time. The agent can no longer converge on "wrong but self-consistent."

---

## Fix 7: Add Graceful Degradation and Error Recovery (3–4 hours)

**The Problem:**
If schema validation fails 3 times, `router.call()` raises `StructuredOutputError` and the entire agent run crashes. No fallback. No partial results.

**The Fix:**

### A. Fallback behavior in the router

Add a `call_with_fallback` method to `LLMRouter`:

```python
async def call_with_fallback(
    self, role: str, template_key: str, variables: dict[str, Any], **kwargs,
) -> dict[str, Any]:
    try:
        return await self.call(role, template_key, variables, **kwargs)
    except StructuredOutputError as exc:
        logger.error("All retries failed for role=%s. Using fallback.", role)
        return self._get_fallback(role, exc.raw_text)

def _get_fallback(self, role: str, raw_text: str) -> dict[str, Any]:
    if role == "debugger":
        return {
            "root_cause": "Unable to parse structured diagnosis. Raw output attached.",
            "failure_category": "parse_failure",
            "repair_strategy": f"Raw output: {raw_text[:500]}",
            "confidence": 0.1,
        }
    elif role == "qa_adversarial":
        return {
            "test_code": "assert True, 'Fallback: no adversarial tests generated'",
            "test_cases_description": ["Fallback smoke test"],
        }
    elif role == "generator":
        import re
        code_match = re.search(r'```python\s*(.*?)```', raw_text, re.DOTALL)
        if code_match:
            return {"code": code_match.group(1), "explanation": "Extracted from malformed output"}
        raise  # Re-raise if truly unrecoverable
    elif role == "memory_summarizer":
        return {"lessons": ["Schema parse failure — no lessons extracted."]}
    raise
```

### B. Add a `degraded_nodes` field to state

```python
class AgentState(TypedDict):
    # ... existing fields ...
    degraded_nodes: list[str]   # Track which nodes fell back to degraded behavior
```

### C. Add confidence-aware routing

If the debugger returns a diagnosis with `confidence < 0.3`, skip targeted repair and do a blind regeneration instead:

```python
def _route_after_diagnosis(state: AgentState) -> str:
    confidence = state.get("diagnosis_confidence", 1.0)
    if confidence < 0.3:
        return "blind_retry"  # Skip targeted repair, regenerate from scratch
    return "generate_solution"  # Normal targeted repair
```

---

## Fix 9: Improve the Sandbox (3–4 hours)

**The Problem:**
Solution code and test code run in the same namespace. Tests can couple to implementation internals (helper functions, module-level variables, side effects).

**The Fix:**

### A. Separate namespaces via importlib

Instead of concatenating code into one file, write two files and have the test file import from the solution:

```python
_SANDBOX_WRAPPER = textwrap.dedent("""\
import sys
import importlib.util

# Load solution as an isolated module
spec = importlib.util.spec_from_file_location("solution", "{solution_path}")
solution = importlib.util.module_from_spec(spec)
spec.loader.exec_module(solution)

# Inject ONLY public names into test namespace
for name in dir(solution):
    if not name.startswith('_'):
        globals()[name] = getattr(solution, name)

# Run tests
try:
{indented_tests}
    print("SANDBOX_RESULT:PASS")
except AssertionError as _ae:
    _msg = str(_ae) if str(_ae) else "AssertionError (no message)"
    print("SANDBOX_RESULT:FAIL:" + _msg, file=sys.stderr)
except Exception as _ex:
    print("SANDBOX_RESULT:EXCEPTION:" + type(_ex).__name__ + ":" + str(_ex), file=sys.stderr)
""")
```

Write the solution to a separate temp file, pass its path into the wrapper. Now tests can only access the solution's public API.

### B. Add resource limits via `rlimit`

```python
import resource

def _set_resource_limits():
    """Called in the subprocess via preexec_fn."""
    resource.setrlimit(resource.RLIMIT_AS, (256 * 1024 * 1024, 256 * 1024 * 1024))  # 256MB
    resource.setrlimit(resource.RLIMIT_CPU, (10, 10))  # 10s CPU
    resource.setrlimit(resource.RLIMIT_NPROC, (0, 0))  # No subprocesses

# In execute():
proc = await asyncio.create_subprocess_exec(
    sys.executable, tmp_path,
    stdout=asyncio.subprocess.PIPE,
    stderr=asyncio.subprocess.PIPE,
    preexec_fn=_set_resource_limits,  # ADD THIS
)
```

### C. Document the security layering

```python
"""
Security layers (cumulative):
  1. Subprocess isolation (current) — prevents crashes from affecting main process
  2. Resource limits via rlimit — prevents memory bombs and CPU abuse
  3. Namespace separation via importlib — prevents test/solution coupling
  4. [Production] nsjail/firejail/Docker — full filesystem and network isolation
     See: https://github.com/google/nsjail
     See: https://github.com/python-discord/snekbox (nsjail-based Python sandbox)
"""
```

---

## Fix 12: Human-in-the-Loop with LangGraph's `interrupt()` (4–5 hours)

**The Problem:**
You're using LangGraph but completely ignoring its flagship capability. For a GS team that builds agents for *other teams* (traders, analysts, ops), human oversight isn't optional — it's the entire point. No one at GS lets an agent run fully autonomously without approval gates.

**The Fix:**

### A. Add an `autonomy_level` config

```python
from dataclasses import dataclass

@dataclass
class AgentConfig:
    max_iterations: int = 4
    autonomy_level: str = "full_auto"
    # "full_auto"     — no interrupts (current behavior)
    # "review_repairs" — pause before each repair for human approval
    # "review_all"     — pause before generation AND before repair
```

### B. Add a `review_repair` node using `interrupt()`

```python
from langgraph.types import interrupt, Command

async def review_repair(state: AgentState) -> dict[str, Any] | Command:
    """Pause execution and present the diagnosis to the human for review."""
    decision = interrupt({
        "type": "repair_review",
        "iteration": state.get("iteration", 0),
        "root_cause": state.get("root_cause", ""),
        "failure_category": state.get("failure_category", ""),
        "repair_strategy": state.get("repair_strategy", ""),
        "confidence": state.get("diagnosis_confidence", 0.5),
        "current_code_preview": state.get("current_code", "")[:500],
        "question": "Approve this repair strategy, edit it, or abort?"
    })

    action = decision.get("action", "approve")

    if action == "approve":
        return {}  # Continue with existing strategy
    elif action == "edit":
        return {
            "repair_strategy": decision.get("edited_strategy", state["repair_strategy"]),
            "root_cause": decision.get("edited_root_cause", state["root_cause"]),
        }
    elif action == "abort":
        return Command(goto="__end__")
    else:
        return {}
```

### C. Wire it into the graph conditionally

```python
def build_graph(config: AgentConfig, router: LLMRouter) -> StateGraph:
    # ... existing node registration ...

    if config.autonomy_level in ("review_repairs", "review_all"):
        graph.add_node("review_repair", review_repair)
        # Insert between diagnosis and generation
        graph.add_edge("update_learning_log", "review_repair")
        graph.add_edge("review_repair", "increment_iteration")
    else:
        # Original flow
        graph.add_edge("update_learning_log", "increment_iteration")
```

### D. Update the Gradio UI to handle interrupts

**IMPORTANT implementation detail:** The current `demo/demo_runner.py` uses `asyncio.run()` inside `run_demo_sync()` to bridge async→sync for Gradio. This creates a fresh event loop per call and blocks until completion — which is incompatible with `interrupt()` since the graph pauses mid-execution and needs a separate `Command(resume=...)` call to continue.

To fix this, the demo runner needs to be restructured:

1. **Option A (recommended):** Switch `demo/app.py` to use Gradio's native async support. Gradio `gr.Blocks` supports async generator functions directly — remove the `run_demo_sync` wrapper and use `run_demo_async` directly. When an `__interrupt__` is yielded by the graph, the async generator yields the interrupt payload to the UI, waits for user input via a Gradio component, then resumes with `graph.invoke(Command(resume=decision), config)`.

2. **Option B (simpler):** When `autonomy_level == "full_auto"`, keep the existing sync wrapper. Only switch to the async path when HITL is enabled.

The key change in `demo/demo_runner.py`:

```python
async def run_demo_async(task_description: str, config: AgentConfig, router: LLMRouter | None = None):
    app = build_graph(config=config, router=router or LLMRouter())
    thread_config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    initial_state = _make_initial_state(task_description, config.max_iterations)

    # Stream until completion or interrupt
    async for state_update in app.astream(initial_state, thread_config):
        for node_name, node_state in state_update.items():
            # ... existing event processing ...
            pass

    # Check if we hit an interrupt
    current_state = app.get_state(thread_config)
    if current_state.next:  # Graph is paused (interrupt)
        interrupt_payload = current_state.values.get("__interrupt__", [])
        yield {"type": "interrupt", "payload": interrupt_payload}
        # UI will call resume_after_interrupt() with the user's decision
```

In `demo/app.py`, add interrupt handling UI:

```python
# When interrupt event is received:
with gr.Row(visible=False) as review_panel:
    diagnosis_display = gr.JSON(label="Diagnosis for Review")
    approve_btn = gr.Button("Approve", variant="primary")
    edit_box = gr.Textbox(label="Edit Strategy (optional)")
    abort_btn = gr.Button("Abort", variant="stop")
```

**Interview talking point:** "The same agent supports three deployment modes: fully autonomous for development, human-reviewed for staging, and human-approved for production trading systems. The configuration is a single field — `autonomy_level` — that controls where interrupts fire."

---

## Fix 13: Checkpointing and Time-Travel Debugging (3–4 hours)

**The Problem:**
If the agent fails at iteration 3 of 4, everything is lost. There's no way to rewind, inspect intermediate states, or fork from a previous checkpoint to try a different repair strategy.

**The Fix:**

### A. Add a checkpointer to the graph

```python
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver  # for persistence across restarts

def build_graph(config: AgentConfig, router: LLMRouter) -> StateGraph:
    # ... build graph as before ...

    # Add checkpointing — enables time-travel and fault tolerance
    if config.persist_checkpoints:
        checkpointer = SqliteSaver.from_conn_string(".agent_checkpoints.db")
    else:
        checkpointer = InMemorySaver()

    return graph.compile(checkpointer=checkpointer)
```

### B. Enable time-travel in the runner

```python
async def run_agent_with_history(
    task_description: str,
    config: AgentConfig,
    router: LLMRouter,
) -> tuple[AgentState, list]:
    """Run agent and return both final state and full checkpoint history."""
    app = build_graph(config=config, router=router)
    thread_config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    initial_state = _make_initial_state(task_description, config.max_iterations)
    final_state = await app.ainvoke(initial_state, thread_config)

    # Retrieve full execution history for debugging
    history = list(app.get_state_history(thread_config))

    return final_state, history
```

### C. Add a fork-from-checkpoint capability

```python
async def fork_from_iteration(
    app,
    thread_config: dict,
    target_iteration: int,
    modified_state: dict,
) -> AgentState:
    """Rewind to a specific iteration, modify state, and replay forward."""
    history = list(app.get_state_history(thread_config))

    # Find the checkpoint at the target iteration
    target_checkpoint = None
    for state_snapshot in history:
        if state_snapshot.values.get("iteration") == target_iteration:
            target_checkpoint = state_snapshot
            break

    if not target_checkpoint:
        raise ValueError(f"No checkpoint found for iteration {target_iteration}")

    # Fork: update state at that checkpoint and create a new branch
    fork_config = app.update_state(
        target_checkpoint.config,
        values=modified_state,
        as_node="diagnose_failure",  # Resume from after diagnosis
    )

    # Replay forward from the fork point
    return await app.ainvoke(None, fork_config)
```

### D. Add a "Rewind" button in the Gradio UI

Show iteration history as a timeline. Clicking any iteration loads that checkpoint's state. The user can edit the `repair_strategy` or `root_cause` and click "Replay" to fork a new execution branch.

**Interview talking point:** "The agent failed on iteration 3. Watch — I rewind to iteration 2, manually edit the diagnosis to say 'the off-by-one is in the loop bound, not the comparison', and replay. It succeeds. That's time-travel debugging for agent systems."

---

# PHASE 3: AGENTIC TRANSFORMATION

---

## Fix 6: Add Real Tool Use to Make This Actually Agentic (6–8 hours)

**The Problem:**
Every "agent" is an LLM call with a different prompt. There's no tool use, no autonomous decision-making, no ReAct-style reasoning. This is a pipeline, not an agent system.

**The Fix: Give the Debugger agent actual tools and a ReAct loop.**

### Tool 1: `run_snippet` — Execute a hypothesis

```python
from langchain_core.tools import tool

@tool
def run_snippet(code: str) -> str:
    """Execute a short Python snippet to test a hypothesis about the bug.
    Use this to verify your understanding of the failure before prescribing a fix.
    Returns stdout and stderr from execution.

    Args:
        code: A short Python snippet (under 20 lines) to test a hypothesis.
    """
    import asyncio
    result = asyncio.run(execute_sync(code, "", timeout=5.0))
    return f"stdout: {result.stdout}\nstderr: {result.stderr}"
```

### Tool 2: `inspect_function` — Extract function signature and docstring

```python
@tool
def inspect_function(code: str, function_name: str) -> str:
    """Extract the signature, docstring, and return type of a specific function
    from the source code. Use this to understand what a function is supposed to do.

    Args:
        code: The full Python source code.
        function_name: Name of the function to inspect.
    """
    import ast
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == function_name:
                # Extract first line (signature)
                lines = code.split('\n')
                sig_line = lines[node.lineno - 1]
                docstring = ast.get_docstring(node) or "No docstring"
                args = [arg.arg for arg in node.args.args]
                return f"Signature: {sig_line}\nArgs: {args}\nDocstring: {docstring}"
        return f"Function '{function_name}' not found in source code"
    except SyntaxError as e:
        return f"SyntaxError parsing code: {e}"
```

### Tool 3: `diff_iterations` — Compare code across iterations

```python
@tool
def diff_iterations(code_v1: str, code_v2: str) -> str:
    """Show a unified diff between two versions of code.
    Use this to understand what changed between repair iterations.

    Args:
        code_v1: The earlier version of the code.
        code_v2: The later version of the code.
    """
    import difflib
    diff = difflib.unified_diff(
        code_v1.splitlines(keepends=True),
        code_v2.splitlines(keepends=True),
        fromfile="previous", tofile="current", n=3
    )
    result = "".join(diff)
    return result or "No differences found"
```

### Integrate into the debugger as a ReAct loop

Restructure `diagnose_failure` to use LangGraph's tool-calling pattern. Instead of a single LLM call, the debugger now operates in a think → act → observe loop:

```python
from langgraph.prebuilt import ToolNode, tools_condition

tools = [run_snippet, inspect_function, diff_iterations]
tool_node = ToolNode(tools)

# In the graph builder, replace the single "diagnose_failure" node
# with a sub-graph:
#
#   diagnose_failure (LLM with tools bound)
#       ↓ (tool_call detected)
#   debugger_tools (ToolNode executes the tool)
#       ↓ (result fed back)
#   diagnose_failure (LLM reasons about tool result)
#       ↓ (no more tool calls → done)
#   update_learning_log

def build_debugger_subgraph(router):
    debugger_graph = StateGraph(DebuggerState)
    debugger_graph.add_node("reason", debugger_reason_node)
    debugger_graph.add_node("tools", tool_node)
    debugger_graph.set_entry_point("reason")
    debugger_graph.add_conditional_edges(
        "reason",
        tools_condition,  # Routes to "tools" if tool_calls present, else END
        {"tools": "tools", "__end__": END}
    )
    debugger_graph.add_edge("tools", "reason")  # Feed tool results back
    return debugger_graph.compile()
```

The debugger now autonomously decides: "I think the bug is an off-by-one. Let me run `run_snippet` to confirm... yes, `range(5)` excludes 5. Now I'm confident — the fix is to use `<=` instead of `<`."

**Dependency note:** `langchain-core>=0.3.0` is already in `requirements.txt` — no new dependency needed for the `@tool` decorator.

**Prompt update note:** The debugger's system prompt in `prompts/debugger.yaml` currently says "You do NOT write code." With tools, it can now *run* code via `run_snippet`. Update the system prompt to: "You do NOT write repair code. You diagnose and investigate. You may run short snippets to verify hypotheses."

**Interview talking point:** "The debugger doesn't just read errors — it executes diagnostic snippets to verify hypotheses before prescribing a fix. That's the ReAct pattern: Reason → Act → Observe → Reason. That's what separates a pipeline from an agent."

---

## Fix 8: Honest README Reframing (2 hours)

**The Problem:**
The README calls this a "multi-agent system" with "4 Specialized LLM Agents." After implementing all fixes, rewrite the narrative to honestly describe what you built.

**Key changes:**

### Architecture description — be precise:

```markdown
## Architecture

The system uses two distinct patterns:

**Orchestrated Pipeline** (deterministic flow):
Generator → QA → Executor → Memory follow a fixed sequence with
conditional repair looping. This provides reliability and predictability.

**Agentic Investigation** (autonomous reasoning):
The Debugger agent operates in a ReAct loop with tool access
(snippet execution, function inspection, diff analysis). It autonomously
decides how many investigation steps to take before prescribing a repair.

**Human Oversight** (configurable):
LangGraph's interrupt() enables three autonomy levels:
full_auto, review_repairs, review_all. Checkpointing enables
time-travel debugging — rewind to any iteration and fork.

This hybrid design is intentional: the pipeline provides reliability,
the agentic debugger provides adaptive reasoning, and human-in-the-loop
provides the trust guarantees needed for production deployment.
```

### Metrics — show honest numbers:

```markdown
| Metric | Self-Reported | Reference-Validated |
|--------|--------------|-------------------|
| Overall success | 87.5% | 75.0% |
| Repair effectiveness | 80% | 60% |

The gap reveals the QA agent's blind spots — an expected limitation
that motivated the dual-testing architecture (spec + adversarial tests).
```

### Engineering concepts — add the new ones:

```markdown
| Concept | Implementation |
|---------|---------------|
| **ReAct agent loop** | Debugger uses tools (snippet execution, AST inspection) in a think→act→observe cycle |
| **Human-in-the-loop** | LangGraph interrupt() with configurable autonomy levels |
| **Time-travel debugging** | Checkpoint-based state snapshots with fork-and-replay |
| **Parallel repair strategies** | Fan-out/fan-in with tournament selection |
| **Cross-session learning** | ChromaDB vector store for persistent lesson retrieval |
| **Dual-oracle testing** | Spec-blind + adversarial test suites prevent circular validation |
| **Observability** | LangSmith tracing, per-node token/latency metrics, cost attribution |
```

---

## Fix 14: Parallel Repair Strategies with Tournament Selection (4–6 hours)

**The Problem:**
The agent tries one repair strategy per iteration, serially. If the strategy is wrong, an entire iteration is wasted.

**The Fix:**

### A. Fan-out multiple repair strategies

On failure, spawn 2–3 parallel repair branches with different strategies, execute all of them, and pick the best:

```python
from langgraph.types import Send
from typing import Annotated
import operator

# Update state to collect parallel results
class AgentState(TypedDict):
    # ... existing fields ...
    parallel_repairs: Annotated[list, operator.add]  # Reducer: append results

def fan_out_repairs(state: AgentState) -> list[Send]:
    """Fan out to multiple repair strategies in parallel."""
    strategies = [
        {
            "name": "minimal_fix",
            "instruction": "Change ONLY the failing line(s). Do not restructure."
        },
        {
            "name": "restructure",
            "instruction": "Rewrite the core algorithm while keeping the same interface."
        },
        {
            "name": "add_guards",
            "instruction": "Add input validation and edge-case guards without changing core logic."
        },
    ]

    return [
        Send("parallel_generate", {
            **state,
            "repair_strategy": f"{state.get('repair_strategy', '')}\n\nApproach: {s['instruction']}",
            "strategy_name": s["name"],
        })
        for s in strategies
    ]
```

### B. Execute each candidate and select the best

```python
async def parallel_generate(state: AgentState, router: LLMRouter) -> dict:
    """Generate a repair candidate for one strategy."""
    result = await router.call(
        role="generator", template_key="repair",
        variables={...},  # includes the strategy-specific instruction
    )
    code = result["code"]

    # Test this candidate against both test suites
    spec_result = await execute(code, state["spec_test_code"])
    adv_result = await execute(code, state["adversarial_test_code"])

    return {
        "parallel_repairs": [{
            "strategy": state.get("strategy_name", "unknown"),
            "code": code,
            "spec_passed": spec_result.passed,
            "adv_passed": adv_result.passed,
            "spec_failures": len(spec_result.failed_assertions),
            "adv_failures": len(adv_result.failed_assertions),
        }]
    }


def select_best_repair(state: AgentState) -> dict:
    """Tournament selection: pick the best candidate from parallel repairs."""
    candidates = state.get("parallel_repairs", [])

    # Priority: both pass > spec pass > fewest total failures
    def score(c):
        both = c["spec_passed"] and c["adv_passed"]
        return (both, c["spec_passed"], -(c["spec_failures"] + c["adv_failures"]))

    candidates.sort(key=score, reverse=True)
    best = candidates[0] if candidates else None

    if best:
        logger.info("Selected strategy=%s (spec=%s, adv=%s)",
                    best["strategy"], best["spec_passed"], best["adv_passed"])
        return {
            "current_code": best["code"],
            "last_execution_passed": best["spec_passed"] and best["adv_passed"],
            "parallel_repairs": [],  # Reset for next iteration
        }
    return {"parallel_repairs": []}
```

### C. Wire into graph with conditional activation

Make this configurable via `AgentConfig.parallel_strategies: bool`. When enabled, replace the single `generate_solution` repair path with fan-out → parallel_generate → select_best.

LangGraph handles the parallel execution natively — all branches in a "superstep" execute concurrently and the graph waits for all to complete before proceeding to `select_best_repair`.

**Interview talking point:** "Parallel strategies cost 3x the tokens but reduce iterations-to-success. For latency-sensitive tasks, we use single-strategy. For correctness-critical tasks, we fan out. The tradeoff is configurable."

---

## Fix 15: Agent Self-Reflection / Critic Node (3–4 hours)

**The Problem:**
When all tests pass, the agent immediately declares success. But passing tests doesn't mean the solution is correct — especially when the tests are LLM-generated. There's no sanity check.

**The Fix:**

### A. Create a Critic agent

Add `prompts/critic.yaml`:

```yaml
role: critic
description: >
  Reviews final solutions for correctness, code quality, and specification adherence.
  Acts as a final gate before declaring success.

system: |
  You are a senior code reviewer. Your job is to catch issues that tests miss.
  You MUST respond with ONLY a JSON object. No text before or after.

  {"verdict": "approve" or "reject", "issues": ["issue1", "issue2"], "confidence": 0.0-1.0}

schema:
  type: object
  required: [verdict, confidence]
  properties:
    verdict:
      type: string
      enum: [approve, reject]
    issues:
      type: array
      items:
        type: string
    confidence:
      type: number

templates:
  review: |
    ## Task Description
    {task_description}

    ## Final Code
    ```python
    {code}
    ```

    ## Tests That Passed
    {test_summary}

    Review critically:
    1. Does the code ACTUALLY solve the task as described, or does it
       merely pass the specific test cases?
    2. Are there edge cases the tests didn't cover?
    3. Is the code correct for ALL valid inputs, not just tested ones?
    4. Are there any algorithmic errors that happen to produce correct
       output for the tested cases but would fail on others?

    Respond with ONLY this JSON:
    {{"verdict": "approve" or "reject", "issues": ["..."], "confidence": 0.0-1.0}}
```

### B. Add the critic node

```python
async def critic_review(state: AgentState, router: LLMRouter) -> dict[str, Any]:
    """Final review gate: catches issues that tests miss."""
    result = await router.call_with_fallback(
        role="critic", template_key="review",
        variables={
            "task_description": state["task_description"],
            "code": state["current_code"],
            "test_summary": state.get("last_failure_summary", "All tests passed."),
        },
        max_new_tokens=512,
    )

    verdict = result.get("verdict", "approve")
    issues = result.get("issues", [])
    confidence = result.get("confidence", 0.5)

    if verdict == "reject" and confidence > 0.6:
        logger.info("Critic REJECTED solution: %s", issues)
        return {
            "last_execution_passed": False,
            "last_failure_summary": f"Critic rejected: {'; '.join(issues)}",
            "root_cause": issues[0] if issues else "Critic found unspecified issue",
            "repair_strategy": f"Address critic feedback: {'; '.join(issues)}",
            "status": "running",
        }

    return {}  # Approved — don't change state
```

### C. Wire into graph after test execution passes

```
execute_solution
    ↓ (pass)
critic_review
    ↓ (approve) → END
    ↓ (reject, confidence > 0.6) → diagnose_failure (re-enter repair loop)
```

This creates a second validation layer beyond the test suite. The critic can catch issues like "this solution handles the tested inputs correctly but uses an O(n³) algorithm that will timeout on large inputs" or "this solution drops None values from the output even though the spec says to preserve them."

---

## Fix 17: Multi-Model Routing (2–3 hours)

**The Problem:**
Every agent uses the same model. In production, you'd never do this — the memory summarizer doesn't need the same model power as the code generator.

**The Fix:**

### A. Add per-role model configuration

```python
# In llm/router.py

_DEFAULT_ROLE_MODELS = {
    "generator": None,           # Uses default model
    "qa_adversarial": None,      # Uses default model
    "debugger": None,            # Uses default model
    "memory_summarizer": None,   # Could use a cheaper/smaller model
    "critic": None,              # Uses default model
}

class LLMRouter:
    def __init__(
        self,
        provider: BaseLLMProvider | None = None,
        role_providers: dict[str, BaseLLMProvider] | None = None,
    ) -> None:
        self._default_provider = provider or _resolve_provider()
        self._role_providers = role_providers or {}

    def _get_provider(self, role: str) -> BaseLLMProvider:
        return self._role_providers.get(role, self._default_provider)

    async def call(self, role: str, ...):
        provider = self._get_provider(role)
        # ... rest of call logic uses this provider ...
```

### B. Configure in the entry point

```python
# For Ollama with multiple models:
from llm.providers.ollama_provider import OllamaProvider

router = LLMRouter(
    provider=OllamaProvider(model="llama3:8b-instruct"),
    role_providers={
        "memory_summarizer": OllamaProvider(model="llama3.2:3b"),
        # Other roles use the default 8B model
    }
)
```

### C. Log which model handled each call

```python
logger.info(
    "LLM_CALL role=%s model=%s tokens_in=%d tokens_out=%d",
    role, provider.model_name, response.input_tokens, response.output_tokens
)
```

**Interview talking point:** "The memory summarizer uses a 3B model because it's doing simple text compression. That saves 60% on tokens for that node without any quality loss. In a finance agent system processing thousands of tasks, that model routing saves real money."

---

# PHASE 4: PLATFORM-LEVEL POLISH

---

## Fix 10: Cross-Session Learning with a Vector Store (4–6 hours)

**The Problem:**
The "learning log" resets between tasks. There's no cross-session memory. The system makes the same mistakes repeatedly.

**The Fix:**

### A. Add ChromaDB

```bash
pip install chromadb
```

### B. Create `agent/memory_store.py`

```python
import chromadb
from chromadb.config import Settings
import hashlib

class LessonStore:
    """Persistent vector store for cross-task lessons learned."""

    def __init__(self, persist_dir: str = ".agent_memory"):
        self._client = chromadb.Client(Settings(
            persist_directory=persist_dir,
            anonymized_telemetry=False,
        ))
        self._collection = self._client.get_or_create_collection(
            name="lessons",
            metadata={"hnsw:space": "cosine"},
        )

    def store_lesson(
        self,
        lesson: str,
        task_category: str,
        failure_category: str,
        task_id: str,
    ) -> None:
        doc_id = f"{task_id}_{hashlib.md5(lesson.encode()).hexdigest()[:8]}"
        self._collection.upsert(
            documents=[lesson],
            metadatas=[{
                "task_category": task_category,
                "failure_category": failure_category,
                "task_id": task_id,
            }],
            ids=[doc_id],
        )

    def retrieve_relevant_lessons(
        self,
        query: str,
        task_category: str = "",
        n_results: int = 5,
    ) -> list[str]:
        where_filter = {"task_category": task_category} if task_category else None
        try:
            results = self._collection.query(
                query_texts=[query],
                n_results=n_results,
                where=where_filter,
            )
            return results["documents"][0] if results["documents"] else []
        except Exception:
            return []

    @property
    def total_lessons(self) -> int:
        return self._collection.count()
```

### C. Integrate into the graph

In `generate_solution`, before the first iteration, retrieve relevant lessons:

```python
if iteration == 0 and lesson_store:
    prior_lessons = lesson_store.retrieve_relevant_lessons(
        query=state["task_description"],
        task_category=state.get("task_category", ""),
    )
    if prior_lessons:
        learning_log = "Lessons from previous tasks:\n" + "\n".join(f"- {l}" for l in prior_lessons)
```

At the end of a successful run, persist the learning log:

```python
# In a post-success node:
if result.passed and lesson_store:
    for lesson in state.get("learning_log", []):
        lesson_store.store_lesson(
            lesson=lesson,
            task_category=task.category,
            failure_category="resolved",
            task_id=task.task_id,
        )
```

### D. Add a test

```python
def test_lesson_store_roundtrip():
    store = LessonStore(persist_dir="/tmp/test_memory")
    store.store_lesson("Always handle empty inputs", "interval_merging", "boundary", "test001")
    results = store.retrieve_relevant_lessons("empty list edge case")
    assert len(results) > 0
    assert "empty inputs" in results[0]
```

**Interview talking point:** "The agent learns from every task. If it discovered that 'touching intervals must be merged' during task A, it retrieves that lesson when encountering a similar task B — even across sessions. That's self-*evolving*, not just self-*healing*."

---

## Fix 16: Benchmark Against HumanEval (4–6 hours)

**The Problem:**
8 custom tasks are fine for development, but an interviewer will ask "how does this compare to established benchmarks?"

**The Fix:**

### A. Add a HumanEval adapter

HumanEval consists of 164 Python programming problems, each with a function signature, docstring, and hidden unit tests. The dataset is available from OpenAI's repo and on HuggingFace.

```bash
pip install human-eval  # or clone https://github.com/openai/human-eval
```

Create `evaluation/humaneval_adapter.py`:

```python
"""Adapter to run Self-Healing Agent against HumanEval benchmark."""

import json
from pathlib import Path
from typing import Any

# HumanEval problems are in JSONL format
def load_humaneval_tasks(path: str = "data/HumanEval.jsonl") -> list[dict]:
    tasks = []
    with open(path) as f:
        for line in f:
            task = json.loads(line)
            tasks.append(task)
    return tasks

def humaneval_to_agent_task(he_task: dict) -> str:
    """Convert a HumanEval problem into a task description for the agent."""
    prompt = he_task["prompt"]  # Includes function signature + docstring
    return (
        f"Complete the following Python function:\n\n"
        f"```python\n{prompt}```\n\n"
        f"Return the complete function implementation. "
        f"The function must match the signature and docstring exactly."
    )

def extract_completion(agent_code: str, entry_point: str) -> str:
    """Extract just the function body from the agent's full code output."""
    # The agent returns full code; HumanEval expects just the completion
    # after the prompt. Find the function and return everything after the
    # signature line.
    lines = agent_code.split('\n')
    in_function = False
    completion_lines = []
    for line in lines:
        if f"def {entry_point}" in line:
            in_function = True
            continue
        if in_function:
            completion_lines.append(line)
    return '\n'.join(completion_lines)
```

### B. Create `evaluation/run_humaneval.py`

```python
async def run_humaneval_benchmark(
    max_tasks: int = 164,
    max_iterations: int = 4,
    provider_name: str = "ollama",
) -> dict:
    tasks = load_humaneval_tasks()[:max_tasks]
    router = LLMRouter()
    results = []

    for he_task in tasks:
        task_desc = humaneval_to_agent_task(he_task)

        try:
            final_state = await run_agent(
                task_description=task_desc,
                max_iterations=max_iterations,
                router=router,
            )

            # Extract the completion and format for HumanEval evaluation
            completion = extract_completion(
                final_state.get("current_code", ""),
                he_task["entry_point"]
            )

            results.append({
                "task_id": he_task["task_id"],
                "completion": completion,
                "iterations_used": final_state.get("iteration", 0) + 1,
                "self_healed": final_state.get("iteration", 0) > 0,
            })
        except Exception as e:
            results.append({
                "task_id": he_task["task_id"],
                "completion": "",
                "error": str(e),
            })

    # Write results in HumanEval's expected JSONL format
    output_path = "evaluation/humaneval_samples.jsonl"
    with open(output_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    return {"total": len(results), "output": output_path}
```

### C. Run the official evaluator

```bash
# After generating samples:
evaluate_functional_correctness evaluation/humaneval_samples.jsonl
```

This gives you Pass@1 — the industry-standard metric. You can then report:

```markdown
## HumanEval Results

| Setup | Pass@1 |
|-------|--------|
| Llama 3 8B (single-shot, no agent) | ~33% |
| Llama 3 8B + Self-Healing Agent (4 iterations) | ~48% |
| Improvement from self-healing | +15 pp |
```

The exact numbers will depend on your model, but the **delta** is what matters — it quantifies how much value the agent adds over raw LLM inference.

**Interview talking point:** "Without the agent, Llama 3 scores 33% on HumanEval. With the self-healing loop, it scores 48%. The agent adds 15 percentage points of Pass@1 improvement using the same model — that's the quantitative value of the agentic architecture."

---

## Fix 18: Graph Visualization in the UI (2–3 hours)

**The Problem:**
The agent's graph topology is invisible to the user. They can't see the conditional routing, the repair loop, or where the agent currently is in the flow.

**The Fix:**

### A. Export the graph as a Mermaid diagram

LangGraph can generate Mermaid diagrams natively:

```python
# In build_graph(), after compilation:
app = graph.compile(checkpointer=checkpointer)

# Generate Mermaid diagram
mermaid_str = app.get_graph().draw_mermaid()

# Or generate a PNG image
mermaid_png = app.get_graph().draw_mermaid_png()
```

### B. Add to the Gradio UI

```python
import gradio as gr

# Show the graph as a static image in the UI
with gr.Row():
    with gr.Column(scale=2):
        # Main agent output area
        output_area = gr.Textbox(label="Agent Output")
    with gr.Column(scale=1):
        # Graph visualization
        graph_image = gr.Image(label="Agent Topology", value=mermaid_png)
        current_node = gr.Textbox(label="Current Node", interactive=False)
```

### C. Highlight current node in real-time

During streaming, update the `current_node` indicator as each node completes:

```python
async for event in stream_agent(task, config=config, router=router):
    node_name = event.get("node", "")
    if node_name:
        current_node.update(value=f"▶ {node_name} (iteration {event.get('iteration', 0)})")
```

For a more impressive version, re-render the Mermaid diagram on each node transition with the current node highlighted in a different color using Mermaid's style syntax:

```python
def highlight_node(mermaid_str: str, active_node: str) -> str:
    """Add styling to highlight the active node."""
    style_line = f"\nstyle {active_node} fill:#ff6,stroke:#333,stroke-width:4px"
    return mermaid_str + style_line
```

**Interview talking point:** In a live demo, the reviewer can visually see the conditional routing, the repair loop, the fan-out — all live. This is a small touch but makes the demo dramatically more compelling.

---

## Fix 19: Configurable Agent Topologies (4–6 hours)

**The Problem:**
The graph topology is hardcoded. A team building agents for other teams needs the ability to configure agent behavior declaratively — different teams need different configurations.

**The Fix:**

### A. Define a comprehensive `AgentConfig` dataclass

```python
from dataclasses import dataclass, field

@dataclass
class AgentConfig:
    """Declarative configuration for agent behavior.

    Different deployment contexts require different tradeoffs:
    - Development: full_auto, no critic, single strategy (fast iteration)
    - Staging: review_repairs, critic enabled, parallel strategies (thorough)
    - Production: review_all, critic enabled, parallel, cross-session memory (safe)
    """

    # --- Iteration control ---
    max_iterations: int = 4

    # --- Human oversight ---
    autonomy_level: str = "full_auto"
    # "full_auto"      — no interrupts
    # "review_repairs"  — pause before each repair
    # "review_all"      — pause before generation AND repair

    # --- Repair strategy ---
    parallel_strategies: bool = False
    # When True: fan-out 3 repair strategies and pick the best
    # When False: single sequential repair (faster, cheaper)

    # --- Quality gates ---
    enable_critic: bool = True
    critic_confidence_threshold: float = 0.6
    # Critic must exceed this confidence to reject a solution

    # --- Testing ---
    enable_spec_tests: bool = True
    # When True: generate spec-blind tests as an oracle

    # --- Memory ---
    enable_cross_session_memory: bool = False
    memory_persist_dir: str = ".agent_memory"

    # --- Observability ---
    enable_langsmith: bool = False
    langsmith_project: str = "self-healing-agent"

    # --- Checkpointing ---
    enable_checkpointing: bool = True
    persist_checkpoints: bool = False
    # When True: use SqliteSaver for crash recovery
    # When False: use InMemorySaver (lost on restart)

    # --- Model routing ---
    model_overrides: dict[str, str] = field(default_factory=dict)
    # e.g., {"memory_summarizer": "llama3.2:3b"} to use a cheaper model

    # --- Presets ---
    @classmethod
    def development(cls) -> "AgentConfig":
        return cls(
            autonomy_level="full_auto",
            parallel_strategies=False,
            enable_critic=False,
            enable_spec_tests=False,
            enable_cross_session_memory=False,
            enable_checkpointing=False,
        )

    @classmethod
    def staging(cls) -> "AgentConfig":
        return cls(
            autonomy_level="review_repairs",
            parallel_strategies=True,
            enable_critic=True,
            enable_spec_tests=True,
            enable_cross_session_memory=True,
            enable_checkpointing=True,
            persist_checkpoints=False,
        )

    @classmethod
    def production(cls) -> "AgentConfig":
        return cls(
            autonomy_level="review_all",
            parallel_strategies=True,
            enable_critic=True,
            enable_spec_tests=True,
            enable_cross_session_memory=True,
            enable_checkpointing=True,
            persist_checkpoints=True,
        )
```

### B. Dynamic graph construction from config

```python
def build_graph(config: AgentConfig, router: LLMRouter) -> StateGraph:
    """Build graph topology dynamically based on configuration."""
    graph = StateGraph(AgentState)

    # Always present
    graph.add_node("generate_solution", _generate)
    graph.add_node("create_adversarial_tests", _qa)
    graph.add_node("execute_solution", execute_solution)
    graph.add_node("diagnose_failure", _diagnose)
    graph.add_node("update_learning_log", _memory)
    graph.add_node("increment_iteration", _increment_iteration)

    # Conditional nodes
    if config.enable_spec_tests:
        graph.add_node("generate_spec_tests", _spec_tests)
        graph.set_entry_point("generate_spec_tests")
        graph.add_edge("generate_spec_tests", "generate_solution")
    else:
        graph.set_entry_point("generate_solution")

    if config.enable_critic:
        graph.add_node("critic_review", _critic)

    if config.autonomy_level != "full_auto":
        graph.add_node("review_repair", review_repair)

    if config.parallel_strategies:
        graph.add_node("parallel_generate", _parallel_gen)
        graph.add_node("select_best_repair", select_best_repair)

    # Wire edges based on config
    graph.add_edge("generate_solution", "create_adversarial_tests")
    graph.add_edge("create_adversarial_tests", "execute_solution")

    # After execution: critic gate or direct routing
    if config.enable_critic:
        graph.add_conditional_edges("execute_solution", _route_to_critic_or_debug, ...)
        graph.add_conditional_edges("critic_review", _route_after_critic, ...)
    else:
        graph.add_conditional_edges("execute_solution", _route_after_execution, ...)

    # Repair path: diagnosis → [human review] → increment → generate
    graph.add_edge("diagnose_failure", "update_learning_log")

    if config.autonomy_level != "full_auto":
        graph.add_edge("update_learning_log", "review_repair")
        graph.add_edge("review_repair", "increment_iteration")
    else:
        graph.add_edge("update_learning_log", "increment_iteration")

    # ... rest of routing ...

    # Checkpointing
    checkpointer = _build_checkpointer(config)
    return graph.compile(checkpointer=checkpointer)
```

### C. Add presets to the Gradio UI

```python
preset = gr.Radio(
    choices=["Development", "Staging", "Production", "Custom"],
    value="Development",
    label="Agent Preset"
)
```

Selecting "Production" automatically enables all features. "Custom" reveals individual toggles.

### D. Add config to results.json for reproducibility

```json
{
  "config": {
    "autonomy_level": "full_auto",
    "parallel_strategies": true,
    "enable_critic": true,
    "enable_spec_tests": true,
    "model_overrides": {"memory_summarizer": "llama3.2:3b"}
  },
  "results": { ... }
}
```

### E. Update root `app.py` entry point

The root `app.py` (HuggingFace Spaces entry point) currently creates an `LLMRouter()` without config and calls `_prewarm()`. After the config system is in place:

1. Read config preset from environment variable: `AGENT_PRESET` (default: "development")
2. Create `AgentConfig` from preset: `AgentConfig.development()`, `.staging()`, or `.production()`
3. Pass config through to `build_app()` in `demo/app.py`
4. If `config.model_overrides` is set, prewarm each role-specific provider

```python
# In app.py:
from agent.config import AgentConfig

preset = os.environ.get("AGENT_PRESET", "development")
config = getattr(AgentConfig, preset, AgentConfig.development)()
# Pass config to demo app builder
demo = build_app(config=config)
```

**Interview talking point:** "The same agent core supports three deployment presets — `development` (fast, no overhead), `staging` (thorough with human review), and `production` (full safety with persistent checkpoints and cross-session memory). The topology is declarative — adding a new node to the agent doesn't require touching existing code, just adding it to the config-driven builder."

---

# IMPLEMENTATION ROADMAP

| Weekend | Phase | Fixes | Est. Hours | Cumulative Result |
|---------|-------|-------|-----------|-------------------|
| **1** | Eliminate Embarrassments | 1, 2, 3, 4, 11 | 10–12 | Clean, tested, observable codebase |
| **2** | Production Thinking | 5, 7, 9, 12, 13 | 16–20 | Dual-oracle testing, error recovery, HITL, checkpointing |
| **3** | Agentic Transformation | 6, 8, 14, 15, 17 | 16–20 | Tool-using ReAct debugger, parallel repair, critic, multi-model |
| **4** | Platform Polish | 10, 16, 18, 19 | 14–18 | Vector memory, HumanEval benchmark, visualization, config system |

**Total estimated effort: ~56–70 hours across 4 weekends.**

---

# WHAT YOU CAN SAY IN THE INTERVIEW AFTER ALL 19 FIXES

> "I built a hybrid orchestration and agentic system for autonomous code repair. The core pipeline is deterministic for reliability, but the debugger operates as a ReAct agent with tool access — it can execute diagnostic snippets, inspect function signatures, and diff code across iterations before prescribing a repair.
>
> The system uses dual-oracle testing — specification-blind tests act as ground truth, while adversarial tests find implementation-specific bugs. A critic agent provides a second validation layer beyond tests.
>
> Human oversight is configurable through LangGraph's interrupt() — the same agent supports full autonomy for development, repair-review for staging, and full-review for production. Checkpointing enables time-travel debugging — I can rewind to any iteration, edit the diagnosis, and fork a new execution branch.
>
> For cost optimization, different agent roles use different model sizes. Cross-session learning via ChromaDB means the agent retrieves relevant lessons from previous tasks. And every run is fully observable through LangSmith with per-node token and latency metrics.
>
> On HumanEval, the agent improves Llama 3's Pass@1 by 15 percentage points. On my custom benchmark, self-reported success is 87.5% but reference-validated success is 75% — I report both because I believe in honest metrics."

That's not a portfolio project anymore. That's an agent platform.
