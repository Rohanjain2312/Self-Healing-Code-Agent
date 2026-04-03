# CLAUDE.md — Self-Healing Code Agent

## Project Overview

A LangGraph-based autonomous agent that generates Python code, adversarially tests it, diagnoses failures, and iteratively repairs solutions. Currently being upgraded from a pipeline into a genuine agentic system via a 19-fix plan.

## Upgrade Plan

The full implementation plan is in `self_healing_agent_upgrade_prompt.xml` at the project root. Reference it for every fix. Implement fixes **in phase order**: Phase 1 (fixes 1,2,3,4,11) → Phase 2 (5,7,9,12,13) → Phase 3 (6,8,14,15,17) → Phase 4 (10,16,18,19).

## Context window overflow

The XML alone is 1,328 lines. Your repo is ~3,600 lines of Python. It will start forgetting earlier fix details or making changes that conflict with fixes it implemented 30 minutes ago.

Mitigation:
I will be asking a fresh Claude Code session for each phase. At the start of each session, I will say: "Read CLAUDE.md and self_healing_agent_upgrade_prompt.xml. Implement Phase N fixes only."


## Roadblocks Claude Code Will Hit During Implementation.md

This file contains the potential roadblocks claude code may hit while implementing the changes along with the mitigation steps. Keep these in mind as well as you make the updates.

## Repo Structure

```
agent/               LangGraph state machine + nodes
  config.py          AgentConfig dataclass (created in Fix 12)
  graph.py           build_graph(), run_agent(), stream_agent()
  state.py           AgentState TypedDict
  metrics.py         NodeMetrics, RunMetrics (created in Fix 4)
  memory_store.py    ChromaDB LessonStore (created in Fix 10)
  tools.py           Debugger tools: run_snippet, inspect_function, diff_iterations (created in Fix 6)
  events.py          Event constructors
  nodes/
    generate_solution.py
    create_adversarial_tests.py
    execute_solution.py
    diagnose_failure.py
    update_learning_log.py
    generate_spec_tests.py   (created in Fix 5)
    review_repair.py         (created in Fix 12)
    critic_review.py         (created in Fix 15)
    parallel_generate.py     (created in Fix 14)
llm/
  router.py          LLMRouter — single point of contact for inference
  base.py            BaseLLMProvider abstract class
  context_builder.py Token-aware context builder (Fix 1: has a bug to fix)
  schema_validator.py JSON parse + validation + coercion
  prompt_loader.py   YAML prompt loader
  providers/
    ollama_provider.py
    hf_provider.py
    mock_provider.py
sandbox/
  python_executor.py Subprocess sandbox for code execution
prompts/             YAML templates per agent role
  generator.yaml
  qa_adversarial.yaml
  debugger.yaml
  memory_summarizer.yaml
  critic.yaml        (created in Fix 15)
evaluation/
  benchmark_tasks.py 8 task definitions (Fix 2: add reference_test_code)
  metrics.py         TaskResult, compute_summary
  run_benchmark.py   Benchmark runner
  humaneval_adapter.py  (created in Fix 16)
  run_humaneval.py      (created in Fix 16)
demo/
  app.py             Gradio UI
tests/               All tests use LLM_PROVIDER=mock
```

## Key Conventions

- **Async everywhere**: All node functions are `async`. Use `await` for LLM calls and sandbox execution.
- **State is TypedDict**: `AgentState` in `agent/state.py`. Nodes return partial dicts merged into state.
- **Prompts in YAML**: Each role has a YAML file in `prompts/` with system prompt, schema, and templates.
- **Router pattern**: Nodes never call providers directly. Always go through `LLMRouter.call()` or `call_with_fallback()`.
- **Mock provider for tests**: `LLM_PROVIDER=mock` uses `MockProvider` with hardcoded JSON responses. No GPU needed.
- **Dependency injection**: Router is bound to nodes via `functools.partial` before adding to graph.
- **Events for streaming**: Nodes append event dicts to `state["events"]` for Gradio UI streaming.

## How to Run

```bash
# Tests (always use mock provider)
LLM_PROVIDER=mock pytest -v

# Run with Ollama
ollama pull llama3
python app.py

# Run with HuggingFace
LLM_PROVIDER=huggingface python app.py

# Benchmark
python -m evaluation.run_benchmark --provider ollama --max-iterations 4
```

## Coding Rules

- Python 3.11+ type hints: `dict[str, Any]` not `Dict[str, Any]`
- Docstrings on every module, class, and public function explaining design decisions
- Logging via `logging.getLogger(__name__)`
- All new features must be optional and configurable via `AgentConfig`
- Never break the default development flow — `AgentConfig.development()` must produce the same behavior as the current hardcoded graph
- Add new dependencies to `requirements.txt`
- No hardcoded API keys — use environment variables
- Run `LLM_PROVIDER=mock pytest -v` after every fix to confirm nothing is broken

## State Fields (AgentState)

```
task_description: str       # Set once at entry
max_iterations: int         # Set once at entry
current_code: str           # Updated each iteration
current_test_code: str      # Adversarial tests (per iteration)
spec_test_code: str         # Spec-blind tests (generated once) [Fix 5]
last_execution_passed: bool
last_failure_summary: str
root_cause: str
failure_category: str
repair_strategy: str
learning_log: list[str]     # Rolling max 5 lessons
iteration: int
iteration_history: list[IterationRecord]
status: str                 # running | success | failed | max_iterations_reached
events: list[dict]          # For UI streaming
degraded_nodes: list[str]   # Nodes that used fallback [Fix 7]
parallel_repairs: Annotated[list, operator.add]  # Fan-out results [Fix 14]
```

## Graph Topology (Target — after all fixes)

```
[generate_spec_tests] (once, if enabled)
    ↓
generate_solution
    ↓
create_adversarial_tests
    ↓
execute_solution (runs BOTH spec + adversarial tests)
    ↓ (pass) → [critic_review] (if enabled)
                    ↓ (approve) → END
                    ↓ (reject) → diagnose_failure
    ↓ (fail) → diagnose_failure (ReAct loop with tools)
                    ↓
               update_learning_log
                    ↓
               [review_repair] (if autonomy_level != full_auto, uses interrupt())
                    ↓
               increment_iteration
                    ↓
               generate_solution (or fan_out_repairs if parallel_strategies enabled)
```

## Common Gotchas

- `context_builder.py` has a bug (Fix 1): truncation computes trimmed vars but returns original. Fix first.
- `prompt_loader.py` needs a new `get_raw_template()` function (Fix 1): the router needs the raw template before substitution to pass to `build_context()`.
- `test_graph_mock.py` accepts `"running"` as terminal status — that's wrong, fix in Fix 3.
- The `reference_tests` field in `BenchmarkTask` exists but is never populated — Fix 2.
- **Existing tests you must NOT duplicate**: `test_schema_validator.py` already has tests for markdown fences, prose prefix, invalid JSON, nested dict coercion. `test_sandbox.py` already has `test_timeout_enforcement` for infinite loops.
- LangGraph's `interrupt()` requires a checkpointer. Always set one when using HITL.
- `Send()` for parallel fan-out requires `Annotated[list, operator.add]` reducer on the collection field.
- Schema validation retries 3 times then crashes — use `call_with_fallback()` for non-critical roles.
- **Token counts vary by provider**: Ollama returns real counts, HuggingFace returns -1, Mock returns word-count approximations. Metrics code must handle -1 gracefully.
- **langchain-core is already in requirements.txt** — do not add it again for Fix 6 tools.
- **demo_runner.py uses `asyncio.run()` sync wrapper** — this is incompatible with `interrupt()`. For HITL (Fix 12), switch to Gradio's native async support or keep the sync wrapper only for `full_auto` mode.
- **Every new node needs event updates**: Add event type constants to `agent/events.py`, formatters to `framework/streaming.py`, and handlers to `demo/demo_runner.py`'s `DemoUIState.apply_event()`.
