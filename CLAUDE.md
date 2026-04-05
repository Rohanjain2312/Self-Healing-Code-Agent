# CLAUDE.md — Self-Healing Code Agent

## Project Overview

LangGraph-based autonomous agent that generates Python code, adversarially tests it,
diagnoses failures via a ReAct loop with tools, and iteratively repairs solutions.
Supports human-in-the-loop review via LangGraph `interrupt()` with configurable autonomy
levels. Generator role uses a weak local model (Ollama/HuggingFace) to intentionally fail
on edge cases; all other roles (QA, debugger, critic, memory_summarizer) use Claude via
the Anthropic API.

## Provider Configuration

- **Generator role:** `HuggingFaceProvider` (HF Spaces) or `OllamaProvider` (local MacBook)
- **All other roles:** `AnthropicProvider` (Claude Haiku 4.5)
- Detection is automatic via `build_router_with_generator_override()` in `llm/router.py`
- **Required env var:** `ANTHROPIC_API_KEY`
- **Optional env vars:**
  - `OLLAMA_GENERATOR_MODEL` — Ollama model for generator (default: `llama3.2:3b`)
  - `LLM_PROVIDER` — force provider: `mock | huggingface | ollama | anthropic`
  - `OLLAMA_BASE_URL` — Ollama host (default: `http://localhost:11434`)

## Repo Structure

```
agent/
  config.py              AgentConfig dataclass (all feature flags)
  graph.py               build_graph(), run_agent(), stream_agent()
  state.py               AgentState TypedDict
  metrics.py             NodeMetrics, RunMetrics
  memory_store.py        ChromaDB LessonStore
  tools.py               Debugger tools: run_snippet, inspect_function, diff_iterations
  events.py              Event constructors
  hf_memory_sync.py      HuggingFace Datasets backend for cross-session lessons
  nodes/
    generate_solution.py
    create_adversarial_tests.py
    execute_solution.py
    diagnose_failure.py
    update_learning_log.py
    generate_spec_tests.py
    review_repair.py
    critic_review.py
    parallel_generate.py
llm/
  router.py              LLMRouter + build_router_with_generator_override()
  base.py                BaseLLMProvider abstract class
  context_builder.py     Token-aware context builder
  schema_validator.py    JSON parse + validation + coercion
  prompt_loader.py       YAML prompt loader
  providers/
    ollama_provider.py
    hf_provider.py
    anthropic_provider.py
    mock_provider.py
sandbox/
  python_executor.py     Subprocess sandbox for code execution
prompts/                 YAML templates per agent role
  generator.yaml
  qa_adversarial.yaml
  debugger.yaml
  memory_summarizer.yaml
  critic.yaml
evaluation/
  benchmark_tasks.py     Task definitions with reference tests
  metrics.py             TaskResult, compute_summary
  run_benchmark.py       Benchmark runner
  humaneval_adapter.py
  run_humaneval.py
demo/
  app.py                 Gradio UI (HITL review panel, async streaming)
  demo_runner.py         run_demo_async(), resume_demo_async(), AgentSession
framework/
  streaming.py           format_event_for_timeline(), build_timeline_text()
tests/                   All tests use LLM_PROVIDER=mock
docs/
  upgrade/               Historical upgrade artifacts (upgrade prompt, verification, plan)
```

## Key Conventions

- **Async everywhere:** All node functions are `async`. Use `await` for LLM calls and sandbox execution.
- **State is TypedDict:** `AgentState` in `agent/state.py`. Nodes return partial dicts merged into state.
- **Prompts in YAML:** Each role has a YAML file in `prompts/` with system prompt, schema, and templates.
- **Router pattern:** Nodes never call providers directly. Always go through `LLMRouter.call()` or `call_with_fallback()`.
- **Mock provider for tests:** `LLM_PROVIDER=mock` uses `MockProvider` with hardcoded JSON responses. No GPU needed.
- **Dependency injection:** Router is bound to nodes via `functools.partial` before adding to graph.
- **Events for streaming:** Nodes append event dicts to `state["events"]` for Gradio UI streaming.
  `events` uses `Annotated[list, operator.add]` reducer — nodes return the FULL cumulative list;
  `stream_agent()` uses `total_seen` to extract only new events.

## Graph Topology (current)

```
[generate_spec_tests] ──────┐   (parallel fan-in at create_adversarial_tests)
                            ↓
generate_solution ──────────→ create_adversarial_tests
                                        ↓
                              execute_solution (runs BOTH spec + adversarial tests)
                                        ↓ (pass) → critic_review
                                                        ↓ (approve) → END
                                                        ↓ (reject) → diagnose_failure
                                        ↓ (fail) → diagnose_failure (ReAct loop with tools)
                                                        ↓
                                               update_learning_log
                                                        ↓
                                               [review_repair]  ← HITL interrupt()
                                                  (when autonomy_level != full_auto)
                                                        ↓
                                               increment_iteration
                                                        ↓ (max iter) → END
                                                        ↓
                                               generate_solution (repair loop)
```

## AgentConfig Defaults (production-grade)

```python
AgentConfig()  # defaults:
  autonomy_level = "review_repairs"   # HITL pause before each repair
  enable_critic = True
  enable_spec_tests = True
  enable_cross_session_memory = True
  enable_checkpointing = True         # InMemorySaver by default
  persist_checkpoints = False
  parallel_strategies = False
  max_iterations = 4
```

For full-auto (no HITL, required for sync-only paths):
```python
config = AgentConfig(autonomy_level="full_auto")
```

## How to Run

```bash
# Tests (always use mock provider)
LLM_PROVIDER=mock pytest -v

# Run locally with Ollama generator + Claude for other roles
ANTHROPIC_API_KEY=sk-ant-... python app.py

# Run with HuggingFace generator + Claude for other roles (HF Spaces)
LLM_PROVIDER=huggingface ANTHROPIC_API_KEY=sk-ant-... python app.py

# Run all roles with Claude (no local model)
LLM_PROVIDER=anthropic ANTHROPIC_API_KEY=sk-ant-... python app.py

# Benchmark
python -m evaluation.run_benchmark --provider ollama --max-iterations 4
```

## State Fields (AgentState)

```
task_description: str       # Set once at entry
max_iterations: int         # Set once at entry
current_code: str           # Updated each iteration
current_test_code: str      # Adversarial tests (per iteration)
spec_test_code: str         # Spec-blind tests (generated once)
last_execution_passed: bool
last_failure_summary: str
root_cause: str
failure_category: str
repair_strategy: str
diagnosis_confidence: float # 0.0–1.0; below 0.3 triggers blind retry
learning_log: list[str]     # Rolling max 5 lessons
iteration: int
iteration_history: list[IterationRecord]
status: str                 # running | success | failed | max_iterations_reached
events: Annotated[list[dict], operator.add]  # For UI streaming
degraded_nodes: list[str]   # Nodes that used fallback
parallel_repairs: Annotated[list, operator.add]  # Fan-out results
strategy_name: str          # Which parallel strategy this branch runs
```

## Common Gotchas

- **Checkpointer required for interrupt():** `build_graph()` attaches `InMemorySaver` when
  `autonomy_level != "full_auto"` or `enable_checkpointing = True`.
- **`asyncio.run()` incompatible with interrupt():** `run_demo_sync()` raises `NotImplementedError`
  for non-full_auto configs. Use `run_demo_async()` + `resume_demo_async()` for HITL.
- **HuggingFace provider returns -1 for token counts:** Metrics code must handle -1 as "unknown".
- **Events list is cumulative:** Each node reads `state["events"]` and appends, returning the full list.
  The `operator.add` reducer means the stored state gets duplicates; `stream_agent()` and
  `run_demo_async()` use `total_seen` to extract only new events. Pass `events_seen` through
  `AgentSession` when resuming after HITL pause.
- **Upgrade artifacts** are in `docs/upgrade/` — not relevant to normal development.
