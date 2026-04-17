---
title: Self-Healing Code Agent
sdk: gradio
sdk_version: 6.6.0
app_file: app.py
pinned: false
license: mit
short_description: Autonomous agent that self-heals Python code errors.
---

# Self-Healing Code Agent

[![Live Demo](https://img.shields.io/badge/🤗%20Live%20Demo-HF%20Spaces-blue)](https://huggingface.co/spaces/rohanjain2312/Self-Healing-Code-Agent)
[![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.3+-orange)](https://github.com/langchain-ai/langgraph)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## About

**Rohan Jain** — MS Machine Learning, University of Maryland

MS ML student at UMD with a background in data science and analytics, transitioning into applied LLM systems engineering. Focused on building AI systems that are reliable and observable in real execution environments — not just accurate in notebooks. This project was built to explore what it takes to make an autonomous agent self-correct in a live Python execution environment, moving well beyond one-shot prompting.

| | |
|---|---|
| 🐙 GitHub | [github.com/Rohanjain2312](https://github.com/Rohanjain2312) |
| 🤗 HuggingFace | [huggingface.co/rohanjain2312](https://huggingface.co/rohanjain2312) |
| 💼 LinkedIn | [linkedin.com/in/jaroh23](https://www.linkedin.com/in/jaroh23/) |
| 📧 Email | jaroh23@umd.edu |

---

## What It Is

An autonomous agent that generates Python code, adversarially tests it with edge cases, diagnoses failures through structured root-cause analysis, and iteratively repairs the solution — all without human input. The core problem: LLMs produce incorrect code on the first attempt more often than not. This system treats that as a solvable engineering problem by wrapping the LLM in a structured, self-correcting feedback loop.

| Run it now | |
|---|---|
| 🤗 [HF Spaces — no setup required](https://huggingface.co/spaces/rohanjain2312/Self-Healing-Code-Agent) | Runs on CPU, expect 30–90s per agent step |
| 🔬 [Google Colab — GPU](colab/Self_Healing_Agent.ipynb) | T4 GPU, public `gradio.live` link via `share=True` |

---

## Demo

![Self-Healing Code Agent — live run](assets/screenshots/demo_run.png)

*Live run: the agent generated a Python solution, tested it adversarially, diagnosed a failure, and applied a targeted repair — all autonomously. Execution Timeline (bottom left) and Learning Log (right) update in real time.*

---

## How to Run It

> ⚠️ Deploying to HuggingFace Spaces? See [`docs/deployment-issues.md`](docs/deployment-issues.md) for a full log of known failure modes and fixes encountered during this build.

### Prerequisites

- Python 3.11+
- `ANTHROPIC_API_KEY` — required for all non-generator roles (QA, debugger, critic, memory summarizer)
- Ollama running locally (`ollama serve`) — optional, used only for the generator role

```bash
pip install -r requirements.txt
# Option 1: Ollama generator + Claude for smart roles (recommended locally)
ollama pull llama3.2:3b
ANTHROPIC_API_KEY=sk-ant-... python app.py

# Option 2: All roles use Claude (no Ollama needed; repair loop triggers less often)
LLM_PROVIDER=anthropic ANTHROPIC_API_KEY=sk-ant-... python app.py
```

### With HuggingFace Transformers (HF Spaces / GPU)

```bash
pip install transformers torch accelerate
# LLM_PROVIDER=huggingface is REQUIRED to route the generator role to the local 3B model.
# Without it, the generator silently falls back to Claude (repair loop rarely triggers).
LLM_PROVIDER=huggingface ANTHROPIC_API_KEY=sk-ant-... python app.py
```

### Mock mode (no models required)

```bash
LLM_PROVIDER=mock python app.py
```

### Running Benchmarks

```bash
# Full benchmark (requires real LLM provider)
python -m evaluation.run_benchmark --provider ollama --max-iterations 4

# Single task
python -m evaluation.run_benchmark --task-ids interval_merge_001 --provider ollama
```

### Running Tests

```bash
# All tests (uses mock provider — no models needed)
LLM_PROVIDER=mock pytest

# Single test file
LLM_PROVIDER=mock pytest tests/test_sandbox.py -v
```

---

## Architecture

This system has three architectural layers with meaningfully different properties:

### Layer 1: Orchestrated Pipeline (deterministic flow)

The Generator, QA, Executor, and Memory Summarizer nodes run in a fixed order. They are LLM-augmented steps, not autonomous agents — each receives a structured prompt, calls the LLM once, validates the output against a JSON schema, and passes results downstream. The LangGraph state machine handles routing.

### Layer 2: Agentic Investigation (autonomous decision-making)

The Debugger is the one genuinely agentic component. It runs a **ReAct loop**: think → use tool → observe → repeat → conclude. It can invoke three tools before issuing its final diagnosis:
- `run_snippet`: execute a Python snippet to test an edge-case hypothesis
- `inspect_function`: parse AST to verify function signatures and docstrings
- `diff_iterations`: compare code across repair attempts to track convergence

This is where actual autonomous multi-step reasoning happens — the LLM decides which tools to use, in what order, and when to stop investigating.

### Layer 3: Human Oversight (configurable autonomy)

Configurable via `AgentConfig.autonomy_level`:
- `review_repairs` (default): pause before each repair for human approval via `interrupt()`
- `full_auto`: no interrupts — fully autonomous repair loop (used by the HF Spaces deployment, set in `app.py`)
- `review_all`: pause before generation AND before each repair

> The HF Spaces entry point (`app.py` at the repo root) explicitly overrides this to `full_auto` because LangGraph's `interrupt()` pause/resume flow can desynchronize with Gradio's streaming generator on hosted deployments.

```
[generate_spec_tests]  ← once (spec-blind oracle tests, if enabled)
        ↓
generate_solution      ← generates initial code
        ↓
create_adversarial_tests  ← QA hunts for edge cases in the code
        ↓
execute_solution       ← runs BOTH spec + adversarial tests in sandbox
        ↓ (pass) → [critic_review]  ← sanity-checks correctness (if enabled)
                       ↓ (approve) → END
                       ↓ (reject) → diagnose_failure
        ↓ (fail) → diagnose_failure  ← ReAct loop with tools
                       ↓
                  update_learning_log
                       ↓
                  [review_repair]  ← HITL interrupt() (if not full_auto)
                       ↓
                  increment_iteration
                       ↓
                  generate_solution  (or fan_out_repairs if parallel_strategies)
```

---

## Agent Roles

| Role | Type | Description | Prompt |
|------|------|-------------|--------|
| Generator | Pipeline node | Writes initial code; applies targeted repairs guided by diagnosis | `generator.yaml` |
| QA Adversarial | Pipeline node | Generates hostile edge-case tests designed to break the solution | `qa_adversarial.yaml` |
| Debugger | **ReAct agent** | Root-cause analysis with tool use — runs think/act/observe loop | `debugger.yaml` |
| Memory Summarizer | Pipeline node | Compresses iteration history into ≤5 bullet lessons | `memory_summarizer.yaml` |
| Critic | Pipeline node | Sanity-checks passing solutions for correctness issues the tests missed | `critic.yaml` |

---

## Engineering Concepts Demonstrated

| Concept | Implementation |
|---------|----------------|
| **ReAct agent loop** | Debugger runs think → use tool → observe → repeat before issuing diagnosis |
| **Dual-oracle testing** | Spec-blind tests (generated before code exists) + adversarial tests (generated after) — both must pass |
| **Human-in-the-loop** | `interrupt()` pauses the graph for human review; `AgentConfig.autonomy_level` controls when |
| **Parallel repair strategies** | Fan-out via LangGraph `Send()` — 3 strategies run concurrently, tournament selection picks winner |
| **Agent self-reflection** | Critic node reviews passing solutions for correctness issues the test suite missed |
| **Time-travel debugging** | Checkpointer stores all states; `fork_from_iteration()` rewinds and replays with modified state |
| **Confidence-aware routing** | Low-confidence diagnoses route to blind retry instead of targeted repair |
| **Structured outputs + schema validation** | Every LLM call validated against a typed JSON schema; coercion + regex salvage handle malformed output |
| **Prompt engineering + versioning** | YAML prompt files per role, git-versioned, hot-reloadable — decoupled from agent code |
| **Token / context management** | Rolling memory summarizer (max 5 lessons) + token-aware truncation with re-render |
| **Provider-agnostic inference** | Unified LLM router: Ollama → HuggingFace → Mock. Per-role model overrides supported |
| **Observability** | LangSmith tracing (opt-in), per-node metrics, degraded-node tracking, event stream |

---

## Benchmark Results

> Run conditions: `llama3` via Ollama, local CPU, max 4 iterations per task. 8 tasks across 6 categories.
> Reference-validated results use held-out ground-truth test suites not seen by the agent.

| Metric | Self-Reported | Reference-Validated |
|--------|--------------|---------------------|
| Tasks evaluated | 8 | 8 |
| First-pass success | 3 / 8 (37%) | — |
| Healed after repair | 4 / 5 initially-failing (80%) | — |
| Final success rate | 7 / 8 (87%) | *run with `--validate-reference` to generate* |
| Avg iterations per task | 1.875 | — |
| Unresolved | 1 — word frequency with complex tie-breaking | — |

*Self-reported: agent's own generated tests pass. Reference-validated: held-out ground-truth assertions pass.*

| Category | Success |
|----------|---------|
| Interval merging | 100% |
| Data normalization | 100% |
| Log processing | 100% |
| Data transformation | 100% |
| Boundary conditions | 100% |
| Text processing | 50% |

---

## Limitations

| Limitation | Why It Happens | How to Overcome |
|------------|----------------|-----------------|
| **Slow inference on HF Spaces (30–90s/step)** | Free tier = CPU only, no GPU | Upgrade to HF Spaces Pro (A100) or swap to an API-hosted model via the router |
| **Schema instability on small models** | 3B models frequently truncate or mis-format JSON — JSON-encoding Python source roughly doubles character count under tight token limits | Use 8B+ model, or an API provider with native structured output support |
| **No cross-session memory** | The learning log resets on every new task | Add ChromaDB vector store — scaffolded in `agent/memory_store.py`, enable via `AgentConfig.enable_cross_session_memory` |
| **Single-file execution sandbox** | Subprocess executor runs one file — cannot handle solutions spanning multiple modules | Extend sandbox to write a temp package directory with `__init__.py` |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Agent orchestration | LangGraph 0.3+ (async state machine, `Send()` fan-out, `interrupt()` HITL, checkpointing) |
| LLM inference | Llama-3.2-3B-Instruct (HF Spaces) · Llama-3.1-8B (Colab) · Ollama (local) |
| UI & deployment | Gradio 6.6 · HuggingFace Spaces |
| Prompt management | YAML templates per agent role, git-versioned, hot-reloadable |
| Schema validation | jsonschema + custom coercion + regex salvage fallback |
| Execution sandbox | Python subprocess with rlimit resource limits and structured output markers |
| Async runtime | asyncio · AnyIO · LangGraph async nodes · async event bus (pub/sub) |
| Testing | pytest · pytest-asyncio · mock provider (no GPU required) |

---

## Development Notes

The agent architecture, workflow design, LangGraph state machine topology, YAML prompt schemas, and all key engineering decisions were designed and authored by Rohan Jain. [Claude Code](https://claude.ai/code) was used as an implementation accelerator to handle repetitive boilerplate, file scaffolding, and iterative debugging — similar to how a senior engineer uses Copilot or a junior developer for implementation tasks while retaining full design ownership.

All architectural choices, agent interaction patterns, structured output schemas, and the self-healing repair logic reflect the author's original engineering judgment.

---

## Project Structure

```
agent/          LangGraph state machine and node implementations
  nodes/        Individual agent nodes (generate, QA, execute, debug, critic, etc.)
  tools.py      Debugger tools: run_snippet, inspect_function, diff_iterations
  config.py     AgentConfig — all feature flags in one place
  metrics.py    Per-node and per-run metrics tracking
framework/      Async event bus and streaming infrastructure
llm/            Unified LLM router, providers, prompt loading
  providers/    Ollama, HuggingFace, Mock implementations
sandbox/        Safe Python subprocess execution environment
prompts/        YAML prompt templates per agent role
evaluation/     Benchmark harness, metrics, and precomputed results
demo/           Gradio application for HuggingFace Spaces
tests/          Pytest test suite (mock provider, no GPU needed)
```

## Environment Variables

| Variable | Required? | Default | Description |
|----------|-----------|---------|-------------|
| `ANTHROPIC_API_KEY` | **Required** | — | Anthropic API key for Claude. All non-generator roles (QA, debugger, critic, memory summarizer) use Claude. Without this key the agent falls back to Mock. |
| `LLM_PROVIDER` | Required for HF Spaces | auto | `huggingface` \| `ollama` \| `anthropic` \| `mock`. Set to `huggingface` on HF Spaces to route the generator to the local 3B model. |
| `OLLAMA_GENERATOR_MODEL` | Optional | `llama3.2:3b` | Ollama model used for the generator role when Ollama is reachable. |
| `OLLAMA_BASE_URL` | Optional | `http://localhost:11434` | Ollama server URL |
| `HF_MODEL` | Optional | `meta-llama/Llama-3.2-3B-Instruct` | HuggingFace model ID for the generator role |
| `HF_TOKEN` | Optional | — | HuggingFace token for cross-session lesson persistence to a HF Dataset |
| `USE_4BIT` | Optional | unset | Enable 4-bit quantization for the HF generator (requires bitsandbytes) |
| `LANGCHAIN_TRACING_V2` | Optional | unset | Enable LangSmith tracing |
| `LANGCHAIN_API_KEY` | Optional | unset | LangSmith API key |
| `LANGCHAIN_PROJECT` | Optional | `self-healing-agent` | LangSmith project name |
