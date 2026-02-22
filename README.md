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
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2+-orange)](https://github.com/langchain-ai/langgraph)
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
- Ollama running locally (`ollama serve`) with `llama3` pulled

```bash
pip install -r requirements.txt
ollama pull llama3
python app.py
```

### With HuggingFace Transformers (GPU)

```bash
pip install transformers torch accelerate
LLM_PROVIDER=huggingface python app.py
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

## How It Works + Engineering Concepts

### Pipeline

```
User Task
    ↓
Generator Agent (LLM)
    ↓
QA Adversarial Agent → generates edge-case tests
    ↓
Execution Sandbox → safe subprocess with timeout
    ↓
Failure Detection
    ↓
Debugger Agent → structured root cause analysis
    ↓
Rolling Memory Summarizer → max 5 bullet lessons
    ↓
Repair Loop (back to Generator)
    ↓
Success / Max Iterations
```

### 4 Specialized LLM Agents

| Agent | Role | Prompt File |
|-------|------|-------------|
| Generator | Writes initial code; applies targeted repairs guided by diagnosis | `prompts/generator.yaml` |
| QA Adversarial | Generates hostile edge-case tests designed to break the solution | `prompts/qa_adversarial.yaml` |
| Debugger | Structured root-cause analysis — outputs `root_cause`, `failure_category`, `repair_strategy` | `prompts/debugger.yaml` |
| Memory Summarizer | Compresses iteration history into ≤5 bullet lessons to prevent context overflow | `prompts/memory_summarizer.yaml` |

### Engineering Concepts Demonstrated

| Concept | Implementation |
|---------|----------------|
| **Structured outputs + schema validation** | Every LLM call validated against a typed JSON schema; `_coerce_parsed()` handles type mismatches; regex salvage recovers truncated JSON as a last resort |
| **Prompt engineering + versioning** | 4 YAML prompt files, versioned in git, hot-reloadable — prompts are fully decoupled from agent code |
| **Token / context management** | Rolling memory summarizer hard-caps at 5 lessons; `max_new_tokens` tuned independently per agent role to balance latency vs. output completeness |
| **Multi-turn agent reasoning** | LangGraph state machine with conditional routing — task context, generated code, diagnosis, and lessons are all carried forward across up to 4 repair iterations |
| **Failure detection + self-healing** | Subprocess sandbox captures stdout/stderr with structured output markers → failure summary injected into debugger → targeted repair, not blind rewrite |
| **Provider-agnostic inference** | Unified LLM router: Ollama (local dev) → HuggingFace Transformers (GPU) → Mock (CI) — agent nodes require zero changes when switching providers |

---

## Benchmark Results

> Run conditions: `llama3` via Ollama, local CPU, max 4 iterations per task. 8 tasks across 6 categories.

| Metric | Result |
|--------|--------|
| Tasks evaluated | 8 |
| First-pass success | 3 / 8 (37%) |
| Healed after repair | 4 / 5 initially-failing (80%) |
| Repair effectiveness | 80% |
| Avg iterations per task | 1.875 |
| Unresolved | 1 — word frequency with complex tie-breaking logic |

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
| **Schema instability on small models** | 3B models frequently truncate or mis-format JSON — JSON-encoding Python source roughly doubles character count under tight token limits | Use 8B+ model, or an API provider with native function calling / structured output support |
| **No cross-session memory** | The learning log resets on every new task — lessons from prior runs are not persisted | Add a vector store (ChromaDB, FAISS) to persist and retrieve lessons across sessions |
| **Single-file execution sandbox** | Subprocess executor runs one file — cannot handle solutions spanning multiple modules | Extend sandbox to write a temp package directory with `__init__.py` and support relative imports |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Agent orchestration | LangGraph 0.2+ (async state machine, conditional routing) |
| LLM inference | Llama-3.2-3B-Instruct (HF Spaces) · Llama-3.1-8B (Colab) · Ollama (local) |
| UI & deployment | Gradio 6.6 · HuggingFace Spaces |
| Prompt management | YAML templates per agent role, git-versioned, hot-reloadable |
| Schema validation | jsonschema + custom coercion + regex salvage fallback |
| Execution sandbox | Python subprocess with wall-clock timeout and structured output markers |
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

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | auto | `ollama` \| `huggingface` \| `mock` |
| `OLLAMA_MODEL` | `llama3` | Ollama model name |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `HF_MODEL` | `meta-llama/Llama-3.2-3B-Instruct` | HuggingFace model ID |
| `USE_4BIT` | unset | Enable 4-bit quantization |
