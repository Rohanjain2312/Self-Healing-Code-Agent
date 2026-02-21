---
title: Self-Healing Code Agent
sdk: gradio
sdk_version: 4.19.2
app_file: app.py
pinned: false
license: mit
short_description: An autonomous agent that detects and fixes its own code errors.
---

# Self-Healing Code Agent

This project is an autonomous Machine Learning agent designed to monitor, diagnose, and repair code execution errors in real-time. 

## Project Links
- **GitHub Repository:** https://github.com/Rohanjain2312/Self-Healing-Code-Agent
- **Hugging Face Space:** https://huggingface.co/spaces/rohanjain2312/Self-Healing-Code-Agent

## Technical Stack
- **LLM:** Meta-Llama-3.2-3B-Instruct
- **Framework:** Gradio / Python
- **Environment:** Hugging Face Spaces

## Implementation Strategy
1. **Trace Analysis:** Capture traceback data from failed executions.
2. **Context Injection:** Feed error logs and source code into the LLM.
3. **Recursive Correction:** Apply suggested fixes and re-test until the execution succeeds.

## Architecture

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

## Running Locally

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

## Running Benchmarks

```bash
# Full benchmark (requires real LLM provider)
python -m evaluation.run_benchmark --provider ollama --max-iterations 4

# Single task
python -m evaluation.run_benchmark --task-ids interval_merge_001 --provider ollama
```

## Running Tests

```bash
# All tests (uses mock provider — no models needed)
LLM_PROVIDER=mock pytest

# Single test file
LLM_PROVIDER=mock pytest tests/test_sandbox.py -v
```

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
