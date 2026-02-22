# Deployment Issues & Fixes — Self-Healing Code Agent

> Reusable knowledge base built from deploying this project to HuggingFace Spaces.
> Each issue is a standalone entry covering the exact error, root cause, fix, and prevention.
> Designed to be consumed by humans and LLM agents on future builds.

---

## Issue Index

| # | Error / Symptom | Where | Status |
|---|-----------------|--------|--------|
| 1 | `sync_to_hf.yml` never triggered | GitHub Actions | Fixed |
| 2 | `short_description` over 60 chars | HF Spaces README frontmatter | Fixed |
| 3 | `ModuleNotFoundError: No module named 'audioop'` | HF Spaces runtime (Python 3.13) | Fixed |
| 4 | `DeprecationWarning: Use dtype instead of torch_dtype` | HuggingFace Transformers | Fixed |
| 5 | `RuntimeError: cannot schedule new futures after interpreter shutdown` | app.py startup (daemon thread) | Fixed |
| 6 | `DeprecationWarning` / `RuntimeError` on `asyncio.get_event_loop()` | Python 3.13 + async code | Fixed |
| 7 | `TypeError: Blocks.launch() got an unexpected keyword argument 'theme'` | Gradio 5→6 migration | Fixed |
| 8 | "Connection errored out" — Space running but unreachable | HF Spaces reverse proxy | Fixed |
| 9 | `TypeError: 'NoneType' object does not support the asynchronous context manager protocol` | Gradio 5 queue internals | Fixed |
| 10 | `safe_get_lock()` returns `None` (root cause of #9) | Gradio 5.x + Python 3.13 | Fixed → upgrade to Gradio 6 |
| 11 | `JSON parse failed: Expecting ':' delimiter: line 1 column 455` | `llm/schema_validator.py` | Fixed |

---

## Issue 1: GitHub Actions Workflow in Wrong Folder

**Error / Symptom:**
```
Workflow never triggered on push. HF Space never received new code.
```

**Where it occurred:** GitHub Actions / repository root

**Root cause:** `sync_to_hf.yml` was placed in the repository root. GitHub Actions only reads workflow files from `.github/workflows/` — files anywhere else are silently ignored.

**Why it happens in LLM agent systems:** AI code generation tools and scaffolding assistants frequently emit files to the project root. Without explicit path awareness, the workflow file ends up in the wrong location with no error feedback.

**Fix applied:**
```bash
# Move file to correct location
mkdir -p .github/workflows
git mv sync_to_hf.yml .github/workflows/sync_to_hf.yml
git commit -m "fix: move sync workflow to correct .github/workflows/ path"
git push
```

**Prevention for future builds:** Always scaffold `.github/workflows/` at project initialization. Add a sanity check: `ls .github/workflows/` before assuming CI is wired up.

---

## Issue 2: HF Spaces `short_description` Over 60 Characters

**Error / Symptom:**
```
ERROR: Validation error in README frontmatter: short_description exceeds maximum length (60 characters)
```

**Where it occurred:** HuggingFace Spaces — push rejected at metadata validation step

**Root cause:** The HF Spaces `short_description` frontmatter field has a hard 60-character limit enforced server-side. Longer values cause the entire push to be rejected.

**Why it happens in LLM agent systems:** Agents tend to write descriptive short descriptions. Without a character count check, it's easy to write a 70–90 char description that reads well but fails validation silently until push time.

**Fix applied:**
```yaml
# Before (73 chars — rejected)
short_description: Autonomous agent that generates, tests, diagnoses, and self-heals Python code.

# After (52 chars — accepted)
short_description: Autonomous agent that self-heals Python code errors.
```

**Prevention for future builds:** Keep `short_description` to one short clause. Count characters before committing: `echo -n "your description" | wc -c`.

---

## Issue 3: `audioop` Missing — Gradio 4 on Python 3.13

**Error / Symptom:**
```
ModuleNotFoundError: No module named 'audioop'
  File "...pydub/utils.py", line 11, in <module>
    import audioop
```

**Where it occurred:** HuggingFace Spaces runtime on startup (Python 3.13)

**Root cause:** Python 3.13 removed the `audioop` C extension from the standard library. Gradio 4.x depended on `pydub`, which imports `audioop` unconditionally at module load time. The entire Gradio import chain fails.

**Why it happens in LLM agent systems:** HF Spaces periodically updates its default Python version. A pinned old Gradio version that worked at project start can break silently when the runtime is updated. LLM-generated `requirements.txt` files often pin whatever was current at generation time.

**Fix applied:**
```
# requirements.txt
# Before
gradio==4.19.2

# After
gradio==6.6.0
```

Also update `README.md` frontmatter:
```yaml
sdk_version: 6.6.0
```

**Prevention for future builds:** Always pin `gradio>=5.0.0` for Python 3.13+ environments. Check the HF Spaces default Python version before choosing a Gradio version. Prefer Gradio 6+ for new projects.

---

## Issue 4: `torch_dtype` Deprecation in HuggingFace Transformers

**Error / Symptom:**
```
FutureWarning: `torch_dtype` is deprecated. Use `dtype` instead.
```

**Where it occurred:** `llm/providers/hf_provider.py` — `pipeline()` call in `_load_pipeline()`

**Root cause:** The `torch_dtype` keyword argument in `transformers.pipeline()` was renamed to `dtype` in Transformers 4.40+. The old argument still worked but emitted a deprecation warning, which in strict environments would raise.

**Why it happens in LLM agent systems:** Agent provider code is often written against older Transformers docs or examples. The rename is silent until warnings are surfaced in logs.

**Fix applied:**
```python
# Before
kwargs = {
    "model": self._model_id,
    "device_map": self._device_map,
    "torch_dtype": torch.float16,
}

# After
kwargs = {
    "model": self._model_id,
    "device_map": self._device_map,
    "dtype": torch.float16,
}
```

**Prevention for future builds:** Use `dtype=` in all new `pipeline()` calls. When upgrading `transformers`, search for `torch_dtype` in your codebase and rename.

---

## Issue 5: Daemon Thread Killed Mid-Download

**Error / Symptom:**
```
RuntimeError: cannot schedule new futures after interpreter shutdown
  File ".../concurrent/futures/thread.py", line 169, in submit
```

**Where it occurred:** `app.py` startup — model pre-warm code running in a daemon thread

**Root cause:** A daemon thread was spawned to pre-download model weights in the background while Gradio started. Python's interpreter shutdown sequence kills all daemon threads when the main thread exits its setup phase. The `ThreadPoolExecutor` inside the HuggingFace provider's `_ensure_loaded()` call was mid-execution when it was killed.

**Why it happens in LLM agent systems:** Model pre-warming is a standard pattern for reducing first-request latency. Daemon threads feel like the right tool (background, non-blocking) but are wrong here — the download *must* complete before the server accepts requests.

**Fix applied:**
```python
# Before — daemon thread (wrong)
import threading
t = threading.Thread(target=lambda: asyncio.run(_prewarm()), daemon=True)
t.start()

# After — synchronous blocking call before launch() (correct)
def _prewarm() -> None:
    try:
        from llm.router import LLMRouter
        router = LLMRouter()
        if hasattr(router.provider, "_ensure_loaded"):
            asyncio.run(router.provider._ensure_loaded())
    except Exception as exc:
        logger.warning("Pre-warm failed (non-fatal): %s", exc)

_prewarm()  # blocks until complete
demo = build_app()
demo.launch(...)
```

**Prevention for future builds:** Never pre-warm in daemon threads. Pre-warm must finish before `launch()` is called. Wrap in `try/except` so a pre-warm failure doesn't crash the server.

---

## Issue 6: `asyncio.get_event_loop()` Deprecated on Python 3.13

**Error / Symptom:**
```
DeprecationWarning: There is no current event loop
RuntimeError: no running event loop
```

**Where it occurred:** `llm/providers/hf_provider.py`, `framework/event_bus.py` — any call site using `asyncio.get_event_loop()` outside an async context

**Root cause:** Python 3.10 deprecated `asyncio.get_event_loop()` when called with no running loop (it used to silently create one). Python 3.12+ promotes this to a `DeprecationWarning`. Python 3.13 can raise `RuntimeError` in some call sites. Code written against Python 3.9 patterns breaks.

**Why it happens in LLM agent systems:** LLM agent frameworks make heavy use of `asyncio`. Boilerplate code from tutorials, Stack Overflow, and AI assistants frequently uses the older `get_event_loop()` pattern. The breakage is Python-version dependent, making it easy to miss in testing.

**Fix applied:**
```python
# Before (hf_provider.py — inside async function)
loop = asyncio.get_event_loop()
result = await loop.run_in_executor(None, self._load_pipeline)

# After
loop = asyncio.get_running_loop()  # safe inside async context
result = await loop.run_in_executor(None, self._load_pipeline)

# Before (event_bus.py — sync wrapper)
loop = asyncio.get_event_loop()
loop.run_until_complete(bus.emit(event))

# After
def emit_sync(bus, event):
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(bus.emit(event))
    except RuntimeError:
        asyncio.run(bus.emit(event))
```

**Prevention for future builds:**
- Inside `async def`: use `asyncio.get_running_loop()` — always safe, always correct
- At sync entry points: use `asyncio.run(coroutine)` — creates and manages its own loop
- Never use `asyncio.get_event_loop()` in new code

---

## Issue 7: `theme=` Removed from `gr.Blocks()` / `launch()` in Gradio 5→6

**Error / Symptom:**
```
# Gradio 5
UserWarning: The `theme` and `css` parameters have been moved to `launch()`.

# Gradio 6
TypeError: Blocks.launch() got an unexpected keyword argument 'theme'
```

**Where it occurred:** `demo/app.py` and `app.py` — `gr.Blocks()` and `demo.launch()` calls

**Root cause:** Gradio 5 moved `theme` and `css` from `gr.Blocks()` to `launch()`. Gradio 6 then removed `theme` from `launch()` entirely — it belongs back in `gr.Blocks()`. The API moved twice across two major versions.

**Why it happens in LLM agent systems:** Gradio is a fast-moving library. AI-generated UI code targets whatever Gradio version was in the training data. The `theme=` argument breakage is particularly deceptive because the error message from Gradio 5 points you in the wrong direction.

**Fix applied (for Gradio 6):**
```python
# Before
with gr.Blocks(theme=gr.themes.Soft(), title="...") as app:
    ...
demo.launch(theme=..., css=...)

# After
css = "..."
with gr.Blocks(title="...", css=css) as app:  # css goes here in Gradio 6
    ...
demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
# theme= removed entirely (use default or pass gr.themes.X() to gr.Blocks())
```

**Prevention for future builds:** Check the Gradio changelog for any major version bump. Always run `python -W error app.py` locally to surface deprecation warnings before deploying.

---

## Issue 8: Server Bound to 127.0.0.1 — HF Spaces Proxy Can't Reach It

**Error / Symptom:**
```
# In browser
Connection errored out.

# In Space logs
INFO:     127.0.0.1:XXXXX - "GET / HTTP/1.1" 200 OK
# (Space running and healthy, but browser can't connect)
```

**Where it occurred:** HuggingFace Spaces — the reverse proxy routing external traffic to the container

**Root cause:** `demo.launch()` defaults to binding on `127.0.0.1` (loopback only). HF Spaces routes external HTTP traffic through a reverse proxy that connects to the container on all network interfaces. With loopback-only binding, the proxy's connection attempt is refused — the server is running but unreachable from outside the container.

**Why it happens in LLM agent systems:** Default Gradio `launch()` is designed for local development. Moving to a containerized or proxied environment requires explicit network binding. This is a common "works locally, fails in production" failure.

**Fix applied:**
```python
# Before
demo.launch()

# After
demo.launch(
    server_name="0.0.0.0",  # bind on all interfaces
    server_port=int(os.environ.get("PORT", 7860)),  # respect HF Spaces PORT env var
)
```

**Prevention for future builds:** Always use `server_name="0.0.0.0"` in any containerized, proxied, or cloud deployment. Always read `PORT` from the environment. Make this a template in your app.py boilerplate.

---

## Issue 9: `pending_message_lock` is `None` — Gradio 5 Queue Crash

**Error / Symptom:**
```
TypeError: 'NoneType' object does not support the asynchronous context manager protocol
  File ".../gradio/queueing.py", line XXX, in push
    async with self.pending_message_lock:
```

**Where it occurred:** Gradio 5 queue internals — triggered on every user interaction (button click, form submit)

**Root cause:** See Issue 10 for the underlying source bug. The symptom: `self.pending_message_lock` inside the Gradio queue is `None` instead of an `asyncio.Lock`. Every interaction that touches the queue crashes immediately with this `TypeError`.

**Why it happens in LLM agent systems:** Streaming agent output through Gradio's queue is the standard pattern for real-time UI updates. The queue being silently broken is catastrophic — every interaction fails, but the Space shows as healthy.

**Fix applied at the time:** Adding `demo.queue()` before `demo.launch()` partially initialized the lock. However this was not a complete fix — the root cause (Issue 10) required upgrading to Gradio 6.

**Permanent fix:** Upgrade to `gradio==6.6.0` (see Issue 10).

**Prevention for future builds:** Use Gradio 6+ with Python 3.13. Always call `demo.queue()` before `demo.launch()` for streaming apps. Test the queue explicitly by sending a request and verifying the streaming response arrives.

---

## Issue 10: `safe_get_lock()` Returns `None` — Root Cause of Issue 9

**Error / Symptom:** (Same as Issue 9 — this is the underlying source bug)

**Where it occurred:** `gradio/utils.py` → `safe_get_lock()` — called at Gradio import time in Python 3.13

**Root cause:**
```python
# Gradio 5.x source (simplified)
def safe_get_lock():
    try:
        loop = asyncio.get_event_loop()  # raises DeprecationWarning / returns None on Python 3.13
        return asyncio.Lock()
    except RuntimeError:
        return None  # <-- this None becomes pending_message_lock
```

On Python 3.13, `asyncio.get_event_loop()` with no running loop returns `None` or raises. The `except RuntimeError` catches it and returns `None`. This `None` is stored as `pending_message_lock` for the lifetime of the process.

**Fix in Gradio 6.0:**
```python
# Gradio 6.x source (simplified)
def safe_get_lock():
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
    return asyncio.Lock()
```

**Fix applied:** Upgrade `requirements.txt`:
```
gradio==6.6.0
```

Update `README.md` frontmatter:
```yaml
sdk_version: 6.6.0
```

**Prevention for future builds:** Pin `gradio>=6.0.0` for any deployment on Python 3.13. This is a hard compatibility boundary — Gradio 4/5 + Python 3.13 = broken queue with no obvious error at startup.

---

## Issue 11: JSON Parse Failure from Truncated LLM Output

**Error / Symptom:**
```
[ERROR] Agent encountered an error: JSON parse failed: Expecting ':' delimiter: line 1 column 455
```

**Where it occurred:** `llm/schema_validator.py` — during structured output parsing after every LLM inference call

**Root cause:** The generator agent is prompted to return a JSON object containing a `"code"` field with the full Python source. With `max_new_tokens=1024`, the 3B model runs out of token budget mid-output — the JSON is cut off partway through the `"code"` string value. The `json.loads()` call fails on the truncated text.

The token budget math: A 200-line Python solution is ~3,000 characters of source. JSON-encoding it (escaping newlines as `\n`, quotes as `\"`, etc.) roughly doubles the character count to ~6,000 characters. At ~4 chars/token that's ~1,500 tokens — already over the 1,024 limit before any JSON wrapper overhead.

**Why it happens in LLM agent systems:** Structured output (JSON-wrapped code or tool calls) is fundamentally longer than raw text generation. Token limits tuned for conversational text are systematically too small for code-generation agents. This failure is silent until it hits production — the model generates what it can, the JSON truncates, the parser fails.

**Fix applied:**

1. Raise `max_new_tokens` across all agent nodes:
```python
# agent/nodes/generate_solution.py
max_new_tokens=2048  # was 1024

# agent/nodes/create_adversarial_tests.py
max_new_tokens=1024  # was 768

# agent/nodes/diagnose_failure.py
max_new_tokens=768   # was 512
```

2. Add salvage fallback in `llm/schema_validator.py`:
```python
def _salvage_code_field(text: str) -> dict[str, Any] | None:
    """Extract code field from truncated JSON using regex as last resort."""
    pattern = re.compile(r'"code"\s*:\s*"((?:[^"\\]|\\.)*)', re.DOTALL)
    match = pattern.search(text)
    if match:
        raw_code = match.group(1)
        try:
            code = raw_code.encode("utf-8").decode("unicode_escape")
        except Exception:
            code = raw_code
        if code.strip():
            return {"code": code}
    return None

def parse_and_validate(text: str, schema: dict) -> dict:
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        salvaged = _salvage_code_field(text)
        if salvaged:
            return salvaged
        raise StructuredOutputError(f"JSON parse failed: {e}")
    ...
```

**Prevention for future builds:**
- Set `max_new_tokens` to at least 2× the expected output length for any structured-output agent
- Always implement a salvage/fallback parser for JSON-wrapped outputs — truncation is inevitable on small models with constrained token budgets
- Log token usage per call to detect budget exhaustion early
- Consider using models with native function calling (which handle structured output within the token budget more efficiently)

---

*Last updated: 2025 — Self-Healing Code Agent build log*
