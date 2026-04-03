## Roadblocks Claude Code Will Hit During Implementation

**2. LangGraph API version mismatches**

The plan references `interrupt()`, `Command`, `Send()`, `InMemorySaver`, `SqliteSaver`, `ToolNode`, `tools_condition`. These APIs have changed across LangGraph versions. Your `requirements.txt` says `langgraph>=0.2.0` but some of these features (especially `interrupt()` and `Command`) were introduced in later versions. `SqliteSaver` may require a separate package (`langgraph-checkpoint-sqlite`).

*Mitigation:* Before starting, pin your LangGraph version. Run `pip install langgraph --upgrade` and check what version you get. If it's below 0.3.x, several HITL features won't work. You may need `langgraph>=0.3.0` and `langgraph-checkpoint-sqlite` as separate dependencies.

**3. Ollama doesn't support native tool calling**

Fix 6 (ReAct debugger with tools) assumes tool calling. Ollama with `llama3` doesn't support OpenAI-style function calling. The plan mentions this and suggests a JSON-based tool-use format as a workaround, but Claude Code might still try to use `bind_tools()` and `ToolNode` from LangGraph, which requires a model that returns `tool_calls` in its response. With Ollama, the LLM will just return plain text.

*Mitigation:* The debugger's ReAct loop will likely need to be implemented as a manual parse loop (check if the LLM output contains a tool-use JSON structure, execute the tool, feed the result back) rather than using LangGraph's built-in `ToolNode` + `tools_condition` pattern. Be prepared to guide Claude Code toward this approach if it gets stuck.

**4. Circular import risks**

The plan adds cross-references between modules that don't currently depend on each other. For example, `agent/nodes/generate_solution.py` will need access to `agent/memory_store.py` (Fix 10), `agent/graph.py` will need `agent/config.py` (Fix 12), and `sandbox/python_executor.py` is imported by `agent/tools.py` (Fix 6) which is imported by `agent/nodes/diagnose_failure.py`. Claude Code tends to create circular imports when wiring things up.

*Mitigation:* If you see `ImportError: cannot import name X from partially initialized module`, the fix is usually lazy imports inside functions rather than top-level imports.

**5. The HITL + Gradio integration is the hardest part**

Fix 12 requires restructuring `demo_runner.py` from a sync wrapper around `asyncio.run()` to a native async Gradio flow that can pause at interrupts and resume. This is architecturally complex and Claude Code may produce code that either deadlocks (waiting for interrupt input that never comes), or crashes because the Gradio event loop and LangGraph's checkpointer interact badly.

*Mitigation:* Implement Fix 12 with `autonomy_level="full_auto"` working first (no actual interrupts firing). Then test the interrupt path separately. If the Gradio integration proves too complex, a fallback is to support HITL only in the CLI/notebook runner, not in the Gradio UI.

**6. ChromaDB version/platform issues (Fix 10)**

ChromaDB has had breaking API changes between versions. The `Settings` import path, `persist_directory` handling, and client initialization have all changed. On some platforms (especially HF Spaces free tier), ChromaDB's SQLite dependency can fail.

*Mitigation:* Pin `chromadb>=0.4.0,<0.6.0` in requirements.txt. Make the memory store completely optional — if ChromaDB import fails, the agent should work without it.

**7. HumanEval security sandbox requirement (Fix 16)**

OpenAI's `human-eval` package deliberately comments out the code execution function for safety. You have to manually uncomment it. Claude Code might not know this and will wonder why evaluation returns zero results.

*Mitigation:* After Claude Code creates the adapter, you'll need to manually enable execution in the `human-eval` package's `execution.py` file per the instructions in their README.

**8. Existing tests may break during Phase 2–3 refactoring**

When `agent/graph.py` is refactored to accept `AgentConfig` (Fix 12), the existing `test_graph_mock.py` tests call `run_agent()` without a config. If Claude Code changes the function signature without providing defaults, all existing tests break.

*Mitigation:* The CLAUDE.md says "never break the default development flow" but Claude Code may not follow this perfectly. After each phase, run `LLM_PROVIDER=mock pytest -v` and fix any regressions before moving on.