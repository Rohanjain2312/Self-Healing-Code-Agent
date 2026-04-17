"""
Declarative configuration for the Self-Healing Code Agent.

Every feature flag, autonomy switch, and routing override the agent supports
lives in this single :class:`AgentConfig` dataclass. The graph builder
(``agent.graph.build_graph``), the demo runner (``demo.demo_runner``), and the
Gradio UI all read from this object — there is no other source of truth.

WHY ONE BIG DATACLASS
---------------------
Centralizing flags here means:
  - ``build_graph(config)`` can decide topology (e.g. whether to insert the
    HITL ``review_repair`` node) from a single argument.
  - Tests can construct a fully-customized run with ``AgentConfig(...)`` and
    pass it to the graph without monkey-patching env vars.
  - The Gradio UI can serialize/echo the active config at the top of the
    timeline so users see exactly what's enabled.

AUTONOMY LEVELS — IMPORTANT
---------------------------
The default is ``"review_repairs"`` because the local CLI / dev experience is
the most common way this agent is run, and HITL is the more interesting
behavior to demo there. **Web deployments** (HF Spaces, any hosted Gradio)
should override this to ``"full_auto"`` because LangGraph's ``interrupt()``
pause + Gradio's streaming generator can desynchronize on resume.

Mapping:
  full_auto      — never interrupts; agent runs end-to-end autonomously.
  review_repairs — pauses before each repair iteration via interrupt(); user
                   sees a "Approve / Reject / Edit" panel in the UI.
  review_all     — pauses before initial generation AND before each repair.

PRODUCTION DEFAULTS
-------------------
All quality features (critic, spec tests, cross-session memory, checkpointing)
are enabled by default. Disable individually only if measuring the impact of
a specific feature in benchmarks.
"""

from dataclasses import dataclass, field


@dataclass
class AgentConfig:
    """Full configuration for a single agent run.

    Pass an instance to :func:`agent.graph.build_graph` to construct a graph
    that respects these settings. The dataclass is immutable in practice — do
    not mutate fields after construction; build a new instance instead.
    """

    # ── Core ──────────────────────────────────────────────────────────────────
    # Hard cap on the number of generate→test→diagnose→repair cycles. The graph
    # ENDs immediately when this is reached, even if the latest solution still
    # fails (status becomes "max_iterations_reached").
    max_iterations: int = 4

    # ── Autonomy / HITL ───────────────────────────────────────────────────────
    # See module docstring for the meaning of each level. Default is
    # "review_repairs" because that's the most informative behavior for local
    # development. HF Spaces / hosted Gradio deployments should override this
    # to "full_auto" in their entry point (see app.py at repo root).
    autonomy_level: str = "review_repairs"

    # ── Parallel repair strategies (Fix 14) ───────────────────────────────────
    # When True, after a failed iteration the graph fans out via LangGraph's
    # Send() to run 3 different repair strategies in parallel and tournament-
    # selects the best one. Disabled by default because it triples LLM cost
    # per repair iteration.
    parallel_strategies: bool = False

    # ── Critic node (Fix 15) ──────────────────────────────────────────────────
    # When True, every passing solution is reviewed by a "critic" LLM call
    # before END — catches correctness issues the test suite missed (e.g. a
    # solution that silently swallows exceptions and returns the wrong type).
    enable_critic: bool = True
    critic_confidence_threshold: float = 0.6  # critic verdicts below this are ignored

    # ── Dual-oracle spec tests (Fix 5) ────────────────────────────────────────
    # When True, before any code is generated we generate a "spec test" suite
    # purely from the task description (the generator never sees these tests).
    # The execute_solution node then runs BOTH spec tests AND the adversarial
    # tests — both must pass for the iteration to succeed. This catches the
    # failure mode where the generator and adversarial tester agree on a wrong
    # interpretation of the spec.
    enable_spec_tests: bool = True

    # ── Cross-session memory via ChromaDB (Fix 10) ────────────────────────────
    # When True, lessons from past runs are loaded at iteration 0 (giving the
    # agent a head start on similar tasks) and successful runs append back to
    # the store. Persists to disk in `memory_persist_dir` and optionally syncs
    # to a HuggingFace Dataset if HF_TOKEN is set.
    enable_cross_session_memory: bool = True
    memory_persist_dir: str = ".agent_memory"

    # ── Observability ─────────────────────────────────────────────────────────
    # LangSmith provides per-run traces of every LLM call with prompt/response
    # diffing. Off by default to avoid sending data externally; opt in with
    # this flag + LANGCHAIN_API_KEY env var.
    enable_langsmith: bool = False
    langsmith_project: str = "self-healing-agent"

    # ── Checkpointing (Fix 13) ────────────────────────────────────────────────
    # InMemorySaver is the default. It is REQUIRED for interrupt() to work
    # (LangGraph needs to pickle state at the pause point). HF Spaces' SQLite
    # is ephemeral so we don't bother persisting to disk by default.
    enable_checkpointing: bool = True
    persist_checkpoints: bool = False

    # ── Model routing overrides ───────────────────────────────────────────────
    # Map role name → model name. Currently only OllamaProvider honors model
    # overrides; HF and Mock ignore the model name and use whatever they were
    # constructed with. Example: {"generator": "llama3.2:3b"}.
    model_overrides: dict[str, str] = field(default_factory=dict)
