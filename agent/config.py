"""
Declarative configuration for agent behavior.

AgentConfig centralizes all feature flags in one place so the graph topology,
node behavior, and UI can all adapt from a single source of truth.

Autonomy levels:
  full_auto      — no human interrupts (default; required for Gradio/HF Spaces
                   since interrupt() is incompatible with the demo runner)
  review_repairs — pause before each repair for human approval via interrupt()
  review_all     — pause before initial generation AND before each repair

All defaults are production-grade settings: critic, spec tests, cross-session
memory, and checkpointing are all enabled out of the box.
"""

from dataclasses import dataclass, field


@dataclass
class AgentConfig:
    """Full configuration for a single agent run."""

    # ── Core ──────────────────────────────────────────────────────────────────
    max_iterations: int = 4

    # ── Autonomy / HITL ───────────────────────────────────────────────────────
    # "full_auto" | "review_repairs" | "review_all"
    # Keep full_auto as default — interrupt() cannot be used in Gradio on HF Spaces.
    autonomy_level: str = "full_auto"

    # ── Parallel repair strategies (Fix 14) ───────────────────────────────────
    parallel_strategies: bool = False

    # ── Critic node (Fix 15) ──────────────────────────────────────────────────
    enable_critic: bool = True
    critic_confidence_threshold: float = 0.6

    # ── Dual-oracle spec tests (Fix 5) ────────────────────────────────────────
    enable_spec_tests: bool = True

    # ── Cross-session memory via ChromaDB (Fix 10) ────────────────────────────
    enable_cross_session_memory: bool = True
    memory_persist_dir: str = ".agent_memory"

    # ── Observability ─────────────────────────────────────────────────────────
    enable_langsmith: bool = False
    langsmith_project: str = "self-healing-agent"

    # ── Checkpointing (Fix 13) ────────────────────────────────────────────────
    enable_checkpointing: bool = True
    # SQLite on HF Spaces is ephemeral — keep in-memory by default
    persist_checkpoints: bool = False

    # ── Model routing overrides ───────────────────────────────────────────────
    # Map role name → model name to override per-role model selection
    model_overrides: dict[str, str] = field(default_factory=dict)
