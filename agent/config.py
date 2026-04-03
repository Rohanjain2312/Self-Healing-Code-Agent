"""
Declarative configuration for agent behavior.

AgentConfig centralizes all feature flags in one place so the graph topology,
node behavior, and UI can all adapt from a single source of truth.

Autonomy levels:
  full_auto      — no human interrupts (default, matches existing behavior)
  review_repairs — pause before each repair for human approval via interrupt()
  review_all     — pause before initial generation AND before each repair

Design decision: all new features default to False/disabled so that
AgentConfig.development() always reproduces the original behavior without
requiring any migration of existing call sites.
"""

from dataclasses import dataclass, field


@dataclass
class AgentConfig:
    """Full configuration for a single agent run."""

    # ── Core ──────────────────────────────────────────────────────────────────
    max_iterations: int = 4

    # ── Autonomy / HITL ───────────────────────────────────────────────────────
    # "full_auto" | "review_repairs" | "review_all"
    autonomy_level: str = "full_auto"

    # ── Parallel repair strategies (Fix 14) ───────────────────────────────────
    parallel_strategies: bool = False

    # ── Critic node (Fix 15) ──────────────────────────────────────────────────
    enable_critic: bool = True
    critic_confidence_threshold: float = 0.6

    # ── Dual-oracle spec tests (Fix 5) ────────────────────────────────────────
    enable_spec_tests: bool = True

    # ── Cross-session memory via ChromaDB (Fix 10) ────────────────────────────
    enable_cross_session_memory: bool = False
    memory_persist_dir: str = ".agent_memory"

    # ── Observability ─────────────────────────────────────────────────────────
    enable_langsmith: bool = False
    langsmith_project: str = "self-healing-agent"

    # ── Checkpointing (Fix 13) ────────────────────────────────────────────────
    enable_checkpointing: bool = True
    persist_checkpoints: bool = False   # True → SqliteSaver; False → InMemorySaver

    # ── Model routing overrides ───────────────────────────────────────────────
    # Map role name → model name to override per-role model selection
    model_overrides: dict[str, str] = field(default_factory=dict)

    # ── Preset factory methods ────────────────────────────────────────────────

    @classmethod
    def development(cls) -> "AgentConfig":
        """
        Mirrors the original hardcoded graph — no new features enabled.

        Use this when you want to reproduce pre-upgrade behavior exactly,
        or to run the mock test suite without triggering any HITL/critic paths.
        """
        return cls(
            autonomy_level="full_auto",
            parallel_strategies=False,
            enable_critic=False,
            enable_spec_tests=False,
            enable_cross_session_memory=False,
            enable_checkpointing=False,
            persist_checkpoints=False,
        )

    @classmethod
    def staging(cls) -> "AgentConfig":
        """
        All agentic features on; human review before each repair; in-memory checkpoints.
        """
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
        """
        Full production mode: review all actions, persistent SQLite checkpoints.
        """
        return cls(
            autonomy_level="review_all",
            parallel_strategies=True,
            enable_critic=True,
            enable_spec_tests=True,
            enable_cross_session_memory=True,
            enable_checkpointing=True,
            persist_checkpoints=True,
        )
