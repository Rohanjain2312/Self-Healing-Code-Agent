"""
Per-node and per-run metrics for observability and cost tracking.

Design decisions:
- NodeMetrics is collected inside each agent node and accumulated into RunMetrics.
- Token counts follow the provider convention: Ollama returns real counts,
  HuggingFace returns -1 (unknown), Mock returns word-count approximations.
  All aggregation methods treat -1 as "unknown" — they are excluded from sums
  rather than skewing totals to zero.
- estimated_cost_usd uses Llama-3 / mid-tier open-source pricing as a rough
  reference (input: $0.15/M, output: $0.60/M). This is for relative comparison
  across runs, not billing accuracy.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class NodeMetrics:
    """Latency, token, and error metrics for a single node execution."""

    node_name: str
    start_time: float = 0.0
    end_time: float = 0.0
    input_tokens: int = 0    # -1 means unknown (HuggingFace provider)
    output_tokens: int = 0   # -1 means unknown
    llm_calls: int = 0
    retries: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def latency_seconds(self) -> float:
        """Wall-clock time from node entry to exit."""
        return self.end_time - self.start_time

    @property
    def estimated_cost_usd(self) -> float:
        """
        Rough cost estimate using mid-tier open-source LLM pricing.

        Returns 0.0 if token counts are unknown (-1).
        """
        if self.input_tokens < 0 or self.output_tokens < 0:
            return 0.0
        return (self.input_tokens * 0.00015 + self.output_tokens * 0.0006) / 1000


@dataclass
class RunMetrics:
    """Aggregated metrics across all nodes in a single agent run."""

    task_id: str
    total_latency: float = 0.0
    total_input_tokens: int = 0   # sum of known counts only
    total_output_tokens: int = 0  # sum of known counts only
    total_llm_calls: int = 0
    node_metrics: list[NodeMetrics] = field(default_factory=list)
    total_retries: int = 0

    @property
    def estimated_total_cost_usd(self) -> float:
        """Sum of per-node cost estimates (excludes nodes with unknown token counts)."""
        return sum(n.estimated_cost_usd for n in self.node_metrics)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a flat dict suitable for JSON logging and results.json."""
        return {
            "task_id": self.task_id,
            "total_latency_s": round(self.total_latency, 2),
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_llm_calls": self.total_llm_calls,
            "total_retries": self.total_retries,
            "estimated_cost_usd": round(self.estimated_total_cost_usd, 6),
            "nodes": [
                {
                    "name": n.node_name,
                    "latency_s": round(n.latency_seconds, 2),
                    "tokens_in": n.input_tokens,
                    "tokens_out": n.output_tokens,
                }
                for n in self.node_metrics
            ],
        }
