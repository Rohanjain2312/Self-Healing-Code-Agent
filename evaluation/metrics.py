"""
Evaluation metrics for the Self-Healing Code Agent benchmark.

Metrics computed per task and aggregated across the benchmark set:
  - first_pass_success: solved on iteration 0 without any repair
  - healed_success: solved after at least one repair iteration
  - repair_effectiveness: fraction of initially-failing tasks that were healed
  - avg_iterations: mean iterations across all tasks (including successes)
  - category_success_rates: per-failure-category healing rates

Results are saved as evaluation/results.json for the demo Performance tab.
"""

import json
import statistics
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any


@dataclass
class TaskResult:
    """Metrics for a single benchmark task run."""
    task_id: str
    task_description: str
    category: str  # benchmark category label (e.g. "interval_merging")
    success: bool
    first_pass: bool  # True if passed on iteration 0
    iterations_used: int
    failure_categories: list[str] = field(default_factory=list)
    final_code: str = ""
    error: str = ""  # non-empty if agent crashed (not task failure)


@dataclass
class BenchmarkSummary:
    """Aggregated metrics across all benchmark tasks."""
    total_tasks: int
    first_pass_success: int
    healed_success: int
    total_failures: int
    repair_effectiveness: float  # healed / (total - first_pass)
    avg_iterations: float
    category_success_rates: dict[str, float]
    provider: str
    model: str
    run_timestamp: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compute_summary(
    results: list[TaskResult],
    provider: str = "",
    model: str = "",
) -> BenchmarkSummary:
    """Compute aggregated metrics from a list of per-task results."""
    import datetime

    total = len(results)
    first_pass = sum(1 for r in results if r.success and r.first_pass)
    healed = sum(1 for r in results if r.success and not r.first_pass)
    failures = total - first_pass - healed

    # repair_effectiveness = of tasks that failed first pass, what fraction healed
    initially_failing = total - first_pass
    repair_effectiveness = (healed / initially_failing) if initially_failing > 0 else 1.0

    iterations = [r.iterations_used for r in results]
    avg_iter = statistics.mean(iterations) if iterations else 0.0

    # Per-category success rates
    category_results: dict[str, list[bool]] = defaultdict(list)
    for r in results:
        category_results[r.category].append(r.success)

    category_rates = {
        cat: sum(successes) / len(successes)
        for cat, successes in category_results.items()
    }

    return BenchmarkSummary(
        total_tasks=total,
        first_pass_success=first_pass,
        healed_success=healed,
        total_failures=failures,
        repair_effectiveness=round(repair_effectiveness, 3),
        avg_iterations=round(avg_iter, 2),
        category_success_rates=category_rates,
        provider=provider,
        model=model,
        run_timestamp=datetime.datetime.utcnow().isoformat() + "Z",
    )


def save_results(
    results: list[TaskResult],
    summary: BenchmarkSummary,
    output_path: str | Path = "evaluation/results.json",
) -> None:
    """Persist results and summary to JSON for demo Performance tab."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "summary": summary.to_dict(),
        "tasks": [asdict(r) for r in results],
    }

    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)


def load_results(
    path: str | Path = "evaluation/results.json",
) -> dict[str, Any]:
    """Load precomputed results for the demo Performance tab."""
    path = Path(path)
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)
