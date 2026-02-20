"""
Benchmark runner — executes all tasks through the agent and collects metrics.

Intended to run on Colab GPU or a machine with a real LLM provider.
Results are saved to evaluation/results.json for the HF Spaces demo.

Usage:
    python -m evaluation.run_benchmark
    python -m evaluation.run_benchmark --max-iterations 4 --provider ollama
    python -m evaluation.run_benchmark --task-ids interval_merge_001,csv_normalize_001
"""

import argparse
import asyncio
import logging
import os
import sys
import time
from pathlib import Path

# Ensure project root is on path when run as __main__
sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.graph import run_agent
from agent.state import AgentState
from evaluation.benchmark_tasks import BENCHMARK_TASKS, BenchmarkTask
from evaluation.metrics import TaskResult, compute_summary, save_results
from llm.router import LLMRouter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)


def _extract_task_result(
    task: BenchmarkTask,
    final_state: AgentState,
    elapsed: float,
) -> TaskResult:
    """Convert final agent state into a TaskResult for metrics."""
    success = final_state.get("status") == "success" or final_state.get("last_execution_passed", False)
    iterations_used = final_state.get("iteration", 0)
    first_pass = success and iterations_used == 0

    # Collect failure categories seen across iterations
    failure_categories = [
        record.get("failure_category", "")
        for record in final_state.get("iteration_history", [])
        if record.get("failure_category")
    ]

    return TaskResult(
        task_id=task.task_id,
        task_description=task.description[:200],
        category=task.category,
        success=success,
        first_pass=first_pass,
        iterations_used=iterations_used + 1,  # 0-indexed → 1-indexed count
        failure_categories=failure_categories,
        final_code=final_state.get("current_code", ""),
    )


async def run_single_task(
    task: BenchmarkTask,
    router: LLMRouter,
    max_iterations: int,
) -> TaskResult:
    """Run a single benchmark task and return its result."""
    logger.info("Running task: %s (%s)", task.task_id, task.category)
    start = time.monotonic()

    try:
        final_state = await run_agent(
            task_description=task.description,
            max_iterations=max_iterations,
            router=router,
        )
        elapsed = time.monotonic() - start
        result = _extract_task_result(task, final_state, elapsed)

    except Exception as exc:
        elapsed = time.monotonic() - start
        logger.error("Task %s crashed: %s", task.task_id, exc, exc_info=True)
        result = TaskResult(
            task_id=task.task_id,
            task_description=task.description[:200],
            category=task.category,
            success=False,
            first_pass=False,
            iterations_used=0,
            error=str(exc),
        )

    status = "PASS" if result.success else "FAIL"
    logger.info(
        "Task %s: %s (iterations=%d, elapsed=%.1fs)",
        task.task_id,
        status,
        result.iterations_used,
        elapsed,
    )
    return result


async def run_benchmark(
    task_ids: list[str] | None = None,
    max_iterations: int = 4,
    provider_name: str | None = None,
    output_path: str = "evaluation/results.json",
    concurrency: int = 1,
) -> None:
    """
    Run the full benchmark and save results.

    Args:
        task_ids: If set, run only these task IDs. Otherwise run all.
        max_iterations: Max repair iterations per task.
        provider_name: Override LLM_PROVIDER env var.
        output_path: Where to write results.json.
        concurrency: Number of tasks to run in parallel (default 1 for safety).
    """
    if provider_name:
        os.environ["LLM_PROVIDER"] = provider_name

    router = LLMRouter()
    logger.info(
        "Benchmark start: provider=%s model=%s",
        router.provider.provider_name,
        router.provider.model_name,
    )

    tasks = BENCHMARK_TASKS
    if task_ids:
        task_id_set = set(task_ids)
        tasks = [t for t in tasks if t.task_id in task_id_set]

    if not tasks:
        logger.error("No matching tasks found.")
        return

    logger.info("Running %d tasks (max_iterations=%d)", len(tasks), max_iterations)

    # Run with controlled concurrency to avoid OOM on CPU machines
    semaphore = asyncio.Semaphore(concurrency)

    async def _run_with_semaphore(task: BenchmarkTask) -> TaskResult:
        async with semaphore:
            return await run_single_task(task, router, max_iterations)

    all_results = await asyncio.gather(*[_run_with_semaphore(t) for t in tasks])

    summary = compute_summary(
        list(all_results),
        provider=router.provider.provider_name,
        model=router.provider.model_name,
    )

    save_results(list(all_results), summary, output_path)

    # Print summary table
    print("\n=== Benchmark Summary ===")
    print(f"Total tasks:          {summary.total_tasks}")
    print(f"First-pass success:   {summary.first_pass_success}")
    print(f"Healed success:       {summary.healed_success}")
    print(f"Total failures:       {summary.total_failures}")
    print(f"Repair effectiveness: {summary.repair_effectiveness:.1%}")
    print(f"Avg iterations:       {summary.avg_iterations:.2f}")
    print("\nCategory success rates:")
    for cat, rate in sorted(summary.category_success_rates.items()):
        print(f"  {cat}: {rate:.1%}")
    print(f"\nResults saved to: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Self-Healing Code Agent benchmark"
    )
    parser.add_argument(
        "--task-ids",
        type=str,
        default=None,
        help="Comma-separated task IDs to run (default: all)",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=4,
        help="Maximum repair iterations per task (default: 4)",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default=None,
        help="LLM provider: ollama | huggingface | mock",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="evaluation/results.json",
        help="Output path for results JSON",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Parallel tasks (default 1 for CPU safety)",
    )
    args = parser.parse_args()

    task_ids = args.task_ids.split(",") if args.task_ids else None

    asyncio.run(
        run_benchmark(
            task_ids=task_ids,
            max_iterations=args.max_iterations,
            provider_name=args.provider,
            output_path=args.output,
            concurrency=args.concurrency,
        )
    )


if __name__ == "__main__":
    main()
