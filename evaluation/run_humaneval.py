"""
HumanEval benchmark runner (Fix 16).

Runs the self-healing agent against HumanEval problems and writes completions
in the format expected by the official HumanEval evaluator (evaluate_functional_correctness).

Quick start:
    # 1. Download HumanEval dataset
    wget https://github.com/openai/human-eval/raw/master/data/HumanEval.jsonl.gz
    gunzip HumanEval.jsonl.gz -c > evaluation/HumanEval.jsonl

    # 2. Run the benchmark
    python -m evaluation.run_humaneval \\
        --tasks-path evaluation/HumanEval.jsonl \\
        --output evaluation/humaneval_samples.jsonl \\
        --max-tasks 20 \\
        --provider ollama

    # 3. Evaluate with the official evaluator (requires: pip install human-eval)
    evaluate_functional_correctness evaluation/humaneval_samples.jsonl

Design decisions:
  - Tasks run sequentially (not in parallel) to avoid overwhelming the local LLM.
  - enable_spec_tests and enable_critic are disabled by default for speed; the
    HumanEval test suite provides its own oracle.
  - Output is written incrementally (flush after each task) so partial results
    survive if the run is interrupted.
"""

import argparse
import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any

from evaluation.humaneval_adapter import (
    load_humaneval_tasks,
    humaneval_to_agent_task,
    extract_completion,
)
from agent.graph import run_agent
from agent.config import AgentConfig

logger = logging.getLogger(__name__)


async def _run_single_task(
    he_task: dict,
    config: AgentConfig,
) -> dict[str, Any]:
    """
    Run one HumanEval problem through the agent.

    Returns a dict with task_id, completion, and status for logging.
    Falls back to a stub completion on any agent error.
    """
    task_description = humaneval_to_agent_task(he_task)
    entry_point = he_task.get("entry_point", "solution")
    task_id = he_task.get("task_id", "")

    try:
        final_state = await run_agent(
            task_description=task_description,
            max_iterations=config.max_iterations,
            config=config,
        )
        agent_code = final_state.get("current_code", "")
        completion = extract_completion(agent_code, entry_point)
        status = final_state.get("status", "unknown")
    except Exception as exc:
        logger.warning("Task %s raised exception: %s", task_id, exc)
        completion = f"    pass  # agent error: {exc}\n"
        status = "error"

    return {
        "task_id": task_id,
        "completion": completion,
        "status": status,
    }


async def run_humaneval_benchmark(
    tasks_path: str | Path,
    output_path: str | Path,
    max_tasks: int | None = None,
    max_iterations: int = 3,
    provider: str = "ollama",
) -> dict[str, Any]:
    """
    Run the agent against all (or a subset of) HumanEval problems.

    Writes completions in JSONL format: {"task_id": ..., "completion": ...}
    This format is consumed directly by evaluate_functional_correctness.

    Args:
        tasks_path: Path to HumanEval.jsonl
        output_path: Where to write the samples JSONL
        max_tasks: Limit number of tasks (None = all 164)
        max_iterations: Max repair iterations per task
        provider: LLM provider name ("ollama", "huggingface", "mock")

    Returns:
        Summary dict with total, completed, errors.
    """
    os.environ.setdefault("LLM_PROVIDER", provider)

    tasks = load_humaneval_tasks(tasks_path)
    if max_tasks is not None:
        tasks = tasks[:max_tasks]

    config = AgentConfig(
        max_iterations=max_iterations,
        enable_spec_tests=False,     # HumanEval provides its own test oracle
        enable_critic=False,         # Skip to reduce inference time
        enable_checkpointing=False,
        parallel_strategies=False,
        enable_cross_session_memory=False,
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    stats: dict[str, Any] = {"total": len(tasks), "completed": 0, "errors": 0}

    with open(output_path, "w") as out_f:
        for i, he_task in enumerate(tasks):
            task_id = he_task.get("task_id", f"task_{i}")
            logger.info(
                "HumanEval task %d/%d: %s", i + 1, len(tasks), task_id
            )

            result = await _run_single_task(he_task, config)

            if result["status"] == "error":
                stats["errors"] += 1
            else:
                stats["completed"] += 1

            # Write in the format expected by evaluate_functional_correctness
            out_f.write(json.dumps({
                "task_id": result["task_id"],
                "completion": result["completion"],
            }) + "\n")
            out_f.flush()

    logger.info(
        "Done: %d/%d tasks, %d errors. Samples written to: %s",
        stats["completed"], stats["total"], stats["errors"], output_path,
    )
    return stats


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Run the self-healing agent against the HumanEval benchmark."
    )
    parser.add_argument(
        "--tasks-path",
        default="evaluation/HumanEval.jsonl",
        help="Path to HumanEval.jsonl (download from github.com/openai/human-eval)",
    )
    parser.add_argument(
        "--output",
        default="evaluation/humaneval_samples.jsonl",
        help="Output JSONL path for completions (for evaluate_functional_correctness)",
    )
    parser.add_argument(
        "--max-tasks",
        type=int,
        default=None,
        help="Run only the first N tasks (default: all 164)",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=3,
        help="Max repair iterations per task (default: 3)",
    )
    parser.add_argument(
        "--provider",
        default="ollama",
        choices=["ollama", "huggingface", "mock"],
        help="LLM provider to use",
    )
    args = parser.parse_args()

    stats = asyncio.run(run_humaneval_benchmark(
        tasks_path=args.tasks_path,
        output_path=args.output,
        max_tasks=args.max_tasks,
        max_iterations=args.max_iterations,
        provider=args.provider,
    ))

    print("\nHumanEval Benchmark Complete")
    print(f"  Total tasks:  {stats['total']}")
    print(f"  Completed:    {stats['completed']}")
    print(f"  Errors:       {stats['errors']}")
    print("\nTo compute Pass@1:")
    print(f"  evaluate_functional_correctness {args.output}")


if __name__ == "__main__":
    main()
