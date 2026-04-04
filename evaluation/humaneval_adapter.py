"""
HumanEval benchmark adapter (Fix 16).

Converts HumanEval problems (164 coding tasks from OpenAI) into agent task
descriptions, and extracts agent completions back to the format expected by
the official HumanEval evaluator.

HumanEval problem format (JSONL):
  {
    "task_id": "HumanEval/0",
    "prompt": "from typing import List\\ndef has_close_elements(numbers: List[float], ...",
    "canonical_solution": "    for idx, elem in enumerate(numbers)...",
    "test": "def check(candidate):\\n    assert ...",
    "entry_point": "has_close_elements"
  }

Usage:
    tasks = load_humaneval_tasks("evaluation/HumanEval.jsonl")
    for task in tasks:
        agent_task = humaneval_to_agent_task(task)
        # ... run agent ...
        completion = extract_completion(agent_code, task["entry_point"])

Download HumanEval.jsonl from:
    https://github.com/openai/human-eval/raw/master/data/HumanEval.jsonl.gz
"""

import json
import re
from pathlib import Path


def load_humaneval_tasks(path: str | Path) -> list[dict]:
    """
    Load HumanEval problems from a JSONL file.

    Args:
        path: Path to HumanEval.jsonl (or .jsonl.gz after decompression).

    Returns:
        List of problem dicts, each with keys: task_id, prompt, canonical_solution,
        test, entry_point.
    """
    path = Path(path)
    tasks = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                tasks.append(json.loads(line))
    return tasks


def humaneval_to_agent_task(he_task: dict) -> str:
    """
    Convert a HumanEval problem dict into an agent task description string.

    Embeds the function signature, docstring, and type hints from the prompt
    so the agent understands exactly what to implement.

    Args:
        he_task: A HumanEval problem dict.

    Returns:
        Natural language task description string for the agent.
    """
    prompt = he_task.get("prompt", "")
    entry_point = he_task.get("entry_point", "")
    task_id = he_task.get("task_id", "")

    return (
        f"Complete the following Python function implementation.\n\n"
        f"Task ID: {task_id}\n"
        f"Function to implement: `{entry_point}`\n\n"
        f"```python\n{prompt}\n```\n\n"
        f"Implement the function body. The implementation must pass all edge cases. "
        f"Return a complete, runnable Python function with no placeholder or stub code."
    )


def extract_completion(agent_code: str, entry_point: str) -> str:
    """
    Extract the function body matching entry_point from agent-generated code.

    The HumanEval evaluator concatenates the original prompt with the completion
    and executes the result. This function extracts only the portion of the
    agent's code starting from the function definition, so it slots in cleanly
    after the prompt.

    Args:
        agent_code: Full code string returned by the agent.
        entry_point: The function name expected by HumanEval (e.g. "has_close_elements").

    Returns:
        The extracted completion string. Falls back to the full agent_code if
        the function definition cannot be found.
    """
    if not agent_code.strip():
        return f"    pass\n"

    # Match the function definition at the start of a line
    pattern = rf"(?:^|\n)(def\s+{re.escape(entry_point)}\s*\()"
    match = re.search(pattern, agent_code)
    if match:
        # Return from the start of the def line onward
        start = match.start(1) if match.group(0).startswith("\n") else match.start(0)
        # Adjust if the match starts with a newline
        if agent_code[match.start(0)] == "\n":
            start = match.start(0) + 1
        else:
            start = match.start(0)
        return agent_code[start:]

    # Fallback: return all code — evaluator will try to execute it
    return agent_code
