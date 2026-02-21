"""
Gradio application for Hugging Face Spaces.

Two tabs:
  Tab 1 — Run Live Agent
    - Task input (textarea + example buttons)
    - Execution timeline (live-streamed text)
    - Code snapshot (live-updated code block)
    - Learning log (live-updated lesson list)

  Tab 2 — Performance
    - Precomputed benchmark results loaded from evaluation/results.json
    - Bar chart: healed vs first-pass success per category
    - Iteration distribution histogram
    - Summary metrics table

Design principles:
  - Streaming updates via Gradio generators (no page reload)
  - No chain-of-thought or internal prompts exposed
  - Tab 2 loads precomputed data — no live inference
  - Respects HF Spaces free tier limits
"""

import logging
import os
import sys
from pathlib import Path

# Ensure project root is importable when running from demo/
_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import gradio as gr

from demo.demo_runner import EXAMPLE_TASKS, run_demo_sync
from evaluation.metrics import load_results

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Performance tab: chart builders
# ---------------------------------------------------------------------------

def _build_performance_components():
    """
    Build the performance tab content from precomputed results.json.
    Returns (summary_text, category_chart_data, iteration_chart_data).
    """
    results = load_results(_PROJECT_ROOT / "evaluation" / "results.json")
    if not results:
        return (
            "No benchmark results found. Run `python -m evaluation.run_benchmark` first.",
            None,
            None,
        )

    summary = results.get("summary", {})
    tasks = results.get("tasks", [])

    # Summary text
    total = summary.get("total_tasks", 0)
    first_pass = summary.get("first_pass_success", 0)
    healed = summary.get("healed_success", 0)
    failures = summary.get("total_failures", 0)
    repair_eff = summary.get("repair_effectiveness", 0)
    avg_iter = summary.get("avg_iterations", 0)
    provider = summary.get("provider", "unknown")
    model = summary.get("model", "unknown")

    summary_text = (
        f"**Provider:** {provider} / {model}\n\n"
        f"| Metric | Value |\n"
        f"|--------|-------|\n"
        f"| Total Tasks | {total} |\n"
        f"| First-Pass Success | {first_pass} ({first_pass/total:.0%} of total) |\n"
        f"| Healed Success | {healed} ({healed/total:.0%} of total) |\n"
        f"| Unresolved Failures | {failures} |\n"
        f"| Repair Effectiveness | {repair_eff:.0%} |\n"
        f"| Avg Iterations | {avg_iter:.2f} |\n"
    )

    # Category success rates
    cat_rates = summary.get("category_success_rates", {})
    cat_data = {
        "Category": list(cat_rates.keys()),
        "Success Rate": [round(v * 100, 1) for v in cat_rates.values()],
    }

    # Iteration distribution
    iter_counts = {}
    for task in tasks:
        n = task.get("iterations_used", 1)
        iter_counts[n] = iter_counts.get(n, 0) + 1
    iter_data = {
        "Iterations": list(iter_counts.keys()),
        "Tasks": list(iter_counts.values()),
    }

    return summary_text, cat_data, iter_data


# ---------------------------------------------------------------------------
# Gradio UI construction
# ---------------------------------------------------------------------------

def build_app() -> gr.Blocks:
    summary_text, cat_data, iter_data = _build_performance_components()

    css = """
    .timeline-box { font-family: monospace; font-size: 0.85rem; }
    .code-box { font-family: monospace; }
    .lessons-box { font-size: 0.9rem; }
    """

    with gr.Blocks(title="Self-Healing Code Agent") as app:
        gr.Markdown(
            """
# Self-Healing Code Agent

An autonomous agent that generates Python code, adversarially tests it,
diagnoses failures, and repairs solutions through structured iteration.

**Architecture:** Generator → QA Adversarial → Sandbox Execution → Debugger → Repair Loop
            """
        )

        # ---- Tab 1: Run Live Agent ----
        with gr.Tab("Run Live Agent"):
            with gr.Row():
                with gr.Column(scale=2):
                    task_input = gr.Textbox(
                        label="Task Description",
                        placeholder="Describe the Python function you want the agent to implement...",
                        lines=6,
                        elem_classes=["task-input"],
                    )
                    with gr.Row():
                        run_btn = gr.Button("Run Agent", variant="primary")
                        clear_btn = gr.Button("Clear", variant="secondary")

                    gr.Markdown("**Example Tasks:**")
                    for i, ex in enumerate(EXAMPLE_TASKS[:3]):
                        gr.Button(
                            f"Example {i + 1}: {ex[:60]}...",
                            size="sm",
                        ).click(
                            fn=lambda t=ex: t,
                            outputs=task_input,
                        )

                with gr.Column(scale=1):
                    learning_log = gr.Textbox(
                        label="Learning Log",
                        lines=10,
                        interactive=False,
                        value="No lessons recorded yet.",
                        elem_classes=["lessons-box"],
                    )

            with gr.Row():
                with gr.Column():
                    timeline = gr.Textbox(
                        label="Execution Timeline",
                        lines=15,
                        interactive=False,
                        value="Enter a task and click Run Agent.",
                        elem_classes=["timeline-box"],
                    )

            with gr.Row():
                with gr.Column():
                    code_output = gr.Code(
                        label="Code Snapshot (latest generated)",
                        language="python",
                        interactive=False,
                        value="# Code will appear here during agent execution.",
                    )

            def _clear():
                return (
                    "",
                    "Enter a task and click Run Agent.",
                    "# Code will appear here during agent execution.",
                    "No lessons recorded yet.",
                )

            clear_btn.click(
                fn=_clear,
                outputs=[task_input, timeline, code_output, learning_log],
            )

            def _run_streaming(task: str):
                """Generator function for Gradio streaming."""
                if not task or not task.strip():
                    yield (
                        "Please enter a task description.",
                        "# No task provided.",
                        "No lessons recorded yet.",
                    )
                    return

                for timeline_text, code_text, lessons_text in run_demo_sync(task):
                    yield timeline_text, code_text, lessons_text

            run_btn.click(
                fn=_run_streaming,
                inputs=[task_input],
                outputs=[timeline, code_output, learning_log],
            )

        # ---- Tab 2: Performance ----
        with gr.Tab("Performance"):
            gr.Markdown(
                """
## Benchmark Results

Results from running the agent against 8 interview + real-world hybrid tasks.
Benchmark runs are executed on GPU (Colab) — results loaded from precomputed data.
                """
            )

            if summary_text:
                gr.Markdown(summary_text)
            else:
                gr.Markdown("No precomputed results found.")

            if cat_data and cat_data.get("Category"):
                gr.BarPlot(
                    value=cat_data,
                    x="Category",
                    y="Success Rate",
                    title="Success Rate by Category (%)",
                    color="Category",
                )

            if iter_data and iter_data.get("Iterations"):
                gr.BarPlot(
                    value=iter_data,
                    x="Iterations",
                    y="Tasks",
                    title="Iteration Distribution (how many iterations tasks required)",
                    color="Iterations",
                )

            gr.Markdown(
                """
---
**Metric Definitions:**

- **First-Pass Success**: Solved correctly on the first generation attempt (no repair needed).
- **Healed Success**: Solved after at least one repair iteration.
- **Repair Effectiveness**: `healed / (total - first_pass)` — fraction of initially-failing tasks resolved.
- **Avg Iterations**: Mean iterations across all tasks (1 = first-pass success).
            """
            )

    return app


def main() -> None:
    app = build_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
        share=False,
        theme=gr.themes.Soft(),
    )


if __name__ == "__main__":
    main()
