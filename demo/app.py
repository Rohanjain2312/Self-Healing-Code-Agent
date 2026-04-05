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
  - Compatible with Gradio 5.x (Python 3.13 safe)
"""

import logging
import os
import sys
from pathlib import Path

# Ensure project root is importable when running from demo/
_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import gradio as gr  # noqa: E402

from agent.config import AgentConfig  # noqa: E402
from demo.demo_runner import (  # noqa: E402
    EXAMPLE_TASKS,
    AgentSession,
    run_demo_async,
    resume_demo_async,
)
from evaluation.metrics import load_results  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Performance tab: chart builders (matplotlib-based for Gradio 5 compat)
# ---------------------------------------------------------------------------

def _make_category_chart(cat_data: dict):
    """Build a matplotlib bar chart for category success rates."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        categories = cat_data.get("Category", [])
        rates = cat_data.get("Success Rate", [])
        if not categories:
            return None

        fig, ax = plt.subplots(figsize=(8, 4))
        bars = ax.bar(categories, rates, color="#4C9BE8")
        ax.set_ylim(0, 100)
        ax.set_ylabel("Success Rate (%)")
        ax.set_title("Success Rate by Category")
        ax.bar_label(bars, fmt="%.0f%%", padding=3)
        plt.xticks(rotation=20, ha="right")
        plt.tight_layout()
        return fig
    except Exception:
        return None


def _make_iteration_chart(iter_data: dict):
    """Build a matplotlib bar chart for iteration distribution."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        iterations = iter_data.get("Iterations", [])
        tasks = iter_data.get("Tasks", [])
        if not iterations:
            return None

        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.bar([str(i) for i in iterations], tasks, color="#58B68A")
        ax.set_xlabel("Iterations")
        ax.set_ylabel("Number of Tasks")
        ax.set_title("Iteration Distribution")
        ax.bar_label(bars, padding=3)
        plt.tight_layout()
        return fig
    except Exception:
        return None


def _build_performance_components():
    """
    Build the performance tab content from precomputed results.json.
    Returns (summary_text, cat_fig, iter_fig).
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

    return summary_text, _make_category_chart(cat_data), _make_iteration_chart(iter_data)


# ---------------------------------------------------------------------------
# Graph Topology — single production SVG (parallel spec+solution entry)
# ---------------------------------------------------------------------------

_SVG_DEVELOPMENT = """
<div style="background:white;padding:20px;border-radius:8px;text-align:center;overflow:auto;">
<svg width="100%" viewBox="0 0 680 580" xmlns="http://www.w3.org/2000/svg">
  <defs><marker id="ah" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M2 1L8 5L2 9" fill="none" stroke="#888" stroke-width="1.5" stroke-linecap="round"/></marker></defs>
  <text x="340" y="28" text-anchor="middle" font-family="system-ui,sans-serif" font-size="15" font-weight="600" fill="#333">Development preset — core pipeline</text>
  <text x="340" y="48" text-anchor="middle" font-family="system-ui,sans-serif" font-size="12" fill="#888">AgentConfig.development() — no critic, no spec tests, no HITL</text>

  <rect x="230" y="72" width="220" height="44" rx="8" fill="#185FA5" stroke="#0C447C" stroke-width="0.5"/>
  <text x="340" y="98" text-anchor="middle" font-family="system-ui,sans-serif" font-size="14" font-weight="500" fill="white">generate_solution</text>
  <line x1="340" y1="116" x2="340" y2="140" stroke="#888" stroke-width="1.5" marker-end="url(#ah)"/>

  <rect x="210" y="140" width="260" height="44" rx="8" fill="#185FA5" stroke="#0C447C" stroke-width="0.5"/>
  <text x="340" y="166" text-anchor="middle" font-family="system-ui,sans-serif" font-size="14" font-weight="500" fill="white">create_adversarial_tests</text>
  <line x1="340" y1="184" x2="340" y2="208" stroke="#888" stroke-width="1.5" marker-end="url(#ah)"/>

  <rect x="230" y="208" width="220" height="44" rx="8" fill="#185FA5" stroke="#0C447C" stroke-width="0.5"/>
  <text x="340" y="234" text-anchor="middle" font-family="system-ui,sans-serif" font-size="14" font-weight="500" fill="white">execute_solution</text>
  <line x1="450" y1="230" x2="556" y2="230" stroke="#888" stroke-width="1.5" marker-end="url(#ah)"/>
  <text x="503" y="222" font-family="system-ui,sans-serif" font-size="12" fill="#888">pass</text>
  <rect x="562" y="210" width="80" height="40" rx="20" fill="#0F6E56" stroke="#085041" stroke-width="0.5"/>
  <text x="602" y="234" text-anchor="middle" font-family="system-ui,sans-serif" font-size="14" font-weight="500" fill="white">END</text>

  <line x1="340" y1="252" x2="340" y2="288" stroke="#888" stroke-width="1.5" marker-end="url(#ah)"/>
  <text x="356" y="274" font-family="system-ui,sans-serif" font-size="12" fill="#888">fail</text>

  <rect x="230" y="288" width="220" height="56" rx="8" fill="#993C1D" stroke="#712B13" stroke-width="0.5"/>
  <text x="340" y="312" text-anchor="middle" font-family="system-ui,sans-serif" font-size="14" font-weight="500" fill="white">diagnose_failure</text>
  <text x="340" y="332" text-anchor="middle" font-family="system-ui,sans-serif" font-size="12" fill="#F0997B">ReAct loop with tools</text>
  <line x1="340" y1="344" x2="340" y2="368" stroke="#888" stroke-width="1.5" marker-end="url(#ah)"/>

  <rect x="230" y="368" width="220" height="44" rx="8" fill="#185FA5" stroke="#0C447C" stroke-width="0.5"/>
  <text x="340" y="394" text-anchor="middle" font-family="system-ui,sans-serif" font-size="14" font-weight="500" fill="white">update_learning_log</text>
  <line x1="340" y1="412" x2="340" y2="436" stroke="#888" stroke-width="1.5" marker-end="url(#ah)"/>

  <rect x="230" y="436" width="220" height="44" rx="8" fill="#5F5E5A" stroke="#444441" stroke-width="0.5"/>
  <text x="340" y="462" text-anchor="middle" font-family="system-ui,sans-serif" font-size="14" font-weight="500" fill="white">increment_iteration</text>
  <line x1="450" y1="458" x2="556" y2="458" stroke="#888" stroke-width="1.5" marker-end="url(#ah)"/>
  <text x="498" y="450" font-family="system-ui,sans-serif" font-size="12" fill="#888">max iter</text>
  <rect x="562" y="438" width="80" height="40" rx="20" fill="#A32D2D" stroke="#791F1F" stroke-width="0.5"/>
  <text x="602" y="462" text-anchor="middle" font-family="system-ui,sans-serif" font-size="14" font-weight="500" fill="white">END</text>

  <path d="M230 458 L140 458 L140 94 L228 94" fill="none" stroke="#888" stroke-width="1.5" marker-end="url(#ah)"/>
  <text x="120" y="280" text-anchor="middle" font-family="system-ui,sans-serif" font-size="12" fill="#888" transform="rotate(-90 120 280)">repair loop</text>
</svg>
</div>
"""

_SVG_STAGING = """
<div style="background:white;padding:20px;border-radius:8px;text-align:center;overflow:auto;">
<svg width="100%" viewBox="0 0 720 900" xmlns="http://www.w3.org/2000/svg">
  <defs><marker id="ah" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M2 1L8 5L2 9" fill="none" stroke="#888" stroke-width="1.5" stroke-linecap="round"/></marker></defs>
  <text x="360" y="28" text-anchor="middle" font-family="system-ui,sans-serif" font-size="15" font-weight="600" fill="#333">Staging preset — all agent features enabled</text>
  <text x="360" y="48" text-anchor="middle" font-family="system-ui,sans-serif" font-size="12" fill="#888">AgentConfig.staging() — critic + spec tests + HITL + in-memory checkpoints</text>

  <!-- generate_spec_tests -->
  <rect x="230" y="68" width="260" height="56" rx="8" fill="#0F6E56" stroke="#085041" stroke-width="0.5"/>
  <text x="360" y="93" text-anchor="middle" font-family="system-ui,sans-serif" font-size="14" font-weight="500" fill="white">generate_spec_tests</text>
  <text x="360" y="113" text-anchor="middle" font-family="system-ui,sans-serif" font-size="12" fill="#7ECFB8">Blind oracle from spec only</text>
  <line x1="360" y1="124" x2="360" y2="148" stroke="#888" stroke-width="1.5" marker-end="url(#ah)"/>

  <!-- generate_solution -->
  <rect x="240" y="148" width="240" height="44" rx="8" fill="#185FA5" stroke="#0C447C" stroke-width="0.5"/>
  <text x="360" y="174" text-anchor="middle" font-family="system-ui,sans-serif" font-size="14" font-weight="500" fill="white">generate_solution</text>
  <line x1="360" y1="192" x2="360" y2="216" stroke="#888" stroke-width="1.5" marker-end="url(#ah)"/>

  <!-- create_adversarial_tests -->
  <rect x="220" y="216" width="280" height="44" rx="8" fill="#185FA5" stroke="#0C447C" stroke-width="0.5"/>
  <text x="360" y="242" text-anchor="middle" font-family="system-ui,sans-serif" font-size="14" font-weight="500" fill="white">create_adversarial_tests</text>
  <line x1="360" y1="260" x2="360" y2="284" stroke="#888" stroke-width="1.5" marker-end="url(#ah)"/>

  <!-- execute_solution -->
  <rect x="240" y="284" width="240" height="56" rx="8" fill="#185FA5" stroke="#0C447C" stroke-width="0.5"/>
  <text x="360" y="309" text-anchor="middle" font-family="system-ui,sans-serif" font-size="14" font-weight="500" fill="white">execute_solution</text>
  <text x="360" y="329" text-anchor="middle" font-family="system-ui,sans-serif" font-size="12" fill="#8BB8E8">Runs BOTH spec + adversarial</text>

  <!-- pass → critic_review -->
  <line x1="480" y1="312" x2="556" y2="312" stroke="#888" stroke-width="1.5" marker-end="url(#ah)"/>
  <text x="518" y="304" font-family="system-ui,sans-serif" font-size="12" fill="#888">pass</text>

  <!-- critic_review -->
  <rect x="556" y="290" width="140" height="56" rx="8" fill="#534AB7" stroke="#3A3388" stroke-width="0.5"/>
  <text x="626" y="315" text-anchor="middle" font-family="system-ui,sans-serif" font-size="14" font-weight="500" fill="white">critic_review</text>
  <text x="626" y="335" text-anchor="middle" font-family="system-ui,sans-serif" font-size="12" fill="#B8B4F0">Self-reflection gate</text>

  <!-- critic approve → END -->
  <line x1="626" y1="290" x2="626" y2="256" stroke="#888" stroke-width="1.5" marker-end="url(#ah)"/>
  <text x="636" y="276" font-family="system-ui,sans-serif" font-size="12" fill="#888">approve</text>
  <rect x="586" y="216" width="80" height="40" rx="20" fill="#0F6E56" stroke="#085041" stroke-width="0.5"/>
  <text x="626" y="240" text-anchor="middle" font-family="system-ui,sans-serif" font-size="14" font-weight="500" fill="white">END</text>

  <!-- critic reject → diagnose_failure (diagonal) -->
  <line x1="556" y1="340" x2="488" y2="400" stroke="#888" stroke-width="1.5" stroke-dasharray="5 3" marker-end="url(#ah)"/>
  <text x="536" y="366" font-family="system-ui,sans-serif" font-size="12" fill="#888">reject</text>

  <!-- fail → diagnose_failure -->
  <line x1="360" y1="340" x2="360" y2="380" stroke="#888" stroke-width="1.5" marker-end="url(#ah)"/>
  <text x="376" y="364" font-family="system-ui,sans-serif" font-size="12" fill="#888">fail</text>

  <!-- diagnose_failure -->
  <rect x="240" y="380" width="240" height="56" rx="8" fill="#993C1D" stroke="#712B13" stroke-width="0.5"/>
  <text x="360" y="405" text-anchor="middle" font-family="system-ui,sans-serif" font-size="14" font-weight="500" fill="white">diagnose_failure</text>
  <text x="360" y="425" text-anchor="middle" font-family="system-ui,sans-serif" font-size="12" fill="#F0997B">ReAct loop with tools</text>
  <line x1="360" y1="436" x2="360" y2="460" stroke="#888" stroke-width="1.5" marker-end="url(#ah)"/>

  <!-- update_learning_log -->
  <rect x="240" y="460" width="240" height="44" rx="8" fill="#185FA5" stroke="#0C447C" stroke-width="0.5"/>
  <text x="360" y="486" text-anchor="middle" font-family="system-ui,sans-serif" font-size="14" font-weight="500" fill="white">update_learning_log</text>
  <line x1="360" y1="504" x2="360" y2="528" stroke="#888" stroke-width="1.5" marker-end="url(#ah)"/>

  <!-- review_repair -->
  <rect x="240" y="528" width="240" height="56" rx="8" fill="#993556" stroke="#712441" stroke-width="0.5"/>
  <text x="360" y="553" text-anchor="middle" font-family="system-ui,sans-serif" font-size="14" font-weight="500" fill="white">review_repair</text>
  <text x="360" y="573" text-anchor="middle" font-family="system-ui,sans-serif" font-size="12" fill="#F0A0C0">Human-in-the-loop interrupt</text>

  <!-- review_repair abort → END -->
  <line x1="480" y1="556" x2="556" y2="556" stroke="#888" stroke-width="1.5" marker-end="url(#ah)"/>
  <text x="518" y="548" font-family="system-ui,sans-serif" font-size="12" fill="#888">abort</text>
  <rect x="556" y="536" width="80" height="40" rx="20" fill="#A32D2D" stroke="#791F1F" stroke-width="0.5"/>
  <text x="596" y="560" text-anchor="middle" font-family="system-ui,sans-serif" font-size="14" font-weight="500" fill="white">END</text>

  <!-- review_repair approve → increment_iteration -->
  <line x1="360" y1="584" x2="360" y2="608" stroke="#888" stroke-width="1.5" marker-end="url(#ah)"/>
  <text x="376" y="600" font-family="system-ui,sans-serif" font-size="12" fill="#888">approve</text>

  <!-- increment_iteration -->
  <rect x="240" y="608" width="240" height="44" rx="8" fill="#5F5E5A" stroke="#444441" stroke-width="0.5"/>
  <text x="360" y="634" text-anchor="middle" font-family="system-ui,sans-serif" font-size="14" font-weight="500" fill="white">increment_iteration</text>

  <!-- max iter → END -->
  <line x1="480" y1="630" x2="556" y2="630" stroke="#888" stroke-width="1.5" marker-end="url(#ah)"/>
  <text x="518" y="622" font-family="system-ui,sans-serif" font-size="12" fill="#888">max iter</text>
  <rect x="556" y="610" width="80" height="40" rx="20" fill="#A32D2D" stroke="#791F1F" stroke-width="0.5"/>
  <text x="596" y="634" text-anchor="middle" font-family="system-ui,sans-serif" font-size="14" font-weight="500" fill="white">END</text>

  <!-- repair loop back to generate_solution -->
  <path d="M240 630 L140 630 L140 170 L238 170" fill="none" stroke="#888" stroke-width="1.5" marker-end="url(#ah)"/>
  <text x="120" y="400" text-anchor="middle" font-family="system-ui,sans-serif" font-size="12" fill="#888" transform="rotate(-90 120 400)">repair loop</text>

  <!-- Legend -->
  <rect x="60" y="740" width="600" height="120" rx="8" fill="#F8F8F8" stroke="#DDD" stroke-width="1"/>
  <text x="360" y="762" text-anchor="middle" font-family="system-ui,sans-serif" font-size="12" font-weight="600" fill="#555">Legend</text>
  <rect x="80" y="774" width="14" height="14" rx="3" fill="#185FA5"/>
  <text x="100" y="786" font-family="system-ui,sans-serif" font-size="12" fill="#555">Core pipeline node</text>
  <rect x="80" y="800" width="14" height="14" rx="3" fill="#0F6E56"/>
  <text x="100" y="812" font-family="system-ui,sans-serif" font-size="12" fill="#555">Spec test generator / success END</text>
  <rect x="280" y="774" width="14" height="14" rx="3" fill="#993C1D"/>
  <text x="300" y="786" font-family="system-ui,sans-serif" font-size="12" fill="#555">Diagnosis (ReAct)</text>
  <rect x="280" y="800" width="14" height="14" rx="3" fill="#534AB7"/>
  <text x="300" y="812" font-family="system-ui,sans-serif" font-size="12" fill="#555">Critic review gate</text>
  <rect x="490" y="774" width="14" height="14" rx="3" fill="#993556"/>
  <text x="510" y="786" font-family="system-ui,sans-serif" font-size="12" fill="#555">HITL interrupt</text>
  <rect x="490" y="800" width="14" height="14" rx="3" fill="#A32D2D"/>
  <text x="510" y="812" font-family="system-ui,sans-serif" font-size="12" fill="#555">Failure / max-iter END</text>
</svg>
</div>
"""

_SVG_PRODUCTION = """
<div style="background:white;padding:20px;border-radius:8px;text-align:center;overflow:auto;">
<svg width="100%" viewBox="0 0 820 940" xmlns="http://www.w3.org/2000/svg">
  <defs><marker id="ah" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M2 1L8 5L2 9" fill="none" stroke="#888" stroke-width="1.5" stroke-linecap="round"/></marker>
  <marker id="ahd" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M2 1L8 5L2 9" fill="none" stroke="#BA7517" stroke-width="1.5" stroke-linecap="round"/></marker></defs>
  <text x="410" y="28" text-anchor="middle" font-family="system-ui,sans-serif" font-size="15" font-weight="600" fill="#333">Production preset — full agent platform</text>
  <text x="410" y="48" text-anchor="middle" font-family="system-ui,sans-serif" font-size="12" fill="#888">AgentConfig.production() — all features + SQLite checkpoints + HF cross-session memory</text>

  <!-- HF Dataset Sync box (right side, spans top portion) -->
  <rect x="660" y="80" width="140" height="80" rx="8" fill="#FEF3CD" stroke="#BA7517" stroke-width="1.5" stroke-dasharray="6 3"/>
  <text x="730" y="106" text-anchor="middle" font-family="system-ui,sans-serif" font-size="13" font-weight="600" fill="#BA7517">HF Dataset</text>
  <text x="730" y="124" text-anchor="middle" font-family="system-ui,sans-serif" font-size="13" font-weight="600" fill="#BA7517">Sync</text>
  <text x="730" y="148" text-anchor="middle" font-family="system-ui,sans-serif" font-size="11" fill="#BA7517">cross-session lessons</text>

  <!-- retrieve arrow: HF Dataset → generate_solution -->
  <line x1="660" y1="115" x2="510" y2="175" stroke="#BA7517" stroke-width="1.5" stroke-dasharray="5 3" marker-end="url(#ahd)"/>
  <text x="582" y="140" text-anchor="middle" font-family="system-ui,sans-serif" font-size="11" fill="#BA7517">retrieve</text>

  <!-- ── Same node topology as staging, shifted right slightly ── -->

  <!-- generate_spec_tests -->
  <rect x="280" y="68" width="260" height="56" rx="8" fill="#0F6E56" stroke="#085041" stroke-width="0.5"/>
  <text x="410" y="93" text-anchor="middle" font-family="system-ui,sans-serif" font-size="14" font-weight="500" fill="white">generate_spec_tests</text>
  <text x="410" y="113" text-anchor="middle" font-family="system-ui,sans-serif" font-size="12" fill="#7ECFB8">Blind oracle from spec only</text>
  <line x1="410" y1="124" x2="410" y2="148" stroke="#888" stroke-width="1.5" marker-end="url(#ah)"/>

  <!-- generate_solution -->
  <rect x="290" y="148" width="240" height="44" rx="8" fill="#185FA5" stroke="#0C447C" stroke-width="0.5"/>
  <text x="410" y="174" text-anchor="middle" font-family="system-ui,sans-serif" font-size="14" font-weight="500" fill="white">generate_solution</text>
  <line x1="410" y1="192" x2="410" y2="216" stroke="#888" stroke-width="1.5" marker-end="url(#ah)"/>

  <!-- create_adversarial_tests -->
  <rect x="270" y="216" width="280" height="44" rx="8" fill="#185FA5" stroke="#0C447C" stroke-width="0.5"/>
  <text x="410" y="242" text-anchor="middle" font-family="system-ui,sans-serif" font-size="14" font-weight="500" fill="white">create_adversarial_tests</text>
  <line x1="410" y1="260" x2="410" y2="284" stroke="#888" stroke-width="1.5" marker-end="url(#ah)"/>

  <!-- execute_solution -->
  <rect x="290" y="284" width="240" height="56" rx="8" fill="#185FA5" stroke="#0C447C" stroke-width="0.5"/>
  <text x="410" y="309" text-anchor="middle" font-family="system-ui,sans-serif" font-size="14" font-weight="500" fill="white">execute_solution</text>
  <text x="410" y="329" text-anchor="middle" font-family="system-ui,sans-serif" font-size="12" fill="#8BB8E8">Runs BOTH spec + adversarial</text>

  <!-- pass → critic_review -->
  <line x1="530" y1="312" x2="598" y2="312" stroke="#888" stroke-width="1.5" marker-end="url(#ah)"/>
  <text x="564" y="304" font-family="system-ui,sans-serif" font-size="12" fill="#888">pass</text>

  <!-- critic_review -->
  <rect x="598" y="290" width="140" height="56" rx="8" fill="#534AB7" stroke="#3A3388" stroke-width="0.5"/>
  <text x="668" y="315" text-anchor="middle" font-family="system-ui,sans-serif" font-size="14" font-weight="500" fill="white">critic_review</text>
  <text x="668" y="335" text-anchor="middle" font-family="system-ui,sans-serif" font-size="12" fill="#B8B4F0">Self-reflection gate</text>

  <!-- critic approve → END -->
  <line x1="668" y1="290" x2="668" y2="256" stroke="#888" stroke-width="1.5" marker-end="url(#ah)"/>
  <text x="678" y="276" font-family="system-ui,sans-serif" font-size="12" fill="#888">approve</text>
  <rect x="628" y="216" width="80" height="40" rx="20" fill="#0F6E56" stroke="#085041" stroke-width="0.5"/>
  <text x="668" y="240" text-anchor="middle" font-family="system-ui,sans-serif" font-size="14" font-weight="500" fill="white">END</text>

  <!-- critic reject → diagnose_failure -->
  <line x1="598" y1="340" x2="538" y2="400" stroke="#888" stroke-width="1.5" stroke-dasharray="5 3" marker-end="url(#ah)"/>
  <text x="578" y="366" font-family="system-ui,sans-serif" font-size="12" fill="#888">reject</text>

  <!-- fail → diagnose_failure -->
  <line x1="410" y1="340" x2="410" y2="380" stroke="#888" stroke-width="1.5" marker-end="url(#ah)"/>
  <text x="426" y="364" font-family="system-ui,sans-serif" font-size="12" fill="#888">fail</text>

  <!-- diagnose_failure -->
  <rect x="290" y="380" width="240" height="56" rx="8" fill="#993C1D" stroke="#712B13" stroke-width="0.5"/>
  <text x="410" y="405" text-anchor="middle" font-family="system-ui,sans-serif" font-size="14" font-weight="500" fill="white">diagnose_failure</text>
  <text x="410" y="425" text-anchor="middle" font-family="system-ui,sans-serif" font-size="12" fill="#F0997B">ReAct loop with tools</text>
  <line x1="410" y1="436" x2="410" y2="460" stroke="#888" stroke-width="1.5" marker-end="url(#ah)"/>

  <!-- update_learning_log -->
  <rect x="290" y="460" width="240" height="44" rx="8" fill="#185FA5" stroke="#0C447C" stroke-width="0.5"/>
  <text x="410" y="486" text-anchor="middle" font-family="system-ui,sans-serif" font-size="14" font-weight="500" fill="white">update_learning_log</text>
  <line x1="410" y1="504" x2="410" y2="528" stroke="#888" stroke-width="1.5" marker-end="url(#ah)"/>

  <!-- review_repair -->
  <rect x="290" y="528" width="240" height="56" rx="8" fill="#993556" stroke="#712441" stroke-width="0.5"/>
  <text x="410" y="553" text-anchor="middle" font-family="system-ui,sans-serif" font-size="14" font-weight="500" fill="white">review_repair</text>
  <text x="410" y="573" text-anchor="middle" font-family="system-ui,sans-serif" font-size="12" fill="#F0A0C0">Human-in-the-loop interrupt</text>

  <!-- review_repair abort → END -->
  <line x1="530" y1="556" x2="598" y2="556" stroke="#888" stroke-width="1.5" marker-end="url(#ah)"/>
  <text x="564" y="548" font-family="system-ui,sans-serif" font-size="12" fill="#888">abort</text>
  <rect x="598" y="536" width="80" height="40" rx="20" fill="#A32D2D" stroke="#791F1F" stroke-width="0.5"/>
  <text x="638" y="560" text-anchor="middle" font-family="system-ui,sans-serif" font-size="14" font-weight="500" fill="white">END</text>

  <!-- review_repair approve → increment_iteration -->
  <line x1="410" y1="584" x2="410" y2="608" stroke="#888" stroke-width="1.5" marker-end="url(#ah)"/>
  <text x="426" y="600" font-family="system-ui,sans-serif" font-size="12" fill="#888">approve</text>

  <!-- increment_iteration -->
  <rect x="290" y="608" width="240" height="44" rx="8" fill="#5F5E5A" stroke="#444441" stroke-width="0.5"/>
  <text x="410" y="634" text-anchor="middle" font-family="system-ui,sans-serif" font-size="14" font-weight="500" fill="white">increment_iteration</text>

  <!-- max iter → END -->
  <line x1="530" y1="630" x2="598" y2="630" stroke="#888" stroke-width="1.5" marker-end="url(#ah)"/>
  <text x="564" y="622" font-family="system-ui,sans-serif" font-size="12" fill="#888">max iter</text>
  <rect x="598" y="610" width="80" height="40" rx="20" fill="#A32D2D" stroke="#791F1F" stroke-width="0.5"/>
  <text x="638" y="634" text-anchor="middle" font-family="system-ui,sans-serif" font-size="14" font-weight="500" fill="white">END</text>

  <!-- repair loop back -->
  <path d="M290 630 L180 630 L180 170 L288 170" fill="none" stroke="#888" stroke-width="1.5" marker-end="url(#ah)"/>
  <text x="160" y="400" text-anchor="middle" font-family="system-ui,sans-serif" font-size="12" fill="#888" transform="rotate(-90 160 400)">repair loop</text>

  <!-- persist arrow: update_learning_log → HF Dataset Sync -->
  <path d="M530 482 L730 482 L730 162" fill="none" stroke="#BA7517" stroke-width="1.5" stroke-dasharray="5 3" marker-end="url(#ahd)"/>
  <text x="648" y="474" font-family="system-ui,sans-serif" font-size="11" fill="#BA7517">persist lessons</text>

  <!-- SQLite checkpoint annotation -->
  <rect x="660" y="600" width="130" height="64" rx="8" fill="#F0F0FF" stroke="#7B7ACF" stroke-width="1.5" stroke-dasharray="6 3"/>
  <text x="725" y="622" text-anchor="middle" font-family="system-ui,sans-serif" font-size="12" font-weight="600" fill="#534AB7">SQLite</text>
  <text x="725" y="640" text-anchor="middle" font-family="system-ui,sans-serif" font-size="12" font-weight="600" fill="#534AB7">Checkpoints</text>
  <text x="725" y="656" text-anchor="middle" font-family="system-ui,sans-serif" font-size="11" fill="#7B7ACF">time-travel debugging</text>
  <line x1="660" y1="630" x2="532" y2="630" stroke="#7B7ACF" stroke-width="1.5" stroke-dasharray="5 3" marker-end="url(#ah)"/>

  <!-- Legend -->
  <rect x="40" y="770" width="740" height="140" rx="8" fill="#F8F8F8" stroke="#DDD" stroke-width="1"/>
  <text x="410" y="792" text-anchor="middle" font-family="system-ui,sans-serif" font-size="12" font-weight="600" fill="#555">Legend</text>
  <rect x="60" y="804" width="14" height="14" rx="3" fill="#185FA5"/>
  <text x="80" y="816" font-family="system-ui,sans-serif" font-size="12" fill="#555">Core pipeline node</text>
  <rect x="60" y="828" width="14" height="14" rx="3" fill="#0F6E56"/>
  <text x="80" y="840" font-family="system-ui,sans-serif" font-size="12" fill="#555">Spec test generator / success END</text>
  <rect x="60" y="852" width="14" height="14" rx="3" fill="#993C1D"/>
  <text x="80" y="864" font-family="system-ui,sans-serif" font-size="12" fill="#555">Diagnosis (ReAct)</text>
  <rect x="260" y="804" width="14" height="14" rx="3" fill="#534AB7"/>
  <text x="280" y="816" font-family="system-ui,sans-serif" font-size="12" fill="#555">Critic review gate</text>
  <rect x="260" y="828" width="14" height="14" rx="3" fill="#993556"/>
  <text x="280" y="840" font-family="system-ui,sans-serif" font-size="12" fill="#555">HITL interrupt</text>
  <rect x="260" y="852" width="14" height="14" rx="3" fill="#A32D2D"/>
  <text x="280" y="864" font-family="system-ui,sans-serif" font-size="12" fill="#555">Failure / max-iter END</text>
  <rect x="480" y="804" width="14" height="14" rx="3" fill="#FEF3CD" stroke="#BA7517" stroke-width="1.5"/>
  <text x="500" y="816" font-family="system-ui,sans-serif" font-size="12" fill="#555">Cross-session storage (HF)</text>
  <rect x="480" y="828" width="14" height="14" rx="3" fill="#F0F0FF" stroke="#7B7ACF" stroke-width="1.5"/>
  <text x="500" y="840" font-family="system-ui,sans-serif" font-size="12" fill="#555">SQLite checkpoints</text>
  <text x="500" y="864" font-family="system-ui,sans-serif" font-size="11" fill="#888">Dashed border = external persistent storage</text>
</svg>
</div>
"""

# ---------------------------------------------------------------------------
# Gradio UI construction
# ---------------------------------------------------------------------------

# Single production SVG — parallel spec+solution entry, no HITL (full_auto),
# critic always present, HF Dataset sync and SQLite checkpoint annotations.
_PRODUCTION_SVG = """
<div style="background:white;padding:20px;border-radius:8px;text-align:center;overflow:auto;">
<svg width="100%" viewBox="0 0 680 940" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <marker id="ah" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M2 1L8 5L2 9" fill="none" stroke="#888" stroke-width="1.5" stroke-linecap="round"/>
    </marker>
    <marker id="ahd" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M2 1L8 5L2 9" fill="none" stroke="#BA7517" stroke-width="1.5" stroke-linecap="round"/>
    </marker>
  </defs>

  <!-- Title -->
  <text x="340" y="28" text-anchor="middle" font-family="system-ui,sans-serif" font-size="15" font-weight="600" fill="#333">Production topology — all features enabled</text>
  <text x="340" y="48" text-anchor="middle" font-family="system-ui,sans-serif" font-size="12" fill="#888">AgentConfig() defaults: critic + spec tests + HF memory + checkpointing</text>

  <!-- USER QUERY entry pill -->
  <rect x="250" y="64" width="180" height="36" rx="18" fill="#5F5E5A" stroke="#444441" stroke-width="0.5"/>
  <text x="340" y="86" text-anchor="middle" font-family="system-ui,sans-serif" font-size="14" font-weight="500" fill="white">User Query</text>

  <!-- Fork arrows to parallel nodes -->
  <line x1="280" y1="100" x2="180" y2="128" stroke="#888" stroke-width="1.5" marker-end="url(#ah)"/>
  <line x1="400" y1="100" x2="490" y2="128" stroke="#888" stroke-width="1.5" marker-end="url(#ah)"/>

  <!-- PARALLEL BAND label -->
  <text x="340" y="118" text-anchor="middle" font-family="system-ui,sans-serif" font-size="11" fill="#aaa">parallel</text>

  <!-- generate_spec_tests (left) -->
  <rect x="60" y="128" width="220" height="56" rx="8" fill="#0F6E56" stroke="#085041" stroke-width="0.5"/>
  <text x="170" y="153" text-anchor="middle" font-family="system-ui,sans-serif" font-size="14" font-weight="500" fill="white">generate_spec_tests</text>
  <text x="170" y="173" text-anchor="middle" font-family="system-ui,sans-serif" font-size="12" fill="#7ECFB8">Blind oracle from spec</text>

  <!-- generate_solution (right) -->
  <rect x="400" y="128" width="220" height="56" rx="8" fill="#185FA5" stroke="#0C447C" stroke-width="0.5"/>
  <text x="510" y="153" text-anchor="middle" font-family="system-ui,sans-serif" font-size="14" font-weight="500" fill="white">generate_solution</text>
  <text x="510" y="173" text-anchor="middle" font-family="system-ui,sans-serif" font-size="12" fill="#8BB8E8">retrieve past lessons</text>

  <!-- HF Dataset Sync annotation (top right, connected to generate_solution) -->
  <rect x="510" y="56" width="150" height="64" rx="8" fill="#FEF3CD" stroke="#BA7517" stroke-width="1.5" stroke-dasharray="6 3"/>
  <text x="585" y="80" text-anchor="middle" font-family="system-ui,sans-serif" font-size="12" font-weight="600" fill="#BA7517">HF Dataset Sync</text>
  <text x="585" y="97" text-anchor="middle" font-family="system-ui,sans-serif" font-size="11" fill="#BA7517">cross-session lessons</text>
  <text x="585" y="112" text-anchor="middle" font-family="system-ui,sans-serif" font-size="11" fill="#BA7517">cosine dedup (0.92)</text>
  <line x1="510" y1="88" x2="460" y2="132" stroke="#BA7517" stroke-width="1.5" stroke-dasharray="5 3" marker-end="url(#ahd)"/>
  <text x="476" y="110" text-anchor="middle" font-family="system-ui,sans-serif" font-size="11" fill="#BA7517">retrieve</text>

  <!-- Join arrows to create_adversarial_tests -->
  <line x1="170" y1="184" x2="280" y2="232" stroke="#888" stroke-width="1.5" marker-end="url(#ah)"/>
  <line x1="510" y1="184" x2="400" y2="232" stroke="#888" stroke-width="1.5" marker-end="url(#ah)"/>

  <!-- create_adversarial_tests -->
  <rect x="200" y="232" width="280" height="44" rx="8" fill="#185FA5" stroke="#0C447C" stroke-width="0.5"/>
  <text x="340" y="258" text-anchor="middle" font-family="system-ui,sans-serif" font-size="14" font-weight="500" fill="white">create_adversarial_tests</text>
  <line x1="340" y1="276" x2="340" y2="300" stroke="#888" stroke-width="1.5" marker-end="url(#ah)"/>

  <!-- execute_solution -->
  <rect x="210" y="300" width="260" height="56" rx="8" fill="#185FA5" stroke="#0C447C" stroke-width="0.5"/>
  <text x="340" y="325" text-anchor="middle" font-family="system-ui,sans-serif" font-size="14" font-weight="500" fill="white">execute_solution</text>
  <text x="340" y="345" text-anchor="middle" font-family="system-ui,sans-serif" font-size="12" fill="#8BB8E8">Runs BOTH spec + adversarial</text>

  <!-- pass → critic_review -->
  <line x1="470" y1="328" x2="534" y2="328" stroke="#888" stroke-width="1.5" marker-end="url(#ah)"/>
  <text x="502" y="320" font-family="system-ui,sans-serif" font-size="12" fill="#888">pass</text>

  <!-- critic_review -->
  <rect x="534" y="306" width="126" height="56" rx="8" fill="#534AB7" stroke="#3A3388" stroke-width="0.5"/>
  <text x="597" y="331" text-anchor="middle" font-family="system-ui,sans-serif" font-size="14" font-weight="500" fill="white">critic_review</text>
  <text x="597" y="351" text-anchor="middle" font-family="system-ui,sans-serif" font-size="12" fill="#B8B4F0">Self-reflection gate</text>

  <!-- critic approve → END (success) -->
  <line x1="597" y1="306" x2="597" y2="272" stroke="#888" stroke-width="1.5" marker-end="url(#ah)"/>
  <text x="609" y="292" font-family="system-ui,sans-serif" font-size="12" fill="#888">approve</text>
  <rect x="557" y="232" width="80" height="40" rx="20" fill="#0F6E56" stroke="#085041" stroke-width="0.5"/>
  <text x="597" y="256" text-anchor="middle" font-family="system-ui,sans-serif" font-size="14" font-weight="500" fill="white">END</text>

  <!-- critic reject → diagnose_failure -->
  <line x1="534" y1="356" x2="474" y2="408" stroke="#888" stroke-width="1.5" stroke-dasharray="5 3" marker-end="url(#ah)"/>
  <text x="516" y="380" font-family="system-ui,sans-serif" font-size="12" fill="#888">reject</text>

  <!-- fail → diagnose_failure -->
  <line x1="340" y1="356" x2="340" y2="396" stroke="#888" stroke-width="1.5" marker-end="url(#ah)"/>
  <text x="356" y="380" font-family="system-ui,sans-serif" font-size="12" fill="#888">fail</text>

  <!-- diagnose_failure -->
  <rect x="210" y="396" width="260" height="56" rx="8" fill="#993C1D" stroke="#712B13" stroke-width="0.5"/>
  <text x="340" y="421" text-anchor="middle" font-family="system-ui,sans-serif" font-size="14" font-weight="500" fill="white">diagnose_failure</text>
  <text x="340" y="441" text-anchor="middle" font-family="system-ui,sans-serif" font-size="12" fill="#F0997B">ReAct loop with tools</text>
  <line x1="340" y1="452" x2="340" y2="476" stroke="#888" stroke-width="1.5" marker-end="url(#ah)"/>

  <!-- update_learning_log -->
  <rect x="210" y="476" width="260" height="44" rx="8" fill="#185FA5" stroke="#0C447C" stroke-width="0.5"/>
  <text x="340" y="502" text-anchor="middle" font-family="system-ui,sans-serif" font-size="14" font-weight="500" fill="white">update_learning_log</text>
  <line x1="340" y1="520" x2="340" y2="544" stroke="#888" stroke-width="1.5" marker-end="url(#ah)"/>

  <!-- persist → HF Dataset Sync dashed arrow (right side) -->
  <path d="M470 498 L585 498 L585 120" fill="none" stroke="#BA7517" stroke-width="1.5" stroke-dasharray="5 3" marker-end="url(#ahd)"/>
  <text x="540" y="490" font-family="system-ui,sans-serif" font-size="11" fill="#BA7517">persist lessons</text>

  <!-- increment_iteration -->
  <rect x="210" y="544" width="260" height="44" rx="8" fill="#5F5E5A" stroke="#444441" stroke-width="0.5"/>
  <text x="340" y="570" text-anchor="middle" font-family="system-ui,sans-serif" font-size="14" font-weight="500" fill="white">increment_iteration</text>

  <!-- SQLite checkpoints annotation (right of increment_iteration) -->
  <rect x="490" y="536" width="138" height="56" rx="8" fill="#F0F0FF" stroke="#7B7ACF" stroke-width="1.5" stroke-dasharray="6 3"/>
  <text x="559" y="558" text-anchor="middle" font-family="system-ui,sans-serif" font-size="12" font-weight="600" fill="#534AB7">SQLite</text>
  <text x="559" y="574" text-anchor="middle" font-family="system-ui,sans-serif" font-size="12" font-weight="600" fill="#534AB7">Checkpoints</text>
  <text x="559" y="585" text-anchor="middle" font-family="system-ui,sans-serif" font-size="10" fill="#7B7ACF">time-travel debug</text>
  <line x1="490" y1="566" x2="472" y2="566" stroke="#7B7ACF" stroke-width="1.5" stroke-dasharray="5 3" marker-end="url(#ah)"/>

  <!-- max iter → END (failure) -->
  <line x1="340" y1="588" x2="340" y2="612" stroke="#888" stroke-width="1.5" marker-end="url(#ah)"/>
  <text x="356" y="604" font-family="system-ui,sans-serif" font-size="12" fill="#888">max iter?</text>

  <!-- routing diamond placeholder lines -->
  <line x1="210" y1="566" x2="120" y2="566" stroke="#888" stroke-width="1.5"/>
  <line x1="120" y1="566" x2="120" y2="156" stroke="#888" stroke-width="1.5"/>
  <line x1="120" y1="156" x2="400" y2="156" stroke="#888" stroke-width="1.5" marker-end="url(#ah)"/>
  <text x="100" y="370" text-anchor="middle" font-family="system-ui,sans-serif" font-size="12" fill="#888" transform="rotate(-90 100 370)">repair loop → generate_solution</text>

  <!-- END (max iterations) -->
  <rect x="290" y="612" width="100" height="40" rx="20" fill="#A32D2D" stroke="#791F1F" stroke-width="0.5"/>
  <text x="340" y="636" text-anchor="middle" font-family="system-ui,sans-serif" font-size="14" font-weight="500" fill="white">END</text>

  <!-- Legend -->
  <rect x="20" y="672" width="640" height="244" rx="8" fill="#F8F8F8" stroke="#DDD" stroke-width="1"/>
  <text x="340" y="694" text-anchor="middle" font-family="system-ui,sans-serif" font-size="12" font-weight="600" fill="#555">Legend</text>
  <rect x="40" y="706" width="14" height="14" rx="3" fill="#185FA5"/>
  <text x="60" y="718" font-family="system-ui,sans-serif" font-size="12" fill="#555">Core pipeline node</text>
  <rect x="40" y="728" width="14" height="14" rx="3" fill="#0F6E56"/>
  <text x="60" y="740" font-family="system-ui,sans-serif" font-size="12" fill="#555">Spec test generator / success END</text>
  <rect x="40" y="750" width="14" height="14" rx="3" fill="#993C1D"/>
  <text x="60" y="762" font-family="system-ui,sans-serif" font-size="12" fill="#555">Diagnosis — ReAct loop with tools</text>
  <rect x="40" y="772" width="14" height="14" rx="3" fill="#5F5E5A"/>
  <text x="60" y="784" font-family="system-ui,sans-serif" font-size="12" fill="#555">Control flow / entry</text>
  <rect x="240" y="706" width="14" height="14" rx="3" fill="#534AB7"/>
  <text x="260" y="718" font-family="system-ui,sans-serif" font-size="12" fill="#555">Critic review gate</text>
  <rect x="240" y="728" width="14" height="14" rx="3" fill="#A32D2D"/>
  <text x="260" y="740" font-family="system-ui,sans-serif" font-size="12" fill="#555">Failure / max-iter END</text>
  <rect x="240" y="750" width="14" height="14" rx="3" fill="#FEF3CD" stroke="#BA7517" stroke-width="1.5"/>
  <text x="260" y="762" font-family="system-ui,sans-serif" font-size="12" fill="#555">HF Dataset Sync (external)</text>
  <rect x="240" y="772" width="14" height="14" rx="3" fill="#F0F0FF" stroke="#7B7ACF" stroke-width="1.5"/>
  <text x="260" y="784" font-family="system-ui,sans-serif" font-size="12" fill="#555">SQLite Checkpoints (external)</text>
  <text x="340" y="814" text-anchor="middle" font-family="system-ui,sans-serif" font-size="11" fill="#888">Dashed border = external persistent storage  ·  Dashed line = async side-effect</text>
  <text x="340" y="832" text-anchor="middle" font-family="system-ui,sans-serif" font-size="11" fill="#888">Parallel fan-in: generate_spec_tests + generate_solution both complete before create_adversarial_tests starts</text>
  <text x="340" y="850" text-anchor="middle" font-family="system-ui,sans-serif" font-size="11" fill="#888">Repair loop returns to generate_solution only (spec tests run once per task, not per iteration)</text>
  <text x="340" y="868" text-anchor="middle" font-family="system-ui,sans-serif" font-size="11" fill="#888">autonomy_level=full_auto — no HITL interrupt (required for Gradio/HF Spaces compatibility)</text>
  <text x="340" y="886" text-anchor="middle" font-family="system-ui,sans-serif" font-size="11" fill="#888">Provider: Claude Haiku 4.5 via Anthropic API (auto-selected when ANTHROPIC_API_KEY is set)</text>
  <text x="340" y="904" text-anchor="middle" font-family="system-ui,sans-serif" font-size="11" fill="#888">Semantic dedup threshold: cosine similarity &gt; 0.92 filters paraphrase lessons before HF sync</text>
</svg>
</div>
"""


def build_app(
    config: AgentConfig | None = None,
    router: "LLMRouter | None" = None,
) -> gr.Blocks:
    """
    Build the Gradio application.

    Args:
        config: AgentConfig to use. Defaults to AgentConfig() (production defaults).
        router: Optional pre-built LLMRouter. If None, run_demo_async builds one
                via build_router_with_generator_override() on first call.
    """
    if config is None:
        config = AgentConfig()

    summary_text, cat_fig, iter_fig = _build_performance_components()

    css = """
    .timeline-box { font-family: monospace; font-size: 0.85rem; }
    .code-box { font-family: monospace; }
    .lessons-box { font-size: 0.9rem; }
    """

    with gr.Blocks(title="Self-Healing Code Agent", css=css) as app:
        gr.Markdown(
            """
# Self-Healing Code Agent

An autonomous agent that generates Python code, adversarially tests it,
diagnoses failures, and repairs solutions through structured iteration.

**Architecture:** Generator → QA Adversarial → Sandbox Execution → Debugger → Repair Loop
            """
        )

        # ── Tab 1: Run Live Agent ────────────────────────────────────────────
        with gr.Tab("Run Live Agent"):

            # Holds the live graph + thread config between HITL pause and resume
            agent_session_state = gr.State(value=None)

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
                    for i, ex in enumerate(EXAMPLE_TASKS[:4]):
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

            # HITL review panel — hidden until an interrupt fires
            with gr.Group(visible=False) as review_panel:
                gr.Markdown("### ⏸ Repair Review — Awaiting Human Decision")
                review_info = gr.JSON(label="Diagnosis", value=None)
                edited_strategy = gr.Textbox(
                    label="Edit repair strategy (optional — leave blank to approve as-is)",
                    lines=3,
                    placeholder="Leave blank to approve the diagnosis unchanged...",
                )
                with gr.Row():
                    approve_btn = gr.Button("Approve", variant="primary")
                    abort_btn = gr.Button("Abort", variant="stop")

            def _clear():
                return (
                    "",
                    "Enter a task and click Run Agent.",
                    "# Code will appear here during agent execution.",
                    "No lessons recorded yet.",
                    gr.update(visible=False),
                    None,
                )

            clear_btn.click(
                fn=_clear,
                outputs=[task_input, timeline, code_output, learning_log,
                         review_panel, agent_session_state],
            )

            async def _run_streaming(task: str, session):
                """Async generator for Gradio streaming with HITL support."""
                if not task or not task.strip():
                    yield (
                        "Please enter a task description.",
                        "# No task provided.",
                        "No lessons recorded yet.",
                        gr.update(visible=False),
                        None,
                    )
                    return

                local_session = None

                async for item in run_demo_async(task, config=config, router=router):
                    # Session handshake — store the live graph reference
                    if isinstance(item, dict) and item.get("type") == "session":
                        local_session = AgentSession(
                            app=item["app"],
                            thread_config=item["thread_config"],
                            events_seen=0,
                        )
                        continue

                    # Interrupt event — show review panel and stop streaming
                    if isinstance(item, dict) and item.get("type") == "repair_review":
                        if local_session is not None:
                            local_session.events_seen = item.get("events_seen", 0)
                        payload = item.get("payload", {})
                        yield (
                            gr.update(),                   # timeline unchanged
                            gr.update(),                   # code unchanged
                            gr.update(),                   # lessons unchanged
                            gr.update(visible=True),       # show review panel
                            local_session,
                        )
                        return  # stop generator, wait for user button click

                    # Normal UI tuple (timeline, code, lessons)
                    if isinstance(item, tuple) and len(item) == 3:
                        timeline_text, code_text, lessons_text = item
                        yield (
                            timeline_text,
                            code_text,
                            lessons_text,
                            gr.update(visible=False),
                            local_session,
                        )

            run_btn.click(
                fn=_run_streaming,
                inputs=[task_input, agent_session_state],
                outputs=[timeline, code_output, learning_log,
                         review_panel, agent_session_state],
            )

            async def _approve_repair(edited: str, session):
                """Resume graph after human approves (optionally editing strategy)."""
                if session is None:
                    yield (gr.update(), gr.update(), gr.update(), gr.update(visible=False))
                    return
                decision = (
                    {"action": "edit", "edited_strategy": edited}
                    if edited and edited.strip()
                    else {"action": "approve"}
                )
                async for timeline_t, code_t, lessons_t in resume_demo_async(
                    session, decision, router=router, config=config
                ):
                    yield (timeline_t, code_t, lessons_t, gr.update(visible=False))

            approve_btn.click(
                fn=_approve_repair,
                inputs=[edited_strategy, agent_session_state],
                outputs=[timeline, code_output, learning_log, review_panel],
            )

            async def _abort_repair(session):
                """Resume graph with abort decision."""
                if session is None:
                    yield (gr.update(), gr.update(), gr.update(), gr.update(visible=False))
                    return
                decision = {"action": "abort"}
                async for timeline_t, code_t, lessons_t in resume_demo_async(
                    session, decision, router=router, config=config
                ):
                    yield (timeline_t, code_t, lessons_t, gr.update(visible=False))

            abort_btn.click(
                fn=_abort_repair,
                inputs=[agent_session_state],
                outputs=[timeline, code_output, learning_log, review_panel],
            )

        # ── Tab 2: Graph Topology ─────────────────────────────────────────────
        with gr.Tab("Graph Topology"):
            gr.Markdown("## Agent graph topology")
            gr.Markdown("Production configuration — all features enabled.")
            gr.HTML(value=_PRODUCTION_SVG)

        # ── Tab 3: Performance ───────────────────────────────────────────────
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

            if cat_fig is not None:
                gr.Plot(value=cat_fig, label="Success Rate by Category")

            if iter_fig is not None:
                gr.Plot(value=iter_fig, label="Iteration Distribution")

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
    app = build_app(config=AgentConfig())
    app.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
        share=False,
    )


if __name__ == "__main__":
    main()
