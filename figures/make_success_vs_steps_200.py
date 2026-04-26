#!/usr/bin/env python3
"""Plot perfect-score rate vs. step budget for the 200-step hard-task runs.

Matches the style of figures/success_vs_steps_rubrics.pdf.
"""
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 11,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "legend.fontsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.pad": 5,
    "ytick.major.pad": 5,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

MODEL_STYLES = {
    "Claude Opus 4.6":  {"color": "#d62728", "linestyle": "-",  "marker": "D"},
    "Qwen3.5-9B":       {"color": "#2ca02c", "linestyle": "--", "marker": "h"},
}
MARKER_EVERY = 16


def perfect_curve(tasks, max_steps=200):
    """Cumulative count of tasks whose rubric_scores are all 1 and whose
    num_steps <= budget, divided by total tasks."""
    total = len(tasks)
    counts = np.zeros(max_steps, dtype=int)
    for t in tasks:
        rs = t.get("rubric_scores")
        if not rs:
            continue
        if not all(v == 1 for v in rs.values()):
            continue
        n = t.get("num_steps", max_steps)
        counts[min(n, max_steps) - 1:] += 1
    return np.arange(1, max_steps + 1), counts / total, total


def main():
    # Canonical task set: 120 tasks from final_journeys_pass.json plus 80 LLM-generated hard tasks = 200
    valid_ids = {t["task_id"] for t in json.load(open("/home/ljang/final_journeys_pass.json"))}

    opus_clean = json.load(open(
        "/home/jykoh/OSWorld/runs_journeys_cua_opus46_final_clean_200steps/"
        "eval_results_gemini_judge_full_traj_per_rubric_response_first_notrunc.json"))
    opus_hard = json.load(open(
        "/home/jykoh/OSWorld/runs_journeys_cua_opus46_round2_200steps/"
        "eval_results_gemini_judge_full_traj_per_rubric_response_first_notrunc_postrefactor.json"))
    opus_tasks = [t for t in opus_clean["tasks"] if t["task_id"] in valid_ids] + list(opus_hard["tasks"])

    # Qwen only has 200-step runs for the 80 hard tasks.
    qwen_hard = json.load(open(
        "/home/jykoh/OSWorld/runs_journeys_cua_qwen35_9b_round2/"
        "eval_results_gemini_judge_full_traj_per_rubric_response_first_notrunc_postrefactor.json"))
    qwen_tasks = list(qwen_hard["tasks"])

    fig, ax = plt.subplots(figsize=(10, 4))
    steps, rates, n = perfect_curve(opus_tasks, max_steps=200)
    style = MODEL_STYLES["Claude Opus 4.6"]
    ax.plot(steps, rates * 100,
            label=f"Claude Opus 4.6 (n={n}, all tasks)",
            color=style["color"], linestyle=style["linestyle"],
            marker=style["marker"], markevery=MARKER_EVERY,
            markersize=5, linewidth=1.8)
    print(f"Opus 4.6: n={n}, final={rates[-1]*100:.2f}%")

    # Opus curve restricted to the same hard-task subset Qwen was evaluated on,
    # for an apples-to-apples comparison.
    qwen_ids = {t["task_id"] for t in qwen_tasks}
    opus_hard_subset = [t for t in opus_tasks if t["task_id"] in qwen_ids]
    steps, rates, n = perfect_curve(opus_hard_subset, max_steps=200)
    ax.plot(steps, rates * 100,
            label=f"Claude Opus 4.6 (n={n}, hard subset)",
            color=style["color"], linestyle=":",
            marker=style["marker"], markevery=MARKER_EVERY,
            markersize=5, linewidth=1.4, alpha=0.75)
    print(f"Opus 4.6 (hard subset): n={n}, final={rates[-1]*100:.2f}%")

    steps, rates, n = perfect_curve(qwen_tasks, max_steps=200)
    style = MODEL_STYLES["Qwen3.5-9B"]
    ax.plot(steps, rates * 100,
            label=f"Qwen3.5-9B (n={n}, hard subset)",
            color=style["color"], linestyle=style["linestyle"],
            marker=style["marker"], markevery=MARKER_EVERY,
            markersize=5, linewidth=1.8)
    print(f"Qwen3.5-9B: n={n}, final={rates[-1]*100:.2f}%")

    ax.axvline(100, color="0.5", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.text(101, 2, "100-step cap", fontsize=9, color="0.4")

    ax.set_xlabel("Step Budget")
    ax.set_ylabel("Perfect Score Rate (%)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}"))
    ax.set_ylim(0, None)
    ax.set_xlim(0, 200)

    ax.grid(True, which="major", linestyle="-", linewidth=0.4, alpha=0.35)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(frameon=True, fancybox=False, edgecolor="0.7", framealpha=0.9)

    fig.tight_layout()
    out = Path(__file__).parent / "success_vs_steps_200_hard.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
