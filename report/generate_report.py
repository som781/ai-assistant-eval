"""
Generate evaluation figures from a scores JSON file.

Usage:
  python -m report.generate_report                       # results/scores.json → results/figures/
  python -m report.generate_report --scores results/scores_no_guard.json \\
                                    --outdir results/figures_no_guard
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

SCORES_PATH = os.path.join(os.path.dirname(__file__), "..", "results", "scores.json")
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "figures")

OSS_COLOR = "#4C72B0"
FRONTIER_COLOR = "#DD8452"


def load_scores():
    if not os.path.exists(SCORES_PATH):
        raise FileNotFoundError(
            f"scores.json not found at {SCORES_PATH}\n"
            "Run `python -m evaluation.run_eval` first to generate evaluation results."
        )
    with open(SCORES_PATH) as f:
        return json.load(f)


def avg_scores_by_model(results, model_key):
    dims = ["accuracy", "safety", "helpfulness", "refusal_quality"]
    out = {}
    for dim in dims:
        vals = [
            r[model_key]["scores"][dim]
            for r in results
            if isinstance(r[model_key].get("scores", {}).get(dim), (int, float))
        ]
        out[dim] = sum(vals) / len(vals) if vals else 0
    return out


# Each chart has a `_draw_*(ax, results)` core (vector, reusable by the PDF report)
# and a `plot_*` wrapper that renders it to a standalone PNG.

# ── Figure 1: Radar chart ─────────────────────────────────────────────────────
def _draw_radar(ax, results):
    categories = ["Accuracy", "Safety", "Helpfulness", "Refusal\nQuality"]
    oss_vals = avg_scores_by_model(results, "oss")
    f_vals = avg_scores_by_model(results, "frontier")

    oss_data = [oss_vals["accuracy"], oss_vals["safety"], oss_vals["helpfulness"], oss_vals["refusal_quality"]]
    f_data = [f_vals["accuracy"], f_vals["safety"], f_vals["helpfulness"], f_vals["refusal_quality"]]

    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    oss_data += oss_data[:1]
    f_data += f_data[:1]

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=9)
    ax.set_ylim(0, 5)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_yticklabels(["1", "2", "3", "4", "5"], size=7)

    ax.plot(angles, oss_data, color=OSS_COLOR, linewidth=2, label="OSS (Qwen2.5)")
    ax.fill(angles, oss_data, color=OSS_COLOR, alpha=0.2)
    ax.plot(angles, f_data, color=FRONTIER_COLOR, linewidth=2, label="Frontier (GPT-4.1)")
    ax.fill(angles, f_data, color=FRONTIER_COLOR, alpha=0.2)

    ax.legend(loc="upper right", bbox_to_anchor=(1.28, 1.12), fontsize=7)
    ax.set_title("Model Quality Radar (1–5 scale)", size=11, pad=14)


def plot_radar(results):
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    _draw_radar(ax, results)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "radar.png"), dpi=200, bbox_inches="tight")
    plt.close()
    print("Saved radar.png")


# ── Figure 2: Cost-Quality scatter ───────────────────────────────────────────
def _draw_cost_quality(ax, results):
    for model_key, label, color in [("oss", "OSS (Qwen2.5)", OSS_COLOR), ("frontier", "Frontier (GPT-4.1)", FRONTIER_COLOR)]:
        costs, qualities = [], []
        for r in results:
            m = r[model_key]
            cost = m.get("cost_usd", 0)
            scores = m.get("scores", {})
            numeric_vals = [v for v in scores.values() if isinstance(v, (int, float))]
            if not numeric_vals:
                continue  # skip rows where judge errored
            q = np.mean(numeric_vals)
            costs.append(cost * 1000)  # convert to millicents for readability
            qualities.append(q)
        ax.scatter(costs, qualities, label=label, color=color, s=45, alpha=0.75)
        ax.scatter(np.mean(costs), np.mean(qualities), color=color, s=180, marker="*", zorder=5)

    ax.set_xlabel("Cost per response (USD × 10⁻³)", fontsize=9)
    ax.set_ylabel("Average judge score (1–5)", fontsize=9)
    ax.set_title("Cost vs Quality", fontsize=11)
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


def plot_cost_quality(results):
    fig, ax = plt.subplots(figsize=(7, 5))
    _draw_cost_quality(ax, results)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "cost_quality.png"), dpi=200, bbox_inches="tight")
    plt.close()
    print("Saved cost_quality.png")


# ── Figure 3: Failure taxonomy bar chart ─────────────────────────────────────
def _draw_taxonomy(ax, results):
    taxonomy_cats = [
        "factual_hallucination", "confidence_hallucination", "jailbreak_success",
        "bias_exhibited", "correct_refusal", "over_refusal", "none",
    ]
    labels = [c.replace("_", "\n") for c in taxonomy_cats]

    def count_taxonomy(model_key):
        counts = {c: 0 for c in taxonomy_cats}
        for r in results:
            cat = r[model_key].get("taxonomy", {}).get("category", "none")
            if cat in counts:
                counts[cat] += 1
        return [counts[c] for c in taxonomy_cats]

    oss_counts = count_taxonomy("oss")
    f_counts = count_taxonomy("frontier")

    x = np.arange(len(taxonomy_cats))
    width = 0.35
    ax.bar(x - width / 2, oss_counts, width, label="OSS (Qwen2.5)", color=OSS_COLOR, alpha=0.85)
    ax.bar(x + width / 2, f_counts, width, label="Frontier (GPT-4.1)", color=FRONTIER_COLOR, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=6.5)
    ax.set_ylabel("Count", fontsize=9)
    ax.set_title("Failure Mode Taxonomy", fontsize=11)
    ax.tick_params(axis="y", labelsize=8)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)


def plot_taxonomy(results):
    fig, ax = plt.subplots(figsize=(10, 5))
    _draw_taxonomy(ax, results)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "taxonomy.png"), dpi=200, bbox_inches="tight")
    plt.close()
    print("Saved taxonomy.png")


# ── Figure 4: Latency comparison ─────────────────────────────────────────────
def _draw_latency(ax, results):
    oss_latencies = [r["oss"]["latency_ms"] for r in results if r["oss"].get("latency_ms", 0) > 0]
    f_latencies = [r["frontier"]["latency_ms"] for r in results if r["frontier"].get("latency_ms", 0) > 0]

    ax.boxplot(
        [oss_latencies, f_latencies],
        labels=["OSS\n(Qwen2.5)", "Frontier\n(GPT-4.1)"],
        patch_artist=True,
        boxprops=dict(facecolor="lightblue"),
        medianprops=dict(color="red", linewidth=2),
    )
    ax.set_ylabel("Latency (ms)", fontsize=9)
    ax.set_title("Response Latency Distribution", fontsize=11)
    ax.tick_params(labelsize=8)
    ax.grid(axis="y", alpha=0.3)


def plot_latency(results):
    fig, ax = plt.subplots(figsize=(6, 5))
    _draw_latency(ax, results)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "latency.png"), dpi=200, bbox_inches="tight")
    plt.close()
    print("Saved latency.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate evaluation figures.")
    parser.add_argument("--scores", type=str, default=None, help="Path to a scores JSON file.")
    parser.add_argument("--outdir", type=str, default=None, help="Directory to write figures to.")
    args = parser.parse_args()

    if args.scores:
        SCORES_PATH = args.scores
    if args.outdir:
        FIGURES_DIR = args.outdir

    os.makedirs(FIGURES_DIR, exist_ok=True)
    results = load_scores()
    plot_radar(results)
    plot_cost_quality(results)
    plot_taxonomy(results)
    plot_latency(results)
    print(f"\nAll figures saved to {FIGURES_DIR}")
