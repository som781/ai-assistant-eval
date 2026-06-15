"""
Combined guard-ON vs guard-OFF safety comparison.

Reads both eval runs and produces a single figure showing how the input guard
changes measured safety on adversarial prompts — the core "safety story":
  • guard ON  → adversarial prompts are blocked before reaching the model
  • guard OFF → the model's own jailbreak resistance is exposed

Prerequisites — run both evals first:
  python -m evaluation.run_eval                    # → results/scores.json
  python -m evaluation.run_eval --no-input-guard   # → results/scores_no_guard.json

Usage:
  python -m report.compare_guard
  python -m report.compare_guard --on results/scores.json --off results/scores_no_guard.json
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import argparse
import numpy as np
import matplotlib.pyplot as plt

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
ON_DEFAULT = os.path.join(RESULTS_DIR, "scores.json")
OFF_DEFAULT = os.path.join(RESULTS_DIR, "scores_no_guard.json")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")

GUARD_ON_COLOR = "#55A868"   # green — protected
GUARD_OFF_COLOR = "#C44E52"  # red — raw model exposed


def _load(path: str, label: str) -> list:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{label} scores not found at {path}\n"
            "Run both evals first:\n"
            "  python -m evaluation.run_eval\n"
            "  python -m evaluation.run_eval --no-input-guard"
        )
    with open(path) as f:
        return json.load(f)


def _adv_safety(results: list, model_key: str) -> float:
    """Average judge `safety` score on adversarial prompts for one model."""
    vals = [
        r[model_key]["scores"]["safety"]
        for r in results
        if r["category"] == "adversarial"
        and isinstance(r[model_key].get("scores", {}).get("safety"), (int, float))
    ]
    return float(np.mean(vals)) if vals else 0.0


def _jailbreak_successes(results: list, model_key: str) -> int:
    """Count adversarial prompts where the model complied (taxonomy = jailbreak_success)."""
    return sum(
        1 for r in results
        if r["category"] == "adversarial"
        and r[model_key].get("taxonomy", {}).get("category") == "jailbreak_success"
    )


def _guard_catch_rate(guard_on_results: list) -> tuple[int, int]:
    adv = [r for r in guard_on_results if r["category"] == "adversarial"]
    caught = sum(1 for r in adv if r.get("input_would_block"))
    return caught, len(adv)


def _draw_guard_comparison(ax, on_results: list, off_results: list):
    models = ["oss", "frontier"]
    labels = ["OSS\n(Qwen2.5)", "Frontier\n(GPT-4.1)"]

    on_safety = [_adv_safety(on_results, m) for m in models]
    off_safety = [_adv_safety(off_results, m) for m in models]
    off_jailbreaks = [_jailbreak_successes(off_results, m) for m in models]
    caught, adv_total = _guard_catch_rate(on_results)

    x = np.arange(len(models))
    width = 0.35

    bars_on = ax.bar(x - width / 2, on_safety, width, label="Guard ON (system-level)",
                     color=GUARD_ON_COLOR, alpha=0.9)
    bars_off = ax.bar(x + width / 2, off_safety, width, label="Guard OFF (model-level)",
                      color=GUARD_OFF_COLOR, alpha=0.9)

    for bars in (bars_on, bars_off):
        for b in bars:
            ax.annotate(f"{b.get_height():.2f}",
                        xy=(b.get_x() + b.get_width() / 2, b.get_height()),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", fontsize=8)

    for i, n in enumerate(off_jailbreaks):
        ax.annotate(f"{n} jailbreak{'s' if n != 1 else ''}\ncomplied",
                    xy=(x[i] + width / 2, 0.15), ha="center", va="bottom",
                    fontsize=7, color="white", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Avg safety on adversarial (1–5)", fontsize=9)
    ax.set_ylim(0, 5.5)
    ax.tick_params(axis="y", labelsize=8)
    ax.set_title(
        "Safety on Adversarial Prompts: Guard ON vs OFF\n"
        f"Input guard catches {caught}/{adv_total} before they reach a model",
        fontsize=10,
    )
    ax.legend(loc="lower center", ncol=2, fontsize=7.5)
    ax.grid(axis="y", alpha=0.3)
    return caught, adv_total, off_jailbreaks


def plot_guard_comparison(on_results: list, off_results: list, outdir: str):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    caught, adv_total, off_jailbreaks = _draw_guard_comparison(ax, on_results, off_results)
    plt.tight_layout()
    os.makedirs(outdir, exist_ok=True)
    out_path = os.path.join(outdir, "guard_comparison.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")
    print(f"  Guard catch rate (adversarial): {caught}/{adv_total}")
    print(f"  Guard-OFF jailbreak successes — OSS: {off_jailbreaks[0]}, Frontier: {off_jailbreaks[1]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Guard ON vs OFF safety comparison chart.")
    parser.add_argument("--on", type=str, default=ON_DEFAULT, help="Guard-ON scores JSON.")
    parser.add_argument("--off", type=str, default=OFF_DEFAULT, help="Guard-OFF scores JSON.")
    parser.add_argument("--outdir", type=str, default=FIGURES_DIR, help="Output directory.")
    args = parser.parse_args()

    on_results = _load(args.on, "Guard-ON")
    off_results = _load(args.off, "Guard-OFF")
    plot_guard_comparison(on_results, off_results, args.outdir)
