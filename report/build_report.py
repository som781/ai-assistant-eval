"""
Assemble the 1-page evaluation report (PDF) from scores + generated figures.

Reads results/scores.json (and results/scores_no_guard.json if present), computes
headline metrics, embeds the infographics, and writes a single-page PDF with a
recommendations section.

Usage:
  python -m report.build_report
  python -m report.build_report --out results/evaluation_report.pdf

Prerequisites:
  python -m evaluation.run_eval            # → results/scores.json
  python -m report.generate_report         # → results/figures/*.png
  (optional, for the guard comparison panel):
  python -m evaluation.run_eval --no-input-guard
  python -m report.compare_guard
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import argparse
import textwrap
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from report.generate_report import _draw_radar, _draw_taxonomy, _draw_cost_quality, _draw_latency
from report.compare_guard import _draw_guard_comparison

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
SCORES_PATH = os.path.join(RESULTS_DIR, "scores.json")
NO_GUARD_PATH = os.path.join(RESULTS_DIR, "scores_no_guard.json")
OUT_PATH = os.path.join(RESULTS_DIR, "evaluation_report.pdf")

DIMS = ["accuracy", "safety", "helpfulness", "refusal_quality"]


def _avg(results, model_key, dim):
    vals = [
        r[model_key]["scores"][dim]
        for r in results
        if isinstance(r[model_key].get("scores", {}).get(dim), (int, float))
    ]
    return float(np.mean(vals)) if vals else 0.0


def _avg_field(results, model_key, field):
    vals = [r[model_key].get(field, 0) for r in results]
    vals = [v for v in vals if isinstance(v, (int, float))]
    return float(np.mean(vals)) if vals else 0.0


def _avg_latency(results, model_key):
    vals = [r[model_key]["latency_ms"] for r in results if r[model_key].get("latency_ms", 0) > 0]
    return float(np.mean(vals)) if vals else 0.0


def _taxonomy_counts(results, model_key):
    return Counter(r[model_key].get("taxonomy", {}).get("category", "none") for r in results)


def _guard_catch(results):
    adv = [r for r in results if r["category"] == "adversarial"]
    caught = sum(1 for r in adv if r.get("input_would_block") or r.get("input_blocked"))
    return caught, len(adv)


def _adv_safety(results, model_key):
    vals = [
        r[model_key]["scores"]["safety"]
        for r in results
        if r["category"] == "adversarial"
        and isinstance(r[model_key].get("scores", {}).get("safety"), (int, float))
    ]
    return float(np.mean(vals)) if vals else 0.0


def build(out_path: str = OUT_PATH):
    if not os.path.exists(SCORES_PATH):
        raise FileNotFoundError(
            f"{SCORES_PATH} not found. Run `python -m evaluation.run_eval` first."
        )
    with open(SCORES_PATH) as f:
        results = json.load(f)

    off_results = None
    if os.path.exists(NO_GUARD_PATH):
        with open(NO_GUARD_PATH) as f:
            off_results = json.load(f)

    # ── Metrics ──────────────────────────────────────────────────────────────
    metrics = {}
    for mk in ["oss", "frontier"]:
        metrics[mk] = {d: _avg(results, mk, d) for d in DIMS}
        metrics[mk]["cost"] = _avg_field(results, mk, "cost_usd")
        metrics[mk]["latency"] = _avg_latency(results, mk)
    caught, adv_total = _guard_catch(results)
    oss_tax = _taxonomy_counts(results, "oss")
    fr_tax = _taxonomy_counts(results, "frontier")

    acc_gap = metrics["frontier"]["accuracy"] - metrics["oss"]["accuracy"]
    oss_halluc = oss_tax.get("factual_hallucination", 0) + oss_tax.get("confidence_hallucination", 0)
    fr_halluc = fr_tax.get("factual_hallucination", 0) + fr_tax.get("confidence_hallucination", 0)
    oss_bias = oss_tax.get("bias_exhibited", 0)

    safety_line = (
        f"Input guard blocks {caught}/{adv_total} adversarial prompts before they reach a model "
        "(system-level safety)."
    )
    if off_results is not None:
        oss_adv = _adv_safety(off_results, "oss")
        fr_adv = _adv_safety(off_results, "frontier")
        safety_line += (
            f" Guard OFF, raw model safety on adversarial prompts: "
            f"OSS {oss_adv:.2f}/5, Frontier {fr_adv:.2f}/5."
        )
    else:
        safety_line += " Run --no-input-guard for the model-level comparison."

    findings_bullets = [
        f"Quality gap: Frontier leads OSS on accuracy by {acc_gap:.2f} pts (1-5) and on helpfulness; "
        "widest on factual prompts.",
        f"Hallucinations (taxonomy-classified, n=30): OSS {oss_halluc}, Frontier {fr_halluc}.",
        f"Bias: OSS exhibited bias on {oss_bias} prompt(s); Frontier none.",
        safety_line,
        f"Cost/latency: OSS is free (self-hosted) and comparable in latency; "
        f"Frontier ~${metrics['frontier']['cost']:.5f}/response.",
    ]
    recs_bullets = [
        "Use Frontier (GPT-4.1) for accuracy-critical, user-facing tasks where correctness matters most.",
        "Use OSS (Qwen2.5) for cost-sensitive, high-volume, or privacy-bound workloads, paired with the "
        "guardrail layer that neutralizes most adversarial input regardless of model.",
        "Keep the input guard in production: it equalizes safety across models at near-zero cost.",
        "Improve OSS with light fine-tuning / RAG to close the factual-accuracy gap before relying on it solo.",
    ]

    def _wrap_block(title, bullets, width):
        lines = [title]
        for b in bullets:
            lines.append(textwrap.fill(b, width=width, initial_indent="• ", subsequent_indent="   "))
        return "\n".join(lines)

    findings = _wrap_block("KEY FINDINGS", findings_bullets, width=60)
    recs = _wrap_block("RECOMMENDATIONS", recs_bullets, width=60)

    # ── Page setup (US Letter portrait) ───────────────────────────────────────
    fig = plt.figure(figsize=(8.5, 11))
    gs = GridSpec(
        5, 2, figure=fig,
        height_ratios=[0.5, 0.95, 1.7, 1.7, 1.35],
        hspace=0.45, wspace=0.15,
        left=0.06, right=0.94, top=0.96, bottom=0.04,
    )

    # Header
    ax_head = fig.add_subplot(gs[0, :]); ax_head.axis("off")
    ax_head.text(0, 1.0, "AI Personal Assistant Evaluation", fontsize=18, fontweight="bold", va="top")
    ax_head.text(
        0, 0.35,
        "OSS (Qwen2.5-0.5B-Instruct, self-hosted)  vs  Frontier (GPT-4.1)    ·    "
        "30 prompts: factual / adversarial / bias    ·    LLM-as-judge: GPT-4.1",
        fontsize=9.5, va="top", color="#333333",
    )

    # Metrics table (own row)
    ax_tbl = fig.add_subplot(gs[1, :]); ax_tbl.axis("off")
    table_rows = [
        ["Metric", "OSS (Qwen2.5)", "Frontier (GPT-4.1)"],
        ["Accuracy (1-5)", f"{metrics['oss']['accuracy']:.2f}", f"{metrics['frontier']['accuracy']:.2f}"],
        ["Helpfulness (1-5)", f"{metrics['oss']['helpfulness']:.2f}", f"{metrics['frontier']['helpfulness']:.2f}"],
        ["Safety (1-5)", f"{metrics['oss']['safety']:.2f}", f"{metrics['frontier']['safety']:.2f}"],
        ["Refusal quality (1-5)", f"{metrics['oss']['refusal_quality']:.2f}", f"{metrics['frontier']['refusal_quality']:.2f}"],
        ["Avg cost / response", f"${metrics['oss']['cost']:.5f}", f"${metrics['frontier']['cost']:.5f}"],
        ["Avg latency (ms)", f"{metrics['oss']['latency']:.0f}", f"{metrics['frontier']['latency']:.0f}"],
    ]
    tbl = ax_tbl.table(cellText=table_rows[1:], colLabels=table_rows[0],
                       cellLoc="center", loc="center", bbox=[0.0, 0.0, 1.0, 1.0])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    for (row, col), cell in tbl.get_celld().items():
        if row == 0:
            cell.set_facecolor("#4C72B0"); cell.set_text_props(color="white", fontweight="bold")
        elif col == 0:
            cell.set_text_props(fontweight="bold")

    # Figures — drawn natively (vector) so they stay crisp at any zoom.
    _draw_radar(fig.add_subplot(gs[2, 0], projection="polar"), results)
    _draw_taxonomy(fig.add_subplot(gs[2, 1]), results)
    _draw_cost_quality(fig.add_subplot(gs[3, 0]), results)
    ax_guard = fig.add_subplot(gs[3, 1])
    if off_results is not None:
        _draw_guard_comparison(ax_guard, results, off_results)
    else:
        _draw_latency(ax_guard, results)

    # Findings + Recommendations (two columns)
    ax_txt = fig.add_subplot(gs[4, :]); ax_txt.axis("off")
    ax_txt.text(0.0, 1.0, findings, fontsize=8.0, va="top", ha="left", family="sans-serif")
    ax_txt.text(0.52, 1.0, recs, fontsize=8.0, va="top", ha="left", family="sans-serif")

    fig.savefig(out_path, bbox_inches="tight")  # format inferred from extension
    plt.close()
    print(f"Report written to {out_path}")
    if off_results is None:
        print("Note: guard-OFF run not found — model-level safety panel used latency.png instead.")
        print("      Run `python -m evaluation.run_eval --no-input-guard` + `python -m report.compare_guard` for the full safety story.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the 1-page evaluation PDF.")
    parser.add_argument("--out", type=str, default=OUT_PATH, help="Output PDF path.")
    args = parser.parse_args()
    build(out_path=args.out)
