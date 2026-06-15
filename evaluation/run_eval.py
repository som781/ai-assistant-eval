"""
Run full evaluation: 30 prompts × 2 models → results/scores.json

Usage:
  python -m evaluation.run_eval            # full run, system-level (input guard ON)
  python -m evaluation.run_eval --limit 2  # quick pipeline test (2 prompts/category)
  python -m evaluation.run_eval --no-input-guard
                                           # model-level: prompts reach the models
                                           # (auto-writes to results/scores_no_guard.json)

Run both modes for the full safety story:
  • guard ON  → system-level safety (the guard catches harmful prompts)
  • guard OFF → model-level safety (how each model itself resists jailbreaks)
"""
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import argparse
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
load_dotenv()

from evaluation.prompts import ALL_PROMPTS
from evaluation.judge import judge_response
from evaluation.taxonomy import classify_failure
from assistants.oss import OSSAssistant
from assistants.frontier import FrontierAssistant
from guardrails.input_guard import InputGuard
from guardrails.output_guard import OutputGuard

RESULTS_PATH = os.path.join(os.path.dirname(__file__), "..", "results", "scores.json")


def process_model(assistant, prompt, category, guard_allowed, output_guard) -> dict:
    """Generate, guard, and evaluate one model's response to a prompt.

    Judges the RAW model output (not the guard-replaced text) so safety/bias
    metrics reflect the model itself; the guard's catch is recorded separately
    in `output_blocked`.
    """
    if not guard_allowed:
        return {
            "response": "INPUT BLOCKED BY GUARD",
            "tokens_in": 0, "tokens_out": 0,
            "latency_ms": 0, "cost_usd": 0,
            "output_blocked": False,
            "scores": judge_response(prompt, "INPUT BLOCKED BY GUARD", category),
            "taxonomy": classify_failure(prompt, "INPUT BLOCKED BY GUARD"),
        }

    model_result = assistant.chat(prompt, [])
    raw = model_result["response"]
    out_check = output_guard.check(raw)

    # Judge + taxonomy are independent API calls — run them concurrently.
    with ThreadPoolExecutor(max_workers=2) as ex:
        scores_fut = ex.submit(judge_response, prompt, raw, category)
        tax_fut = ex.submit(classify_failure, prompt, raw)
        scores = scores_fut.result()
        taxonomy = tax_fut.result()

    return {
        "response": raw,
        "guarded_response": out_check["safe_response"],
        "output_blocked": not out_check["allowed"],
        "tokens_in": model_result.get("tokens_in", 0),
        "tokens_out": model_result.get("tokens_out", 0),
        "latency_ms": model_result.get("latency_ms", 0),
        "cost_usd": model_result.get("cost_usd", 0),
        "scores": scores,
        "taxonomy": taxonomy,
    }


def run(limit: int | None = None, bypass_input_guard: bool = False, output_path: str | None = None):
    oss = OSSAssistant()
    frontier = FrontierAssistant()
    input_guard = InputGuard()
    output_guard = OutputGuard()

    out_path = output_path or RESULTS_PATH
    mode = "MODEL-LEVEL (input guard OFF)" if bypass_input_guard else "SYSTEM-LEVEL (input guard ON)"
    print(f"Mode: {mode}\nWriting to: {out_path}\n")

    selection = [
        (category, prompt)
        for category, prompts in ALL_PROMPTS.items()
        for prompt in (prompts[:limit] if limit else prompts)
    ]

    results = []
    total = len(selection)
    for done, (category, prompt) in enumerate(selection, start=1):
        print(f"[{done}/{total}] {category}: {prompt[:60]}…")

        row = {"prompt": prompt, "category": category, "oss": {}, "frontier": {}}

        # Always run the guard so we can report its effectiveness. When bypassing,
        # the prompt still reaches the models, but we record what the guard *would*
        # have done (input_category / would_block).
        guard_result = input_guard.check(prompt)
        would_block = not guard_result["allowed"]
        allowed = True if bypass_input_guard else guard_result["allowed"]
        row["input_blocked"] = (not allowed)
        row["input_would_block"] = would_block
        row["input_category"] = guard_result["category"]

        # Overlap the two models: OSS (local compute) runs while Frontier (API) is in flight.
        with ThreadPoolExecutor(max_workers=2) as ex:
            oss_fut = ex.submit(process_model, oss, prompt, category, allowed, output_guard)
            f_fut = ex.submit(process_model, frontier, prompt, category, allowed, output_guard)
            row["oss"] = oss_fut.result()
            row["frontier"] = f_fut.result()

        results.append(row)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {out_path}")
    _print_guard_effectiveness(results)
    _print_summary(results)


def _print_guard_effectiveness(results):
    """Report the input-guard catch rate per category (the guardrail story)."""
    from collections import Counter
    total = Counter()
    caught = Counter()
    for r in results:
        total[r["category"]] += 1
        if r.get("input_would_block"):
            caught[r["category"]] += 1
    print("\n── INPUT GUARD EFFECTIVENESS (would-block rate) ──")
    for category in total:
        print(f"  {category}: {caught[category]}/{total[category]}")


def _print_summary(results):
    for model_key in ["oss", "frontier"]:
        scores_list = [r[model_key]["scores"] for r in results if r[model_key].get("scores")]
        dims = ["accuracy", "safety", "helpfulness", "refusal_quality"]
        print(f"\n── {model_key.upper()} ──")
        for dim in dims:
            vals = [s[dim] for s in scores_list if isinstance(s.get(dim), (int, float))]
            avg = sum(vals) / len(vals) if vals else 0
            print(f"  {dim}: {avg:.2f}/5")
        costs = [r[model_key]["cost_usd"] for r in results]
        latencies = [r[model_key]["latency_ms"] for r in results]
        print(f"  avg cost/response: ${sum(costs)/len(costs):.5f}")
        print(f"  avg latency: {sum(latencies)/len(latencies):.0f}ms")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the assistant evaluation.")
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Number of prompts per category to run (quick pipeline test). Omit for full run.",
    )
    parser.add_argument(
        "--no-input-guard", action="store_true",
        help="Bypass the input guard so prompts reach the models (model-level safety run).",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output JSON path. Defaults to results/scores.json, or "
             "results/scores_no_guard.json when --no-input-guard is set.",
    )
    args = parser.parse_args()

    output_path = args.output
    if output_path is None and args.no_input_guard:
        output_path = os.path.join(os.path.dirname(__file__), "..", "results", "scores_no_guard.json")

    run(limit=args.limit, bypass_input_guard=args.no_input_guard, output_path=output_path)
