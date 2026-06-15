# AI Personal Assistant Evaluation

Two personal assistants — one open-source, one frontier — built, compared, and evaluated head-to-head.

## Assistants

| | OSS | Frontier |
|---|---|---|
| **Model** | Qwen2.5-0.5B-Instruct | GPT-4.1 |
| **Runtime** | Self-hosted via `transformers` (local CPU/MPS/CUDA) | OpenAI API |
| **Cost/response** | Free (self-hosted compute) | ~$0.002–0.008 |
| **Tool use** | Prompt-based JSON | Native function calling |
| **HF Spaces** | [Live demo](#) | — |

> **Note:** The HF free *serverless* Inference API no longer serves Qwen2.5-0.5B-Instruct,
> so the OSS assistant runs the model in-process via `transformers`. This is the same code
> path used by the HF Spaces deployment (`spaces_app.py`). First run downloads the model (~1 GB).

## Setup

```bash
git clone https://github.com/som781/ai-assistant-eval
cd ai-assistant-eval
pip install -r requirements.txt
cp .env.example .env
# Fill in your keys in .env
```

## Run

```bash
# App: Chat (side-by-side demo) + Evaluation (benchmark + charts) tabs
streamlit run app/app.py

# Full evaluation (30 prompts × 2 models)
python -m evaluation.run_eval

# Generate report figures
python -m report.generate_report

# Smoke test individual assistants
python -m assistants.oss
python -m assistants.frontier
```

## Architecture

```
User message
    │
    ├── InputGuard (GPT-4o-mini classifier)
    │       └── blocked → fixed refusal
    │
    ├── SlidingWindowMemory (last 6 turns; older → GPT-4o-mini summary)
    │
    ├── OSSAssistant (Qwen2.5 self-hosted via transformers)
    │   └── Prompt-based tool use
    │
    ├── FrontierAssistant (GPT-4.1 via OpenAI)
    │   └── Native function calling
    │
    ├── OutputGuard (PII regex + GPT-4o-mini toxicity check)
    │
    └── Langfuse trace (latency, tokens, cost, guardrail decisions)
```

## Evaluation Framework

- **30 prompts** across 3 categories: factual (10), adversarial (10), bias/sensitive (10)
- **LLM-as-judge**: GPT-4.1 scores each response on accuracy, safety, helpfulness, refusal_quality (1–5)
- **Judges raw model output**, then records the guard's action separately — so safety/bias metrics reflect the *model*, not the guardrail masking it
- **Failure taxonomy**: Each response classified into one of 7 failure modes
- **In-app view**: the Streamlit Evaluation tab loads the saved benchmark and renders the metrics table + charts; a small live sample can be run from the UI

### Two-mode safety evaluation

Safety is measured at two layers, because they answer different questions:

| Mode | Command | Measures |
|------|---------|----------|
| **System-level** (guard ON) | `python -m evaluation.run_eval` | Does the *system* refuse harmful prompts? (input guard catches them) |
| **Model-level** (guard OFF) | `python -m evaluation.run_eval --no-input-guard` | How does each *model itself* resist jailbreaks when prompts reach it? |

The input guard blocks ~9/10 adversarial prompts before they reach a model, so the
guard-ON run alone makes both models look identical on jailbreak resistance. The
guard-OFF run (`results/scores_no_guard.json`) exposes the real model-level
difference, while the guard's catch rate is still reported in both runs. Generate
the second set of figures with:

```bash
python -m report.generate_report --scores results/scores_no_guard.json --outdir results/figures_no_guard
```

Then produce the combined **guard-ON vs guard-OFF** chart (`results/figures/guard_comparison.png`),
which overlays adversarial-prompt safety in both modes and annotates the guard's catch
rate and any raw-model jailbreak successes:

```bash
python -m report.compare_guard
```

## Key Design Decisions

**Why Qwen2.5-0.5B-Instruct?**
Small enough to self-host on a laptop (CPU/MPS) or a free CPU-tier Space, and representative of constrained open-source deployments. It runs in-process via `transformers`, as the Hugging Face serverless Inference API no longer hosts this model.

**Why GPT-4.1?**
Latest GPT-4 variant, with strong native function-calling support and predictable per-token pricing.

**Why sliding window + compression memory?**
Naive full-history context breaks at ~10 turns for a 512-token model. Compressing older turns to a 3-sentence summary mirrors production-grade memory patterns without adding a vector store dependency.

**Why GPT-4o-mini for guardrails/judge internals?**
Guardrail calls (input classifier, output toxicity, compression) are latency-sensitive and cheap. GPT-4o-mini is fast and inexpensive (~5–10× cheaper than GPT-4.1). The judge uses GPT-4.1 for accuracy.

**Judge bias note:** GPT-4.1 judging GPT-4.1 has a known self-serving bias. We mitigate this with a structured rubric and note it in the report. In production, a separate model or human eval would be preferred.

## Tradeoffs

| Decision | What we gained | What we gave up |
|----------|---------------|-----------------|
| Self-hosted Qwen2.5-0.5B via `transformers` | Zero per-token cost, runs offline, matches HF Spaces deploy | ~1 GB model download, local compute |
| Prompt-based tool use for OSS | No fine-tuning needed; works via a parse → execute → re-generate loop | The 0.5B model triggers tools inconsistently (see note below) |
| GPT-4o-mini for guards | Fast + cheap | Slightly less accurate than GPT-4.1 |
| Stub web search | No API key needed | Not a real search result |
| In-memory session state | Simple, no DB | State lost on page refresh |

**Note on OSS tool use:** the tool-call machinery (parse JSON → execute → feed result back → regenerate in natural language) is fully implemented and verified. However, Qwen2.5-**0.5B** is small enough that it doesn't reliably *decide* when to call `get_current_date` — it tends to either over-call or skip it. A conservative system prompt curbs the over-calling. The frontier assistant (GPT-4.1) uses native function calling and invokes tools reliably. A larger OSS model (3B+) would close most of this gap.

## What I'd Improve With More Time

1. **Real web search** via Brave Search or SerpAPI instead of the stub
2. **Vector memory** (ChromaDB/Pinecone) for long-term cross-session recall
3. **Fine-tuned guard model** (Llama Guard) instead of GPT-4o-mini classifier
4. **Multi-judge panel** — rotate judge models to reduce bias
5. **Streaming responses** in the Streamlit UI
6. **Larger OSS model** (3B+) to close the tool-use and accuracy gap

## Evaluation Results

See `results/figures/` for:
- `radar.png` — quality dimensions radar chart
- `cost_quality.png` — cost vs quality scatter
- `taxonomy.png` — failure mode breakdown
- `latency.png` — latency distribution

Full evaluation report: `results/evaluation_report.pdf`
