import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import uuid
import json
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from assistants.oss import OSSAssistant
from assistants.frontier import FrontierAssistant
from memory.sliding_window import SlidingWindowMemory
from guardrails.input_guard import InputGuard
from guardrails.output_guard import OutputGuard
from observability.tracer import trace_turn, get_langfuse
from evaluation.prompts import ALL_PROMPTS
from evaluation.run_eval import process_model
from report.generate_report import avg_scores_by_model, _draw_radar, _draw_taxonomy, _draw_cost_quality

st.set_page_config(page_title="AI Assistant Evaluation", layout="wide")


@st.cache_resource
def load_assistants():
    return OSSAssistant(), FrontierAssistant()


@st.cache_resource
def load_guards():
    return InputGuard(), OutputGuard()


oss, frontier = load_assistants()
input_guard, output_guard = load_guards()

OSS_NAME = oss.get_model_name().split("/")[-1]
FRONTIER_NAME = frontier.get_model_name()
SCORES_PATH = os.path.join(os.path.dirname(__file__), "..", "results", "scores.json")

# ── Session state ──────────────────────────────────────────────────────────────
for key, default in {
    "chat_oss_history": [],
    "chat_frontier_history": [],
    "oss_memory": None,
    "frontier_memory": None,
    "eval_results": None,
    "session_id": str(uuid.uuid4()),
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

for mem_key in ["oss_memory", "frontier_memory"]:
    if st.session_state[mem_key] is None:
        st.session_state[mem_key] = SlidingWindowMemory()


def _compute_response(model_key, message, memory):
    """Pure compute path — thread-safe (no st.session_state access)."""
    history, was_compressed = memory.get_history()
    assistant = oss if model_key == "oss" else frontier
    result = assistant.chat(message, history)
    out_check = output_guard.check(result["response"])
    result["response"] = out_check["safe_response"]
    result["output_blocked"] = not out_check["allowed"]
    result["was_compressed"] = was_compressed
    result["model_name"] = assistant.get_model_name()
    return result


def run_pair(message, specs, session_id):
    """Run both assistants concurrently, then commit memory + traces on the main thread."""
    with ThreadPoolExecutor(max_workers=len(specs)) as ex:
        futures = [ex.submit(_compute_response, mk, message, mem) for mk, mem in specs]
        results = [f.result() for f in futures]
    for (_, mem), result in zip(specs, results):
        mem.add_turn(message, result["response"])
        trace_turn(
            session_id=session_id, model_name=result["model_name"],
            user_message=message, response=result["response"],
            tokens_in=result["tokens_in"], tokens_out=result["tokens_out"],
            latency_ms=result["latency_ms"], cost_usd=result["cost_usd"],
        )
    return results


def reset_chat():
    st.session_state.chat_oss_history = []
    st.session_state.chat_frontier_history = []
    st.session_state.oss_memory.reset()
    st.session_state.frontier_memory.reset()


def _meta_caption(result, show_cost):
    bits = [f"{result['latency_ms']:.0f} ms", f"{result['tokens_in']}→{result['tokens_out']} tokens"]
    bits.append(f"${result['cost_usd']:.5f}" if show_cost else "free (self-hosted)")
    if result.get("was_compressed"):
        bits.append("memory compressed")
    if result.get("output_blocked"):
        bits.append("output filtered")
    return "  ·  ".join(bits)


def _avg_field(results, mk, field):
    vals = [r[mk].get(field, 0) for r in results if isinstance(r[mk].get(field), (int, float))]
    return sum(vals) / len(vals) if vals else 0.0


def _load_saved_results():
    with open(SCORES_PATH) as f:
        return {"results": json.load(f), "bypassed": False, "source": "saved"}


def render_eval_results(results, bypassed):
    n_adv = sum(1 for r in results if r["category"] == "adversarial")
    caught = sum(1 for r in results if r["category"] == "adversarial" and r.get("input_would_block"))

    st.markdown("##### Summary")
    oss_s, fr_s = avg_scores_by_model(results, "oss"), avg_scores_by_model(results, "frontier")
    st.table({
        "Metric": ["Accuracy", "Safety", "Helpfulness", "Refusal quality", "Avg cost / resp", "Avg latency (ms)"],
        "OSS": [f"{oss_s['accuracy']:.2f}", f"{oss_s['safety']:.2f}", f"{oss_s['helpfulness']:.2f}",
                f"{oss_s['refusal_quality']:.2f}", f"${_avg_field(results,'oss','cost_usd'):.5f}",
                f"{_avg_field(results,'oss','latency_ms'):.0f}"],
        "Frontier": [f"{fr_s['accuracy']:.2f}", f"{fr_s['safety']:.2f}", f"{fr_s['helpfulness']:.2f}",
                     f"{fr_s['refusal_quality']:.2f}", f"${_avg_field(results,'frontier','cost_usd'):.5f}",
                     f"{_avg_field(results,'frontier','latency_ms'):.0f}"],
    })
    if n_adv:
        msg = f"Input guard would block {caught}/{n_adv} adversarial prompts."
        msg += " Guard bypassed — models answered them directly (model-level safety)." if bypassed \
            else " Blocked prompts never reached the models (system-level safety)."
        st.caption(msg)

    st.markdown("##### Charts")
    g1, g2 = st.columns(2)
    with g1:
        fig, ax = plt.subplots(figsize=(4, 4), subplot_kw=dict(polar=True))
        _draw_radar(ax, results)
        st.pyplot(fig, use_container_width=True)
    with g2:
        fig, ax = plt.subplots(figsize=(5, 4))
        _draw_taxonomy(ax, results)
        st.pyplot(fig, use_container_width=True)
    fig, ax = plt.subplots(figsize=(7, 3.5))
    _draw_cost_quality(ax, results)
    st.pyplot(fig, use_container_width=True)

    with st.expander(f"Per-prompt responses ({len(results)})"):
        for r in results:
            st.markdown(f"**[{r['category']}]** {r['prompt']}")
            cols = st.columns(2)
            for col, mk, label in [(cols[0], "oss", "OSS"), (cols[1], "frontier", "Frontier")]:
                with col:
                    st.markdown(f":{'blue' if mk == 'oss' else 'orange'}[{label}]")
                    st.write(r[mk]["response"])
                    sc = r[mk].get("scores", {})
                    st.caption(f"acc {sc.get('accuracy','-')} · safe {sc.get('safety','-')} · "
                               f"help {sc.get('helpfulness','-')} · {r[mk].get('taxonomy',{}).get('category','-')}")
            st.divider()


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("AI Assistant Evaluation")
    st.markdown(
        f":blue[**OSS**] `{OSS_NAME}` — self-hosted, free  \n"
        f":orange[**Frontier**] `{FRONTIER_NAME}` — hosted API"
    )
    with st.expander("About"):
        st.markdown(
            "- **Chat** — try both assistants live, side by side.\n"
            "- **Evaluation** — automated benchmark (LLM-as-judge) across "
            "factual, adversarial, and bias prompts.\n\n"
            "Both assistants share short-term memory, input/output guardrails, "
            "tool use, and Langfuse tracing."
        )
    st.caption("Langfuse tracing: on" if get_langfuse() else "Langfuse tracing: off")
    st.divider()
    if st.button("Reset chat", use_container_width=True):
        reset_chat()
        st.session_state.session_id = str(uuid.uuid4())
        st.rerun()


tab_chat, tab_eval = st.tabs(["Chat", "Evaluation"])

# ── Tab: Chat ──────────────────────────────────────────────────────────────────
with tab_chat:
    st.caption("Type once — both assistants answer in parallel. Quality, speed, and cost appear under each reply.")

    c1, c2 = st.columns(2)
    c1.markdown(f"##### :blue[OSS] · `{OSS_NAME}`")
    c2.markdown(f"##### :orange[Frontier] · `{FRONTIER_NAME}`")

    if not st.session_state.chat_oss_history:
        st.info("Ask a question in the box below. Try \"What's the capital of Australia?\"")

    for oss_turn, fr_turn in zip(st.session_state.chat_oss_history, st.session_state.chat_frontier_history):
        col_o, col_f = st.columns(2)
        for col, turn in [(col_o, oss_turn), (col_f, fr_turn)]:
            with col, st.chat_message(turn["role"]):
                st.write(turn["content"])
                if turn["role"] == "assistant" and turn.get("meta"):
                    st.caption(turn["meta"])

    if prompt := st.chat_input("Message both assistants…", key="chat_input"):
        guard = input_guard.check(prompt)
        st.session_state.chat_oss_history.append({"role": "user", "content": prompt})
        st.session_state.chat_frontier_history.append({"role": "user", "content": prompt})
        if not guard["allowed"]:
            blocked = f"_Blocked by input guard ({guard['category']})._ {guard['refusal']}"
            st.session_state.chat_oss_history.append({"role": "assistant", "content": blocked, "meta": ""})
            st.session_state.chat_frontier_history.append({"role": "assistant", "content": blocked, "meta": ""})
            st.rerun()
        with st.spinner("Both assistants are thinking…"):
            oss_r, fr_r = run_pair(
                prompt,
                [("oss", st.session_state.oss_memory), ("frontier", st.session_state.frontier_memory)],
                st.session_state.session_id,
            )
        st.session_state.chat_oss_history.append(
            {"role": "assistant", "content": oss_r["response"], "meta": _meta_caption(oss_r, False)})
        st.session_state.chat_frontier_history.append(
            {"role": "assistant", "content": fr_r["response"], "meta": _meta_caption(fr_r, True)})
        st.rerun()


# ── Tab: Evaluation ────────────────────────────────────────────────────────────
with tab_eval:
    st.caption("Automated benchmark — prompts (factual / adversarial / bias) scored by an LLM judge (GPT-4.1). "
               "This is the core comparison the project produces.")

    # Show the saved full benchmark by default, so the tab is informative on arrival.
    if st.session_state.eval_results is None and os.path.exists(SCORES_PATH):
        st.session_state.eval_results = _load_saved_results()

    with st.expander("Run a fresh sample live (optional)"):
        st.caption("Runs the same pipeline on a small sample so you can watch it work. "
                   "The full run is via CLI: `python -m evaluation.run_eval`.")
        cc1, cc2 = st.columns(2)
        n = cc1.select_slider("Prompts per category", options=[1, 2, 3, 5], value=2)
        bypass = cc2.checkbox("Bypass input guard (test raw model safety)", value=False)
        if st.button(f"Run sample  (~{n * 3} prompts × 2 models)", type="primary"):
            selection = [(c, p) for c, ps in ALL_PROMPTS.items() for p in ps[:n]]
            progress = st.progress(0.0, text="Starting…")
            rows = []
            for i, (cat, prompt) in enumerate(selection):
                progress.progress(i / len(selection), text=f"[{i+1}/{len(selection)}] {cat}: {prompt[:45]}…")
                g = input_guard.check(prompt)
                allowed = True if bypass else g["allowed"]
                rows.append({
                    "prompt": prompt, "category": cat, "input_blocked": not allowed,
                    "input_would_block": not g["allowed"], "input_category": g["category"],
                    "oss": process_model(oss, prompt, cat, allowed, output_guard),
                    "frontier": process_model(frontier, prompt, cat, allowed, output_guard),
                })
            progress.empty()
            st.session_state.eval_results = {"results": rows, "bypassed": bypass, "source": "live"}
            st.rerun()

    state = st.session_state.eval_results
    if not state:
        st.info("No saved results found. Generate them with `python -m evaluation.run_eval`.")
    else:
        if state.get("source") == "live":
            left, right = st.columns([4, 1])
            left.caption(f"Showing a live sample — {len(state['results'])} prompts"
                         + (" · input guard bypassed" if state["bypassed"] else ""))
            if right.button("Full benchmark", use_container_width=True) and os.path.exists(SCORES_PATH):
                st.session_state.eval_results = _load_saved_results()
                st.rerun()
        else:
            st.caption(f"Showing the saved full benchmark — {len(state['results'])} prompts.")
        render_eval_results(state["results"], state["bypassed"])
