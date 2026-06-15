import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import uuid
import random
import json
from concurrent.futures import ThreadPoolExecutor
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from assistants.oss import OSSAssistant
from assistants.frontier import FrontierAssistant
from memory.sliding_window import SlidingWindowMemory
from guardrails.input_guard import InputGuard
from guardrails.output_guard import OutputGuard
from observability.tracer import trace_turn, get_langfuse

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

# ── Session state init ────────────────────────────────────────────────────────
for key, default in {
    "chat_oss_history": [],
    "chat_frontier_history": [],
    "oss_memory": None,
    "frontier_memory": None,
    "arena_oss_memory": None,       # separate from chat memory
    "arena_frontier_memory": None,
    "arena_history": [],
    "arena_assignment": None,   # {"A": "oss"|"frontier", "B": ...}
    "arena_scores": {"A": 0, "B": 0},
    "arena_revealed": False,
    "arena_turn": 0,
    "session_id": str(uuid.uuid4()),
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

if st.session_state.oss_memory is None:
    st.session_state.oss_memory = SlidingWindowMemory()
if st.session_state.frontier_memory is None:
    st.session_state.frontier_memory = SlidingWindowMemory()
if st.session_state.arena_oss_memory is None:
    st.session_state.arena_oss_memory = SlidingWindowMemory()
if st.session_state.arena_frontier_memory is None:
    st.session_state.arena_frontier_memory = SlidingWindowMemory()
if st.session_state.arena_assignment is None:
    models = ["oss", "frontier"]
    random.shuffle(models)
    st.session_state.arena_assignment = {"A": models[0], "B": models[1]}

ARENA_REVEAL_AFTER = 3  # rounds before the Reveal button unlocks


def _compute_response(model_key: str, message: str, memory: SlidingWindowMemory) -> dict:
    """Pure compute path — safe to run in a worker thread (no st.session_state access).

    OSS (local generation) and Frontier (API I/O) overlap well, so total turn
    latency becomes max(oss, frontier) instead of their sum.
    """
    history, was_compressed = memory.get_history()
    assistant = oss if model_key == "oss" else frontier
    result = assistant.chat(message, history)

    out_check = output_guard.check(result["response"])
    result["response"] = out_check["safe_response"]
    result["output_blocked"] = not out_check["allowed"]
    result["was_compressed"] = was_compressed
    result["model_name"] = assistant.get_model_name()
    return result


def run_pair(message: str, specs: "list[tuple[str, SlidingWindowMemory]]", session_id: str) -> list:
    """Run both assistants concurrently, then commit memory + traces on the main thread."""
    with ThreadPoolExecutor(max_workers=len(specs)) as ex:
        futures = [ex.submit(_compute_response, model_key, message, mem) for model_key, mem in specs]
        results = [f.result() for f in futures]

    for (_, mem), result in zip(specs, results):
        mem.add_turn(message, result["response"])
        trace_turn(
            session_id=session_id,
            model_name=result["model_name"],
            user_message=message,
            response=result["response"],
            tokens_in=result["tokens_in"],
            tokens_out=result["tokens_out"],
            latency_ms=result["latency_ms"],
            cost_usd=result["cost_usd"],
        )
    return results


def reset_chat():
    st.session_state.chat_oss_history = []
    st.session_state.chat_frontier_history = []
    st.session_state.oss_memory.reset()
    st.session_state.frontier_memory.reset()


def reset_arena():
    st.session_state.arena_history = []
    st.session_state.arena_scores = {"A": 0, "B": 0}
    st.session_state.arena_revealed = False
    st.session_state.arena_turn = 0
    models = ["oss", "frontier"]
    random.shuffle(models)
    st.session_state.arena_assignment = {"A": models[0], "B": models[1]}
    st.session_state.arena_oss_memory.reset()
    st.session_state.arena_frontier_memory.reset()


def _meta_caption(result: dict, show_cost: bool) -> str:
    bits = [
        f"{result['latency_ms']:.0f} ms",
        f"{result['tokens_in']}→{result['tokens_out']} tokens",
    ]
    bits.append(f"${result['cost_usd']:.5f}" if show_cost else "free (self-hosted)")
    if result.get("was_compressed"):
        bits.append("memory compressed")
    if result.get("output_blocked"):
        bits.append("output filtered")
    return "  ·  ".join(bits)


# ════════════════════════════════════════════════════════════════════════════════
# SIDEBAR — orientation + controls
# ════════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.title("AI Assistant Evaluation")
    st.markdown("Compare an open-source assistant with a frontier one, side by side.")

    st.markdown("**Assistants**")
    st.markdown(
        f"- :blue[**OSS**] — `{OSS_NAME}`  \n"
        f"  self-hosted, free, runs locally\n"
        f"- :orange[**Frontier**] — `{FRONTIER_NAME}`  \n"
        f"  hosted API, paid per token"
    )

    with st.expander("How to use this app"):
        st.markdown(
            "**Chat** — type once; both assistants answer in parallel so you can "
            "compare quality, speed, and cost.\n\n"
            "**Blind Arena** — answers are hidden as Model A / Model B. Vote for the "
            "better one each round, then reveal which was which. This removes brand "
            "bias from your judgement."
        )

    st.markdown("**Under the hood**")
    st.markdown(
        "- Short-term memory (sliding window + summary)\n"
        "- Input and output safety guardrails\n"
        "- Tool use (date / web-search stub)\n"
        "- Langfuse tracing"
    )
    lf_on = get_langfuse() is not None
    st.caption("Langfuse tracing: on" if lf_on else "Langfuse tracing: off (no keys set)")

    st.divider()
    if st.button("Reset everything", use_container_width=True):
        reset_chat()
        reset_arena()
        st.session_state.session_id = str(uuid.uuid4())
        st.rerun()


# ════════════════════════════════════════════════════════════════════════════════
tab1, tab2 = st.tabs(["Chat (side-by-side)", "Blind Arena"])

# ────────────────────────────────────────────────────────────────────────────────
# TAB 1 — CHAT MODE
# ────────────────────────────────────────────────────────────────────────────────
with tab1:
    head_l, head_r = st.columns([5, 1])
    with head_l:
        st.subheader("Side-by-side chat")
        st.caption("Type a message below — it goes to both assistants at once. Metrics appear under each reply.")
    with head_r:
        if st.button("Clear chat", key="clear_chat", use_container_width=True):
            reset_chat()
            st.rerun()

    col_oss, col_frontier = st.columns(2)
    with col_oss:
        st.markdown(f"##### :blue[OSS] · `{OSS_NAME}`")
    with col_frontier:
        st.markdown(f"##### :orange[Frontier] · `{FRONTIER_NAME}`")

    if not st.session_state.chat_oss_history:
        st.info(
            "Start by typing a question in the box at the bottom of the page. "
            "Try \"What's the capital of Australia?\", or \"What is today's date?\" to see tool use."
        )

    for oss_turn, fr_turn in zip(st.session_state.chat_oss_history, st.session_state.chat_frontier_history):
        c_oss, c_fr = st.columns(2)
        with c_oss:
            with st.chat_message(oss_turn["role"]):
                st.write(oss_turn["content"])
                if oss_turn["role"] == "assistant" and oss_turn.get("meta"):
                    st.caption(oss_turn["meta"])
        with c_fr:
            with st.chat_message(fr_turn["role"]):
                st.write(fr_turn["content"])
                if fr_turn["role"] == "assistant" and fr_turn.get("meta"):
                    st.caption(fr_turn["meta"])

    if prompt := st.chat_input("Type your message to both assistants…", key="chat_input"):
        guard = input_guard.check(prompt)
        st.session_state.chat_oss_history.append({"role": "user", "content": prompt})
        st.session_state.chat_frontier_history.append({"role": "user", "content": prompt})

        if not guard["allowed"]:
            blocked_msg = f"_Blocked by input guard ({guard['category']})._ {guard['refusal']}"
            st.session_state.chat_oss_history.append({"role": "assistant", "content": blocked_msg, "meta": ""})
            st.session_state.chat_frontier_history.append({"role": "assistant", "content": blocked_msg, "meta": ""})
            st.rerun()

        with st.spinner("Both assistants are thinking…"):
            oss_result, f_result = run_pair(
                prompt,
                [("oss", st.session_state.oss_memory), ("frontier", st.session_state.frontier_memory)],
                st.session_state.session_id,
            )
        st.session_state.chat_oss_history.append({
            "role": "assistant", "content": oss_result["response"],
            "meta": _meta_caption(oss_result, show_cost=False),
        })
        st.session_state.chat_frontier_history.append({
            "role": "assistant", "content": f_result["response"],
            "meta": _meta_caption(f_result, show_cost=True),
        })
        st.rerun()


# ────────────────────────────────────────────────────────────────────────────────
# TAB 2 — BLIND ARENA
# ────────────────────────────────────────────────────────────────────────────────
with tab2:
    st.subheader("Blind Arena")
    st.caption(
        "Judge the assistants without knowing which is which, to remove brand bias. "
        f"Vote each round; after {ARENA_REVEAL_AFTER} rounds you can reveal the identities."
    )

    assignment = st.session_state.arena_assignment
    arena_memories = {
        "oss": st.session_state.arena_oss_memory,
        "frontier": st.session_state.arena_frontier_memory,
    }
    scores = st.session_state.arena_scores
    history = st.session_state.arena_history
    awaiting_vote = bool(history) and history[-1]["winner"] is None

    st.markdown(
        "**Steps:**  1. Ask a question   2. Read Model A vs Model B   "
        "3. Vote for the better answer   4. Reveal after a few rounds."
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Model A wins", scores["A"])
    m2.metric("Model B wins", scores["B"])
    m3.metric("Rounds played", st.session_state.arena_turn)
    with m4:
        if not st.session_state.arena_revealed:
            disabled = st.session_state.arena_turn < ARENA_REVEAL_AFTER
            if st.button("Reveal identities", disabled=disabled, use_container_width=True,
                         help=f"Play at least {ARENA_REVEAL_AFTER} rounds to unlock"):
                st.session_state.arena_revealed = True
                st.rerun()
        if st.button("New arena", key="reset_arena", use_container_width=True):
            reset_arena()
            st.rerun()

    if st.session_state.arena_revealed:
        def _name(side):
            return OSS_NAME if assignment[side] == "oss" else FRONTIER_NAME
        st.success(f"Model A = {assignment['A'].upper()} (`{_name('A')}`)   ·   "
                   f"Model B = {assignment['B'].upper()} (`{_name('B')}`)")
        total = scores["A"] + scores["B"]
        if total > 0:
            oss_wins = scores[next(k for k, v in assignment.items() if v == "oss")]
            fr_wins = scores[next(k for k, v in assignment.items() if v == "frontier")]
            st.info(f"OSS won {oss_wins}/{total} · Frontier won {fr_wins}/{total} of your votes.")

    st.divider()

    if not history:
        st.info(
            "Ask your first question below to begin round 1. Good tests: a tricky factual "
            "question, or something subjective like \"Explain recursion to a five-year-old.\""
        )

    for idx in range(len(history) - 1, -1, -1):
        entry = history[idx]
        is_latest = idx == len(history) - 1
        st.markdown(f"**Round {idx + 1}** — you asked: _{entry['user']}_")
        ca, cb = st.columns(2)
        with ca:
            with st.container(border=True):
                st.markdown("**Model A**")
                st.write(entry["A"])
                st.caption(f"{entry['A_latency']:.0f} ms")
        with cb:
            with st.container(border=True):
                st.markdown("**Model B**")
                st.write(entry["B"])
                st.caption(f"{entry['B_latency']:.0f} ms")

        if entry["winner"] is None and is_latest and not st.session_state.arena_revealed:
            st.markdown("**Which answer is better?**")
            v1, v2, v3 = st.columns(3)
            if v1.button("Model A wins", key=f"vote_a_{idx}", use_container_width=True):
                entry["winner"] = "A"; scores["A"] += 1; st.rerun()
            if v2.button("Model B wins", key=f"vote_b_{idx}", use_container_width=True):
                entry["winner"] = "B"; scores["B"] += 1; st.rerun()
            if v3.button("Tie / skip", key=f"vote_tie_{idx}", use_container_width=True):
                entry["winner"] = "tie"; st.rerun()
        elif entry["winner"]:
            label = {"A": "Model A", "B": "Model B", "tie": "Tie"}[entry["winner"]]
            st.caption(f"Your pick: {label}")
        st.divider()

    if not st.session_state.arena_revealed and not awaiting_vote:
        next_round = st.session_state.arena_turn + 1
        with st.form("arena_form", clear_on_submit=True):
            st.markdown(f"**Ask round {next_round}:**")
            arena_prompt = st.text_input(
                "Your message", placeholder="Type a question for both models…",
                key="arena_text", label_visibility="collapsed",
            )
            submitted = st.form_submit_button("Send to both models")

        if submitted and arena_prompt:
            guard = input_guard.check(arena_prompt)
            if not guard["allowed"]:
                st.warning(f"Input blocked ({guard['category']}): {guard['refusal']}")
            else:
                a_model, b_model = assignment["A"], assignment["B"]
                with st.spinner("Both models are thinking…"):
                    a_result, b_result = run_pair(
                        arena_prompt,
                        [(a_model, arena_memories[a_model]), (b_model, arena_memories[b_model])],
                        st.session_state.session_id,
                    )
                history.append({
                    "user": arena_prompt,
                    "A": a_result["response"], "B": b_result["response"],
                    "A_latency": a_result["latency_ms"], "B_latency": b_result["latency_ms"],
                    "winner": None,
                })
                st.session_state.arena_turn += 1
                st.rerun()
    elif awaiting_vote and not st.session_state.arena_revealed:
        st.info("Vote on the round above to unlock the next question.")
