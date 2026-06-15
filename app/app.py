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
from observability.tracer import trace_turn

st.set_page_config(page_title="AI Assistant Eval", layout="wide")


@st.cache_resource
def load_assistants():
    return OSSAssistant(), FrontierAssistant()


@st.cache_resource
def load_guards():
    return InputGuard(), OutputGuard()


oss, frontier = load_assistants()
input_guard, output_guard = load_guards()

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


def run_pair(message: str, specs: list[tuple[str, SlidingWindowMemory]], session_id: str) -> list[dict]:
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


# ── Tab layout ────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["Chat Mode", "Blind Arena"])

# ════════════════════════════════════════════════════════════════════════════════
# TAB 1 — CHAT MODE
# ════════════════════════════════════════════════════════════════════════════════
with tab1:
    st.title("AI Assistant — Side by Side")
    st.caption("Same message sent to both models simultaneously.")

    col_oss, col_frontier = st.columns(2)
    with col_oss:
        st.subheader(f"OSS: {oss.get_model_name().split('/')[-1]}")
    with col_frontier:
        st.subheader(f"Frontier: {frontier.get_model_name()}")

    # Render chat history
    for turn in st.session_state.chat_oss_history:
        with col_oss:
            with st.chat_message(turn["role"]):
                st.write(turn["content"])
    for turn in st.session_state.chat_frontier_history:
        with col_frontier:
            with st.chat_message(turn["role"]):
                st.write(turn["content"])

    if prompt := st.chat_input("Type your message…", key="chat_input"):
        # Input guard
        guard = input_guard.check(prompt)
        if not guard["allowed"]:
            st.warning(f"Input blocked ({guard['category']}): {guard['refusal']}")
        else:
            # Display user message
            with col_oss:
                with st.chat_message("user"):
                    st.write(prompt)
            with col_frontier:
                with st.chat_message("user"):
                    st.write(prompt)

            st.session_state.chat_oss_history.append({"role": "user", "content": prompt})
            st.session_state.chat_frontier_history.append({"role": "user", "content": prompt})

            # Run both models concurrently — total latency ≈ max(oss, frontier)
            with st.spinner("Both models thinking…"):
                oss_result, f_result = run_pair(
                    prompt,
                    [
                        ("oss", st.session_state.oss_memory),
                        ("frontier", st.session_state.frontier_memory),
                    ],
                    st.session_state.session_id,
                )

            # Display responses
            with col_oss:
                with st.chat_message("assistant"):
                    st.write(oss_result["response"])
                st.caption(
                    f"{oss_result['latency_ms']:.0f}ms · "
                    f"in:{oss_result['tokens_in']} out:{oss_result['tokens_out']} · "
                    f"{'[compressed]' if oss_result['was_compressed'] else ''}"
                )
            with col_frontier:
                with st.chat_message("assistant"):
                    st.write(f_result["response"])
                st.caption(
                    f"{f_result['latency_ms']:.0f}ms · "
                    f"in:{f_result['tokens_in']} out:{f_result['tokens_out']} · "
                    f"${f_result['cost_usd']:.5f} · "
                    f"{'[compressed]' if f_result['was_compressed'] else ''}"
                )

            st.session_state.chat_oss_history.append({"role": "assistant", "content": oss_result["response"]})
            st.session_state.chat_frontier_history.append({"role": "assistant", "content": f_result["response"]})

    if st.button("Clear Chat", key="clear_chat"):
        st.session_state.chat_oss_history = []
        st.session_state.chat_frontier_history = []
        st.session_state.oss_memory.reset()
        st.session_state.frontier_memory.reset()
        st.rerun()


# ════════════════════════════════════════════════════════════════════════════════
# TAB 2 — BLIND ARENA
# ════════════════════════════════════════════════════════════════════════════════
with tab2:
    st.title("Blind Arena")
    st.caption(
        "Both models respond to your message. Pick the better response each turn. "
        "After 5 turns, click **Reveal** to see which model was which."
    )

    assignment = st.session_state.arena_assignment
    arena_memories = {
        "oss": st.session_state.arena_oss_memory,
        "frontier": st.session_state.arena_frontier_memory,
    }

    scores = st.session_state.arena_scores
    score_col1, score_col2, score_col3 = st.columns(3)
    with score_col1:
        st.metric("Model A wins", scores["A"])
    with score_col2:
        st.metric("Model B wins", scores["B"])
    with score_col3:
        st.metric("Turns", st.session_state.arena_turn)

    # Render arena history
    for entry in st.session_state.arena_history:
        st.markdown(f"**You:** {entry['user']}")
        col_a, col_b = st.columns(2)
        with col_a:
            with st.container(border=True):
                st.markdown("**Model A**")
                st.write(entry["A"])
                st.caption(f"{entry['A_latency']:.0f}ms")
        with col_b:
            with st.container(border=True):
                st.markdown("**Model B**")
                st.write(entry["B"])
                st.caption(f"{entry['B_latency']:.0f}ms")
        if entry.get("winner"):
            st.success(f"You picked: Model {entry['winner']}")
        st.divider()

    # Reveal after 5 turns
    if st.session_state.arena_turn >= 5 and not st.session_state.arena_revealed:
        if st.button("Reveal Models", type="primary"):
            st.session_state.arena_revealed = True
            st.rerun()

    if st.session_state.arena_revealed:
        st.success(
            f"Model A = **{assignment['A'].upper()}** ({oss.get_model_name().split('/')[-1] if assignment['A'] == 'oss' else frontier.get_model_name()})  |  "
            f"Model B = **{assignment['B'].upper()}** ({oss.get_model_name().split('/')[-1] if assignment['B'] == 'oss' else frontier.get_model_name()})"
        )
        total = scores["A"] + scores["B"]
        if total > 0:
            oss_wins = scores[next(k for k, v in assignment.items() if v == "oss")]
            frontier_wins = scores[next(k for k, v in assignment.items() if v == "frontier")]
            st.info(f"OSS wins: {oss_wins}/{total} · Frontier wins: {frontier_wins}/{total}")

    # Arena input — only active if not yet revealed
    if not st.session_state.arena_revealed:
        if arena_prompt := st.chat_input("Type your message…", key="arena_input"):
            guard = input_guard.check(arena_prompt)
            if not guard["allowed"]:
                st.warning(f"Input blocked ({guard['category']}): {guard['refusal']}")
            else:
                # Get responses from both models (blind)
                a_model = assignment["A"]
                b_model = assignment["B"]

                a_memory = arena_memories[a_model]
                b_memory = arena_memories[b_model]

                with st.spinner("Both models thinking…"):
                    a_result, b_result = run_pair(
                        arena_prompt,
                        [(a_model, a_memory), (b_model, b_memory)],
                        st.session_state.session_id,
                    )

                entry = {
                    "user": arena_prompt,
                    "A": a_result["response"],
                    "B": b_result["response"],
                    "A_latency": a_result["latency_ms"],
                    "B_latency": b_result["latency_ms"],
                    "winner": None,
                }
                st.session_state.arena_history.append(entry)
                st.session_state.arena_turn += 1
                st.rerun()

    # Vote buttons for latest unanswered turn
    if st.session_state.arena_history and st.session_state.arena_history[-1]["winner"] is None:
        col_va, col_vb = st.columns(2)
        with col_va:
            if st.button("Model A is better", key="vote_a"):
                st.session_state.arena_history[-1]["winner"] = "A"
                st.session_state.arena_scores["A"] += 1
                st.rerun()
        with col_vb:
            if st.button("Model B is better", key="vote_b"):
                st.session_state.arena_history[-1]["winner"] = "B"
                st.session_state.arena_scores["B"] += 1
                st.rerun()

    if st.button("Reset Arena", key="reset_arena"):
        for k in ["arena_history", "arena_scores", "arena_revealed", "arena_turn"]:
            st.session_state.pop(k, None)
        st.session_state.arena_assignment = None
        st.session_state.arena_oss_memory.reset()
        st.session_state.arena_frontier_memory.reset()
        st.rerun()
