from __future__ import annotations

import os
from langfuse import Langfuse

_langfuse: Langfuse | None = None


def get_langfuse() -> Langfuse | None:
    global _langfuse
    if _langfuse is None:
        pk = os.getenv("LANGFUSE_PUBLIC_KEY")
        sk = os.getenv("LANGFUSE_SECRET_KEY")
        host = os.getenv("LANGFUSE_BASE_URL", "https://cloud.langfuse.com")
        if pk and sk:
            _langfuse = Langfuse(public_key=pk, secret_key=sk, host=host)
    return _langfuse


def trace_turn(
    session_id: str,
    model_name: str,
    user_message: str,
    response: str,
    tokens_in: int,
    tokens_out: int,
    latency_ms: float,
    cost_usd: float,
    guardrail_input: dict | None = None,
    guardrail_output: dict | None = None,
):
    lf = get_langfuse()
    if lf is None:
        return

    # Observability is best-effort — a tracing failure must never break a turn.
    try:
        with lf.start_as_current_generation(
            name="llm_call",
            model=model_name,
            input=user_message,
            metadata={"latency_ms": latency_ms, "cost_usd": cost_usd},
        ) as generation:
            generation.update(
                output=response,
                usage_details={
                    "input": tokens_in,
                    "output": tokens_out,
                    "total": tokens_in + tokens_out,
                },
            )
            lf.update_current_trace(
                name="assistant_turn",
                session_id=session_id,
                metadata={
                    "model": model_name,
                    "guardrail_input_category": guardrail_input.get("category") if guardrail_input else None,
                    "guardrail_output_allowed": guardrail_output.get("allowed") if guardrail_output else None,
                },
            )
        lf.flush()
    except Exception as e:
        print(f"[langfuse] trace skipped: {e}")
