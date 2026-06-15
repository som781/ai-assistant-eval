from __future__ import annotations

import re
import json
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from assistants.base import BaseAssistant
from tools.assistant_tools import OSS_TOOL_PROMPT, dispatch_tool

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

SYSTEM_PROMPT = (
    "You are a helpful, concise personal assistant.\n\n" + OSS_TOOL_PROMPT
)


def _pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _try_parse_tool_call(text: str) -> dict | None:
    """Return a tool-call dict if the model emitted one as JSON, else None."""
    text = text.strip()
    candidates = [text]
    m = re.search(r"\{.*?\}", text, re.DOTALL)
    if m:
        candidates.append(m.group(0))
    for c in candidates:
        try:
            obj = json.loads(c)
        except Exception:
            continue
        if isinstance(obj, dict) and "tool" in obj:
            return obj
    return None


class OSSAssistant(BaseAssistant):
    def __init__(self):
        self._device = _pick_device()
        self._tokenizer = AutoTokenizer.from_pretrained(MODEL)
        self._model = AutoModelForCausalLM.from_pretrained(
            MODEL, torch_dtype="auto"
        ).to(self._device)

    def get_model_name(self) -> str:
        return MODEL

    def get_cost_per_1k_tokens(self) -> dict:
        return {"input": 0.0, "output": 0.0}  # self-hosted — no per-token cost

    def _generate(self, messages: list[dict]) -> tuple[str, int, int, float]:
        """One generation pass. Returns (text, tokens_in, tokens_out, latency_ms)."""
        text = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tokenizer([text], return_tensors="pt").to(self._device)

        t0 = time.time()
        with torch.no_grad():
            generated = self._model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.7,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        latency_ms = (time.time() - t0) * 1000

        prompt_len = inputs.input_ids.shape[1]
        output_ids = generated[0][prompt_len:]
        out_text = self._tokenizer.decode(output_ids, skip_special_tokens=True).strip()
        return out_text, int(prompt_len), int(output_ids.shape[0]), latency_ms

    def chat(self, message: str, history: list[dict]) -> dict:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.extend(history)
        messages.append({"role": "user", "content": message})

        out_text, tokens_in, tokens_out, latency_ms = self._generate(messages)

        # Prompt-based tool use: if the model emitted a tool call, execute it and
        # feed the result back for a second, natural-language generation.
        tool_call = _try_parse_tool_call(out_text)
        if tool_call:
            tool_name = tool_call.get("tool", "")
            args = {k: v for k, v in tool_call.items() if k != "tool"}
            tool_result = dispatch_tool(tool_name, args)

            messages.append({"role": "assistant", "content": out_text})
            messages.append({
                "role": "user",
                "content": (
                    f"Tool '{tool_name}' returned: {tool_result}\n"
                    "Use this to answer my previous question in natural language. "
                    "Do not output JSON."
                ),
            })
            out_text, t_in2, t_out2, lat2 = self._generate(messages)
            tokens_in += t_in2
            tokens_out += t_out2
            latency_ms += lat2

        return {
            "response": out_text,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "latency_ms": latency_ms,
            "cost_usd": 0.0,
        }


if __name__ == "__main__":
    assistant = OSSAssistant()
    history = []
    for user_msg in ["Hi, what can you do?", "What is 2 + 2?", "What is today's date?"]:
        print(f"User: {user_msg}")
        result = assistant.chat(user_msg, history)
        print(f"Assistant: {result['response']}")
        print(f"  [{result['latency_ms']:.0f}ms | in:{result['tokens_in']} out:{result['tokens_out']}]\n")
        history.append({"role": "user", "content": user_msg})
        history.append({"role": "assistant", "content": result["response"]})
