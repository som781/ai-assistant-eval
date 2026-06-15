from __future__ import annotations

import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from assistants.base import BaseAssistant
from tools.assistant_tools import OSS_TOOL_PROMPT

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

    def chat(self, message: str, history: list[dict]) -> dict:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.extend(history)
        messages.append({"role": "user", "content": message})

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

        # Strip the prompt tokens; decode only the newly generated portion.
        prompt_len = inputs.input_ids.shape[1]
        output_ids = generated[0][prompt_len:]
        response_text = self._tokenizer.decode(output_ids, skip_special_tokens=True).strip()

        return {
            "response": response_text,
            "tokens_in": int(prompt_len),
            "tokens_out": int(output_ids.shape[0]),
            "latency_ms": latency_ms,
            "cost_usd": 0.0,
        }


if __name__ == "__main__":
    assistant = OSSAssistant()
    history = []
    for user_msg in ["Hi, what can you do?", "What is the capital of France?", "Thanks!"]:
        print(f"User: {user_msg}")
        result = assistant.chat(user_msg, history)
        print(f"Assistant: {result['response']}")
        print(f"  [{result['latency_ms']:.0f}ms | in:{result['tokens_in']} out:{result['tokens_out']}]\n")
        history.append({"role": "user", "content": user_msg})
        history.append({"role": "assistant", "content": result["response"]})
