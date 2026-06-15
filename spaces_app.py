"""
HF Spaces deployment — Qwen2.5-0.5B-Instruct, self-hosted via transformers.
Deploy this file as the root app.py on a Hugging Face Space (CPU basic is enough
for the 0.5B model). Add `transformers`, `torch`, and `gradio` to the Space's
requirements.txt.
"""
import time
import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
SYSTEM_PROMPT = "You are a helpful, concise personal assistant."

device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype="auto").to(device)


def respond(message: str, history: list[list[str]]) -> str:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for user_msg, assistant_msg in history:
        messages.append({"role": "user", "content": user_msg})
        if assistant_msg:
            messages.append({"role": "assistant", "content": assistant_msg})
    messages.append({"role": "user", "content": message})

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(device)

    t0 = time.time()
    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id,
        )
    latency_ms = (time.time() - t0) * 1000

    prompt_len = inputs.input_ids.shape[1]
    output_ids = generated[0][prompt_len:]
    response = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
    print(f"latency={latency_ms:.0f}ms in={prompt_len} out={output_ids.shape[0]}")
    return response


demo = gr.ChatInterface(
    fn=respond,
    title="Qwen2.5-0.5B-Instruct Personal Assistant",
    description=f"A lightweight personal assistant self-hosted via transformers. Model: `{MODEL}`",
    examples=[
        "What's the capital of Japan?",
        "Explain quantum computing in simple terms.",
        "What can you help me with?",
    ],
)

if __name__ == "__main__":
    demo.launch()
