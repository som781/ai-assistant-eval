import os
import json
import time
from openai import OpenAI
from assistants.base import BaseAssistant
from tools.assistant_tools import get_current_date, search_web_stub, TOOL_SCHEMAS

MODEL = "gpt-4.1"
COST_PER_1K = {"input": 0.002, "output": 0.008}  # gpt-4.1 pricing (USD)

SYSTEM_PROMPT = "You are a helpful, concise personal assistant. Use tools when appropriate."


class FrontierAssistant(BaseAssistant):
    def __init__(self):
        self._client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def get_model_name(self) -> str:
        return MODEL

    def get_cost_per_1k_tokens(self) -> dict:
        return COST_PER_1K

    def chat(self, message: str, history: list[dict]) -> dict:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.extend(history)
        messages.append({"role": "user", "content": message})

        t0 = time.time()
        response = self._client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOL_SCHEMAS,
            tool_choice="auto",
            max_tokens=512,
            temperature=0.7,
        )

        # Handle tool calls if present
        msg = response.choices[0].message
        tokens_in = response.usage.prompt_tokens
        tokens_out = response.usage.completion_tokens
        if msg.tool_calls:
            messages.append(msg)
            for tc in msg.tool_calls:
                fn_name = tc.function.name
                fn_args = json.loads(tc.function.arguments)
                if fn_name == "get_current_date":
                    tool_result = get_current_date()
                elif fn_name == "search_web_stub":
                    tool_result = search_web_stub(fn_args.get("query", ""))
                else:
                    tool_result = "Unknown tool"
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": tool_result,
                })
            # Second call with tool results — accumulate tokens from both calls
            response2 = self._client.chat.completions.create(
                model=MODEL,
                messages=messages,
                max_tokens=512,
                temperature=0.7,
            )
            msg = response2.choices[0].message
            tokens_in += response2.usage.prompt_tokens
            tokens_out += response2.usage.completion_tokens

        latency_ms = (time.time() - t0) * 1000
        cost = (tokens_in / 1000 * COST_PER_1K["input"]) + (tokens_out / 1000 * COST_PER_1K["output"])

        return {
            "response": msg.content,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "latency_ms": latency_ms,
            "cost_usd": cost,
        }


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    assistant = FrontierAssistant()
    history = []
    for user_msg in ["Hi, what can you do?", "What is today's date?", "Thanks!"]:
        print(f"User: {user_msg}")
        result = assistant.chat(user_msg, history)
        print(f"Assistant: {result['response']}")
        print(f"  [{result['latency_ms']:.0f}ms | in:{result['tokens_in']} out:{result['tokens_out']} | ${result['cost_usd']:.5f}]\n")
        history.append({"role": "user", "content": user_msg})
        history.append({"role": "assistant", "content": result["response"]})
