import re
import os
import json
from openai import OpenAI

SAFE_FALLBACK = "[Response blocked by output safety filter]"

# Simple PII patterns
_PII_PATTERNS = [
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),           # SSN
    re.compile(r"\b4[0-9]{12}(?:[0-9]{3})?\b"),      # Visa card
    re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),  # Email (basic)
]

TOXICITY_PROMPT = """You are a content safety filter. Determine if the assistant response contains any of:
- Hate speech or discriminatory content
- Instructions for harmful or illegal activities
- Explicit sexual content

Respond with ONLY: {"toxic": true} or {"toxic": false}"""


class OutputGuard:
    def __init__(self):
        self._client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def check(self, response: str) -> dict:
        """
        Returns:
            {"allowed": bool, "reason": str, "safe_response": str}
        """
        # PII check (fast, no API call)
        for pattern in _PII_PATTERNS:
            if pattern.search(response):
                return {
                    "allowed": False,
                    "reason": "PII detected in output",
                    "safe_response": SAFE_FALLBACK,
                }

        # Toxicity check
        try:
            result = self._client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": TOXICITY_PROMPT},
                    {"role": "user", "content": response},
                ],
                max_tokens=20,
                temperature=0,
            )
            raw = result.choices[0].message.content.strip()
            parsed = json.loads(raw)
            if parsed.get("toxic"):
                return {
                    "allowed": False,
                    "reason": "Toxic content detected in output",
                    "safe_response": SAFE_FALLBACK,
                }
        except Exception:
            pass  # On classifier failure, allow output through

        return {"allowed": True, "reason": "clean", "safe_response": response}
