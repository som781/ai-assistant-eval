import os
from openai import OpenAI

WINDOW_SIZE = 6  # number of recent turns to keep in full


def compress_history(history: list[dict], client: OpenAI) -> list[dict]:
    """
    When history exceeds WINDOW_SIZE turns, compress the oldest turns into
    a 3-sentence summary and return a trimmed history. WINDOW_SIZE is counted
    in turns; each turn is 2 messages (user + assistant).
    """
    keep_messages = WINDOW_SIZE * 2  # retain the last WINDOW_SIZE turns in full
    if len(history) <= keep_messages:
        return history

    old_turns = history[:-keep_messages]
    recent_turns = history[-keep_messages:]

    conversation_text = "\n".join(
        f"{m['role'].capitalize()}: {m['content']}" for m in old_turns
    )
    summary_response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "Summarize the following conversation in exactly 3 concise sentences, capturing key facts and context.",
            },
            {"role": "user", "content": conversation_text},
        ],
        max_tokens=150,
        temperature=0.3,
    )
    summary = summary_response.choices[0].message.content.strip()

    compressed = [{"role": "system", "content": f"[Earlier conversation summary]: {summary}"}]
    compressed.extend(recent_turns)
    return compressed


class SlidingWindowMemory:
    """Maintains conversation history with automatic compression."""

    def __init__(self):
        self._client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self._history: list[dict] = []
        self.compressed_at: list[int] = []  # turn numbers where compression happened

    def add_turn(self, user_msg: str, assistant_msg: str):
        self._history.append({"role": "user", "content": user_msg})
        self._history.append({"role": "assistant", "content": assistant_msg})

    def get_history(self) -> tuple[list[dict], bool]:
        """Returns (history, was_compressed)."""
        if self.turn_count() > WINDOW_SIZE:
            compressed = compress_history(self._history, self._client)
            was_compressed = len(compressed) < len(self._history)
            self._history = compressed  # write back so next call doesn't re-summarize
            return self._history, was_compressed
        return self._history, False

    def reset(self):
        self._history = []
        self.compressed_at = []

    def turn_count(self) -> int:
        return len(self._history) // 2
