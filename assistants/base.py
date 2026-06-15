from abc import ABC, abstractmethod


class BaseAssistant(ABC):
    @abstractmethod
    def chat(self, message: str, history: list[dict]) -> dict:
        """
        Args:
            message: user input
            history: list of {"role": "user"|"assistant", "content": str}
        Returns:
            dict with keys: response (str), tokens_in (int), tokens_out (int),
                            latency_ms (float), cost_usd (float)
        """

    @abstractmethod
    def get_model_name(self) -> str: ...

    @abstractmethod
    def get_cost_per_1k_tokens(self) -> dict:
        """Returns {"input": float, "output": float} in USD per 1k tokens."""
