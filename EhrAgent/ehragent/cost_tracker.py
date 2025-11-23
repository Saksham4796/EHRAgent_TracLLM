import threading
from typing import Dict, Tuple

import openai
from langchain_community.callbacks.openai_info import get_openai_token_cost_for_model

# Pricing table (per 1K tokens) based on current OpenAI pricing.
# Keep the keys lowercase for easier matching.
PRICING_PER_1K = {
    # GPT-4o family
    "gpt-4o": {"prompt": 0.0025, "completion": 0.010},
    "gpt-4o-2024-05-13": {"prompt": 0.005, "completion": 0.015},
    "gpt-4o-2024-08-06": {"prompt": 0.0025, "completion": 0.010},
    "chatgpt-4o-latest": {"prompt": 0.005, "completion": 0.015},
    # GPT-4o-mini family
    "gpt-4o-mini": {"prompt": 0.00015, "completion": 0.0006},
    "gpt-4o-mini-2024-07-18": {"prompt": 0.00015, "completion": 0.0006},
    # o1 family
    "o1-preview": {"prompt": 0.015, "completion": 0.060},
    "o1-preview-2024-09-12": {"prompt": 0.015, "completion": 0.060},
    "o1-mini": {"prompt": 0.003, "completion": 0.012},
    "o1-mini-2024-09-12": {"prompt": 0.003, "completion": 0.012},
    # GPT-4 Turbo family
    "gpt-4-turbo": {"prompt": 0.010, "completion": 0.030},
    "gpt-4-turbo-2024-04-09": {"prompt": 0.010, "completion": 0.030},
    "gpt-4-0125-preview": {"prompt": 0.010, "completion": 0.030},
    "gpt-4-1106-preview": {"prompt": 0.010, "completion": 0.030},
    "gpt-4-vision-preview": {"prompt": 0.010, "completion": 0.030},
    # GPT-4 family
    "gpt-4": {"prompt": 0.030, "completion": 0.060},
    "gpt-4-0613": {"prompt": 0.030, "completion": 0.060},
    "gpt-4-32k": {"prompt": 0.060, "completion": 0.120},
    # GPT-3.5 (common fallback)
    "gpt-3.5-turbo": {"prompt": 0.0005, "completion": 0.0015},
    "gpt-3.5-turbo-0125": {"prompt": 0.0005, "completion": 0.0015},
    "gpt-3.5-turbo-1106": {"prompt": 0.001, "completion": 0.002},
    "gpt-3.5-turbo-instruct": {"prompt": 0.0015, "completion": 0.002},
}


class LangChainOpenAICostTracker:
    """
    Global OpenAI cost tracker that hooks the OpenAI client and uses
    LangChain's pricing helper to compute spend.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_cost = 0.0
        self._patched = False
        self._orig_chat_create = None
        self._orig_completion_create = None

    def _capture_usage(self, model: str, usage) -> None:
        if not usage:
            return
        prompt_tokens = getattr(usage, "prompt_tokens", None) or getattr(usage, "prompt_tokens", 0)
        completion_tokens = getattr(usage, "completion_tokens", None) or getattr(usage, "completion_tokens", 0)
        if isinstance(prompt_tokens, dict):
            prompt_tokens = prompt_tokens.get("prompt_tokens", 0)
        if isinstance(completion_tokens, dict):
            completion_tokens = completion_tokens.get("completion_tokens", 0)

        prompt_tokens = prompt_tokens or 0
        completion_tokens = completion_tokens or 0

        model_key = (model or "").lower()
        if model_key in PRICING_PER_1K:
            pricing = PRICING_PER_1K[model_key]
            cost = (prompt_tokens / 1000.0) * pricing["prompt"] + (completion_tokens / 1000.0) * pricing["completion"]
        else:
            # Fallback to LangChain's helper if we don't have a price entry
            try:
                cost = get_openai_token_cost_for_model(
                    model or "",
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                )
            except Exception:
                cost = 0.0

        with self._lock:
            self.prompt_tokens += prompt_tokens
            self.completion_tokens += completion_tokens
            self.total_cost += cost

    def patch_openai(self) -> None:
        """Monkey-patch the OpenAI client so every call is costed."""
        if self._patched:
            return

        try:
            from openai.resources.chat.completions import Completions as ChatCompletions
        except Exception:
            ChatCompletions = None

        try:
            from openai.resources.completions import Completions as LegacyCompletions
        except Exception:
            LegacyCompletions = None

        if ChatCompletions:
            self._orig_chat_create = ChatCompletions.create

            def patched_chat_create(completions_self, *args, **kwargs):
                response = self._orig_chat_create(completions_self, *args, **kwargs)
                model = kwargs.get("model") or getattr(response, "model", None) or ""
                usage = getattr(response, "usage", None)
                self._capture_usage(model, usage)
                return response

            ChatCompletions.create = patched_chat_create

        if LegacyCompletions:
            self._orig_completion_create = LegacyCompletions.create

            def patched_completion_create(completions_self, *args, **kwargs):
                response = self._orig_completion_create(completions_self, *args, **kwargs)
                model = kwargs.get("model") or getattr(response, "model", None) or ""
                usage = getattr(response, "usage", None)
                self._capture_usage(model, usage)
                return response

            LegacyCompletions.create = patched_completion_create

        self._patched = True

    def snapshot(self) -> Tuple[int, int, float]:
        with self._lock:
            return (self.prompt_tokens, self.completion_tokens, self.total_cost)

    def delta_since(self, snapshot: Tuple[int, int, float]) -> Dict[str, float]:
        with self._lock:
            prompt_delta = self.prompt_tokens - snapshot[0]
            completion_delta = self.completion_tokens - snapshot[1]
            cost_delta = self.total_cost - snapshot[2]
        return {
            "prompt_tokens": prompt_delta,
            "completion_tokens": completion_delta,
            "total_tokens": prompt_delta + completion_delta,
            "cost_usd": cost_delta,
        }

    def summary(self) -> Dict[str, float]:
        with self._lock:
            return {
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "total_tokens": self.prompt_tokens + self.completion_tokens,
                "total_cost_usd": self.total_cost,
            }
