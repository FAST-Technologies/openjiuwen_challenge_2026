# llm_client.py
"""LLM-клиент на базе openJiuwen (openjiuwen.core.foundation.llm.Model).

Все обращения к моделям идут через openJiuwen Model -> OpenAI-совместимый прокси организаторов.
Учёт токенов: 10 входных = 1 выходной; reasoning-токены считаются как выходные.
"""
import asyncio
import logging
import random

from openjiuwen.core.foundation.llm import Model
from openjiuwen.core.foundation.llm.schema.config import ModelClientConfig, ModelRequestConfig

logger = logging.getLogger(__name__)
ZERO = {"prompt_tokens": 0, "completion_tokens": 0}


def usage_from_message(msg, prompt_chars: int = 0, text: str = "") -> dict:
    """Достаёт расход из AssistantMessage.usage_metadata; при его отсутствии оценивает по длине."""
    u = getattr(msg, "usage_metadata", None)
    if u is not None and (u.input_tokens or u.output_tokens or u.total_tokens):
        prompt = int(u.input_tokens or 0)
        completion = max(int(u.output_tokens or 0),
                         int(u.total_tokens or 0) - prompt,
                         int(u.reasoning_tokens or 0))
        return {"prompt_tokens": prompt, "completion_tokens": max(completion, 0)}
    # консервативная оценка (≈3.5 символа на токен)
    return {"prompt_tokens": int(prompt_chars / 3.5), "completion_tokens": int(len(text) / 3.5) + 1}


class BudgetedModel(Model):
    """Model, который сам записывает расход каждого вызова (используется ReAct-агентом openJiuwen)."""

    def __init__(self, client_config, request_config, tracker, cap_cost: float):
        super().__init__(client_config, request_config)
        self._tracker = tracker
        self.cap_cost = cap_cost
        self.spent = 0.0

    async def invoke(self, messages, **kwargs):
        if self.spent >= self.cap_cost:
            raise RuntimeError(f"escalation cap reached ({self.spent:.0f}/{self.cap_cost:.0f})")
        chars = len(str(messages))
        msg = await super().invoke(messages, **kwargs)
        usage = usage_from_message(msg, chars, str(getattr(msg, "content", "") or ""))
        await self._tracker.add_usage(usage["prompt_tokens"], usage["completion_tokens"])
        self.spent += usage["completion_tokens"] + usage["prompt_tokens"] / 10.0
        return msg


class LLMClient:
    def __init__(self, api_key: str, base_url: str, concurrency: int = 3, timeout: float = 120.0):
        self.api_key, self.base_url, self.timeout = api_key, base_url, timeout
        self._client_cfg = ModelClientConfig(client_provider="OpenAI", api_key=api_key,
                                             api_base=base_url, timeout=timeout)
        self._models: dict[str, Model] = {}
        self._semaphore = asyncio.Semaphore(concurrency)

    def _model(self, name: str) -> Model:
        if name not in self._models:
            self._models[name] = Model(self._client_cfg, ModelRequestConfig(model_name=name))
        return self._models[name]

    def budgeted_model(self, name: str, tracker, cap_cost: float, max_tokens: int = 700) -> BudgetedModel:
        return BudgetedModel(self._client_cfg,
                             ModelRequestConfig(model_name=name, temperature=0.1, max_tokens=max_tokens),
                             tracker, cap_cost)

    async def call(self, prompt: str, model: str = "qwen3.6-35b-a3b-noreason", max_tokens: int = 800,
                   system: str | None = None, max_retries: int = 3) -> tuple[str, dict]:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        chars = sum(len(m["content"]) for m in messages)

        async with self._semaphore:
            for attempt in range(max_retries):
                try:
                    msg = await self._model(model).invoke(messages, temperature=0.1, max_tokens=max_tokens,
                                                          timeout=self.timeout)
                    text = str(getattr(msg, "content", "") or "").strip()
                    return text, usage_from_message(msg, chars, text)
                except Exception as e:  # ошибки openJiuwen/сети/429
                    delay = min(2 ** attempt * 2, 20) + random.random()
                    logger.warning(f"LLM error ({model}, попытка {attempt + 1}/{max_retries}): "
                                   f"{type(e).__name__}: {str(e)[:200]}; повтор через {delay:.1f}с")
                    await asyncio.sleep(delay)
        return "", dict(ZERO)