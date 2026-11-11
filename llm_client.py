# llm_client.py
import asyncio
import logging
import random
from openai import AsyncOpenAI, APIError, APIConnectionError, APITimeoutError, RateLimitError

logger = logging.getLogger(__name__)
ZERO = {"prompt_tokens": 0, "completion_tokens": 0}

class LLMClient:
    def __init__(self, api_key: str, base_url: str, concurrency: int = 3):
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url,
                                  max_retries=0, timeout=120)
        self._semaphore = asyncio.Semaphore(concurrency)

    async def call(self, prompt: str, model: str = "qwen3.6-35b-a3b-noreason",
                   max_tokens: int = 800, system: str | None = None,
                   max_retries: int = 3) -> tuple[str, dict]:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        async with self._semaphore:
            for attempt in range(max_retries):
                try:
                    r = await self.client.chat.completions.create(
                        model=model, messages=messages,
                        temperature=0.1, max_tokens=max_tokens)
                    u = r.usage
                    usage = {
                        "prompt_tokens": (u.prompt_tokens if u else 0) or 0,
                        "completion_tokens": (u.completion_tokens if u else 0) or 0,
                    }
                    text = (r.choices[0].message.content or "").strip()
                    if r.choices[0].finish_reason == "length":
                        logger.warning(f"{model}: ответ обрезан по max_tokens={max_tokens}")
                    return text, usage
                except RateLimitError:
                    delay = min(2 ** attempt * 2, 20) + random.random()
                    logger.warning(f"429, ждём {delay:.1f}с")
                    await asyncio.sleep(delay)
                except (APITimeoutError, APIConnectionError, APIError) as e:
                    logger.error(f"API error ({model}, попытка {attempt + 1}): {e}")
                    await asyncio.sleep(1 + random.random())
        return "", dict(ZERO)