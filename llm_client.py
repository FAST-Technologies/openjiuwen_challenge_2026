import os
import asyncio
import logging
from openai import AsyncOpenAI, APIError, RateLimitError

logger = logging.getLogger(__name__)

class LLMClient:
    def __init__(self, api_key: str, base_url: str):
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self._semaphore = asyncio.Semaphore(3) # Макс 3 параллельных запроса

    async def call(self, prompt: str, model: str = "qwen3.6-35b-a3b-noreason", max_retries: int = 3) -> tuple[str, dict]:
        async with self._semaphore:
            for attempt in range(max_retries):
                try:
                    response = await self.client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.1,
                        timeout=60
                    )
                    usage = {
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens
                    }
                    return response.choices[0].message.content.strip(), usage
                except RateLimitError as e:
                    wait_time = (2 ** attempt) + (asyncio.get_event_loop().time() % 1)
                    logger.warning(f"Rate limit. Ждем {wait_time:.1f}с...")
                    await asyncio.sleep(wait_time)
                except APIError as e:
                    logger.error(f"API Error (attempt {attempt+1}/{model}): {e}")
                    if attempt == max_retries - 1:
                        return None, {"prompt_tokens": 0, "completion_tokens": 0}
        return None, {"prompt_tokens": 0, "completion_tokens": 0}