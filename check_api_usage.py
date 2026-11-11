
import asyncio
import json
import os
from openai import AsyncOpenAI


async def main():
    api_key = os.environ["LLM_API_KEY"]
    base_url = os.environ.get(
        "LLM_BASE_URL",
        "https://api.neuraldeep.ru/v1"
    )
    model = os.environ.get(
        "LLM_MODEL",
        "qwen3.6-35b-a3b-noreason"
    )

    client = AsyncOpenAI(
        api_key=api_key,
        base_url=base_url,
        max_retries=0,
        timeout=120,
    )

    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": "Ответь одним словом: OK"
                }
            ],
            max_tokens=32,
            temperature=0,
        )

        print("=== RESPONSE ===")
        print("model:", response.model)
        print("finish_reason:", response.choices[0].finish_reason)
        print("content:", response.choices[0].message.content)

        print("\n=== USAGE ===")
        usage = response.usage

        if usage is None:
            print("usage: None")
        else:
            print("type:", type(usage).__name__)
            print("repr:", repr(usage))
            print(
                "model_dump:",
                json.dumps(
                    usage.model_dump(),
                    ensure_ascii=False,
                    indent=2,
                    default=str,
                ),
            )

    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
