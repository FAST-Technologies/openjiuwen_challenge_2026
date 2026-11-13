# test_llm_client.py
import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fake_llm_server import FakeLLM
from llm_client import LLMClient


async def main():
    srv = await FakeLLM(["hello 42", "second"], reasoning_tokens=100).start()
    try:
        client = LLMClient("sk-test", srv.base_url, concurrency=2, timeout=20)
        text, usage = await client.call("Say 42", model="qwen3.6-35b-a3b", max_tokens=123, system="be brief")
        print("text:", repr(text), "usage:", usage)
        req = srv.requests[0]
        print("model:", req["model"], "max_tokens:", req.get("max_tokens"), "roles:", [m["role"] for m in req["messages"]])
        assert text == "hello 42"
        assert usage["prompt_tokens"] > 0 and usage["completion_tokens"] >= 130, usage  # reasoning учтён как выходные
        assert req["model"] == "qwen3.6-35b-a3b" and req.get("max_tokens") == 123
        print("OK")
    finally:
        await srv.stop()


asyncio.run(main())
