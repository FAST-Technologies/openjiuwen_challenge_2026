# fake_llm_server.py
"""Минимальный OpenAI-совместимый сервер для офлайн-тестов (aiohttp, приходит вместе с openjiuwen)."""
import json

from aiohttp import web


class FakeLLM:
    """script: список ответов. str -> текст; dict {"tool": name, "args": {...}} -> вызов инструмента."""

    def __init__(self, script=(), reasoning_tokens=0, responder=None):
        self.responder = responder
        self.script = list(script)
        self.requests = []
        self.reasoning_tokens = reasoning_tokens
        self._runner = None
        self.base_url = None

    async def _handle(self, request: web.Request):
        body = await request.json()
        self.requests.append(body)
        if self.responder is not None:
            item = self.responder(body)
        else:
            item = self.script.pop(0) if self.script else "NO_SCRIPT"
        prompt_tokens = max(len(json.dumps(body.get("messages", []))) // 4, 1)
        message = {"role": "assistant", "content": item if isinstance(item, str) else None}
        finish = "stop"
        if isinstance(item, dict):
            message["tool_calls"] = [{"id": f"call_{len(self.requests)}", "type": "function",
                                      "function": {"name": item["tool"],
                                                   "arguments": json.dumps(item.get("args", {}))}}]
            finish = "tool_calls"
        completion = 30 + self.reasoning_tokens
        return web.json_response({
            "id": "fake", "object": "chat.completion", "created": 0, "model": body.get("model", ""),
            "choices": [{"index": 0, "message": message, "finish_reason": finish}],
            "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion,
                      "total_tokens": prompt_tokens + completion,
                      "completion_tokens_details": {"reasoning_tokens": self.reasoning_tokens}},
        })

    async def start(self):
        app = web.Application()
        app.router.add_post("/v1/chat/completions", self._handle)
        self._runner = web.AppRunner(app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        self.base_url = f"http://127.0.0.1:{port}/v1"
        return self

    async def stop(self):
        if self._runner:
            await self._runner.cleanup()
