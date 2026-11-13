# test_escalation.py
import asyncio
import os
import sys
import tempfile
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from oj_setup import configure_openjiuwen_logging

configure_openjiuwen_logging()

from fake_llm_server import FakeLLM
from llm_client import LLMClient
from escalation import Escalator, build_tools


class Tracker:
    def __init__(self, budget=100000.0):
        self.max_budget, self.spent = budget, 0.0

    async def add_usage(self, p, c):
        self.spent += c + p / 10

    async def get_remaining(self):
        return self.max_budget - self.spent


async def main():
    tmp = Path(tempfile.mkdtemp())
    task_dir = tmp / "t1"
    (task_dir / "workspace").mkdir(parents=True)
    (task_dir / "workspace" / "data.txt").write_text("a=2\nb=40\n")
    task = {"task_id": "t1", "path": task_dir}

    # 1) инструменты: пути ограничены каталогом, тесты менять нельзя
    tools = {t.card.name: t for t in build_tools(task_dir / "workspace")}
    print("tools:", sorted(tools))
    assert (await tools["read_file"].invoke({"path": "../../etc/passwd"})).startswith("error")
    assert (await tools["write_file"].invoke({"path": "test_x.py", "content": "x"})).startswith("error")
    assert (await tools["write_file"].invoke({"path": "ok.py", "content": "print(5)"})) == "ok"
    print("run_file:", (await tools["run_file"].invoke({"path": "ok.py"})).strip().replace("\n", " | "))

    # 2) ReAct через openJiuwen: инструмент -> финальный ответ
    srv = await FakeLLM([{"tool": "read_file", "args": {"path": "data.txt"}},
                         {"tool": "run_python", "args": {"code": "print(2+40)"}},
                         "42"]).start()
    try:
        tr = Tracker()
        llm = LLMClient("k", srv.base_url, timeout=30)
        esc = Escalator(llm, tr, cap_cost=6000, max_iterations=6)
        ans, spent = await esc.run(task, "Compute a+b from data.txt")
        print("answer:", repr(ans), "spent:", round(spent, 1), "tracker:", round(tr.spent, 1), "requests:", len(srv.requests))
        print("system prompt chars:", len(srv.requests[0]["messages"][0]["content"]), "| tools sent:", len(srv.requests[0].get("tools", [])),
              "| max_tokens:", srv.requests[0].get("max_tokens"))
        assert ans == "42" and len(srv.requests) == 3 and tr.spent > 0 and abs(tr.spent - spent) < 1e-6
    finally:
        await srv.stop()

    srv = await FakeLLM([{"tool": "list_dir", "args": {"path": "."}}] * 10, reasoning_tokens=400).start()
    try:
        tr = Tracker()
        esc = Escalator(LLMClient("k", srv.base_url, timeout=30), tr, cap_cost=500, max_iterations=8)
        ans, spent = await esc.run(task, "loop forever")
        print("capped: answer=", ans, "spent=", round(spent, 1), "requests=", len(srv.requests))
        assert ans is None and len(srv.requests) <= 3
    finally:
        await srv.stop()

    # 3) мало бюджета -> эскалация не стартует
    tr = Tracker(1000.0); tr.spent = 900.0
    esc = Escalator(LLMClient("k", "http://127.0.0.1:1/v1"), tr)
    ans, spent = await esc.run(task, "x")
    assert ans is None and spent == 0.0
    print("OK")


asyncio.run(main())
