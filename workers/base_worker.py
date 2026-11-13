# workers/base_worker.py
import asyncio
import json
import logging
import os
import re
import subprocess
import sys
import textwrap
import time
from pathlib import Path

logger = logging.getLogger(__name__)

TRACE_FILE = os.environ.get("TRACE_FILE", "/tmp/trajectories.jsonl")
FENCE = re.compile(r"```[A-Za-z0-9_+-]*[ \t]*\n(.*?)```", re.DOTALL)


def split_sentences(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.?!])\s+", (text or "").strip()) if s]


def extract_code(text: str) -> str:
    """Достаёт код из ```-блока (или берёт весь текст) и снимает общий отступ."""
    m = FENCE.search(text or "")
    code = m.group(1) if m else (text or "")
    return textwrap.dedent(code).strip("\n").rstrip() + "\n"


def normalize_answer(raw: str, check: str | None) -> str | None:
    raw = (raw or "").strip()
    if check == "exact_int":
        nums = re.findall(r"-?\d+(?:\.\d+)?", raw.replace(",", ""))
        if not nums:
            return None
        v = float(nums[-1])
        return str(int(v)) if v == int(v) else None
    raw = raw.strip().strip("\"'").strip()
    return raw or None


async def run_py(code: str, cwd: Path, timeout: float = 20) -> tuple[int, str, str]:
    """Запуск python-кода без shell (аргументы списком)."""
    def _run():
        try:
            p = subprocess.run([sys.executable, "-B", "-c", code], cwd=str(cwd),
                               capture_output=True, text=True, timeout=timeout)
            return p.returncode, p.stdout, p.stderr
        except subprocess.TimeoutExpired:
            return 124, "", "timeout"
    return await asyncio.to_thread(_run)


class BaseWorker:
    def __init__(self, llm_client, budget_tracker):
        self.llm_client = llm_client
        self.budget_tracker = budget_tracker
        self.escalator = None  # openJiuwen ReAct-агент (escalation.Escalator), выставляется в run_agent
        self._tokens: dict[str, float] = {}

    def estimate(self, task: dict) -> tuple[str, float]:
        """Оценка до запуска, без обращения к LLM: ('free'|'llm'|'unsolvable', ожидаемые токены)."""
        return "llm", 1000.0

    async def escalate(self, task: dict, goal: str) -> str | None:
        """Последняя попытка: ReAct-агент openJiuwen с инструментами. Расход уже учтён в BudgetTracker."""
        if self.escalator is None:
            return None
        answer, spent = await self.escalator.run(task, goal)
        self._tokens[task["task_id"]] = self._tokens.get(task["task_id"], 0.0) + spent
        return answer

    # ---------- файлы ----------
    def save_answer(self, task_path: Path, answer: str):
        answer_file = task_path / "answer.txt"
        with open(answer_file, "w", encoding="utf-8") as f:
            f.write(str(answer).strip())
        logger.info(f"✅ Сохранён ответ: {answer_file}")

    def clear_stale_answer(self, task: dict):
        try:
            (task["path"] / "answer.txt").unlink()
        except FileNotFoundError:
            pass
        except OSError as e:
            logger.warning(f"Не удалось удалить старый answer.txt: {e}")

    # ---------- LLM ----------
    async def ask(self, task: dict, prompt: str, model: str, max_tokens: int,
                  system: str | None = None) -> str:
        text, usage = await self.llm_client.call(prompt, model=model,
                                                 max_tokens=max_tokens, system=system)
        await self.budget_tracker.add_usage(usage["prompt_tokens"], usage["completion_tokens"])
        cost = usage["completion_tokens"] + usage["prompt_tokens"] / 10.0
        self._tokens[task["task_id"]] = self._tokens.get(task["task_id"], 0.0) + cost
        return text or ""

    # ---------- траектории (для README) ----------
    def trace(self, task: dict, method: str, ok: bool, note: str = ""):
        rec = {"ts": round(time.time(), 1), "task_id": task["task_id"],
               "type": task.get("task_type"), "reward": task.get("reward"),
               "method": method, "ok": ok, "tokens": round(self._tokens.get(task["task_id"], 0.0), 1),
               "est": round(float(task.get("estimated_cost", 0.0)), 1), "plan": task.get("plan"),
               "note": note[:300]}
        logger.info(f"TRACE {json.dumps(rec, ensure_ascii=False)}")
        try:
            with open(TRACE_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except OSError:
            pass

    async def solve(self, task: dict) -> bool:
        raise NotImplementedError