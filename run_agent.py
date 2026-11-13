# run_agent.py
import argparse
import asyncio
import json
import logging
import os
import sys
import time

from oj_setup import configure_openjiuwen_logging

configure_openjiuwen_logging(os.environ.get("OJ_LOG_LEVEL", "WARNING"))

import openjiuwen  # noqa: E402  харнесс построен на openJiuwen (llm_client.py, escalation.py)

from budget_tracker import BudgetTracker  # noqa: E402
from escalation import Escalator  # noqa: E402
from llm_client import LLMClient  # noqa: E402
from task_router import TaskRouter  # noqa: E402
from workers import MathWorker, SearchWorker, DataWorker, CodingWorker  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger("harness")

SAFETY_MARGIN = 0.95        # прокси обрывает ответы резко, оставляем 5% запаса
TASK_TIMEOUT = 420          # секунд на одну задачу (с эскалацией)
FINISH_BEFORE_END = 300     # новые задачи не стартуем за 5 минут до лимита


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tasks-dir", default=os.environ.get("TASKS", "/data/tasks"))
    p.add_argument("--concurrency", type=int, default=int(os.environ.get("CONCURRENCY", "3")))
    p.add_argument("--time-limit", type=float, default=float(os.environ.get("TIME_LIMIT_SEC", "3600")))
    return p.parse_args()


def write_summary(record: dict):
    """Необязательная сводка в $OUT (файл или каталог). Основной результат — answer.txt рядом с task.json."""
    out = os.environ.get("OUT")
    if not out:
        return
    try:
        path = os.path.join(out, "results.jsonl") if os.path.isdir(out) else out
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError:
        pass


async def process_task(task, workers_map, tracker, deadline):
    tid = task["task_id"]
    if task["plan"] == "unsolvable":
        stale = task["path"] / "answer.txt"
        if stale.exists():
            stale.unlink()
        logger.info(f"⛔ {tid}: невыполнима по данным задачи, пропуск (0 токенов)")
        write_summary({"task_id": tid, "solved": False, "reason": "unsolvable"})
        return
    worker = workers_map[task["task_type"]]
    if not await tracker.can_afford(task["estimated_cost"]):
        logger.warning(f"⏭️ {tid}: недостаточно бюджета (нужно ~{task['estimated_cost']:.0f})")
        return
    timeout = min(TASK_TIMEOUT, deadline - time.monotonic())
    if timeout <= 0:
        return
    logger.info(f"🚀 {tid} ({task['task_type']}, reward={task['reward']:g}, plan={task['plan']}, est={task['estimated_cost']:.0f})")
    try:
        ok = await asyncio.wait_for(worker.solve(task), timeout=timeout)
    except Exception as e:  # одна задача не должна ронять остальные
        logger.warning(f"💥 {tid}: {type(e).__name__}: {e}")
        ok = False
    logger.info(f"{'✅' if ok else '⏭️'} {tid}: {'решена' if ok else 'без ответа'}")
    write_summary({"task_id": tid, "solved": bool(ok)})


async def consumer(queue, workers_map, tracker, deadline):
    while time.monotonic() < deadline - FINISH_BEFORE_END:
        try:
            task = queue.get_nowait()
        except asyncio.QueueEmpty:
            return
        await process_task(task, workers_map, tracker, deadline)


async def main():
    args = parse_args()
    api_key = os.environ.get("LLM_API_KEY")
    base_url = os.environ.get("LLM_BASE_URL", "https://api.neuraldeep.ru/v1")
    budget = float(os.environ.get("BUDGET", "100000"))
    if not api_key:
        logger.error("Не задана переменная окружения LLM_API_KEY")
        sys.exit(1)

    deadline = time.monotonic() + args.time_limit
    tracker = BudgetTracker(budget * SAFETY_MARGIN)
    llm = LLMClient(api_key, base_url, concurrency=args.concurrency)
    escalator = Escalator(llm, tracker, cap_cost=min(6000.0, budget * 0.06))

    workers_map = {
        "math": MathWorker(llm, tracker),
        "search": SearchWorker(llm, tracker),
        "data_analysis": DataWorker(llm, tracker),
        "coding": CodingWorker(llm, tracker),
    }
    for w in workers_map.values():
        w.escalator = escalator

    tasks = TaskRouter(args.tasks_dir, workers_map).get_prioritized_tasks()
    plans = {p: sum(1 for t in tasks if t["plan"] == p) for p in ("free", "llm", "unsolvable")}
    logger.info(f"openJiuwen {openjiuwen.__version__}; задач: {len(tasks)} {plans}; бюджет: {budget:.0f}; dir: {args.tasks_dir}; "
                f"ожидаемый расход: {sum(t['estimated_cost'] for t in tasks):.0f}")
    logger.info(f"Задач: {len(tasks)}, бюджет: {budget:.0f}, dir: {args.tasks_dir}")

    queue = asyncio.Queue()
    for t in tasks:
        queue.put_nowait(t)
    await asyncio.gather(*[consumer(queue, workers_map, tracker, deadline) for _ in range(args.concurrency)])
    logger.info(f"🏁 Готово. Потрачено: {tracker.spent:.1f}, остаток: {await tracker.get_remaining():.1f}")

if __name__ == "__main__":
    asyncio.run(main())