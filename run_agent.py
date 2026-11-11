# run_agent.py
import argparse
import asyncio
import logging
import os
import sys
import time

try:  # харнесс на openJiuwen; сбой импорта не должен ронять прогон
    import openjiuwen  # noqa: F401
except Exception as e:  # pragma: no cover
    print(f"openjiuwen import failed: {e}", file=sys.stderr)

from budget_tracker import BudgetTracker
from llm_client import LLMClient
from task_router import TaskRouter
from workers import MathWorker, SearchWorker, DataWorker, CodingWorker

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

SAFETY_MARGIN = 0.95          # прокси обрывает ответы резко, оставляем запас
TASK_TIMEOUT = 300            # секунд на одну задачу
FINISH_BEFORE_END = 300       # не начинаем новые задачи за 5 мин до лимита


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tasks-dir", default=os.environ.get("TASKS", "/data/tasks"))
    p.add_argument("--concurrency", type=int,
                   default=int(os.environ.get("CONCURRENCY", "3")))
    p.add_argument("--time-limit", type=float,
                   default=float(os.environ.get("TIME_LIMIT_SEC", "3600")))
    return p.parse_args()


async def process_task(task, workers_map, tracker, deadline):
    worker = workers_map.get(task.get("task_type"))
    if not worker:
        logger.warning(f"Неизвестный тип задачи: {task.get('task_type')}")
        return
    estimated = task.get("estimated_cost", 1000)
    if not await tracker.can_afford(estimated):
        logger.warning(f"⏭️ Пропуск {task['task_id']}: бюджет (нужно ~{estimated})")
        return
    timeout = min(TASK_TIMEOUT, deadline - time.monotonic())
    if timeout <= 0:
        return
    logger.info(f"🚀 Старт: {task['task_id']} ({task['task_type']}, reward={task['reward']})")
    try:
        ok = await asyncio.wait_for(worker.solve(task), timeout=timeout)
    except Exception as e:  # одна задача не должна ронять остальные
        logger.warning(f"💥 {task['task_id']}: {type(e).__name__}: {e}")
        return
    logger.info(f"{'✅' if ok else '⏭️'} {task['task_id']}: {'решена' if ok else 'без ответа'}")

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
    tasks = TaskRouter(args.tasks_dir).get_prioritized_tasks()
    logger.info(f"Задач: {len(tasks)}, бюджет: {budget:.0f}, dir: {args.tasks_dir}")

    workers_map = {
        "math": MathWorker(llm, tracker),
        "search": SearchWorker(llm, tracker),
        "data_analysis": DataWorker(llm, tracker),
        "coding": CodingWorker(llm, tracker),
    }
    queue = asyncio.Queue()
    for t in tasks:
        queue.put_nowait(t)

    await asyncio.gather(*[consumer(queue, workers_map, tracker, deadline)
                           for _ in range(args.concurrency)])
    logger.info(f"🏁 Готово. Остаток бюджета: {await tracker.get_remaining():.1f}")

if __name__ == "__main__":
    asyncio.run(main())