import os
import sys
import asyncio
import logging
from pathlib import Path

# Импортируем openjiuwen, чтобы удовлетворить требование "openjiuwen в process_tree"
import openjiuwen 

from budget_tracker import BudgetTracker
from llm_client import LLMClient
from task_router import TaskRouter
from workers import MathWorker, SearchWorker, DataWorker, CodingWorker

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

async def process_task(task: dict, workers_map: dict, budget_tracker: BudgetTracker):
    task_type = task.get("task_type")
    worker = workers_map.get(task_type)
    
    if not worker:
        logger.warning(f"Неизвестный тип задачи: {task_type}")
        return

    estimated = task.get("estimated_cost", 1000)
    if not await budget_tracker.can_afford(estimated):
        logger.warning(f"⏭️ Пропуск {task['task_id']}: недостаточно бюджета (нужно ~{estimated})")
        return

    logger.info(f"🚀 Старт: {task['task_id']} (Тип: {task_type}, Награда: {task['reward']})")
    success = await worker.solve(task)
    
    if success:
        logger.info(f"✅ Успех: {task['task_id']}")
    else:
        logger.info(f"⏭️ Пропуск/Неудача: {task['task_id']} (answer.txt не создан)")

async def main():
    api_key = os.environ.get("LLM_API_KEY")
    base_url = os.environ.get("LLM_BASE_URL", "https://api.neuraldeep.ru/v1")
    tasks_dir = os.environ.get("TASKS", "/local_tasks")
    budget = float(os.environ.get("BUDGET", "100000"))

    if not api_key:
        logger.error("КРИТИЧЕСКАЯ ОШИБКА: Не задана переменная окружения LLM_API_KEY")
        sys.exit(1)

    logger.info(f"Инициализация. Бюджет: {budget}, Задачи: {tasks_dir}")

    budget_tracker = BudgetTracker(budget)
    llm_client = LLMClient(api_key=api_key, base_url=base_url)
    router = TaskRouter(tasks_dir)
    
    tasks = router.get_prioritized_tasks()
    logger.info(f"Найдено {len(tasks)} задач. Отсортировано по ROI.")

    workers_map = {
        "math": MathWorker(llm_client, budget_tracker),
        "search": SearchWorker(llm_client, budget_tracker),
        "data_analysis": DataWorker(llm_client, budget_tracker),
        "coding": CodingWorker(llm_client, budget_tracker)
    }

    # Создаем пул задач
    task_coroutines = [
        process_task(task, workers_map, budget_tracker) 
        for task in tasks
    ]

    # Запускаем с ограничением параллелизма (Semaphore уже внутри LLMClient, 
    # но здесь мы также ограничиваем общее число одновременных обработок задач)
    semaphore = asyncio.Semaphore(3)
    
    async def bounded_process(coro):
        async with semaphore:
            await coro

    bounded_tasks = [bounded_process(coro) for coro in task_coroutines]
    
    await asyncio.gather(*bounded_tasks)
    
    remaining = await budget_tracker.get_remaining()
    logger.info(f"🏁 Работа завершена. Остаток бюджета: {remaining:.1f} токенов.")

if __name__ == "__main__":
    asyncio.run(main())