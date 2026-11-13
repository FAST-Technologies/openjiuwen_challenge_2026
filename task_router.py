# task_router.py
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

MIN_COST = 10.0  # пол для знаменателя ROI, чтобы бесплатные задачи не давали бесконечность


class TaskRouter:
    """Читает задачи, оценивает расход ДО запуска (без обращений к LLM) и упорядочивает по ROI.

    Оценку даёт сам воркер (worker.estimate): 'free' — решается детерминированно (0 токенов),
    'unsolvable' — по данным задачи ответа получить нельзя, 'llm' — нужен вызов модели,
    ожидаемый расход считается по размеру входных данных.
    """

    def __init__(self, tasks_dir: str, workers_map: dict):
        self.tasks_dir = Path(tasks_dir)
        self.workers_map = workers_map

    def load(self) -> list[dict]:
        tasks = []
        for item in sorted(self.tasks_dir.iterdir()):
            task_file = item / "task.json"
            if not (item.is_dir() and task_file.exists()):
                continue
            try:
                data = json.loads(task_file.read_text(encoding="utf-8"))
                data["task_id"] = data.get("task_id") or item.name
                data["reward"] = float(data.get("reward", 1))
                data["instruction"] = data.get("instruction", "")
            except Exception as e:
                logger.warning(f"{item.name}: task.json не прочитан ({e}) — пропуск")
                continue
            data["path"] = item
            tasks.append(data)
        return tasks

    def get_prioritized_tasks(self) -> list[dict]:
        tasks = self.load()
        for t in tasks:
            worker = self.workers_map.get(t.get("task_type"))
            if worker is None:
                t["plan"], t["estimated_cost"] = "unsolvable", 0.0
                continue
            try:
                t["plan"], t["estimated_cost"] = worker.estimate(t)
            except Exception as e:  # оценка не должна ронять запуск
                logger.warning(f"{t['task_id']}: ошибка оценки ({type(e).__name__}: {e}) — считаем как llm")
                t["plan"], t["estimated_cost"] = "llm", 1000.0
        solvable = [t for t in tasks if t["plan"] != "unsolvable"]
        skipped = [t for t in tasks if t["plan"] == "unsolvable"]
        solvable.sort(key=lambda t: t["reward"] / max(t["estimated_cost"], MIN_COST), reverse=True)
        return solvable + skipped
