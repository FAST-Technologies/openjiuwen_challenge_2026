import os
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class TaskRouter:
    def __init__(self, tasks_dir: str):
        self.tasks_dir = Path(tasks_dir)

    def get_prioritized_tasks(self) -> list[dict]:
        tasks = []
        for item in self.tasks_dir.iterdir():
            if item.is_dir():
                task_file = item / "task.json"
                if task_file.exists():
                    with open(task_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        data["path"] = item
                        data["estimated_cost"] = self._estimate_cost(item, data)
                        tasks.append(data)
        
        # Сортировка по ROI: reward / estimated_cost (по убыванию)
        tasks.sort(key=lambda x: x["reward"] / max(x["estimated_cost"], 1), reverse=True)
        return tasks

    def _estimate_cost(self, task_path: Path, data: dict) -> int:
        task_type = data.get("task_type", "unknown")
        base_cost = {"math": 300, "search": 400, "data_analysis": 800, "coding": 2000}.get(task_type, 1000)
        
        # Простые эвристики для корректировки
        if task_type == "coding":
            mylib = task_path / "workspace" / "mylib.py"
            if mylib.exists():
                base_cost += len(mylib.read_text(encoding="utf-8")) // 2
        elif task_type == "data_analysis":
            csv_file = task_path / "workspace" / "sales.csv"
            if csv_file.exists():
                with open(csv_file, "r", encoding="utf-8") as f:
                    lines = sum(1 for _ in f)
                    base_cost += lines // 10 # +10 токенов на каждые 100 строк
                    
        return base_cost