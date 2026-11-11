import asyncio
import subprocess
from pathlib import Path
from .base_worker import BaseWorker
import logging

logger = logging.getLogger(__name__)

class DataWorker(BaseWorker):
    async def solve(self, task: dict) -> bool:
        workspace = task["path"] / "workspace"
        prompt = f"""Напиши Python-скрипт для решения задачи.
Задача: {task['instruction']}
Требования:
1. Используй стандартный модуль `csv`.
2. Читай файл строго так: open('sales.csv', 'r') (имя файла в кавычках!).
3. Выведи ТОЛЬКО итоговое число или строку через print().
4. Не пиши никаких объяснений, только сам код."""
        
        code, usage = await self.llm_client.call(prompt, model="qwen3.6-35b-a3b-noreason")
        await self.budget_tracker.add_usage(usage["prompt_tokens"], usage["completion_tokens"])
        
        if not code: return False
        code = code.replace("```python", "").replace("```", "").strip()

        script_path = workspace / "solve_data.py"
        try:
            # 1. Записываем код в файл (никаких проблем с кавычками bash)
            script_path.write_text(code, encoding="utf-8")
            
            # 2. Запускаем как обычный скрипт
            result = await asyncio.to_thread(
                subprocess.run, ["python3", "solve_data.py"], 
                cwd=workspace, capture_output=True, text=True, timeout=20
            )
            
            if result.returncode == 0 and result.stdout.strip():
                self.save_answer(task["path"], result.stdout.strip())
                return True
            else:
                logger.warning(f"Data failed. Stderr: {result.stderr}")
        except Exception as e:
            logger.warning(f"Data execution exception: {e}")
        finally:
            # 3. Чистим за собой временный файл
            if script_path.exists():
                script_path.unlink()
        return False