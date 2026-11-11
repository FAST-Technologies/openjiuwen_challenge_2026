import asyncio
import subprocess
import re
from .base_worker import BaseWorker
import logging

logger = logging.getLogger(__name__)

class MathWorker(BaseWorker):
    async def solve(self, task: dict) -> bool:
        # Используем task["path"], так как папки workspace у math задач может не быть
        task_dir = task["path"]
        prompt = f"Реши задачу. Верни ТОЛЬКО команду, начинающуюся с 'python3 -c'. Никаких markdown, никаких объяснений, никакого перенаправления вывода в файл (никаких >).\nЗадача: {task['instruction']}"
        
        code, usage = await self.llm_client.call(prompt, model="qwen3.6-35b-a3b-noreason")
        await self.budget_tracker.add_usage(usage["prompt_tokens"], usage["completion_tokens"])
        
        if not code: return False
        
        # Агрессивная очистка от markdown
        code = code.replace("```python", "").replace("```bash", "").replace("```", "").strip()
        
        # ЗАЩИТА: принудительно удаляем любые попытки записи в файл, если LLM их добавила
        code = re.sub(r'\s*>\s*answer\.txt.*', '', code).strip()
        
        # Если LLM забыла добавить python3 -c, добавляем сами
        if not code.startswith("python3"):
            # Экранируем внутренние кавычки, если они есть, для безопасности
            code = f'python3 -c "{code}"'

        try:
            result = await asyncio.to_thread(
                subprocess.run, code, shell=True, cwd=task_dir, capture_output=True, text=True, timeout=15
            )
            if result.returncode == 0 and result.stdout.strip():
                self.save_answer(task["path"], result.stdout.strip())
                return True
            else:
                logger.warning(f"Math failed. Code: {code[:60]}... | Stderr: {result.stderr}")
        except Exception as e:
            logger.warning(f"Math execution exception: {e}")
        return False