import asyncio
import subprocess
import re
import ast
from pathlib import Path
from .base_worker import BaseWorker
import logging

logger = logging.getLogger(__name__)

class CodingWorker(BaseWorker):
    async def solve(self, task: dict) -> bool:
        workspace = task["path"] / "workspace"
        mylib_path = workspace / "mylib.py"
        test_path = workspace / "test_mylib.py"
        
        if not mylib_path.exists() or not test_path.exists():
            return False

        mylib_code = mylib_path.read_text(encoding="utf-8")
        test_code = test_path.read_text(encoding="utf-8")

        required_funcs = "все исходные функции"
        try:
            tree = ast.parse(mylib_code)
            func_names = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
            if func_names:
                required_funcs = ", ".join(func_names)
        except Exception:
            pass

        prompt = f"""В файле mylib.py есть ошибки, из-за которых падают тесты.
КРИТИЧЕСКИЕ ТРЕБОВАНИЯ:
1. Верни ПОЛНЫЙ, валидный Python-код файла mylib.py.
2. ТЫ ОБЯЗАН СОХРАНИТЬ СЛЕДУЮЩИЕ ФУНКЦИИ: {required_funcs}. Не удаляй ни одну из них.
3. СТРОГО СОБЛЮДАЙ ОТСТУПЫ: все `def` должны начинаться с начала строки (0 пробелов). Вложенный код — ровно 4 пробела. Никаких случайных пробелов перед `def`.
4. Никаких объяснений, никаких мыслей вслух. Только код внутри ```python ... ```.

Исходный mylib.py:
{mylib_code}

Тесты (test_mylib.py):
{test_code}"""
        
        raw_response, usage = await self.llm_client.call(prompt, model="qwen3.6-35b-a3b")
        await self.budget_tracker.add_usage(usage["prompt_tokens"], usage["completion_tokens"])
        
        if not raw_response: return False
        
        code_blocks = re.findall(r'```python\s*(.*?)\s*```', raw_response, re.DOTALL | re.IGNORECASE)
        if code_blocks:
            fixed_code = code_blocks[0].strip()
        else:
            fixed_code = raw_response.strip()
            match = re.search(r'(def\s+\w+.*?:)', fixed_code, re.DOTALL)
            if match:
                fixed_code = fixed_code[match.start():].strip()

        for attempt in range(2):
            try:
                mylib_path.write_text(fixed_code, encoding="utf-8")
                
                result = await asyncio.to_thread(
                    subprocess.run, ["python3", "test_mylib.py"], 
                    cwd=workspace, capture_output=True, text=True, timeout=30
                )
                
                if result.returncode == 0:
                    self.save_answer(task["path"], "DONE")
                    return True
                else:
                    logger.warning(f"Coding attempt {attempt+1} failed. Stderr: {result.stderr[:600]}")
                    
                    if attempt == 0:
                        # Специальное указание на IndentationError, если он есть
                        indent_warning = " УБЕДИСЬ, ЧТО ВСЕ `def` НАЧИНАЮТСЯ С НАЧАЛА СТРОКИ (0 пробелов)." if "IndentationError" in result.stderr else ""
                        error_prompt = f"Твой код вызвал ошибку:\n{result.stderr}\n\nИСПРАВЬ ТОЛЬКО ошибку. ВЕРНИ ПОЛНЫЙ КОД mylib.py. СОХРАНИ ФУНКЦИИ: {required_funcs}.{indent_warning} ТОЛЬКО КОД, БЕЗ ОБЪЯСНЕНИЙ."
                        
                        raw_response2, usage2 = await self.llm_client.call(error_prompt, model="qwen3.6-35b-a3b")
                        await self.budget_tracker.add_usage(usage2["prompt_tokens"], usage2["completion_tokens"])
                        
                        code_blocks2 = re.findall(r'```python\s*(.*?)\s*```', raw_response2, re.DOTALL | re.IGNORECASE)
                        fixed_code = code_blocks2[0].strip() if code_blocks2 else raw_response2.strip()
                        
            except Exception as e:
                logger.warning(f"Coding execution exception: {e}")
                break
                
        logger.warning(f"❌ Задача {task['task_id']} не решена (coding).")
        return False