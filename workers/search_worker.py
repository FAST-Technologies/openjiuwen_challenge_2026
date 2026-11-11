import os
from pathlib import Path
from .base_worker import BaseWorker
import logging

logger = logging.getLogger(__name__)

class SearchWorker(BaseWorker):
    async def solve(self, task: dict) -> bool:
        workspace = task["path"] / "workspace"
        kb_dir = workspace / "kb"
        
        if not kb_dir.exists():
            logger.warning(f"Search failed: directory kb/ not found in {workspace}")
            return False
            
        # 1. Читаем все .md файлы из kb/ локально (это мгновенно и надежно)
        context = ""
        try:
            for filename in os.listdir(kb_dir):
                if filename.endswith(".md"):
                    filepath = kb_dir / filename
                    content = filepath.read_text(encoding="utf-8")
                    context += f"\n--- Файл: {filename} ---\n{content}\n"
        except Exception as e:
            logger.warning(f"Search failed to read kb/: {e}")
            return False
            
        # Защита от гигантских файлов (обрезаем, если больше 15000 символов)
        if len(context) > 15000:
            context = context[:15000] + "\n... (текст обрезан из-за длины)"
            
        # 2. Формируем промпт с уже готовым контекстом
        prompt = f"""У тебя есть база знаний из markdown файлов:
{context}

Вопрос: {task['instruction']}

ЗАДАЧА: Найди ответ на вопрос в предоставленном тексте.
ЖЕСТКИЕ ТРЕБОВАНИЯ:
1. Выведи ТОЛЬКО искомое значение (например, год, название города или компании).
2. Никаких объяснений, никаких дополнительных слов, никаких кавычек. Только сам ответ.
3. Если ответа нет в тексте, выведи ТОЛЬКО 'NOT FOUND'.
"""
        
        # 3. Получаем прямой ответ от LLM (без генерации кода!)
        response, usage = await self.llm_client.call(prompt, model="qwen3.6-35b-a3b-noreason")
        await self.budget_tracker.add_usage(usage["prompt_tokens"], usage["completion_tokens"])
        
        if not response:
            return False
            
        # 4. Очищаем ответ от возможных кавычек или пробелов, которые могла добавить LLM
        answer = response.strip().strip('"').strip("'").strip()
        
        if answer and answer.upper() != "NOT FOUND":
            self.save_answer(task["path"], answer)
            return True
        else:
            logger.warning(f"Search failed: LLM returned empty or NOT FOUND. Response: {response}")
            return False