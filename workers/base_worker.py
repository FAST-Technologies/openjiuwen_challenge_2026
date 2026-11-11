from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class BaseWorker:
    def __init__(self, llm_client, budget_tracker):
        self.llm_client = llm_client
        self.budget_tracker = budget_tracker

    def save_answer(self, task_path: Path, answer: str):
        answer_file = task_path / "answer.txt"
        with open(answer_file, "w", encoding="utf-8") as f:
            f.write(str(answer).strip())
        logger.info(f"✅ Сохранен ответ: {answer_file}")

    async def solve(self, task: dict) -> bool:
        raise NotImplementedError