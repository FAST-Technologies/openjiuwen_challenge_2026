# budget_tracker.py
import asyncio
import logging

logger = logging.getLogger(__name__)

class BudgetTracker:
    def __init__(self, max_budget: float):
        self.max_budget = max_budget
        self.spent = 0.0
        self._lock = asyncio.Lock()

    async def add_usage(self, prompt_tokens: int, completion_tokens: int) -> float:
        async with self._lock:
            cost = completion_tokens + (prompt_tokens / 10.0)
            self.spent += cost
            logger.info(f"💰 Расход: +{cost:.1f} (Всего: {self.spent:.1f} / {self.max_budget})")
            return self.spent

    async def can_afford(self, estimated_cost: float) -> bool:
        async with self._lock:
            return (self.spent + estimated_cost) <= self.max_budget

    async def get_remaining(self) -> float:
        async with self._lock:
            return self.max_budget - self.spent