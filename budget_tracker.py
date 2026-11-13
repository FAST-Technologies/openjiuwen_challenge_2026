
# budget_tracker.py
import asyncio
import logging
import uuid
from contextvars import ContextVar
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class BudgetTracker:
    def __init__(self, max_budget: float):
        if max_budget < 0:
            raise ValueError("max_budget must be non-negative")

        self.max_budget = float(max_budget)
        self.spent = 0.0

        self._lock = asyncio.Lock()
        self._reservations: dict[str, float] = {}
        self._current_reservation = ContextVar(
            f"budget_reservation_{id(self)}",
            default=None,
        )

    @property
    def reserved(self) -> float:
        """Текущая сумма неиспользованных резервов."""
        return sum(self._reservations.values())

    async def try_reserve(self, estimated_cost: float) -> str | None:
        """
        Атомарно резервирует бюджет.

        Возвращает ID резерва или None, если свободного бюджета недостаточно.
        """
        estimated_cost = float(estimated_cost)

        if estimated_cost < 0:
            raise ValueError("estimated_cost must be non-negative")

        async with self._lock:
            available = self.max_budget - self.spent - self.reserved

            if estimated_cost > available:
                logger.info(
                    "Резерв отклонён: требуется %.1f, доступно %.1f",
                    estimated_cost,
                    max(available, 0.0),
                )
                return None

            reservation_id = uuid.uuid4().hex
            self._reservations[reservation_id] = estimated_cost

            logger.info(
                "Резерв создан: %.1f (занято резервами: %.1f)",
                estimated_cost,
                self.reserved,
            )
            return reservation_id

    @contextmanager
    def use_reservation(self, reservation_id: str):
        """
        Привязывает резерв к текущему async-контексту.

        Все add_usage() внутри этого контекста будут уменьшать
        соответствующий резерв.
        """
        token = self._current_reservation.set(reservation_id)
        try:
            yield
        finally:
            self._current_reservation.reset(token)

    async def release_reservation(self, reservation_id: str) -> float:
        """Освобождает неиспользованный резерв и возвращает его остаток."""
        async with self._lock:
            remaining = self._reservations.pop(reservation_id, 0.0)

            logger.info(
                "Резерв освобождён: %.1f",
                remaining,
            )
            return remaining

    async def add_usage(
        self,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> float:
        """Списывает фактическую стоимость API-вызова."""
        if prompt_tokens < 0 or completion_tokens < 0:
            raise ValueError("Token counts must be non-negative")

        cost = completion_tokens + prompt_tokens / 10.0
        reservation_id = self._current_reservation.get()

        async with self._lock:
            self.spent += cost

            if reservation_id is not None:
                reserved_before = self._reservations.get(reservation_id)

                if reserved_before is not None:
                    consumed = min(cost, reserved_before)
                    remaining = reserved_before - consumed

                    if remaining > 0:
                        self._reservations[reservation_id] = remaining
                    else:
                        self._reservations.pop(reservation_id, None)

                    if cost > reserved_before:
                        logger.warning(
                            "Превышен резерв задачи: расход %.1f, "
                            "остаток резерва %.1f, превышение %.1f",
                            cost,
                            reserved_before,
                            cost - reserved_before,
                        )

            total_committed = self.spent + self.reserved

            if total_committed > self.max_budget:
                logger.error(
                    "Бюджет превышен: потрачено %.1f + резерв %.1f "
                    "> лимит %.1f",
                    self.spent,
                    self.reserved,
                    self.max_budget,
                )

            logger.info(
                "Расход: +%.1f (потрачено: %.1f / %.1f; резервы: %.1f)",
                cost,
                self.spent,
                self.max_budget,
                self.reserved,
            )

            return self.spent

    async def can_afford(self, estimated_cost: float) -> bool:
        """Проверяет доступный бюджет с учётом действующих резервов."""
        async with self._lock:
            return (
                self.spent + self.reserved + estimated_cost
                <= self.max_budget
            )

    async def get_remaining(self) -> float:
        """Бюджет, ещё не потраченный и не зарезервированный."""
        async with self._lock:
            return self.max_budget - self.spent - self.reserved
