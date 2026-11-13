# escalation.py
"""Эскалация: ReAct-агент openJiuwen с инструментами, ограниченными каталогом задачи.

Включается только когда дешёвые пути (детерминированный, одиночный LLM-вызов) не сработали.
Расход каждого вызова модели пишется в BudgetTracker (BudgetedModel), на задачу действует жёсткий потолок.
"""
import asyncio
import logging
import subprocess
import sys
from pathlib import Path

from openjiuwen.core.foundation.tool import LocalFunction, ToolCard
from openjiuwen.core.single_agent import AgentCard, ReActAgent, ReActAgentConfig

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You solve one small task inside a working directory using the provided tools. "
    "Be efficient: few steps, no long explanations. Never edit test files. "
    "When finished, reply with ONLY the final answer (for coding tasks: the word DONE)."
)
MAX_OUT = 1500


def _schema(**props):
    return {"type": "object", "properties": {k: {"type": "string", "description": v} for k, v in props.items()},
            "required": list(props), "additionalProperties": False}


def build_tools(cwd: Path) -> list[LocalFunction]:
    root = cwd.resolve()

    def safe(rel: str) -> Path:
        p = (root / (rel or ".")).resolve()
        if p != root and root not in p.parents:
            raise ValueError("path outside the task directory")
        return p

    def list_dir(path: str = ".") -> str:
        try:
            p = safe(path)
            return "\n".join(sorted(x.name + ("/" if x.is_dir() else "") for x in p.iterdir()))[:MAX_OUT]
        except Exception as e:
            return f"error: {e}"

    def read_file(path: str) -> str:
        try:
            return safe(path).read_text(encoding="utf-8", errors="replace")[:4000]
        except Exception as e:
            return f"error: {e}"

    def write_file(path: str, content: str) -> str:
        try:
            p = safe(path)
            if p.name.startswith("test_") or p.name == "answer.txt":
                return "error: this file must not be modified"
            p.write_text(content, encoding="utf-8")
            return "ok"
        except Exception as e:
            return f"error: {e}"

    def _run(argv: list[str]) -> str:
        try:
            r = subprocess.run(argv, cwd=str(root), capture_output=True, text=True, timeout=20)
            return f"exit={r.returncode}\n{(r.stdout + r.stderr)[-MAX_OUT:]}"
        except subprocess.TimeoutExpired:
            return "error: timeout"

    def run_python(code: str) -> str:
        return _run([sys.executable, "-B", "-c", code])

    def run_file(path: str) -> str:
        try:
            return _run([sys.executable, "-B", str(safe(path))])
        except Exception as e:
            return f"error: {e}"

    specs = [
        ("list_dir", "List files in a directory (relative path).", _schema(path="relative directory"), list_dir),
        ("read_file", "Read a text file (relative path).", _schema(path="relative file path"), read_file),
        ("write_file", "Create/overwrite a file (not tests).",
         _schema(path="relative file path", content="full file content"), write_file),
        ("run_python", "Run a Python 3 snippet in the working directory; returns exit code and output.",
         _schema(code="python code"), run_python),
        ("run_file", "Run a Python file in the working directory (e.g. a test file).",
         _schema(path="relative file path"), run_file),
    ]
    return [LocalFunction(card=ToolCard(name=n, description=d, input_params=s), func=f) for n, d, s, f in specs]


class Escalator:
    def __init__(self, llm_client, tracker, model_name: str = "qwen3.6-35b-a3b",
                 cap_cost: float = 6000.0, max_iterations: int = 6, max_tokens: int = 900,
                 min_remaining_ratio: float = 0.25):
        self.llm_client, self.tracker = llm_client, tracker
        self.model_name, self.cap_cost = model_name, cap_cost
        self.max_iterations, self.max_tokens = max_iterations, max_tokens
        self.min_remaining_ratio = min_remaining_ratio

    async def run(self, task: dict, goal: str) -> tuple[str | None, float]:
        """-> (финальный ответ агента | None, потраченные токены). Исключения наружу не выходят."""
        if await self.tracker.get_remaining() < self.tracker.max_budget * self.min_remaining_ratio:
            logger.info(f"{task['task_id']}: эскалация пропущена — мало бюджета")
            return None, 0.0
        ws = task["path"] / "workspace"
        cwd = ws if ws.exists() else task["path"]
        model = self.llm_client.budgeted_model(self.model_name, self.tracker, self.cap_cost, self.max_tokens)
        try:
            agent = ReActAgent(AgentCard(name=f"solver_{task['task_id']}", description="task solver"))
            cfg = (ReActAgentConfig()
                   .configure_model_client("OpenAI", self.llm_client.api_key, self.llm_client.base_url, self.model_name)
                   .configure_prompt_template([{"role": "system", "content": SYSTEM_PROMPT}])
                   .configure_max_iterations(self.max_iterations))
            agent.configure(cfg)
            agent.set_llm(model)
            for tool in build_tools(cwd):
                agent.ability_manager.add_ability(tool.card, tool)
            result = await asyncio.wait_for(
                agent.invoke({"query": goal, "conversation_id": f"esc_{task['task_id']}"}), timeout=240)
        except Exception as e:
            logger.warning(f"{task['task_id']}: эскалация не удалась: {type(e).__name__}: {str(e)[:200]}")
            return None, model.spent
        out = result.get("output") if isinstance(result, dict) else str(result)
        ok = isinstance(result, dict) and result.get("result_type") == "answer"
        return (str(out).strip() if ok and out else None), model.spent
