# workers/data_worker.py
import csv
import logging
import re

from .base_worker import BaseWorker, extract_code, normalize_answer, run_py, split_sentences

logger = logging.getLogger(__name__)

P_RANGE = re.compile(r"how many units of product '([^']+)' were sold in months? (\d+)(?: through| to| and|-) (\d+)", re.I)
P_ONE = re.compile(r"how many units of product '([^']+)' were sold in month (\d+)\b", re.I)
TAIL = re.compile(r"^(use |work only|when you are done)", re.I)


def _find_csv(workspace, instruction):
    m = re.search(r"file ([\w.\-]+\.csv)", instruction)
    if m and (workspace / m.group(1)).exists():
        return workspace / m.group(1)
    files = sorted(workspace.glob("*.csv"))
    return files[0] if len(files) == 1 else None


def _required_columns(instruction):
    m = re.search(r"columns\s+([A-Za-z_][\w,]*)", instruction)
    return [c for c in m.group(1).split(",") if c] if m else []

def deterministic_answer(instr: str, rows: list[dict]) -> int | None:
    m = P_RANGE.search(instr)
    if m:
        product, lo, hi = m.group(1).lower(), int(m.group(2)), int(m.group(3))
    else:
        m = P_ONE.search(instr)
        if not m:
            return None
        product, lo, hi = m.group(1).lower(), int(m.group(2)), int(m.group(2))
    total, bad = 0, 0
    for r in rows:
        try:
            month, units = int(r["month"]), int(r["units"])
        except (KeyError, ValueError, TypeError):
            bad += 1
            continue
        if (r.get("product") or "").strip().lower() == product and lo <= month <= hi:
            total += units
    if bad:
        logger.warning(f"data: пропущено {bad} некорректных строк")
    return total


class DataWorker(BaseWorker):
    def _inspect(self, task: dict):
        """-> ('unsolvable', причина) | ('ok', csv_path, header, rows)."""
        workspace = task["path"] / "workspace"
        csv_path = _find_csv(workspace, task["instruction"]) if workspace.exists() else None
        if csv_path is None:
            return "unsolvable", "csv not found"
        try:
            with open(csv_path, newline="", encoding="utf-8", errors="replace") as f:
                rows = list(csv.DictReader(f))
        except Exception as e:
            return "unsolvable", f"csv unreadable: {e}"
        if not rows:
            return "unsolvable", "empty csv"
        header = list(rows[0].keys())
        missing = [c for c in _required_columns(task["instruction"]) if c not in header]
        if missing:
            return "unsolvable", f"columns missing {missing}"
        return "ok", csv_path, header, rows

    def estimate(self, task: dict) -> tuple[str, float]:
        info = self._inspect(task)
        if info[0] != "ok":
            return "unsolvable", 0.0
        if deterministic_answer(task["instruction"], info[3]) is not None:
            return "free", 0.0
        return "llm", 120 + len(task["instruction"]) / 35 + len(",".join(info[2])) / 35

    async def solve(self, task: dict) -> bool:
        self.clear_stale_answer(task)
        instr, check = task["instruction"], task.get("check")
        info = self._inspect(task)
        if info[0] != "ok":
            self.trace(task, "precheck", False, f"{info[1]} -> unsolvable")
            return False
        _, csv_path, header, rows = info

        # 1) без LLM
        res = deterministic_answer(instr, rows)
        if res is not None:
            self.save_answer(task["path"], str(res))
            self.trace(task, "deterministic", True)
            return True

        # 2) fallback: LLM пишет скрипт, запускаем без записи файлов
        workspace = task["path"] / "workspace"
        question = " ".join(s for s in split_sentences(instr) if not TAIL.match(s))
        sample = "\n".join(",".join(r.values()) for r in rows[:3])
        prompt = (f"Write a Python 3 script using only the stdlib `csv` module. Open the file with open('{csv_path.name}', newline='').\n"
                  f"Columns: {','.join(header)}. Sample rows:\n{sample}\n"
                  f"Task: {question}\nThe script must print ONLY the final answer. Return only code in one ```python block.")
        text = await self.ask(task, prompt, "qwen3.6-35b-a3b-noreason", max_tokens=600)
        rc, out, err = await run_py(extract_code(text), workspace, timeout=20)
        ans = normalize_answer(out, check) if rc == 0 else None
        if ans is None:
            logger.warning(f"data fallback failed rc={rc} err={err[:200]}")
            reply = await self.escalate(task, f"File {csv_path.name} has columns {','.join(header)}. {question} "
                                              "Use run_python to compute it. Reply with ONLY the final answer.")
            ans = normalize_answer(reply, check) if reply else None
            if ans is None:
                self.trace(task, "llm+agent", False, err)
                return False
            self.save_answer(task["path"], ans)
            self.trace(task, "agent", True)
            return True
        self.save_answer(task["path"], ans)
        self.trace(task, "llm", True)
        return True