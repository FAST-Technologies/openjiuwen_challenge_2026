import logging
import re

from .base_worker import BaseWorker, extract_code, normalize_answer, run_py, split_sentences

logger = logging.getLogger(__name__)

IGNORE = re.compile(r"^(how many|compute|use |work only|when you are done|write )", re.I)
START = re.compile(r"starts? with (\d+)\b", re.I)

# (regex, функция (n, match) -> новое n); порядок важен
STEPS = [
    (re.compile(r"split equally into (\d+) \w+ and it keeps one share.*rounded up", re.I),
     lambda n, m: -(-n // int(m[1]))),
    (re.compile(r"split equally into (\d+) \w+ and it keeps one share", re.I),
     lambda n, m: n // int(m[1])),
    (re.compile(r"\bdoubled\b", re.I), lambda n, m: n * 2),
    (re.compile(r"\btripled\b", re.I), lambda n, m: n * 3),
    (re.compile(r"ships (\d+) \w+ per day for (\d+) days?", re.I),
     lambda n, m: n - int(m[1]) * int(m[2])),
    (re.compile(r"receives (\d+) deliveries of (\d+) \w+ each", re.I),
     lambda n, m: n + int(m[1]) * int(m[2])),
    (re.compile(r"^(?:it |the warehouse )?(?:ships|sells|loses|removes) (\d+) \w+$", re.I),
     lambda n, m: n - int(m[1])),
    (re.compile(r"^(?:it |the warehouse )?(?:receives|gets|adds) (\d+) \w+$", re.I),
     lambda n, m: n + int(m[1])),
]


def solve_deterministic(instruction: str) -> int | None:
    """Разбирает шаблонные задачи про склад. None — если хоть одно предложение не распознано."""
    n = None
    for s in split_sentences(instruction):
        s = re.sub(r"^then\s+", "", s.strip().rstrip(".?!").strip(), flags=re.I)
        if not s or IGNORE.match(s):
            continue
        m = START.search(s)
        if m:
            if n is not None:
                return None
            n = int(m.group(1))
            continue
        if n is None:
            return None
        for rx, fn in STEPS:
            m = rx.search(s)
            if m:
                n = fn(n, m)
                break
        else:
            return None
    return n


class MathWorker(BaseWorker):
    async def solve(self, task: dict) -> bool:
        self.clear_stale_answer(task)
        check = task.get("check") or "exact_int"

        # 1) без LLM
        try:
            res = solve_deterministic(task["instruction"])
        except Exception as e:
            logger.warning(f"math deterministic error: {e}")
            res = None
        if res is not None:
            self.save_answer(task["path"], str(res))
            self.trace(task, "deterministic", True)
            return True

        # 2) fallback: LLM пишет программу, считаем сами
        problem = " ".join(s for s in split_sentences(task["instruction"])
                           if not IGNORE.match(s) or s.lower().startswith("how many"))
        prompt = ("Write a short Python 3 program that solves the word problem and prints ONLY the final integer. "
                  "Apply every step literally and in order; 'rounded down' means integer floor division.\n"
                  f"Problem: {problem}\nReturn only the code in one ```python block.")
        text = await self.ask(task, prompt, "qwen3.6-35b-a3b-noreason", max_tokens=400)
        code = extract_code(text)
        rc, out, err = await run_py(code, task["path"], timeout=15)
        ans = normalize_answer(out, check) if rc == 0 else None
        if ans is None:
            logger.warning(f"math fallback failed: rc={rc} err={err[:200]}")
            self.trace(task, "llm", False, err)
            return False
        self.save_answer(task["path"], ans)
        self.trace(task, "llm", True)
        return True