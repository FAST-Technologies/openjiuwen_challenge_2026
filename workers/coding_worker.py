import ast
import asyncio
import copy
import logging
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from .base_worker import BaseWorker, extract_code

logger = logging.getLogger(__name__)

HUGE = 10 ** 6
CMP = {ast.Lt: ast.Gt, ast.Gt: ast.Lt, ast.LtE: ast.GtE, ast.GtE: ast.LtE, ast.Eq: ast.NotEq, ast.NotEq: ast.Eq}
BIN = {ast.Add: [ast.Sub], ast.Sub: [ast.Add], ast.Mult: [ast.Div], ast.Div: [ast.FloorDiv, ast.Mult],
       ast.FloorDiv: [ast.Div]}
NAMES = {"sorted": ["reversed"], "reversed": ["sorted"], "min": ["max"], "max": ["min"],
         "any": ["all"], "all": ["any"]}


def run_tests(workdir: Path, test_file: str, timeout: float = 15):
    """-> (returncode, число упавших тестов, вывод). Ошибка импорта/синтаксиса = HUGE-1."""
    try:
        p = subprocess.run([sys.executable, "-B", test_file], cwd=str(workdir),
                           capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return 1, HUGE, "timeout"
    out = p.stderr + p.stdout
    if p.returncode == 0:
        return 0, 0, out
    m = re.search(r"FAILED \((.*?)\)", out)
    n = sum(int(x) for x in re.findall(r"(?:failures|errors)=(\d+)", m.group(1))) if m else HUGE - 1
    return p.returncode, n, out


def _variants(node):
    """[(приоритет, функция-мутатор)] для узла AST."""
    v = []
    if isinstance(node, ast.Compare) and len(node.ops) == 1 and type(node.ops[0]) in CMP:
        t = CMP[type(node.ops[0])]
        v.append((0, lambda n, t=t: n.ops.__setitem__(0, t())))
    if isinstance(node, (ast.BinOp, ast.AugAssign)) and type(node.op) in BIN:
        for t in BIN[type(node.op)]:
            v.append((0, lambda n, t=t: setattr(n, "op", t())))
    if isinstance(node, ast.BoolOp):
        t = ast.Or if isinstance(node.op, ast.And) else ast.And
        v.append((0, lambda n, t=t: setattr(n, "op", t())))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in NAMES:
        for name in NAMES[node.func.id]:
            v.append((0, lambda n, name=name: setattr(n.func, "id", name)))
    if isinstance(node, ast.Constant) and type(node.value) is int:
        for d in (1, -1):
            v.append((1, lambda n, d=d: setattr(n, "value", n.value + d)))
    return v


def mutants(tree):
    """Одиночные мутации: (приоритет, индекс, исходник)."""
    nodes = list(ast.walk(tree))
    for i, node in enumerate(nodes):
        for j, (prio, fn) in enumerate(_variants(node)):
            t2 = copy.deepcopy(tree)
            fn(list(ast.walk(t2))[i])
            try:
                yield prio, i * 10 + j, ast.unparse(ast.fix_missing_locations(t2))
            except Exception:
                continue


class CodingWorker(BaseWorker):
    async def solve(self, task: dict) -> bool:
        self.clear_stale_answer(task)
        ws = task["path"] / "workspace"
        instr = task["instruction"]
        m = re.search(r"bugs in ([\w.]+\.py)", instr)
        lib = m.group(1) if m else "mylib.py"
        m = re.search(r"python3 ([\w.]+\.py)", instr)
        test = m.group(1) if m else "test_mylib.py"
        lib_path, test_path = ws / lib, ws / test
        if not lib_path.exists() or not test_path.exists():
            self.trace(task, "precheck", False, "files missing")
            return False

        orig = lib_path.read_text(encoding="utf-8")
        test_src = test_path.read_text(encoding="utf-8")
        rc, n0, out0 = await asyncio.to_thread(run_tests, ws, test)
        if rc == 0:
            self.save_answer(task["path"], "DONE")
            self.trace(task, "already-passing", True)
            return True

        # 1) перебор одиночных мутаций (0 токенов)
        fixed = await asyncio.to_thread(self._mutation_repair, ws, lib, test, orig, n0)
        if fixed is not None:
            lib_path.write_text(fixed, encoding="utf-8")
            if (await asyncio.to_thread(run_tests, ws, test))[0] == 0:
                self.save_answer(task["path"], "DONE")
                self.trace(task, "mutation", True)
                return True
            lib_path.write_text(orig, encoding="utf-8")

        # 2) LLM
        if await self._llm_repair(task, ws, lib, test, orig, test_src, out0):
            self.save_answer(task["path"], "DONE")
            self.trace(task, "llm", True)
            return True
        lib_path.write_text(orig, encoding="utf-8")
        self.trace(task, "llm", False, "not fixed")
        return False

    # ------------------------------------------------------------
    def _mutation_repair(self, ws: Path, lib: str, test: str, orig: str, score0: int,
                         budget_s: float = 45.0, rounds: int = 5):
        try:
            tree = ast.parse(orig)
        except SyntaxError:
            return None
        tmp = Path(tempfile.mkdtemp(prefix="mut_"))
        try:
            for f in ws.iterdir():
                if f.is_file():
                    shutil.copy(f, tmp / f.name)
            deadline = time.monotonic() + budget_s
            cur_tree, cur_src, cur_score = tree, orig, score0
            for _ in range(rounds):
                best = None
                for prio, idx, src in mutants(cur_tree):
                    if time.monotonic() > deadline:
                        break
                    (tmp / lib).write_text(src, encoding="utf-8")
                    rc, score, _ = run_tests(tmp, test, timeout=8)
                    key = (score, prio, idx)
                    if score < cur_score and (best is None or key < best[0]):
                        best = (key, src, score)
                if best is None:
                    return None
                _, cur_src, cur_score = best
                if cur_score == 0:
                    return cur_src
                cur_tree = ast.parse(cur_src)
            return None
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    async def _llm_repair(self, task, ws, lib, test, orig, test_src, first_out) -> bool:
        try:
            names = {n.name for n in ast.walk(ast.parse(orig)) if isinstance(n, ast.FunctionDef)}
        except SyntaxError:
            names = set()
        feedback, current = first_out[-1500:], orig
        for model, mt in (("qwen3.6-35b-a3b-noreason", 1500), ("qwen3.6-35b-a3b", 4000)):
            prompt = (f"Fix the bugs in {lib} so that all tests in {test} pass. Do not change the tests. "
                      f"Keep every existing function name and signature.\n"
                      f"Return the COMPLETE corrected {lib} in one ```python block and nothing else.\n\n"
                      f"{lib}:\n```python\n{current}\n```\n\n{test}:\n```python\n{test_src}\n```\n\n"
                      f"Current test output:\n{feedback}")
            code = extract_code(await self.ask(task, prompt, model, max_tokens=mt))
            try:
                tree = ast.parse(code)
            except SyntaxError as e:
                feedback = f"Your previous answer had a SyntaxError: {e}"
                continue
            have = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
            if names - have:
                feedback = f"Your previous answer lost functions: {sorted(names - have)}"
                continue
            (ws / lib).write_text(code, encoding="utf-8")
            rc, _, out = await asyncio.to_thread(run_tests, ws, test)
            if rc == 0:
                return True
            feedback, current = out[-1500:], code
        return False