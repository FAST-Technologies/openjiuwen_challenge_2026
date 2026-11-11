
import asyncio
import ast
import json
import tempfile
from pathlib import Path

from workers.base_worker import BaseWorker
from workers.math_worker import MathWorker, solve_deterministic
from workers.data_worker import DataWorker
from workers.search_worker import SearchWorker, KB, resolve
from workers.coding_worker import CodingWorker


# ---------- Test infrastructure ----------

class Tracker:
    def __init__(self):
        self.spent = 0.0

    async def add_usage(self, prompt_tokens, completion_tokens):
        self.spent += completion_tokens + prompt_tokens / 10


class StubLLM:
    def __init__(self, replies=()):
        self.replies = list(replies)
        self.calls = 0

    async def call(self, prompt, model=None, max_tokens=None, system=None):
        self.calls += 1
        if not self.replies:
            raise AssertionError("Unexpected LLM call")
        return self.replies.pop(0), {
            "prompt_tokens": 100,
            "completion_tokens": 20,
        }


class Checker:
    def __init__(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="worker_tests_")
        self.root = Path(self.tmp.name)
        self.passed = 0
        self.failed = 0
        self.unsupported_count = 0

    def close(self):
        self.tmp.cleanup()

    def result(self, name, ok, detail=""):
        if ok:
            self.passed += 1
            status = "PASS"
        else:
            self.failed += 1
            status = "FAIL"
        print(f"[{status}] {name} {detail}")

    def unsupported(self, name, detail=""):
        self.unsupported_count += 1
        print(f"[UNSUPPORTED] {name} {detail}")

    def make_task(self, task_id, task_type, instruction,
                  files=None, check="exact_int"):
        path = self.root / task_id
        ws = path / "workspace"
        ws.mkdir(parents=True, exist_ok=True)

        task = {
            "task_id": task_id,
            "task_type": task_type,
            "instruction": instruction,
            "check": check,
            "reward": 1,
            "path": path,
        }

        (path / "task.json").write_text(
            json.dumps({
                "task_id": task_id,
                "task_type": task_type,
                "instruction": instruction,
                "check": check,
            }),
            encoding="utf-8",
        )

        for name, content in (files or {}).items():
            p = ws / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8")

        return task

    def worker(self, cls, replies=()):
        return cls(StubLLM(replies), Tracker())

    @staticmethod
    def answer(task):
        p = task["path"] / "answer.txt"
        return p.read_text(encoding="utf-8").strip() if p.exists() else None


# ---------- MathWorker ----------

async def test_math(c):
    print("\n=== MathWorker ===")

    worker = c.worker(MathWorker)

    cases = [
        (
            "math_original_1",
            "The warehouse starts with 1752 items. "
            "It split equally into 2 boxes and it keeps one share rounded down. "
            "Then doubled. "
            "Then split equally into 3 boxes and it keeps one share rounded down. "
            "Then doubled. "
            "Then split equally into 3 boxes and it keeps one share rounded down.",
            "389",
        ),
        (
            "math_negative",
            "The warehouse starts with 50 items. "
            "It ships 65 items. Then receives 13 items.",
            "-2",
        ),
    ]

    for tid, instruction, expected in cases:
        task = c.make_task(tid, "math", instruction)
        ok = await worker.solve(task)
        c.result(
            tid,
            ok and c.answer(task) == expected,
            f"expected={expected}, actual={c.answer(task)!r}",
        )

    # Direct checks of deterministic parser.
    checks = [
        ("math subtraction", "The warehouse starts with 50 items. It ships 20 items.", 30),
        ("math addition", "The warehouse starts with 50 items. It receives 20 items.", 70),
        ("math multiplication", "The warehouse starts with 10 items. It doubled.", 20),
        ("math floor division", "The warehouse starts with 17 items. It split equally into 3 boxes and it keeps one share.", 5),
    ]

    for name, instruction, expected in checks:
        actual = solve_deterministic(instruction)
        c.result(name, actual == expected, f"actual={actual!r}")

    # These forms are not implemented by the deterministic grammar.
    c.unsupported("math percentages", "requires LLM fallback")
    c.unsupported("math fractions", "requires LLM fallback")
    c.unsupported("math arbitrary negative arithmetic", "not a general expression parser")

    # Verify fallback execution and token accounting.
    fallback = c.worker(MathWorker, ["print(207)"])
    task = c.make_task(
        "math_percentage_fallback",
        "math",
        "Compute 200 increased by 15% and then decreased by 10%.",
    )
    ok = await fallback.solve(task)

    c.result(
        "math LLM fallback",
        ok and c.answer(task) == "207" and fallback.llm_client.calls == 1,
        f"answer={c.answer(task)!r}",
    )

    c.result(
        "math fallback token accounting",
        fallback.budget_tracker.spent == 30,
        f"spent={fallback.budget_tracker.spent}",
    )


# ---------- DataWorker ----------

async def test_data(c):
    print("\n=== DataWorker ===")

    worker = c.worker(DataWorker)

    csv_data = (
        "product,month,units\n"
        "widget,7,100\n"
        "widget,8,58\n"
        "widget,9,50\n"
        "gadget,7,999\n"
        "widget,5,20\n"
    )

    task = c.make_task(
        "data_range",
        "data_analysis",
        "How many units of product 'widget' were sold in months 7 through 8?",
        {"sales.csv": csv_data},
    )

    ok = await worker.solve(task)
    c.result(
        "data sum over inclusive month range",
        ok and c.answer(task) == "158",
        f"answer={c.answer(task)!r}",
    )

    # One-month query.
    task = c.make_task(
        "data_one_month",
        "data_analysis",
        "How many units of product 'widget' were sold in month 9?",
        {"sales.csv": csv_data},
    )

    ok = await worker.solve(task)
    c.result(
        "data one month",
        ok and c.answer(task) == "50",
        f"answer={c.answer(task)!r}",
    )

    # Extra irrelevant column must not prevent solving.
    task = c.make_task(
        "data_extra_column",
        "data_analysis",
        "How many units of product 'widget' were sold in months 7 through 8?",
        {
            "sales.csv":
                "product,month,units,discount\n"
                "widget,7,10,2\n"
                "widget,8,20,3\n"
        },
    )

    ok = await worker.solve(task)
    c.result(
        "data ignores irrelevant extra column",
        ok and c.answer(task) == "30",
        f"answer={c.answer(task)!r}",
    )

    # Missing required column.
    task = c.make_task(
        "data_missing_units",
        "data_analysis",
        "Use columns product,month,units. "
        "How many units of product 'widget' were sold in months 7 through 8?",
        {"sales.csv": "product,month\nwidget,7\n"},
    )

    ok = await worker.solve(task)
    c.result(
        "data missing required column",
        not ok and c.answer(task) is None,
        f"ok={ok}, answer={c.answer(task)!r}",
    )

    # Malformed rows should be skipped by the deterministic implementation.
    task = c.make_task(
        "data_bad_row",
        "data_analysis",
        "How many units of product 'widget' were sold in months 7 through 9?",
        {
            "sales.csv":
                "product,month,units\n"
                "widget,7,10\n"
                "widget,8,\n"
                "widget,9,5\n"
        },
    )

    ok = await worker.solve(task)
    c.result(
        "data malformed numeric row",
        ok and c.answer(task) == "15",
        f"answer={c.answer(task)!r}",
    )

    # LLM fallback mechanics.
    fallback = c.worker(DataWorker, [
        "```python\nprint(159)\n```"
    ])

    task = c.make_task(
        "data_fallback",
        "data_analysis",
        "Calculate the total quantity for region east.",
        {
            "sales.csv":
                "region,quantity\n"
                "east,100\n"
                "west,20\n"
                "east,59\n"
        },
    )

    ok = await fallback.solve(task)
    c.result(
        "data fallback script execution",
        ok and c.answer(task) == "159",
        f"answer={c.answer(task)!r}",
    )

    c.result(
        "data fallback token accounting",
        fallback.budget_tracker.spent == 30,
        f"spent={fallback.budget_tracker.spent}",
    )

    c.unsupported("data grouping and aggregation", "not in deterministic parser")
    c.unsupported("data average calculations", "not in deterministic parser")
    c.unsupported("data multiple filters", "requires fallback")
    c.unsupported("data alternative column names", "deterministic parser expects product/month/units")


# ---------- SearchWorker ----------

async def test_search(c):
    print("\n=== SearchWorker ===")

    kb_dir = c.root / "search_fixture" / "workspace" / "kb"
    kb_dir.mkdir(parents=True)

    docs = {
        "irina.md": (
            "# Irina Smith\n"
            "Born: 1983\n"
            "Employer: Dune Robotics\n"
        ),
        "dune.md": (
            "# Dune Robotics\n"
            "Headquarters: Osaka\n"
            "Founded: 1999\n"
        ),
        "alex.md": (
            "# Alex Brown\n"
            "Born: 1975\n"
        ),
        "nova1.md": (
            "# Nova Labs\n"
            "Headquarters: Oslo\n"
        ),
        "nova2.md": (
            "# Nova Labs\n"
            "Headquarters: Riga\n"
        ),
    }

    for name, content in docs.items():
        (kb_dir / name).write_text(content, encoding="utf-8")

    kb = KB(kb_dir)

    ans, status = resolve(kb, "When was Irina Smith born?")
    c.result(
        "search direct person field",
        status == "ok" and ans == "1983",
        f"status={status}, answer={ans!r}",
    )

    ans, status = resolve(kb, "Where is Dune Robotics headquartered?")
    c.result(
        "search company headquarters",
        status == "ok" and ans == "Osaka",
        f"status={status}, answer={ans!r}",
    )

    ans, status = resolve(kb, "What year was Irina Smith born?")
    c.result(
        "search born synonym",
        status == "ok" and ans == "1983",
        f"status={status}, answer={ans!r}",
    )

    c.result(
        "search duplicate titles are ambiguous",
        kb.get("Nova Labs") is None,
    )

    ans, status = resolve(kb, "When was Unknown Person born?")
    c.result(
        "search unknown entity detection",
        status == "unknown_entity",
        f"status={status}",
    )

    ans, status = resolve(kb, "When was I. Smith born?")
    c.unsupported(
        "search abbreviated aliases",
        f"status={status}; alias index is not implemented",
    )

    # A duplicate title forces resolve() into fallback.
    # The stub response checks fallback plumbing only, not factual accuracy.
    fallback = c.worker(SearchWorker, ["Oslo"])

    task = c.make_task(
        "search_ambiguous_fallback",
        "search",
        "Where is Nova Labs headquartered?",
        {
            "kb/nova1.md": "# Nova Labs\nHeadquarters: Oslo\n",
            "kb/nova2.md": "# Nova Labs\nHeadquarters: Riga\n",
        },
        check=None,
    )

    ok = await fallback.solve(task)
    c.result(
        "search ambiguous entity LLM fallback plumbing",
        ok and c.answer(task) == "Oslo" and fallback.llm_client.calls == 1,
        f"answer={c.answer(task)!r}",
    )

    c.result(
        "search fallback token accounting",
        fallback.budget_tracker.spent == 30,
        f"spent={fallback.budget_tracker.spent}",
    )

    c.unsupported(
        "search chain longer than supported context expansion",
        "KB.context() expands at most three frontiers",
    )


# ---------- CodingWorker ----------

async def test_coding(c):
    print("\n=== CodingWorker ===")

    lib3 = """def reverse_words(text):
    return sorted(text.split())

def total(a, b):
    return a - b
"""

    tests3 = """import unittest
from mylib import reverse_words, total

class T(unittest.TestCase):
    def test_reverse(self):
        self.assertEqual(
            reverse_words("one two three"),
            ["three", "two", "one"]
        )

    def test_total(self):
        self.assertEqual(total(4, 5), 9)

if __name__ == "__main__":
    unittest.main()
"""

    lib7 = """def maximum(a, b):
    return a if a < b else b

def count_even(values):
    return sum(1 for x in values if x % 2 == 1)
"""

    tests7 = """import unittest
from mylib import maximum, count_even

class T(unittest.TestCase):
    def test_maximum(self):
        self.assertEqual(maximum(3, 8), 8)
        self.assertEqual(maximum(-5, -2), -2)

    def test_count_even(self):
        self.assertEqual(count_even([1, 2, 3, 4, 0]), 3)

if __name__ == "__main__":
    unittest.main()
"""

    for tid, lib, tests in [
        ("coding_reverse_total", lib3, tests3),
        ("coding_maximum_even", lib7, tests7),
    ]:
        task = c.make_task(
            tid,
            "coding",
            "Fix the bugs in mylib.py and run python3 test_mylib.py.",
            {
                "mylib.py": lib,
                "test_mylib.py": tests,
            },
        )

        ws = task["path"] / "workspace"
        lib_path = ws / "mylib.py"
        test_path = ws / "test_mylib.py"

        test_before = test_path.read_bytes()

        original_tree = ast.parse(lib)
        original_signatures = {
            n.name: ast.dump(n.args)
            for n in ast.walk(original_tree)
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        }

        worker = c.worker(CodingWorker)

        try:
            ok = await worker.solve(task)
        except Exception as exc:
            ok = False
            c.result(
                tid + " worker raised exception",
                False,
                f"{type(exc).__name__}: {exc}",
            )
            continue

        c.result(
            tid + " repaired",
            ok and c.answer(task) == "DONE",
            f"answer={c.answer(task)!r}",
        )

        c.result(
            tid + " test file unchanged",
            test_path.read_bytes() == test_before,
        )

        try:
            repaired_tree = ast.parse(lib_path.read_text(encoding="utf-8"))
            repaired_signatures = {
                n.name: ast.dump(n.args)
                for n in ast.walk(repaired_tree)
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
            }
            syntax_ok = True
        except SyntaxError:
            syntax_ok = False
            repaired_signatures = {}

        c.result(tid + " valid Python syntax", syntax_ok)
        c.result(
            tid + " function signatures preserved",
            all(
                repaired_signatures.get(name) == signature
                for name, signature in original_signatures.items()
            ),
        )

    # Edge cases and multiple defects are not guaranteed by the current
    # greedy single-mutation repair algorithm.
    c.unsupported(
        "coding multiple independent bugs in one function",
        "greedy mutation search may fail to find a multi-edit solution",
    )
    c.unsupported(
        "coding exhaustive edge-case generation",
        "worker relies on supplied tests; it does not generate tests",
    )

    # Failed LLM repair must restore original source.
    broken = "def greet(name):\n    return 'hello'\n"

    tests = """import unittest
from mylib import greet

class T(unittest.TestCase):
    def test_greet(self):
        self.assertEqual(greet("Ada"), "Hello, Ada!")

if __name__ == "__main__":
    unittest.main()
"""

    task = c.make_task(
        "coding_rollback",
        "coding",
        "Fix the bugs in mylib.py and run python3 test_mylib.py.",
        {
            "mylib.py": broken,
            "test_mylib.py": tests,
        },
    )

    ws = task["path"] / "workspace"
    test_before = (ws / "test_mylib.py").read_bytes()

    worker = c.worker(CodingWorker, [
        "not valid python",
        "```python\ndef greet(name):\n    return 'still wrong'\n```",
    ])

    ok = await worker.solve(task)

    source_after = (ws / "mylib.py").read_text(encoding="utf-8")

    c.result(
        "coding failed repair does not save answer",
        not ok and c.answer(task) is None,
        f"answer={c.answer(task)!r}",
    )

    c.result(
        "coding rollback restores original source",
        source_after == broken,
    )

    c.result(
        "coding failure does not modify tests",
        (ws / "test_mylib.py").read_bytes() == test_before,
    )

    c.result(
        "coding failed LLM token accounting",
        worker.budget_tracker.spent == 60,
        f"spent={worker.budget_tracker.spent}",
    )


# ---------- Main ----------

async def main():
    c = Checker()

    try:
        await test_math(c)
        await test_data(c)
        await test_search(c)
        await test_coding(c)

        print("\n=== SUMMARY ===")
        print("PASS:", c.passed)
        print("FAIL:", c.failed)
        print("UNSUPPORTED:", c.unsupported_count)

        print(
            "Temporary test directory:",
            c.root,
        )

        return 1 if c.failed else 0

    finally:
        c.close()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
