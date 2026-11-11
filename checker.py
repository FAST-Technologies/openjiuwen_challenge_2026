"""Офлайн-проверка воркеров: LLM заменён заглушкой, сети нет."""
import asyncio, json, os, sys, tempfile
from pathlib import Path

os.environ["TRACE_FILE"] = os.path.join(tempfile.gettempdir(), "selftest_traj.jsonl")
from workers import MathWorker, SearchWorker, DataWorker, CodingWorker

TAIL_ANS = " Work only inside the current directory. When you are done, write ONLY the final answer to the file answer.txt in the current directory (for example with bash: echo 42 > answer.txt)."


class Tracker:
    def __init__(self): self.spent = 0.0
    async def add_usage(self, p, c): self.spent += c + p / 10


class Stub:
    """Если очередь ответов пуста, а LLM вызвали — тест падает."""
    def __init__(self, replies=()): self.replies = list(replies); self.calls = 0
    async def call(self, prompt, model="", max_tokens=0, system=None):
        self.calls += 1
        assert self.replies, f"LLM вызван неожиданно (model={model})"
        return self.replies.pop(0), {"prompt_tokens": 100, "completion_tokens": 20}


def mk(root, tid, ttype, instr, check, files=None, reward=1):
    d = root / tid
    (d / "workspace").mkdir(parents=True)
    (d / "task.json").write_text(json.dumps({"task_id": tid, "task_type": ttype, "instruction": instr,
                                             "reward": reward, "check": check}))
    for name, content in (files or {}).items():
        p = d / "workspace" / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
    return {"task_id": tid, "task_type": ttype, "instruction": instr, "check": check, "reward": reward, "path": d}


def ans(t):
    p = t["path"] / "answer.txt"
    return p.read_text() if p.exists() else None


SALES = """order_id,region,product,month,units,price
1,west,widget,5,5,33
2,north,sprocket,6,28,28
3,north,gizmo,1,39,17
4,west,gadget,5,23,98
5,west,gadget,12,44,31
6,north,gadget,3,22,72
7,east,widget,10,29,90
8,south,widget,8,44,57
9,east,sprocket,7,43,37
10,south,widget,8,48,15
11,east,widget,9,18,22
12,south,gizmo,6,40,41
13,east,gadget,12,20,54
14,west,widget,1,39,29
15,east,gadget,4,15,86
16,west,gizmo,12,44,77
17,west,widget,7,45,77
18,west,widget,3,29,13
19,east,gadget,8,34,67
20,north,widget,8,21,44
"""
LIB3 = """def reverse_words(s):
    return ' '.join(sorted(s.split()))


def total(xs):
    s = 0
    for x in xs:
        s -= x
    return s


def maximum(xs):
    best = xs[0]
    for x in xs:
        if x > best:
            best = x
    return best


def average(xs):
    return sum(xs) / len(xs)


def first_n(xs, n):
    return xs[:n]


def count_even(xs):
    return sum(1 for x in xs if x % 2 == 0)
"""
TEST3 = """import unittest

import mylib as m


class T(unittest.TestCase):
    def test_reverse_words(self):
        self.assertEqual(m.reverse_words('c a b'), 'b a c')

    def test_total(self):
        self.assertEqual(m.total([1, 2, 3]), 6)

    def test_maximum(self):
        self.assertEqual(m.maximum([3, 9, 2]), 9)

    def test_average(self):
        self.assertEqual(m.average([2, 4, 6]), 4)

    def test_first_n(self):
        self.assertEqual(m.first_n([5, 6, 7, 8], 3), [5, 6, 7])

    def test_count_even(self):
        self.assertEqual(m.count_even([1, 2, 4, 6, 7]), 3)


if __name__ == '__main__':
    unittest.main()
"""
LIB7 = """def maximum(xs):
    best = xs[0]
    for x in xs:
        if x < best:
            best = x
    return best


def total(xs):
    s = 0
    for x in xs:
        s += x
    return s


def reverse_words(s):
    return ' '.join(reversed(s.split()))


def count_even(xs):
    return sum(1 for x in xs if x % 2 == 1)
"""
TEST7 = """import unittest

import mylib as m


class T(unittest.TestCase):
    def test_maximum(self):
        self.assertEqual(m.maximum([3, 9, 2]), 9)

    def test_total(self):
        self.assertEqual(m.total([1, 2, 3]), 6)

    def test_reverse_words(self):
        self.assertEqual(m.reverse_words('c a b'), 'b a c')

    def test_count_even(self):
        self.assertEqual(m.count_even([1, 2, 4, 6, 7]), 3)


if __name__ == '__main__':
    unittest.main()
"""
CODE_INSTR = ("Run `python3 test_mylib.py` in the current directory; some tests fail. Fix the bugs in mylib.py "
              "(do not edit the tests) until all tests pass. Work only inside the current directory. When you are done, "
              "write ONLY the word DONE to the file answer.txt in the current directory (for example with bash: echo 42 > answer.txt).")
KB = {
    "person_irina_smith.md": "# Irina Smith\n\nBorn: 1983\nEmployer: Arktika Labs\nMentor: Kira Petrov\n",
    "person_kira_petrov.md": "# Kira Petrov\n\nBorn: 1975\nEmployer: Dune Robotics\n",
    "person_sergei_garcia.md": "# Sergei Garcia\n\nBorn: 1990\nEmployer: Dune Robotics\n",
    "company_arktika_labs.md": "# Arktika Labs\n\nHeadquarters: Medan\nFounded: 1982\n",
    "company_dune_robotics.md": "# Dune Robotics\n\nHeadquarters: Osaka\nFounded: 1999\n",
}
KB_PRE = "The directory kb/ contains a small knowledge base of markdown files. "
KB_POST = " Use the file tools or bash (ls, grep, cat) to find it." + TAIL_ANS


async def main():
    root = Path(tempfile.mkdtemp(prefix="selftest_"))
    tr = Tracker()
    results = []

    def check(name, got, want, calls=None, want_calls=0):
        ok = got == want and (calls is None or calls == want_calls)
        results.append(ok)
        print(("PASS " if ok else "FAIL ") + f"{name}: got={got!r} want={want!r}" + (f" llm_calls={calls}" if calls is not None else ""))

    # ---- math
    stub = Stub(); w = MathWorker(stub, tr)
    t = mk(root, "math_0000", "math",
           "A warehouse starts with 1752 boxes. Then the stock is split equally into 2 warehouses and it keeps one share (rounded down). Then the stock is doubled by merging with a sister warehouse. Then the stock is split equally into 3 warehouses and it keeps one share (rounded down). Then the stock is doubled by merging with a sister warehouse. Then the stock is split equally into 3 warehouses and it keeps one share (rounded down). How many boxes does it have at the end? Compute it with the bash tool (python3 -c)." + TAIL_ANS, "exact_int")
    await w.solve(t); check("math_0000 (det)", ans(t), "389", stub.calls)
    t = mk(root, "math_0004", "math",
           "A warehouse starts with 1669 boxes. It ships 36 boxes per day for 3 days. It receives 2 deliveries of 87 boxes each. How many boxes does it have at the end? Compute it with the bash tool (python3 -c)." + TAIL_ANS, "exact_int")
    await w.solve(t); check("math_0004 (det)", ans(t), "1735", stub.calls)
    stub = Stub(["```python\nprint(7*6)\n```"]); w = MathWorker(stub, tr)
    t = mk(root, "math_9001", "math", "A shop has 7 shelves with 6 jars each after a recount. How many jars are there?" + TAIL_ANS, "exact_int")
    await w.solve(t); check("math fallback (LLM)", ans(t), "42", stub.calls, 1)

    # ---- data
    stub = Stub(); w = DataWorker(stub, tr)
    t = mk(root, "data_analysis_0002", "data_analysis",
           "The file sales.csv has columns order_id,region,product,month,units,price (20 rows). How many units of product 'widget' were sold in months 7 through 8 inclusive? Use bash with python3 to compute it." + TAIL_ANS,
           "exact_int", {"sales.csv": SALES})
    await w.solve(t); check("data_analysis_0002 (det)", ans(t), "158", stub.calls)
    t = mk(root, "data_9002", "data_analysis",
           "The file sales.csv has columns order_id,region,product,month,units,price,discount (20 rows). How many units of product 'widget' were sold in months 7 through 8 inclusive?" + TAIL_ANS,
           "exact_int", {"sales.csv": SALES})
    await w.solve(t); check("data unsolvable (missing column)", ans(t), None, stub.calls)
    stub = Stub(["```python\nimport csv\nprint(sum(int(r['units']) for r in csv.DictReader(open('sales.csv')) if r['region']=='east'))\n```"]); w = DataWorker(stub, tr)
    t = mk(root, "data_9003", "data_analysis",
           "The file sales.csv has columns order_id,region,product,month,units,price (20 rows). What is the total number of units sold in region 'east'?" + TAIL_ANS,
           "exact_int", {"sales.csv": SALES})
    await w.solve(t); check("data fallback (LLM)", ans(t), "159", stub.calls, 1)

    # ---- search
    kbf = {f"kb/{k}": v for k, v in KB.items()}
    stub = Stub(); w = SearchWorker(stub, tr)
    for tid, q, want in [
        ("search_0001", "In which year was Irina Smith born?", "1983"),
        ("search_0005", "In which city is the headquarters of Sergei Garcia's employer?", "Osaka"),
        ("search_9004", "In which year was Irina Smith's mentor born?", "1975"),
        ("search_9005", "In which year was Dune Robotics founded?", "1999"),
        ("search_9006", "Which city hosts the headquarters of the employer of Irina Smith's mentor?", "Osaka"),
    ]:
        t = mk(root, tid, "search", KB_PRE + q + KB_POST, "exact_str", kbf)
        await w.solve(t); check(f"{tid} (det)", ans(t), want, stub.calls)
    t = mk(root, "search_9007", "search", KB_PRE + "In which year was Zoltan Quimby born?" + KB_POST, "exact_str", kbf)
    await w.solve(t); check("search unsolvable (unknown person)", ans(t), None, stub.calls)
    stub = Stub(["Medan"]); w = SearchWorker(stub, tr)
    t = mk(root, "search_9008", "search", KB_PRE + "Where is Irina Smith's employer located?" + KB_POST, "exact_str", kbf)
    await w.solve(t); check("search wording variant", ans(t), "Medan", None)
    print("   (calls:", stub.calls, ")")

    # ---- coding
    stub = Stub(); w = CodingWorker(stub, tr)
    t = mk(root, "coding_0003", "coding", CODE_INSTR, "unittest", {"mylib.py": LIB3, "test_mylib.py": TEST3})
    await w.solve(t); check("coding_0003 (mutation, 2 bugs)", ans(t), "DONE", stub.calls)
    t = mk(root, "coding_0007", "coding", CODE_INSTR, "unittest", {"mylib.py": LIB7, "test_mylib.py": TEST7})
    await w.solve(t); check("coding_0007 (mutation, 2 bugs)", ans(t), "DONE", stub.calls)
    # баг, который мутациями не лечится -> LLM
    lib = "def greet(name):\n    return 'Hi ' + name\n"
    test = "import unittest\nimport mylib as m\n\nclass T(unittest.TestCase):\n    def test_g(self):\n        self.assertEqual(m.greet('Bob'), 'Hello, Bob!')\n\nif __name__ == '__main__':\n    unittest.main()\n"
    stub = Stub(["```python\n    def greet(name):\n        return 'Hello, ' + name + '!'\n```"]); w = CodingWorker(stub, tr)
    t = mk(root, "coding_9009", "coding", CODE_INSTR, "unittest", {"mylib.py": lib, "test_mylib.py": test})
    await w.solve(t); check("coding LLM fallback (indented reply)", ans(t), "DONE", stub.calls, 1)
    # LLM не справился -> без answer.txt, исходник восстановлен
    stub = Stub(["```python\ndef greet(name):\n    return 'x'\n```", "```python\ndef greet(name):\n    return 'y'\n```"]); w = CodingWorker(stub, tr)
    t = mk(root, "coding_9010", "coding", CODE_INSTR, "unittest", {"mylib.py": lib, "test_mylib.py": test})
    await w.solve(t)
    restored = (t["path"] / "workspace" / "mylib.py").read_text() == lib
    check("coding failure: no answer + source restored", (ans(t), restored), (None, True))

    print(f"\n{sum(results)}/{len(results)} passed; tokens counted by tracker: {tr.spent:.1f}")
    return all(results)


if __name__ == "__main__":
    sys.exit(0 if asyncio.run(main()) else 1)