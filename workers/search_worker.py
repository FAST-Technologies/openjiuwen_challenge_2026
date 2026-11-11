import logging
import re
from pathlib import Path

from .base_worker import BaseWorker, normalize_answer, split_sentences

logger = logging.getLogger(__name__)

SYN = {
    "born": ["born", "birth"],
    "headquarters": ["headquarters", "headquartered", "hq", "located", "where", "city"],
    "founded": ["founded", "established"],
    "employer": ["employer", "employed", "works for", "works at"],
}
STOP = {"In", "Which", "What", "Who", "When", "Where", "How", "The", "Is", "Does", "Do", "Did", "Was", "Were", "Name"}


class KB:
    def __init__(self, kb_dir: Path):
        self.entities: dict[str, list[dict]] = {}
        self.keys: set[str] = set()
        for p in sorted(kb_dir.glob("*.md")):
            text = p.read_text(encoding="utf-8", errors="replace")
            title = next((l[2:].strip() for l in text.splitlines() if l.startswith("# ")), "")
            if not title:
                title = p.stem.split("_", 1)[-1].replace("_", " ").title()
            fields = {}
            for line in text.splitlines():
                m = re.match(r"^\s*[-*]?\s*([A-Za-z][A-Za-z _]*?)\s*:\s*(.+?)\s*$", line)
                if m:
                    k = m.group(1).strip().lower()
                    fields[k] = m.group(2).strip()
                    self.keys.add(k)
            self.entities.setdefault(title.lower(), []).append(
                {"title": title, "file": p.name, "text": text, "fields": fields})

    def get(self, name: str):
        lst = self.entities.get((name or "").strip().lower())
        return lst[0] if lst and len(lst) == 1 else None

    def mentions(self, q: str):
        ql = q.lower()
        found = []
        for title in sorted(self.entities, key=len, reverse=True):
            for m in re.finditer(r"(?<!\w)" + re.escape(title) + r"(?!\w)", ql):
                if all(m.end() <= s or m.start() >= e for _, s, e in found):
                    found.append((title, m.start(), m.end()))
        return sorted(found, key=lambda x: x[1])

    def context(self, titles: list[str], limit: int = 6000) -> str:
        seen, order, frontier = set(), [], list(titles)
        for _ in range(3):  # сами сущности + ссылки на 2 шага
            nxt = []
            for t in frontier:
                lst = self.entities.get(t.lower(), [])
                for e in lst:
                    if e["file"] in seen:
                        continue
                    seen.add(e["file"])
                    order.append(e)
                    nxt += [v for v in e["fields"].values() if v.lower() in self.entities]
            frontier = nxt
        if not order:
            order = [e for lst in self.entities.values() for e in lst]
            limit = 8000
        ctx = "".join(f"--- {e['file']} ---\n{e['text'].strip()}\n" for e in order)
        return ctx[:limit]


def resolve(kb: KB, q: str):
    """-> (answer|None, status) status: ok | unparsed | missing | unknown_entity"""
    ms = kb.mentions(q)
    if not ms:
        cands = []
        for c in re.findall(r"\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)+\b", q):
            words = c.split()
            while words and words[0] in STOP:
                words.pop(0)
            if len(words) >= 2:
                cands.append(" ".join(words))
        return None, ("unknown_entity" if cands else "unparsed")
    if len(ms) != 1:
        return None, "unparsed"
    title, s, e = ms[0]
    ent = kb.get(title)
    if ent is None:
        return None, "unparsed"

    ql = q.lower()
    hits = []
    for key in kb.keys:
        for pat in SYN.get(key, [key]):
            m = next((m for m in re.finditer(r"\b" + re.escape(pat) + r"\b", ql)
                      if m.end() <= s or m.start() >= e), None)
            if m:
                hits.append((m.start(), key))
                break
    if not hits:
        return None, "unparsed"
    after = [k for p, k in sorted(hits) if p >= e]
    before = [k for p, k in sorted(hits, reverse=True) if p < s]
    order = after + before

    cur = ent
    for i, key in enumerate(order):
        val = cur["fields"].get(key)
        if val is None:
            return None, "missing"
        if i == len(order) - 1:
            # итог — ссылка на другую сущность, а вопрос не «кто/какая компания» -> не уверены, отдаём LLM
            if kb.get(val) is not None and not re.search(
                    r"\b(who|whom|which (company|employer|person)|what (company|employer))\b", ql):
                return None, "unparsed"
            return val, "ok"
        cur = kb.get(val)
        if cur is None:
            return None, "missing"
    return None, "unparsed"


class SearchWorker(BaseWorker):
    async def solve(self, task: dict) -> bool:
        self.clear_stale_answer(task)
        kb_dir = task["path"] / "workspace" / "kb"
        if not kb_dir.exists():
            self.trace(task, "precheck", False, "kb/ not found")
            return False
        kb = KB(kb_dir)
        q = " ".join(s for s in split_sentences(task["instruction"]) if s.endswith("?")) or task["instruction"]

        ans, status = resolve(kb, q)
        if status == "ok":
            self.save_answer(task["path"], ans)
            self.trace(task, "deterministic", True)
            return True
        if status == "unknown_entity":
            self.trace(task, "precheck", False, "entity from question not in kb -> unsolvable")
            return False

        # fallback: в промпт идут только релевантные файлы
        titles = [t for t, _, _ in kb.mentions(q)]
        prompt = (f"Knowledge base files:\n{kb.context(titles)}\n\nQuestion: {q}\n"
                  "Reply with ONLY the answer value exactly as written in the files. "
                  "If the files do not contain the answer, reply NOT FOUND.")
        text = await self.ask(task, prompt, "qwen3.6-35b-a3b-noreason", max_tokens=40)
        ans = normalize_answer(text, task.get("check"))
        if not ans or "NOT FOUND" in ans.upper():
            self.trace(task, "llm", False, f"status={status}; {text[:80]}")
            return False
        self.save_answer(task["path"], ans)
        self.trace(task, "llm", True, f"status={status}")
        return True