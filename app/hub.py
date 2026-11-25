"""Application Knowledge Hub core.

Ingestion -> chunking with context enrichment -> hybrid index (BM25 + vectors)
-> security-trimmed, graph-aware retrieval -> grounded, cited answers.

Everything is dependency-free so the demo runs anywhere. `HashingEmbedder` is a
stand-in for Azure OpenAI embeddings: swap it for any object with `embed(text)`.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter, deque
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

TOKEN_RE = re.compile(r"[a-z0-9]+")
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
STOPWORDS = set(
    "a an and are as at be by can do does for from has have how i if in into is it its "
    "of on or so that the their then there this to was we what when where which who why "
    "will with you your my me any most our".split()
)

# Small enterprise glossary: maps how people ask to how documents are written.
# In production this comes from the governed taxonomy (e.g. Purview glossary).
GLOSSARY = {
    "down": ["unavailable", "outage", "timeout"],
    "outage": ["unavailable", "incident"],
    "fix": ["runbook", "resolution", "restart"],
    "card": ["cardholder", "pci"],
    "service": ["engine", "system"],
    "label": ["labels"],
    "labels": ["label"],
    "printed": ["print"],
    "reversed": ["reverse", "reversal"],
    "return": ["returned", "returns", "refund"],
}

# Freshness SLA in days, per document type.
FRESHNESS_SLA = {
    "ticket": 180, "incident": 180, "metadata": 90,
    "architecture": 365, "requirements": 365, "process": 365, "policy": 365,
}

ABSTAIN_BELOW = 0.30  # confidence threshold for "I don't know"


def tokenize(text: str) -> list[str]:
    return [t for t in TOKEN_RE.findall(text.lower()) if t not in STOPWORDS]


def expand(tokens: list[str]) -> list[str]:
    out = list(tokens)
    for t in tokens:
        out += GLOSSARY.get(t, [])
    return out


class HashingEmbedder:
    """Sparse hashed features (words, bigrams, 4-char grams), L2-normalised.

    Char grams make 'pricing' ~ 'price', bigrams capture phrases. Good enough to
    show the vector leg of hybrid search without an API key.
    """

    dim = 2048

    def embed(self, text: str) -> dict[int, float]:
        toks = expand(tokenize(text))
        feats = toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:])]
        feats += [t[i : i + 4] for t in toks if len(t) > 4 for i in range(len(t) - 3)]
        vec: Counter[int] = Counter()
        for f in feats:
            vec[int(hashlib.md5(f.encode()).hexdigest()[:8], 16) % self.dim] += 1.0
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        return {i: v / norm for i, v in vec.items()}


def cosine(a: dict[int, float], b: dict[int, float]) -> float:
    if len(a) > len(b):
        a, b = b, a
    return sum(v * b.get(i, 0.0) for i, v in a.items())


@dataclass
class Chunk:
    id: str
    doc_id: str
    section: str
    text: str
    tokens: list[str]
    vector: dict[int, float]


@dataclass
class Document:
    id: str
    source: str
    doc_type: str
    app: str
    title: str
    url: str
    owner: str
    updated_at: str
    acl: list[str]
    body: str
    chunks: list[Chunk] = field(default_factory=list, repr=False)

    def age_days(self, today: date) -> int:
        return (today - date.fromisoformat(self.updated_at)).days

    def is_stale(self, today: date) -> bool:
        return self.age_days(today) > FRESHNESS_SLA.get(self.doc_type, 365)

    def meta(self, today: date) -> dict:
        return {
            "doc_id": self.id, "title": self.title, "url": self.url, "source": self.source,
            "doc_type": self.doc_type, "app": self.app, "owner": self.owner or None,
            "updated_at": self.updated_at, "stale": self.is_stale(today),
        }


class KnowledgeHub:
    def __init__(self, embedder=None, today: date | None = None):
        self.embedder = embedder or HashingEmbedder()
        self.today = today or date.today()
        self.users: dict[str, dict] = {}
        self.apps: dict[str, dict] = {}
        self.docs: dict[str, Document] = {}
        self.chunks: list[Chunk] = []
        self.eval_set: list[dict] = []
        self.gaps: deque[dict] = deque(maxlen=50)  # abstained queries = knowledge gaps
        self._df: Counter[str] = Counter()
        self._avgdl = 1.0

    # ---------- ingestion ----------

    @classmethod
    def from_seed(cls, path: str | Path, **kw) -> "KnowledgeHub":
        seed = json.loads(Path(path).read_text())
        hub = cls(**kw)
        hub.users = seed["users"]
        hub.apps = {a["id"]: a for a in seed["applications"]}
        hub.eval_set = seed.get("eval_set", [])
        for d in seed["documents"]:
            hub.ingest(d, reindex=False)
        hub._reindex()
        return hub

    def ingest(self, raw: dict, reindex: bool = True) -> Document:
        """Upsert a connector-normalised document (Confluence, Jira, SharePoint...)."""
        doc = Document(**{k: raw[k] for k in Document.__dataclass_fields__ if k != "chunks"})
        doc.chunks = list(self._chunk(doc))
        self.docs[doc.id] = doc
        self.chunks = [c for d in self.docs.values() for c in d.chunks]
        if reindex:
            self._reindex()
        return doc

    def _chunk(self, doc: Document, max_words: int = 60, overlap: int = 15):
        """Split on headings, then window long sections. Each chunk's embedding
        is enriched with title/app/section so it carries its own context."""
        app_name = self.apps.get(doc.app, {}).get("name", doc.app)
        sections = re.split(r"^##\s+", doc.body, flags=re.M)
        n = 0
        for sec in sections:
            sec = sec.strip()
            if not sec:
                continue
            heading, _, text = sec.partition("\n")
            if not text:
                heading, text = "Body", heading
            words = text.split()
            step = max_words - overlap
            for start in range(0, max(len(words) - overlap, 1), step):
                piece = " ".join(words[start : start + max_words])
                context = f"{doc.title}. {app_name}. {heading}. "
                yield Chunk(
                    id=f"{doc.id}#{n}", doc_id=doc.id, section=heading.strip(), text=piece,
                    tokens=tokenize(context + piece),
                    vector=self.embedder.embed(context + piece),
                )
                n += 1

    def _reindex(self):
        self._df = Counter(t for c in self.chunks for t in set(c.tokens))
        self._avgdl = sum(len(c.tokens) for c in self.chunks) / max(len(self.chunks), 1)

    # ---------- access control ----------

    def groups(self, user: str) -> set[str]:
        if user not in self.users:
            raise KeyError(user)
        return set(self.users[user]["groups"])

    def can_read(self, user: str, doc: Document) -> bool:
        return bool(self.groups(user) & set(doc.acl))

    # ---------- knowledge graph ----------

    def neighbours(self, app_id: str) -> set[str]:
        a = self.apps.get(app_id, {})
        out = set(a.get("depends_on", [])) | set(a.get("integrates_with", []))
        out |= {o for o, x in self.apps.items() if app_id in x.get("depends_on", []) + x.get("integrates_with", [])}
        return out - {app_id}

    def impacted_by(self, app_id: str) -> list[str]:
        """Transitive downstream: everything that breaks if `app_id` goes down."""
        seen, queue = [], [app_id]
        while queue:
            cur = queue.pop(0)
            for o, x in self.apps.items():
                if cur in x.get("depends_on", []) and o not in seen and o != app_id:
                    seen.append(o)
                    queue.append(o)
        return seen

    def linked_apps(self, query: str) -> set[str]:
        """Entity linking: which applications does the query mention?"""
        q = " ".join(tokenize(query))
        hits = set()
        for aid, a in self.apps.items():
            name_toks = tokenize(a["name"])
            if aid in q.split() or (name_toks and " ".join(name_toks) in q) or (
                len(name_toks) > 1 and name_toks[0] in q.split()
            ):
                hits.add(aid)
        return hits

    def app_view(self, app_id: str, user: str) -> dict:
        a = self.apps[app_id]
        return {
            **a,
            "owner": a.get("owner") or None,
            "depended_on_by": sorted(o for o, x in self.apps.items() if app_id in x.get("depends_on", [])),
            "impact_if_down": self.impacted_by(app_id),
            "documents": [d.meta(self.today) for d in self.docs.values() if d.app == app_id and self.can_read(user, d)],
        }

    # ---------- retrieval ----------

    def _bm25(self, q: list[str], c: Chunk, k1=1.5, b=0.75) -> float:
        n, tf, dl = len(self.chunks), Counter(c.tokens), len(c.tokens)
        s = 0.0
        for t in set(q):
            if t in tf:
                idf = math.log(1 + (n - self._df[t] + 0.5) / (self._df[t] + 0.5))
                s += idf * tf[t] * (k1 + 1) / (tf[t] + k1 * (1 - b + b * dl / self._avgdl))
        return s

    def search(self, query: str, user: str, k: int = 5, app: str | None = None,
               source: str | None = None, doc_type: str | None = None,
               expand_graph: bool = True) -> list[dict]:
        """Hybrid retrieval: BM25 + vector, fused with RRF, then graph & freshness signals.

        Security trimming happens *before* ranking, so unauthorised content never
        influences scores or leaks through snippets.
        """
        allowed_apps = None
        if app:
            allowed_apps = {app} | (self.neighbours(app) if expand_graph else set())
        pool = [
            c for c in self.chunks
            if self.can_read(user, d := self.docs[c.doc_id])
            and (allowed_apps is None or d.app in allowed_apps)
            and (source is None or d.source == source)
            and (doc_type is None or d.doc_type == doc_type)
        ]
        if not pool:
            return []

        q_tokens = expand(tokenize(query))
        q_vec = self.embedder.embed(query)
        kw = {c.id: self._bm25(q_tokens, c) for c in pool}
        vec = {c.id: cosine(q_vec, c.vector) for c in pool}
        kw_rank = {cid: r for r, cid in enumerate(sorted((i for i in kw if kw[i] > 0), key=kw.get, reverse=True), 1)}
        vec_rank = {cid: r for r, cid in enumerate(sorted((i for i in vec if vec[i] > 0), key=vec.get, reverse=True), 1)}

        mentioned = self.linked_apps(query) | ({app} if app else set())
        related = set().union(*(self.neighbours(m) for m in mentioned)) if mentioned else set()

        results = []
        for c in pool:
            if c.id not in kw_rank and c.id not in vec_rank:
                continue
            d = self.docs[c.doc_id]
            rrf = sum(1 / (60 + r[c.id]) for r in (kw_rank, vec_rank) if c.id in r)
            graph = 1.25 if d.app in mentioned else 1.1 if d.app in related else 1.0
            fresh = 0.85 if d.is_stale(self.today) else 1.0
            coverage = len(set(tokenize(query)) & set(expand(c.tokens))) / max(len(set(tokenize(query))), 1)
            results.append({
                "chunk_id": c.id, "section": c.section, "text": c.text,
                "score": round(rrf * graph * fresh * 1000, 3),
                "signals": {
                    "bm25": round(kw[c.id], 3), "vector": round(vec[c.id], 3),
                    "graph_boost": graph, "freshness_penalty": fresh,
                    "term_coverage": round(coverage, 2),
                },
                **d.meta(self.today),
            })
        results.sort(key=lambda r: r["score"], reverse=True)

        # Diversify: at most two chunks per document in the final list.
        per_doc: Counter[str] = Counter()
        out = []
        for r in results:
            if per_doc[r["doc_id"]] < 2:
                per_doc[r["doc_id"]] += 1
                out.append(r)
            if len(out) == k:
                break
        return out

    # ---------- grounding ----------

    def ask(self, question: str, user: str, k: int = 4, **filters) -> dict:
        """Extractive, cited answer. Abstains instead of guessing when evidence is weak.

        The sentence-selection step is where an LLM (Azure OpenAI) would sit in
        production; the retrieval, citations and abstain logic stay identical.
        """
        hits = self.search(question, user, k=k, **filters)
        confidence = 0.0
        if hits:
            top = hits[0]["signals"]
            confidence = round(min(1.0, 0.5 * top["term_coverage"] + 0.5 * min(top["vector"] / 0.35, 1.0)), 2)

        if not hits or confidence < ABSTAIN_BELOW:
            self.gaps.append({"question": question, "user": user, "confidence": confidence, "at": self.today.isoformat()})
            return {
                "question": question, "answer": None, "abstained": True, "confidence": confidence,
                "reason": "No sufficiently relevant content you are authorised to read.",
                "citations": [], "warnings": [],
            }

        # Only compose from evidence that covers the question about as well as the best hit.
        best = hits[0]["signals"]["term_coverage"]
        hits = [h for h in hits if h["signals"]["term_coverage"] >= 0.75 * best]
        q = set(expand(tokenize(question)))
        q_vec = self.embedder.embed(question)
        cands = []
        for n, h in enumerate(hits, 1):
            for s in SENTENCE_RE.split(h["text"]):
                st = set(tokenize(s))
                if st:
                    score = len(q & st) / len(st) ** 0.5 + cosine(q_vec, self.embedder.embed(s)) + 0.15 / n
                    cands.append((score, n, s.strip()))
        ranked = sorted(cands, reverse=True)
        picked = [c for c in ranked[:3] if c[0] >= 0.6 * ranked[0][0]]  # drop weak filler
        picked.sort(key=lambda x: (x[1], hits[x[1] - 1]["text"].find(x[2])))  # keep source order

        renum = {n: i for i, n in enumerate(sorted({n for _, n, _ in picked}), 1)}  # refs run 1..n
        picked = [(s, renum[n], t) for s, n, t in picked]
        hits = {renum[n]: hits[n - 1] for n in renum}
        citations = [{"ref": n, **{k: hits[n][k] for k in ("doc_id", "title", "url", "section", "source", "updated_at", "stale")}} for n in sorted(hits)]
        warnings = [f"[{c['ref']}] {c['doc_id']} is past its freshness SLA (updated {c['updated_at']})." for c in citations if c["stale"]]
        return {
            "question": question,
            "answer": " ".join(f"{s} [{n}]" for _, n, s in picked),
            "abstained": False, "confidence": confidence,
            "citations": citations, "warnings": warnings,
        }

    # ---------- governance ----------

    def quality_report(self) -> dict:
        issues = []
        for d in self.docs.values():
            if not d.owner:
                issues.append({"type": "missing_owner", "doc_id": d.id})
            if d.is_stale(self.today):
                issues.append({"type": "stale", "doc_id": d.id, "age_days": d.age_days(self.today),
                               "sla_days": FRESHNESS_SLA.get(d.doc_type, 365)})
            if not d.acl:
                issues.append({"type": "no_acl", "doc_id": d.id})
        documented = {d.app for d in self.docs.values()}
        for aid, a in self.apps.items():
            if not a.get("owner"):
                issues.append({"type": "app_missing_owner", "app": aid})
            if aid not in documented:
                issues.append({"type": "undocumented_app", "app": aid})
        healthy = len(self.docs) - len({i["doc_id"] for i in issues if "doc_id" in i})
        return {
            "documents": len(self.docs), "chunks": len(self.chunks), "applications": len(self.apps),
            "doc_health_pct": round(100 * healthy / max(len(self.docs), 1), 1),
            "issues": issues,
            "knowledge_gaps": list(self.gaps),
        }

    def evaluate(self, k: int = 5) -> dict:
        """Retrieval eval over the golden set: recall@k, MRR, hit rate."""
        rows = []
        for case in self.eval_set:
            got = list(dict.fromkeys(h["doc_id"] for h in self.search(case["query"], case["user"], k=k)))
            exp = set(case["expected"])
            first = next((i for i, d in enumerate(got, 1) if d in exp), None)
            rows.append({
                "query": case["query"], "user": case["user"], "expected": case["expected"], "retrieved": got,
                "recall": round(len(exp & set(got)) / len(exp), 2), "rr": round(1 / first, 2) if first else 0.0,
            })
        n = max(len(rows), 1)
        return {
            "k": k, "cases": len(rows),
            "recall_at_k": round(sum(r["recall"] for r in rows) / n, 3),
            "mrr": round(sum(r["rr"] for r in rows) / n, 3),
            "hit_rate": round(sum(r["rr"] > 0 for r in rows) / n, 3),
            "results": rows,
        }
