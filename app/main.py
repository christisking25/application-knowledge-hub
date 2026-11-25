"""Application Knowledge Hub API.

Identity: the `X-User` header stands in for an Entra ID token. Every retrieval
path is security-trimmed to that user's groups.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from .hub import KnowledgeHub

hub = KnowledgeHub.from_seed(Path(__file__).parent.parent / "data" / "seed.json")

app = FastAPI(
    title="Application Knowledge Hub",
    version="0.1.0",
    description=(
        "Governed context layer for AI agents: ingestion, hybrid retrieval (BM25 + vector + "
        "metadata + knowledge graph), role-based security trimming, cited answers that abstain "
        "when unsure, quality monitoring and retrieval evaluation.\n\n"
        "Try `X-User`: **alice** (architect), **sam** (store ops), **riley** (SRE)."
    ),
)


def current_user(x_user: str = Header("alice", description="alice | sam | riley")) -> str:
    if x_user not in hub.users:
        raise HTTPException(401, f"Unknown user '{x_user}'. Try one of: {', '.join(hub.users)}")
    return x_user


class Filters(BaseModel):
    app: str | None = Field(None, description="Application id, e.g. orderhub. Neighbours in the graph are included.")
    source: Literal["confluence", "jira", "sharepoint", "servicenow", "app_registry"] | None = None
    doc_type: str | None = None
    expand_graph: bool = True


class SearchRequest(Filters):
    query: str = Field(..., examples=["What happens at checkout if the pricing service is down?"])
    k: int = Field(5, ge=1, le=20)


class AskRequest(Filters):
    question: str = Field(..., examples=["Are loyalty points reversed when an order is returned?"])


class IngestDocument(BaseModel):
    id: str
    source: str
    doc_type: str
    app: str
    title: str
    url: str
    owner: str = ""
    updated_at: str = Field(..., examples=["2026-11-20"])
    acl: list[str] = Field(..., examples=[["engineering"]])
    body: str = Field(..., description="Markdown-ish text; '## ' headings become chunk boundaries.")


def _filters(r: Filters) -> dict:
    return r.model_dump(include={"app", "source", "doc_type", "expand_graph"})


@app.get("/", tags=["meta"])
def root():
    return {
        "service": "Application Knowledge Hub", "docs": "/docs",
        "documents": len(hub.docs), "chunks": len(hub.chunks), "applications": len(hub.apps),
        "users": {u: v["name"] for u, v in hub.users.items()},
    }


@app.get("/health", tags=["meta"])
def health():
    return {"status": "ok"}


@app.post("/search", tags=["retrieval"])
def search(req: SearchRequest, user: str = Depends(current_user)):
    """Hybrid search with per-signal scores so ranking is explainable."""
    return {"user": user, "results": hub.search(req.query, user, k=req.k, **_filters(req))}


@app.post("/ask", tags=["retrieval"])
def ask(req: AskRequest, user: str = Depends(current_user)):
    """Grounded answer with numbered citations, confidence, and staleness warnings."""
    return hub.ask(req.question, user, **_filters(req))


@app.get("/apps", tags=["knowledge graph"])
def list_apps():
    return [{"id": a["id"], "name": a["name"], "owner": a.get("owner") or None} for a in hub.apps.values()]


@app.get("/apps/{app_id}", tags=["knowledge graph"])
def get_app(app_id: str, user: str = Depends(current_user)):
    """Capabilities, dependencies, blast radius and the documents you can see."""
    if app_id not in hub.apps:
        raise HTTPException(404, f"No application '{app_id}'")
    return hub.app_view(app_id, user)


@app.get("/graph", tags=["knowledge graph"])
def graph():
    """Nodes and typed edges, ready for a graph DB (Neo4j / Cosmos Gremlin) or a UI."""
    nodes = [{"id": a, "type": "application", "name": x["name"]} for a, x in hub.apps.items()]
    nodes += [{"id": d.id, "type": "document", "name": d.title} for d in hub.docs.values()]
    edges = [{"from": a, "to": t, "type": "depends_on"} for a, x in hub.apps.items() for t in x.get("depends_on", [])]
    edges += [{"from": a, "to": t, "type": "integrates_with"} for a, x in hub.apps.items() for t in x.get("integrates_with", [])]
    edges += [{"from": d.id, "to": d.app, "type": "documents"} for d in hub.docs.values()]
    return {"nodes": nodes, "edges": edges}


@app.post("/ingest", tags=["ingestion"], status_code=201)
def ingest(doc: IngestDocument):
    """Upsert a connector-normalised document; it is chunked, embedded and indexed immediately."""
    if doc.app not in hub.apps:
        raise HTTPException(422, f"Unknown app '{doc.app}'. Register it first.")
    d = hub.ingest(doc.model_dump())
    return {"doc_id": d.id, "chunks": [c.id for c in d.chunks], "stale": d.is_stale(hub.today)}


@app.get("/quality", tags=["governance"])
def quality():
    """Ownership, freshness, ACL coverage, undocumented apps, and knowledge gaps (abstained questions)."""
    return hub.quality_report()


@app.get("/eval", tags=["governance"])
def evaluate(k: int = 5):
    """Retrieval evaluation over the golden set: recall@k, MRR, hit rate."""
    return hub.evaluate(k=k)


# ---------- Agent tool surface (MCP-style definitions) ----------

TOOLS = [
    {
        "name": "search_knowledge",
        "description": "Search governed enterprise application knowledge. Returns cited chunks.",
        "inputSchema": {"type": "object", "required": ["query"], "properties": {
            "query": {"type": "string"}, "app": {"type": "string"}, "k": {"type": "integer", "default": 5}}},
    },
    {
        "name": "answer_with_citations",
        "description": "Answer a question from enterprise knowledge with citations; abstains if evidence is weak.",
        "inputSchema": {"type": "object", "required": ["question"], "properties": {
            "question": {"type": "string"}, "app": {"type": "string"}}},
    },
    {
        "name": "get_application",
        "description": "Get an application's capabilities, dependencies, owner and blast radius.",
        "inputSchema": {"type": "object", "required": ["app_id"], "properties": {"app_id": {"type": "string"}}},
    },
]


@app.get("/agent/tools", tags=["agents"])
def agent_tools():
    """Tool definitions an agent framework (MCP, Semantic Kernel, LangGraph) can register."""
    return {"tools": TOOLS}


@app.post("/agent/tools/{name}", tags=["agents"])
def call_tool(name: str, args: dict, user: str = Depends(current_user)):
    """Invoke a tool on behalf of the signed-in user (on-behalf-of: same security trimming)."""
    try:
        if name == "search_knowledge":
            return {"content": hub.search(args["query"], user, k=args.get("k", 5), app=args.get("app"))}
        if name == "answer_with_citations":
            return {"content": hub.ask(args["question"], user, app=args.get("app"))}
        if name == "get_application":
            return {"content": hub.app_view(args["app_id"], user)}
    except KeyError as e:
        raise HTTPException(422, f"Missing or unknown argument: {e}")
    raise HTTPException(404, f"No tool '{name}'")
