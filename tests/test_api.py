from fastapi.testclient import TestClient

from app.main import app, hub

client = TestClient(app)


def post(path, body, user="alice"):
    return client.post(path, json=body, headers={"X-User": user})


def test_hybrid_search_finds_architecture_and_incident():
    r = post("/search", {"query": "What happens at checkout if the pricing service is down?"})
    ids = [h["doc_id"] for h in r.json()["results"]]
    assert {"CONF-101", "JIRA-OH-482"} <= set(ids[:3])
    assert set(r.json()["results"][0]["signals"]) >= {"bm25", "vector", "graph_boost", "freshness_penalty"}


def test_security_trimming_hides_restricted_docs():
    # SP-501 is architecture-only; Sam (store ops) must never see it.
    ids = [h["doc_id"] for h in post("/search", {"query": "can apps log card numbers"}, "sam").json()["results"]]
    assert "SP-501" not in ids
    assert post("/ask", {"question": "can apps log card numbers"}, "sam").json()["abstained"]
    assert not post("/ask", {"question": "can apps log card numbers"}, "alice").json()["abstained"]


def test_answer_is_cited_and_abstains_off_topic():
    a = post("/ask", {"question": "Are loyalty points reversed when an order is returned?"}, "sam").json()
    assert not a["abstained"] and a["citations"]
    assert "[1]" in a["answer"] or "[2]" in a["answer"]
    off = post("/ask", {"question": "what is on the cafeteria menu today"}).json()
    assert off["abstained"] and off["answer"] is None
    assert any(g["question"].startswith("what is on the cafeteria") for g in client.get("/quality").json()["knowledge_gaps"])


def test_stale_source_flagged():
    a = post("/ask", {"question": "how does the pricing engine compute markdowns"}).json()
    assert any(c["doc_id"] == "CONF-201" and c["stale"] for c in a["citations"])
    assert a["warnings"]


def test_graph_impact_analysis():
    r = client.get("/apps/pricing", headers={"X-User": "alice"}).json()
    assert set(r["impact_if_down"]) == {"orderhub", "storeops"}


def test_ingest_then_retrieve():
    doc = {
        "id": "JIRA-LP-77", "source": "jira", "doc_type": "ticket", "app": "loyalty",
        "title": "LP-77: Double points weekend config", "url": "https://jira.example.com/browse/LP-77",
        "owner": "Customer Tech", "updated_at": hub.today.isoformat(), "acl": ["all-staff"],
        "body": "## Description\nDouble points weekend multiplier is configured in the promo-rules table and expires Sunday at midnight.",
    }
    assert client.post("/ingest", json=doc).status_code == 201
    ids = [h["doc_id"] for h in post("/search", {"query": "double points weekend multiplier"}, "sam").json()["results"]]
    assert ids[0] == "JIRA-LP-77"


def test_eval_and_quality():
    e = client.get("/eval").json()
    assert e["recall_at_k"] >= 0.8 and e["mrr"] >= 0.8
    types = {i["type"] for i in client.get("/quality").json()["issues"]}
    assert {"missing_owner", "stale", "app_missing_owner"} <= types


def test_agent_tool_call_respects_user():
    assert {t["name"] for t in client.get("/agent/tools").json()["tools"]} >= {"search_knowledge", "get_application"}
    r = client.post("/agent/tools/search_knowledge", json={"query": "card numbers PCI"}, headers={"X-User": "sam"})
    assert all(h["doc_id"] != "SP-501" for h in r.json()["content"])


def test_unknown_user_rejected():
    assert post("/search", {"query": "x"}, "mallory").status_code == 401
