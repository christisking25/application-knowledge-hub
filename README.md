# Application Knowledge Hub

A small, working **context layer for AI agents**. It takes enterprise knowledge (Confluence, Jira,
SharePoint, ServiceNow and an application registry), turns it into governed, searchable context,
and serves it to agents through an API, with security trimming, citations and evaluation built in.

The goal is to show the *architecture* end to end in about 600 lines with no cloud dependencies.
Each component lines up with an Azure service that would replace it in production.

## Run it

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/uvicorn app.main:app --reload
# open http://127.0.0.1:8000/docs
.venv/bin/pytest -q
```

Set the `X-User` header to switch identity: `alice` (architect), `sam` (store ops), `riley` (SRE).

## What it does

| Capability | How it works here | Production equivalent |
|---|---|---|
| **Ingestion** | `POST /ingest` upserts connector-normalised docs. They are split on headings, windowed with overlap, and each chunk is enriched with its title, app and section before embedding | Confluence/Jira/SharePoint/ServiceNow connectors → Fabric / Databricks pipelines → Azure AI Search indexers |
| **Hybrid retrieval** | BM25 + vector similarity fused with Reciprocal Rank Fusion, plus metadata filters (`app`, `source`, `doc_type`) | Azure AI Search hybrid + semantic ranker |
| **Relationship-based retrieval** | Entity linking finds apps named in the query and boosts their docs and their graph neighbours. An `app` filter expands to dependencies | Knowledge graph (Neo4j / Cosmos DB Gremlin / Fabric graph) |
| **Taxonomy** | A glossary expands queries ("down" → unavailable/outage/timeout) | Purview business glossary |
| **Security trimming** | Each doc carries an ACL. Chunks the user can't read are removed *before* ranking, so they can't change scores or leak into snippets | Entra ID groups + AI Search security filters, on-behalf-of tokens for agents |
| **Grounded answers** | `POST /ask` picks the best sentences, cites them `[1]`, scores its confidence and **abstains** when evidence is weak | Azure OpenAI with the same retrieval, citation and abstain contract |
| **Freshness** | Each doc type has an SLA. Stale docs get a ranking penalty and a warning on the answer | Lifecycle policies + re-crawl schedules |
| **Explainability** | Every hit returns its `bm25`, `vector`, `graph_boost`, `freshness_penalty` and `term_coverage` scores | Search debug / tracing in AI Foundry |
| **Quality & gaps** | `GET /quality` reports missing owners, stale docs, missing ACLs, undocumented apps and **knowledge gaps** (questions it abstained on) | Purview data quality + Foundry monitoring |
| **Evaluation** | `GET /eval` runs a golden set and reports recall@k, MRR and hit rate | AI Foundry evaluations (groundedness, relevance) in CI |
| **Agent tools** | `GET /agent/tools` returns MCP-style tool schemas. `POST /agent/tools/{name}` runs a tool as the calling user | MCP server / Semantic Kernel plugin / Copilot Studio action |
| **Impact analysis** | `GET /apps/{id}` shows capabilities, dependencies and the transitive blast radius (`impact_if_down`) | Application portfolio management (APM) + graph |

## Demo script (2 minutes)

```bash
H='-H Content-Type:application/json'
# 1. Hybrid + graph retrieval: the architecture doc and the Jira incident both come back, with per-signal scores
curl -s localhost:8000/ask $H -H 'X-User: alice' -d '{"question":"What happens at checkout if the pricing service is down?"}'

# 2. Security trimming: same question, different user, different answer
curl -s localhost:8000/ask $H -H 'X-User: alice' -d '{"question":"can apps log card numbers"}'   # answered from SP-501
curl -s localhost:8000/ask $H -H 'X-User: sam'   -d '{"question":"can apps log card numbers"}'   # abstains: no access

# 3. Staleness warning on a 20-month-old design doc
curl -s localhost:8000/ask $H -d '{"question":"how does the pricing engine compute markdowns"}'

# 4. Blast radius
curl -s localhost:8000/apps/pricing

# 5. Governance + eval
curl -s localhost:8000/quality
curl -s localhost:8000/eval
```

## Layout

```
app/hub.py      core: chunking, embeddings, BM25, RRF fusion, graph, RBAC, grounding, quality, eval
app/main.py     FastAPI routes + agent tool surface
data/seed.json  sample knowledge: 7 apps, 9 docs from 5 sources, 3 users, golden eval set
tests/          9 API tests covering retrieval, RBAC, abstain, staleness, ingest, graph, eval
```

## Design choices

- **Trim first, then rank.** Filtering after ranking can leak restricted content through scores,
  snippets or "no results" timing. Here the candidate pool is built from authorised chunks only.
- **Abstaining beats guessing.** Confidence combines how many query terms the top hit covers with
  its vector similarity. Below the threshold the API returns `answer: null` and logs the question
  as a knowledge gap for content owners to fill.
- **The embedder is pluggable.** `HashingEmbedder` (hashed words, bigrams and 4-char grams) keeps the
  demo free of API keys. Any object with `embed(text)` drops in, e.g. Azure OpenAI
  `text-embedding-3-large`.
- **Context enrichment.** Chunks are embedded together with their document title, app name and
  section heading, so a chunk like "restart the consumer group" still matches "InventorySync".

## Limitations (honest notes)

- Everything is in memory. Production would use Azure AI Search for the index and a graph store for relationships.
- The golden set is small (6 cases) and was written alongside the seed data, so the 1.0 scores show
  the harness works, not that the approach generalises. The next step is a larger set built from real queries.
- `X-User` is a stand-in for validated Entra ID tokens.
- All sample data is fictional.
