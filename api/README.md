# StudyRAG API

A separate FastAPI service that sits next to the existing Streamlit UI and
CLI. It is incremental — neither the Streamlit app nor the CLI was
modified, and they keep working without this service running.

## What it does today

- `GET /health` — liveness + readiness, reports API version, whether
  `DATABASE_URL` is configured, and how many chunks are in the shared
  vector store.
- Pydantic Settings reads env vars (including a reserved `DATABASE_URL`
  for future persistence).
- Database connection is **optional** — the API boots and serves health
  checks with no Postgres available.

## Run it

From the repo root:

```bash
pip install -r requirements-api.txt
uvicorn api.main:app --reload
```

Then open http://127.0.0.1:8000/health or the auto-generated docs at
http://127.0.0.1:8000/docs.

## Configuration (env vars)

| Variable | Default | Meaning |
|---|---|---|
| `api_title` | `StudyRAG API` | FastAPI app title |
| `api_version` | `0.1.0` | Reported by `/health` |
| `DATABASE_URL` | (unset) | Reserved for Phase 2. If unset, no DB engine is created. |
| `vector_db_path` | `./data/chroma_db` | Path to the shared SQLite vector store |

## Architecture note

The API reuses `src/rag/embeddings.EmbeddingStore` directly (read-only in
the health probe) so the Streamlit ingest flow and the API always see the
same chunks. When `query` and `ingest` endpoints land, they'll call the
same `EmbeddingStore` — no data duplication.
