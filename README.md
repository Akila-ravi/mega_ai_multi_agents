# Mega AI

## 5-minute setup
1. Copy env: already included as `.env`.
2. Run: `docker compose up --build`
3. API: `http://localhost:8000/docs`
4. Submit query (SSE): `curl -N -X POST http://localhost:8000/query -H "Content-Type: application/json" -d '{"query":"Explain Kubernetes scheduling"}'`

## Architecture
See [ARCHITECTURE.md](ARCHITECTURE.md)

## Services
- `api`: FastAPI (5 required endpoints)
- `worker`: async DB-backed queue processor
- `postgres`: persistence for jobs/traces/evals/prompt versions
- `logs`: pgAdmin lightweight log/query viewer

## Endpoints (exact 5)
- `POST /query` SSE streaming
- `GET /trace/{job_id}`
- `GET /eval/summary`
- `POST /prompt/approve`
- `POST /eval/retry`

## Limitations
- Web search is a stub for deterministic tests.
- Token streaming is event-based chunk streaming (not provider-native token callbacks).
- NL-to-SQL mapper is constrained and intentionally small for safety.

## AI collaboration attestation
- AI assistant used to scaffold architecture and generate implementation drafts.
- All runtime errors were manually validated in Docker and fixed iteratively.
