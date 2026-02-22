# StudyBuddy v14 — Production Hardening

AI-powered learning platform with production-grade security, caching, rate limiting, and monitoring. Built with FastAPI, Next.js, LangGraph, Qdrant, PostgreSQL.

## Quick Start

```bash
# 1. Set up environment
cp .env.example .env  # Add your OPENAI_API_KEY

# 2. Backend terminal
cd v14-production-hardening
uv sync
uvicorn api.index:app --reload

# 3. Frontend terminal
cd v14-production-hardening/frontend
npm run dev

# 3. Start services
docker compose up

# 4. Access
# Frontend: http://localhost:3000
# API: http://localhost:8000
# HTTPS (with nginx): https://localhost
```

## Architecture

- **FastAPI backend** with LangGraph-based tutor agent
- **Next.js frontend** with dashboard components for costs, alerts, and production readiness
- **PostgreSQL** for relational data (users, programs, flashcards, audit logs, cost records)
- **Qdrant** for vector search (hybrid BM25 + dense + Cohere reranking)
- **Nginx** reverse proxy for TLS termination with HSTS headers

## What's New in v14

### Content Guardrails

- **Input guardrails**: prompt injection detection (12 regex patterns), inappropriate content filtering
- **Output guardrails**: age-inappropriate content redaction
- Integrated into LangGraph agent as `input_guard` -> `agent` -> `output_guard` nodes

### Comprehensive Caching

- **Semantic response cache**: cosine similarity matching (threshold 0.92) with 24h TTL
- **Database-backed embedding cache**: SHA-256 content hashing to avoid re-embedding
- **Batch document indexing**: 100 chunks per batch for efficient ingestion

### Cost Analytics

- Per-feature cost breakdown (`/api/costs/by-feature`)
- Daily cost time series (`/api/costs/daily`)
- Budget alerts with configurable thresholds
- Frontend dashboard with bar charts and budget controls

### Distributed Security

- API key authentication for MCP inter-agent communication
- Immutable audit logging for security events
- TLS/HTTPS via nginx with HSTS headers
- SHA-256 hashed API keys with scopes and expiry

### Tiered Rate Limiting

- Per-tier limits: `free`, `pro`, `enterprise`
- Dynamic limit resolution per request based on user tier
- Categories: `auth`, `chat`, `flashcard_generate`, `document_upload`, `benchmark`
- Graceful degradation threshold at 80%

### Production Checklist

- Automated readiness checks across 5 categories: security, performance, monitoring, database, external
- Color-coded status (green/yellow/red) with overall rollup
- Frontend dashboard component

### Monitoring & Alerting

- P95/P99 latency tracking
- Error rate computation
- Alert rules for budget (80%/95%), error rate (5%/15%), latency (5s/15s)
- Prometheus-format metrics endpoint

### Load Testing

```bash
pip install locust
locust -f tests/locustfile.py --host=http://localhost:8000
```

## Configuration

| Variable               | Required | Description                                         |
| ---------------------- | -------- | --------------------------------------------------- |
| `OPENAI_API_KEY`       | Yes      | OpenAI API key for LLM calls                        |
| `DATABASE_URL`         | Yes      | PostgreSQL connection string                        |
| `QDRANT_URL`           | Yes      | Qdrant vector database URL                          |
| `JWT_SECRET`           | Yes      | Secret key for signing JWT tokens                   |
| `JWT_ALGORITHM`        | No       | JWT algorithm (default: HS256)                      |
| `JWT_EXPIRATION_HOURS` | No       | Token expiry in hours                               |
| `TOGETHER_API_KEY`     | No       | Together AI key for Llama/Mixtral                   |
| `OLLAMA_URL`           | No       | Ollama server URL (default: http://localhost:11434) |
| `COHERE_API_KEY`       | No       | Cohere key for reranking                            |

## API Endpoints (v14 Additions)

### Cost Analytics
- `GET /api/costs/by-feature` — Per-feature cost breakdown
- `GET /api/costs/daily` — Daily cost time series

### Budget Management
- `GET /api/budget` — Budget configuration
- `PUT /api/budget` — Create/update budget
- `GET /api/budget/status` — Spend vs budget

### API Key Administration
- `POST /api/admin/api-keys` — Create API key
- `GET /api/admin/api-keys` — List API keys
- `DELETE /api/admin/api-keys/{id}` — Revoke key

### Audit & Monitoring
- `GET /api/admin/audit-logs` — Query audit logs
- `GET /api/production-checklist` — Run readiness checks
- `GET /api/alerts` — Current triggered alerts
- `GET /api/alerts/rules` — List alert rules

### Existing Endpoints (preserved from v12/v13)
- `POST /api/auth/register`, `/api/auth/login`, `/api/auth/me`
- `GET/POST /api/programs`, `GET/PATCH/DELETE /api/programs/{id}`
- `POST /api/programs/{id}/documents`, `POST /api/programs/{id}/flashcards/generate`
- `POST /api/programs/{id}/chat` — Streaming SSE tutor chat
- `GET /api/models`, `PUT /api/models/config`, `POST /api/benchmark/run`
- `GET /api/health`, `GET /api/metrics`

## TLS Setup (Development)

```bash
./scripts/generate-certs.sh
docker compose up
# Access https://localhost
```

## Project Structure

```
v14-production-hardening/
├── api/
│   ├── index.py                      # FastAPI app with all endpoints
│   ├── auth.py                       # JWT authentication & password hashing
│   ├── rate_limit.py                 # Tiered rate limiting (free/pro/enterprise)
│   ├── agents/
│   │   └── tutor.py                  # LangGraph tutor with guardrail nodes
│   ├── database/
│   │   ├── models.py                 # SQLAlchemy models (+ AuditLog, ApiKey, Budget)
│   │   └── connection.py             # DB init and migrations
│   ├── services/
│   │   ├── guardrails.py             # Input/output content guardrails
│   │   ├── semantic_cache.py         # Cosine-similarity response cache
│   │   ├── cached_embeddings.py      # SHA-256 embedding dedup cache
│   │   ├── indexing.py               # Batch document indexing
│   │   ├── alerting.py               # Alert rules and evaluation
│   │   ├── api_keys.py               # API key management (SHA-256 hashed)
│   │   ├── audit.py                  # Immutable audit logging
│   │   ├── production_checklist.py   # Readiness checks (5 categories)
│   │   ├── monitoring.py             # Prometheus metrics collector
│   │   ├── flashcard.py              # Flashcard generation
│   │   ├── curriculum.py             # Curriculum generation
│   │   ├── retrieval.py              # Document retrieval
│   │   ├── connectors/               # MCP connector implementations
│   │   └── models/                   # Multi-model infrastructure
│   │       ├── config.py             # Model catalog & pricing
│   │       ├── router.py             # Task-based model routing
│   │       ├── fallback.py           # ModelFallbackChain
│   │       ├── benchmark.py          # Performance benchmarking
│   │       ├── ollama_client.py      # Ollama integration
│   │       └── together_client.py    # Together AI integration
│   └── mcp_server/                   # MCP Server exposure
│       ├── server.py                 # FastMCP tools
│       └── agent_card.py             # A2A capability card
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── CostAnalytics.tsx     # v14: Cost analytics dashboard
│   │   │   ├── ProductionChecklist.tsx # v14: Readiness checks UI
│   │   │   ├── CostDashboard.tsx     # Cost tracking UI
│   │   │   ├── ModelSettings.tsx     # Model configuration UI
│   │   │   ├── BenchmarkResults.tsx  # Benchmark UI
│   │   │   └── ...                   # Auth, Study, Flashcard, etc.
│   │   └── lib/
│   │       └── api.ts               # Authenticated fetch wrapper
│   └── Dockerfile
├── tests/
│   └── locustfile.py                 # Locust load testing scenarios
├── scripts/
│   └── generate-certs.sh             # TLS certificate generation
├── nginx.conf                        # Reverse proxy with TLS & HSTS
├── docker-compose.yml                # Full stack (API, frontend, PG, Qdrant, nginx)
├── Dockerfile.api                    # API container
├── langgraph.json                    # LangGraph Cloud config
├── pyproject.toml                    # Python dependencies
└── .env.example                      # Environment template
```
