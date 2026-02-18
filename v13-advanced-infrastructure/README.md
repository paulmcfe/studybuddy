# StudyBuddy v13 — Advanced Infrastructure

AI-powered study assistant with multi-model support (OpenAI, Together AI, Ollama), per-task model routing, cost tracking, and MCP server exposure. Built with FastAPI, LangGraph, Next.js, PostgreSQL, and Qdrant.

## What's New in v13

- **Multi-Model Support** — Use OpenAI, Together AI (Llama 4, Mixtral), or Ollama (local) models
- **Per-Task Model Configuration** — Configure primary and fallback models for flashcards, tutoring, and curriculum
- **Automatic Fallback Chains** — Graceful degradation from open-source to proprietary models on failures
- **Cost Tracking Dashboard** — Monitor token usage and costs across providers
- **Performance Benchmarking** — Compare latency and throughput across models
- **MCP Server Exposure** — Other agents can call StudyBuddy via MCP (generate_flashcards, explain_concept, etc.)
- **Agent-to-Agent Discovery** — AgentCard endpoint for capability advertisement

All v12 features (authentication, rate limiting, monitoring) and v11 features (MCP connectors) are preserved.

## Quick Start (Local Development)

### Prerequisites

- Python 3.12+
- Node.js 20+
- PostgreSQL
- Qdrant (local or cloud)
- [uv](https://github.com/astral-sh/uv) (Python package manager)

### 1. Environment Setup

```bash
cp .env.example .env
# Edit .env with your API keys and database URL
```

### 2. Backend

```bash
uv venv && source .venv/bin/activate
uv sync
uvicorn api.index:app --reload
```

### 3. Frontend

```bash
cd frontend
npm install
npm run dev
```

The app runs at `http://localhost:3000` with the API at `http://localhost:8000`.

## Quick Start (Docker)

```bash
# Copy and configure environment
cp .env.example .env
# Edit .env with your OPENAI_API_KEY and JWT_SECRET_KEY

# Start all services
docker-compose up --build
```

This starts PostgreSQL, Qdrant, the API, and the frontend. The app is accessible at `http://localhost:3000`.

## Environment Variables

| Variable                          | Required | Description                                               |
| --------------------------------- | -------- | --------------------------------------------------------- |
| `OPENAI_API_KEY`                  | Yes      | OpenAI API key for LLM calls                              |
| `DATABASE_URL`                    | Yes      | PostgreSQL connection string                              |
| `QDRANT_URL`                      | Yes      | Qdrant vector database URL                                |
| `JWT_SECRET_KEY`                  | Yes      | Secret key for signing JWT tokens (change in production!) |
| `JWT_ALGORITHM`                   | No       | JWT algorithm (default: HS256)                            |
| `JWT_ACCESS_TOKEN_EXPIRE_MINUTES` | No       | Token expiry in minutes (default: 60)                     |
| `TOGETHER_API_KEY`                | No       | Together AI key for Llama/Mixtral (v13)                   |
| `OLLAMA_URL`                      | No       | Ollama server URL (default: http://localhost:11434)       |
| `LANGSMITH_API_KEY`               | No       | LangSmith tracing key                                     |
| `LANGSMITH_PROJECT`               | No       | LangSmith project name                                    |
| `BRAVE_SEARCH_API_KEY`            | No       | Brave Search API key for web search connector             |
| `GITHUB_TOKEN`                    | No       | GitHub token for repository connector                     |

## Authentication

### Endpoints

| Endpoint                           | Method | Description                          |
| ---------------------------------- | ------ | ------------------------------------ |
| `/api/auth/register`               | POST   | Create account, returns JWT          |
| `/api/auth/login`                  | POST   | Verify credentials, returns JWT      |
| `/api/auth/me`                     | GET    | Current user profile (requires auth) |
| `/api/auth/verify-email`           | POST   | Verify email address                 |
| `/api/auth/request-password-reset` | POST   | Request password reset               |
| `/api/auth/reset-password`         | POST   | Reset password with token            |

All other `/api/` endpoints (except `/api/health` and `/api/metrics`) require a valid JWT in the `Authorization: Bearer <token>` header.

### Default User

On first startup, a default user is created:
- Email: `default@studybuddy.local`
- Password: `studybuddy`

### Example

```bash
# Register a new user
curl -X POST http://localhost:8000/api/auth/register \
  -H "Content-Type: application/json" \
  -d '{"email": "user@example.com", "password": "mypassword"}'

# Login
curl -X POST http://localhost:8000/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email": "user@example.com", "password": "mypassword"}'

# Use the returned token for authenticated requests
curl http://localhost:8000/api/programs \
  -H "Authorization: Bearer <token>"
```

## Monitoring

- **Health Check**: `GET /api/health` — Returns status of PostgreSQL and Qdrant connections
- **Metrics**: `GET /api/metrics` — Prometheus-format request metrics (counts, latency, errors)

## Rate Limits

| Endpoint             | Limit     |
| -------------------- | --------- |
| Register             | 5/minute  |
| Login                | 10/minute |
| Chat                 | 30/minute |
| Flashcard Generation | 20/minute |
| Document Upload      | 10/minute |
| Password Reset       | 5/minute  |
| Benchmark Run        | 5/hour    |

Rate limits are per-user (identified by JWT). Unauthenticated requests are limited by IP. In-memory storage (resets on restart).

## v13 Multi-Model API

### Model Configuration
- `GET /api/models` — List available models with provider status
- `GET /api/models/config` — Get user's model configuration
- `PUT /api/models/config` — Update per-task model assignments

### Cost Tracking
- `GET /api/costs` — Cost summary for authenticated user
- `GET /api/costs/comparison` — Provider cost comparison (hypothetical)

### Benchmarking
- `POST /api/benchmark/run` — Run performance benchmark (rate limited)
- `GET /api/benchmark/history` — Historical benchmark results

### Agent Discovery
- `GET /api/agent-card` — A2A agent capability card

## LangGraph Cloud Deployment

The tutor agent can be deployed as a standalone API via LangGraph Cloud:

```bash
langgraph up
```

Configuration is in `langgraph.json`. The entry point is `api/agents/tutor.py:create_tutor_graph`.

## API Endpoints

### Programs
- `GET /api/programs` — List all programs
- `POST /api/programs` — Create a program
- `GET /api/programs/{id}` — Get program details
- `PATCH /api/programs/{id}` — Update a program
- `DELETE /api/programs/{id}` — Archive a program
- `GET /api/programs/{id}/stats` — Get program statistics

### Documents
- `GET /api/programs/{id}/documents` — List documents
- `POST /api/programs/{id}/documents` — Upload a document
- `DELETE /api/programs/{id}/documents/{doc_id}` — Delete a document

### Flashcards
- `GET /api/programs/{id}/flashcards` — List flashcards
- `GET /api/programs/{id}/due-cards` — Get cards due for review
- `POST /api/programs/{id}/flashcards/generate` — Generate a flashcard
- `POST /api/programs/{id}/flashcards/{card_id}/review` — Record a review

### Curriculum
- `POST /api/programs/{id}/generate-curriculum` — Generate AI curriculum

### Connectors
- `GET /api/programs/{id}/connectors` — List connectors
- `POST /api/programs/{id}/connectors` — Create a connector
- `PATCH /api/programs/{id}/connectors/{cid}` — Update connector config
- `DELETE /api/programs/{id}/connectors/{cid}` — Delete a connector
- `POST /api/programs/{id}/connectors/{cid}/fetch` — Import a URL
- `GET /api/programs/{id}/connectors/{cid}/github/files` — Browse repo files
- `POST /api/programs/{id}/connectors/{cid}/github/import` — Import GitHub files
- `POST /api/programs/{id}/connectors/{cid}/github/sync` — Re-sync GitHub repo

### Chat
- `POST /api/programs/{id}/chat` — Chat with the tutor (streaming SSE)

## Project Structure

```
v13-advanced-infrastructure/
├── api/
│   ├── index.py               # FastAPI app with all endpoints
│   ├── auth.py                # JWT authentication & password hashing
│   ├── rate_limit.py          # slowapi rate limiter
│   ├── agents/
│   │   └── tutor.py           # LangGraph tutor agent + Cloud factory
│   ├── database/
│   │   ├── models.py          # SQLAlchemy models (+ ModelConfig, TokenUsage, BenchmarkResult)
│   │   └── connection.py      # DB init and migrations
│   ├── services/
│   │   ├── monitoring.py      # Prometheus metrics collector
│   │   ├── flashcard.py       # Flashcard generation with ModelRouter
│   │   ├── curriculum.py      # Curriculum generation with ModelRouter
│   │   ├── retrieval.py       # Document retrieval
│   │   ├── connectors/        # MCP connector implementations
│   │   └── models/            # v13: Multi-model infrastructure
│   │       ├── config.py      # Model catalog & pricing
│   │       ├── ollama_client.py # Ollama integration
│   │       ├── together_client.py # Together AI integration
│   │       ├── fallback.py    # ModelFallbackChain
│   │       ├── router.py      # Task-based model routing
│   │       └── benchmark.py   # Performance benchmarking
│   └── mcp_server/            # v13: MCP Server exposure
│       ├── server.py          # FastMCP tools
│       └── agent_card.py      # A2A capability card
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── CostDashboard.tsx    # v13: Cost tracking UI
│   │   │   ├── ModelSettings.tsx    # v13: Model configuration UI
│   │   │   ├── BenchmarkResults.tsx # v13: Benchmark UI
│   │   │   └── ... (existing components)
│   │   └── lib/
│   │       └── api.ts         # Authenticated fetch wrapper
│   └── Dockerfile
├── docker-compose.yml         # Full stack with Ollama
├── Dockerfile.api             # API container
├── langgraph.json             # LangGraph Cloud config
├── pyproject.toml             # Python dependencies (v13.0.0)
├── .env.example               # Environment template with Together AI, Ollama
└── README.md
```
