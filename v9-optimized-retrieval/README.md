# StudyBuddy v9 - Advanced Retrieval & Evaluation

An intelligent AI Engineering tutor with advanced retrieval capabilities and comprehensive evaluation infrastructure for measuring improvements.

## What's New in v9

StudyBuddy v9 transforms basic vector search into a sophisticated retrieval pipeline:

- **Semantic Chunking**: Markdown-aware + embedding-based chunking that respects document structure
- **Hybrid Search**: Dense vector search + BM25 keyword matching with Reciprocal Rank Fusion
- **Cohere Reranking**: Cross-encoder reranking for precision on the final results
- **RAG-Fusion**: Multi-query expansion for comprehensive coverage on complex questions
- **Adaptive Retriever**: Automatically selects strategy based on query complexity
- **Retrieval Comparison Tool**: Side-by-side comparison with RAGAS metrics

## Features

- **All v8 Features**: Evaluation infrastructure, synthetic test data, LangSmith integration
- **All v7 Features**: Deep research, planning, focus areas, markdown support
- **Semantic Chunking**: Preserves code examples and keeps explanations together
- **Hybrid Search**: Catches exact matches that dense search misses
- **Reranking**: Best results surface to the top
- **RAG-Fusion**: Multiple query angles for research-style questions
- **Retrieval Comparison**: Quantify retrieval improvements with RAGAS metrics

## Architecture

```
                    ┌─────────────┐
                    │  Supervisor │
                    └──────┬──────┘
                           │
         ┌─────────────────┼─────────────────┐
         │         │       │       │         │
         ▼         ▼       ▼       ▼         ▼
    ┌────────┐ ┌───────┐ ┌───────┐ ┌─────────┐
    │ Tutor  │ │ Card  │ │Quality│ │Scheduler│
    └────────┘ │  Gen  │ │Checker│ └─────────┘
         │     └───────┘ └───────┘
         │
         ▼
┌────────────────────────────────────────────┐
│           Advanced Retrieval (v9)          │
│  ┌──────────────────────────────────────┐  │
│  │  Semantic Chunking → Hybrid Search   │  │
│  │  → Cohere Reranking → RAG-Fusion     │  │
│  │  → Adaptive Strategy Selection       │  │
│  └──────────────────────────────────────┘  │
└────────────────────────────────────────────┘
         │
         ▼
┌────────────────────────────────────────────┐
│         Evaluation Module (v8)             │
│  ┌──────────────────────────────────────┐  │
│  │ Testset Generator | Dataset Builder  │  │
│  │ Custom Evaluators | Baseline Runner  │  │
│  │ Dashboard Metrics | Comparison Tool  │  │
│  └──────────────────────────────────────┘  │
└────────────────────────────────────────────┘
```

## Prerequisites

- Python 3.12+
- Node.js 18+
- OpenAI API key
- Cohere API key (for reranking)
- LangSmith API key (for evaluation)

## Setup

### 1. Navigate to the directory

```bash
cd studybuddy/v9-optimized-retrieval
```

### 2. Create virtual environment

```bash
uv sync
```

### 3. Configure environment variables

Create or update `.env`:

```
OPENAI_API_KEY=sk-your-key-here
COHERE_API_KEY=your-cohere-key-here
LANGSMITH_API_KEY=lsv2_your-key-here
LANGSMITH_PROJECT=studybuddy-v9
```

### 4. Set up the frontend

```bash
cd frontend
npm install
cd ..
```

### 5. Run the app (two terminals)

**Terminal 1 - Backend:**
```bash
uv run uvicorn api.index:app --reload --port 8000
```

**Terminal 2 - Frontend:**
```bash
cd frontend
npm run dev
```

### 6. Open the app

Visit `http://localhost:3000` in your browser.

## Retrieval Strategies

v9 provides multiple retrieval strategies that can be selected automatically or manually:

| Strategy | Latency | Best For |
|----------|---------|----------|
| `simple` | 20-50ms | Baseline comparison |
| `hybrid` | 40-80ms | Queries with specific terms |
| `reranked` | 100-250ms | When precision is critical |
| `fusion` | 200-500ms | Complex, multi-faceted queries |
| `adaptive` | Varies | Automatic selection (default) |

## API Endpoints

### NEW in v9: Retrieval Comparison

```bash
curl -X POST http://127.0.0.1:8000/api/evaluation/compare-retrievers \
  -H "Content-Type: application/json" \
  -d '{"queries": ["What is RAG?", "Compare embeddings and BM25"], "k": 5}'
```

Returns side-by-side comparison of simple vs adaptive retrieval with latency metrics.

### Evaluation Endpoints (v8)

```bash
# Generate synthetic test data
curl -X POST http://127.0.0.1:8000/api/evaluation/generate-testset \
  -H "Content-Type: application/json" \
  -d '{"test_size": 50}'

# Run evaluation
curl -X POST http://127.0.0.1:8000/api/evaluation/run \
  -H "Content-Type: application/json" \
  -d '{"dataset_name": "tutoring-eval", "eval_type": "all"}'

# Get dashboard metrics
curl http://127.0.0.1:8000/api/evaluation/dashboard
```

### Core Endpoints

- `POST /api/chat` - Chat with the tutor
- `GET/POST /api/focus-areas` - Manage focus areas
- `POST /api/flashcard` - Generate flashcards
- `GET /api/due-cards` - Get cards due for review
- `POST /api/review` - Record flashcard review
- `GET /api/stats` - Get learning statistics

## Project Structure

```
v9-optimized-retrieval/
├── api/
│   ├── __init__.py
│   ├── index.py              # FastAPI app + all endpoints
│   ├── state.py
│   ├── database/
│   │   ├── models.py
│   │   └── connection.py
│   ├── agents/
│   │   └── ...
│   ├── services/
│   │   ├── indexing.py       # NEW: Semantic chunking
│   │   ├── retrieval.py      # NEW: Advanced retrievers
│   │   └── ...
│   └── evaluation/
│       ├── retrieval_comparison.py  # NEW: Comparison tool
│       └── ...               # v8 evaluation module
├── documents/
│   ├── topic-list.md         # Chapter/topic structure
│   └── ref-*.md              # Knowledge base documents
├── frontend/                  # Next.js frontend
│   ├── src/
│   │   ├── app/
│   │   │   ├── layout.tsx
│   │   │   ├── page.tsx      # Main app with state management
│   │   │   └── globals.css
│   │   └── components/
│   │       ├── HomeScreen.tsx     # Chapter selection + curriculum
│   │       ├── StudyScreen.tsx    # Flashcard display + actions
│   │       ├── Flashcard.tsx      # Card with flip animation
│   │       ├── ChatPanel.tsx      # Slide-up chat
│   │       ├── Sidebar.tsx        # Desktop sidebar with focus areas
│   │       ├── CurriculumModal.tsx # Learning path creation
│   │       ├── FocusAreas.tsx     # Struggle topic display
│   │       └── LoadingDots.tsx
│   ├── public/images/         # Favicon and icons
│   ├── next.config.ts         # API proxy config
│   └── package.json
├── .env
├── pyproject.toml
└── README.md
```

## Differences from v8

| Aspect | v8 | v9 |
|--------|----|----|
| Chunking | Fixed 1000 chars | Semantic (markdown + embedding) |
| Retrieval | Vector search only | Hybrid + reranking + RAG-Fusion |
| Strategy | Single | Adaptive selection |
| Comparison | None | Side-by-side with metrics |
| Dependencies | Base | + cohere, langchain-experimental |

## Measuring Improvements

The comparison endpoint lets you quantify retrieval improvements:

1. **Generate testset** (v8): Create synthetic test cases
2. **Compare retrievers** (v9): Run same queries through old vs new
3. **Measure with RAGAS**: Get context_precision and context_recall scores
4. **Document gains**: Track improvements over baseline

## Important Notes

- Evaluation endpoints are disabled on Vercel (too compute-heavy)
- Run evaluations locally during development
- Cohere reranking requires a Cohere API key (free tier available)
- The adaptive retriever adds latency for complex queries

## Cost Considerations

v9 adds additional API costs:

- **Semantic chunking**: More embedding calls during indexing
- **Cohere reranking**: Per-query cost for reranking calls
- **RAG-Fusion**: Multiple LLM calls for query expansion

Monitor usage at the respective provider dashboards.
