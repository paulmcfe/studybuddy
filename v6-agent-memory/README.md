# StudyBuddy v6 - Agent Memory and Spaced Repetition

An intelligent AI Engineering tutor with persistent memory, flashcard caching, and spaced repetition powered by the SM-2 algorithm.

## What's New in v6

StudyBuddy v6 transforms from an ephemeral multi-agent system (v5) into a truly persistent learning companion:

- **PostgreSQL Persistence**: All data survives restarts in a real database
- **Content-Addressed Flashcard Caching**: Same topic + context = instant results (no regeneration)
- **SM-2 Spaced Repetition**: Optimal review scheduling based on your performance
- **Scheduler Agent**: New agent that manages due cards and learning statistics
- **Custom Memory Store**: Remember user preferences, struggles, and goals across sessions
- **Next.js Frontend**: Modern React-based UI with three-button review system

## Features

- **Multi-Agent Coordination**: Supervisor routes to Tutor, Card Generator, Quality Checker, or Scheduler
- **Persistent Flashcards**: Generated cards are cached and reused
- **Smart Review Scheduling**: Cards you struggle with come back sooner
- **Learning Analytics**: Track your progress, identify struggle areas
- **LangSmith Tracing**: Full observability of agent interactions

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
    └───┬────┘ │  Gen  │ │Checker│ └─────────┘
        │      └───┬───┘ └───────┘      │
        │          │          ▲         │
        │          └──────────┘         │
        └───────────────────────────────┘
                           │
                    ┌──────┴──────┐
                    │ PostgreSQL  │
                    │  Database   │
                    └─────────────┘
```

## Prerequisites

- Python 3.12+
- Node.js 18+
- PostgreSQL (via Homebrew: `brew install postgresql@16`)
- OpenAI API key
- LangSmith API key (optional, for tracing)

## Setup

### 1. Navigate to the directory

```bash
cd studybuddy/v6-agent-memory
```

### 2. Create virtual environment

```bash
uv sync
```

Or with pip:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e .
```

### 3. Set up PostgreSQL

```bash
# Start PostgreSQL
brew services start postgresql@16

# Create the database
createdb studybuddy
```

### 4. Configure environment variables

Create or update `.env`:

```
OPENAI_API_KEY=sk-your-key-here
POSTGRES_URL=postgresql://localhost/studybuddy
LANGSMITH_API_KEY=lsv2_your-key-here
```

### 5. Install frontend dependencies

```bash
cd frontend
npm install
cd ..
```

### 6. Run the app (two terminals)

**Terminal 1 - Backend:**
```bash
cd v6-agent-memory
uv run uvicorn api.index:app --reload --port 8000
```

**Terminal 2 - Frontend:**
```bash
cd v6-agent-memory/frontend
npm run dev
```

### 7. Open the app

Visit `http://localhost:3000` in your browser. The Next.js dev server proxies `/api/*` requests to the FastAPI backend on port 8000.

## API Endpoints

### Chat

```
POST /api/chat
{
    "message": "Explain RAG to me",
    "card_context": null
}
```

Response includes reply, cards, cache_hit status, due_cards, and study_stats.

### Flashcard Generation

```
POST /api/flashcard
{
    "chapter_id": 1,
    "scope": "single",
    "current_topic": null
}
```

### Get Due Cards (NEW in v6)

```
GET /api/due-cards?limit=10&include_new=true
```

Returns cards due for review plus new cards to learn.

### Record Review (NEW in v6)

```
POST /api/review
{
    "flashcard_id": "uuid-here",
    "quality": 4
}
```

Quality ratings (SM-2):
- 0: Complete blackout
- 1: Incorrect but recognized answer
- 2: Incorrect but seemed easy
- 3: Correct with difficulty
- 4: Correct after hesitation
- 5: Perfect recall

### Get Statistics (NEW in v6)

```
GET /api/stats
```

Returns total reviews, due cards, topic performance, and struggle areas.

### Cache Statistics (NEW in v6)

```
GET /api/cache-stats
```

Returns flashcard cache hit statistics.

## Project Structure

```
v6-agent-memory/
├── api/
│   ├── __init__.py
│   ├── index.py              # FastAPI app + multi-agent graph
│   ├── state.py              # Shared state definitions
│   ├── database/
│   │   ├── __init__.py
│   │   ├── models.py         # SQLAlchemy models
│   │   └── connection.py     # PostgreSQL setup
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── tutor.py
│   │   ├── card_generator.py
│   │   ├── quality_checker.py
│   │   ├── supervisor.py
│   │   └── scheduler.py      # Spaced repetition scheduling
│   └── services/
│       ├── __init__.py
│       ├── flashcard_cache.py
│       ├── spaced_repetition.py
│       └── memory_store.py
├── documents/
│   ├── topic-list.md
│   └── ref-*.md
├── frontend/                  # Next.js frontend
│   ├── src/
│   │   ├── app/
│   │   │   ├── layout.tsx
│   │   │   ├── page.tsx
│   │   │   └── globals.css
│   │   └── components/
│   │       ├── HomeScreen.tsx
│   │       ├── StudyScreen.tsx
│   │       ├── Flashcard.tsx
│   │       ├── ChatPanel.tsx
│   │       ├── ReviewButtons.tsx  # No/Took a sec/Yes
│   │       └── ...
│   ├── next.config.ts
│   └── package.json
├── .env
├── pyproject.toml
└── README.md
```

## Database Schema

- **users**: User profiles and preferences
- **memories**: Learning preferences, struggles, goals
- **flashcards**: Cached flashcards with content hashes
- **card_reviews**: SM-2 spaced repetition state per user/card

## LangSmith Tracing

View detailed traces at [smith.langchain.com](https://smith.langchain.com) in the "studybuddy-v6" project.

## Differences from v5

| Aspect | v5 | v6 |
|--------|----|----|
| Persistence | None (in-memory) | PostgreSQL database |
| Agent count | 4 | 5 (+ Scheduler) |
| Flashcard caching | None | Content-addressed |
| Review scheduling | None | SM-2 algorithm |
| User memory | None | Custom SQLAlchemy store |
| Review buttons | Got It / Study More | No / Took a sec / Yes |

## Troubleshooting

**"Agent still initializing"**
- Wait for indexing to complete (watch terminal output)

**Database errors**
- Drop and recreate the database: `dropdb studybuddy && createdb studybuddy`

**POSTGRES_URL not set**
- Ensure your `.env` file contains `POSTGRES_URL=postgresql://localhost/studybuddy`

**No due cards showing**
- Generate some flashcards first, then they'll be scheduled for review

## Cost Considerations

StudyBuddy v6 uses multiple LLM calls per request:
- Supervisor: gpt-4o-mini (routing decision)
- Tutor: gpt-4o-mini (explanations)
- Card Generator: gpt-4o-mini (flashcard creation)
- Quality Checker: gpt-4o-mini (validation)
- Scheduler: gpt-4o-mini (recommendations)

Flashcard caching significantly reduces costs for repeated topic requests.

Monitor usage at platform.openai.com/usage.
