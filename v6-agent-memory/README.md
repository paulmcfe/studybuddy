# StudyBuddy v6 - Agent Memory and Spaced Repetition

An intelligent AI Engineering tutor with persistent memory, flashcard caching, and spaced repetition powered by the SM-2 algorithm.

## What's New in v6

StudyBuddy v6 transforms from an ephemeral multi-agent system (v5) into a truly persistent learning companion:

- **SQLite Persistence**: All data survives restarts in `studybuddy.db`
- **Content-Addressed Flashcard Caching**: Same topic + context = instant results (no regeneration)
- **SM-2 Spaced Repetition**: Optimal review scheduling based on your performance
- **Scheduler Agent**: New agent that manages due cards and learning statistics
- **Custom Memory Store**: Remember user preferences, struggles, and goals across sessions

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
                    │   SQLite    │
                    │  Database   │
                    └─────────────┘
```

## Prerequisites

- Python 3.12+
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

### 3. Configure environment variables

Create or update `.env`:

```
OPENAI_API_KEY=sk-your-key-here
LANGSMITH_API_KEY=lsv2_your-key-here
```

### 4. Run the server

```bash
uv run uvicorn api.index:app --reload --port 8000
```

Or without uv:

```bash
uvicorn api.index:app --reload --port 8000
```

### 5. Open the app

Visit `http://localhost:8000` in your browser.

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
│   │   └── connection.py     # Database setup
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── tutor.py
│   │   ├── card_generator.py
│   │   ├── quality_checker.py
│   │   ├── supervisor.py
│   │   └── scheduler.py      # NEW: Spaced repetition
│   └── services/
│       ├── __init__.py
│       ├── flashcard_cache.py
│       ├── spaced_repetition.py
│       └── memory_store.py
├── documents/
├── frontend/
├── studybuddy.db             # SQLite database (created at runtime)
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
| Persistence | None (in-memory) | SQLite database |
| Agent count | 4 | 5 (+ Scheduler) |
| Flashcard caching | None | Content-addressed |
| Review scheduling | None | SM-2 algorithm |
| User memory | None | Custom SQLAlchemy store |

## Troubleshooting

**"Agent still initializing"**
- Wait for indexing to complete (watch terminal output)

**Database errors**
- Delete `studybuddy.db` and restart to recreate tables

**No due cards showing**
- Generate some flashcards first, then they'll be scheduled for review

## Cost Considerations

StudyBuddy v6 uses multiple LLM calls per request:
- Supervisor: gpt-5-nano (routing decision)
- Tutor: gpt-5-nano (explanations)
- Card Generator: gpt-4o-mini (flashcard creation)
- Quality Checker: gpt-5-nano (validation)
- Scheduler: gpt-5-nano (recommendations)

Flashcard caching significantly reduces costs for repeated topic requests.

Monitor usage at platform.openai.com/usage.
