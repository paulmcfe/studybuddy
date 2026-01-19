# StudyBuddy v7 - Deep Agents with Planning and Delegation

An intelligent AI Engineering tutor with deep research capabilities, multi-step planning, and autonomous task delegation.

## What's New in v7

StudyBuddy v7 transforms from a reactive multi-agent system (v6) into a proactive deep agent:

- **Deep Research Agent**: Autonomous research with scope → research → synthesize workflow
- **Planning Capabilities**: Multi-step task decomposition and execution
- **Focus Areas**: User-configurable learning priorities that guide tutoring
- **Markdown Support**: Rich formatting in tutor responses
- **Enhanced Mobile UI**: Improved responsive design

## Features

- **Multi-Agent Coordination**: Supervisor routes to Tutor, Card Generator, Quality Checker, or Scheduler
- **Deep Research**: Autonomous information gathering and synthesis
- **Persistent Flashcards**: Generated cards are cached and reused
- **Smart Review Scheduling**: SM-2 spaced repetition algorithm
- **Focus Area Guidance**: Personalized tutoring based on learning goals
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
    │ (Deep) │ │  Gen  │ │Checker│ └─────────┘
    └───┬────┘ └───┬───┘ └───────┘      │
        │          │          ▲         │
        │   Planning & Research         │
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
- Node.js 18+
- OpenAI API key
- LangSmith API key (optional, for tracing)

## Setup

### 1. Navigate to the directory

```bash
cd studybuddy/v7-deep-agents
```

### 2. Set up the backend

```bash
# Create .env file
echo "OPENAI_API_KEY=your-key-here" > .env

# Create virtual environment and install dependencies
uv sync
```

### 3. Set up the frontend

```bash
cd frontend
npm install
cd ..
```

### 4. Run the app (two terminals)

**Terminal 1 - Backend:**
```bash
uv run uvicorn api.index:app --reload --port 8000
```

You'll see indexing progress in the terminal:
```
Server started. Document indexing running in background...
Indexing reference guides...
...
Agent ready!
```

**Terminal 2 - Frontend:**
```bash
cd frontend
npm run dev
```

### 5. Open the app

Visit `http://localhost:3000` in your browser.

## API Endpoints

### Chat

```
POST /api/chat
{
    "message": "Explain RAG to me",
    "card_context": null
}
```

Response includes reply (with markdown), cards, cache_hit status, due_cards, and study_stats.

### Focus Areas (NEW in v7)

```
GET /api/focus-areas
```

Returns current focus areas for personalized learning.

```
POST /api/focus-areas
{
    "areas": ["RAG fundamentals", "Vector databases", "Prompt engineering"]
}
```

Sets focus areas to guide tutoring responses.

### Flashcard Generation

```
POST /api/flashcard
{
    "chapter_id": 1,
    "scope": "single",
    "current_topic": null
}
```

### Get Due Cards

```
GET /api/due-cards?limit=10&include_new=true
```

### Record Review

```
POST /api/review
{
    "flashcard_id": "uuid-here",
    "quality": 4
}
```

Quality ratings (SM-2): 0-5 scale from complete blackout to perfect recall.

### Get Statistics

```
GET /api/stats
```

## Project Structure

```
v7-deep-agents/
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
│   │   ├── tutor.py          # Deep research tutor
│   │   ├── card_generator.py
│   │   ├── quality_checker.py
│   │   ├── supervisor.py
│   │   └── scheduler.py
│   └── services/
│       ├── __init__.py
│       ├── flashcard_cache.py
│       ├── spaced_repetition.py
│       └── memory_store.py
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
├── studybuddy.db
├── .env
├── pyproject.toml
└── README.md
```

## Differences from v6

| Aspect | v6 | v7 |
|--------|----|----|
| Tutor capabilities | Reactive responses | Deep research & planning |
| Focus areas | None | User-configurable |
| Response format | Plain text | Markdown support |
| Mobile UI | Basic | Enhanced responsive |

## LangSmith Tracing

View detailed traces at [smith.langchain.com](https://smith.langchain.com) in the "studybuddy-v7" project.

## Cost Considerations

StudyBuddy v7 uses multiple LLM calls per request:
- Supervisor: gpt-4o-mini (routing decision)
- Tutor: gpt-4o-mini (explanations with deep research)
- Card Generator: gpt-4o-mini (flashcard creation)
- Quality Checker: gpt-4o-mini (validation)
- Scheduler: gpt-4o-mini (recommendations)

Flashcard caching significantly reduces costs for repeated topic requests.

Monitor usage at platform.openai.com/usage.
