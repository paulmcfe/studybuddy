# StudyBuddy v5 - Multi-Agent Architecture

An intelligent AI Engineering tutor that uses a multi-agent architecture with the supervisor pattern to generate flashcards, answer questions, and manage spaced repetition review sessions.

## What's New in v5

StudyBuddy v5 transforms from a single agentic RAG system (v4) into a team of specialized agents:

- **Tutor Agent**: Explains concepts conversationally using gpt-5-nano
- **Card Generator Agent**: Creates high-quality flashcards using gpt-4o-mini
- **Quality Checker Agent**: Validates card clarity and usefulness
- **Scheduler Agent**: Manages spaced repetition using the SM-2 algorithm
- **Learning Coordinator (Supervisor)**: Orchestrates all agents based on student needs

## Features

- **Multi-Agent Coordination**: Supervisor routes requests to specialized agents
- **Quality-Checked Flashcards**: All cards pass through quality validation
- **Spaced Repetition**: SM-2 algorithm for optimal review scheduling
- **Interactive Chat**: Ask questions about concepts or flashcards
- **Chapter-Based Study**: Topics organized by chapter with single or cumulative modes
- **LangSmith Tracing**: Full observability of agent interactions

## Architecture

```
                    ┌─────────────┐
                    │  Supervisor │
                    └──────┬──────┘
                           │
         ┌─────────────────┼─────────────────┐
         │                 │                 │
         ▼                 ▼                 ▼
    ┌─────────┐      ┌───────────┐     ┌───────────┐
    │  Tutor  │      │   Card    │     │ Scheduler │
    └────┬────┘      │ Generator │     └───────────┘
         │           └─────┬─────┘
         │                 │
         │                 ▼
         │          ┌───────────┐
         └─────────►│  Quality  │
                    │  Checker  │
                    └───────────┘
```

## Prerequisites

- Python 3.12+
- OpenAI API key
- LangSmith API key (optional, for tracing)

## Setup

### 1. Navigate to the directory

```bash
cd studybuddy/v5-multi-agent
```

### 2. Create virtual environment

```bash
uv sync
```

Or with pip:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -e .
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
    "card_context": null  // Optional: current flashcard
}
```

Response includes reply, generated cards, and due review count.

### Flashcard Generation

```
POST /api/flashcard
{
    "chapter_id": 1,
    "scope": "single",  // or "cumulative"
    "current_topic": null  // Optional: for "Study More"
}
```

### Spaced Repetition

```
GET /api/cards/due
```

Returns cards due for review based on SM-2 scheduling.

```
POST /api/cards/{card_id}/review
{
    "got_it": true,
    "hesitation": false
}
```

Records review result and updates next review date.

## SM-2 Algorithm

The Scheduler uses the SM-2 spaced repetition algorithm:

- **New cards**: Review after 1 day
- **Successful review**: Interval = previous interval × ease factor
- **Failed review**: Reset to 1 day
- **Ease factor**: Adjusts based on performance (minimum 1.3)

Quality ratings from UI:
- "Got It" with confidence → 5 (perfect recall)
- "Got It" with hesitation → 4 (correct with hesitation)
- "Study More" → 2 (needs review soon)

## Project Structure

```
v5-multi-agent/
├── api/
│   ├── __init__.py
│   ├── index.py              # FastAPI app + multi-agent graph
│   ├── state.py              # Shared state definitions
│   ├── spaced_repetition.py  # SM-2 algorithm
│   └── agents/
│       ├── __init__.py
│       ├── tutor.py          # Tutor agent
│       ├── card_generator.py # Card Generator agent
│       ├── quality_checker.py# Quality Checker agent
│       ├── scheduler.py      # Scheduler agent
│       └── supervisor.py     # Supervisor/Coordinator
├── frontend/
│   ├── index.html            # Single-file app
│   └── images/               # Favicon and icons
├── documents/
│   ├── topic-list.md         # Chapter/topic structure
│   └── ref-*.md              # Knowledge base documents
├── .env                      # API keys
├── pyproject.toml
└── README.md
```

## LangSmith Tracing

View detailed traces at [smith.langchain.com](https://smith.langchain.com) in the "studybuddy-v5" project.

Traces show:
- Supervisor routing decisions
- Agent invocations and responses
- Quality check results
- Full multi-agent coordination flow

## Differences from v4

| Aspect | v4 | v5 |
|--------|----|----|
| Agent count | 1 (monolithic) | 5 (specialized) |
| Routing | Simple node graph | Supervisor pattern |
| Card quality | Generated directly | Generated → Quality checked |
| Spaced repetition | None | SM-2 algorithm |
| Models | gpt-4o-mini only | gpt-5-nano + gpt-4o-mini |

## Troubleshooting

**"Agent still initializing"**
- Wait for indexing to complete (watch terminal output)

**Low quality flashcards**
- Cards now pass through Quality Checker
- Check LangSmith traces for quality check results

**Spaced repetition not working**
- Card states are stored in memory (resets on restart)
- For persistence, modify `spaced_repetition.py` to use SQLite

## Cost Considerations

StudyBuddy v5 uses multiple LLM calls per request:
- Supervisor: gpt-5-nano (routing decision)
- Tutor: gpt-5-nano (explanations)
- Card Generator: gpt-4o-mini (flashcard creation)
- Quality Checker: gpt-5-nano (validation)
- Scheduler: gpt-5-nano (study summaries)

Monitor usage at platform.openai.com/usage.
