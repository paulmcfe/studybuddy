# StudyBuddy v5 - Multi-Agent Architecture

An intelligent AI Engineering tutor that uses a multi-agent architecture with the supervisor pattern to generate quality-checked flashcards and answer questions.

## What's New in v5

StudyBuddy v5 transforms from a single agentic RAG system (v4) into a team of specialized agents:

- **Tutor Agent**: Explains concepts conversationally using gpt-4o
- **Card Generator Agent**: Creates high-quality flashcards using gpt-4o-mini
- **Quality Checker Agent**: Validates card clarity and usefulness
- **Learning Coordinator (Supervisor)**: Orchestrates all agents based on student needs
- **Next.js Frontend**: Modern React-based UI with chapter selection and chat panel

## Features

- **Multi-Agent Coordination**: Supervisor routes requests to specialized agents
- **Quality-Checked Flashcards**: All cards pass through quality validation
- **Interactive Chat**: Ask questions about concepts or flashcards
- **Chapter-Based Study**: Topics organized by chapter with single or cumulative modes
- **Two-Button Review**: "Got It" for new topics, "Study More" for same topic
- **LangSmith Tracing**: Full observability of agent interactions

## Architecture

```
                    ┌─────────────┐
                    │  Supervisor │
                    └──────┬──────┘
                           │
              ┌────────────┼────────────┐
              │            │            │
              ▼            ▼            ▼
         ┌─────────┐  ┌───────────┐  ┌─────────┐
         │  Tutor  │  │   Card    │  │ Quality │
         └────┬────┘  │ Generator │  │ Checker │
              │       └─────┬─────┘  └─────────┘
              │             │              ▲
              │             └──────────────┘
              │                   │
              └───────────────────┘
```

## Prerequisites

- Python 3.12+
- Node.js 18+
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
uv pip install -e .
```

### 3. Configure environment variables

Create or update `.env`:

```
OPENAI_API_KEY=sk-your-key-here
LANGSMITH_API_KEY=lsv2_your-key-here
```

### 4. Install frontend dependencies

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

Visit `http://localhost:3000` in your browser. The Next.js dev server proxies `/api/*` requests to the FastAPI backend on port 8000.

## API Endpoints

### Chat

```
POST /api/chat
{
    "message": "Explain RAG to me",
    "card_context": null  // Optional: current flashcard
}
```

Response includes reply and generated cards.

### Flashcard Generation

```
POST /api/flashcard
{
    "chapter_id": 1,
    "scope": "single",  // or "cumulative"
    "current_topic": null  // Optional: for "Study More"
}
```

## Project Structure

```
v5-multi-agent/
├── api/
│   ├── __init__.py
│   ├── index.py              # FastAPI app + multi-agent graph
│   ├── state.py              # Shared state definitions
│   └── agents/
│       ├── __init__.py
│       ├── tutor.py          # Tutor agent
│       ├── card_generator.py # Card Generator agent
│       ├── quality_checker.py# Quality Checker agent
│       └── supervisor.py     # Supervisor/Coordinator
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
│   │       └── ...
│   ├── next.config.ts
│   └── package.json
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
| Agent count | 1 (monolithic) | 4 (specialized) |
| Routing | Simple node graph | Supervisor pattern |
| Card quality | Generated directly | Generated → Quality checked |
| Models | gpt-4o-mini only | gpt-4o + gpt-4o-mini |
| Review buttons | Got It / Study More | Got It / Study More |

## Troubleshooting

**"Agent still initializing"**
- Wait for indexing to complete (watch terminal output)

**Low quality flashcards**
- Cards now pass through Quality Checker
- Check LangSmith traces for quality check results

## Cost Considerations

StudyBuddy v5 uses multiple LLM calls per request:
- Supervisor: gpt-4o (routing decision)
- Tutor: gpt-4o (explanations)
- Card Generator: gpt-4o-mini (flashcard creation)
- Quality Checker: gpt-4o (validation)

Monitor usage at platform.openai.com/usage.
