# StudyBuddy v4 - Agentic RAG with LangGraph

An intelligent AI Engineering tutor that uses LangGraph-based agentic RAG to answer questions about RAG, agents, embeddings, LangChain, LangGraph, and more.

## Features

- **Intelligent Query Analysis**: Determines query type (factual, conceptual, procedural, comparison) and complexity
- **Adaptive Retrieval**: Adjusts number of documents retrieved based on query complexity (simple=2, moderate=4, complex=6)
- **Self-Evaluation**: Assesses retrieval quality and retries if confidence is low (max 2 iterations)
- **Confidence Scoring**: Shows how confident the system is in its answer (visual bar + percentage)
- **Flashcard Generation**: Suggests flashcards for key concepts worth remembering
- **LangSmith Tracing**: Full observability of the agent's decision process

## Architecture

```
Query --> [Analyze] --> [Retrieve] --> [Evaluate] --> [Generate] --> Response
              |              ^              |
              |              |______________|  (retry if low confidence)
              |
              +--> [Direct Answer]  (if no retrieval needed)
```

### Nodes

- **analyze**: Classifies query type, complexity, determines if retrieval is needed
- **retrieve**: Performs adaptive multi-query retrieval with deduplication
- **evaluate**: Scores confidence (0-1), decides if retrieval is sufficient
- **generate**: Creates educational response with optional flashcard
- **direct**: Answers simple/conversational queries without retrieval

## Prerequisites

- Python 3.12+
- OpenAI API key
- LangSmith API key (for tracing)

## Setup

### 1. Clone and navigate

```bash
cd studybuddy/v4-agentic-rag
```

### 2. Create virtual environment

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

Or with uv:

```bash
uv sync
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

You'll see indexing progress in the terminal and status bar:
```
Server started. Document indexing running in background...
Indexing reference guides...
Indexed 42 chunks from Agentic Rag Pattern
Indexed 38 chunks from Agents And Agency
...
Total: 847 chunks indexed from 33 guides
Agent ready!
```

## Knowledge Base

The system indexes 33 AI Engineering reference guides covering:

- RAG fundamentals and patterns
- Embeddings and similarity search
- Vector databases (Qdrant)
- Agents and agency
- LangChain and LangGraph
- Tool use and memory systems
- Evaluation metrics
- Chunking strategies
- And more...

## Testing Different Query Types

**Questions that trigger retrieval:**
- "What is RAG and why is it useful?" (conceptual, moderate)
- "Compare different chunking strategies" (comparison, complex)
- "How does cosine similarity work?" (factual, simple)

**Questions answered directly:**
- "What's 2 + 2?" (no retrieval needed)
- "Hi, how are you?" (conversational)

## UI Features

- **Confidence Bar**: Visual indicator showing answer confidence (green/yellow/red)
- **Flashcard**: Click to flip between question and answer
- **Analysis Toggle**: Expand to see query type, complexity, retrieval stats

## LangSmith Tracing

View detailed traces at [smith.langchain.com](https://smith.langchain.com) in the "studybuddy-v4" project.

Traces show:
- Query analysis decisions
- Retrieval queries and results
- Evaluation confidence scores
- Generation inputs and outputs

## Project Structure

```
v4-agentic-rag/
├── api/
│   ├── index.py              # FastAPI app with LangGraph agent
│   └── requirements.txt      # Python dependencies
├── frontend/
│   ├── index.html            # Chat interface
│   ├── app.js                # JavaScript with confidence/flashcard UI
│   └── styles.css            # Styling including new components
├── documents/                # 33 ref-*.md knowledge base files
├── .env                      # API keys (not in git)
├── .gitignore
├── pyproject.toml
└── README.md
```

## Cost Considerations

StudyBuddy v4 uses:
- GPT-4o-mini for query analysis, evaluation, and generation
- text-embedding-3-small for embeddings

Multiple LLM calls per request (analyze + evaluate + generate), so costs are higher than v3. Monitor usage at platform.openai.com/usage.

## What's Next

In Chapter 5, we'll expand StudyBuddy into a multi-agent system with separate agents for tutoring, flashcard generation, quality checking, and scheduling—all coordinated by a supervisor.

## Troubleshooting

**"Agent still initializing"**
- Wait for indexing to complete (watch terminal output)

**LangSmith traces not appearing**
- Verify LANGSMITH_API_KEY is set correctly
- Check project name is "studybuddy-v4"

**Low confidence on all queries**
- Ensure documents directory contains ref-*.md files
- Check indexing completed successfully

**JSON parsing errors in logs**
- Normal for some edge cases; fallback defaults are used
