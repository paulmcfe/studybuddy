# StudyBuddy v4 - Agentic RAG with LangGraph

An intelligent AI Engineering tutor that uses LangGraph-based agentic RAG to generate flashcards and answer questions about RAG, agents, embeddings, LangChain, LangGraph, and more.

## Features

- **Interactive Flashcards**: Study with AI-generated flashcards scoped to specific chapters or cumulatively
- **Intelligent Chat**: Ask questions about the current flashcard or any topic
- **Adaptive Retrieval**: Adjusts retrieval based on query complexity
- **Chapter-Based Organization**: Topics organized by chapter with single or cumulative study modes
- **Responsive Design**: Works on desktop (sidebar) and mobile (full-screen)
- **Accessibility**: WCAG 2.1 AA compliant with keyboard navigation, screen reader support, and reduced motion preferences
- **LangSmith Tracing**: Full observability of the agent's decision process

## Architecture

```
Flashcard Generation:
Topic List --> Select Topic --> Retrieve Context --> Generate Card --> Display

Chat Flow:
Query --> [Analyze] --> [Retrieve] --> [Evaluate] --> [Generate] --> Response
              |              ^              |
              |              |______________|  (retry if low confidence)
              |
              +--> [Direct Answer]  (if no retrieval needed)
```

### Agent Nodes

- **analyze**: Classifies query type, complexity, determines if retrieval is needed
- **retrieve**: Performs adaptive multi-query retrieval with deduplication
- **evaluate**: Scores confidence (0-1), decides if retrieval is sufficient
- **generate**: Creates educational response with optional flashcard suggestion
- **direct**: Answers simple/conversational queries without retrieval

## Prerequisites

- Python 3.12+
- OpenAI API key
- LangSmith API key (optional, for tracing)

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

You'll see indexing progress in the terminal:
```
Server started. Document indexing running in background...
Indexing reference guides...
Indexed 42 chunks from Agentic Rag Pattern
Indexed 38 chunks from Agents And Agency
...
Total: 847 chunks indexed from 33 guides
Parsed 14 chapters from topic-list.md
Agent ready!
```

## Customizing the Topic List

StudyBuddy generates flashcards based on topics defined in `documents/topic-list.md`. You can customize this file to create your own curriculum.

### Format

The topic list uses a simple markdown format:

```markdown
# Chapter 1: Your Chapter Title

## Topic Name
- Subtopic 1
- Subtopic 2
- Subtopic 3

## Another Topic
- More subtopics here

---

# Chapter 2: Next Chapter

## Topic in Chapter 2
- Subtopics...
```

### Structure Rules

1. **Chapters**: Use `# Chapter N: Title` format where N is a number
2. **Topics/Sections**: Use `## Topic Name` under each chapter
3. **Subtopics**: Use `- Subtopic` bullet points under each topic
4. **Separators**: Use `---` between chapters (optional, for readability)

### How It Works

- When studying a **single chapter**, flashcards are generated from topics in that chapter only
- When studying **cumulatively** (e.g., "Chapters 1-3"), topics from all chapters up to and including the selected one are included
- The **subtopics** help guide the LLM to generate more specific, targeted flashcards
- Topics are selected randomly, so repeated studying covers different areas

### Example: Custom Topic List

```markdown
# Chapter 1: Python Basics

## Variables and Data Types
- Integers and floats
- Strings and string methods
- Lists and tuples
- Dictionaries

## Control Flow
- If/elif/else statements
- For loops
- While loops
- List comprehensions

---

# Chapter 2: Functions

## Function Basics
- Defining functions
- Parameters and arguments
- Return values

## Advanced Functions
- Lambda functions
- Decorators
- Generators
```

### Tips

- Keep topic names concise but descriptive
- Include 2-5 subtopics per topic for best results
- Subtopics should be specific concepts, not full sentences
- The reference documents (`ref-*.md`) provide context for flashcard generation, so ensure your topics align with your knowledge base

## Knowledge Base

The system indexes markdown files from the `documents/` directory:

- Files matching `ref-*.md` are indexed into the vector store
- The `topic-list.md` file defines the chapter/topic structure
- Add your own reference documents to expand the knowledge base

## UI Features

- **Chapter Selection**: Choose which chapter to study
- **Scope Toggle**: Study single chapter or cumulative (hidden for Chapter 1)
- **Flashcards**: Tap/click to flip between question and answer
- **Got It**: Mark as understood and get a new random topic
- **Study More**: Get another flashcard on the same topic
- **Chat**: Ask questions about the current card or any topic

## Keyboard Shortcuts

- **Space**: Flip the flashcard
- **Escape**: Close the chat panel
- **Enter** (in chat): Send message

## Accessibility

StudyBuddy v4 is designed to be accessible:

- Full keyboard navigation
- Screen reader announcements for dynamic content
- ARIA labels and roles throughout
- Respects `prefers-reduced-motion` for animations
- High contrast text (WCAG AAA compliant)
- Focus indicators on all interactive elements

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
│   ├── index.html            # Single-file app (HTML, CSS, JS)
│   └── images/               # Favicon and icons
├── documents/
│   ├── topic-list.md         # Chapter/topic structure
│   └── ref-*.md              # Knowledge base documents
├── .env                      # API keys (not in git)
├── .gitignore
├── pyproject.toml
└── README.md
```

## Deployment

StudyBuddy v4 is configured for Vercel deployment:

- `vercel.json` defines the build and routing configuration
- API routes are handled by the Python serverless function
- Frontend is served as static files
- Deployment is triggered only when `v4-agentic-rag/` or `vercel.json` changes

## Cost Considerations

StudyBuddy v4 uses:
- GPT-4o-mini for query analysis, evaluation, and generation
- text-embedding-3-small for embeddings

Multiple LLM calls per flashcard and chat request. Monitor usage at platform.openai.com/usage.

## Troubleshooting

**"Agent still initializing"**
- Wait for indexing to complete (watch terminal output)

**No chapters appearing**
- Ensure `documents/topic-list.md` exists and follows the correct format
- Check terminal for parsing errors

**LangSmith traces not appearing**
- Verify LANGSMITH_API_KEY is set correctly
- Check project name is "studybuddy-v4"

**Low quality flashcards**
- Ensure `documents/` contains relevant `ref-*.md` files
- Check that topics in `topic-list.md` align with your knowledge base

**JSON parsing errors in logs**
- Normal for some edge cases; fallback defaults are used
