# StudyBuddy v10 - Full Stack Application

A full-stack learning application that lets you study any subject with AI-powered tutoring, flashcards, and spaced repetition.

## Features

- **Learning Programs**: Create separate programs for different subjects, each with its own knowledge base
- **Document Upload**: Upload PDF, Markdown, and text files to build your knowledge base
- **AI Curriculum Generation**: Generate comprehensive curricula for any topic
- **Flashcard Generation**: AI-generated flashcards from your documents
- **Spaced Repetition**: SM-2 algorithm for optimal review scheduling
- **Chat Interface**: Real-time streaming chat with a tutor that uses your documents
- **Progress Tracking**: Dashboard with statistics and upcoming reviews

## Prerequisites

- Python 3.12+
- Node.js 18+
- PostgreSQL
- Qdrant (vector database)
- OpenAI API key

## Setup

### 1. Install Dependencies

```bash
cd v10-full-stack

# Install Python dependencies (using uv)
uv sync

# Install frontend dependencies
cd frontend
npm install
cd ..
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env with your API keys and database URL
```

### 3. Start Qdrant

```bash
# Using Docker
docker run -p 6333:6333 qdrant/qdrant

# Or download binary from https://qdrant.tech/documentation/quick-start/
```

### 4. Create PostgreSQL Database (if not already created)

```bash
# Check if database exists
psql -l | grep studybuddy

# If no output, create it
createdb studybuddy
```

Note: v10 uses the same `studybuddy` database as previous versions. New tables are created automatically on startup.

### 5. Start the Application

```bash
# Terminal 1: Start the API server
uv run uvicorn api.index:app --reload --port 8000

# Terminal 2: Start the frontend dev server
cd frontend
npm run dev
```

Open http://localhost:3000 in your browser.

## API Endpoints

### Programs
- `GET /api/programs` - List all programs
- `POST /api/programs` - Create a program
- `GET /api/programs/{id}` - Get program details
- `PATCH /api/programs/{id}` - Update a program
- `DELETE /api/programs/{id}` - Archive a program

### Documents
- `GET /api/programs/{id}/documents` - List documents
- `POST /api/programs/{id}/documents` - Upload a document
- `DELETE /api/programs/{id}/documents/{doc_id}` - Delete a document

### Flashcards
- `GET /api/programs/{id}/flashcards` - List flashcards
- `GET /api/programs/{id}/due-cards` - Get cards due for review
- `POST /api/programs/{id}/flashcards/generate` - Generate a flashcard
- `POST /api/programs/{id}/flashcards/{card_id}/review` - Record a review

### Curriculum
- `POST /api/programs/{id}/generate-curriculum` - Generate AI curriculum

### WebSocket
- `WS /ws/study/{program_id}` - Real-time chat with tutor

## Project Structure

```
v10-full-stack/
├── api/
│   ├── index.py              # FastAPI app
│   ├── database/
│   │   ├── models.py         # SQLAlchemy models
│   │   └── connection.py     # Database connection
│   └── services/
│       ├── retrieval.py      # Vector search
│       ├── indexing.py       # Document processing
│       ├── curriculum.py     # Curriculum generation
│       └── flashcard.py      # Flashcard service
├── frontend/                 # Next.js frontend
│   ├── src/
│   │   ├── app/
│   │   │   ├── layout.tsx
│   │   │   ├── page.tsx
│   │   │   └── globals.css
│   │   └── components/
│   │       ├── Sidebar.tsx
│   │       ├── Dashboard.tsx
│   │       ├── StudyInterface.tsx
│   │       ├── DocumentManager.tsx
│   │       ├── CreateProgram.tsx
│   │       └── Flashcard.tsx
│   ├── next.config.ts
│   └── package.json
├── uploads/                  # Uploaded documents
├── pyproject.toml
└── README.md
```

## Development

### Backend
The API is built with FastAPI and uses:
- SQLAlchemy for PostgreSQL
- Qdrant for vector search
- LangChain for LLM orchestration
- WebSockets for real-time chat

### Frontend
The frontend is built with:
- Next.js 15 with App Router
- React 19
- Vanilla CSS with CSS variables

## What's Next

- Chapter 11: MCP connectors for external content import
- Chapter 12: Production deployment with authentication
