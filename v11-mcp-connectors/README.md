# StudyBuddy v11 - MCP Connectors

A full-stack learning application with AI-powered tutoring, flashcards, and spaced repetition. Version 11 adds MCP connectors for importing content from external sources and augmenting the tutor with web search.

## What's New in v11

- **Import from URL**: Paste a web page URL to import it as a learning document (uses the Fetch MCP server)
- **Import from GitHub**: Connect a repository, browse its files, and selectively import markdown documentation
- **Web Search for the Tutor**: Enable Brave Search so the tutor can search the web when your documents don't have the answer
- **Incremental Sync**: GitHub connector tracks content hashes — re-syncing only re-imports files that changed
- **Connector Management UI**: Dedicated Connectors page for configuring and managing all three connector types

## Features

All v10 features plus MCP connectors:

- **Learning Programs**: Create separate programs for different subjects, each with its own knowledge base
- **Document Upload**: Upload PDF, Markdown, and text files to build your knowledge base
- **MCP Connectors**: Import content from URLs, GitHub repos, and augment the tutor with web search
- **AI Curriculum Generation**: Generate comprehensive curricula for any topic
- **Flashcard Generation**: AI-generated flashcards from your documents
- **Spaced Repetition**: SM-2 algorithm for optimal review scheduling
- **Chat Interface**: Real-time streaming chat with a tutor that uses your documents
- **Search-Augmented Tutoring**: LangGraph agent with Brave Search for web-augmented answers
- **Progress Tracking**: Dashboard with statistics and upcoming reviews

## Prerequisites

- Python 3.12+
- Node.js 18+
- PostgreSQL
- Qdrant (vector database)
- OpenAI API key

**Optional (for connectors):**
- GitHub Personal Access Token (for GitHub connector)
- Brave Search API key (for web search — free tier at https://brave.com/search/api/)

## Setup

### 1. Install Dependencies

```bash
cd v11-mcp-connectors

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

Required variables:
- `OPENAI_API_KEY` — OpenAI API key
- `POSTGRES_URL` — PostgreSQL connection string

Optional variables:
- `QDRANT_URL` — Qdrant URL (defaults to `http://localhost:6333`)
- `COHERE_API_KEY` — Cohere API key for reranking
- `GITHUB_TOKEN` — GitHub Personal Access Token for the GitHub connector
- `BRAVE_API_KEY` — Brave Search API key for the web search connector

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

Note: v11 uses the same `studybuddy` database as previous versions. New tables (`connector_configs`) and columns are created automatically on startup.

### 5. Start the Application

```bash
# Terminal 1: Start the API server
uv run uvicorn api.index:app --reload --port 8000

# Terminal 2: Start the frontend dev server
cd frontend
npm run dev
```

Open http://localhost:3000 in your browser.

## Usage

### Creating a Learning Program

1. Click **"+ New Program"** in the sidebar
2. Enter a name for your subject (e.g., "Machine Learning", "Spanish")
3. Enter a description of what you want to learn (e.g., "Machine learning fundamentals including supervised and unsupervised learning")
4. Choose a curriculum option:
   - **Start Empty** - Create a blank program without a predefined topic list. You can upload your own documents and the AI tutor will use them directly.
   - **Generate with AI** - The AI will generate a structured curriculum based on your description, with chapters and topics. This creates a topic list that organizes your learning and helps generate focused flashcards.
5. Click **Create** to start your new program

When you create a program, the app automatically generates an initial set of flashcards in the background based on your curriculum or description.

**When to use each option:**

- Use **Generate with AI** when learning a new subject from scratch and you want a structured learning path with organized topics
- Use **Start Empty** when you have your own study materials (textbooks, lecture notes, articles) and want the AI to work directly from those documents

### Building Your Knowledge Base

Upload documents to give the AI tutor context about your subject:

1. Select a program from the sidebar
2. Click **"Upload Documents"** or go to the Documents view
3. Drag and drop files or click to browse
4. Supported formats: PDF, Markdown (.md), and plain text (.txt)
5. Maximum file size: 50MB per file
6. Documents are automatically indexed for AI retrieval

### Using Connectors

Go to the **Connectors** tab to import content from external sources:

**Import from URL:**
1. Click the "Import from URL" card
2. Paste a web page URL
3. Click **Import** — the page is fetched, converted to markdown, and indexed

**Import from GitHub:**
1. Click the "Import from GitHub" card
2. Enter the repository (e.g., `owner/repo`), a Personal Access Token, and branch
3. Click **Browse Files** to see available markdown files
4. Select files and click **Import Selected**
5. Use **Sync** later to detect and re-import changed files

**Web Search (Brave Search):**
1. Click the "Web Search" card
2. Enter your Brave Search API key and click **Save**
3. The tutor can now search the web when your documents don't cover a topic
4. Web search results include source URLs for verification

### Studying with the AI Tutor

The Chat interface lets you have conversations with an AI tutor that references your uploaded documents:

1. Select a program and click **"Start Studying"**
2. Type questions in the chat input
3. The tutor searches your documents and provides relevant answers
4. If Brave Search is enabled, the tutor can also search the web when needed
5. Use this to ask clarifying questions, get explanations, or test your understanding

The chat supports rich formatting including markdown (headers, lists, code blocks) and LaTeX math equations.

### Using Flashcards

Flashcards use spaced repetition (SM-2 algorithm) to optimize your review schedule:

1. Go to the **Flashcards** tab in the Study view
2. Review your due cards or click **"Generate New Card"** to create more
3. Click on a card to reveal the answer
4. Rate your recall:
   - **No** - You didn't remember (card shown again soon)
   - **Took a sec** - You remembered with effort
   - **Yes** - Easy recall (card scheduled for later)

Flashcards follow the "minimum information principle" - each card tests one atomic fact with a specific, direct question for effective learning.

### Tracking Progress

The Dashboard shows your learning statistics:

- **Documents Indexed** - Number of processed documents
- **Flashcards** - Total cards in your collection
- **Due for Review** - Cards scheduled for today
- **Mastered** - Cards you've learned well

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

### Connectors
- `GET /api/programs/{id}/connectors` - List connectors
- `POST /api/programs/{id}/connectors` - Create a connector
- `GET /api/programs/{id}/connectors/{cid}` - Get connector details
- `PATCH /api/programs/{id}/connectors/{cid}` - Update connector config
- `DELETE /api/programs/{id}/connectors/{cid}` - Delete a connector

### Connector Actions
- `POST /api/programs/{id}/connectors/{cid}/fetch` - Import a URL
- `GET /api/programs/{id}/connectors/{cid}/github/files` - Browse repo files
- `POST /api/programs/{id}/connectors/{cid}/github/import` - Import GitHub files
- `POST /api/programs/{id}/connectors/{cid}/github/sync` - Re-sync GitHub repo

### Chat
- `POST /api/programs/{id}/chat` - Chat with the tutor (uses LangGraph agent if Brave Search is configured)

## Project Structure

```
v11-mcp-connectors/
├── api/
│   ├── index.py              # FastAPI app with connector endpoints
│   ├── agents/
│   │   └── tutor.py          # LangGraph tutor agent with Brave Search
│   ├── database/
│   │   ├── models.py         # SQLAlchemy models (+ ConnectorConfig)
│   │   └── connection.py     # Database connection
│   └── services/
│       ├── connectors/       # MCP connector implementations
│       │   ├── mcp_client.py       # MCP server configurations
│       │   ├── fetch_connector.py  # URL import via Fetch MCP
│       │   ├── github_connector.py # GitHub import via REST API
│       │   └── brave_connector.py  # Brave Search tool provider
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
│   │       ├── ConnectorManager.tsx  # Connector management UI
│   │       ├── CreateProgram.tsx
│   │       └── Flashcard.tsx
│   ├── next.config.ts
│   └── package.json
├── uploads/                  # Uploaded and imported documents
├── pyproject.toml
└── README.md
```

## Development

### Backend
The API is built with FastAPI and uses:
- SQLAlchemy for PostgreSQL
- Qdrant for vector search
- LangChain + LangGraph for LLM orchestration and agents
- langchain-mcp-adapters for MCP server integration
- httpx for GitHub REST API calls

### Frontend
The frontend is built with:
- Next.js 15 with App Router
- React 19
- Vanilla CSS with CSS variables

## What's Next

- Chapter 12: Production deployment with authentication and multi-user support
