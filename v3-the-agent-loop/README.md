# StudyBuddy v3 - The Agent Loop

Your AI tutor is now an agent. This version uses LangChain 1.0 and Qdrant to build a system that reasons about when to search and when to answer directly.

## What This Is

StudyBuddy v3 transforms the RAG chatbot into a proper agent. Instead of blindly retrieving context for every question, it thinks about whether it needs to search the study materials. Ask "What's 2+2?" and it answers directly. Ask about Sherlock Holmes and it searches the indexed stories first.

This version demonstrates:
- LangChain 1.0's `create_agent` API
- Qdrant vector database with in-memory mode
- Tool-based architecture with the `@tool` decorator
- ReAct-style reasoning (reason → act → observe)
- Background document indexing with progress tracking
- Production-ready document processing pipeline

## What's an Agent?

An agent is an LLM that can use tools. Instead of just generating text, it:

1. **Reasons** about what to do next
2. **Acts** by calling tools (like searching documents)
3. **Observes** the results
4. **Repeats** until it has enough information to answer

This is called the "agent loop." The system prompt encourages step-by-step thinking, and the model decides when tools are needed.

## Prerequisites

- Python 3.12+
- An OpenAI API key (grab one at platform.openai.com)
- Git and GitHub account
- Vercel account (free tier works great)

## Local Setup

### 1. Clone the repo

```bash
git clone https://github.com/yourusername/studybuddy.git
cd studybuddy/v3-the-agent-loop
```

### 2. Set up the project

```bash
# Create .env file
echo "OPENAI_API_KEY=your-key-here" > .env

# Create virtual environment and install dependencies
uv sync
```

### 3. Add study materials

Place `.txt` files in the `documents/` directory. The default setup includes 12 Sherlock Holmes stories from "The Adventures of Sherlock Holmes."

### 4. Run the app

```bash
uv run uvicorn api.index:app --reload --port 8000
```

You'll see documents being indexed in the background:
```
Server started. Document indexing running in background...
Indexing stories...
Indexed 47 chunks from A Scandal In Bohemia
Indexed 52 chunks from The Red Headed League
...
```

Visit `http://localhost:8000` in your browser. The UI shows indexing progress.

### 5. Test it out

Try questions that require searching:
- "Who is Irene Adler?"
- "What was the mystery in The Red-Headed League?"
- "How does Holmes solve the case of the speckled band?"

Then try questions the agent can answer directly:
- "What's the capital of France?"
- "Explain what a vector database is"

Watch how the agent decides whether to search or answer directly.

## Project Structure

```
v3-the-agent-loop/
├── api/
│   ├── index.py         # FastAPI app with LangChain agent
│   └── requirements.txt # Vercel dependencies
├── documents/           # Your study materials (.txt files)
│   ├── 01-a-scandal-in-bohemia.txt
│   ├── 02-the-red-headed-league.txt
│   └── ...
├── frontend/
│   ├── index.html       # Chat interface
│   ├── styles.css       # Styling
│   └── app.js           # Frontend logic with status polling
├── .env                 # Your API keys (never commit!)
├── .gitignore           # Keeps secrets out of git
├── pyproject.toml       # Python dependencies
└── README.md            # You are here
```

## Key Components

### The Agent

```python
from langchain.agents import create_agent

agent = create_agent(
    model="gpt-5-nano",
    tools=tools,
    system_prompt=SYSTEM_PROMPT
)
```

### The Search Tool

```python
@tool
def search_materials(query: str) -> str:
    """Search the indexed study materials for information."""
    results = vector_store.similarity_search(query, k=3)
    # Format and return results...
```

### Document Processing

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
```

### Vector Storage

```python
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

client = QdrantClient(":memory:")  # In-memory for development
vector_store = QdrantVectorStore(
    client=client,
    collection_name="study_materials",
    embedding=embeddings
)
```

## Deploying to Vercel

**Note:** Vercel's serverless environment resets between requests, so documents need to be re-indexed on each cold start. For production, consider using Qdrant Cloud for persistent storage.

```bash
# From the repo root
vercel

# Set your environment variable
vercel env add OPENAI_API_KEY

# Deploy to production
vercel --prod
```

## Customizing Your StudyBuddy

### Add your own study materials

Drop `.txt` files into the `documents/` directory. They'll be automatically indexed on startup. File names become document names (dashes and underscores converted to spaces, title-cased).

### Adjust chunking parameters

In `api/index.py`, modify the text splitter:

```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,    # Characters per chunk
    chunk_overlap=50   # Overlap between chunks
)
```

### Change the system prompt

Edit `SYSTEM_PROMPT` in `api/index.py` to change how the agent behaves:

```python
SYSTEM_PROMPT = """You are StudyBuddy, an AI tutoring assistant...
Think step-by-step:
1. Consider whether you need to search
2. If you do, use the search_materials tool
3. Use retrieved information to provide clear explanations
..."""
```

### Add more tools

Create new tools with the `@tool` decorator:

```python
@tool
def get_definition(term: str) -> str:
    """Look up a definition for a term."""
    # Your implementation
    pass

tools = [search_materials, get_definition]
```

## What's Next?

In Chapter 4, we'll rebuild using LangGraph for complete control over the agent's state machine. We'll add reflection, confidence scoring, and observability with LangSmith.

## Troubleshooting

**Backend won't start:**
- Check your OpenAI API key is set in `.env`
- Make sure you're in the virtual environment
- Verify LangChain 1.0+ is installed: `pip list | grep langchain`

**Documents not indexing:**
- Check the `documents/` directory exists and contains `.txt` files
- Look at terminal output for indexing progress
- Check `/api/status` endpoint for indexing state

**Agent not using tools:**
- The agent decides when tools are needed based on the question
- For questions about indexed content, it should search automatically
- Check the system prompt encourages tool usage

**Import errors:**
- LangChain 1.0 has new import paths:
  - `from langchain_qdrant import QdrantVectorStore`
  - `from langchain_text_splitters import RecursiveCharacterTextSplitter`
  - `from langchain.agents import create_agent`

**Deployment issues:**
- Make sure `.env` is in `.gitignore`
- Verify environment variables are set in Vercel dashboard
- Check logs: `vercel logs` for error details

## Cost Considerations

StudyBuddy v3 uses:
- GPT-5-nano for agent reasoning (~$0.15/M input, ~$0.60/M output tokens)
- text-embedding-3-small for embeddings (~$0.02/M tokens)

Agent conversations may use more tokens due to reasoning traces. A typical conversation costs less than $0.02. Monitor usage at platform.openai.com/usage.

## Contributing

This is your learning project! Fork it, modify it, make it your own. If you build something cool, share it with the community.

## License

MIT - do whatever you want with this.

## Questions?

Hit up the AI Makerspace community or open an issue. We're here to help.

---

Built as part of the AI Engineering Bootcamp. Keep building.
