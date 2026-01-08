from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from contextlib import asynccontextmanager
from pathlib import Path
import threading
import os
from dotenv import load_dotenv

# Load .env from parent directory (v3-the-agent-loop/)
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)

# Check if running on Vercel
IS_VERCEL = os.environ.get("VERCEL") == "1"

# ============== LangChain Agent Setup ==============

from langchain_qdrant import QdrantVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain.tools import tool
from langchain.agents import create_agent
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

# Initialize embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Create in-memory Qdrant client
qdrant_client = QdrantClient(":memory:")

# Collection name
COLLECTION_NAME = "study_materials"

# Create collection with proper dimensions (1536 for text-embedding-3-small)
qdrant_client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
)

# Create LangChain vector store wrapper
vector_store = QdrantVectorStore(
    client=qdrant_client,
    collection_name=COLLECTION_NAME,
    embedding=embeddings
)


def index_document(file_path: str, doc_name: str):
    """Load, chunk, and index a document."""
    loader = TextLoader(file_path)
    documents = loader.load()

    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len
    )
    chunks = text_splitter.split_documents(documents)

    # Add metadata
    for chunk in chunks:
        chunk.metadata['source'] = doc_name

    # Index in Qdrant
    vector_store.add_documents(chunks)

    return len(chunks)


@tool
def search_materials(query: str) -> str:
    """
    Search the indexed study materials for information about a topic.
    Use this when you need to find specific information from the
    student's uploaded study materials.

    Args:
        query: The search term or question to look up
    """
    results = vector_store.similarity_search(query, k=3)

    if not results:
        return "No relevant information found in study materials."

    # Format results with source attribution
    formatted = []
    for doc in results:
        source = doc.metadata.get('source', 'Unknown')
        formatted.append(f"[From {source}]:\n{doc.page_content}")

    return "\n\n".join(formatted)


# System prompt
SYSTEM_PROMPT = """You are StudyBuddy, an AI tutoring assistant helping students study Sherlock Holmes stories.

IMPORTANT: The student has uploaded study materials (Sherlock Holmes stories) that you MUST search
before answering questions about characters, plots, or events. Always use the search_materials tool
first for any question about:
- Characters (Holmes, Watson, Irene Adler, Moriarty, etc.)
- Story plots or mysteries
- Specific events or quotes
- Anything related to the study materials

Only answer from general knowledge for questions completely unrelated to the study materials
(like math questions or general facts).

When you search, cite which story the information comes from."""

# Create tools list
tools = [search_materials]

# Create the agent using LangChain 1.0 API
agent = create_agent(
    model="gpt-5-nano",
    tools=tools,
    system_prompt=SYSTEM_PROMPT
)


# Track indexing status
indexing_status = {"done": False, "count": 0, "chunks": 0, "current_file": ""}


def index_all_documents():
    """Index all documents from the documents directory."""
    if indexing_status["done"]:
        return indexing_status["chunks"]

    documents_dir = Path(__file__).parent.parent / "documents"
    if not documents_dir.exists():
        print(f"Documents directory not found: {documents_dir}")
        indexing_status["done"] = True
        return 0

    story_files = sorted(documents_dir.glob("*.txt"))
    if not story_files:
        print("No .txt files found in documents directory")
        indexing_status["done"] = True
        return 0

    print("Indexing stories...")
    total_chunks = 0
    for filepath in story_files:
        doc_name = filepath.stem.replace("-", " ").replace("_", " ").title()
        indexing_status["current_file"] = doc_name
        num_chunks = index_document(str(filepath), doc_name)
        total_chunks += num_chunks
        indexing_status["chunks"] = total_chunks
        print(f"Indexed {num_chunks} chunks from {doc_name}")

    print(f"\nTotal: {total_chunks} chunks indexed from {len(story_files)} stories")

    indexing_status["done"] = True
    indexing_status["count"] = len(story_files)
    indexing_status["current_file"] = ""

    return total_chunks


# ============== FastAPI App ==============

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start background indexing on startup."""
    # Start indexing in a background thread
    thread = threading.Thread(target=index_all_documents, daemon=True)
    thread.start()
    print("Server started. Document indexing running in background...")
    yield


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)


class ChatRequest(BaseModel):
    message: str


@app.get("/api/status")
def get_status():
    """Check indexing status."""
    return {
        "indexing_complete": indexing_status["done"],
        "documents_indexed": indexing_status["count"],
        "chunks_in_db": indexing_status["chunks"],
        "current_file": indexing_status["current_file"]
    }


@app.post("/api/chat")
def chat(request: ChatRequest):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not configured")

    try:
        # Ensure documents are indexed
        index_all_documents()

        # Run the agent using LangChain 1.0 API
        response = agent.invoke({
            "messages": [{"role": "user", "content": request.message}]
        })

        # Extract reasoning from message history
        messages = response["messages"]
        reasoning_parts = []
        final_answer = ""

        for msg in messages:
            msg_type = getattr(msg, 'type', None)

            # Skip the user message
            if msg_type == 'human':
                continue

            # Check for tool calls (agent deciding to use a tool)
            if msg_type == 'ai' and hasattr(msg, 'tool_calls') and msg.tool_calls:
                for tool_call in msg.tool_calls:
                    reasoning_parts.append(f"Action: {tool_call['name']}")
                    reasoning_parts.append(f"Input: {tool_call['args']}")

            # Check for tool responses (ToolMessage)
            elif msg_type == 'tool':
                tool_name = getattr(msg, 'name', 'search_materials')
                content = msg.content if msg.content else ''
                # Truncate long content
                display_content = content[:800] + '...' if len(content) > 800 else content
                reasoning_parts.append(f"Observation from {tool_name}:\n{display_content}")

            # The final AI message is the answer (has content but no tool calls)
            elif msg_type == 'ai' and msg.content:
                tool_calls = getattr(msg, 'tool_calls', [])
                if not tool_calls:
                    final_answer = msg.content

        reasoning = "\n\n".join(reasoning_parts) if reasoning_parts else None

        return {"reply": final_answer, "reasoning": reasoning}

    except Exception as e:
        error_detail = f"{type(e).__name__}: {str(e)}"
        raise HTTPException(status_code=500, detail=error_detail)


# Serve frontend static files (local development only)
if not IS_VERCEL:
    frontend_path = Path(__file__).parent.parent / "frontend"
    if frontend_path.exists():
        app.mount("/", StaticFiles(directory=frontend_path, html=True), name="frontend")
