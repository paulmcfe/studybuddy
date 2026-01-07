from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from contextlib import asynccontextmanager
from pathlib import Path
import asyncio
import os
from dotenv import load_dotenv

# Load .env from parent directory (v2-rag-from-scratch/)
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)

# Import RAG functions after loading env (so OpenAI client gets the key)
from api.rag import answer_question, index_documents_directory, vector_db

# Track indexing status
indexing_status = {"done": False, "count": 0, "chunks": 0}


async def index_documents_background():
    """Index documents in background thread to avoid blocking startup."""
    documents_path = Path(__file__).parent.parent / "documents"
    # Run in thread pool to avoid blocking the event loop
    loop = asyncio.get_event_loop()
    count = await loop.run_in_executor(None, index_documents_directory, str(documents_path))
    indexing_status["done"] = True
    indexing_status["count"] = count
    indexing_status["chunks"] = len(vector_db.vectors)
    print(f"Indexing complete. {count} documents indexed. {len(vector_db.vectors)} chunks in vector DB.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start background indexing on startup."""
    # Start indexing in background (don't await - let it run while server starts)
    asyncio.create_task(index_documents_background())
    print("Server started. Document indexing running in background...")
    yield


app = FastAPI(lifespan=lifespan)

# CORS so the frontend can talk to backend
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
        "chunks_in_db": indexing_status["chunks"] if indexing_status["done"] else len(vector_db.vectors)
    }


@app.post("/api/chat")
def chat(request: ChatRequest):
    if not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not configured")

    try:
        reply = answer_question(request.message)
        return {"reply": reply}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


# Serve frontend static files (must be after API routes)
frontend_path = Path(__file__).parent.parent / "frontend"
if frontend_path.exists():
    app.mount("/", StaticFiles(directory=frontend_path, html=True), name="frontend")
