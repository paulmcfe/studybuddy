from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from pathlib import Path
import os
from dotenv import load_dotenv

# Load .env from parent directory (v2-rag-from-scratch/)
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)

# Check if running on Vercel
IS_VERCEL = os.environ.get("VERCEL") == "1"

# Import RAG functions - try both import styles for local vs Vercel
try:
    from api.rag import answer_question, index_study_material, vector_db
except ImportError:
    from rag import answer_question, index_study_material, vector_db

app = FastAPI()

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
        "indexing_complete": len(vector_db.vectors) > 0,
        "chunks_in_db": len(vector_db.vectors)
    }


@app.post("/api/chat")
def chat(request: ChatRequest):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not configured")

    try:
        reply = answer_question(request.message)
        return {"reply": reply}

    except Exception as e:
        # Include more detail for debugging
        import traceback
        error_detail = f"{type(e).__name__}: {str(e)}"
        raise HTTPException(status_code=500, detail=error_detail)


# Serve frontend static files (local development only)
# On Vercel, static files are served separately
if not IS_VERCEL:
    frontend_path = Path(__file__).parent.parent / "frontend"
    if frontend_path.exists():
        app.mount("/", StaticFiles(directory=frontend_path, html=True), name="frontend")
