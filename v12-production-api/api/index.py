"""StudyBuddy v12 - Production API

FastAPI application with:
- JWT authentication with registration, login, password reset (v12)
- Multi-user data isolation (v12)
- Rate limiting per user (v12)
- Production monitoring and health checks (v12)
- Learning program CRUD
- Document upload and indexing
- Flashcard generation and spaced repetition
- HTTP streaming for chat
- Curriculum generation
- MCP Connectors: Fetch/URL, GitHub, Brave Search (v11)
"""

import asyncio
import os
import re
import hashlib
import logging
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from fastapi import (
    FastAPI,
    HTTPException,
    Depends,
    Request,
    UploadFile,
    File,
    BackgroundTasks,
    Query,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse, PlainTextResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
import aiofiles

from dotenv import load_dotenv

load_dotenv()

from .database import (
    init_database,
    get_db_dependency,
    get_or_create_user,
    LearningProgram,
    ConnectorConfig,
    Document,
    Flashcard,
    Conversation,
    Message,
    ProgramStats,
    User,
)
from .auth import (
    get_current_user,
    hash_password,
    verify_password,
    create_access_token,
    decode_access_token,
)
from .rate_limit import limiter
from .services.monitoring import metrics, normalize_path

logger = logging.getLogger(__name__)
from .services.retrieval import (
    get_program_retriever,
    get_program_vector_store,
    ensure_collection_exists,
)
from .services.indexing import (
    load_document,
    chunk_documents,
    index_document_to_program,
)
from .services.curriculum import (
    generate_curriculum,
    generate_curriculum_from_documents,
    parse_topic_list,
    topic_list_to_markdown,
    count_topics,
)
from .services.flashcard import (
    generate_flashcard,
    get_due_flashcards,
    update_flashcard_sm2,
    get_flashcard_stats,
)

# Initialize database on startup
init_database()

# Create FastAPI app
app = FastAPI(
    title="StudyBuddy v12",
    description="Production learning application with authentication, monitoring, and AI-powered tutoring",
    version="12.0.0",
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# v12: Rate limiting
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


# v12: Monitoring middleware
@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    """Collect request metrics for every HTTP request."""
    metrics.increment_active()
    start_time = time.time()
    try:
        response = await call_next(request)
        duration = time.time() - start_time
        path = normalize_path(request.url.path)
        metrics.record_request(request.method, path, response.status_code, duration)
        return response
    finally:
        metrics.decrement_active()


# File upload configuration
# Use /tmp on Vercel (read-only filesystem), local uploads/ otherwise
if os.environ.get("VERCEL"):
    UPLOAD_DIR = Path("/tmp/uploads")
else:
    UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)
ALLOWED_TYPES = {"application/pdf", "text/markdown", "text/plain"}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB


# ============== v12: Auth Pydantic Models ==============


class RegisterRequest(BaseModel):
    email: str
    password: str


class LoginRequest(BaseModel):
    email: str
    password: str


class PasswordResetRequest(BaseModel):
    email: str


class PasswordResetConfirm(BaseModel):
    token: str
    new_password: str


# ============== Pydantic Models ==============


class ProgramCreate(BaseModel):
    name: str
    description: Optional[str] = None
    topic_list_markdown: Optional[str] = None


class ProgramUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    topic_list_markdown: Optional[str] = None
    status: Optional[str] = None


class CurriculumGenerateRequest(BaseModel):
    topic: str
    depth: str = "intermediate"
    chapter_count: int = 8


class FlashcardGenerateRequest(BaseModel):
    topic: Optional[str] = None  # Optional - auto-selects topic if not provided


class FlashcardReviewRequest(BaseModel):
    quality: int  # 0-5


class ChatMessage(BaseModel):
    message: str
    conversation_id: Optional[str] = None


class ChatMessageItem(BaseModel):
    role: str  # 'user' or 'assistant'
    content: str


class ChatRequest(BaseModel):
    message: str
    history: list[ChatMessageItem] = []


# v11: Connector request models

class ConnectorCreate(BaseModel):
    connector_type: str  # "fetch", "github", "brave_search"
    name: str
    config: dict = {}


class ConnectorUpdate(BaseModel):
    name: Optional[str] = None
    config: Optional[dict] = None


class FetchImportRequest(BaseModel):
    url: str


class GitHubImportRequest(BaseModel):
    file_paths: list[str]


# ============== v12: Authentication Endpoints ==============


@app.post("/api/auth/register")
@limiter.limit("5/minute")
def register(
    request_obj: RegisterRequest,
    request: Request,
    db: Session = Depends(get_db_dependency),
):
    """Register a new user account.

    Creates the user, generates a verification token (logged to console),
    and returns a JWT so the user can start using the app immediately.
    """
    # Validate email format
    if not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", request_obj.email):
        raise HTTPException(status_code=400, detail="Invalid email format")

    if len(request_obj.password) < 8:
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters")

    # Check email uniqueness
    existing = db.query(User).filter_by(email=request_obj.email).first()
    if existing:
        raise HTTPException(status_code=409, detail="Email already registered")

    # Create user
    verification_token = str(uuid.uuid4())
    user = User(
        id=str(uuid.uuid4()),
        email=request_obj.email,
        password_hash=hash_password(request_obj.password),
        is_active=True,
        is_verified=False,
        verification_token=verification_token,
        preferences={},
    )

    db.add(user)
    db.commit()
    db.refresh(user)

    # Log verification URL (educational simplification - no email service)
    print(f"[AUTH] Verification URL for {user.email}: /api/auth/verify-email?token={verification_token}")

    # Return JWT token
    token = create_access_token(user.id, user.email)
    return {
        "access_token": token,
        "token_type": "bearer",
        "user_id": user.id,
        "email": user.email,
    }


@app.post("/api/auth/login")
@limiter.limit("10/minute")
def login(
    request_obj: LoginRequest,
    request: Request,
    db: Session = Depends(get_db_dependency),
):
    """Authenticate and return a JWT token."""
    user = db.query(User).filter_by(email=request_obj.email).first()

    if not user or not user.password_hash:
        raise HTTPException(status_code=401, detail="Invalid email or password")

    if not verify_password(request_obj.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    if not user.is_active:
        raise HTTPException(status_code=401, detail="Account is deactivated")

    token = create_access_token(user.id, user.email)
    return {
        "access_token": token,
        "token_type": "bearer",
        "user_id": user.id,
        "email": user.email,
    }


@app.post("/api/auth/verify-email")
def verify_email(
    token: str = Query(...),
    db: Session = Depends(get_db_dependency),
):
    """Verify a user's email address using the verification token."""
    user = db.query(User).filter_by(verification_token=token).first()

    if not user:
        raise HTTPException(status_code=400, detail="Invalid verification token")

    user.is_verified = True
    user.verification_token = None
    db.commit()

    return {"status": "verified", "email": user.email}


@app.post("/api/auth/request-password-reset")
@limiter.limit("5/minute")
def request_password_reset(
    request_obj: PasswordResetRequest,
    request: Request,
    db: Session = Depends(get_db_dependency),
):
    """Request a password reset token.

    Always returns 200 to avoid email enumeration.
    The reset URL is logged to console (educational simplification).
    """
    user = db.query(User).filter_by(email=request_obj.email).first()

    if user:
        reset_token = str(uuid.uuid4())
        user.reset_token = reset_token
        user.reset_token_expires = datetime.utcnow() + timedelta(hours=1)
        db.commit()
        print(f"[AUTH] Password reset URL for {user.email}: /api/auth/reset-password (token: {reset_token})")

    return {"status": "ok", "message": "If that email exists, a reset link has been sent"}


@app.post("/api/auth/reset-password")
def reset_password(
    request_obj: PasswordResetConfirm,
    db: Session = Depends(get_db_dependency),
):
    """Reset password using a valid reset token."""
    user = db.query(User).filter_by(reset_token=request_obj.token).first()

    if not user:
        raise HTTPException(status_code=400, detail="Invalid reset token")

    if user.reset_token_expires and user.reset_token_expires < datetime.utcnow():
        raise HTTPException(status_code=400, detail="Reset token has expired")

    if len(request_obj.new_password) < 8:
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters")

    user.password_hash = hash_password(request_obj.new_password)
    user.reset_token = None
    user.reset_token_expires = None
    db.commit()

    return {"status": "ok", "message": "Password has been reset"}


@app.get("/api/auth/me")
def get_me(
    current_user: User = Depends(get_current_user),
):
    """Get the current user's profile."""
    return {
        "user_id": current_user.id,
        "email": current_user.email,
        "is_verified": current_user.is_verified,
        "created_at": current_user.created_at.isoformat() if current_user.created_at else None,
    }


# ============== Helper Functions ==============


async def get_program_or_404(
    program_id: str,
    user_id: str,
    db: Session,
) -> LearningProgram:
    """Get a program by ID, ensuring it belongs to the user."""
    program = db.query(LearningProgram).filter(
        LearningProgram.id == program_id,
        LearningProgram.user_id == user_id,
    ).first()

    if not program:
        raise HTTPException(status_code=404, detail="Program not found")

    return program


# ============== Program Endpoints ==============


@app.get("/api/programs")
def list_programs(
    db: Session = Depends(get_db_dependency),
    current_user: User = Depends(get_current_user),
):
    """List all programs for the current user."""
    user_id = current_user.id

    programs = db.query(LearningProgram).filter(
        LearningProgram.user_id == user_id,
        LearningProgram.status == "active",
    ).order_by(LearningProgram.updated_at.desc()).all()

    return {
        "programs": [
            {
                "id": p.id,
                "name": p.name,
                "description": p.description,
                "topic_count": count_topics(p.topic_list) if p.topic_list else 0,
                "document_count": len(p.documents),
                "flashcard_count": len(p.flashcards),
                "created_at": p.created_at.isoformat(),
                "updated_at": p.updated_at.isoformat(),
            }
            for p in programs
        ]
    }


@app.post("/api/programs")
def create_program(
    request: ProgramCreate,
    db: Session = Depends(get_db_dependency),
    current_user: User = Depends(get_current_user),
):
    """Create a new learning program."""
    user_id = current_user.id

    # Parse topic list if provided
    topic_list = {}
    if request.topic_list_markdown:
        topic_list = parse_topic_list(request.topic_list_markdown)

    program = LearningProgram(
        user_id=user_id,
        name=request.name,
        description=request.description,
        topic_list=topic_list,
    )

    db.add(program)
    db.commit()
    db.refresh(program)

    # Ensure Qdrant collection exists
    ensure_collection_exists(program.qdrant_collection)

    return {
        "id": program.id,
        "name": program.name,
        "description": program.description,
        "document_count": 0,
        "flashcard_count": 0,
        "qdrant_collection": program.qdrant_collection,
        "created_at": program.created_at.isoformat(),
    }


@app.get("/api/programs/{program_id}")
async def get_program(
    program_id: str,
    db: Session = Depends(get_db_dependency),
    current_user: User = Depends(get_current_user),
):
    """Get a program by ID."""
    user_id = current_user.id
    program = await get_program_or_404(program_id, user_id, db)

    return {
        "id": program.id,
        "name": program.name,
        "description": program.description,
        "topic_list": program.topic_list,
        "topic_list_markdown": topic_list_to_markdown(program.topic_list) if program.topic_list else "",
        "document_count": len(program.documents),
        "flashcard_count": len(program.flashcards),
        "created_at": program.created_at.isoformat(),
        "updated_at": program.updated_at.isoformat(),
    }


@app.patch("/api/programs/{program_id}")
async def update_program(
    program_id: str,
    request: ProgramUpdate,
    db: Session = Depends(get_db_dependency),
    current_user: User = Depends(get_current_user),
):
    """Update a program."""
    user_id = current_user.id
    program = await get_program_or_404(program_id, user_id, db)

    if request.name is not None:
        program.name = request.name
    if request.description is not None:
        program.description = request.description
    if request.topic_list_markdown is not None:
        program.topic_list = parse_topic_list(request.topic_list_markdown)
    if request.status is not None:
        program.status = request.status

    program.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(program)

    return {"status": "updated", "id": program.id}


@app.delete("/api/programs/{program_id}")
async def delete_program(
    program_id: str,
    db: Session = Depends(get_db_dependency),
    current_user: User = Depends(get_current_user),
):
    """Delete a program and all associated data."""
    from sqlalchemy import text

    user_id = current_user.id
    program = await get_program_or_404(program_id, user_id, db)

    # Delete in correct order due to foreign key constraints:
    # 1. flashcards (references learning_programs)
    # 2. messages (references conversations)
    # 3. conversations (references learning_programs)
    # 4. documents (references learning_programs and connector_configs)
    # 5. connector_configs (references learning_programs)
    # 6. learning_programs

    # Delete flashcards
    db.query(Flashcard).filter(Flashcard.program_id == program_id).delete()

    # Delete messages for all conversations in this program
    db.execute(text("""
        DELETE FROM messages
        WHERE conversation_id IN (
            SELECT id FROM conversations WHERE program_id = :program_id
        )
    """), {"program_id": program_id})

    # Delete conversations
    db.execute(text("DELETE FROM conversations WHERE program_id = :program_id"), {"program_id": program_id})

    # Delete documents (before connector_configs due to FK)
    db.query(Document).filter(Document.program_id == program_id).delete()

    # Delete connector configs (v11)
    db.query(ConnectorConfig).filter(ConnectorConfig.program_id == program_id).delete()

    # Delete the program
    db.delete(program)
    db.commit()

    return {"status": "deleted", "id": program_id}


# ============== Curriculum Generation ==============


async def generate_initial_flashcards(program_id: str, num_cards: int = 6):
    """Background task to generate initial flashcards for a new program."""
    from .database.connection import SessionLocal

    db = SessionLocal()
    try:
        program = db.query(LearningProgram).filter(
            LearningProgram.id == program_id
        ).first()

        if not program:
            return

        retriever = get_program_retriever(program)

        # Check if program has a curriculum
        topic_list = program.topic_list or {}
        has_curriculum = bool(topic_list.get("chapters", []))

        if has_curriculum:
            # Use curriculum-based generation
            for _ in range(num_cards):
                topic = await get_next_curriculum_topic(program, db)
                if not topic:
                    topic = f"{program.name} concepts"

                docs = retriever.search(topic, k=3)
                if docs:
                    context = "\n\n".join(doc.page_content for doc in docs)
                else:
                    context = f"General knowledge about {topic}. No specific documents available."

                await generate_flashcard(
                    topic=topic,
                    context=context,
                    program_id=program_id,
                    db=db,
                    program_name=program.name,
                    program_description=program.description or "",
                )
        else:
            # No curriculum - extract diverse chunks from documents
            # Fetch more chunks than needed to ensure diversity
            all_docs = retriever.search(program.name, k=num_cards * 3)

            if not all_docs:
                # No documents indexed yet, use general topic
                for i in range(num_cards):
                    topic = f"{program.name}: concept {i + 1}"
                    context = f"General knowledge about {program.name}."
                    await generate_flashcard(
                        topic=topic,
                        context=context,
                        program_id=program_id,
                        db=db,
                        program_name=program.name,
                        program_description=program.description or "",
                    )
            else:
                # Use each chunk as basis for a unique flashcard
                used_contexts = set()
                cards_generated = 0

                for doc in all_docs:
                    if cards_generated >= num_cards:
                        break

                    # Skip if we've already used very similar content
                    content_preview = doc.page_content[:200]
                    if content_preview in used_contexts:
                        continue
                    used_contexts.add(content_preview)

                    # Extract topic from the chunk content (first line or phrase)
                    lines = doc.page_content.strip().split('\n')
                    first_line = lines[0].strip() if lines else program.name
                    # Clean up the topic - limit length and remove special chars
                    topic = first_line[:80].strip('# ').strip()
                    if not topic or len(topic) < 5:
                        topic = f"{program.name}: section {cards_generated + 1}"

                    context = doc.page_content

                    await generate_flashcard(
                        topic=topic,
                        context=context,
                        program_id=program_id,
                        db=db,
                        program_name=program.name,
                        program_description=program.description or "",
                    )
                    cards_generated += 1

    except Exception as e:
        print(f"Error generating initial flashcards: {e}")
    finally:
        db.close()


@app.post("/api/programs/{program_id}/generate-curriculum")
async def generate_program_curriculum(
    program_id: str,
    request: CurriculumGenerateRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
):
    """Generate a curriculum for a program using AI."""
    from .database.connection import SessionLocal

    # Step 1: Validate program exists (quick DB check, then release connection)
    db = SessionLocal()
    try:
        user_id = current_user.id
        program = await get_program_or_404(program_id, user_id, db)
        # Just verify it exists
    finally:
        db.close()

    # Step 2: Generate curriculum (slow OpenAI call - no DB connection held)
    markdown = await generate_curriculum(
        topic=request.topic,
        depth=request.depth,
        chapter_count=request.chapter_count,
    )

    # Step 3: Save results (quick DB operation)
    db = SessionLocal()
    try:
        program = db.query(LearningProgram).filter(
            LearningProgram.id == program_id
        ).first()

        topic_list = parse_topic_list(markdown)
        program.topic_list = topic_list
        program.updated_at = datetime.utcnow()
        db.commit()
    finally:
        db.close()

    # Generate initial flashcards in the background
    background_tasks.add_task(generate_initial_flashcards, program_id, 6)

    return {
        "status": "generated",
        "topic_count": count_topics(topic_list),
        "markdown": markdown,
    }


# ============== Document Endpoints ==============


@app.get("/api/programs/{program_id}/documents")
async def list_documents(
    program_id: str,
    db: Session = Depends(get_db_dependency),
    current_user: User = Depends(get_current_user),
):
    """List all documents in a program."""
    user_id = current_user.id
    program = await get_program_or_404(program_id, user_id, db)

    return {
        "documents": [
            {
                "id": d.id,
                "filename": d.filename,
                "file_type": d.file_type,
                "file_size": d.file_size,
                "status": d.status,
                "chunks_count": d.chunks_count,
                "created_at": d.created_at.isoformat(),
                "indexed_at": d.indexed_at.isoformat() if d.indexed_at else None,
                "error_message": d.error_message,
                "source_type": d.source_type or "upload",
                "source_url": d.source_url,
            }
            for d in program.documents
        ]
    }


@app.post("/api/programs/{program_id}/documents")
@limiter.limit("10/minute")
async def upload_document(
    program_id: str,
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    db: Session = Depends(get_db_dependency),
    current_user: User = Depends(get_current_user),
):
    """Upload a document to a learning program."""
    user_id = current_user.id
    program = await get_program_or_404(program_id, user_id, db)

    # Validate file type
    content_type = file.content_type or ""
    filename = file.filename or "unknown"

    # Check extension for markdown files (content_type may be wrong)
    if filename.endswith(".md") or filename.endswith(".markdown"):
        content_type = "text/markdown"
    elif filename.endswith(".txt"):
        content_type = "text/plain"

    if content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed: PDF, Markdown, plain text",
        )

    # Read and validate file size
    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"File exceeds {MAX_FILE_SIZE // 1024 // 1024}MB limit",
        )

    # Check for duplicates
    content_hash = hashlib.sha256(content).hexdigest()
    existing = db.query(Document).filter(
        Document.program_id == program_id,
        Document.content_hash == content_hash,
    ).first()

    if existing:
        raise HTTPException(
            status_code=409,
            detail=f"Document already uploaded as '{existing.filename}'",
        )

    # Save uploaded file
    file_dir = UPLOAD_DIR / program_id
    file_dir.mkdir(parents=True, exist_ok=True)
    file_path = file_dir / filename

    async with aiofiles.open(file_path, "wb") as f:
        await f.write(content)

    # Create document record
    document = Document(
        program_id=program_id,
        filename=filename,
        file_type=content_type,
        file_size=len(content),
        file_path=str(file_path),
        content_hash=content_hash,
        status="pending",
    )

    db.add(document)
    db.commit()
    db.refresh(document)

    # Queue indexing
    background_tasks.add_task(
        index_document_background,
        document_id=document.id,
        program_id=program_id,
    )

    return {
        "document_id": document.id,
        "status": "pending",
        "message": "Document uploaded. Indexing in progress.",
    }


async def index_document_background(document_id: str, program_id: str):
    """Background task to index a document."""
    from .database.connection import SessionLocal

    db = SessionLocal()
    try:
        document = db.query(Document).filter(Document.id == document_id).first()
        program = db.query(LearningProgram).filter(LearningProgram.id == program_id).first()

        if not document or not program:
            return

        document.status = "processing"
        db.commit()

        try:
            chunks_count = await index_document_to_program(document, program)

            document.status = "indexed"
            document.chunks_count = chunks_count
            document.indexed_at = datetime.utcnow()
            db.commit()

            # Generate/update curriculum from all indexed documents
            await generate_curriculum_from_indexed_documents(program_id)

            # Generate flashcards if the program has fewer than 6
            existing_count = db.query(Flashcard).filter(
                Flashcard.program_id == program_id
            ).count()

            if existing_count < 6:
                cards_to_generate = 6 - existing_count
                await generate_initial_flashcards(program_id, cards_to_generate)

        except Exception as e:
            document.status = "failed"
            document.error_message = str(e)
            db.commit()
            raise

    finally:
        db.close()


async def generate_curriculum_from_indexed_documents(program_id: str):
    """Generate a curriculum based on indexed document content."""
    from .database.connection import SessionLocal
    from qdrant_client.models import Filter, FieldCondition, MatchValue

    db = SessionLocal()
    try:
        program = db.query(LearningProgram).filter(
            LearningProgram.id == program_id
        ).first()

        if not program:
            print(f"Program {program_id} not found for curriculum generation")
            return

        # Get all indexed documents for this program
        indexed_docs = db.query(Document).filter(
            Document.program_id == program_id,
            Document.status == "indexed",
        ).all()

        if not indexed_docs:
            print(f"No indexed documents for program {program_id}")
            return

        # Sample chunks from EVERY document using Qdrant scroll
        # This ensures all documents contribute to the curriculum,
        # regardless of how different their content is from the program description
        from .services.retrieval import get_qdrant_client
        qdrant = get_qdrant_client()
        all_chunks = []
        chunks_per_doc = max(2, 15 // len(indexed_docs))

        for doc in indexed_docs:
            try:
                points, _ = qdrant.scroll(
                    collection_name=program.qdrant_collection,
                    scroll_filter=Filter(
                        must=[
                            FieldCondition(
                                key="metadata.document_id",
                                match=MatchValue(value=doc.id),
                            )
                        ]
                    ),
                    limit=chunks_per_doc,
                    with_payload=True,
                    with_vectors=False,
                )
                for point in points:
                    content = point.payload.get("page_content", "")
                    if content:
                        all_chunks.append(content)
            except Exception as e:
                print(f"Failed to scroll chunks for document {doc.id}: {e}")

        if not all_chunks:
            print(f"No content found in vector store for program {program_id}")
            return

        print(f"Found {len(all_chunks)} chunks from {len(indexed_docs)} documents for curriculum generation")

        # Combine document content for curriculum generation
        document_content = "\n\n---\n\n".join(all_chunks[:20])

        # Generate curriculum from document content
        print(f"Generating curriculum for program {program_id}...")
        markdown = await generate_curriculum_from_documents(
            program_name=program.name,
            program_description=program.description or "",
            document_content=document_content,
            chapter_count=6,
        )

        # Parse and save the curriculum
        topic_list = parse_topic_list(markdown)
        program.topic_list = topic_list
        program.updated_at = datetime.utcnow()
        db.commit()

        print(f"Generated curriculum for program {program_id} with {count_topics(topic_list)} topics")

    except Exception as e:
        import traceback
        print(f"Error generating curriculum from documents: {e}")
        traceback.print_exc()
    finally:
        db.close()


@app.delete("/api/programs/{program_id}/documents/{document_id}")
async def delete_document(
    program_id: str,
    document_id: str,
    db: Session = Depends(get_db_dependency),
    current_user: User = Depends(get_current_user),
):
    """Delete a document from a program."""
    user_id = current_user.id
    program = await get_program_or_404(program_id, user_id, db)

    document = db.query(Document).filter(
        Document.id == document_id,
        Document.program_id == program_id,
    ).first()

    if not document:
        raise HTTPException(status_code=404, detail="Document not found")

    # Delete from Qdrant
    from .services.indexing import delete_document_from_program
    await delete_document_from_program(document, program)

    # Delete file
    if document.file_path and Path(document.file_path).exists():
        Path(document.file_path).unlink()

    # Delete record
    db.delete(document)
    db.commit()

    # Mark program as updated so stats reflect curriculum is stale
    program.updated_at = datetime.utcnow()
    db.commit()

    # Check if any documents remain and regenerate curriculum
    remaining_docs = db.query(Document).filter(
        Document.program_id == program_id,
        Document.status == "indexed",
    ).count()

    if remaining_docs > 0:
        # Regenerate curriculum from remaining documents
        asyncio.create_task(generate_curriculum_from_indexed_documents(program_id))
    else:
        # No documents left - clear the curriculum
        program.topic_list = None
        db.commit()

    return {"status": "deleted", "id": document_id}


# ============== Flashcard Endpoints ==============


@app.get("/api/programs/{program_id}/flashcards")
async def list_flashcards(
    program_id: str,
    topic: Optional[str] = None,
    db: Session = Depends(get_db_dependency),
    current_user: User = Depends(get_current_user),
):
    """List flashcards in a program."""
    user_id = current_user.id
    await get_program_or_404(program_id, user_id, db)

    query = db.query(Flashcard).filter(Flashcard.program_id == program_id)

    if topic:
        query = query.filter(Flashcard.topic == topic)

    flashcards = query.order_by(Flashcard.created_at.desc()).all()

    return {
        "flashcards": [
            {
                "id": c.id,
                "topic": c.topic,
                "question": c.question,
                "answer": c.answer,
                "interval": c.interval,
                "ease_factor": c.ease_factor,
                "next_review": c.next_review.isoformat() if c.next_review else None,
                "created_at": c.created_at.isoformat(),
            }
            for c in flashcards
        ]
    }


@app.get("/api/programs/{program_id}/due-cards")
async def get_due_cards(
    program_id: str,
    limit: int = Query(default=10, ge=1, le=50),
    db: Session = Depends(get_db_dependency),
    current_user: User = Depends(get_current_user),
):
    """Get flashcards due for review."""
    user_id = current_user.id
    await get_program_or_404(program_id, user_id, db)

    due_cards = get_due_flashcards(db, program_id, limit)

    # Get total flashcard count for this program
    total_cards = db.query(Flashcard).filter(Flashcard.program_id == program_id).count()

    return {
        "cards": [
            {
                "id": c.id,
                "topic": c.topic,
                "question": c.question,
                "answer": c.answer,
                "interval": c.interval,
                "repetitions": c.repetitions,
            }
            for c in due_cards
        ],
        "count": len(due_cards),
        "total_cards": total_cards,
    }


@app.post("/api/programs/{program_id}/flashcards/generate")
@limiter.limit("20/minute")
async def generate_flashcard_endpoint(
    program_id: str,
    request: Request,
    request_obj: FlashcardGenerateRequest = None,
    db: Session = Depends(get_db_dependency),
    current_user: User = Depends(get_current_user),
):
    """Generate a flashcard automatically or for a specific topic.

    If no topic is provided, automatically selects the next topic from:
    1. The program's curriculum/topic list
    2. Random content from uploaded documents
    3. Falls back to LLM knowledge about the program's subject
    """
    user_id = current_user.id
    program = await get_program_or_404(program_id, user_id, db)

    # Get topic - either from request or auto-select
    topic = None
    if request_obj and request_obj.topic:
        topic = request_obj.topic

    # Try to get context from documents
    retriever = get_program_retriever(program)
    context = ""
    docs = []

    if topic:
        # Search for specific topic
        docs = retriever.search(topic, k=3)
    else:
        # Auto-select: try curriculum topics first
        topic = await get_next_curriculum_topic(program, db)

        if topic:
            docs = retriever.search(topic, k=3)
        else:
            # No curriculum - sample random content from documents
            # Get existing flashcard contexts to avoid duplicates
            existing_contexts = set(
                fc.source_context[:200] for fc in db.query(Flashcard).filter(
                    Flashcard.program_id == program_id
                ).all() if fc.source_context
            )

            # Search for more chunks to have options
            import random
            all_docs = retriever.search(program.name, k=20)

            # Filter out chunks already used for flashcards
            unused_docs = [
                doc for doc in all_docs
                if doc.page_content[:200] not in existing_contexts
            ]

            if unused_docs:
                doc = random.choice(unused_docs)
                docs = [doc]
            elif all_docs:
                # All chunks used, pick a random one anyway (will likely return cached card)
                doc = random.choice(all_docs)
                docs = [doc]

            topic = f"{program.name} concepts"

    if docs:
        context = "\n\n".join(doc.page_content for doc in docs)
    else:
        # No documents - use LLM knowledge with clear scope boundaries
        if not topic:
            topic = program.name
        description = program.description or program.name

        # Get count of existing flashcards to ensure unique hash for each generation
        existing_count = db.query(Flashcard).filter(
            Flashcard.program_id == program_id
        ).count()

        context = f"""Subject: {program.name}
Description: {description}
Card number: {existing_count + 1}

CRITICAL REQUIREMENTS:
1. Generate a flashcard testing ACTUAL SUBJECT KNOWLEDGE from {program.name}
2. The card must test a specific fact, formula, definition, or concept from the subject itself
3. Do NOT create cards about: study habits, course expectations, how to learn, time management, or any meta-topics about studying
4. Do NOT include concepts from more advanced courses (e.g., no calculus in precalculus)
5. Focus on testable facts: formulas, definitions, properties, examples, and problem-solving techniques
6. Generate a DIFFERENT card than any previous cards - vary the specific topic and concept tested

For precalculus, good topics include: trigonometric identities, unit circle values, polynomial properties, logarithm rules, function transformations, etc."""

    # Generate flashcard
    card = await generate_flashcard(
        topic=topic,
        context=context,
        program_id=program_id,
        db=db,
        program_name=program.name,
        program_description=program.description or "",
    )

    if not card:
        raise HTTPException(
            status_code=500,
            detail="Failed to generate flashcard",
        )

    return {
        "id": card.id,
        "topic": card.topic,
        "question": card.question,
        "answer": card.answer,
        "interval": card.interval,
        "repetitions": card.repetitions,
    }


async def get_next_curriculum_topic(program: LearningProgram, db: Session) -> Optional[str]:
    """Get the next topic from curriculum that doesn't have enough flashcards yet."""
    topic_list = program.topic_list or {}
    chapters = topic_list.get("chapters", [])

    if not chapters:
        return None

    # Flatten all topics from the curriculum
    all_topics = []
    for chapter in chapters:
        chapter_title = chapter.get("title", "")
        for topic in chapter.get("topics", []):
            topic_title = topic.get("title", "")
            if topic_title:
                all_topics.append(f"{chapter_title}: {topic_title}")
            for subtopic in topic.get("subtopics", []):
                if subtopic:
                    all_topics.append(f"{topic_title}: {subtopic}")

    if not all_topics:
        return None

    # Find topics with fewest flashcards
    topic_card_counts = {}
    for topic in all_topics:
        count = db.query(Flashcard).filter(
            Flashcard.program_id == program.id,
            Flashcard.topic == topic,
        ).count()
        topic_card_counts[topic] = count

    # Return topic with fewest cards (prioritize those with 0)
    min_count = min(topic_card_counts.values())
    candidates = [t for t, c in topic_card_counts.items() if c == min_count]

    import random
    return random.choice(candidates) if candidates else None


@app.post("/api/programs/{program_id}/flashcards/{card_id}/review")
async def review_flashcard(
    program_id: str,
    card_id: str,
    request: FlashcardReviewRequest,
    db: Session = Depends(get_db_dependency),
    current_user: User = Depends(get_current_user),
):
    """Record a flashcard review."""
    user_id = current_user.id
    await get_program_or_404(program_id, user_id, db)

    card = db.query(Flashcard).filter(
        Flashcard.id == card_id,
        Flashcard.program_id == program_id,
    ).first()

    if not card:
        raise HTTPException(status_code=404, detail="Flashcard not found")

    if not 0 <= request.quality <= 5:
        raise HTTPException(status_code=400, detail="Quality must be 0-5")

    card = update_flashcard_sm2(card, request.quality, db)

    return {
        "next_review": card.next_review.isoformat(),
        "interval": card.interval,
        "ease_factor": card.ease_factor,
    }


@app.get("/api/programs/{program_id}/stats")
async def get_program_stats(
    program_id: str,
    db: Session = Depends(get_db_dependency),
    current_user: User = Depends(get_current_user),
):
    """Get statistics for a program."""
    user_id = current_user.id
    program = await get_program_or_404(program_id, user_id, db)

    flashcard_stats = get_flashcard_stats(db, program_id)

    # Check if any documents were recently indexed (within last 30 seconds)
    # or if the program was recently modified (e.g. document deleted)
    # This indicates curriculum might still be regenerating
    recent_threshold = datetime.utcnow() - timedelta(seconds=30)
    recently_indexed = any(
        d.indexed_at and d.indexed_at > recent_threshold
        for d in program.documents
        if d.status == "indexed"
    )
    # Also flag as recently changed if program was updated recently
    # (covers document deletions that trigger curriculum regeneration)
    program_recently_updated = (
        program.updated_at and program.updated_at > recent_threshold
    )
    recently_changed = recently_indexed or program_recently_updated

    return {
        "program_id": program_id,
        "name": program.name,
        "documents": {
            "total": len(program.documents),
            "indexed": sum(1 for d in program.documents if d.status == "indexed"),
            "pending": sum(1 for d in program.documents if d.status == "pending"),
            "failed": sum(1 for d in program.documents if d.status == "failed"),
            "recently_indexed": recently_changed,
        },
        "flashcards": flashcard_stats,
        "topics": {
            "total": count_topics(program.topic_list) if program.topic_list else 0,
        },
    }


# ============== Chat Endpoints ==============


@app.post("/api/programs/{program_id}/chat")
@limiter.limit("30/minute")
async def chat_with_tutor(
    program_id: str,
    request_obj: ChatRequest,
    request: Request,
    db: Session = Depends(get_db_dependency),
    current_user: User = Depends(get_current_user),
):
    """Chat with the AI tutor using HTTP streaming.

    v11: If a Brave Search connector is configured for this program,
    the tutor can search the web for additional context when documents
    don't have sufficient information.
    """
    from langchain_openai import ChatOpenAI

    user_id = current_user.id
    program = await get_program_or_404(program_id, user_id, db)

    # Initialize retriever
    retriever = get_program_retriever(program)

    # Retrieve context from documents
    docs = retriever.search(request_obj.message, k=3)
    context = "\n\n".join(doc.page_content for doc in docs) if docs else ""

    # v11: Check for Brave Search connector
    brave_connector = (
        db.query(ConnectorConfig)
        .filter(
            ConnectorConfig.program_id == program_id,
            ConnectorConfig.connector_type == "brave_search",
        )
        .first()
    )

    brave_api_key = None
    if brave_connector:
        config = brave_connector.config or {}
        if config.get("enabled", True):
            brave_api_key = config.get("api_key") or os.environ.get("BRAVE_API_KEY")

    # v11: Use agent with Brave Search if configured
    if brave_api_key:
        return await _chat_with_agent(program, request_obj, context, brave_api_key)

    # Default v10 flow: direct LLM with context
    system_prompt = f"""You are a helpful tutor for the learning program: {program.name}

Use the following context to answer questions accurately and helpfully:

{context}

If the context doesn't contain relevant information, say so honestly."""

    # Build messages
    messages = [("system", system_prompt)]
    for msg in request_obj.history:
        messages.append((msg.role, msg.content))
    messages.append(("user", request_obj.message))

    # Create streaming LLM
    llm = ChatOpenAI(model="gpt-4o", streaming=True)

    async def generate():
        """Stream the response chunks."""
        import json
        async for chunk in llm.astream(messages):
            if chunk.content:
                # JSON-encode to preserve newlines in SSE format
                yield f"data: {json.dumps(chunk.content)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


async def _chat_with_agent(
    program: LearningProgram,
    request: ChatRequest,
    context: str,
    brave_api_key: str,
):
    """Chat using the LangChain agent with Brave Search tool.

    Creates a tutor agent that can search the web when document
    context is insufficient.
    """
    import json
    from langchain_core.messages import HumanMessage, AIMessage
    from .agents.tutor import create_tutor_with_search

    agent, mcp_client = await create_tutor_with_search(
        program_name=program.name,
        context=context,
        brave_api_key=brave_api_key,
    )

    if not agent:
        # Fallback: shouldn't happen, but just in case
        raise HTTPException(
            status_code=500,
            detail="Failed to create tutor agent with search",
        )

    # Build message history
    messages = []
    for msg in request.history:
        if msg.role == "user":
            messages.append(HumanMessage(content=msg.content))
        else:
            messages.append(AIMessage(content=msg.content))
    messages.append(HumanMessage(content=request.message))

    async def generate():
        """Stream the agent's response."""
        try:
            async for event in agent.astream(
                {"messages": messages},
                stream_mode="messages",
            ):
                # Handle different event formats
                if isinstance(event, tuple):
                    msg, metadata = event
                    if hasattr(msg, "content") and msg.content and isinstance(msg.content, str):
                        # Only stream AI messages, not tool calls
                        if not hasattr(msg, "tool_calls") or not msg.tool_calls:
                            yield f"data: {json.dumps(msg.content)}\n\n"
                elif hasattr(event, "content") and event.content:
                    if isinstance(event.content, str):
                        yield f"data: {json.dumps(event.content)}\n\n"
        except (TypeError, AttributeError):
            # Fallback: invoke without streaming
            result = await agent.ainvoke({"messages": messages})
            final_messages = result.get("messages", [])
            if final_messages:
                last_msg = final_messages[-1]
                if hasattr(last_msg, "content") and last_msg.content:
                    yield f"data: {json.dumps(last_msg.content)}\n\n"

        yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


# ============== v11: Connector Endpoints ==============


VALID_CONNECTOR_TYPES = {"fetch", "github", "brave_search"}


@app.get("/api/programs/{program_id}/connectors")
async def list_connectors(
    program_id: str,
    db: Session = Depends(get_db_dependency),
    current_user: User = Depends(get_current_user),
):
    """List all connectors for a program."""
    user_id = current_user.id
    await get_program_or_404(program_id, user_id, db)

    connectors = (
        db.query(ConnectorConfig)
        .filter(ConnectorConfig.program_id == program_id)
        .order_by(ConnectorConfig.created_at.desc())
        .all()
    )

    return {
        "connectors": [
            {
                "id": c.id,
                "connector_type": c.connector_type,
                "name": c.name,
                "config": _safe_config(c),
                "status": c.status,
                "last_sync_at": c.last_sync_at.isoformat() if c.last_sync_at else None,
                "error_message": c.error_message,
                "created_at": c.created_at.isoformat(),
                "updated_at": c.updated_at.isoformat(),
            }
            for c in connectors
        ],
        "env": {
            "brave_api_key": bool(os.environ.get("BRAVE_API_KEY")),
            "github_token": bool(os.environ.get("GITHUB_TOKEN")),
        },
    }


def _safe_config(connector: ConnectorConfig) -> dict:
    """Return connector config with sensitive fields masked."""
    config = dict(connector.config or {})
    # Mask tokens and API keys
    for key in ("token", "api_key"):
        if key in config and config[key]:
            config[key] = config[key][:4] + "..." + config[key][-4:]
    return config


@app.post("/api/programs/{program_id}/connectors")
async def create_connector(
    program_id: str,
    request: ConnectorCreate,
    db: Session = Depends(get_db_dependency),
    current_user: User = Depends(get_current_user),
):
    """Create a new connector configuration."""
    user_id = current_user.id
    await get_program_or_404(program_id, user_id, db)

    if request.connector_type not in VALID_CONNECTOR_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid connector type. Must be one of: {', '.join(VALID_CONNECTOR_TYPES)}",
        )

    # For brave_search, only allow one per program
    if request.connector_type == "brave_search":
        existing = (
            db.query(ConnectorConfig)
            .filter(
                ConnectorConfig.program_id == program_id,
                ConnectorConfig.connector_type == "brave_search",
            )
            .first()
        )
        if existing:
            raise HTTPException(
                status_code=409,
                detail="A Brave Search connector already exists for this program",
            )

    connector = ConnectorConfig(
        program_id=program_id,
        connector_type=request.connector_type,
        name=request.name,
        config=request.config,
    )

    db.add(connector)
    db.commit()
    db.refresh(connector)

    return {
        "id": connector.id,
        "connector_type": connector.connector_type,
        "name": connector.name,
        "status": connector.status,
        "created_at": connector.created_at.isoformat(),
    }


@app.get("/api/programs/{program_id}/connectors/{connector_id}")
async def get_connector(
    program_id: str,
    connector_id: str,
    db: Session = Depends(get_db_dependency),
    current_user: User = Depends(get_current_user),
):
    """Get connector details including sync status."""
    user_id = current_user.id
    await get_program_or_404(program_id, user_id, db)

    connector = (
        db.query(ConnectorConfig)
        .filter(
            ConnectorConfig.id == connector_id,
            ConnectorConfig.program_id == program_id,
        )
        .first()
    )

    if not connector:
        raise HTTPException(status_code=404, detail="Connector not found")

    # Count documents imported by this connector
    doc_count = (
        db.query(Document)
        .filter(Document.connector_id == connector_id)
        .count()
    )

    return {
        "id": connector.id,
        "connector_type": connector.connector_type,
        "name": connector.name,
        "config": _safe_config(connector),
        "status": connector.status,
        "last_sync_at": connector.last_sync_at.isoformat() if connector.last_sync_at else None,
        "error_message": connector.error_message,
        "sync_state": connector.sync_state,
        "document_count": doc_count,
        "created_at": connector.created_at.isoformat(),
        "updated_at": connector.updated_at.isoformat(),
    }


@app.patch("/api/programs/{program_id}/connectors/{connector_id}")
async def update_connector(
    program_id: str,
    connector_id: str,
    request: ConnectorUpdate,
    db: Session = Depends(get_db_dependency),
    current_user: User = Depends(get_current_user),
):
    """Update connector configuration."""
    user_id = current_user.id
    await get_program_or_404(program_id, user_id, db)

    connector = (
        db.query(ConnectorConfig)
        .filter(
            ConnectorConfig.id == connector_id,
            ConnectorConfig.program_id == program_id,
        )
        .first()
    )

    if not connector:
        raise HTTPException(status_code=404, detail="Connector not found")

    if request.name is not None:
        connector.name = request.name
    if request.config is not None:
        # Merge with existing config to allow partial updates
        current_config = dict(connector.config or {})
        current_config.update(request.config)
        connector.config = current_config

    connector.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(connector)

    return {"status": "updated", "id": connector.id}


@app.delete("/api/programs/{program_id}/connectors/{connector_id}")
async def delete_connector(
    program_id: str,
    connector_id: str,
    delete_documents: bool = Query(default=False),
    db: Session = Depends(get_db_dependency),
    current_user: User = Depends(get_current_user),
):
    """Delete a connector and optionally its imported documents."""
    user_id = current_user.id
    program = await get_program_or_404(program_id, user_id, db)

    connector = (
        db.query(ConnectorConfig)
        .filter(
            ConnectorConfig.id == connector_id,
            ConnectorConfig.program_id == program_id,
        )
        .first()
    )

    if not connector:
        raise HTTPException(status_code=404, detail="Connector not found")

    if delete_documents:
        # Delete all documents imported by this connector
        imported_docs = (
            db.query(Document)
            .filter(Document.connector_id == connector_id)
            .all()
        )
        for doc in imported_docs:
            # Remove vectors from Qdrant
            from .services.indexing import delete_document_from_program
            try:
                await delete_document_from_program(doc, program)
            except Exception as e:
                logger.warning(f"Failed to delete vectors for document {doc.id}: {e}")

            # Remove file from disk
            if doc.file_path and Path(doc.file_path).exists():
                Path(doc.file_path).unlink()

            db.delete(doc)
    else:
        # Just clear the connector_id on documents
        db.query(Document).filter(
            Document.connector_id == connector_id
        ).update({"connector_id": None})

    db.delete(connector)
    db.commit()

    return {"status": "deleted", "id": connector_id}


# --- Fetch/URL Connector Actions ---


@app.post("/api/programs/{program_id}/connectors/{connector_id}/fetch")
async def fetch_url_import(
    program_id: str,
    connector_id: str,
    request: FetchImportRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db_dependency),
    current_user: User = Depends(get_current_user),
):
    """Import a URL as a document via the Fetch MCP connector.

    Fetches the web page content, saves it as a markdown document,
    and triggers the indexing pipeline.
    """
    user_id = current_user.id
    program = await get_program_or_404(program_id, user_id, db)

    connector = (
        db.query(ConnectorConfig)
        .filter(
            ConnectorConfig.id == connector_id,
            ConnectorConfig.program_id == program_id,
            ConnectorConfig.connector_type == "fetch",
        )
        .first()
    )

    if not connector:
        raise HTTPException(status_code=404, detail="Fetch connector not found")

    # Validate URL
    from urllib.parse import urlparse
    parsed = urlparse(request.url)
    if not parsed.scheme or not parsed.netloc:
        raise HTTPException(status_code=400, detail="Invalid URL")

    # Import the URL (this creates the document record)
    try:
        from .services.connectors.fetch_connector import import_url
        document = await import_url(
            url=request.url,
            program=program,
            connector=connector,
            db=db,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Fetch import failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch URL: {e}",
        )

    # Queue indexing (reuses existing pipeline)
    background_tasks.add_task(
        index_document_background,
        document_id=document.id,
        program_id=program_id,
    )

    return {
        "document_id": document.id,
        "filename": document.filename,
        "status": "pending",
        "message": f"Successfully imported {request.url}",
    }


# --- GitHub Connector Actions ---


@app.get("/api/programs/{program_id}/connectors/{connector_id}/github/files")
async def list_github_files(
    program_id: str,
    connector_id: str,
    path: str = Query(default="", description="Directory path to browse"),
    branch: str = Query(default="main", description="Branch name"),
    db: Session = Depends(get_db_dependency),
    current_user: User = Depends(get_current_user),
):
    """Browse files in the connected GitHub repository."""
    user_id = current_user.id
    await get_program_or_404(program_id, user_id, db)

    connector = (
        db.query(ConnectorConfig)
        .filter(
            ConnectorConfig.id == connector_id,
            ConnectorConfig.program_id == program_id,
            ConnectorConfig.connector_type == "github",
        )
        .first()
    )

    if not connector:
        raise HTTPException(status_code=404, detail="GitHub connector not found")

    config = connector.config or {}
    owner = config.get("owner")
    repo = config.get("repo")
    token = config.get("token", os.environ.get("GITHUB_TOKEN", ""))

    if not owner or not repo:
        raise HTTPException(
            status_code=400,
            detail="GitHub connector not configured: missing owner/repo",
        )

    if not token:
        raise HTTPException(
            status_code=400,
            detail="GitHub token not configured",
        )

    try:
        from .services.connectors.github_connector import list_repo_files
        files = await list_repo_files(
            owner=owner,
            repo=repo,
            token=token,
            path=path,
            branch=branch or config.get("branch", "main"),
        )
    except Exception as e:
        logger.error(f"GitHub file listing failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to browse repository: {e}",
        )

    return {"files": files, "path": path, "branch": branch}


@app.post("/api/programs/{program_id}/connectors/{connector_id}/github/import")
async def import_github_files_endpoint(
    program_id: str,
    connector_id: str,
    request: GitHubImportRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db_dependency),
    current_user: User = Depends(get_current_user),
):
    """Import selected files from the connected GitHub repository."""
    user_id = current_user.id
    program = await get_program_or_404(program_id, user_id, db)

    connector = (
        db.query(ConnectorConfig)
        .filter(
            ConnectorConfig.id == connector_id,
            ConnectorConfig.program_id == program_id,
            ConnectorConfig.connector_type == "github",
        )
        .first()
    )

    if not connector:
        raise HTTPException(status_code=404, detail="GitHub connector not found")

    if not request.file_paths:
        raise HTTPException(status_code=400, detail="No files selected for import")

    config = connector.config or {}
    owner = config.get("owner")
    repo = config.get("repo")
    token = config.get("token", os.environ.get("GITHUB_TOKEN", ""))
    branch = config.get("branch", "main")

    if not owner or not repo or not token:
        raise HTTPException(
            status_code=400,
            detail="GitHub connector not fully configured",
        )

    # Update connector status
    connector.status = "syncing"
    db.commit()

    # Run import as background task
    background_tasks.add_task(
        _github_import_background,
        owner=owner,
        repo=repo,
        token=token,
        file_paths=request.file_paths,
        branch=branch,
        program_id=program_id,
        connector_id=connector_id,
    )

    return {
        "status": "importing",
        "file_count": len(request.file_paths),
        "message": f"Importing {len(request.file_paths)} files from {owner}/{repo}",
    }


async def _github_import_background(
    owner: str,
    repo: str,
    token: str,
    file_paths: list[str],
    branch: str,
    program_id: str,
    connector_id: str,
):
    """Background task to import GitHub files and trigger indexing."""
    from .database.connection import SessionLocal

    db = SessionLocal()
    try:
        program = db.query(LearningProgram).filter(
            LearningProgram.id == program_id
        ).first()
        connector = db.query(ConnectorConfig).filter(
            ConnectorConfig.id == connector_id
        ).first()

        if not program or not connector:
            return

        from .services.connectors.github_connector import import_github_files

        documents = await import_github_files(
            owner=owner,
            repo=repo,
            token=token,
            file_paths=file_paths,
            branch=branch,
            program=program,
            connector=connector,
            db=db,
        )

        # Index each imported document
        for doc in documents:
            await index_document_background(doc.id, program_id)

    except Exception as e:
        logger.error(f"GitHub import background task failed: {e}")
        # Update connector status
        try:
            connector = db.query(ConnectorConfig).filter(
                ConnectorConfig.id == connector_id
            ).first()
            if connector:
                connector.status = "failed"
                connector.error_message = str(e)
                db.commit()
        except Exception:
            pass
    finally:
        db.close()


@app.post("/api/programs/{program_id}/connectors/{connector_id}/github/sync")
async def sync_github_connector(
    program_id: str,
    connector_id: str,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db_dependency),
    current_user: User = Depends(get_current_user),
):
    """Re-sync the GitHub connector: check for changes and re-import updated files."""
    user_id = current_user.id
    await get_program_or_404(program_id, user_id, db)

    connector = (
        db.query(ConnectorConfig)
        .filter(
            ConnectorConfig.id == connector_id,
            ConnectorConfig.program_id == program_id,
            ConnectorConfig.connector_type == "github",
        )
        .first()
    )

    if not connector:
        raise HTTPException(status_code=404, detail="GitHub connector not found")

    sync_state = connector.sync_state or {}
    if not sync_state:
        raise HTTPException(
            status_code=400,
            detail="No files have been imported yet. Use the import endpoint first.",
        )

    config = connector.config or {}
    owner = config.get("owner")
    repo = config.get("repo")
    token = config.get("token", os.environ.get("GITHUB_TOKEN", ""))
    branch = config.get("branch", "main")

    # Re-import all previously imported files (incremental sync checks hashes)
    file_paths = list(sync_state.keys())

    connector.status = "syncing"
    db.commit()

    background_tasks.add_task(
        _github_import_background,
        owner=owner,
        repo=repo,
        token=token,
        file_paths=file_paths,
        branch=branch,
        program_id=program_id,
        connector_id=connector_id,
    )

    return {
        "status": "syncing",
        "file_count": len(file_paths),
        "message": f"Re-syncing {len(file_paths)} files from {owner}/{repo}",
    }


# ============== v12: Health Check & Monitoring ==============


@app.get("/api/health")
def health_check():
    """Enhanced health check with dependency status.

    Checks PostgreSQL and Qdrant connectivity. Returns per-dependency
    status so operators can quickly identify which service is down.
    """
    from sqlalchemy import text as sa_text

    checks = {"api": "healthy"}

    # Check PostgreSQL
    try:
        from .database.connection import engine
        with engine.connect() as conn:
            conn.execute(sa_text("SELECT 1"))
        checks["postgres"] = "healthy"
    except Exception as e:
        checks["postgres"] = f"unhealthy: {str(e)}"

    # Check Qdrant
    try:
        from .services.retrieval import get_qdrant_client
        client = get_qdrant_client()
        client.get_collections()
        checks["qdrant"] = "healthy"
    except Exception as e:
        checks["qdrant"] = f"unhealthy: {str(e)}"

    overall = "healthy" if all(v == "healthy" for v in checks.values()) else "degraded"

    return {
        "status": overall,
        "version": "12.0.0",
        "timestamp": datetime.utcnow().isoformat(),
        "checks": checks,
    }


@app.get("/api/metrics")
def get_metrics():
    """Prometheus-compatible metrics endpoint.

    Returns request counts, latency averages, error counts,
    and active request gauge in Prometheus text format.
    """
    return PlainTextResponse(
        content=metrics.to_prometheus(),
        media_type="text/plain; version=0.0.4",
    )


# ============== Static Files (Frontend) ==============

# Serve frontend if it exists
FRONTEND_DIR = Path(__file__).parent.parent / "frontend" / "dist"
if FRONTEND_DIR.exists():
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIR / "assets"), name="assets")

    @app.get("/")
    async def serve_frontend():
        return FileResponse(FRONTEND_DIR / "index.html")

    @app.get("/{path:path}")
    async def serve_frontend_routes(path: str):
        # Try to serve static file first
        file_path = FRONTEND_DIR / path
        if file_path.exists() and file_path.is_file():
            return FileResponse(file_path)
        # Fall back to index.html for SPA routing
        return FileResponse(FRONTEND_DIR / "index.html")
