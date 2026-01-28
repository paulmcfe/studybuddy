"""StudyBuddy v10 - Full Stack API

FastAPI application with:
- Learning program CRUD
- Document upload and indexing
- Flashcard generation and spaced repetition
- HTTP streaming for chat
- Curriculum generation
"""

import os
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from fastapi import (
    FastAPI,
    HTTPException,
    Depends,
    UploadFile,
    File,
    BackgroundTasks,
    Query,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session
import aiofiles

from dotenv import load_dotenv

load_dotenv()

from .database import (
    init_database,
    get_db_dependency,
    get_or_create_user,
    LearningProgram,
    Document,
    Flashcard,
    Conversation,
    Message,
    ProgramStats,
)
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
    title="StudyBuddy v10",
    description="Full-stack learning application with AI-powered tutoring",
    version="10.0.0",
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# File upload configuration
# Use /tmp on Vercel (read-only filesystem), local uploads/ otherwise
if os.environ.get("VERCEL"):
    UPLOAD_DIR = Path("/tmp/uploads")
else:
    UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)
ALLOWED_TYPES = {"application/pdf", "text/markdown", "text/plain"}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB


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


# ============== Helper Functions ==============


def get_current_user_id() -> str:
    """Get current user ID. In v10, this is hardcoded; auth comes in Chapter 12."""
    return "default"


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
):
    """List all programs for the current user."""
    user_id = get_current_user_id()
    get_or_create_user(db, user_id)

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
):
    """Create a new learning program."""
    user_id = get_current_user_id()
    get_or_create_user(db, user_id)

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
):
    """Get a program by ID."""
    user_id = get_current_user_id()
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
):
    """Update a program."""
    user_id = get_current_user_id()
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
):
    """Delete a program and all associated data."""
    from sqlalchemy import text

    user_id = get_current_user_id()
    program = await get_program_or_404(program_id, user_id, db)

    # Delete in correct order due to foreign key constraints:
    # 1. flashcards (references learning_programs)
    # 2. messages (references conversations)
    # 3. conversations (references learning_programs)
    # 4. documents (references learning_programs)
    # 5. learning_programs

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

    # Delete documents
    db.query(Document).filter(Document.program_id == program_id).delete()

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
):
    """Generate a curriculum for a program using AI."""
    from .database.connection import SessionLocal

    # Step 1: Validate program exists (quick DB check, then release connection)
    db = SessionLocal()
    try:
        user_id = get_current_user_id()
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
):
    """List all documents in a program."""
    user_id = get_current_user_id()
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
            }
            for d in program.documents
        ]
    }


@app.post("/api/programs/{program_id}/documents")
async def upload_document(
    program_id: str,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    db: Session = Depends(get_db_dependency),
):
    """Upload a document to a learning program."""
    user_id = get_current_user_id()
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

    db = SessionLocal()
    try:
        program = db.query(LearningProgram).filter(
            LearningProgram.id == program_id
        ).first()

        if not program:
            print(f"Program {program_id} not found for curriculum generation")
            return

        # Get document content from the vector store
        retriever = get_program_retriever(program)

        # Use multiple search queries to get diverse content from all documents
        search_queries = [
            program.name,
            program.description or program.name,
            "introduction overview",
            "key concepts main topics",
        ]

        all_docs = []
        seen_content = set()

        for query in search_queries:
            docs = retriever.search(query, k=10)
            for doc in docs:
                # Deduplicate by content hash
                content_hash = hash(doc.page_content[:200])
                if content_hash not in seen_content:
                    seen_content.add(content_hash)
                    all_docs.append(doc)

        if not all_docs:
            print(f"No content found in vector store for program {program_id}")
            return

        print(f"Found {len(all_docs)} unique document chunks for curriculum generation")

        # Combine document content for curriculum generation
        # Take varied samples to cover the breadth of the material
        document_content = "\n\n---\n\n".join(
            doc.page_content for doc in all_docs[:15]  # Limit to prevent token overflow
        )

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
):
    """Delete a document from a program."""
    user_id = get_current_user_id()
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
        program.updated_at = datetime.utcnow()
        db.commit()

    return {"status": "deleted", "id": document_id}


# ============== Flashcard Endpoints ==============


@app.get("/api/programs/{program_id}/flashcards")
async def list_flashcards(
    program_id: str,
    topic: Optional[str] = None,
    db: Session = Depends(get_db_dependency),
):
    """List flashcards in a program."""
    user_id = get_current_user_id()
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
):
    """Get flashcards due for review."""
    user_id = get_current_user_id()
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
async def generate_flashcard_endpoint(
    program_id: str,
    request: FlashcardGenerateRequest = None,
    db: Session = Depends(get_db_dependency),
):
    """Generate a flashcard automatically or for a specific topic.

    If no topic is provided, automatically selects the next topic from:
    1. The program's curriculum/topic list
    2. Random content from uploaded documents
    3. Falls back to LLM knowledge about the program's subject
    """
    user_id = get_current_user_id()
    program = await get_program_or_404(program_id, user_id, db)

    # Get topic - either from request or auto-select
    topic = None
    if request and request.topic:
        topic = request.topic

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
):
    """Record a flashcard review."""
    user_id = get_current_user_id()
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
):
    """Get statistics for a program."""
    user_id = get_current_user_id()
    program = await get_program_or_404(program_id, user_id, db)

    flashcard_stats = get_flashcard_stats(db, program_id)

    # Check if any documents were recently indexed (within last 30 seconds)
    # This indicates curriculum might still be regenerating
    recent_threshold = datetime.utcnow() - timedelta(seconds=30)
    recently_indexed = any(
        d.indexed_at and d.indexed_at > recent_threshold
        for d in program.documents
        if d.status == "indexed"
    )

    return {
        "program_id": program_id,
        "name": program.name,
        "documents": {
            "total": len(program.documents),
            "indexed": sum(1 for d in program.documents if d.status == "indexed"),
            "pending": sum(1 for d in program.documents if d.status == "pending"),
            "failed": sum(1 for d in program.documents if d.status == "failed"),
            "recently_indexed": recently_indexed,
        },
        "flashcards": flashcard_stats,
        "topics": {
            "total": count_topics(program.topic_list) if program.topic_list else 0,
        },
    }


# ============== Chat Endpoints ==============


@app.post("/api/programs/{program_id}/chat")
async def chat_with_tutor(
    program_id: str,
    request: ChatRequest,
    db: Session = Depends(get_db_dependency),
):
    """Chat with the AI tutor using HTTP streaming."""
    from langchain_openai import ChatOpenAI

    user_id = get_current_user_id()
    program = await get_program_or_404(program_id, user_id, db)

    # Initialize retriever
    retriever = get_program_retriever(program)

    # Retrieve context
    docs = retriever.search(request.message, k=3)
    context = "\n\n".join(doc.page_content for doc in docs) if docs else ""

    # Build system prompt
    system_prompt = f"""You are a helpful tutor for the learning program: {program.name}

Use the following context to answer questions accurately and helpfully:

{context}

If the context doesn't contain relevant information, say so honestly."""

    # Build messages
    messages = [("system", system_prompt)]
    for msg in request.history:
        messages.append((msg.role, msg.content))
    messages.append(("user", request.message))

    # Create streaming LLM
    llm = ChatOpenAI(model="gpt-4o", streaming=True)

    async def generate():
        """Stream the response chunks."""
        async for chunk in llm.astream(messages):
            if chunk.content:
                yield f"data: {chunk.content}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


# ============== Health Check ==============


@app.get("/api/health")
def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "version": "10.0.0",
        "timestamp": datetime.utcnow().isoformat(),
    }


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
