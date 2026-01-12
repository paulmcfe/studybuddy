"""StudyBuddy v7 - Deep Agents with Curriculum Planning.

This is the main FastAPI application that orchestrates multiple specialized
agents with persistence, memory, flashcard caching, spaced repetition, and
curriculum planning.

New in v7:
- Curriculum Planner agent for learning path management
- Curriculum progress tracking with checkpoints
- Study session management
- Backtracking and adaptive learning paths
- Hybrid flashcard generation (first module upfront, others on-demand)
"""

from pathlib import Path
from dotenv import load_dotenv

# Load environment variables FIRST (before any imports that need them)
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from contextlib import asynccontextmanager
import os
import re
import random
import threading
from typing import Optional
from sqlalchemy.orm import Session

# LangChain imports
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader

# LangGraph imports
from langgraph.graph import StateGraph, END

# Qdrant imports
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

# Local imports (after dotenv loaded)
from .state import StudyBuddyState, create_initial_state
from .agents.tutor import create_tutor_agent, tutor_explain
from .agents.card_generator import (
    create_card_generator_agent,
    generate_cards,
    generate_single_card,
)
from .agents.quality_checker import (
    create_quality_checker_agent,
    check_cards_batch,
)
from .agents.supervisor import (
    create_supervisor_agent,
    route_request,
    should_generate_cards,
)
from .agents.scheduler import (
    create_scheduler_agent,
    get_due_cards,
    get_new_cards,
    record_review,
    get_study_stats,
)
from .agents.curriculum_planner import (
    create_curriculum_planner,
    generate_curriculum,
    generate_curriculum_flashcards,
    evaluate_checkpoint,
    summarize_learning_progress,
    get_session_context,
    resume_study_session,
    handle_learning_struggle,
)
from .database.connection import get_db, init_database, get_db_dependency
from .services.flashcard_cache import (
    get_or_generate_flashcards,
    get_cache_stats,
)
from .services.spaced_repetition import quality_from_button
from .services.memory_store import MemoryStore
from .services.background_generator import BackgroundGenerator, prefetch_status
from .services.curriculum_service import (
    create_curriculum as create_curriculum_record,
    get_user_curricula,
    get_curriculum_by_id,
    get_curriculum_progress,
    update_curriculum_progress,
    increment_module_attempts,
    get_module_performance,
    start_study_session,
    end_study_session,
)

# Check if running on Vercel
IS_VERCEL = os.environ.get("VERCEL") == "1"

# LangSmith setup
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_PROJECT"] = "studybuddy-v7"

# ============== Vector Store Setup ==============

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
qdrant_client = QdrantClient(":memory:")

COLLECTION_NAME = "ai_engineering_guides_v7"

qdrant_client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
)

vector_store = QdrantVectorStore(
    client=qdrant_client, collection_name=COLLECTION_NAME, embedding=embeddings
)

# ============== Document Indexing ==============

indexing_status = {"done": False, "count": 0, "chunks": 0, "current_file": ""}


def index_reference_guides():
    """Index all reference guide markdown files."""
    documents_dir = Path(__file__).parent.parent / "documents"
    if not documents_dir.exists():
        print(f"Documents directory not found: {documents_dir}")
        indexing_status["done"] = True
        return 0

    guide_files = sorted(documents_dir.glob("ref-*.md"))
    if not guide_files:
        print("No ref-*.md files found in documents directory")
        indexing_status["done"] = True
        return 0

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=100, length_function=len
    )

    print("Indexing reference guides...")
    total_chunks = 0
    for filepath in guide_files:
        doc_name = filepath.stem.replace("ref-", "").replace("-", " ").title()
        indexing_status["current_file"] = doc_name

        loader = TextLoader(str(filepath))
        documents = loader.load()
        chunks = text_splitter.split_documents(documents)

        # Add source metadata
        for chunk in chunks:
            chunk.metadata["source"] = doc_name

        vector_store.add_documents(chunks)
        total_chunks += len(chunks)
        indexing_status["chunks"] = total_chunks
        print(f"Indexed {len(chunks)} chunks from {doc_name}")

    print(f"\nTotal: {total_chunks} chunks indexed from {len(guide_files)} guides")

    indexing_status["done"] = True
    indexing_status["count"] = len(guide_files)
    indexing_status["current_file"] = ""

    return total_chunks


# ============== Topic List Parsing ==============

_cached_chapters = None


def parse_topic_list() -> list[dict]:
    """Parse topic-list.md into structured chapter data. Cached after first call."""
    global _cached_chapters
    if _cached_chapters is not None:
        return _cached_chapters

    topic_file = Path(__file__).parent.parent / "documents" / "topic-list.md"
    if not topic_file.exists():
        print(f"topic-list.md not found at {topic_file}")
        return []

    content = topic_file.read_text()
    chapters = []
    current_chapter = None
    current_section = None

    for line in content.split("\n"):
        line = line.rstrip()

        # Match chapter heading: # Chapter X: Title
        chapter_match = re.match(r"^# Chapter (\d+):\s*(.+)$", line)
        if chapter_match:
            if current_chapter:
                chapters.append(current_chapter)
            current_chapter = {
                "id": int(chapter_match.group(1)),
                "title": chapter_match.group(2).strip(),
                "sections": [],
            }
            current_section = None
            continue

        # Match section heading: ## Section Name
        section_match = re.match(r"^## (.+)$", line)
        if section_match and current_chapter:
            current_section = {"name": section_match.group(1).strip(), "subtopics": []}
            current_chapter["sections"].append(current_section)
            continue

        # Match subtopic: - Subtopic text
        subtopic_match = re.match(r"^- (.+)$", line)
        if subtopic_match and current_section:
            current_section["subtopics"].append(subtopic_match.group(1).strip())

    # Add the last chapter
    if current_chapter:
        chapters.append(current_chapter)

    _cached_chapters = chapters
    print(f"Parsed {len(chapters)} chapters from topic-list.md")
    return chapters


def get_topics_for_scope(chapter_id: int, scope: str) -> list[dict]:
    """Get sections/topics for the given scope."""
    chapters = parse_topic_list()
    topics = []

    if scope == "cumulative":
        target_chapters = [c for c in chapters if c["id"] <= chapter_id]
    else:
        target_chapters = [c for c in chapters if c["id"] == chapter_id]

    for chapter in target_chapters:
        for section in chapter["sections"]:
            topics.append(
                {
                    "chapter_id": chapter["id"],
                    "chapter_title": chapter["title"],
                    "section_name": section["name"],
                    "subtopics": section["subtopics"],
                }
            )

    return topics


# ============== Agent Setup ==============

# Create the specialized agents
tutor_llm = create_tutor_agent("gpt-4o")
card_generator_llm = create_card_generator_agent("gpt-4o-mini")
quality_checker_llm = create_quality_checker_agent("gpt-4o")
supervisor_llm = create_supervisor_agent("gpt-4o")
scheduler_llm = create_scheduler_agent("gpt-4o")
curriculum_planner_llm = create_curriculum_planner("gpt-4o")  # v7: Deep agent

def search_materials(query: str, k: int = 4) -> str:
    """Search the knowledge base for relevant content."""
    try:
        results = vector_store.similarity_search(query, k=k)
    except ValueError as e:
        # Qdrant can have shape mismatches if collection is corrupted
        print(f"Vector store search error: {e}")
        return ""

    if not results:
        return ""

    formatted = []
    for doc in results:
        source = doc.metadata.get("source", "Unknown")
        formatted.append(f"[{source}]:\n{doc.page_content}")

    return "\n\n---\n\n".join(formatted)


# Initialize background generator now that search_materials is defined
background_generator = BackgroundGenerator(
    card_generator_llm=card_generator_llm,
    search_func=search_materials,
    get_topics_func=get_topics_for_scope,
)


# ============== LangGraph Node Implementations ==============


def supervisor_node(state: StudyBuddyState) -> dict:
    """Route the request to the appropriate agent."""
    query = state.get("query", "")

    context = {
        "current_mode": state.get("current_mode", "learning"),
        "pending_cards": state.get("pending_cards", []),
        "card_context": state.get("card_context"),
        "due_cards_count": len(state.get("due_cards", [])),
    }

    # Check if we just got an explanation (for auto card generation)
    if state.get("response") and should_generate_cards(state.get("response", "")):
        context["recent_explanation"] = True

    decision = route_request(supervisor_llm, query, context)

    return {"next_agent": decision.get("next_agent", "respond")}


def tutor_node(state: StudyBuddyState) -> dict:
    """Have the Tutor explain a concept."""
    query = state.get("query", "")
    card_context = state.get("card_context")

    # Search for relevant context
    context = search_materials(query)

    # Get the explanation
    explanation = tutor_explain(tutor_llm, query, context, card_context)

    return {"response": explanation, "current_topic": query[:50]}


def card_generator_node(state: StudyBuddyState) -> dict:
    """Generate flashcards with caching."""
    topic = state.get("current_topic", "AI Engineering")
    explanation = state.get("response", "")
    context = search_materials(topic)

    # Use caching service
    with get_db() as db:
        def generator_func(t, c):
            return generate_cards(card_generator_llm, t, c, explanation)

        cards, cache_hit = get_or_generate_flashcards(
            topic=topic,
            source_context=context,
            generator_func=generator_func,
            db=db,
        )

    return {"pending_cards": cards, "cache_hit": cache_hit}


def quality_checker_node(state: StudyBuddyState) -> dict:
    """Check quality of pending cards."""
    pending = state.get("pending_cards", [])

    if not pending:
        return {"pending_cards": [], "approved_cards": []}

    # Check each card
    approved = check_cards_batch(quality_checker_llm, pending)

    return {"pending_cards": [], "approved_cards": approved}


def scheduler_node(state: StudyBuddyState) -> dict:
    """Get due cards and study statistics."""
    user_id = state.get("user_id", "default")

    with get_db() as db:
        due = get_due_cards(db, user_id, limit=10)
        stats = get_study_stats(db, user_id)

    return {
        "due_cards": due,
        "study_stats": stats,
    }


def curriculum_planner_node(state: StudyBuddyState) -> dict:
    """Handle curriculum planning operations.

    This node implements the deep agent pattern:
    - Curriculum generation with backward chaining
    - Progress tracking and checkpoint evaluation
    - Session resumption with appropriate context
    - Backtracking when students struggle
    """
    query = state.get("query", "").lower()
    user_id = state.get("user_id", "default")
    curriculum_id = state.get("curriculum_id")

    # Determine operation from query
    if any(word in query for word in ["create", "plan", "learning path", "curriculum for"]):
        # Generate new curriculum
        context = search_materials(query, k=6)
        curriculum_data = generate_curriculum(
            curriculum_planner_llm,
            goal=query,
            context=context,
        )

        if "error" in curriculum_data:
            return {
                "response": f"Sorry, I couldn't create a curriculum: {curriculum_data.get('error')}",
            }

        # Save curriculum to database
        with get_db() as db:
            curriculum = create_curriculum_record(
                db=db,
                user_id=user_id,
                goal=curriculum_data.get("goal", query),
                curriculum_data=curriculum_data,
                weekly_hours=10,
            )

            # Generate flashcards for first module (hybrid strategy)
            card_results = generate_curriculum_flashcards(
                curriculum=curriculum_data,
                card_generator_llm=card_generator_llm,
                search_func=search_materials,
                db=db,
                module_id=None,  # First module only
            )

        modules = curriculum_data.get("modules", [])
        first_module = modules[0] if modules else {}

        return {
            "response": f"Created curriculum for: {curriculum_data.get('goal')}\n\n"
                       f"Total modules: {len(modules)}\n"
                       f"Estimated duration: {curriculum_data.get('estimated_duration_weeks', 8)} weeks\n"
                       f"Flashcards generated for first module: {card_results.get('total_cards', 0)}\n\n"
                       f"Starting with: {first_module.get('title', 'Module 1')}",
            "curriculum_id": curriculum.id,
            "curriculum_goal": curriculum_data.get("goal"),
            "current_module": first_module,
        }

    elif any(word in query for word in ["resume", "continue", "back to"]):
        # Resume existing curriculum
        if not curriculum_id:
            # Try to find user's active curriculum
            with get_db() as db:
                curricula = get_user_curricula(db, user_id)
                active = [c for c in curricula if c.get("status") == "active"]
                if active:
                    curriculum_id = active[0]["id"]

        if not curriculum_id:
            return {
                "response": "You don't have an active curriculum. Would you like to create one?",
            }

        with get_db() as db:
            session_result = resume_study_session(
                user_id, curriculum_id, db, curriculum_planner_llm
            )

        return {
            "response": session_result.get("message", "Welcome back!"),
            "session_context": session_result.get("context"),
            "curriculum_id": curriculum_id,
        }

    elif any(word in query for word in ["checkpoint", "evaluate", "test me", "ready to advance"]):
        # Evaluate module checkpoint
        if not curriculum_id:
            return {
                "response": "Please select a curriculum first to evaluate checkpoints.",
            }

        with get_db() as db:
            curriculum = get_curriculum_by_id(db, curriculum_id)
            progress = get_curriculum_progress(db, curriculum_id, user_id)

            if not curriculum or not progress:
                return {"response": "Curriculum not found."}

            # Get current module
            current_module = None
            for m in curriculum.curriculum_data.get("modules", []):
                if m.get("id") == progress.current_module_id:
                    current_module = m
                    break

            if not current_module:
                return {"response": "No current module to evaluate."}

            # Get performance for this module's topics
            performance = get_module_performance(
                db, user_id, current_module.get("topics", [])
            )

            # Evaluate checkpoint
            checkpoint_result = evaluate_checkpoint(
                curriculum_planner_llm,
                current_module,
                performance,
            )

            if checkpoint_result.get("passed"):
                # Mark module as complete, advance to next
                modules = curriculum.curriculum_data.get("modules", [])
                current_idx = next(
                    (i for i, m in enumerate(modules) if m.get("id") == current_module.get("id")),
                    -1
                )

                next_module = modules[current_idx + 1] if current_idx + 1 < len(modules) else None

                update_curriculum_progress(
                    db=db,
                    curriculum_id=curriculum_id,
                    user_id=user_id,
                    completed_module_id=current_module.get("id"),
                    current_module_id=next_module.get("id") if next_module else None,
                    checkpoint_result={
                        "module_id": current_module.get("id"),
                        **checkpoint_result
                    },
                )

                if next_module:
                    # Generate flashcards for next module (background prefetch)
                    generate_curriculum_flashcards(
                        curriculum=curriculum.curriculum_data,
                        card_generator_llm=card_generator_llm,
                        search_func=search_materials,
                        db=db,
                        module_id=next_module.get("id"),
                    )

                    return {
                        "response": f"Congratulations! You've completed '{current_module.get('title')}'!\n\n"
                                   f"{checkpoint_result.get('feedback', '')}\n\n"
                                   f"Moving on to: {next_module.get('title')}",
                        "checkpoint_result": checkpoint_result,
                        "current_module": next_module,
                    }
                else:
                    return {
                        "response": f"Amazing! You've completed the entire curriculum!\n\n"
                                   f"{checkpoint_result.get('feedback', '')}",
                        "checkpoint_result": checkpoint_result,
                    }
            else:
                # Checkpoint not passed - suggest reinforcement
                attempts = increment_module_attempts(db, curriculum_id, user_id, current_module.get("id"))
                struggle_result = handle_learning_struggle(
                    curriculum.curriculum_data,
                    current_module,
                    attempts,
                    curriculum_planner_llm,
                )

                return {
                    "response": f"Not quite ready yet for '{current_module.get('title')}'.\n\n"
                               f"{checkpoint_result.get('feedback', '')}\n\n"
                               f"Recommendation: {struggle_result.get('recommendation', 'Keep practicing!')}",
                    "checkpoint_result": checkpoint_result,
                }

    elif any(word in query for word in ["progress", "how am i doing", "status"]):
        # Get curriculum progress
        if not curriculum_id:
            with get_db() as db:
                curricula = get_user_curricula(db, user_id)
                if not curricula:
                    return {"response": "You don't have any curricula yet. Would you like to create one?"}

                # Return summary of all curricula
                summaries = []
                for c in curricula:
                    summaries.append(
                        f"- {c['goal']}: {c['progress_percentage']:.0f}% complete"
                    )

                return {
                    "response": f"Your learning paths:\n" + "\n".join(summaries),
                    "curriculum_progress": curricula,
                }

        with get_db() as db:
            curriculum = get_curriculum_by_id(db, curriculum_id)
            progress = get_curriculum_progress(db, curriculum_id, user_id)

            if not curriculum:
                return {"response": "Curriculum not found."}

            # Generate progress summary
            summary = summarize_learning_progress(
                curriculum=curriculum.curriculum_data,
                completed_modules=progress.completed_modules if progress else [],
                current_module=progress.current_module_id if progress else "",
                struggle_areas=[],
                llm=curriculum_planner_llm,
            )

            return {
                "response": summary,
                "curriculum_progress": {
                    "goal": curriculum.goal,
                    "progress_percentage": progress.progress_percentage if progress else 0,
                    "completed_modules": progress.completed_modules if progress else [],
                    "current_module_id": progress.current_module_id if progress else None,
                },
            }

    else:
        # Generic curriculum help
        return {
            "response": "I can help you with curriculum planning! Try:\n"
                       "- 'Create a learning path for RAG systems'\n"
                       "- 'Resume my curriculum'\n"
                       "- 'How's my progress?'\n"
                       "- 'I'm ready for the checkpoint'",
        }


def respond_node(state: StudyBuddyState) -> dict:
    """Final response node - just passes through."""
    return {}


def route_after_supervisor(state: StudyBuddyState) -> str:
    """Route based on supervisor decision."""
    next_agent = state.get("next_agent", "respond")

    # v7: added curriculum_planner
    if next_agent in ["tutor", "card_generator", "quality_checker", "scheduler", "curriculum_planner"]:
        return next_agent

    return "respond"


def route_after_tutor(state: StudyBuddyState) -> str:
    """After tutor, check if we should generate cards."""
    response = state.get("response", "")

    if should_generate_cards(response):
        return "card_generator"

    return "respond"


def route_after_card_generator(state: StudyBuddyState) -> str:
    """After generating cards, always check quality."""
    pending = state.get("pending_cards", [])

    if pending:
        return "quality_checker"

    return "respond"


# ============== Build the Multi-Agent Graph ==============


def build_studybuddy_graph():
    """Build the v7 multi-agent graph with curriculum planning."""
    graph = StateGraph(StudyBuddyState)

    # Add nodes (v7: added curriculum_planner)
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("tutor", tutor_node)
    graph.add_node("card_generator", card_generator_node)
    graph.add_node("quality_checker", quality_checker_node)
    graph.add_node("scheduler", scheduler_node)
    graph.add_node("curriculum_planner", curriculum_planner_node)
    graph.add_node("respond", respond_node)

    # Entry point
    graph.set_entry_point("supervisor")

    # Supervisor routes to workers (v7: added curriculum_planner)
    graph.add_conditional_edges(
        "supervisor",
        route_after_supervisor,
        {
            "tutor": "tutor",
            "card_generator": "card_generator",
            "quality_checker": "quality_checker",
            "scheduler": "scheduler",
            "curriculum_planner": "curriculum_planner",
            "respond": "respond",
        },
    )

    # Tutor can trigger card generation
    graph.add_conditional_edges(
        "tutor",
        route_after_tutor,
        {"card_generator": "card_generator", "respond": "respond"},
    )

    # Card generator always goes to quality checker
    graph.add_conditional_edges(
        "card_generator",
        route_after_card_generator,
        {"quality_checker": "quality_checker", "respond": "respond"},
    )

    # Quality checker goes to respond
    graph.add_edge("quality_checker", "respond")

    # Scheduler goes to respond
    graph.add_edge("scheduler", "respond")

    # Curriculum planner goes to respond
    graph.add_edge("curriculum_planner", "respond")

    # Respond ends the graph
    graph.add_edge("respond", END)

    return graph.compile()


# Create the agent (will be initialized after indexing)
studybuddy = None
_initialized = False
_init_lock = threading.Lock()


def ensure_initialized():
    """Ensure documents are indexed and agent is ready. Thread-safe."""
    global studybuddy, _initialized
    if _initialized:
        return

    with _init_lock:
        # Double-check after acquiring lock
        if _initialized:
            return

        print("Initializing StudyBuddy v7...")

        # Initialize database
        init_database()

        # Index documents
        index_reference_guides()

        # Build graph
        studybuddy = build_studybuddy_graph()

        _initialized = True
        print("Multi-agent system with curriculum planning ready!")


# ============== FastAPI Application ==============


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize on startup for local development."""
    if not IS_VERCEL:
        # Local: use background thread for faster startup
        def init():
            ensure_initialized()
            # After indexing completes, start prefetching Chapter 1
            print("Starting background prefetch for Chapter 1...")
            background_generator.start_prefetch(1)

        thread = threading.Thread(target=init, daemon=True)
        thread.start()
        print("Server started. Initialization running in background...")
    yield


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============== Request/Response Models ==============


class ChatRequest(BaseModel):
    message: str
    chapter_id: Optional[int] = None
    scope: Optional[str] = None
    card_context: Optional[dict] = None


class ChatResponse(BaseModel):
    reply: str
    cards: list[dict] = []
    cache_hit: bool = False
    due_cards: list[dict] = []
    study_stats: Optional[dict] = None


class FlashcardRequest(BaseModel):
    chapter_id: int
    scope: str = "single"
    current_topic: Optional[str] = None
    previous_question: Optional[str] = None  # For "Still Learning" to get a different card


class FlashcardResponse(BaseModel):
    question: str
    answer: str
    topic: str
    source: str
    flashcard_id: Optional[str] = None


class ReviewRequest(BaseModel):
    flashcard_id: str
    button: str  # "no", "took_a_sec", or "yes"
    response_time_ms: Optional[int] = None


class ReviewResponse(BaseModel):
    flashcard_id: str
    next_review: str
    interval_days: int
    ease_factor: float
    message: str


class DueCardsResponse(BaseModel):
    due_cards: list[dict]
    new_cards: list[dict]
    stats: dict


class StatsResponse(BaseModel):
    total_reviews: int
    cards_due_now: int
    reviews_today: int
    average_ease_factor: float
    topic_performance: dict
    struggle_areas: list[str]
    unique_cards_studied: int


class PrefetchRequest(BaseModel):
    chapter_id: int


class PrefetchResponse(BaseModel):
    chapter_id: int
    state: str  # "started", "already_in_progress", "already_completed", "no_topics"
    total_topics: int
    message: str


class PrefetchStatusResponse(BaseModel):
    chapter_id: int
    state: str  # "idle", "in_progress", "completed", "error"
    total_topics: int
    completed_topics: int
    current_topic: str
    cards_generated: int
    progress_percent: float


# ============== API Endpoints ==============


@app.get("/api/status")
def get_status():
    """Check indexing status. On Vercel, triggers initialization."""
    if IS_VERCEL:
        ensure_initialized()

    return {
        "indexing_complete": indexing_status["done"],
        "documents_indexed": indexing_status["count"],
        "chunks_in_db": indexing_status["chunks"],
        "current_file": indexing_status["current_file"],
        "db_backend": "postgres",
    }


@app.get("/api/chapters")
def get_chapters():
    """Get all chapters parsed from topic-list.md."""
    chapters = parse_topic_list()
    return {"chapters": chapters}


@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """Handle chat requests through the multi-agent system."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not configured")

    # On Vercel, initialize synchronously on first request
    if IS_VERCEL:
        ensure_initialized()
    elif studybuddy is None:
        raise HTTPException(
            status_code=503,
            detail="Agent still initializing. Please wait for indexing to complete.",
        )

    try:
        # Build the initial state (v6: includes user_id)
        initial_state = create_initial_state(
            query=request.message,
            user_id="default",
            card_context=request.card_context,
        )

        result = studybuddy.invoke(initial_state)

        return ChatResponse(
            reply=result.get("response", "I couldn't generate a response."),
            cards=result.get("approved_cards", []),
            cache_hit=result.get("cache_hit", False),
            due_cards=result.get("due_cards", []),
            study_stats=result.get("study_stats"),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {str(e)}")


@app.post("/api/flashcard", response_model=FlashcardResponse)
def generate_flashcard(request: FlashcardRequest):
    """Generate a single flashcard scoped to selected chapters."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not configured")

    if IS_VERCEL:
        ensure_initialized()
    elif not indexing_status["done"]:
        raise HTTPException(status_code=503, detail="Still indexing documents.")

    # Get topics for the requested scope
    topics = get_topics_for_scope(request.chapter_id, request.scope)
    if not topics:
        raise HTTPException(
            status_code=400, detail=f"No topics found for chapter {request.chapter_id}"
        )

    # Select topic (honor current_topic for "Study More", else random)
    if request.current_topic:
        matching = [t for t in topics if t["section_name"] == request.current_topic]
        selected_topic = matching[0] if matching else random.choice(topics)
    else:
        selected_topic = random.choice(topics)

    topic_name = selected_topic["section_name"]
    subtopics = selected_topic["subtopics"]

    # Search for context
    context = search_materials(topic_name, k=4)

    # First, check if we have cached cards for this topic
    from .services.flashcard_cache import get_cached_flashcards

    try:
        with get_db() as db:
            cached_cards = get_cached_flashcards(topic_name, context, db)

            if cached_cards:
                # Filter out the previous question if in "Still Learning" flow
                available_cards = list(cached_cards)
                if request.previous_question:
                    available_cards = [
                        c for c in cached_cards
                        if c.question != request.previous_question
                    ]
                    # If no other cards available, skip cache and generate new via LLM
                    if not available_cards:
                        cached_cards = None  # Force LLM generation below

                if cached_cards and available_cards:
                    # Return a random cached card
                    selected_card = random.choice(available_cards)
                    return FlashcardResponse(
                        question=selected_card.question,
                        answer=selected_card.answer,
                        topic=topic_name,
                        source="cache",
                        flashcard_id=selected_card.id,
                    )
    except Exception as e:
        # Log cache lookup error but continue to LLM generation
        print(f"Cache lookup error: {e}")

    # No cached cards - generate via LLM
    card = generate_single_card(
        card_generator_llm,
        topic_name,
        subtopics,
        context,
        previous_question=request.previous_question,
    )
    if card is None:
        raise HTTPException(status_code=500, detail="Failed to generate flashcard")

    # Quality check
    approved = check_cards_batch(quality_checker_llm, [card])
    if approved:
        card = approved[0]

    # Save to database so we have an ID for spaced repetition tracking
    from .services.flashcard_cache import cache_flashcards

    try:
        with get_db() as db:
            saved_cards = cache_flashcards(
                topic=topic_name,
                source_context=context,
                cards_data=[card],
                db=db,
                chapter_id=request.chapter_id,
            )
            flashcard_id = saved_cards[0].id if saved_cards else None
    except Exception as e:
        print(f"Failed to cache generated card: {e}")
        flashcard_id = None

    return FlashcardResponse(
        question=card.get("question", ""),
        answer=card.get("answer", ""),
        topic=topic_name,
        source="rag" if context else "llm",
        flashcard_id=flashcard_id,
    )


# ============== v6 Spaced Repetition Endpoints ==============


@app.get("/api/due-cards", response_model=DueCardsResponse)
def get_due_cards_endpoint(
    limit: int = 10,
    topic: Optional[str] = None,
    include_new: bool = True,
    db: Session = Depends(get_db_dependency),
):
    """Get flashcards due for review."""
    due = get_due_cards(db, "default", limit, topic)
    new = get_new_cards(db, "default", 5, topic) if include_new else []
    stats = get_study_stats(db, "default")

    return DueCardsResponse(due_cards=due, new_cards=new, stats=stats)


@app.post("/api/review", response_model=ReviewResponse)
def record_review_endpoint(
    request: ReviewRequest,
    db: Session = Depends(get_db_dependency),
):
    """Record a flashcard review result."""
    # Convert button choice to SM-2 quality rating
    quality = quality_from_button(request.button)

    result = record_review(
        db=db,
        user_id="default",
        flashcard_id=request.flashcard_id,
        quality=quality,
        response_time_ms=request.response_time_ms,
    )

    return ReviewResponse(
        flashcard_id=result["flashcard_id"],
        next_review=result["next_review"],
        interval_days=result["interval_days"],
        ease_factor=result["ease_factor"],
        message=f"Next review in {result['interval_days']} days",
    )


@app.get("/api/stats", response_model=StatsResponse)
def get_stats_endpoint(db: Session = Depends(get_db_dependency)):
    """Get user study statistics."""
    stats = get_study_stats(db, "default")
    return StatsResponse(**stats)


@app.get("/api/cache-stats")
def get_cache_stats_endpoint(db: Session = Depends(get_db_dependency)):
    """Get flashcard cache statistics."""
    return get_cache_stats(db)


# ============== v7 Curriculum Endpoints ==============


class CurriculumRequest(BaseModel):
    goal: str
    current_knowledge: list[str] = []
    weekly_hours: int = 10


class CurriculumResponse(BaseModel):
    curriculum_id: str
    curriculum: dict
    flashcards_generated: int
    estimated_duration_weeks: int


class CurriculumProgressResponse(BaseModel):
    curriculum_id: str
    goal: str
    progress_percentage: float
    current_module: Optional[dict]
    completed_modules: list[str]
    status: str


class ResumeSessionRequest(BaseModel):
    curriculum_id: str


class ResumeSessionResponse(BaseModel):
    message: str
    action: str
    context: dict
    review_topics: list[str] = []


class CheckpointRequest(BaseModel):
    curriculum_id: str
    module_id: Optional[str] = None  # None = current module


class CheckpointResponse(BaseModel):
    passed: bool
    confidence: float
    feedback: str
    recommendations: list[str]
    next_action: str
    metrics: dict


@app.post("/api/curriculum", response_model=CurriculumResponse)
def create_curriculum_endpoint(
    request: CurriculumRequest,
    db: Session = Depends(get_db_dependency),
):
    """Generate a personalized learning curriculum.

    Creates a structured curriculum using backward chaining, then
    generates flashcards for the first module (hybrid strategy).
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not configured")

    if IS_VERCEL:
        ensure_initialized()

    # Search for relevant context
    context = search_materials(request.goal, k=6)

    # Generate curriculum using backward chaining
    curriculum_data = generate_curriculum(
        curriculum_planner_llm,
        goal=request.goal,
        current_knowledge=request.current_knowledge,
        available_time_hours=request.weekly_hours,
        context=context,
    )

    if "error" in curriculum_data:
        raise HTTPException(status_code=500, detail=curriculum_data["error"])

    # Save curriculum to database
    curriculum = create_curriculum_record(
        db=db,
        user_id="default",
        goal=curriculum_data.get("goal", request.goal),
        curriculum_data=curriculum_data,
        weekly_hours=request.weekly_hours,
    )

    # Generate flashcards for first module (hybrid strategy)
    card_results = generate_curriculum_flashcards(
        curriculum=curriculum_data,
        card_generator_llm=card_generator_llm,
        search_func=search_materials,
        db=db,
        module_id=None,  # First module only
    )

    return CurriculumResponse(
        curriculum_id=curriculum.id,
        curriculum=curriculum_data,
        flashcards_generated=card_results.get("total_cards", 0),
        estimated_duration_weeks=curriculum_data.get("estimated_duration_weeks", 8),
    )


@app.get("/api/curriculum/{curriculum_id}", response_model=CurriculumProgressResponse)
def get_curriculum_endpoint(
    curriculum_id: str,
    db: Session = Depends(get_db_dependency),
):
    """Get progress on a specific curriculum."""
    curriculum = get_curriculum_by_id(db, curriculum_id)
    if not curriculum:
        raise HTTPException(status_code=404, detail="Curriculum not found")

    progress = get_curriculum_progress(db, curriculum_id, "default")

    # Get current module details
    current_module = None
    if progress and progress.current_module_id:
        for m in curriculum.curriculum_data.get("modules", []):
            if m.get("id") == progress.current_module_id:
                current_module = m
                break

    return CurriculumProgressResponse(
        curriculum_id=curriculum.id,
        goal=curriculum.goal,
        progress_percentage=progress.progress_percentage if progress else 0,
        current_module=current_module,
        completed_modules=progress.completed_modules if progress else [],
        status=curriculum.status,
    )


@app.get("/api/curricula")
def list_curricula_endpoint(db: Session = Depends(get_db_dependency)):
    """List all curricula for the current user."""
    curricula = get_user_curricula(db, "default")
    return {"curricula": curricula}


@app.post("/api/curriculum/resume", response_model=ResumeSessionResponse)
def resume_curriculum_endpoint(
    request: ResumeSessionRequest,
    db: Session = Depends(get_db_dependency),
):
    """Resume studying a curriculum with appropriate context."""
    curriculum = get_curriculum_by_id(db, request.curriculum_id)
    if not curriculum:
        raise HTTPException(status_code=404, detail="Curriculum not found")

    result = resume_study_session(
        user_id="default",
        curriculum_id=request.curriculum_id,
        db=db,
        llm=curriculum_planner_llm,
    )

    return ResumeSessionResponse(
        message=result.get("message", "Welcome back!"),
        action=result.get("action", "continue"),
        context=result.get("context", {}),
        review_topics=result.get("review_topics", []),
    )


@app.post("/api/curriculum/checkpoint", response_model=CheckpointResponse)
def evaluate_checkpoint_endpoint(
    request: CheckpointRequest,
    db: Session = Depends(get_db_dependency),
):
    """Evaluate a module checkpoint to determine if student can advance."""
    curriculum = get_curriculum_by_id(db, request.curriculum_id)
    if not curriculum:
        raise HTTPException(status_code=404, detail="Curriculum not found")

    progress = get_curriculum_progress(db, request.curriculum_id, "default")
    if not progress:
        raise HTTPException(status_code=400, detail="No progress found for curriculum")

    # Determine module to evaluate
    module_id = request.module_id or progress.current_module_id
    if not module_id:
        raise HTTPException(status_code=400, detail="No module to evaluate")

    # Find the module
    current_module = None
    for m in curriculum.curriculum_data.get("modules", []):
        if m.get("id") == module_id:
            current_module = m
            break

    if not current_module:
        raise HTTPException(status_code=404, detail="Module not found")

    # Get performance metrics
    performance = get_module_performance(
        db, "default", current_module.get("topics", [])
    )

    # Evaluate checkpoint
    result = evaluate_checkpoint(
        curriculum_planner_llm,
        current_module,
        performance,
    )

    # Determine next action
    if result.get("passed"):
        next_action = "advance"
    else:
        attempts = increment_module_attempts(
            db, request.curriculum_id, "default", module_id
        )
        struggle = handle_learning_struggle(
            curriculum.curriculum_data, current_module, attempts, curriculum_planner_llm
        )
        next_action = struggle.get("action", "reinforce")

    return CheckpointResponse(
        passed=result.get("passed", False),
        confidence=result.get("confidence", 0.5),
        feedback=result.get("feedback", ""),
        recommendations=result.get("recommendations", []),
        next_action=next_action,
        metrics=result.get("metrics", {}),
    )


@app.post("/api/curriculum/{curriculum_id}/module/{module_id}/flashcards")
def generate_module_flashcards_endpoint(
    curriculum_id: str,
    module_id: str,
    db: Session = Depends(get_db_dependency),
):
    """Generate flashcards for a specific module (on-demand)."""
    curriculum = get_curriculum_by_id(db, curriculum_id)
    if not curriculum:
        raise HTTPException(status_code=404, detail="Curriculum not found")

    # Find the module
    module = None
    for m in curriculum.curriculum_data.get("modules", []):
        if m.get("id") == module_id:
            module = m
            break

    if not module:
        raise HTTPException(status_code=404, detail="Module not found")

    # Generate flashcards
    result = generate_curriculum_flashcards(
        curriculum=curriculum.curriculum_data,
        card_generator_llm=card_generator_llm,
        search_func=search_materials,
        db=db,
        module_id=module_id,
    )

    return {
        "module_id": module_id,
        "cards_generated": result.get("total_cards", 0),
        "module_info": result.get("modules", {}).get(module_id, {}),
    }


class CurriculumFlashcardRequest(BaseModel):
    curriculum_id: str
    module_id: Optional[str] = None  # None = current module
    current_topic: Optional[str] = None  # For "Still Learning" flow
    previous_question: Optional[str] = None


@app.post("/api/curriculum/flashcard", response_model=FlashcardResponse)
def get_curriculum_flashcard(
    request: CurriculumFlashcardRequest,
    db: Session = Depends(get_db_dependency),
):
    """Get a flashcard from the current curriculum module.

    Returns a random flashcard from the module's topics, prioritizing
    spaced repetition (due cards) when available.
    """
    # 1. Get curriculum and progress
    curriculum = get_curriculum_by_id(db, request.curriculum_id)
    if not curriculum:
        raise HTTPException(status_code=404, detail="Curriculum not found")

    progress = get_curriculum_progress(db, request.curriculum_id, "default")

    # 2. Determine module to use
    module_id = request.module_id
    if not module_id and progress:
        module_id = progress.current_module_id

    # Find the module
    current_module = None
    modules = curriculum.curriculum_data.get("modules", [])
    if module_id:
        for m in modules:
            if m.get("id") == module_id:
                current_module = m
                break

    # Fallback to first module if no current module
    if not current_module and modules:
        current_module = modules[0]

    if not current_module:
        raise HTTPException(status_code=400, detail="No modules in curriculum")

    # 3. Get module's topics
    module_topics = current_module.get("topics", [])
    if not module_topics:
        raise HTTPException(
            status_code=400,
            detail=f"No topics defined for module: {current_module.get('title')}"
        )

    # 4. Try to find flashcards by topic
    from .database.models import Flashcard

    # Build query for flashcards matching any of the module's topics
    topic_filter = module_topics if isinstance(module_topics, list) else [module_topics]

    # First try due cards (spaced repetition)
    due_cards = get_due_cards(db, "default", limit=20, topic_filter=topic_filter)

    if due_cards:
        # Filter out previous question if in "Still Learning" flow
        available_cards = due_cards
        if request.previous_question:
            available_cards = [
                c for c in due_cards if c.get("question") != request.previous_question
            ]

        if available_cards:
            selected = random.choice(available_cards)
            return FlashcardResponse(
                question=selected.get("question", ""),
                answer=selected.get("answer", ""),
                topic=selected.get("topic", module_topics[0]),
                source="curriculum",
                flashcard_id=selected.get("flashcard_id"),
            )

    # Then try new cards (not yet reviewed)
    new_cards = get_new_cards(db, "default", limit=20, topic_filter=topic_filter)

    if new_cards:
        available_cards = new_cards
        if request.previous_question:
            available_cards = [
                c for c in new_cards if c.get("question") != request.previous_question
            ]

        if available_cards:
            selected = random.choice(available_cards)
            return FlashcardResponse(
                question=selected.get("question", ""),
                answer=selected.get("answer", ""),
                topic=selected.get("topic", module_topics[0]),
                source="curriculum",
                flashcard_id=selected.get("flashcard_id"),
            )

    # 5. No cards found - generate on-demand
    # Pick a random topic from the module
    selected_topic = request.current_topic if request.current_topic in module_topics else random.choice(module_topics)

    context = search_materials(selected_topic, k=4)

    card = generate_single_card(
        card_generator_llm,
        selected_topic,
        [],  # No subtopics for curriculum
        context,
        previous_question=request.previous_question,
    )

    if card is None:
        raise HTTPException(status_code=500, detail="Failed to generate flashcard")

    # Quality check
    approved = check_cards_batch(quality_checker_llm, [card])
    if approved:
        card = approved[0]

    # Save to database
    from .services.flashcard_cache import cache_flashcards

    try:
        saved_cards = cache_flashcards(
            topic=selected_topic,
            source_context=context,
            cards_data=[card],
            db=db,
            chapter_id=None,  # Curriculum cards don't belong to chapters
        )
        flashcard_id = saved_cards[0].id if saved_cards else None
    except Exception as e:
        print(f"Failed to cache generated card: {e}")
        flashcard_id = None

    return FlashcardResponse(
        question=card.get("question", ""),
        answer=card.get("answer", ""),
        topic=selected_topic,
        source="curriculum-generated",
        flashcard_id=flashcard_id,
    )


# ============== Background Prefetch Endpoints ==============


@app.post("/api/prefetch", response_model=PrefetchResponse)
def start_prefetch(request: PrefetchRequest):
    """Start background flashcard generation for a chapter.

    Returns immediately. Use /api/prefetch-status/{chapter_id} to check progress.
    """
    if not indexing_status["done"]:
        raise HTTPException(
            status_code=503,
            detail="Document indexing not complete. Please wait.",
        )

    result = background_generator.start_prefetch(request.chapter_id)
    return PrefetchResponse(**result)


@app.get("/api/prefetch-status/{chapter_id}", response_model=PrefetchStatusResponse)
def get_prefetch_status(chapter_id: int):
    """Get background generation progress for a chapter."""
    status = prefetch_status.get(chapter_id)

    if status is None:
        return PrefetchStatusResponse(
            chapter_id=chapter_id,
            state="idle",
            total_topics=0,
            completed_topics=0,
            current_topic="",
            cards_generated=0,
            progress_percent=0.0,
        )

    progress = (
        status["completed_topics"] / status["total_topics"] * 100
        if status["total_topics"] > 0
        else 0
    )

    return PrefetchStatusResponse(
        chapter_id=chapter_id,
        state=status["state"],
        total_topics=status["total_topics"],
        completed_topics=status["completed_topics"],
        current_topic=status.get("current_topic", ""),
        cards_generated=status["cards_generated"],
        progress_percent=round(progress, 1),
    )


# Serve static files (local development only)
if not IS_VERCEL:
    frontend_path = Path(__file__).parent.parent / "frontend"
    if frontend_path.exists():
        app.mount(
            "/", StaticFiles(directory=frontend_path, html=True), name="frontend"
        )
