"""StudyBuddy v5 - Multi-Agent Architecture with Supervisor Pattern.

This is the main FastAPI application that orchestrates multiple specialized
agents using the supervisor pattern to provide an intelligent tutoring experience.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from contextlib import asynccontextmanager
from pathlib import Path
import os
import re
import random
import threading
from typing import Optional
from dotenv import load_dotenv

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

# Local imports
from .state import StudyBuddyState
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

# Load environment variables from parent directory
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)

# Check if running on Vercel
IS_VERCEL = os.environ.get("VERCEL") == "1"

# LangSmith setup
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_PROJECT"] = "studybuddy-v5"

# ============== Vector Store Setup ==============

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
qdrant_client = QdrantClient(":memory:")

COLLECTION_NAME = "ai_engineering_guides_v5"

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


def search_materials(query: str, k: int = 4) -> str:
    """Search the knowledge base for relevant content."""
    results = vector_store.similarity_search(query, k=k)

    if not results:
        return ""

    formatted = []
    for doc in results:
        source = doc.metadata.get("source", "Unknown")
        formatted.append(f"[{source}]:\n{doc.page_content}")

    return "\n\n---\n\n".join(formatted)


# ============== LangGraph Node Implementations ==============


def supervisor_node(state: StudyBuddyState) -> dict:
    """Route the request to the appropriate agent."""
    query = state.get("query", "")

    context = {
        "current_mode": state.get("current_mode", "learning"),
        "pending_cards": state.get("pending_cards", []),
        "card_context": state.get("card_context"),
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
    """Generate flashcards."""
    topic = state.get("current_topic", "AI Engineering")
    explanation = state.get("response", "")

    # Search for additional context
    context = search_materials(topic)

    # Generate cards
    cards = generate_cards(card_generator_llm, topic, context, explanation)

    return {"pending_cards": cards}


def quality_checker_node(state: StudyBuddyState) -> dict:
    """Check quality of pending cards."""
    pending = state.get("pending_cards", [])

    if not pending:
        return {"pending_cards": [], "approved_cards": []}

    # Check each card
    approved = check_cards_batch(quality_checker_llm, pending)

    return {"pending_cards": [], "approved_cards": approved}


def respond_node(state: StudyBuddyState) -> dict:
    """Final response node - just passes through."""
    return {}


def route_after_supervisor(state: StudyBuddyState) -> str:
    """Route based on supervisor decision."""
    next_agent = state.get("next_agent", "respond")

    if next_agent in ["tutor", "card_generator", "quality_checker"]:
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
    """Build the complete multi-agent graph."""
    graph = StateGraph(StudyBuddyState)

    # Add nodes
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("tutor", tutor_node)
    graph.add_node("card_generator", card_generator_node)
    graph.add_node("quality_checker", quality_checker_node)
    graph.add_node("respond", respond_node)

    # Entry point
    graph.set_entry_point("supervisor")

    # Supervisor routes to workers
    graph.add_conditional_edges(
        "supervisor",
        route_after_supervisor,
        {
            "tutor": "tutor",
            "card_generator": "card_generator",
            "quality_checker": "quality_checker",
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

    # Respond ends the graph
    graph.add_edge("respond", END)

    return graph.compile()


# Create the agent (will be initialized after indexing)
studybuddy = None
_initialized = False


def ensure_initialized():
    """Ensure documents are indexed and agent is ready. Thread-safe."""
    global studybuddy, _initialized
    if _initialized:
        return

    print("Initializing StudyBuddy v5...")
    index_reference_guides()
    studybuddy = build_studybuddy_graph()
    _initialized = True
    print("Multi-agent system ready!")


# ============== FastAPI Application ==============


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize on startup for local development."""
    if not IS_VERCEL:
        # Local: use background thread for faster startup
        def init():
            ensure_initialized()

        thread = threading.Thread(target=init, daemon=True)
        thread.start()
        print("Server started. Document indexing running in background...")
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


class FlashcardRequest(BaseModel):
    chapter_id: int
    scope: str = "single"
    current_topic: Optional[str] = None
    previous_question: Optional[str] = None


class FlashcardResponse(BaseModel):
    question: str
    answer: str
    topic: str
    source: str


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
        # Build the initial state
        initial_state = {
            "query": request.message,
            "messages": [],
            "current_mode": "learning",
            "pending_cards": [],
            "approved_cards": [],
            "current_topic": "",
            "card_context": request.card_context,
            "response": "",
            "next_agent": None,
        }

        result = studybuddy.invoke(initial_state)

        return ChatResponse(
            reply=result.get("response", "I couldn't generate a response."),
            cards=result.get("approved_cards", []),
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

    # Generate card
    card = generate_single_card(
        card_generator_llm, topic_name, subtopics, context, request.previous_question
    )
    if card is None:
        raise HTTPException(status_code=500, detail="Failed to generate flashcard")

    # Quality check
    approved = check_cards_batch(quality_checker_llm, [card])
    if approved:
        card = approved[0]

    return FlashcardResponse(
        question=card.get("question", ""),
        answer=card.get("answer", ""),
        topic=topic_name,
        source="rag" if context else "llm",
    )


# Serve static files (local development only)
if not IS_VERCEL:
    frontend_path = Path(__file__).parent.parent / "frontend"
    if frontend_path.exists():
        app.mount(
            "/", StaticFiles(directory=frontend_path, html=True), name="frontend"
        )
