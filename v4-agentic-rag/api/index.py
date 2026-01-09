from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from contextlib import asynccontextmanager
from pathlib import Path
import threading
import os
import json
import re
import random
from typing import TypedDict, Annotated, Literal, Optional
from dotenv import load_dotenv

# LangChain imports
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader

# LangGraph imports
from langgraph.graph import StateGraph, END, add_messages

# Qdrant imports
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

# Load environment variables from parent directory
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)

# Check if running on Vercel
IS_VERCEL = os.environ.get("VERCEL") == "1"

# LangSmith setup (required)
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_PROJECT"] = "studybuddy-v4"

# ============== Vector Store Setup ==============

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
qdrant_client = QdrantClient(":memory:")

COLLECTION_NAME = "ai_engineering_guides"

qdrant_client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
)

vector_store = QdrantVectorStore(
    client=qdrant_client,
    collection_name=COLLECTION_NAME,
    embedding=embeddings
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
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len
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
                "sections": []
            }
            current_section = None
            continue

        # Match section heading: ## Section Name
        section_match = re.match(r"^## (.+)$", line)
        if section_match and current_chapter:
            current_section = {
                "name": section_match.group(1).strip(),
                "subtopics": []
            }
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
    """Get sections/topics for the given scope.

    Returns list of dicts with 'chapter_id', 'chapter_title', 'section_name', 'subtopics'.
    """
    chapters = parse_topic_list()
    topics = []

    if scope == "cumulative":
        target_chapters = [c for c in chapters if c["id"] <= chapter_id]
    else:
        target_chapters = [c for c in chapters if c["id"] == chapter_id]

    for chapter in target_chapters:
        for section in chapter["sections"]:
            topics.append({
                "chapter_id": chapter["id"],
                "chapter_title": chapter["title"],
                "section_name": section["name"],
                "subtopics": section["subtopics"]
            })

    return topics


# ============== LangGraph State ==============

class StudyBuddyState(TypedDict):
    # Conversation
    messages: Annotated[list, add_messages]

    # Query analysis
    query: str
    query_type: str
    complexity: str
    needs_retrieval: bool
    search_queries: list[str]

    # Context (for flashcard-based chat)
    card_context: dict | None  # {question, answer, topic} of current flashcard

    # Retrieval
    retrieved_docs: list
    retrieval_sufficient: bool
    iteration: int

    # Response
    response: str
    confidence: float

    # Flashcard
    flashcard_suggestion: dict | None


# ============== LLM Setup ==============

llm = ChatOpenAI(model="gpt-4o-mini")


# ============== Node Implementations ==============

def analyze_query_node(state: StudyBuddyState) -> dict:
    """Analyze the incoming query to determine handling strategy."""
    query = state["query"]

    analysis_prompt = f"""You are StudyBuddy, an AI engineering tutor. Analyze this student query:

Query: "{query}"

Your knowledge base covers: RAG, embeddings, vector databases, agents, LangChain,
LangGraph, tool use, memory systems, evaluation, chunking, and related AI engineering topics.

Determine:
1. query_type: "factual" | "conceptual" | "procedural" | "comparison"
2. complexity: "simple" | "moderate" | "complex"
3. needs_retrieval: true if this needs info from the knowledge base, false if
   it's general knowledge or conversational (like "hi" or "what is 2+2")
4. search_queries: if retrieval needed, list 1-3 effective search queries

Respond with JSON only, no markdown formatting."""

    response = llm.invoke(analysis_prompt)

    try:
        # Handle potential markdown code blocks in response
        content = response.content.strip()
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        analysis = json.loads(content)
    except json.JSONDecodeError:
        # Fallback for parsing errors
        analysis = {
            "query_type": "conceptual",
            "complexity": "moderate",
            "needs_retrieval": True,
            "search_queries": [query]
        }

    return {
        "query_type": analysis.get("query_type", "conceptual"),
        "complexity": analysis.get("complexity", "moderate"),
        "needs_retrieval": analysis.get("needs_retrieval", True),
        "search_queries": analysis.get("search_queries", [query]),
        "iteration": 0
    }


def retrieve_node(state: StudyBuddyState) -> dict:
    """Retrieve relevant documents with adaptive k based on complexity."""
    complexity = state.get("complexity", "moderate")
    search_queries = state.get("search_queries", [state["query"]])

    # Adaptive k based on complexity
    k_map = {"simple": 2, "moderate": 4, "complex": 6}
    k = k_map.get(complexity, 3)

    # Multi-query retrieval with deduplication
    all_docs = []
    seen = set()

    for query in search_queries:
        docs = vector_store.similarity_search(query, k=k)
        for doc in docs:
            content_hash = hash(doc.page_content[:200])
            if content_hash not in seen:
                seen.add(content_hash)
                all_docs.append(doc)

    return {"retrieved_docs": all_docs}


def evaluate_node(state: StudyBuddyState) -> dict:
    """Evaluate if retrieved documents are sufficient to answer the query."""
    query = state["query"]
    docs = state.get("retrieved_docs", [])
    iteration = state.get("iteration", 0)

    if not docs:
        return {
            "retrieval_sufficient": False,
            "confidence": 0.0,
            "iteration": iteration + 1
        }

    context_preview = "\n".join([doc.page_content[:300] for doc in docs[:5]])

    eval_prompt = f"""Evaluate if these documents can answer the query:

Query: {query}

Retrieved content preview:
{context_preview}

Rate confidence 0-1 that you can fully answer the query with this context.
Consider:
- Is the topic covered?
- Is there enough detail?
- Are there gaps that would need to be filled?

Respond with JSON only, no markdown: {{"confidence": X}}"""

    response = llm.invoke(eval_prompt)

    try:
        content = response.content.strip()
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        result = json.loads(content)
        confidence = result.get("confidence", 0.5)
    except:
        confidence = 0.5

    # Stop iterating after 2 attempts or if confidence is good
    sufficient = confidence >= 0.7 or iteration >= 2

    return {
        "retrieval_sufficient": sufficient,
        "confidence": confidence,
        "iteration": iteration + 1
    }


def generate_node(state: StudyBuddyState) -> dict:
    """Generate educational response with optional flashcard suggestion."""
    query = state["query"]
    docs = state.get("retrieved_docs", [])
    confidence = state.get("confidence", 0.5)
    query_type = state.get("query_type", "conceptual")
    card_context = state.get("card_context")

    # Format context with sources
    context_parts = []
    for doc in docs:
        source = doc.metadata.get("source", "Reference Guide")
        context_parts.append(f"[{source}]:\n{doc.page_content}")
    context = "\n\n---\n\n".join(context_parts)

    # Build card context section if present
    card_section = ""
    if card_context:
        card_section = f"""
The student is currently studying a flashcard:
- Topic: {card_context.get('topic', 'Unknown')}
- Question: {card_context.get('question', '')}
- Answer: {card_context.get('answer', '')}

They're asking for clarification about this card. Focus your explanation on helping them understand this specific concept better.

"""

    generate_prompt = f"""You are StudyBuddy, an AI engineering tutor helping students learn.
{card_section}
Student question: {query}

Reference material:
{context}

Instructions:
- Explain clearly and educationally
- Cite sources when using specific information (e.g., "According to the RAG Fundamentals guide...")
- Use examples to illustrate concepts
- If the context doesn't fully cover the question, acknowledge gaps honestly
- Match your explanation depth to the question complexity

After your explanation, if this covers an important concept worth remembering,
suggest a flashcard in this EXACT format:

FLASHCARD:
Front: [question testing the key concept]
Back: [concise answer]

Only suggest a flashcard for conceptual or factual questions with clear takeaways.
Skip for procedural how-to questions, comparisons, or conversational messages.
Skip if the student is asking about a flashcard they're already studying."""

    response = llm.invoke(generate_prompt)
    content = response.content

    # Extract flashcard if present
    flashcard = None
    if "FLASHCARD:" in content:
        parts = content.split("FLASHCARD:")
        main_response = parts[0].strip()
        flashcard_text = parts[1].strip()

        try:
            lines = flashcard_text.split("\n")
            front = next(l for l in lines if l.startswith("Front:")).replace("Front:", "").strip()
            back = next(l for l in lines if l.startswith("Back:")).replace("Back:", "").strip()
            flashcard = {"front": front, "back": back}
        except:
            flashcard = None
        content = main_response

    return {
        "response": content,
        "confidence": confidence,
        "flashcard_suggestion": flashcard
    }


def direct_answer_node(state: StudyBuddyState) -> dict:
    """Answer without retrieval for simple/conversational queries."""
    query = state["query"]

    response = llm.invoke(
        f"You are StudyBuddy, a friendly AI engineering tutor. Answer this briefly: {query}"
    )

    return {
        "response": response.content,
        "confidence": 0.9,
        "flashcard_suggestion": None,
        "retrieved_docs": []
    }


# ============== Routing Functions ==============

def route_after_analysis(state: StudyBuddyState) -> Literal["retrieve", "direct"]:
    """Route based on whether retrieval is needed."""
    if state.get("needs_retrieval", True):
        return "retrieve"
    return "direct"


def route_after_evaluation(state: StudyBuddyState) -> Literal["generate", "retrieve"]:
    """Route based on retrieval quality evaluation."""
    if state.get("retrieval_sufficient", False):
        return "generate"
    return "retrieve"


# ============== Graph Assembly ==============

def create_studybuddy_graph():
    """Create and compile the StudyBuddy v4 agent graph."""
    graph = StateGraph(StudyBuddyState)

    # Add nodes
    graph.add_node("analyze", analyze_query_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("evaluate", evaluate_node)
    graph.add_node("generate", generate_node)
    graph.add_node("direct", direct_answer_node)

    # Set entry point
    graph.set_entry_point("analyze")

    # Add edges
    graph.add_conditional_edges(
        "analyze",
        route_after_analysis,
        {"retrieve": "retrieve", "direct": "direct"}
    )

    graph.add_edge("retrieve", "evaluate")

    graph.add_conditional_edges(
        "evaluate",
        route_after_evaluation,
        {"generate": "generate", "retrieve": "retrieve"}
    )

    graph.add_edge("generate", END)
    graph.add_edge("direct", END)

    return graph.compile()


# Create the agent (will be initialized after indexing)
studybuddy = None


# ============== FastAPI Application ==============

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start background indexing and create agent on startup."""
    global studybuddy

    def init():
        global studybuddy
        index_reference_guides()
        studybuddy = create_studybuddy_graph()
        print("Agent ready!")

    thread = threading.Thread(target=init, daemon=True)
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
    chapter_id: Optional[int] = None
    scope: Optional[str] = None
    card_context: Optional[dict] = None


class ChatResponse(BaseModel):
    reply: str
    confidence: float
    flashcard: dict | None = None
    analysis: dict | None = None


class FlashcardRequest(BaseModel):
    chapter_id: int
    scope: str = "single"  # "single" or "cumulative"
    current_topic: Optional[str] = None


class FlashcardResponse(BaseModel):
    question: str
    answer: str
    topic: str
    source: str  # "rag" or "llm"


@app.get("/api/status")
def get_status():
    """Check indexing status."""
    return {
        "indexing_complete": indexing_status["done"],
        "documents_indexed": indexing_status["count"],
        "chunks_in_db": indexing_status["chunks"],
        "current_file": indexing_status["current_file"]
    }


@app.get("/api/chapters")
def get_chapters():
    """Get all chapters parsed from topic-list.md."""
    chapters = parse_topic_list()
    return {"chapters": chapters}


@app.post("/api/flashcard", response_model=FlashcardResponse)
def generate_flashcard(request: FlashcardRequest):
    """Generate a single flashcard scoped to selected chapters."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not configured")

    if not indexing_status["done"]:
        raise HTTPException(status_code=503, detail="Still indexing documents. Please wait.")

    # Get topics for the requested scope
    topics = get_topics_for_scope(request.chapter_id, request.scope)
    if not topics:
        raise HTTPException(status_code=400, detail=f"No topics found for chapter {request.chapter_id}")

    # Select a topic
    if request.current_topic:
        # "Study More" mode - find the matching topic
        matching = [t for t in topics if t["section_name"] == request.current_topic]
        if matching:
            selected_topic = matching[0]
        else:
            selected_topic = random.choice(topics)
    else:
        # Pick a random topic
        selected_topic = random.choice(topics)

    topic_name = selected_topic["section_name"]
    subtopics = selected_topic["subtopics"]

    # Build search queries from topic and subtopics
    search_queries = [topic_name]
    if subtopics:
        search_queries.extend(subtopics[:2])

    # Retrieve relevant documents
    all_docs = []
    seen = set()
    for query in search_queries:
        docs = vector_store.similarity_search(query, k=3)
        for doc in docs:
            content_hash = hash(doc.page_content[:200])
            if content_hash not in seen:
                seen.add(content_hash)
                all_docs.append(doc)

    # Determine source based on retrieval quality
    source = "rag" if len(all_docs) >= 2 else "llm"

    # Format context
    if all_docs:
        context = "\n\n---\n\n".join([doc.page_content for doc in all_docs[:5]])
    else:
        context = ""

    # Generate flashcard
    flashcard_prompt = f"""You are creating a study flashcard for the topic: {topic_name}

Subtopics to consider: {', '.join(subtopics) if subtopics else 'General concepts'}

Reference material:
{context if context else 'No specific reference material available - use your knowledge of AI engineering.'}

Create ONE high-quality flashcard that:
- Tests understanding of a key concept (not just recall)
- Has a clear, unambiguous question
- Has a concise but complete answer
- Avoids yes/no questions

Respond with JSON only:
{{"question": "...", "answer": "..."}}"""

    response = llm.invoke(flashcard_prompt)

    try:
        content = response.content.strip()
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        card = json.loads(content)
    except json.JSONDecodeError:
        # Fallback card
        card = {
            "question": f"What is a key concept in {topic_name}?",
            "answer": "Please try again - there was an error generating this card."
        }

    return FlashcardResponse(
        question=card["question"],
        answer=card["answer"],
        topic=topic_name,
        source=source
    )


@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """Handle chat requests through the LangGraph agent."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not configured")

    if studybuddy is None:
        raise HTTPException(status_code=503, detail="Agent still initializing. Please wait for indexing to complete.")

    try:
        # Build the initial state with optional card context
        initial_state = {
            "query": request.message,
            "messages": [],
            "card_context": request.card_context  # Pass through flashcard context if present
        }

        result = studybuddy.invoke(initial_state)

        return ChatResponse(
            reply=result.get("response", "I couldn't generate a response."),
            confidence=result.get("confidence", 0.0),
            flashcard=result.get("flashcard_suggestion"),
            analysis={
                "query_type": result.get("query_type"),
                "complexity": result.get("complexity"),
                "used_retrieval": result.get("needs_retrieval"),
                "docs_retrieved": len(result.get("retrieved_docs", [])),
                "iterations": result.get("iteration", 0)
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {str(e)}")


# Serve static files (local development only)
if not IS_VERCEL:
    frontend_path = Path(__file__).parent.parent / "frontend"
    if frontend_path.exists():
        app.mount("/", StaticFiles(directory=frontend_path, html=True), name="frontend")
