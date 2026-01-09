# Building StudyBuddy v4

Alright, let's put everything together. We're rebuilding StudyBuddy using LangGraph, implementing all the agentic RAG patterns we've discussed, and adding observability with LangSmith. This version is a significant step up in sophistication.

## What We're Building

StudyBuddy v4 transforms from a simple agent into a sophisticated learning assistant with intelligent retrieval. The knowledge base changes too—instead of Sherlock Holmes stories, we're using the AI Engineering reference guides. This makes StudyBuddy a tool that helps you learn the very concepts you're studying in this book. Meta, right?

The new features include LangGraph-based architecture with explicit state management, query analysis that determines complexity and retrieval strategy, dynamic retrieval that adapts based on query needs, evaluation and reflection that assesses answer quality, confidence scoring so the system knows what it doesn't know, flashcard generation for important concepts (the foundation for the multi-agent system in Chapter 5), and LangSmith observability for debugging and improvement.

## Starting Point

We're building on the concepts from v3 but changing the architecture significantly. The vector store and embedding setup remain similar, but the agent logic is completely new. Here's our foundation:

```python
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path
import os
import json
from typing import TypedDict, Annotated, Literal
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

# Load environment variables
load_dotenv()

# LangSmith setup
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_PROJECT"] = "studybuddy-v4"
```

## The Knowledge Base

StudyBuddy v4's knowledge base consists of the reference guides—markdown files covering RAG, agents, LangGraph, embeddings, tool use, memory systems, and more. These files should be placed in a `documents` folder:

```
documents/
├── ref-agentic-rag-pattern.md
├── ref-agents-and-agency.md
├── ref-chunking-strategies.md
├── ref-embeddings.md
├── ref-evaluation-metrics.md
├── ref-langgraph-quickref.md
├── ref-langsmith-quickref.md
├── ref-memory-systems.md
├── ref-rag-fundamentals.md
├── ref-reflection-pattern.md
├── ref-retrieval-patterns.md
├── ref-tool-use.md
└── ... (other ref-*.md files)
```

The indexing code loads and chunks these files:

```python
# Initialize embeddings and vector store
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

def index_reference_guides():
    """Index all reference guide markdown files."""
    documents_dir = Path("documents")
    guide_files = list(documents_dir.glob("ref-*.md"))
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len
    )
    
    total_chunks = 0
    for filepath in guide_files:
        loader = TextLoader(str(filepath))
        documents = loader.load()
        chunks = text_splitter.split_documents(documents)
        
        # Add source metadata
        for chunk in chunks:
            chunk.metadata["source"] = filepath.stem.replace("ref-", "").replace("-", " ").title()
        
        vector_store.add_documents(chunks)
        total_chunks += len(chunks)
    
    return total_chunks
```

## State Definition

Our agent state captures everything needed for agentic RAG:

```python
class StudyBuddyState(TypedDict):
    # Conversation
    messages: Annotated[list, add_messages]
    
    # Query analysis
    query: str
    query_type: str
    complexity: str
    needs_retrieval: bool
    search_queries: list[str]
    
    # Retrieval
    retrieved_docs: list
    retrieval_sufficient: bool
    iteration: int
    
    # Response
    response: str
    confidence: float
    
    # Flashcard (new feature!)
    flashcard_suggestion: dict  # {"front": ..., "back": ...} or None
```

## Node Implementations

Let's implement each node for StudyBuddy.

### Query Analysis Node

```python
llm = ChatOpenAI(model="gpt-5-nano")

def analyze_query_node(state: StudyBuddyState) -> StudyBuddyState:
    """Analyze the incoming query."""
    
    query = state["query"]
    
    analysis_prompt = f"""You are StudyBuddy, an AI engineering tutor. Analyze this student query:

Query: "{query}"

Your knowledge base covers: RAG, embeddings, vector databases, agents, LangChain,
LangGraph, tool use, memory systems, evaluation, and related AI engineering topics.

Determine:
1. query_type: "factual" | "conceptual" | "procedural" | "comparison"
2. complexity: "simple" | "moderate" | "complex"
3. needs_retrieval: true if this needs info from the knowledge base, false if
   it's general knowledge or conversational
4. search_queries: if retrieval needed, list 1-3 effective search queries

Respond with JSON only."""
    
    response = llm.invoke(analysis_prompt)
    
    try:
        analysis = json.loads(response.content)
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
```

### Retrieval Node

```python
def retrieve_node(state: StudyBuddyState) -> StudyBuddyState:
    """Retrieve relevant documents based on analysis."""
    
    complexity = state["complexity"]
    search_queries = state["search_queries"]
    
    # Adaptive k
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
```

### Evaluation Node

```python
def evaluate_node(state: StudyBuddyState) -> StudyBuddyState:
    """Evaluate retrieval quality."""
    
    query = state["query"]
    docs = state["retrieved_docs"]
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
Respond with JSON: {{"confidence": X}}"""
    
    response = llm.invoke(eval_prompt)
    
    try:
        result = json.loads(response.content)
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
```

### Generation Node

```python
def generate_node(state: StudyBuddyState) -> StudyBuddyState:
    """Generate response with flashcard suggestion."""
    
    query = state["query"]
    docs = state["retrieved_docs"]
    confidence = state.get("confidence", 0.5)
    query_type = state.get("query_type", "conceptual")
    
    # Format context with sources
    context_parts = []
    for doc in docs:
        source = doc.metadata.get("source", "Reference Guide")
        context_parts.append(f"[{source}]:\n{doc.page_content}")
    context = "\n\n---\n\n".join(context_parts)
    
    generate_prompt = f"""You are StudyBuddy, an AI engineering tutor.

Student question: {query}

Reference material:
{context}

Instructions:
- Explain clearly and educationally
- Cite sources when using specific information
- Use examples to illustrate concepts
- If the context doesn't fully cover the question, acknowledge gaps
- Match your explanation depth to the question complexity

After your explanation, if this covers an important concept worth remembering,
suggest a flashcard in this format:
FLASHCARD:
Front: [question testing the key concept]
Back: [concise answer]

Only suggest a flashcard for conceptual or factual questions with clear takeaways.
Skip for procedural how-to questions or comparisons."""
    
    response = llm.invoke(generate_prompt)
    content = response.content
    
    # Extract flashcard if present
    flashcard = None
    if "FLASHCARD:" in content:
        parts = content.split("FLASHCARD:")
        main_response = parts[0].strip()
        flashcard_text = parts[1].strip()
        
        # Parse flashcard
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
```

### Direct Answer Node

```python
def direct_answer_node(state: StudyBuddyState) -> StudyBuddyState:
    """Answer without retrieval for simple queries."""
    
    query = state["query"]
    
    response = llm.invoke(
        f"You are StudyBuddy, a friendly AI tutor. Answer this briefly: {query}"
    )
    
    return {
        "response": response.content,
        "confidence": 0.9,
        "flashcard_suggestion": None
    }
```

## Routing Functions

The routing logic determines flow through the graph:

```python
def route_after_analysis(state: StudyBuddyState) -> Literal["retrieve", "direct"]:
    """Route based on whether retrieval is needed."""
    if state.get("needs_retrieval", True):
        return "retrieve"
    return "direct"

def route_after_evaluation(state: StudyBuddyState) -> Literal["generate", "retrieve"]:
    """Route based on retrieval quality."""
    if state.get("retrieval_sufficient", False):
        return "generate"
    return "retrieve"
```

## Building the Graph

Now we assemble everything into the complete graph:

```python
def create_studybuddy_graph():
    """Create the StudyBuddy v4 agent graph."""
    
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

# Create the agent
studybuddy = create_studybuddy_graph()
```

## The API Layer

We expose StudyBuddy through a FastAPI application:

```python
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    reply: str
    confidence: float
    flashcard: dict | None = None
    analysis: dict | None = None

@app.on_event("startup")
async def startup():
    """Index documents on startup."""
    chunks = index_reference_guides()
    print(f"Indexed {chunks} chunks from reference guides")

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Handle chat requests."""
    
    try:
        # Run the agent
        result = studybuddy.invoke({
            "query": request.message,
            "messages": []
        })
        
        return ChatResponse(
            reply=result.get("response", "I couldn't generate a response."),
            confidence=result.get("confidence", 0.0),
            flashcard=result.get("flashcard_suggestion"),
            analysis={
                "query_type": result.get("query_type"),
                "complexity": result.get("complexity"),
                "used_retrieval": result.get("needs_retrieval"),
                "docs_retrieved": len(result.get("retrieved_docs", []))
            }
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

## Testing the System

Let's test StudyBuddy v4 with different types of questions.

**Simple factual query** (should use retrieval, low complexity):

```python
# "What is cosine similarity?"
#
# Expected behavior:
# - Query analysis: factual, simple, needs_retrieval=True
# - Retrieval: k=2, single search query
# - High confidence, quick response
# - Likely suggests a flashcard
```

**Complex conceptual query** (should use multi-query retrieval):

```python
# "Compare different chunking strategies for RAG systems"
#
# Expected behavior:
# - Query analysis: comparison, complex, needs_retrieval=True
# - Retrieval: k=6, multiple search queries
# - Multiple iterations possible if first retrieval insufficient
# - Detailed response citing multiple sources
```

**General knowledge query** (should skip retrieval):

```python
# "How do I write a for loop in Python?"
#
# Expected behavior:
# - Query analysis: procedural, simple, needs_retrieval=False
# - Routes directly to direct_answer_node
# - Quick response from general knowledge
# - No flashcard (basic programming)
```

## Viewing Traces in LangSmith

With `LANGSMITH_TRACING` enabled, every request generates a trace. Go to `smith.langchain.com` and find your "studybuddy-v4" project. You'll see traces for each request showing the analyze node execution and its output, routing decisions (retrieve vs direct), retrieval queries and results, evaluation scores, and generation inputs and outputs.

Click into any trace to see the full execution flow. Use the graph view to see which path through the state machine was taken. If something looks wrong, use time travel to step through and find where things went off track.

## What's Next

StudyBuddy v4 is a sophisticated agentic RAG system. It understands query complexity, adapts its retrieval strategy, evaluates its own results, and knows when it's uncertain. The flashcard suggestions lay the groundwork for the multi-agent system we'll build in Chapter 5.

In the next chapter, we'll expand StudyBuddy into a true multi-agent application. We'll have separate agents for tutoring, flashcard generation, quality checking, and scheduling—all coordinated by a supervisor. The single-agent pattern we built here will become one piece of a larger, more capable system.

Great work making it through this chapter. You now understand how to build agents with explicit state machine control, how to implement intelligent retrieval strategies, and how to observe and debug agent behavior. These skills will serve you well in everything that follows.

Let's keep building.
