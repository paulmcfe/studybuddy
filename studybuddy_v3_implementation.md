# Building StudyBuddy v3

Time to put everything together. We're going to rebuild StudyBuddy as a proper agent using LangChain 1.0's create_agent API, Qdrant for vector storage, and real document indexing. This is where v2's basic RAG becomes v3's intelligent tutoring agent.

## Where We Left Off

StudyBuddy v2 had RAG capabilities but no document indexing. We embedded some sample text directly in the code and used a custom in-memory vector database. It could answer questions based on that hardcoded content, but you couldn't upload your own study materials. It also had no reasoning—it just retrieved and answered every time, regardless of whether retrieval was needed.

That was fine for learning RAG fundamentals, but it's not a real tutoring system. Students need to be able to work with their actual study materials—literature, historical documents, course readings, whatever they're studying. The system needs to reason about whether it should search before answering or whether it already knows enough. And it needs production-grade infrastructure that actually persists data.

## What We're Adding

StudyBuddy v3 adds document upload and indexing. Students can work with any text-based study materials—we'll chunk them, embed them, and store them in Qdrant. The agent can then search these materials when answering questions. This implementation uses the 12 Sherlock Holmes stories from "The Adventures of Sherlock Holmes" as the document set, demonstrating how the system works with literary texts. The same patterns work for any domain—textbooks, course notes, research papers, historical documents, whatever students need to study.

We're also making StudyBuddy a real agent using LangChain 1.0's create_agent. The agent will have tools for searching materials and retrieving additional context. It decides when to use these tools based on the question. If a student asks a simple question the agent already knows, it can answer directly. If they ask about specific study materials, the agent will search before answering.

We're switching to Qdrant for vector storage. We'll start with in-memory mode for simplicity, but the code will be ready to switch to persistent mode when we deploy in later chapters. And we're using LangChain's document loaders and text splitters to handle the indexing pipeline properly.

## Implementation Walkthrough

Let's build this step by step. First, we'll set up Qdrant and create our collection. Then we'll implement document processing and indexing. Next, we'll define tools for the agent. Finally, we'll create the agent itself, index all the stories from the documents directory, and wire everything together.

Setting up Qdrant with LangChain is straightforward. We'll use the in-memory client for now. Here's the code:

```python
from langchain_qdrant import QdrantVectorStore
from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

# Initialize embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Create in-memory Qdrant client
client = QdrantClient(":memory:")

# Collection name
COLLECTION_NAME = "study_materials"

# Create collection with proper dimensions (1536 for text-embedding-3-small)
client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
)

# Create LangChain vector store wrapper
vector_store = QdrantVectorStore(
    client=client,
    collection_name=COLLECTION_NAME,
    embedding=embeddings
)
```

This creates an in-memory Qdrant instance and wraps it in LangChain's QdrantVectorStore. The vector store handles embedding documents and searching for us. We're using cosine distance, which is standard for text embeddings.

Next, we need document processing and indexing. We'll use LangChain's document loaders and text splitters to handle this properly:

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader

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
```

The document loader handles extracting text from files. The text splitter chunks the content using recursive character splitting, which tries to split on natural boundaries like paragraphs and sentences. We add the document name to the metadata so we can track which chunks came from which source.

Now we'll define tools for the agent. We need a search tool that queries the vector store:

```python
from langchain.tools import tool

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
```

The tool decorator tells LangChain this function is a tool. The docstring is crucial—it gets passed to the LLM so it knows when and how to use this tool. We search for the top 3 most similar chunks and format them with source attribution so the agent knows where the information came from.

Now we can create the agent. We'll use LangChain 1.0's create_agent with a system prompt that directs the agent to search for questions about the study materials:

```python
from langchain.agents import create_agent

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
```

The system prompt is doing a lot of work here. It explicitly tells the agent when to use tools—for any question about the study materials. This ensures the agent searches the indexed documents rather than relying solely on its general knowledge.

Now let's index all the documents in the documents directory. We'll do this in a background thread so the server starts immediately:

```python
from pathlib import Path
import threading

# Track indexing status
indexing_status = {"done": False, "count": 0, "chunks": 0, "current_file": ""}

def index_all_documents():
    """Index all documents from the documents directory."""
    if indexing_status["done"]:
        return indexing_status["chunks"]

    documents_dir = Path("documents")
    story_files = sorted(documents_dir.glob("*.txt"))

    print("Indexing stories...")
    total_chunks = 0
    for filepath in story_files:
        doc_name = filepath.stem.replace("-", " ").replace("_", " ").title()
        indexing_status["current_file"] = doc_name
        num_chunks = index_document(str(filepath), doc_name)
        total_chunks += num_chunks
        indexing_status["chunks"] = total_chunks
        print(f"Indexed {num_chunks} chunks from {doc_name}")

    indexing_status["done"] = True
    indexing_status["count"] = len(story_files)
    return total_chunks

# Start indexing in background thread on server startup
thread = threading.Thread(target=index_all_documents, daemon=True)
thread.start()
```

The indexing runs in the background, updating the status dict as it goes. The frontend can poll the `/api/status` endpoint to show progress to the user.

Now let's look at how we invoke the agent and extract its reasoning:

```python
# Question that requires searching
response = agent.invoke({
    "messages": [{"role": "user", "content": "Who is Irene Adler?"}]
})

# The response contains a list of messages
messages = response["messages"]

# Extract reasoning from the message history
reasoning_parts = []
final_answer = ""

for msg in messages:
    msg_type = getattr(msg, 'type', None)

    # Check for tool calls (agent deciding to use a tool)
    if msg_type == 'ai' and hasattr(msg, 'tool_calls') and msg.tool_calls:
        for tool_call in msg.tool_calls:
            reasoning_parts.append(f"Action: {tool_call['name']}")
            reasoning_parts.append(f"Input: {tool_call['args']}")

    # Check for tool responses
    elif msg_type == 'tool':
        tool_name = getattr(msg, 'name', 'search_materials')
        reasoning_parts.append(f"Observation from {tool_name}:\n{msg.content[:500]}...")

    # The final AI message is the answer
    elif msg_type == 'ai' and msg.content:
        if not getattr(msg, 'tool_calls', []):
            final_answer = msg.content

print("Reasoning:", reasoning_parts)
print("Answer:", final_answer)
```

LangChain 1.0's create_agent uses native tool calling rather than text-based reasoning. The message history contains structured data about tool calls:

1. **AIMessage with tool_calls**: The agent decides to call a tool
2. **ToolMessage**: The result from executing the tool
3. **AIMessage with content**: The final answer

We parse this history to extract the agent's reasoning—what tools it called and what it observed—separate from the final answer. This lets us show users what the agent did behind the scenes.

## Exposing Agent Reasoning

One of the key features of v3 is transparency. Users can see what the agent is "thinking"—which tools it called, what queries it made, and what results it found. This is exposed through a "Show reasoning" toggle in the UI.

The API returns both the final answer and the reasoning trace:

```python
return {
    "reply": final_answer,
    "reasoning": "\n\n".join(reasoning_parts) if reasoning_parts else None
}
```

The frontend displays a collapsible section showing the agent's actions when reasoning is present. This helps users understand how the agent arrived at its answer and builds trust in the system.

## Testing Agentic Behavior

To really see the agent in action, try different types of questions:

**Questions that trigger search:**
- "Who is Irene Adler?" → Searches materials, finds info from "A Scandal in Bohemia"
- "What happened in The Red-Headed League?" → Searches for plot details
- "How does Holmes solve the speckled band case?" → Searches for story events

**Questions answered directly:**
- "What's 2 + 2?" → No search needed, answers from general knowledge
- "What's the capital of France?" → General knowledge, no tool call

Watch the reasoning traces. They'll show you exactly what the agent did at each step. The "Show reasoning" toggle reveals:
- **Action**: Which tool was called
- **Input**: The search query used
- **Observation**: What the tool returned

If the agent is making bad decisions, you can diagnose why from the traces. Maybe the system prompt needs adjustment. Maybe the tool description is unclear. The traces give you the visibility to iterate and improve.

## What's Next

StudyBuddy v3 is a real agentic tutoring system. It can reason about when to search, use tools intelligently, and adapt its behavior based on the question. Users can see the agent's reasoning process, building transparency and trust.

In Chapter 4, we'll rebuild this using LangGraph, which gives us complete control over the agent's state machine. We'll implement custom reasoning patterns, add reflection and confidence scoring, and set up observability with LangSmith. We'll also start building toward the flashcard generation feature that will become central to StudyBuddy in later chapters.

Great work. You just built a real agent. Not a chatbot pretending to be an agent. A system that reasons, uses tools, and adapts its behavior intelligently. This is the foundation everything else builds on. Let's keep going.
