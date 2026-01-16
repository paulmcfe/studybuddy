# Building StudyBuddy v2

Alright. Time to build. You've got the concepts. Now let's implement them.

In Chapter 1, you built StudyBuddy v1, a basic tutoring chatbot that could explain concepts but had no access to specific study materials. It was helpful, but limited. If a student asked about a topic not in the model's training data, v1 couldn't help. If they uploaded their textbook and asked questions about it, v1 had no way to reference that content.

StudyBuddy v2 fixes that. We're adding RAG so students can get explanations grounded in specific study materials. Ask about mitosis, and get an answer that references the exact explanations from the indexed content.

## What We're Adding

StudyBuddy v2 adds RAG retrieval and source attribution. When a student asks a question, StudyBuddy retrieves relevant sections from indexed study materials and uses them to generate grounded explanations. The response includes references to which documents the information came from.

Under the hood, we're building a RAG pipeline from scratch. No frameworks yet because we want you to see exactly how everything works. We'll create our own simple vector database in memory, implement chunking, call the OpenAI embeddings API, perform similarity search, and generate augmented prompts.

This is intentionally a learning exercise. In Chapter 3, we'll rebuild this using proper tools like LangChain and Qdrant. But for now, writing it yourself helps you understand what those tools are actually doing.

## Architecture

Like v1, StudyBuddy v2 uses a hybrid architecture: **Next.js frontend** with a **Python/FastAPI backend**. The backend handles the RAG pipeline (chunking, embeddings, retrieval), while the frontend provides a React-based chat interface.

```
v2-rag-from-scratch/
├── api/
│   ├── index.py              # FastAPI + RAG pipeline
│   └── requirements.txt      # Python deps (includes numpy)
├── frontend/
│   ├── src/
│   │   ├── app/
│   │   │   ├── layout.tsx    # Root layout
│   │   │   ├── page.tsx      # Chat page with RAG status
│   │   │   └── globals.css   # Tailwind styles
│   │   └── components/
│   │       ├── Message.tsx       # Message with markdown
│   │       ├── MessageList.tsx   # Scrollable container
│   │       ├── MessageInput.tsx  # Input textarea
│   │       └── LoadingDots.tsx   # Loading animation
│   ├── next.config.ts        # API proxy config
│   └── package.json          # Node.js dependencies
├── .env                      # API keys
├── pyproject.toml
└── README.md
```

The key difference from v1 is the backend. While v1 just forwarded messages to OpenAI, v2's backend includes a complete RAG pipeline: chunking, embedding, storage, retrieval, and augmented generation.

## The RAG Pipeline

The backend has two main flows: indexing (processing documents) and querying (answering questions). When the server starts, it indexes the built-in study materials. When a question comes in, it retrieves relevant chunks and augments the prompt.

Let's walk through the code. First, the vector database:

```python
import numpy as np
from typing import List, Tuple

class SimpleVectorDB:
    def __init__(self):
        self.vectors = []
        self.texts = []
        self.metadata = []

    def add(self, vector: List[float], text: str, meta: dict):
        self.vectors.append(np.array(vector))
        self.texts.append(text)
        self.metadata.append(meta)

    def search(self, query_vector: List[float], k: int = 5):
        query = np.array(query_vector)
        similarities = []
        for vec in self.vectors:
            # Cosine similarity
            similarity = np.dot(query, vec) / (np.linalg.norm(query) * np.linalg.norm(vec))
            similarities.append(similarity)
        top_k = np.argsort(similarities)[-k:][::-1]
        results = []
        for idx in top_k:
            results.append({'text': self.texts[idx], 'metadata': self.metadata[idx],
                'score': float(similarities[idx])})
        return results
```

This is about as simple as a vector database can get. We store vectors, texts, and metadata in parallel lists. When searching, we compute cosine similarity between the query and every stored vector, then return the top k results.

Cosine similarity is computed as the dot product of normalized vectors. We use numpy for the math because it's fast. The `argsort` call finds the indices that would sort the similarities array, we take the last k to get the highest scores, then reverse to get descending order.

Now the chunking function:

```python
def chunk_text(text: str, chunk_size: int = 500,
        overlap: int = 50) -> List[str]:
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = words[i:i + chunk_size]
        chunks.append(' '.join(chunk))
    return chunks
```

Fixed-size chunking by word count. We split the text into words, then take chunks of `chunk_size` words with `overlap` words of overlap. The stride is `chunk_size - overlap`, so each chunk starts where the previous chunk was `overlap` words from finishing.

Embedding and indexing:

```python
from openai import OpenAI

client = OpenAI()
vector_db = SimpleVectorDB()

def embed_text(text: str) -> List[float]:
    response = client.embeddings.create(
        model='text-embedding-3-small',
        input=text
    )
    return response.data[0].embedding

def index_document(text: str, doc_name: str):
    chunks = chunk_text(text)
    for i, chunk in enumerate(chunks):
        embedding = embed_text(chunk)
        vector_db.add(embedding, chunk, {'doc_name': doc_name, 'chunk_id': i})
```

The `embed_text` function calls OpenAI's API to generate embeddings. We're using `text-embedding-3-small` which returns 1,536-dimensional vectors. The `index_document` function chunks the document, embeds each chunk, and stores it in our vector database along with metadata about which document it came from and which chunk number it is.

Query and retrieval:

```python
def retrieve_context(query: str, k: int = 3) -> str:
    query_embedding = embed_text(query)
    results = vector_db.search(query_embedding, k=k)
    context_parts = []
    for result in results:
        context_parts.append(f"[From {result['metadata']['doc_name']}]:\n{result['text']}")
    return '\n\n'.join(context_parts)

def answer_question(question: str) -> str:
    context = retrieve_context(question)
    prompt = f'''You are StudyBuddy, a helpful tutoring assistant.
Given the following context from study materials, answer the student's
question. Be clear and thorough, and cite which document you're
referencing when relevant.

Context:
{context}

Question: {question}'''

    response = client.chat.completions.create(
        model='gpt-5-nano',
        messages=[{
            'role': 'user',
            'content': prompt
        }]
    )
    return response.choices[0].message.content
```

The `retrieve_context` function embeds the query and searches the vector database. It formats the results with document attribution so we know where each chunk came from. The `answer_question` function retrieves context, builds a prompt that includes both the context and the question, and sends it to the LLM.

## Running StudyBuddy v2

v2 uses the same two-terminal setup as v1. The backend indexes study materials on startup, and the frontend polls for indexing status.

**Terminal 1 - Backend (from v2-rag-from-scratch/):**
```bash
uv run uvicorn api.index:app --reload --port 8000
```

**Terminal 2 - Frontend (from v2-rag-from-scratch/frontend/):**
```bash
npm run dev
```

Visit `http://localhost:3000`. You'll see "Indexing..." in the footer while the backend processes the study materials, then "RAG enabled" once it's ready.

The backend includes built-in study material about mitosis, photosynthesis, and the water cycle. When you ask a question, the system embeds it, finds relevant chunks, retrieves them, and generates an answer grounded in that specific text. The response will reference the source material.

## The Frontend: RAG Status

The v2 frontend adds one key feature over v1: RAG status polling. The page uses `useEffect` to poll the `/api/status` endpoint every 2 seconds:

```tsx
const [ragStatus, setRagStatus] = useState<{
  indexing_complete: boolean
  chunks_in_db: number
} | null>(null)

useEffect(() => {
  const checkStatus = async () => {
    try {
      const res = await fetch('/api/status')
      if (res.ok) {
        const data = await res.json()
        setRagStatus(data)
      }
    } catch {
      // Retry on failure - backend might not be ready yet
    }
  }

  checkStatus()
  const interval = setInterval(checkStatus, 2000)
  return () => clearInterval(interval)
}, [])
```

The footer displays the status:

```tsx
<div className="text-center text-xs text-gray-400 py-2">
  StudyBuddy v2 · {ragStatus?.indexing_complete
    ? <span className="text-green-600">RAG enabled</span>
    : <span className="text-amber-500">Indexing...</span>}
  {ragStatus && ragStatus.chunks_in_db > 0 && (
    <span> · {ragStatus.chunks_in_db} chunks</span>
  )}
</div>
```

This gives users feedback about whether the RAG pipeline is ready. The rest of the frontend is identical to v1: React components for messages, markdown rendering, and a chat input.

## Testing the RAG Pipeline

Try these questions to see RAG in action:

- "What happens during prophase?"
- "Explain how photosynthesis works"
- "What are the stages of the water cycle?"

The responses should reference the indexed study materials. This is RAG working end-to-end: embed the query, retrieve relevant chunks, augment the prompt with context, generate a grounded response.

## What's Next

StudyBuddy v2 works, but it's basic. We built everything from scratch to understand the fundamentals. In Chapter 3, we'll rebuild this using proper tools. LangChain will handle document loading and chunking. Qdrant will replace our simple in-memory vector database. We'll add more sophisticated retrieval strategies and better error handling.

But the core pattern stays the same: index documents, embed queries, retrieve context, generate responses. You now understand how that pattern works at a fundamental level. Everything else is optimization and scale.

Great work getting through this chapter. RAG is foundational. You'll use this pattern in almost every AI application you build. Next up: making StudyBuddy truly agentic.
