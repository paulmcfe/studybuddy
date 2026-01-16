# Building StudyBuddy v2

Alright. Time to build. You've got the concepts. Now let's implement them.

In Chapter 1, you built StudyBuddy v1, a basic tutoring chatbot that could explain concepts but had no access to specific study materials. It was helpful, but limited. If a student asked about a topic not in the model's training data, v1 couldn't help. If they uploaded their textbook and asked questions about it, v1 had no way to reference that content.

StudyBuddy v2 fixes that. We're adding document upload and RAG so students can upload their study materials and get explanations grounded in those specific sources. Upload a chapter on mitosis, ask how it works, and get an answer that references the exact diagrams and explanations from that chapter.

## What We're Adding

StudyBuddy v2 adds three major capabilities: document upload, RAG retrieval, and source attribution. Students can upload PDFs or text files with their study materials. When they ask a question, StudyBuddy retrieves relevant sections from those materials and uses them to generate grounded explanations. The response includes references to which documents and sections the information came from.

Under the hood, we're building a RAG pipeline from scratch. No frameworks yet because we want you to see exactly how everything works. We'll create our own simple vector database in memory, implement chunking, call the OpenAI embeddings API, perform similarity search, and generate augmented prompts.

This is intentionally a learning exercise. In Chapter 3, we'll rebuild this using proper tools like LangChain and Qdrant. But for now, writing it yourself helps you understand what those tools are actually doing.

## Implementation Overview

The architecture has two main components: a document processing pipeline that handles uploads and indexing, and a query handler that performs retrieval and generation. When a document is uploaded, we extract the text, chunk it, generate embeddings, and store everything in memory. When a query comes in, we embed the query, search for similar chunks, and augment the prompt with retrieved context.

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

## Testing StudyBuddy v2

Let's test it. We'll set up StudyBuddy to automatically index some text embedded in the code:

```python
# Sample study material embedded directly in the code
STUDY_MATERIAL = """
Mitosis: Cell Division Explained

Mitosis is the process by which a single cell divides to produce two
identical daughter cells. It is essential for growth, repair, and
maintenance of living organisms. Mitosis ensures that each new cell
receives an exact copy of the parent cell's genetic material.

...

"""

def index_document(text: str, doc_name: str):
    chunks = chunk_text(text)
    for i, chunk in enumerate(chunks):
        embedding = embed_text(chunk)
        vector_db.add(embedding, chunk, {'doc_name': doc_name, 'chunk_id': i})

def index_study_material():
    """Index the built-in study material."""
    if vector_db.vectors:
        return len(vector_db.vectors)  # Already indexed
    index_document(STUDY_MATERIAL, "study-guide")
    return len(vector_db.vectors)
```

The system embeds the question, finds the chunks that mention prophase, retrieves them, and generates an answer grounded in that specific text. The response will reference the source material as follows:

```
Source: study-guide, Mitosis: Cell Division Explained.
```

This is RAG working end-to-end. Upload documents, chunk them, embed them, store them, then retrieve relevant chunks when users ask questions. Simple in concept, powerful in practice.

## What's Next

StudyBuddy v2 works, but it's basic. We built everything from scratch to understand the fundamentals. In Chapter 3, we'll rebuild this using proper tools. LangChain will handle document loading and chunking. Qdrant will replace our simple in-memory vector database. We'll add more sophisticated retrieval strategies and better error handling.

But the core pattern stays the same: index documents, embed queries, retrieve context, generate responses. You now understand how that pattern works at a fundamental level. Everything else is optimization and scale.

Great work getting through this chapter. RAG is foundational. You'll use this pattern in almost every AI application you build. Next up: making StudyBuddy truly agentic.
