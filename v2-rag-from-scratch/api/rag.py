import numpy as np
from typing import List
from pathlib import Path
from openai import OpenAI
import os


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
        if not self.vectors:
            return []

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


def chunk_text(text: str, chunk_size: int = 500,
               overlap: int = 50) -> List[str]:
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size - overlap):
        chunk = words[i:i + chunk_size]
        chunks.append(' '.join(chunk))

    return chunks


# Initialize OpenAI client and vector database
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
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


def index_documents_directory(directory_path: str = './documents'):
    """Index all .txt files in the specified directory."""
    doc_path = Path(directory_path)

    if not doc_path.exists():
        print(f"Directory {directory_path} not found. Creating it...")
        doc_path.mkdir(parents=True, exist_ok=True)
        print(f"Please add your study materials as .txt files to {directory_path}")
        return 0

    txt_files = list(doc_path.glob('*.txt'))

    if not txt_files:
        print(f"No .txt files found in {directory_path}")
        return 0

    print(f"Found {len(txt_files)} documents to index...")

    for txt_file in txt_files:
        print(f"Indexing {txt_file.name}...")
        with open(txt_file, 'r', encoding='utf-8') as f:
            text = f.read()

        # Use the filename (without extension) as the document name
        doc_name = txt_file.stem
        index_document(text, doc_name)

    print(f"Successfully indexed {len(txt_files)} documents!")
    return len(txt_files)


def retrieve_context(query: str, k: int = 3) -> str:
    if not vector_db.vectors:
        return ""

    query_embedding = embed_text(query)
    results = vector_db.search(query_embedding, k=k)

    context_parts = []
    for result in results:
        context_parts.append(f"[From {result['metadata']['doc_name']}]:\n{result['text']}")

    return '\n\n'.join(context_parts)


def answer_question(question: str) -> str:
    context = retrieve_context(question)

    if context:
        prompt = f'''You are StudyBuddy, a helpful tutoring assistant.

Given the following context from study materials, answer the student's
question. Be clear and thorough, and cite which document you're
referencing when relevant.

Context:
{context}

Question: {question}'''
    else:
        # Fall back to general knowledge if no documents indexed
        prompt = f'''You are StudyBuddy, a helpful tutoring assistant.

Answer the student's question clearly and thoroughly. Since no study
materials have been uploaded yet, use your general knowledge.

Question: {question}'''

    response = client.chat.completions.create(
        model='gpt-5-nano',
        messages=[{
            'role': 'user',
            'content': prompt
        }]
    )

    return response.choices[0].message.content
