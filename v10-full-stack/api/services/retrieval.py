"""Retrieval service for StudyBuddy v10.

Adapts v9's advanced retrieval strategies for program-scoped collections.
Each learning program gets its own Qdrant collection and retriever instance.
"""

import os
from typing import Optional

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from ..database.models import LearningProgram

# Lazy import for optional dependencies
_bm25_module = None
_cohere_module = None


def _get_bm25_retriever():
    """Lazy load BM25 retriever."""
    global _bm25_module
    if _bm25_module is None:
        try:
            from langchain_community.retrievers import BM25Retriever
            _bm25_module = BM25Retriever
        except ImportError:
            _bm25_module = False
    return _bm25_module if _bm25_module else None


def _get_cohere_client(api_key: str):
    """Lazy load Cohere client."""
    global _cohere_module
    if _cohere_module is None:
        try:
            import cohere
            _cohere_module = cohere
        except ImportError:
            _cohere_module = False

    if _cohere_module:
        return _cohere_module.Client(api_key=api_key)
    return None


# Qdrant configuration
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSION = 1536


def get_qdrant_client() -> QdrantClient:
    """Get a Qdrant client instance."""
    return QdrantClient(url=QDRANT_URL)


def ensure_collection_exists(collection_name: str):
    """Ensure a Qdrant collection exists, creating if needed."""
    client = get_qdrant_client()

    collections = client.get_collections().collections
    exists = any(c.name == collection_name for c in collections)

    if not exists:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=EMBEDDING_DIMENSION,
                distance=Distance.COSINE,
            ),
        )
        print(f"Created Qdrant collection: {collection_name}")

    return client


def get_program_vector_store(program: LearningProgram) -> QdrantVectorStore:
    """Get a vector store for a program's collection."""
    ensure_collection_exists(program.qdrant_collection)

    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

    return QdrantVectorStore(
        client=get_qdrant_client(),
        collection_name=program.qdrant_collection,
        embedding=embeddings,
    )


def get_program_retriever(
    program: LearningProgram,
    documents: Optional[list[Document]] = None,
) -> "SimpleRetriever":
    """Create a retriever for a program's knowledge base.

    If documents are provided, creates a hybrid retriever with BM25.
    Otherwise, creates a simple vector-only retriever.
    """
    vector_store = get_program_vector_store(program)

    # If we have documents and BM25 is available, use hybrid
    BM25Retriever = _get_bm25_retriever()
    if documents and BM25Retriever:
        return HybridRetriever(vector_store, documents)

    return SimpleRetriever(vector_store)


class SimpleRetriever:
    """Basic vector-only retriever."""

    def __init__(self, vector_store: QdrantVectorStore):
        self.vector_store = vector_store

    def search(self, query: str, k: int = 5) -> list[Document]:
        """Simple vector similarity search."""
        return self.vector_store.similarity_search(query, k=k)

    async def ainvoke(self, query: str, k: int = 5) -> list[Document]:
        """Async interface for LangChain compatibility."""
        return self.search(query, k)


class HybridRetriever:
    """Combines dense vector search with BM25 keyword matching."""

    def __init__(
        self,
        vector_store: QdrantVectorStore,
        documents: list[Document],
        dense_weight: float = 0.6,
    ):
        self.vector_store = vector_store
        self.dense_weight = dense_weight

        BM25Retriever = _get_bm25_retriever()
        if BM25Retriever:
            self.bm25_retriever = BM25Retriever.from_documents(documents)
        else:
            self.bm25_retriever = None

    def search(self, query: str, k: int = 5) -> list[Document]:
        """Perform hybrid search with RRF fusion."""
        # Dense search
        dense_results = self.vector_store.similarity_search_with_score(query, k=k * 2)

        if not self.bm25_retriever:
            return [doc for doc, _ in dense_results[:k]]

        # Sparse search
        self.bm25_retriever.k = k * 2
        sparse_docs = self.bm25_retriever.invoke(query)
        sparse_results = [(doc, 1.0 / (i + 1)) for i, doc in enumerate(sparse_docs)]

        # RRF fusion
        combined = self._reciprocal_rank_fusion(
            [dense_results, sparse_results],
            weights=[self.dense_weight, 1 - self.dense_weight],
        )

        return combined[:k]

    def _reciprocal_rank_fusion(
        self,
        result_lists: list[list],
        weights: list[float],
        k: int = 60,
    ) -> list[Document]:
        """Combine ranked lists using weighted RRF."""
        scores = {}

        for results, weight in zip(result_lists, weights):
            for rank, item in enumerate(results):
                doc = item[0] if isinstance(item, tuple) else item
                doc_id = hash(doc.page_content)

                if doc_id not in scores:
                    scores[doc_id] = {"doc": doc, "score": 0}

                scores[doc_id]["score"] += weight * (1 / (k + rank + 1))

        sorted_results = sorted(
            scores.values(), key=lambda x: x["score"], reverse=True
        )

        return [item["doc"] for item in sorted_results]

    async def ainvoke(self, query: str, k: int = 5) -> list[Document]:
        """Async interface for LangChain compatibility."""
        return self.search(query, k)


class RerankedRetriever:
    """Wraps a base retriever with Cohere reranking."""

    def __init__(
        self,
        base_retriever: HybridRetriever | SimpleRetriever,
        rerank_top_n: int = 5,
    ):
        self.base_retriever = base_retriever
        self.rerank_top_n = rerank_top_n
        api_key = os.getenv("COHERE_API_KEY")
        if api_key and api_key != "your_cohere_api_key_here":
            self.cohere_client = _get_cohere_client(api_key)
        else:
            self.cohere_client = None

    def search(self, query: str, k: int = 5) -> list[Document]:
        """Search with base retriever, then rerank results."""
        candidates = self.base_retriever.search(query, k=k * 4)

        if not candidates or not self.cohere_client:
            return candidates[:k]

        try:
            rerank_response = self.cohere_client.rerank(
                query=query,
                documents=[doc.page_content for doc in candidates],
                top_n=min(self.rerank_top_n, len(candidates)),
                model="rerank-english-v3.0",
            )

            reranked = []
            for result in rerank_response.results:
                doc = candidates[result.index]
                doc.metadata["rerank_score"] = result.relevance_score
                reranked.append(doc)

            return reranked[:k]

        except Exception as e:
            print(f"Cohere reranking failed: {e}")
            return candidates[:k]

    async def ainvoke(self, query: str, k: int = 5) -> list[Document]:
        """Async interface for LangChain compatibility."""
        return self.search(query, k)
