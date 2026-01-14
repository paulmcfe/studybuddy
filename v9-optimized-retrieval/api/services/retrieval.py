"""Advanced retrieval service for StudyBuddy v9.

Provides multiple retrieval strategies:
- HybridRetriever: Dense vectors + BM25 keyword matching with RRF fusion
- RerankedRetriever: Adds Cohere cross-encoder reranking for precision
- RAGFusionRetriever: Multi-query expansion with result fusion
- AdaptiveRetriever: Automatically selects strategy based on query complexity
"""

import os
import json
from typing import Callable

import cohere
from langchain_community.retrievers import BM25Retriever
from langchain_qdrant import QdrantVectorStore
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document


class HybridRetriever:
    """
    Combines dense vector search with BM25 keyword matching.
    Uses Reciprocal Rank Fusion to combine results.
    """

    def __init__(
        self,
        vector_store: QdrantVectorStore,
        documents: list[Document],
        dense_weight: float = 0.6,
    ):
        """
        Initialize hybrid retriever.

        Args:
            vector_store: Qdrant vector store for dense search
            documents: Original documents for BM25 (needs raw text)
            dense_weight: Weight for dense results (1-dense_weight for sparse)
        """
        self.vector_store = vector_store
        self.dense_weight = dense_weight
        self.bm25_retriever = BM25Retriever.from_documents(documents)

    def search(self, query: str, k: int = 5) -> list[Document]:
        """Perform hybrid search with RRF fusion."""
        # Dense search
        dense_results = self.vector_store.similarity_search_with_score(query, k=k * 2)

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
                # Handle both (doc, score) tuples and plain docs
                doc = item[0] if isinstance(item, tuple) else item
                doc_id = hash(doc.page_content)

                if doc_id not in scores:
                    scores[doc_id] = {"doc": doc, "score": 0}

                scores[doc_id]["score"] += weight * (1 / (k + rank + 1))

        sorted_results = sorted(
            scores.values(), key=lambda x: x["score"], reverse=True
        )

        return [item["doc"] for item in sorted_results]


class RerankedRetriever:
    """Wraps a base retriever with Cohere reranking."""

    def __init__(
        self,
        base_retriever: HybridRetriever,
        rerank_top_n: int = 5,
    ):
        """
        Initialize reranked retriever.

        Args:
            base_retriever: Retriever to get initial candidates
            rerank_top_n: Number of results to return after reranking
        """
        self.base_retriever = base_retriever
        self.rerank_top_n = rerank_top_n
        api_key = os.getenv("COHERE_API_KEY")
        if api_key and api_key != "your_cohere_api_key_here":
            self.cohere_client = cohere.Client(api_key=api_key)
        else:
            self.cohere_client = None

    def search(self, query: str, k: int = 5) -> list[Document]:
        """Search with base retriever, then rerank results."""
        # Get candidates from base retriever
        candidates = self.base_retriever.search(query, k=k * 4)

        if not candidates:
            return []

        # If no Cohere client, return base results
        if self.cohere_client is None:
            return candidates[:k]

        try:
            # Rerank with Cohere
            rerank_response = self.cohere_client.rerank(
                query=query,
                documents=[doc.page_content for doc in candidates],
                top_n=min(self.rerank_top_n, len(candidates)),
                model="rerank-english-v3.0",
            )

            # Return reranked documents
            reranked = []
            for result in rerank_response.results:
                doc = candidates[result.index]
                doc.metadata["rerank_score"] = result.relevance_score
                reranked.append(doc)

            return reranked[:k]

        except Exception as e:
            print(f"Cohere reranking failed: {e}")
            return candidates[:k]


class RAGFusionRetriever:
    """
    Generates multiple query variations and fuses results.
    Best for complex, multi-faceted questions.
    """

    def __init__(
        self,
        base_retriever: RerankedRetriever | HybridRetriever,
        llm: ChatOpenAI,
    ):
        """
        Initialize RAG-Fusion retriever.

        Args:
            base_retriever: Retriever to search with each query variation
            llm: LLM for generating query variations
        """
        self.base_retriever = base_retriever
        self.llm = llm

    def search(self, query: str, k: int = 5, num_queries: int = 3) -> list[Document]:
        """Generate query variations, search each, fuse results."""
        variations = self._generate_variations(query, num_queries)
        all_queries = [query] + variations

        # Search with each variation
        all_results = []
        for q in all_queries:
            results = self.base_retriever.search(q, k=k * 2)
            ranked = [(doc, 1.0 / (i + 1)) for i, doc in enumerate(results)]
            all_results.append(ranked)

        # Fuse with RRF
        fused = self._reciprocal_rank_fusion(all_results)
        return fused[:k]

    def _generate_variations(self, query: str, n: int) -> list[str]:
        """Use LLM to generate query variations."""
        prompt = f"""Generate {n} alternative search queries for:
"{query}"

Make each query approach the topic from a different angle.
Return ONLY a JSON array of strings, no markdown formatting."""

        try:
            response = self.llm.invoke(prompt)
            content = response.content.strip()

            # Strip markdown code blocks if present
            if content.startswith("```"):
                # Remove ```json or ``` at start and ``` at end
                lines = content.split("\n")
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                content = "\n".join(lines)

            return json.loads(content)
        except (json.JSONDecodeError, Exception) as e:
            print(f"Query variation generation failed: {e}")
            return []

    def _reciprocal_rank_fusion(
        self, result_lists: list[list], k: int = 60
    ) -> list[Document]:
        """Combine multiple ranked lists using RRF."""
        scores = {}

        for results in result_lists:
            for rank, (doc, _) in enumerate(results):
                doc_id = hash(doc.page_content)

                if doc_id not in scores:
                    scores[doc_id] = {"doc": doc, "score": 0}

                scores[doc_id]["score"] += 1 / (k + rank + 1)

        sorted_results = sorted(
            scores.values(), key=lambda x: x["score"], reverse=True
        )

        return [item["doc"] for item in sorted_results]


class AdaptiveRetriever:
    """
    Selects retrieval strategy based on query characteristics.

    - Simple queries (<=4 words): HybridRetriever (fast)
    - Moderate queries: RerankedRetriever (balanced)
    - Complex queries (comparison, relationship, etc.): RAGFusionRetriever (thorough)
    """

    def __init__(
        self,
        vector_store: QdrantVectorStore,
        documents: list[Document],
        llm: ChatOpenAI,
    ):
        """
        Initialize adaptive retriever with all strategies.

        Args:
            vector_store: Qdrant vector store for dense search
            documents: Original documents for BM25
            llm: LLM for RAG-Fusion query generation
        """
        self.hybrid = HybridRetriever(vector_store, documents)
        self.reranked = RerankedRetriever(self.hybrid)
        self.fusion = RAGFusionRetriever(self.reranked, llm)
        self.llm = llm

    def search(self, query: str, k: int = 5) -> list[Document]:
        """Adaptive search based on query complexity."""
        complexity = self._assess_complexity(query)

        if complexity == "simple":
            return self.hybrid.search(query, k)
        elif complexity == "moderate":
            return self.reranked.search(query, k)
        else:
            return self.fusion.search(query, k)

    def _assess_complexity(self, query: str) -> str:
        """Quick heuristic for query complexity."""
        words = query.split()

        if len(words) <= 4:
            return "simple"

        complex_indicators = [
            "compare",
            "difference",
            "relationship",
            "how does",
            "why does",
            "explain how",
            "and",
            "or",
            "versus",
            "vs",
            "between",
        ]

        query_lower = query.lower()
        if any(indicator in query_lower for indicator in complex_indicators):
            return "complex"

        return "moderate"

    def search_with_strategy(
        self, query: str, k: int = 5, strategy: str = "adaptive"
    ) -> list[Document]:
        """
        Search with a specific strategy.

        Args:
            query: Search query
            k: Number of results
            strategy: One of "hybrid", "reranked", "fusion", or "adaptive"
        """
        if strategy == "hybrid":
            return self.hybrid.search(query, k)
        elif strategy == "reranked":
            return self.reranked.search(query, k)
        elif strategy == "fusion":
            return self.fusion.search(query, k)
        else:
            return self.search(query, k)


class SimpleRetriever:
    """
    Basic vector-only retriever for comparison baselines.
    Uses only dense vector search without any enhancements.
    """

    def __init__(self, vector_store: QdrantVectorStore):
        """
        Initialize simple retriever.

        Args:
            vector_store: Qdrant vector store for dense search
        """
        self.vector_store = vector_store

    def search(self, query: str, k: int = 5) -> list[Document]:
        """Simple vector similarity search."""
        return self.vector_store.similarity_search(query, k=k)
