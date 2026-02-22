"""Database-backed embedding cache for StudyBuddy v14.

Wraps an embedding model to avoid re-embedding identical text chunks.
Uses SHA-256 content hashing for cache key lookup in the EmbeddingCache
database table.

This saves significant cost when documents are re-indexed or when
the same content appears across multiple programs.
"""

import hashlib
import logging
from typing import Optional

from langchain_openai import OpenAIEmbeddings

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "text-embedding-3-small"


class DatabaseBackedEmbeddings:
    """Wraps OpenAIEmbeddings with a database-backed cache.

    On embed_documents(), checks the database for cached embeddings
    before calling the underlying model. Only uncached texts are
    sent to the API, and their results are stored for future use.

    Query embeddings are not cached (they are typically unique).
    """

    def __init__(self, underlying: OpenAIEmbeddings, db_session_factory):
        self.underlying = underlying
        self._session_factory = db_session_factory

    @property
    def model(self):
        """Expose model name for compatibility with LangChain vector stores."""
        return self.underlying.model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed documents, using cache for previously seen texts."""
        from ..database.models import EmbeddingCache

        db = self._session_factory()
        try:
            results: list[Optional[list[float]]] = [None] * len(texts)
            uncached_indices: list[int] = []
            uncached_texts: list[str] = []

            # Check cache for each text
            for i, text in enumerate(texts):
                content_hash = hashlib.sha256(text.encode()).hexdigest()
                cached = (
                    db.query(EmbeddingCache)
                    .filter_by(content_hash=content_hash)
                    .first()
                )
                if cached:
                    results[i] = cached.embedding
                else:
                    uncached_indices.append(i)
                    uncached_texts.append(text)

            cache_hits = len(texts) - len(uncached_texts)
            if cache_hits > 0:
                logger.info(
                    f"Embedding cache: {cache_hits}/{len(texts)} hits, "
                    f"{len(uncached_texts)} to embed"
                )

            # Embed uncached texts in batch
            if uncached_texts:
                new_embeddings = self.underlying.embed_documents(uncached_texts)

                for idx, text, embedding in zip(
                    uncached_indices, uncached_texts, new_embeddings
                ):
                    content_hash = hashlib.sha256(text.encode()).hexdigest()
                    db.add(
                        EmbeddingCache(
                            content_hash=content_hash,
                            embedding=embedding,
                            model=self.underlying.model,
                        )
                    )
                    results[idx] = embedding

                db.commit()

            return results  # type: ignore[return-value]
        finally:
            db.close()

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query (no caching — queries are unique)."""
        return self.underlying.embed_query(text)

    async def aembed_query(self, text: str) -> list[float]:
        """Async query embedding (pass-through, no caching)."""
        return await self.underlying.aembed_query(text)

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        """Async document embedding with caching."""
        # For async, delegate to sync version (DB operations are sync)
        return self.embed_documents(texts)


def get_cached_embeddings(db_session_factory=None) -> DatabaseBackedEmbeddings:
    """Create a cached embeddings instance.

    Args:
        db_session_factory: Callable that returns a new DB session.
            Defaults to SessionLocal from the database module.
    """
    if db_session_factory is None:
        from ..database.connection import SessionLocal
        db_session_factory = SessionLocal

    underlying = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    return DatabaseBackedEmbeddings(underlying, db_session_factory)
