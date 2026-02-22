"""Semantic response cache for StudyBuddy v14.

Caches LLM responses and matches new queries by embedding similarity
rather than exact string match. This means semantically similar questions
like "What is photosynthesis?" and "Explain photosynthesis" will hit
the same cache entry.

Uses OpenAI embeddings for query vectors and cosine similarity for matching.
"""

import math
import logging
from datetime import datetime, timedelta
from typing import Optional

from sqlalchemy.orm import Session
from sqlalchemy import or_

logger = logging.getLogger(__name__)

# Similarity threshold — queries must be this similar to use cached response
DEFAULT_SIMILARITY_THRESHOLD = 0.92
DEFAULT_TTL_HOURS = 24
MAX_CANDIDATES = 100


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class SemanticResponseCache:
    """Cache that uses embedding similarity to match similar queries.

    Instead of requiring exact query matches, this cache embeds queries
    and finds cached responses where the query embedding is sufficiently
    similar (above the configured threshold).
    """

    def __init__(
        self,
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
        ttl_hours: int = DEFAULT_TTL_HOURS,
    ):
        self.similarity_threshold = similarity_threshold
        self.ttl_hours = ttl_hours
        self._embeddings = None

    def _get_embeddings(self):
        """Lazy-load embeddings model."""
        if self._embeddings is None:
            from langchain_openai import OpenAIEmbeddings
            self._embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        return self._embeddings

    async def get(
        self,
        query: str,
        program_id: Optional[str],
        task_type: str,
        db: Session,
    ) -> Optional[str]:
        """Check cache for a semantically similar query.

        Args:
            query: The user's query text
            program_id: Learning program ID (can be None for global queries)
            task_type: Type of task ("tutoring", "flashcard_generation", etc.)
            db: Database session

        Returns:
            Cached response text if a similar query is found, None otherwise.
        """
        from ..database.models import SemanticCache

        embeddings = self._get_embeddings()
        query_embedding = await embeddings.aembed_query(query)

        # Fetch recent cache entries for this program + task
        candidates = (
            db.query(SemanticCache)
            .filter(
                SemanticCache.program_id == program_id,
                SemanticCache.task_type == task_type,
                or_(
                    SemanticCache.expires_at.is_(None),
                    SemanticCache.expires_at > datetime.utcnow(),
                ),
            )
            .order_by(SemanticCache.created_at.desc())
            .limit(MAX_CANDIDATES)
            .all()
        )

        best_match = None
        best_similarity = 0.0

        for entry in candidates:
            similarity = cosine_similarity(query_embedding, entry.query_embedding)
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = entry

        if best_match and best_similarity >= self.similarity_threshold:
            best_match.hit_count += 1
            db.commit()
            logger.info(
                f"Semantic cache hit: similarity={best_similarity:.3f} "
                f"task={task_type} program={program_id}"
            )
            return best_match.response_text

        logger.debug(
            f"Semantic cache miss: best_similarity={best_similarity:.3f} "
            f"threshold={self.similarity_threshold} task={task_type}"
        )
        return None

    async def put(
        self,
        query: str,
        response: str,
        program_id: Optional[str],
        task_type: str,
        model_used: str,
        db: Session,
    ):
        """Store a query-response pair in the cache.

        Args:
            query: The user's query text
            response: The LLM response to cache
            program_id: Learning program ID
            task_type: Type of task
            model_used: Which model generated this response
            db: Database session
        """
        from ..database.models import SemanticCache

        embeddings = self._get_embeddings()
        query_embedding = await embeddings.aembed_query(query)

        entry = SemanticCache(
            program_id=program_id,
            query_text=query,
            query_embedding=query_embedding,
            response_text=response,
            model_used=model_used,
            task_type=task_type,
            expires_at=datetime.utcnow() + timedelta(hours=self.ttl_hours),
        )
        db.add(entry)
        db.commit()

        logger.info(f"Cached response: task={task_type} program={program_id}")


# Singleton instance
semantic_cache = SemanticResponseCache()
