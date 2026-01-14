"""Content-addressed flashcard caching.

This module provides hash-based caching for generated flashcards.
Same topic + same source context = same hash = cache hit.

First user to study a topic pays the LLM generation cost.
Every subsequent request for the same content gets instant results.
"""

import hashlib
import uuid
import logging
from datetime import datetime
from sqlalchemy.orm import Session

from ..database.models import Flashcard

logger = logging.getLogger(__name__)


def compute_content_hash(topic: str, source_context: str) -> str:
    """Generate SHA-256 hash of topic + source context.

    This creates a unique identifier based on the input material.
    Same inputs will always produce the same hash.

    Args:
        topic: The topic name
        source_context: The RAG-retrieved context used for generation

    Returns:
        64-character hex string (SHA-256 hash)
    """
    # Normalize inputs for consistent hashing
    content = f"{topic.strip().lower()}|{source_context.strip()}"
    return hashlib.sha256(content.encode()).hexdigest()


def get_cached_flashcards(
    topic: str,
    source_context: str,
    db: Session,
) -> list[Flashcard] | None:
    """Check cache for existing flashcards.

    Args:
        topic: The topic the cards are about
        source_context: The RAG context used to generate cards
        db: Database session

    Returns:
        List of cached Flashcard objects, or None if cache miss
    """
    content_hash = compute_content_hash(topic, source_context)

    cards = db.query(Flashcard).filter_by(content_hash=content_hash).all()

    if cards:
        logger.info(f"Cache HIT for topic '{topic}' - {len(cards)} cards")
        return cards

    logger.info(f"Cache MISS for topic '{topic}'")
    return None


def cache_flashcards(
    topic: str,
    source_context: str,
    cards_data: list[dict],
    db: Session,
    chapter_id: int | None = None,
) -> list[Flashcard]:
    """Store generated flashcards in cache.

    Args:
        topic: The topic of the cards
        source_context: The RAG context used
        cards_data: List of card dictionaries with question/answer
        db: Database session
        chapter_id: Optional chapter identifier

    Returns:
        List of created Flashcard objects
    """
    content_hash = compute_content_hash(topic, source_context)

    flashcards = []
    for card in cards_data:
        flashcard = Flashcard(
            id=str(uuid.uuid4()),
            content_hash=content_hash,
            topic=topic,
            question=card.get("question", ""),
            answer=card.get("answer", ""),
            difficulty=card.get("difficulty", "intermediate"),
            chapter_id=chapter_id,
            quality_score=card.get("quality_score", 3),
            was_revised=card.get("was_revised", False),
            source_context_hash=hashlib.sha256(
                source_context[:500].encode()
            ).hexdigest(),
        )
        db.add(flashcard)
        flashcards.append(flashcard)

    db.commit()
    logger.info(f"Cached {len(flashcards)} cards for topic '{topic}'")

    return flashcards


def get_or_generate_flashcards(
    topic: str,
    source_context: str,
    generator_func,
    db: Session,
    chapter_id: int | None = None,
) -> tuple[list[dict], bool]:
    """Generate flashcards with caching - main entry point.

    Checks cache first, generates only on miss.

    Args:
        topic: Topic to generate cards for
        source_context: RAG context
        generator_func: Function that generates cards (LLM call)
        db: Database session
        chapter_id: Optional chapter ID

    Returns:
        Tuple of (list of flashcard dicts, cache_hit boolean)
    """
    # Check cache first
    cached = get_cached_flashcards(topic, source_context, db)
    if cached:
        # Convert Flashcard objects to dicts
        return (
            [
                {
                    "id": c.id,
                    "question": c.question,
                    "answer": c.answer,
                    "topic": c.topic,
                    "difficulty": c.difficulty,
                }
                for c in cached
            ],
            True,  # cache_hit
        )

    # Cache miss - generate with LLM
    cards_data = generator_func(topic, source_context)

    # Store in cache
    if cards_data:
        cached_cards = cache_flashcards(
            topic=topic,
            source_context=source_context,
            cards_data=cards_data,
            db=db,
            chapter_id=chapter_id,
        )
        # Return with IDs from cached objects
        return (
            [
                {
                    "id": c.id,
                    "question": c.question,
                    "answer": c.answer,
                    "topic": c.topic,
                    "difficulty": c.difficulty,
                }
                for c in cached_cards
            ],
            False,  # cache_hit
        )

    return (cards_data, False)


def get_cache_stats(db: Session) -> dict:
    """Get statistics about the flashcard cache.

    Args:
        db: Database session

    Returns:
        Dictionary with cache statistics
    """
    from sqlalchemy import func

    total_cards = db.query(Flashcard).count()
    unique_hashes = db.query(func.count(func.distinct(Flashcard.content_hash))).scalar()
    topics = db.query(func.count(func.distinct(Flashcard.topic))).scalar()

    return {
        "total_flashcards": total_cards,
        "unique_content_hashes": unique_hashes or 0,
        "topics_covered": topics or 0,
    }
