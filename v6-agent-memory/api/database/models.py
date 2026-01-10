"""SQLAlchemy models for StudyBuddy v6 persistence."""

from datetime import datetime
from sqlalchemy import (
    Column,
    String,
    Integer,
    Float,
    DateTime,
    JSON,
    ForeignKey,
    Text,
    Boolean,
    Index,
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class User(Base):
    """User profile with preferences."""

    __tablename__ = "users"

    id = Column(String(50), primary_key=True)  # 'default' for dev
    created_at = Column(DateTime, default=datetime.utcnow)
    preferences = Column(JSON, default=dict)

    # Relationships
    card_reviews = relationship("CardReview", back_populates="user")
    memories = relationship("Memory", back_populates="user")


class Memory(Base):
    """User memory storage for learning preferences, struggles, goals.

    This is a custom SQLAlchemy-based memory store (no langmem dependency).
    Memories are organized by namespace for easy categorization:
    - preferences: Learning style, favorite topics, etc.
    - struggles: Topics the user finds difficult
    - goals: What the user is studying for
    - sessions: Summaries of past study sessions
    """

    __tablename__ = "memories"

    id = Column(String(36), primary_key=True)
    user_id = Column(String(50), ForeignKey("users.id"), nullable=False, index=True)
    namespace = Column(String(100), nullable=False, index=True)
    key = Column(String(200), nullable=False)
    value = Column(JSON, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    user = relationship("User", back_populates="memories")

    # Composite index for efficient lookups
    __table_args__ = (Index("ix_memory_user_namespace", "user_id", "namespace"),)


class Flashcard(Base):
    """Flashcard with content-addressed caching support.

    The content_hash is a SHA-256 hash of (topic + source_context).
    Same inputs produce the same hash, enabling cache hits.
    """

    __tablename__ = "flashcards"

    id = Column(String(36), primary_key=True)  # UUID
    content_hash = Column(String(64), unique=True, index=True)  # SHA-256

    # Card content
    topic = Column(String(200), nullable=False, index=True)
    question = Column(Text, nullable=False)
    answer = Column(Text, nullable=False)
    difficulty = Column(String(20), default="intermediate")

    # Cache metadata
    source_context_hash = Column(String(64))  # Hash of source material
    chapter_id = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Quality metadata (from quality checker agent)
    quality_score = Column(Integer, default=3)  # 1-5
    was_revised = Column(Boolean, default=False)

    # Relationships
    reviews = relationship("CardReview", back_populates="flashcard")

    # Index for efficient topic + chapter queries
    __table_args__ = (Index("ix_flashcard_topic_chapter", "topic", "chapter_id"),)


class CardReview(Base):
    """Learning history entry for spaced repetition (SM-2 algorithm).

    Tracks each user's progress on each flashcard:
    - ease_factor: How easy this card is for the user (min 1.3, default 2.5)
    - interval: Days until next review
    - repetition_number: Consecutive correct reviews
    - next_review: When this card is due again
    """

    __tablename__ = "card_reviews"

    id = Column(String(36), primary_key=True)  # UUID
    user_id = Column(String(50), ForeignKey("users.id"), nullable=False)
    flashcard_id = Column(String(36), ForeignKey("flashcards.id"), nullable=False)

    # Review data
    reviewed_at = Column(DateTime, default=datetime.utcnow)
    quality = Column(Integer, nullable=False)  # 0-5 SM-2 rating
    response_time_ms = Column(Integer, nullable=True)  # How long user took

    # SM-2 algorithm state
    ease_factor = Column(Float, default=2.5)
    interval = Column(Integer, default=1)  # Days until next review
    repetition_number = Column(Integer, default=0)  # Consecutive correct reviews
    next_review = Column(DateTime, nullable=False)

    # Relationships
    user = relationship("User", back_populates="card_reviews")
    flashcard = relationship("Flashcard", back_populates="reviews")

    # Indexes for efficient due card queries
    __table_args__ = (
        Index("ix_review_user_next", "user_id", "next_review"),
        Index("ix_review_user_flashcard", "user_id", "flashcard_id"),
    )
