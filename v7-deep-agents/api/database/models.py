"""SQLAlchemy models for StudyBuddy v7 persistence.

v7 additions:
- Curriculum: Learning path with modules and prerequisites
- CurriculumProgress: Track progress through a curriculum
- StudySession: Individual study session tracking
"""

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
    curricula = relationship("Curriculum", back_populates="user")


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
    content_hash = Column(String(64), index=True)  # SHA-256 - NOT unique, multiple cards share same hash

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


# ============== v7: Curriculum Planning Models ==============


class Curriculum(Base):
    """A learning curriculum generated for a user goal.

    The curriculum contains:
    - A high-level learning goal
    - Structured modules with prerequisites
    - A learning path (ordered sequence of module IDs)
    - Time estimates
    """

    __tablename__ = "curricula"

    id = Column(String(36), primary_key=True)  # UUID
    user_id = Column(String(50), ForeignKey("users.id"), nullable=False, index=True)

    # Goal and metadata
    goal = Column(Text, nullable=False)
    estimated_duration_weeks = Column(Integer, default=8)
    weekly_hours = Column(Integer, default=10)

    # Curriculum structure (JSON with modules and learning_path)
    # Format: {"modules": [...], "learning_path": ["m1", "m2", ...]}
    curriculum_data = Column(JSON, nullable=False)

    # Status tracking
    status = Column(String(20), default="active")  # active, completed, paused
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    user = relationship("User", back_populates="curricula")
    progress = relationship(
        "CurriculumProgress", back_populates="curriculum", uselist=False
    )
    sessions = relationship("StudySession", back_populates="curriculum")


class CurriculumProgress(Base):
    """Track progress through a curriculum.

    Stores:
    - Current module being studied
    - Completed modules
    - Attempt counts for backtracking logic
    - Checkpoint evaluation results
    - Progress summary for context management
    """

    __tablename__ = "curriculum_progress"

    id = Column(String(36), primary_key=True)  # UUID
    curriculum_id = Column(String(36), ForeignKey("curricula.id"), nullable=False)
    user_id = Column(String(50), ForeignKey("users.id"), nullable=False)

    # Module progress
    current_module_id = Column(String(50), nullable=True)
    completed_modules = Column(JSON, default=list)  # List of module IDs
    module_attempts = Column(JSON, default=dict)  # {module_id: attempt_count}

    # Overall progress
    progress_percentage = Column(Float, default=0.0)
    started_at = Column(DateTime, default=datetime.utcnow)
    last_activity_at = Column(DateTime, default=datetime.utcnow)

    # Checkpoint evaluation history
    # Format: {module_id: {passed, confidence, feedback, evaluated_at}}
    checkpoint_results = Column(JSON, default=dict)

    # Context summary (for long sessions - context compression)
    progress_summary = Column(Text, nullable=True)

    # Relationships
    curriculum = relationship("Curriculum", back_populates="progress")

    __table_args__ = (
        Index("ix_progress_user_curriculum", "user_id", "curriculum_id"),
    )


class StudySession(Base):
    """Track individual study sessions within a curriculum.

    Used for:
    - Session duration tracking
    - Topics studied per session
    - Generating session summaries
    - Determining time since last session for resume logic
    """

    __tablename__ = "study_sessions"

    id = Column(String(36), primary_key=True)  # UUID
    curriculum_id = Column(String(36), ForeignKey("curricula.id"), nullable=True)
    user_id = Column(String(50), ForeignKey("users.id"), nullable=False, index=True)

    # Session timing
    started_at = Column(DateTime, default=datetime.utcnow)
    ended_at = Column(DateTime, nullable=True)
    duration_minutes = Column(Integer, nullable=True)

    # Session content
    topics_studied = Column(JSON, default=list)  # List of topic names
    cards_reviewed = Column(Integer, default=0)
    modules_worked_on = Column(JSON, default=list)  # List of module IDs

    # Session summary (generated by context summarization)
    session_summary = Column(Text, nullable=True)

    # Relationships
    curriculum = relationship("Curriculum", back_populates="sessions")
