"""Shared state definitions for StudyBuddy v9 multi-agent system.

Includes:
- Curriculum planning state
- Session management
- Checkpoint tracking
"""

from typing import TypedDict, Annotated, Literal
from datetime import datetime
from langgraph.graph.message import add_messages


class StudyBuddyState(TypedDict):
    """Shared state for the multi-agent graph."""

    # ============== From v5 ==============

    # Conversation history
    messages: Annotated[list, add_messages]

    # Current mode (v7: added "curriculum")
    current_mode: Literal["learning", "practice", "review", "curriculum"]

    # Query from user
    query: str

    # Flashcard pipeline
    pending_cards: list[dict]  # Cards awaiting quality check
    approved_cards: list[dict]  # Cards that passed quality check

    # Context
    current_topic: str
    card_context: dict | None  # Current flashcard being studied

    # Response
    response: str

    # Supervisor routing
    next_agent: str | None  # Which agent to route to next

    # ============== New for v6 ==============

    # User identification (hardcoded 'default' for dev)
    user_id: str

    # Memory context - retrieved at start of conversation
    user_memories: list[dict]  # Relevant memories for current context

    # Spaced repetition state
    due_cards: list[dict]  # Cards due for review
    study_stats: dict | None  # User's learning statistics

    # Current review (when in practice/review mode)
    current_review: dict | None  # Card being reviewed
    review_started_at: datetime | None

    # Cache metadata
    cache_hit: bool  # Whether flashcard generation used cache

    # ============== New for v7: Curriculum Planning ==============

    # Active curriculum context
    curriculum_id: str | None  # ID of active curriculum
    curriculum_goal: str | None  # The learning goal
    current_module: dict | None  # Current module being studied

    # Progress tracking
    curriculum_progress: dict | None  # {completed_modules, progress_pct, etc.}

    # Session management
    session_id: str | None  # Active study session ID
    session_context: dict | None  # Context loaded for returning students

    # Checkpoint state
    checkpoint_pending: bool  # Whether a checkpoint evaluation is pending
    checkpoint_result: dict | None  # Result of checkpoint evaluation


def create_initial_state(
    query: str,
    user_id: str = "default",
    mode: Literal["learning", "practice", "review", "curriculum"] = "learning",
    card_context: dict | None = None,
    curriculum_id: str | None = None,
) -> StudyBuddyState:
    """Create initial state for a new conversation.

    Args:
        query: The user's message
        user_id: User identifier (default for dev)
        mode: Starting mode
        card_context: Optional current flashcard context
        curriculum_id: Optional active curriculum ID

    Returns:
        Initialized StudyBuddyState
    """
    return StudyBuddyState(
        # v5 fields
        messages=[],
        current_mode=mode,
        query=query,
        pending_cards=[],
        approved_cards=[],
        current_topic="",
        card_context=card_context,
        response="",
        next_agent=None,
        # v6 additions
        user_id=user_id,
        user_memories=[],
        due_cards=[],
        study_stats=None,
        current_review=None,
        review_started_at=None,
        cache_hit=False,
        # v7 additions
        curriculum_id=curriculum_id,
        curriculum_goal=None,
        current_module=None,
        curriculum_progress=None,
        session_id=None,
        session_context=None,
        checkpoint_pending=False,
        checkpoint_result=None,
    )
