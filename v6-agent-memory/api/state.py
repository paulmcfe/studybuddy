"""Shared state definitions for StudyBuddy v6 multi-agent system.

Extended from v5 with:
- User identification
- Memory context
- Scheduling state
- Cache tracking
"""

from typing import TypedDict, Annotated, Literal
from datetime import datetime
from langgraph.graph.message import add_messages


class StudyBuddyState(TypedDict):
    """Shared state for the multi-agent graph."""

    # ============== From v5 ==============

    # Conversation history
    messages: Annotated[list, add_messages]

    # Current mode
    current_mode: Literal["learning", "practice", "review"]

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


def create_initial_state(
    query: str,
    user_id: str = "default",
    mode: Literal["learning", "practice", "review"] = "learning",
    card_context: dict | None = None,
) -> StudyBuddyState:
    """Create initial state for a new conversation.

    Args:
        query: The user's message
        user_id: User identifier (default for dev)
        mode: Starting mode
        card_context: Optional current flashcard context

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
    )
