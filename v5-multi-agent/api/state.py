"""Shared state definitions for StudyBuddy v5 multi-agent system."""

from typing import TypedDict, Annotated, Literal
from langgraph.graph.message import add_messages


class StudyBuddyState(TypedDict):
    """Shared state for the multi-agent graph."""

    # Conversation history
    messages: Annotated[list, add_messages]

    # Current mode
    current_mode: Literal["learning", "practice"]

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
