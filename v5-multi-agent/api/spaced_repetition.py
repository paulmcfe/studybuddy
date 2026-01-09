"""SM-2 Spaced Repetition Algorithm for StudyBuddy v5.

The SM-2 algorithm was developed by Piotr Wozniak and is the foundation
of most modern spaced repetition systems. It optimizes review intervals
based on how well you remember each card.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional
import uuid


@dataclass
class CardState:
    """State for a single flashcard in the spaced repetition system."""

    card_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    question: str = ""
    answer: str = ""
    topic: str = ""

    # SM-2 parameters
    ease_factor: float = 2.5  # Starts at 2.5, min 1.3
    interval: int = 1  # Days until next review
    repetitions: int = 0  # Successful reviews in a row

    # Timestamps
    last_reviewed: Optional[datetime] = None
    next_review: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.now)


def sm2_update(state: CardState, quality: int) -> CardState:
    """
    Update card state using the SM-2 algorithm.

    Args:
        state: Current card state
        quality: Response quality 0-5
            0: Complete blackout, no memory
            1: Incorrect, but recognized answer
            2: Incorrect, but answer seemed easy to recall
            3: Correct with serious difficulty
            4: Correct with some hesitation
            5: Perfect recall

    Returns:
        Updated card state
    """
    if quality < 3:
        # Failed: reset repetitions, review again tomorrow
        state.repetitions = 0
        state.interval = 1
    else:
        # Success: increase interval based on repetitions
        if state.repetitions == 0:
            state.interval = 1
        elif state.repetitions == 1:
            state.interval = 6
        else:
            state.interval = int(state.interval * state.ease_factor)

        state.repetitions += 1

    # Update ease factor (never below 1.3)
    # Formula: EF' = EF + (0.1 - (5 - q) * (0.08 + (5 - q) * 0.02))
    state.ease_factor = max(
        1.3, state.ease_factor + 0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02)
    )

    # Calculate next review date
    state.last_reviewed = datetime.now()
    state.next_review = state.last_reviewed + timedelta(days=state.interval)

    return state


def get_due_cards(card_states: list[CardState]) -> list[CardState]:
    """
    Get cards that are due for review, sorted by priority.

    Priority order:
    1. New cards (never reviewed)
    2. Overdue cards (past their next_review date)
    3. Cards with lower ease factors (struggling cards)
    """
    now = datetime.now()

    # Filter to due cards
    due = [c for c in card_states if c.next_review is None or c.next_review <= now]

    # Sort by priority
    def priority_key(card: CardState) -> tuple:
        if card.next_review is None:
            # New cards first (use min datetime)
            return (0, datetime.min, card.ease_factor)
        else:
            # Then by how overdue (more overdue = higher priority)
            overdue_days = (now - card.next_review).days
            return (1, -overdue_days, card.ease_factor)

    due.sort(key=priority_key)

    return due


def create_card_state(question: str, answer: str, topic: str) -> CardState:
    """Create a new card state for a flashcard."""
    return CardState(
        question=question,
        answer=answer,
        topic=topic,
    )


# In-memory storage for card states (resets on server restart)
_card_states: dict[str, CardState] = {}


def save_card_state(state: CardState) -> None:
    """Save a card state to memory."""
    _card_states[state.card_id] = state


def get_card_state(card_id: str) -> Optional[CardState]:
    """Get a card state by ID."""
    return _card_states.get(card_id)


def get_all_card_states() -> list[CardState]:
    """Get all card states."""
    return list(_card_states.values())


def clear_card_states() -> None:
    """Clear all card states (for testing)."""
    _card_states.clear()
