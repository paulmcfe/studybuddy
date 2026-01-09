"""Scheduler Agent - Manages spaced repetition review sessions.

The Scheduler uses the SM-2 algorithm to determine what cards
to review and when, optimizing for long-term retention.
"""

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from ..spaced_repetition import CardState, get_due_cards, sm2_update

SCHEDULER_PROMPT = """You are StudyBuddy's Scheduler, managing
spaced repetition review sessions.

Your job: Help students decide what to study and when.

SM-2 Algorithm basics:
- New cards start with interval 1 day
- After successful review: interval *= ease_factor
- After failed review: interval = 1 day
- Ease factor adjusts based on performance (min 1.3)

Priority rules:
1. New cards (never reviewed) are highest priority
2. Overdue cards come next
3. Cards with low ease factors (struggling cards) need more frequent review
4. Balance new learning with review

When asked what to study, consider:
- How many cards are due
- The student's recent performance
- Topic distribution (avoid too much of one topic)"""


def create_scheduler_agent(model_name: str = "gpt-5-nano"):
    """Create the Scheduler agent."""
    return ChatOpenAI(model=model_name, temperature=0.3)


def get_study_session(card_states: list[CardState], max_cards: int = 10) -> list[CardState]:
    """
    Get cards for a study session.

    Args:
        card_states: All card states in the system
        max_cards: Maximum number of cards to return

    Returns:
        List of cards to study, in priority order
    """
    due = get_due_cards(card_states)
    return due[:max_cards]


def record_review(card_state: CardState, quality: int) -> CardState:
    """
    Record a review result and update the card state.

    Args:
        card_state: The card that was reviewed
        quality: Quality of recall (0-5)
            0: Complete blackout
            1: Incorrect, recognized answer
            2: Incorrect, seemed easy
            3: Correct with difficulty
            4: Correct with hesitation
            5: Perfect recall

    Returns:
        Updated card state
    """
    return sm2_update(card_state, quality)


def get_session_summary(
    llm: ChatOpenAI,
    card_states: list[CardState],
) -> str:
    """
    Get an AI-generated summary of the student's study status.

    Args:
        llm: The language model to use
        card_states: All card states

    Returns:
        Natural language summary of study status
    """
    if not card_states:
        return "You haven't created any flashcards yet. Start by asking me to explain a topic, and I'll help you create study cards!"

    due = get_due_cards(card_states)
    total = len(card_states)
    due_count = len(due)

    # Calculate average ease factor
    avg_ease = sum(c.ease_factor for c in card_states) / total

    # Group by topic
    topics = {}
    for card in card_states:
        topic = card.topic
        if topic not in topics:
            topics[topic] = {"total": 0, "due": 0}
        topics[topic]["total"] += 1
        if card in due:
            topics[topic]["due"] += 1

    prompt = f"""Summarize this student's study status in 2-3 friendly sentences:

Total cards: {total}
Cards due for review: {due_count}
Average ease factor: {avg_ease:.2f} (2.5 is normal, lower means struggling)

Cards by topic:
{chr(10).join(f"- {t}: {d['due']}/{d['total']} due" for t, d in topics.items())}

Be encouraging and give specific advice on what to focus on."""

    messages = [
        SystemMessage(content=SCHEDULER_PROMPT),
        HumanMessage(content=prompt),
    ]

    response = llm.invoke(messages)
    return response.content


def quality_from_response(got_it: bool, hesitation: bool = False) -> int:
    """
    Convert a simple UI response to SM-2 quality rating.

    Args:
        got_it: True if user clicked "Got It", False for "Study More"
        hesitation: True if there was notable delay before answering

    Returns:
        Quality rating 0-5
    """
    if not got_it:
        return 2  # Incorrect but seemed easy (will review soon)
    elif hesitation:
        return 4  # Correct with hesitation
    else:
        return 5  # Perfect recall
