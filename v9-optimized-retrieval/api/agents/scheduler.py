"""Scheduler Agent - Manages spaced repetition and due cards.

The Scheduler is responsible for:
- Tracking which flashcards need review
- Recording review results and updating SM-2 state
- Providing study statistics
- Prioritizing cards based on urgency and difficulty
"""

import uuid
import json
from datetime import datetime
from collections import defaultdict
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from sqlalchemy.orm import Session
from sqlalchemy import func, or_

from ..database.models import CardReview, Flashcard, User
from ..services.spaced_repetition import calculate_sm2, get_priority_score
from ..services.memory_store import MemoryStore

SCHEDULER_PROMPT = """You are StudyBuddy's Scheduler, responsible for
optimizing the user's learning through spaced repetition.

You track which flashcards need review and prioritize them based on:
1. Cards that are overdue (next_review < now)
2. Cards the user struggles with (low ease factor)
3. Cards from topics the user is currently focusing on

When the user finishes a review, record their performance and calculate
when they should see the card again using spaced repetition.

Your capabilities:
- Get cards due for review
- Record review results
- Provide study statistics
- Identify struggle areas

Always be encouraging about progress while being honest about areas
that need more practice."""


def create_scheduler_agent(model_name: str = "gpt-4o-mini"):
    """Create the Scheduler agent."""
    return ChatOpenAI(model=model_name, temperature=0.3)


def get_due_cards(
    db: Session,
    user_id: str = "default",
    limit: int = 10,
    topic_filter: str | list[str] | None = None,
) -> list[dict]:
    """Get flashcards due for review.

    Args:
        db: Database session
        user_id: User identifier
        limit: Maximum cards to return
        topic_filter: Optional topic(s) to focus on - string or list of strings

    Returns:
        List of due flashcards with review metadata, ordered by priority
    """
    now = datetime.utcnow()

    # Get cards with reviews that are due
    query = (
        db.query(CardReview, Flashcard)
        .join(Flashcard, CardReview.flashcard_id == Flashcard.id)
        .filter(CardReview.user_id == user_id, CardReview.next_review <= now)
    )

    if topic_filter:
        if isinstance(topic_filter, list):
            # Multiple topics - use IN with case-insensitive matching
            conditions = [Flashcard.topic.ilike(f"%{t}%") for t in topic_filter]
            query = query.filter(or_(*conditions))
        else:
            query = query.filter(Flashcard.topic.ilike(f"%{topic_filter}%"))

    # Get all due cards to sort by priority
    results = []
    for review, card in query.all():
        overdue_days = (now - review.next_review).days
        priority = get_priority_score(
            days_overdue=overdue_days,
            ease_factor=review.ease_factor,
            repetition_number=review.repetition_number,
        )
        results.append(
            {
                "flashcard_id": card.id,
                "question": card.question,
                "answer": card.answer,
                "topic": card.topic,
                "difficulty": card.difficulty,
                "ease_factor": review.ease_factor,
                "interval": review.interval,
                "overdue_days": overdue_days,
                "repetition_number": review.repetition_number,
                "priority": priority,
            }
        )

    # Sort by priority (highest first) and limit
    results.sort(key=lambda x: x["priority"], reverse=True)
    return results[:limit]


def get_new_cards(
    db: Session,
    user_id: str = "default",
    limit: int = 5,
    topic_filter: str | list[str] | None = None,
) -> list[dict]:
    """Get cards the user hasn't seen yet.

    Args:
        db: Database session
        user_id: User identifier
        limit: Maximum cards to return
        topic_filter: Optional topic(s) to focus on - string or list of strings

    Returns:
        List of new flashcards
    """
    # Get flashcard IDs user has already reviewed
    reviewed_ids = (
        db.query(CardReview.flashcard_id).filter(CardReview.user_id == user_id).subquery()
    )

    query = db.query(Flashcard).filter(~Flashcard.id.in_(reviewed_ids))

    if topic_filter:
        if isinstance(topic_filter, list):
            # Multiple topics - use IN with case-insensitive matching
            conditions = [Flashcard.topic.ilike(f"%{t}%") for t in topic_filter]
            query = query.filter(or_(*conditions))
        else:
            query = query.filter(Flashcard.topic.ilike(f"%{topic_filter}%"))

    query = query.order_by(Flashcard.created_at.desc()).limit(limit)

    return [
        {
            "flashcard_id": card.id,
            "question": card.question,
            "answer": card.answer,
            "topic": card.topic,
            "difficulty": card.difficulty,
            "is_new": True,
        }
        for card in query.all()
    ]


def record_review(
    db: Session,
    user_id: str,
    flashcard_id: str,
    quality: int,
    response_time_ms: int | None = None,
) -> dict:
    """Record a flashcard review and update spaced repetition state.

    Args:
        db: Database session
        user_id: User identifier
        flashcard_id: The reviewed flashcard
        quality: SM-2 quality rating (0-5)
        response_time_ms: How long user took (optional)

    Returns:
        Updated review state with next review info
    """
    # Get existing review or create new
    review = (
        db.query(CardReview)
        .filter_by(user_id=user_id, flashcard_id=flashcard_id)
        .first()
    )

    if review:
        # Existing card - update with SM-2
        result = calculate_sm2(
            quality=quality,
            repetition_number=review.repetition_number,
            ease_factor=review.ease_factor,
            interval=review.interval,
        )

        review.quality = quality
        review.reviewed_at = datetime.utcnow()
        review.ease_factor = result.new_ease_factor
        review.interval = result.new_interval
        review.repetition_number = result.repetition_number
        review.next_review = result.next_review
        if response_time_ms:
            review.response_time_ms = response_time_ms
    else:
        # First time seeing this card
        result = calculate_sm2(
            quality=quality,
            repetition_number=0,
            ease_factor=2.5,
            interval=1,
        )

        review = CardReview(
            id=str(uuid.uuid4()),
            user_id=user_id,
            flashcard_id=flashcard_id,
            quality=quality,
            ease_factor=result.new_ease_factor,
            interval=result.new_interval,
            repetition_number=result.repetition_number,
            next_review=result.next_review,
            response_time_ms=response_time_ms,
        )
        db.add(review)

    db.commit()

    return {
        "flashcard_id": flashcard_id,
        "next_review": result.next_review.isoformat(),
        "interval_days": result.new_interval,
        "ease_factor": result.new_ease_factor,
        "repetition_number": result.repetition_number,
        "quality": quality,
    }


def get_study_stats(db: Session, user_id: str = "default") -> dict:
    """Get comprehensive study statistics.

    Args:
        db: Database session
        user_id: User identifier

    Returns:
        Statistics dictionary with performance data
    """
    now = datetime.utcnow()

    # Total reviews ever
    total_reviews = db.query(CardReview).filter_by(user_id=user_id).count()

    # Cards due now
    due_count = (
        db.query(CardReview)
        .filter(CardReview.user_id == user_id, CardReview.next_review <= now)
        .count()
    )

    # Reviews today
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    reviews_today = (
        db.query(CardReview)
        .filter(CardReview.user_id == user_id, CardReview.reviewed_at >= today_start)
        .count()
    )

    # Average ease factor (lower = harder cards overall)
    avg_ease = (
        db.query(func.avg(CardReview.ease_factor))
        .filter(CardReview.user_id == user_id)
        .scalar()
        or 2.5
    )

    # Topic breakdown
    topic_stats = defaultdict(lambda: {"total": 0, "correct": 0})
    reviews_with_cards = (
        db.query(CardReview, Flashcard)
        .join(Flashcard, CardReview.flashcard_id == Flashcard.id)
        .filter(CardReview.user_id == user_id)
        .all()
    )

    for review, card in reviews_with_cards:
        topic_stats[card.topic]["total"] += 1
        if review.quality >= 3:
            topic_stats[card.topic]["correct"] += 1

    # Identify struggle areas (< 60% accuracy with 3+ reviews)
    struggle_topics = [
        topic
        for topic, stats in topic_stats.items()
        if stats["total"] >= 3 and stats["correct"] / stats["total"] < 0.6
    ]

    # Persist struggle areas to memory for cross-mode integration
    # This allows tutoring to know what the user struggles with in practice
    memory_store = MemoryStore(db)
    for topic in struggle_topics:
        stats = topic_stats[topic]
        accuracy = stats["correct"] / stats["total"]
        memory_store.put(
            user_id=user_id,
            namespace="struggles",
            key=topic.lower().replace(" ", "_"),
            value={
                "topic": topic,
                "accuracy": round(accuracy, 2),
                "total_reviews": stats["total"],
                "correct_reviews": stats["correct"],
                "identified_at": datetime.utcnow().isoformat(),
            },
        )

    # Unique cards studied
    unique_cards = len(set(r.flashcard_id for r, _ in reviews_with_cards))

    return {
        "total_reviews": total_reviews,
        "cards_due_now": due_count,
        "reviews_today": reviews_today,
        "average_ease_factor": round(avg_ease, 2),
        "topic_performance": dict(topic_stats),
        "struggle_areas": struggle_topics,
        "unique_cards_studied": unique_cards,
    }


def get_study_recommendation(
    llm: ChatOpenAI,
    stats: dict,
    available_topics: list[str],
    user_request: str = "",
) -> dict:
    """Have the Scheduler agent recommend what to study next.

    Args:
        llm: The language model
        stats: Current user statistics from get_study_stats()
        available_topics: Topics with available cards
        user_request: What the user asked for (optional)

    Returns:
        Recommendation dictionary
    """
    context = f"""Current study statistics:
- Cards due for review: {stats['cards_due_now']}
- Reviews completed today: {stats['reviews_today']}
- Average ease factor: {stats['average_ease_factor']}
- Struggle areas: {', '.join(stats['struggle_areas']) or 'None identified'}
- Topics studied: {', '.join(stats['topic_performance'].keys()) or 'None yet'}

Available topics: {', '.join(available_topics) if available_topics else 'Various AI engineering topics'}

User request: {user_request or 'General study session'}

Decide what the user should study next. Consider:
1. If they have overdue cards, prioritize those
2. Balance review with learning new material
3. Focus extra attention on struggle areas
4. Honor user preferences if they want specific topics

Respond with JSON:
{{
    "action": "review_due|learn_new|mixed|specific_topic",
    "topic_focus": "topic name or null",
    "card_count": 5,
    "reasoning": "Brief explanation"
}}"""

    messages = [
        SystemMessage(content=SCHEDULER_PROMPT),
        HumanMessage(content=context),
    ]

    response = llm.invoke(messages)

    try:
        content = response.content.strip()
        # Handle markdown code blocks
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
            content = content.strip()
        return json.loads(content)
    except json.JSONDecodeError:
        # Fallback recommendation
        return {
            "action": "review_due" if stats["cards_due_now"] > 0 else "learn_new",
            "topic_focus": None,
            "card_count": 5,
            "reasoning": "Default recommendation based on due cards",
        }
