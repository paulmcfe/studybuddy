"""Flashcard service for StudyBuddy v10.

Handles flashcard generation, caching, and spaced repetition scheduling.
"""

import hashlib
from datetime import datetime, timedelta
from typing import Optional

from sqlalchemy.orm import Session
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from api.database.models import Flashcard, LearningProgram


FLASHCARD_PROMPT = ChatPromptTemplate.from_template("""You are an expert educator creating flashcards following the minimum information principle.

Topic: {topic}

Context from the learning materials:
{context}

CRITICAL RULES for creating effective flashcards:
1. ONE ATOMIC FACT per card - never ask multiple questions or combine concepts
2. Be SPECIFIC - avoid vague questions like "What is X?" when you can ask about a specific behavior or property
3. Keep questions SHORT and direct (ideally under 15 words)
4. Keep answers BRIEF - one sentence or a few bullet points maximum
5. Test RECALL, not recognition - the learner should produce the answer from memory

BAD flashcard examples (DO NOT create cards like these):
- "What is a dictionary and how do you use it?" (combines multiple concepts)
- "Explain the differences between lists and tuples" (too broad, multiple facts)
- "What are the key features of Python?" (vague, multiple answers)

GOOD flashcard examples:
- Q: "What Python syntax checks if key 'name' exists in dict d?" A: "'name' in d"
- Q: "What does list.append(x) return?" A: "None (it modifies the list in place)"
- Q: "What error is raised when accessing a missing dictionary key with d[key]?" A: "KeyError"

Return ONLY a JSON object with "question" and "answer" fields.
No markdown formatting, just the JSON object.

{{"question": "Your specific, atomic question here", "answer": "Brief, direct answer"}}
""")


def compute_content_hash(topic: str, context: str) -> str:
    """Compute hash for flashcard deduplication."""
    content = f"{topic}:{context}"
    return hashlib.sha256(content.encode()).hexdigest()


async def generate_flashcard(
    topic: str,
    context: str,
    program_id: str,
    db: Session,
) -> Optional[Flashcard]:
    """Generate a flashcard for a topic using retrieved context.

    Checks cache first to avoid regenerating identical cards.
    """
    content_hash = compute_content_hash(topic, context)

    # Check cache
    existing = db.query(Flashcard).filter(
        Flashcard.program_id == program_id,
        Flashcard.content_hash == content_hash,
    ).first()

    if existing:
        return existing

    # Generate new flashcard
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    chain = FLASHCARD_PROMPT | llm

    try:
        result = await chain.ainvoke({
            "topic": topic,
            "context": context,
        })

        # Parse JSON response
        import json
        content = result.content.strip()

        # Handle markdown code blocks
        if content.startswith("```"):
            lines = content.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            content = "\n".join(lines)

        data = json.loads(content)

        # Create flashcard
        card = Flashcard(
            program_id=program_id,
            topic=topic,
            question=data["question"],
            answer=data["answer"],
            source_context=context[:2000],  # Truncate long context
            content_hash=content_hash,
            next_review=datetime.utcnow(),  # Due immediately
        )

        db.add(card)
        db.commit()
        db.refresh(card)

        return card

    except Exception as e:
        print(f"Flashcard generation failed: {e}")
        return None


def get_due_flashcards(
    db: Session,
    program_id: str,
    limit: int = 10,
) -> list[Flashcard]:
    """Get flashcards due for review."""
    now = datetime.utcnow()

    return db.query(Flashcard).filter(
        Flashcard.program_id == program_id,
        Flashcard.next_review <= now,
    ).order_by(
        Flashcard.next_review.asc()
    ).limit(limit).all()


def update_flashcard_sm2(
    card: Flashcard,
    quality: int,
    db: Session,
) -> Flashcard:
    """Update flashcard using SM-2 algorithm.

    Quality ratings:
    0 - Complete blackout
    1 - Incorrect, but recognized answer
    2 - Incorrect, but easy to recall
    3 - Correct with serious difficulty
    4 - Correct with some hesitation
    5 - Perfect response
    """
    # SM-2 algorithm
    if quality >= 3:  # Correct response
        if card.repetitions == 0:
            card.interval = 1
        elif card.repetitions == 1:
            card.interval = 6
        else:
            card.interval = int(card.interval * card.ease_factor)
        card.repetitions += 1
    else:  # Incorrect response
        card.repetitions = 0
        card.interval = 1

    # Update ease factor
    card.ease_factor = max(
        1.3,
        card.ease_factor + 0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02)
    )

    # Schedule next review
    card.next_review = datetime.utcnow() + timedelta(days=card.interval)

    db.commit()
    db.refresh(card)

    return card


def get_flashcard_stats(
    db: Session,
    program_id: str,
) -> dict:
    """Get flashcard statistics for a program."""
    total = db.query(Flashcard).filter(
        Flashcard.program_id == program_id
    ).count()

    due = db.query(Flashcard).filter(
        Flashcard.program_id == program_id,
        Flashcard.next_review <= datetime.utcnow(),
    ).count()

    # Cards with interval > 21 days are considered "mastered"
    mastered = db.query(Flashcard).filter(
        Flashcard.program_id == program_id,
        Flashcard.interval > 21,
    ).count()

    return {
        "total": total,
        "due": due,
        "mastered": mastered,
        "learning": total - mastered,
    }
