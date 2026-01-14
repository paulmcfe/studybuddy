"""SM-2 Spaced Repetition Algorithm implementation.

The SM-2 algorithm (SuperMemo 2) is a classic spaced repetition algorithm
that optimizes learning by scheduling reviews based on performance.

Key concepts:
- Quality (0-5): User's self-assessment of how well they knew the answer
- Ease Factor: How easy the card is for this user (min 1.3, default 2.5)
- Interval: Days until next review
- Repetition Number: Consecutive successful reviews

Quality ratings:
- 0: Complete blackout, no recall
- 1: Incorrect, but remembered upon seeing answer
- 2: Incorrect, but answer seemed easy to recall
- 3: Correct with serious difficulty
- 4: Correct after hesitation
- 5: Perfect response, instant recall
"""

from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass
class SM2Result:
    """Result of SM-2 calculation."""

    new_interval: int  # Days until next review
    new_ease_factor: float  # Updated ease factor
    next_review: datetime  # Absolute next review datetime
    repetition_number: int  # Updated repetition count


def calculate_sm2(
    quality: int,
    repetition_number: int,
    ease_factor: float,
    interval: int,
) -> SM2Result:
    """Calculate next review using SM-2 algorithm.

    Args:
        quality: User's self-assessment (0-5)
        repetition_number: Number of consecutive correct reviews
        ease_factor: How easy the card is for this user (default 2.5)
        interval: Current interval in days

    Returns:
        SM2Result with updated values for next review
    """
    # Constrain quality to valid range
    quality = max(0, min(5, quality))

    if quality < 3:
        # Failed review - reset to beginning
        new_interval = 1
        new_repetition = 0
        # Decrease ease factor (card is harder than we thought)
        new_ease = max(1.3, ease_factor - 0.2)
    else:
        # Successful review - extend interval
        new_repetition = repetition_number + 1

        if new_repetition == 1:
            new_interval = 1  # First success: review tomorrow
        elif new_repetition == 2:
            new_interval = 6  # Second success: review in 6 days
        else:
            # Subsequent successes: multiply interval by ease factor
            new_interval = round(interval * ease_factor)

        # Update ease factor based on quality
        # Formula: EF' = EF + (0.1 - (5-q) * (0.08 + (5-q) * 0.02))
        new_ease = ease_factor + (
            0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02)
        )
        new_ease = max(1.3, new_ease)  # Minimum ease factor is 1.3

    next_review = datetime.utcnow() + timedelta(days=new_interval)

    return SM2Result(
        new_interval=new_interval,
        new_ease_factor=round(new_ease, 2),
        next_review=next_review,
        repetition_number=new_repetition,
    )


def quality_from_button(button: str) -> int:
    """Convert button choice to SM-2 quality rating.

    Three-button UI maps to SM-2 quality ratings:
    - "no": Failed recall, reset interval
    - "took_a_sec": Correct but struggled
    - "yes": Solid recall

    Args:
        button: "no", "took_a_sec", or "yes"

    Returns:
        SM-2 quality rating (1, 3, or 4)
    """
    mapping = {
        "no": 1,  # Failed recall, reset interval
        "took_a_sec": 3,  # Correct but struggled
        "yes": 4,  # Solid recall
    }
    return mapping.get(button, 3)  # Default to 3 if unknown


def get_priority_score(
    days_overdue: int,
    ease_factor: float,
    repetition_number: int,
) -> float:
    """Calculate priority score for card ordering.

    Higher scores should be reviewed first.

    Args:
        days_overdue: How many days past due (negative if not yet due)
        ease_factor: Card's ease factor
        repetition_number: How many times successfully reviewed

    Returns:
        Priority score (higher = more urgent)
    """
    # Base priority from overdue days
    overdue_priority = days_overdue * 2

    # Harder cards (lower ease) get slight priority boost
    difficulty_priority = (2.5 - ease_factor) * 5

    # New cards (low repetition) get slight priority
    newness_priority = max(0, 3 - repetition_number)

    return overdue_priority + difficulty_priority + newness_priority
