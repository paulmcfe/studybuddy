"""Curriculum persistence and progress tracking service.

Manages curriculum lifecycle:
- Creating and storing curricula
- Tracking progress through modules
- Managing study sessions
- Calculating performance metrics for checkpoints
"""

import uuid
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import func

from ..database.models import (
    Curriculum,
    CurriculumProgress,
    StudySession,
    CardReview,
    Flashcard,
)


def create_curriculum(
    db: Session,
    user_id: str,
    goal: str,
    curriculum_data: dict,
    weekly_hours: int = 10,
) -> Curriculum:
    """Create and store a new curriculum with initial progress record.

    Args:
        db: Database session
        user_id: User identifier
        goal: The learning goal
        curriculum_data: Dict with modules and learning_path
        weekly_hours: Weekly time commitment

    Returns:
        Created Curriculum object
    """
    curriculum_id = str(uuid.uuid4())

    # Calculate estimated duration from module hours
    total_hours = sum(
        m.get("estimated_hours", 10) for m in curriculum_data.get("modules", [])
    )
    estimated_weeks = max(1, int(total_hours / weekly_hours) + 1)

    curriculum = Curriculum(
        id=curriculum_id,
        user_id=user_id,
        goal=goal,
        estimated_duration_weeks=estimated_weeks,
        weekly_hours=weekly_hours,
        curriculum_data=curriculum_data,
        status="active",
    )
    db.add(curriculum)

    # Create initial progress record
    first_module_id = (
        curriculum_data.get("learning_path", [None])[0]
        if curriculum_data.get("learning_path")
        else curriculum_data.get("modules", [{}])[0].get("id")
    )

    progress = CurriculumProgress(
        id=str(uuid.uuid4()),
        curriculum_id=curriculum_id,
        user_id=user_id,
        current_module_id=first_module_id,
        completed_modules=[],
        module_attempts={},
        progress_percentage=0.0,
        checkpoint_results={},
    )
    db.add(progress)

    db.commit()
    db.refresh(curriculum)

    return curriculum


def get_user_curricula(db: Session, user_id: str) -> list[dict]:
    """Get all curricula for a user with progress summaries.

    Args:
        db: Database session
        user_id: User identifier

    Returns:
        List of curriculum dicts with progress info
    """
    curricula = db.query(Curriculum).filter_by(user_id=user_id).all()

    results = []
    for c in curricula:
        progress = (
            db.query(CurriculumProgress)
            .filter_by(curriculum_id=c.id, user_id=user_id)
            .first()
        )

        results.append(
            {
                "id": c.id,
                "goal": c.goal,
                "status": c.status,
                "estimated_duration_weeks": c.estimated_duration_weeks,
                "created_at": c.created_at.isoformat(),
                "progress_percentage": progress.progress_percentage if progress else 0,
                "current_module_id": progress.current_module_id if progress else None,
                "completed_modules": progress.completed_modules if progress else [],
            }
        )

    return results


def get_curriculum_by_id(db: Session, curriculum_id: str) -> Curriculum | None:
    """Get a specific curriculum by ID.

    Args:
        db: Database session
        curriculum_id: Curriculum identifier

    Returns:
        Curriculum object or None
    """
    return db.query(Curriculum).filter_by(id=curriculum_id).first()


def get_curriculum_progress(
    db: Session, curriculum_id: str, user_id: str
) -> CurriculumProgress | None:
    """Get progress record for a curriculum.

    Args:
        db: Database session
        curriculum_id: Curriculum identifier
        user_id: User identifier

    Returns:
        CurriculumProgress object or None
    """
    return (
        db.query(CurriculumProgress)
        .filter_by(curriculum_id=curriculum_id, user_id=user_id)
        .first()
    )


def update_curriculum_progress(
    db: Session,
    curriculum_id: str,
    user_id: str,
    current_module_id: str | None = None,
    completed_module_id: str | None = None,
    checkpoint_result: dict | None = None,
    progress_summary: str | None = None,
) -> CurriculumProgress:
    """Update progress on a curriculum.

    Args:
        db: Database session
        curriculum_id: Curriculum identifier
        user_id: User identifier
        current_module_id: Set new current module
        completed_module_id: Mark a module as completed
        checkpoint_result: Checkpoint evaluation result for a module
        progress_summary: Updated context summary

    Returns:
        Updated CurriculumProgress object
    """
    progress = get_curriculum_progress(db, curriculum_id, user_id)

    if not progress:
        # Create progress if it doesn't exist
        progress = CurriculumProgress(
            id=str(uuid.uuid4()),
            curriculum_id=curriculum_id,
            user_id=user_id,
        )
        db.add(progress)

    if current_module_id is not None:
        progress.current_module_id = current_module_id

    if completed_module_id is not None:
        completed = list(progress.completed_modules or [])
        if completed_module_id not in completed:
            completed.append(completed_module_id)
        progress.completed_modules = completed

        # Update progress percentage
        curriculum = get_curriculum_by_id(db, curriculum_id)
        if curriculum:
            total_modules = len(curriculum.curriculum_data.get("modules", []))
            if total_modules > 0:
                progress.progress_percentage = len(completed) / total_modules * 100

    if checkpoint_result is not None:
        results = dict(progress.checkpoint_results or {})
        module_id = checkpoint_result.get("module_id")
        if module_id:
            results[module_id] = {
                "passed": checkpoint_result.get("passed"),
                "confidence": checkpoint_result.get("confidence"),
                "feedback": checkpoint_result.get("feedback"),
                "evaluated_at": datetime.utcnow().isoformat(),
            }
        progress.checkpoint_results = results

    if progress_summary is not None:
        progress.progress_summary = progress_summary

    progress.last_activity_at = datetime.utcnow()

    db.commit()
    db.refresh(progress)

    return progress


def increment_module_attempts(
    db: Session, curriculum_id: str, user_id: str, module_id: str
) -> int:
    """Increment attempt count for a module (for backtracking logic).

    Args:
        db: Database session
        curriculum_id: Curriculum identifier
        user_id: User identifier
        module_id: Module identifier

    Returns:
        New attempt count
    """
    progress = get_curriculum_progress(db, curriculum_id, user_id)
    if not progress:
        return 1

    attempts = dict(progress.module_attempts or {})
    current_attempts = attempts.get(module_id, 0)
    attempts[module_id] = current_attempts + 1
    progress.module_attempts = attempts

    db.commit()

    return attempts[module_id]


def get_module_performance(
    db: Session,
    user_id: str,
    module_topics: list[str],
) -> dict:
    """Get performance metrics for topics in a module.

    Used for checkpoint evaluation to determine if student has mastered
    the module content.

    Args:
        db: Database session
        user_id: User identifier
        module_topics: List of topic names in the module

    Returns:
        Dict with accuracy, ease_factor, and cards_reviewed
    """
    if not module_topics:
        return {
            "accuracy": 0,
            "average_ease_factor": 2.5,
            "cards_reviewed": 0,
            "total_reviews": 0,
        }

    # Get flashcards for these topics
    flashcard_ids = (
        db.query(Flashcard.id).filter(Flashcard.topic.in_(module_topics)).all()
    )
    flashcard_ids = [f[0] for f in flashcard_ids]

    if not flashcard_ids:
        return {
            "accuracy": 0,
            "average_ease_factor": 2.5,
            "cards_reviewed": 0,
            "total_reviews": 0,
        }

    # Get reviews for these flashcards
    reviews = (
        db.query(CardReview)
        .filter(
            CardReview.user_id == user_id, CardReview.flashcard_id.in_(flashcard_ids)
        )
        .all()
    )

    if not reviews:
        return {
            "accuracy": 0,
            "average_ease_factor": 2.5,
            "cards_reviewed": 0,
            "total_reviews": 0,
        }

    # Calculate metrics
    total_reviews = len(reviews)
    correct_reviews = sum(1 for r in reviews if r.quality >= 3)  # 3+ is passing
    accuracy = correct_reviews / total_reviews if total_reviews > 0 else 0

    # Get latest ease factors for unique cards
    card_ease = {}
    for r in sorted(reviews, key=lambda x: x.reviewed_at):
        card_ease[r.flashcard_id] = r.ease_factor

    avg_ease = (
        sum(card_ease.values()) / len(card_ease) if card_ease else 2.5
    )

    return {
        "accuracy": accuracy,
        "average_ease_factor": avg_ease,
        "cards_reviewed": len(card_ease),
        "total_reviews": total_reviews,
    }


def start_study_session(
    db: Session,
    user_id: str,
    curriculum_id: str | None = None,
) -> StudySession:
    """Start a new study session.

    Args:
        db: Database session
        user_id: User identifier
        curriculum_id: Optional curriculum being studied

    Returns:
        Created StudySession object
    """
    session = StudySession(
        id=str(uuid.uuid4()),
        curriculum_id=curriculum_id,
        user_id=user_id,
        started_at=datetime.utcnow(),
        topics_studied=[],
        cards_reviewed=0,
        modules_worked_on=[],
    )
    db.add(session)
    db.commit()
    db.refresh(session)

    return session


def end_study_session(
    db: Session,
    session_id: str,
    topics_studied: list[str] | None = None,
    cards_reviewed: int | None = None,
    modules_worked_on: list[str] | None = None,
    summary: str | None = None,
) -> StudySession | None:
    """End a study session with summary.

    Args:
        db: Database session
        session_id: Session identifier
        topics_studied: Topics covered in session
        cards_reviewed: Number of cards reviewed
        modules_worked_on: Modules studied
        summary: AI-generated session summary

    Returns:
        Updated StudySession or None
    """
    session = db.query(StudySession).filter_by(id=session_id).first()
    if not session:
        return None

    session.ended_at = datetime.utcnow()

    if session.started_at:
        duration = session.ended_at - session.started_at
        session.duration_minutes = int(duration.total_seconds() / 60)

    if topics_studied is not None:
        session.topics_studied = topics_studied

    if cards_reviewed is not None:
        session.cards_reviewed = cards_reviewed

    if modules_worked_on is not None:
        session.modules_worked_on = modules_worked_on

    if summary is not None:
        session.session_summary = summary

    db.commit()
    db.refresh(session)

    return session


def get_recent_sessions(
    db: Session, user_id: str, limit: int = 5
) -> list[StudySession]:
    """Get recent study sessions for a user.

    Args:
        db: Database session
        user_id: User identifier
        limit: Maximum sessions to return

    Returns:
        List of StudySession objects
    """
    return (
        db.query(StudySession)
        .filter_by(user_id=user_id)
        .order_by(StudySession.ended_at.desc())
        .limit(limit)
        .all()
    )


def calculate_days_since_last_session(db: Session, user_id: str) -> int | None:
    """Calculate days since user's last completed study session.

    Args:
        db: Database session
        user_id: User identifier

    Returns:
        Number of days, or None if no prior sessions
    """
    last_session = (
        db.query(StudySession)
        .filter(StudySession.user_id == user_id, StudySession.ended_at.isnot(None))
        .order_by(StudySession.ended_at.desc())
        .first()
    )

    if not last_session or not last_session.ended_at:
        return None

    delta = datetime.utcnow() - last_session.ended_at
    return delta.days
