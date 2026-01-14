"""Curriculum Planner Agent - Deep agent for learning path generation.

The Curriculum Planner transforms StudyBuddy into a proactive learning system.
It implements four deep agent capabilities:

1. Planning and Task Decomposition - Backward chaining for curriculum generation
2. Subagent Delegation - Delegates flashcard creation to Card Generator
3. Context Management - Summarizes progress across long sessions
4. Backtracking - Handles learning struggles with alternative approaches

This is the "deep agent" from Chapter 7 that orchestrates long-running
learning projects across multiple sessions.
"""

import json
import uuid
from datetime import datetime, timedelta
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from sqlalchemy.orm import Session

from .card_generator import generate_cards_batch
from ..services.flashcard_cache import cache_flashcards
from ..services.curriculum_service import (
    get_curriculum_progress,
    get_module_performance,
    get_recent_sessions,
    calculate_days_since_last_session,
)


CURRICULUM_PLANNER_PROMPT = """You are StudyBuddy's Curriculum Planner,
an expert at designing effective learning paths.

Your job: Take a learning goal and create a structured curriculum that
guides the student from where they are to where they want to be.

When creating a curriculum:
1. Break the goal into major topics or modules
2. Identify prerequisites - what must be learned first
3. Sequence topics logically based on dependencies
4. Estimate time needed for each topic
5. Define checkpoints to assess understanding

Use backward chaining: Start from the goal, identify what's needed to
achieve it, then what's needed for that, until you reach foundational
concepts the student likely already knows.

You have access to the AI Engineering reference materials for domain
knowledge about what topics exist and how they relate.

Output format - respond with JSON:
{
    "goal": "The learning objective",
    "estimated_duration_weeks": 8,
    "modules": [
        {
            "id": "m1",
            "title": "Module title",
            "description": "What this covers",
            "prerequisites": [],
            "topics": ["topic1", "topic2"],
            "estimated_hours": 10,
            "checkpoint": "How to assess completion"
        }
    ],
    "learning_path": ["m1", "m2", "m3"]
}"""


def create_curriculum_planner(model_name: str = "gpt-4o") -> ChatOpenAI:
    """Create the Curriculum Planner agent.

    Uses gpt-4o for better reasoning on complex curriculum design.

    Args:
        model_name: Model to use (default gpt-4o)

    Returns:
        ChatOpenAI instance configured for curriculum planning
    """
    return ChatOpenAI(model=model_name, temperature=0.4)


def generate_curriculum(
    llm: ChatOpenAI,
    goal: str,
    current_knowledge: list[str] | None = None,
    available_time_hours: int | None = None,
    context: str = "",
) -> dict:
    """Generate a complete curriculum for a learning goal using backward chaining.

    The curriculum planner uses backward chaining:
    1. Start from the goal
    2. Identify what's needed to achieve it
    3. Identify prerequisites for those
    4. Continue until reaching concepts the student knows
    5. Sequence based on dependencies

    Args:
        llm: The language model
        goal: What the user wants to learn
        current_knowledge: Topics user already knows
        available_time_hours: Weekly time commitment
        context: Retrieved knowledge base content

    Returns:
        Curriculum dictionary with modules and learning path
    """
    current_knowledge = current_knowledge or []

    user_content = f"""Create a curriculum for this learning goal: {goal}

Student's current knowledge: {', '.join(current_knowledge) if current_knowledge else 'Assume foundational programming knowledge'}

{'Weekly time available: ' + str(available_time_hours) + ' hours' if available_time_hours else ''}

Use backward chaining:
1. What does the student need to know to achieve this goal?
2. For each of those, what prerequisites are needed?
3. Continue until you reach topics the student already knows
4. Sequence everything based on dependencies

Reference material for AI engineering topics:
{context if context else 'Use your knowledge of AI engineering fundamentals.'}

Provide the curriculum as JSON."""

    messages = [
        SystemMessage(content=CURRICULUM_PLANNER_PROMPT),
        HumanMessage(content=user_content),
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
        return {"error": "Failed to parse curriculum", "raw": response.content}


def generate_curriculum_flashcards(
    curriculum: dict,
    card_generator_llm: ChatOpenAI,
    search_func,
    db: Session,
    module_id: str | None = None,
) -> dict:
    """Generate flashcards for curriculum topics via Card Generator subagent.

    Implements hybrid generation strategy:
    - If module_id is None: Generate for first module only
    - If module_id is specified: Generate for that specific module

    This delegates to the existing Card Generator agent, demonstrating
    subagent spawning and delegation pattern.

    Args:
        curriculum: Curriculum dictionary with modules
        card_generator_llm: LLM for card generation
        search_func: Function to search knowledge base
        db: Database session
        module_id: Specific module to generate cards for (None = first module)

    Returns:
        Results dict with cards generated per module
    """
    results = {"modules": {}, "total_cards": 0}

    modules = curriculum.get("modules", [])
    if not modules:
        return results

    # Determine which modules to process
    if module_id:
        # Generate for specific module
        modules_to_process = [m for m in modules if m.get("id") == module_id]
    else:
        # Generate for first module only (hybrid strategy)
        modules_to_process = [modules[0]]

    for module in modules_to_process:
        mid = module.get("id", "unknown")
        module_cards = []

        for topic in module.get("topics", []):
            # Get context from knowledge base (RAG)
            context = search_func(topic, k=4)

            # Delegate to Card Generator subagent
            cards = generate_cards_batch(
                card_generator_llm,
                topic=topic,
                context=context,
                count=5,  # 5 cards per topic
            )

            if cards:
                # Cache the generated cards
                cached = cache_flashcards(
                    topic=topic,
                    source_context=context,
                    cards_data=cards,
                    db=db,
                    chapter_id=None,  # Curriculum cards aren't chapter-bound
                )
                module_cards.extend(cached)

        results["modules"][mid] = {
            "title": module.get("title", ""),
            "card_count": len(module_cards),
            "card_ids": [c.id for c in module_cards],
        }
        results["total_cards"] += len(module_cards)

    return results


def evaluate_checkpoint(
    llm: ChatOpenAI,
    module: dict,
    user_performance: dict,
) -> dict:
    """Evaluate whether user has passed a module checkpoint.

    Combines heuristic metrics with LLM assessment for nuanced evaluation.

    Checkpoint criteria:
    - Accuracy >= 70%
    - Ease factor >= 2.3 (cards aren't too difficult)
    - Sufficient cards studied (3+ per topic)

    Args:
        llm: Language model for evaluation
        module: The module being evaluated
        user_performance: Performance metrics from get_module_performance()

    Returns:
        Evaluation dict with passed, confidence, feedback, recommendations
    """
    topics = module.get("topics", [])
    avg_ease = user_performance.get("average_ease_factor", 2.5)
    accuracy = user_performance.get("accuracy", 0)
    cards_studied = user_performance.get("cards_reviewed", 0)

    # Heuristic checkpoint evaluation
    passed = (
        accuracy >= 0.7  # 70% accuracy threshold
        and avg_ease >= 2.3  # Cards aren't too difficult
        and cards_studied >= len(topics) * 3  # Studied enough cards
    )

    # Get LLM assessment for nuance
    prompt = f"""Evaluate this student's checkpoint for module: {module.get('title', 'Unknown')}

Checkpoint criteria: {module.get('checkpoint', 'Demonstrate understanding of key concepts')}

Performance data:
- Topics covered: {', '.join(topics)}
- Flashcard accuracy: {accuracy:.0%}
- Average ease factor: {avg_ease:.2f}
- Cards reviewed: {cards_studied}

Based on this data, has the student sufficiently mastered this module?
What specific areas might need more attention?

Respond with JSON:
{{"passed": true/false, "confidence": 0.0-1.0, "feedback": "...", "recommendations": ["..."]}}"""

    response = llm.invoke(prompt)

    try:
        content = response.content.strip()
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
            content = content.strip()

        evaluation = json.loads(content)
        evaluation["metrics"] = {
            "accuracy": accuracy,
            "ease_factor": avg_ease,
            "cards_studied": cards_studied,
        }
        return evaluation

    except (json.JSONDecodeError, Exception):
        return {
            "passed": passed,
            "confidence": 0.7,
            "feedback": "Checkpoint evaluated based on performance metrics.",
            "recommendations": (
                []
                if passed
                else ["Review the topics and practice more flashcards."]
            ),
            "metrics": {
                "accuracy": accuracy,
                "ease_factor": avg_ease,
                "cards_studied": cards_studied,
            },
        }


def summarize_learning_progress(
    curriculum: dict,
    completed_modules: list[str],
    current_module: str,
    struggle_areas: list[str],
    llm: ChatOpenAI,
) -> str:
    """Create compressed summary of learning progress for context management.

    Called periodically to prevent context window exhaustion during
    long study sessions. Produces a 2-3 sentence summary.

    Args:
        curriculum: The curriculum being studied
        completed_modules: List of completed module IDs
        current_module: Current module ID
        struggle_areas: Topics the student finds difficult
        llm: Language model for summarization

    Returns:
        Compressed progress summary string
    """
    modules = curriculum.get("modules", [])
    total_modules = len(modules)
    progress_pct = len(completed_modules) / total_modules * 100 if total_modules else 0

    # Get module titles for completed modules
    completed_titles = []
    current_title = None
    for m in modules:
        if m.get("id") in completed_modules:
            completed_titles.append(m.get("title", m.get("id")))
        if m.get("id") == current_module:
            current_title = m.get("title", current_module)

    prompt = f"""Summarize this student's learning progress concisely:

Goal: {curriculum.get('goal', 'Learning AI Engineering')}
Progress: {progress_pct:.0f}% complete

Completed modules:
{chr(10).join(f'- {t}' for t in completed_titles) if completed_titles else '- None yet'}

Currently studying: {current_title or 'Not started'}

Struggle areas: {', '.join(struggle_areas) if struggle_areas else 'None identified'}

Provide a 2-3 sentence summary capturing:
1. Overall progress status
2. Current focus area
3. Any concerns or adjustments needed"""

    response = llm.invoke(prompt)
    return response.content.strip()


def get_session_context(
    user_id: str,
    curriculum_id: str,
    db: Session,
) -> dict:
    """Load context for a returning student.

    Combines curriculum state, recent performance, and session history
    into a coherent context object for session resumption.

    Args:
        user_id: User identifier
        curriculum_id: Curriculum identifier
        db: Database session

    Returns:
        Context dict with progress, recent sessions, performance
    """
    from ..services.curriculum_service import (
        get_curriculum_by_id,
        get_curriculum_progress,
    )

    # Get curriculum and progress
    curriculum = get_curriculum_by_id(db, curriculum_id)
    progress = get_curriculum_progress(db, curriculum_id, user_id)

    if not curriculum or not progress:
        return {
            "curriculum_goal": None,
            "current_module": None,
            "modules_completed": [],
            "overall_progress": 0,
            "recent_sessions": [],
            "current_performance": {},
            "days_since_last_session": None,
        }

    # Get current module info
    current_module = None
    current_module_topics = []
    for m in curriculum.curriculum_data.get("modules", []):
        if m.get("id") == progress.current_module_id:
            current_module = m
            current_module_topics = m.get("topics", [])
            break

    # Get performance for current module
    performance = (
        get_module_performance(db, user_id, current_module_topics)
        if current_module_topics
        else {}
    )

    # Get recent sessions
    recent_sessions = get_recent_sessions(db, user_id, limit=5)

    # Calculate days since last session
    days_away = calculate_days_since_last_session(db, user_id)

    return {
        "curriculum_goal": curriculum.goal,
        "current_module": current_module,
        "modules_completed": progress.completed_modules or [],
        "overall_progress": progress.progress_percentage,
        "recent_sessions": [
            {
                "date": s.ended_at.isoformat() if s.ended_at else None,
                "duration_minutes": s.duration_minutes,
                "topics_studied": s.topics_studied,
                "cards_reviewed": s.cards_reviewed,
            }
            for s in recent_sessions
            if s.ended_at
        ],
        "current_performance": performance,
        "days_since_last_session": days_away,
    }


def resume_study_session(
    user_id: str,
    curriculum_id: str,
    db: Session,
    llm: ChatOpenAI,
) -> dict:
    """Resume a student's study session with appropriate context.

    Handles different scenarios based on time since last session:
    - First session ever: Welcome message
    - > 7 days away: Review + continue
    - > 2 days away: Brief re-engagement with summary
    - Recent: Just continue

    Args:
        user_id: User identifier
        curriculum_id: Curriculum identifier
        db: Database session
        llm: Language model for summarization

    Returns:
        Dict with message, action, context, and optional review_topics
    """
    from ..services.curriculum_service import get_curriculum_by_id

    context = get_session_context(user_id, curriculum_id, db)
    days_away = context.get("days_since_last_session")

    curriculum = get_curriculum_by_id(db, curriculum_id)

    if days_away is None:
        # First session ever
        return {
            "message": "Welcome! Let's start your learning journey.",
            "action": "start_curriculum",
            "context": context,
            "review_topics": [],
        }

    elif days_away > 7:
        # Been away a while - need review
        # Get recent module titles for review
        review_modules = context.get("modules_completed", [])[-2:]
        return {
            "message": f"Welcome back! It's been {days_away} days. Let's do a quick review before continuing.",
            "action": "review_then_continue",
            "review_topics": review_modules,
            "context": context,
        }

    elif days_away > 2:
        # Brief break - gentle re-engagement
        if curriculum:
            summary = summarize_learning_progress(
                curriculum=curriculum.curriculum_data,
                completed_modules=context.get("modules_completed", []),
                current_module=context.get("current_module", {}).get("id", ""),
                struggle_areas=[],
                llm=llm,
            )
        else:
            summary = "Ready to continue your studies."

        return {
            "message": f"Welcome back! Here's where we left off: {summary}",
            "action": "continue",
            "context": context,
            "review_topics": [],
        }

    else:
        # Recent session - just continue
        current_title = context.get("current_module", {}).get("title", "your studies")
        return {
            "message": f"Ready to continue with {current_title}?",
            "action": "continue",
            "context": context,
            "review_topics": [],
        }


def handle_learning_struggle(
    curriculum: dict,
    struggling_module: dict,
    attempts: int,
    llm: ChatOpenAI,
) -> dict:
    """Handle backtracking when a student is stuck on a module.

    Implements progressive intervention:
    - Attempts 1-2: Suggest reinforcement (review + more practice)
    - Attempts 3-4: Try alternative learning approach
    - Attempts 5+: Backtrack to prerequisites

    Args:
        curriculum: The curriculum being studied
        struggling_module: Module the student is struggling with
        attempts: Number of checkpoint attempts
        llm: Language model for alternative approach suggestions

    Returns:
        Dict with action and recommendations
    """
    if attempts < 2:
        # First struggle: suggest review and more practice
        return {
            "action": "reinforce",
            "recommendation": "Review foundational concepts and practice more flashcards",
            "additional_cards_needed": 10,
            "backtrack_to": None,
        }

    elif attempts < 4:
        # Continued struggle: try alternative learning approach
        prompt = f"""A student is struggling with this module despite multiple attempts:

Module: {struggling_module.get('title', 'Unknown')}
Topics: {', '.join(struggling_module.get('topics', []))}
Attempts: {attempts}

Suggest alternative approaches:
1. Different ways to explain these concepts
2. Prerequisite topics that might need review
3. Simpler stepping stones to build up to this material

Provide JSON:
{{"alternative_approach": "...", "prerequisite_review": ["..."], "simplified_path": ["..."]}}"""

        response = llm.invoke(prompt)

        try:
            content = response.content.strip()
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
                content = content.strip()

            alternatives = json.loads(content)
            return {
                "action": "alternative_approach",
                "recommendation": alternatives.get(
                    "alternative_approach", "Try a different learning approach"
                ),
                "prerequisite_review": alternatives.get("prerequisite_review", []),
                "simplified_path": alternatives.get("simplified_path", []),
                "backtrack_to": None,
            }

        except (json.JSONDecodeError, Exception):
            return {
                "action": "alternative_approach",
                "recommendation": "Try reviewing the prerequisites and breaking the concepts into smaller pieces.",
                "prerequisite_review": struggling_module.get("prerequisites", []),
                "simplified_path": [],
                "backtrack_to": None,
            }

    else:
        # Persistent struggle: backtrack to prerequisites
        prerequisites = struggling_module.get("prerequisites", [])

        return {
            "action": "backtrack",
            "recommendation": "Return to prerequisite modules to strengthen foundational understanding",
            "backtrack_to": prerequisites if prerequisites else None,
            "reason": "Foundational gaps detected after multiple attempts",
        }
