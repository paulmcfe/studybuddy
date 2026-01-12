# StudyBuddy v7 Agent Definitions
from .tutor import create_tutor_agent, TUTOR_PROMPT
from .card_generator import create_card_generator_agent, CARD_GENERATOR_PROMPT
from .quality_checker import create_quality_checker_agent, QUALITY_CHECKER_PROMPT
from .supervisor import create_supervisor_agent, SUPERVISOR_PROMPT
from .scheduler import create_scheduler_agent, SCHEDULER_PROMPT
from .curriculum_planner import (
    create_curriculum_planner,
    CURRICULUM_PLANNER_PROMPT,
    generate_curriculum,
    generate_curriculum_flashcards,
    evaluate_checkpoint,
    summarize_learning_progress,
    get_session_context,
    resume_study_session,
    handle_learning_struggle,
)

__all__ = [
    "create_tutor_agent",
    "create_card_generator_agent",
    "create_quality_checker_agent",
    "create_supervisor_agent",
    "create_scheduler_agent",
    "create_curriculum_planner",
    "TUTOR_PROMPT",
    "CARD_GENERATOR_PROMPT",
    "QUALITY_CHECKER_PROMPT",
    "SUPERVISOR_PROMPT",
    "SCHEDULER_PROMPT",
    "CURRICULUM_PLANNER_PROMPT",
    "generate_curriculum",
    "generate_curriculum_flashcards",
    "evaluate_checkpoint",
    "summarize_learning_progress",
    "get_session_context",
    "resume_study_session",
    "handle_learning_struggle",
]
