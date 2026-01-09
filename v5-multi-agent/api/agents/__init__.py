# StudyBuddy v5 Agent Definitions
from .tutor import create_tutor_agent, TUTOR_PROMPT
from .card_generator import create_card_generator_agent, CARD_GENERATOR_PROMPT
from .quality_checker import create_quality_checker_agent, QUALITY_CHECKER_PROMPT
from .supervisor import create_supervisor_agent, SUPERVISOR_PROMPT

__all__ = [
    "create_tutor_agent",
    "create_card_generator_agent",
    "create_quality_checker_agent",
    "create_supervisor_agent",
    "TUTOR_PROMPT",
    "CARD_GENERATOR_PROMPT",
    "QUALITY_CHECKER_PROMPT",
    "SUPERVISOR_PROMPT",
]
