"""Database package for StudyBuddy v11."""

from .models import (
    Base,
    User,
    Memory,
    LearningProgram,
    ConnectorConfig,
    Document,
    Flashcard,
    Conversation,
    Message,
    ProgramStats,
)
from .connection import (
    init_database,
    get_db,
    get_db_dependency,
    get_or_create_user,
    SessionLocal,
)

__all__ = [
    "Base",
    "User",
    "Memory",
    "LearningProgram",
    "ConnectorConfig",
    "Document",
    "Flashcard",
    "Conversation",
    "Message",
    "ProgramStats",
    "init_database",
    "get_db",
    "get_db_dependency",
    "get_or_create_user",
    "SessionLocal",
]
