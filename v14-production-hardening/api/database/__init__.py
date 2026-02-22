"""Database package for StudyBuddy v14."""

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
    # v13: Multi-model infrastructure
    ModelConfig,
    TokenUsage,
    BenchmarkResult,
    # v14: Production hardening
    SemanticCache,
    EmbeddingCache,
    BudgetAlert,
    AuditLog,
    ApiKey,
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
    # v13
    "ModelConfig",
    "TokenUsage",
    "BenchmarkResult",
    # v14
    "SemanticCache",
    "EmbeddingCache",
    "BudgetAlert",
    "AuditLog",
    "ApiKey",
    "init_database",
    "get_db",
    "get_db_dependency",
    "get_or_create_user",
    "SessionLocal",
]
