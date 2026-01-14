"""Services module for StudyBuddy v9."""

from .spaced_repetition import calculate_sm2, SM2Result
from .flashcard_cache import (
    compute_content_hash,
    get_cached_flashcards,
    cache_flashcards,
)
from .memory_store import MemoryStore
from .background_generator import BackgroundGenerator, prefetch_status
from .curriculum_service import (
    create_curriculum,
    get_user_curricula,
    get_curriculum_by_id,
    get_curriculum_progress,
    update_curriculum_progress,
    increment_module_attempts,
    get_module_performance,
    start_study_session,
    end_study_session,
    get_recent_sessions,
    calculate_days_since_last_session,
)
# v9: Advanced indexing (retrieval classes imported directly where needed)
from .indexing import chunk_reference_document, chunk_all_documents

# Note: Retrieval classes (HybridRetriever, RerankedRetriever, etc.) should be
# imported directly from api.services.retrieval where needed, to avoid import
# issues with optional dependencies on serverless platforms.

__all__ = [
    "calculate_sm2",
    "SM2Result",
    "compute_content_hash",
    "get_cached_flashcards",
    "cache_flashcards",
    "MemoryStore",
    "BackgroundGenerator",
    "prefetch_status",
    # Curriculum service
    "create_curriculum",
    "get_user_curricula",
    "get_curriculum_by_id",
    "get_curriculum_progress",
    "update_curriculum_progress",
    "increment_module_attempts",
    "get_module_performance",
    "start_study_session",
    "end_study_session",
    "get_recent_sessions",
    "calculate_days_since_last_session",
    # v9: Advanced indexing
    "chunk_reference_document",
    "chunk_all_documents",
]
