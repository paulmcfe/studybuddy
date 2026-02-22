"""SQLAlchemy models for StudyBuddy v14.

v14 adds production hardening on top of v13:
- Semantic response cache (SemanticCache)
- Embedding cache (EmbeddingCache)
- Budget alerts (BudgetAlert)
- Audit logging (AuditLog)
- API keys for inter-agent auth (ApiKey)
- User tier for tiered rate limiting

v13 adds advanced infrastructure on top of v12:
- Multi-model support (OpenAI, Together AI, Ollama)
- Per-task model configuration with fallback chains
- Token usage tracking for cost analysis
- Performance benchmarking storage

v12 adds production features: JWT authentication, multi-user data isolation,
rate limiting, and monitoring on top of v11's MCP Connectors.

v11 added MCP Connectors on top of v10's Learning Programs.
Each program can have external data source connectors:
- Fetch/URL: Import web pages as learning materials
- GitHub: Import docs and markdown from repositories
- Brave Search: Augment the tutor with web search

v10 introduced Learning Programs - containers for subject-specific learning.
Each program has its own:
- Knowledge base (documents indexed in a dedicated Qdrant collection)
- Topic list (curriculum structure)
- Flashcards with spaced repetition
- Progress tracking
"""

from datetime import datetime
import uuid
from sqlalchemy import (
    Column,
    String,
    Integer,
    Float,
    DateTime,
    JSON,
    ForeignKey,
    Text,
    Boolean,
    Index,
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


def generate_uuid() -> str:
    """Generate a UUID string for primary keys."""
    return str(uuid.uuid4())


class User(Base):
    """User profile with authentication.

    v12 adds full JWT authentication with registration, login,
    password reset, and email verification.

    Auth columns are nullable to maintain compatibility with the
    existing "default" user created during development.
    """

    __tablename__ = "users"

    id = Column(String(50), primary_key=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    preferences = Column(JSON, default=dict)

    # v12: Authentication fields
    email = Column(String(255), unique=True, nullable=True, index=True)
    password_hash = Column(String(255), nullable=True)
    is_active = Column(Boolean, default=True)
    is_verified = Column(Boolean, default=False)
    verification_token = Column(String(100), nullable=True)
    reset_token = Column(String(100), nullable=True)
    reset_token_expires = Column(DateTime, nullable=True)

    # v14: Tiered rate limiting
    tier = Column(String(20), default="free")  # "free", "pro", "enterprise"

    # Relationships
    programs = relationship("LearningProgram", back_populates="user")
    memories = relationship("Memory", back_populates="user")


class Memory(Base):
    """User memory storage for learning preferences, struggles, goals.

    Memories are organized by namespace:
    - preferences: Learning style, favorite topics
    - struggles: Topics the user finds difficult
    - goals: What the user is studying for
    - sessions: Summaries of past study sessions
    """

    __tablename__ = "memories"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    user_id = Column(String(50), ForeignKey("users.id"), nullable=False, index=True)
    namespace = Column(String(100), nullable=False, index=True)
    key = Column(String(200), nullable=False)
    value = Column(JSON, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    user = relationship("User", back_populates="memories")

    __table_args__ = (Index("ix_memory_user_namespace", "user_id", "namespace"),)


# ============== v10: Learning Programs ==============


class LearningProgram(Base):
    """A learning program for studying a specific subject.

    Each program is a container with:
    - Its own Qdrant collection for document vectors
    - A topic list (either uploaded or AI-generated)
    - Flashcards scoped to this program
    - Progress tracking
    """

    __tablename__ = "learning_programs"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    user_id = Column(String(50), ForeignKey("users.id"), nullable=False, index=True)

    # Program metadata
    name = Column(String(200), nullable=False)
    description = Column(Text, nullable=True)

    # Topic list structure (markdown parsed into JSON)
    # Format: {"chapters": [{"title": str, "topics": [{"title": str, "subtopics": [...]}]}]}
    topic_list = Column(JSON, default=dict)

    # Settings
    settings = Column(JSON, default=dict)

    # Status
    status = Column(String(20), default="active")  # active, archived
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    user = relationship("User", back_populates="programs")
    documents = relationship("Document", back_populates="program", cascade="all, delete-orphan")
    flashcards = relationship("Flashcard", back_populates="program", cascade="all, delete-orphan")
    conversations = relationship("Conversation", back_populates="program", cascade="all, delete-orphan")
    connectors = relationship("ConnectorConfig", back_populates="program", cascade="all, delete-orphan")

    @property
    def qdrant_collection(self) -> str:
        """Collection name for this program's vectors."""
        return f"program_{self.id}"


# ============== v11: MCP Connectors ==============


class ConnectorConfig(Base):
    """External connector configuration for a learning program.

    Stores connection settings for MCP-based connectors:
    - fetch: Import web pages as learning materials via URL
    - github: Import markdown/docs from GitHub repositories
    - brave_search: Augment the tutor with web search capability
    """

    __tablename__ = "connector_configs"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    program_id = Column(
        String(36), ForeignKey("learning_programs.id"), nullable=False, index=True
    )

    # Connector type: "fetch", "github", "brave_search"
    connector_type = Column(String(50), nullable=False)

    # Display name (e.g., "Python Docs", "langchain-ai/langchain")
    name = Column(String(200), nullable=False)

    # Connector-specific configuration (JSON)
    # fetch: {} (URL provided per-import, no persistent config needed)
    # github: {"owner": "...", "repo": "...", "token": "...", "branch": "main"}
    # brave_search: {"api_key": "...", "enabled": true}
    config = Column(JSON, nullable=False, default=dict)

    # Sync status tracking
    status = Column(String(20), default="configured")  # configured, syncing, synced, failed
    last_sync_at = Column(DateTime, nullable=True)
    error_message = Column(Text, nullable=True)

    # Incremental sync state (JSON)
    # github: {"path/file.md": {"content_hash": "...", "document_id": "...", "synced_at": "..."}}
    sync_state = Column(JSON, default=dict)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    program = relationship("LearningProgram", back_populates="connectors")

    __table_args__ = (
        Index("ix_connector_program_type", "program_id", "connector_type"),
    )


class Document(Base):
    """Document metadata — from file uploads or MCP connector imports.

    The actual vectors are stored in Qdrant, but we track
    document metadata here for management and deduplication.

    v11 adds source tracking: documents can come from file uploads (v10),
    URL imports (Fetch connector), or GitHub imports (GitHub connector).
    """

    __tablename__ = "documents"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    program_id = Column(String(36), ForeignKey("learning_programs.id"), nullable=False, index=True)

    # File metadata
    filename = Column(String(255), nullable=False)
    file_type = Column(String(50))  # application/pdf, text/markdown, text/plain
    file_size = Column(Integer)  # bytes
    file_path = Column(String(500))  # Storage location

    # Indexing status
    status = Column(String(20), default="pending")  # pending, processing, indexed, failed
    chunks_count = Column(Integer, nullable=True)
    indexed_at = Column(DateTime, nullable=True)
    error_message = Column(Text, nullable=True)

    # Deduplication
    content_hash = Column(String(64), index=True)  # SHA-256 hash

    # v11: Source tracking for connector-imported documents
    source_type = Column(String(20), default="upload")  # upload, url, github
    source_url = Column(String(2000), nullable=True)  # Original URL for imported content
    connector_id = Column(
        String(36), ForeignKey("connector_configs.id"), nullable=True
    )

    created_at = Column(DateTime, default=datetime.utcnow)

    program = relationship("LearningProgram", back_populates="documents")

    __table_args__ = (Index("ix_document_program_hash", "program_id", "content_hash"),)


class Flashcard(Base):
    """Flashcard scoped to a learning program.

    Includes spaced repetition fields for the SM-2 algorithm.
    """

    __tablename__ = "flashcards"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    program_id = Column(String(36), ForeignKey("learning_programs.id"), nullable=False, index=True)

    # Card content
    topic = Column(String(200), nullable=False, index=True)
    question = Column(Text, nullable=False)
    answer = Column(Text, nullable=False)
    source_context = Column(Text, nullable=True)  # Retrieved context used to generate

    # Deduplication
    content_hash = Column(String(64), index=True)  # SHA-256 of (topic + source_context)

    # Spaced repetition fields (SM-2 algorithm)
    ease_factor = Column(Float, default=2.5)
    interval = Column(Integer, default=0)  # Days until next review
    repetitions = Column(Integer, default=0)  # Consecutive correct reviews
    next_review = Column(DateTime, nullable=True)

    # Quality metadata
    quality_score = Column(Integer, default=3)  # 1-5
    was_revised = Column(Boolean, default=False)

    created_at = Column(DateTime, default=datetime.utcnow)

    program = relationship("LearningProgram", back_populates="flashcards")

    __table_args__ = (Index("ix_flashcard_program_topic", "program_id", "topic"),)


class Conversation(Base):
    """Chat conversation within a learning program."""

    __tablename__ = "conversations"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    program_id = Column(String(36), ForeignKey("learning_programs.id"), nullable=False, index=True)
    user_id = Column(String(50), ForeignKey("users.id"), nullable=False)

    title = Column(String(200), nullable=True)
    started_at = Column(DateTime, default=datetime.utcnow)
    last_activity = Column(DateTime, default=datetime.utcnow)
    message_count = Column(Integer, default=0)

    program = relationship("LearningProgram", back_populates="conversations")
    messages = relationship("Message", back_populates="conversation", cascade="all, delete-orphan")


class Message(Base):
    """Individual message in a conversation."""

    __tablename__ = "messages"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    conversation_id = Column(String(36), ForeignKey("conversations.id"), nullable=False, index=True)

    role = Column(String(20), nullable=False)  # user, assistant
    content = Column(Text, nullable=False)

    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    tokens_used = Column(Integer, nullable=True)
    model = Column(String(50), nullable=True)

    # Feedback
    feedback = Column(String(20), nullable=True)  # positive, negative
    feedback_text = Column(Text, nullable=True)

    conversation = relationship("Conversation", back_populates="messages")


# ============== Stats and Analytics ==============


class ProgramStats(Base):
    """Aggregated statistics for a learning program.

    Pre-computed stats for efficient dashboard queries.
    Updated after each study session.
    """

    __tablename__ = "program_stats"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    program_id = Column(String(36), ForeignKey("learning_programs.id"), nullable=False, unique=True)

    # Study metrics
    total_flashcards = Column(Integer, default=0)
    cards_reviewed = Column(Integer, default=0)
    cards_mastered = Column(Integer, default=0)  # interval > 21 days

    # Accuracy
    total_reviews = Column(Integer, default=0)
    correct_reviews = Column(Integer, default=0)

    # Time tracking
    total_study_time_minutes = Column(Integer, default=0)
    last_study_date = Column(DateTime, nullable=True)

    # Progress
    topics_studied = Column(Integer, default=0)
    total_topics = Column(Integer, default=0)

    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


# ============== v13: Multi-Model Infrastructure ==============


class ModelConfig(Base):
    """Per-task model configuration for multi-model support.

    Allows users to configure which models to use for different tasks:
    - flashcard_generation: Creating flashcards
    - tutoring: Chat/tutoring responses
    - curriculum: Curriculum generation
    - embedding: Vector embeddings

    Each task can have a primary model and optional fallback.
    """

    __tablename__ = "model_configs"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    user_id = Column(String(50), ForeignKey("users.id"), nullable=True, index=True)

    # Task type: "flashcard_generation", "tutoring", "curriculum", "embedding"
    task_type = Column(String(50), nullable=False)

    # Primary model configuration
    primary_provider = Column(String(50), nullable=False)  # "openai", "together", "ollama"
    primary_model = Column(String(100), nullable=False)

    # Fallback configuration (optional)
    fallback_provider = Column(String(50), nullable=True)
    fallback_model = Column(String(100), nullable=True)

    # Generation settings
    temperature = Column(Float, default=0.7)
    max_tokens = Column(Integer, default=1000)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (Index("ix_model_config_user_task", "user_id", "task_type"),)


class TokenUsage(Base):
    """Track token usage for cost analysis and billing.

    Records every LLM call with token counts and computed costs.
    Used for the cost comparison dashboard and usage analytics.
    """

    __tablename__ = "token_usage"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    user_id = Column(String(50), ForeignKey("users.id"), nullable=False, index=True)
    program_id = Column(String(36), ForeignKey("learning_programs.id"), nullable=True)

    # Model identification
    provider = Column(String(50), nullable=False)  # "openai", "together", "ollama"
    model = Column(String(100), nullable=False)
    task_type = Column(String(50), nullable=False)

    # Token counts
    input_tokens = Column(Integer, nullable=False)
    output_tokens = Column(Integer, nullable=False)

    # Cost (in cents for precision)
    cost_cents = Column(Float, nullable=False)

    # Performance
    latency_ms = Column(Float, nullable=True)

    # Fallback tracking
    was_fallback = Column(Boolean, default=False)

    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("ix_token_usage_user_date", "user_id", "created_at"),
        Index("ix_token_usage_provider", "provider", "model"),
    )


class BenchmarkResult(Base):
    """Store benchmark results for model comparison.

    Records performance metrics from benchmark runs to help
    users compare models across latency, throughput, and quality.
    """

    __tablename__ = "benchmark_results"

    id = Column(String(36), primary_key=True, default=generate_uuid)

    # Model identification
    provider = Column(String(50), nullable=False)
    model = Column(String(100), nullable=False)
    task_type = Column(String(50), nullable=False)
    prompt_template = Column(String(100), nullable=False)

    # Performance metrics
    latency_ms = Column(Float, nullable=False)
    tokens_per_second = Column(Float, nullable=True)
    response_length = Column(Integer, nullable=False)

    # Quality assessment (optional, can be filled in later)
    quality_score = Column(Float, nullable=True)

    # Status
    success = Column(Boolean, default=True)
    error_message = Column(Text, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (Index("ix_benchmark_provider_model", "provider", "model"),)


# ============== v14: Production Hardening ==============


class SemanticCache(Base):
    """Cache for semantically similar query responses.

    Stores query embeddings so that similar future queries can be
    served from cache rather than re-invoking the LLM.
    """

    __tablename__ = "semantic_cache"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    program_id = Column(String(36), ForeignKey("learning_programs.id"), nullable=True, index=True)
    query_text = Column(Text, nullable=False)
    query_embedding = Column(JSON, nullable=False)  # List[float]
    response_text = Column(Text, nullable=False)
    model_used = Column(String(100), nullable=True)
    task_type = Column(String(50), nullable=False)  # "tutoring", "flashcard_generation", etc.
    hit_count = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime, nullable=True)  # TTL

    __table_args__ = (
        Index("ix_semantic_cache_program_task", "program_id", "task_type"),
    )


class EmbeddingCache(Base):
    """Cache for document chunk embeddings to avoid re-embedding.

    Stores the SHA-256 hash of text content and its embedding vector.
    Used by DatabaseBackedEmbeddings wrapper.
    """

    __tablename__ = "embedding_cache"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    content_hash = Column(String(64), unique=True, nullable=False, index=True)
    embedding = Column(JSON, nullable=False)  # List[float]
    model = Column(String(100), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


class BudgetAlert(Base):
    """Configurable spending limits with alert thresholds.

    Users can set a monthly budget and a warning threshold percentage.
    When spending exceeds the threshold, alerts are triggered.
    """

    __tablename__ = "budget_alerts"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    user_id = Column(String(50), ForeignKey("users.id"), nullable=False, index=True)
    monthly_budget_cents = Column(Integer, nullable=False)  # Budget in cents
    warning_threshold_pct = Column(Integer, default=80)  # Alert at N% of budget
    is_active = Column(Boolean, default=True)
    last_alert_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class AuditLog(Base):
    """Immutable audit log for security-relevant events.

    Records agent interactions, data access, authentication events,
    and guardrail triggers. Entries are only inserted, never updated
    or deleted.
    """

    __tablename__ = "audit_logs"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    user_id = Column(String(50), nullable=True, index=True)  # Null for system/agent events
    action = Column(String(100), nullable=False, index=True)
    resource_type = Column(String(50), nullable=True)
    resource_id = Column(String(36), nullable=True)
    source_ip = Column(String(45), nullable=True)
    user_agent = Column(String(500), nullable=True)
    result = Column(String(20), nullable=False)  # "success", "failure", "blocked"
    details = Column(JSON, nullable=True)

    __table_args__ = (
        Index("ix_audit_user_action", "user_id", "action"),
    )


class ApiKey(Base):
    """API keys for inter-agent and MCP authentication.

    External agents calling StudyBuddy's MCP tools must authenticate
    with an API key. Keys are stored as SHA-256 hashes.
    """

    __tablename__ = "api_keys"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    name = Column(String(200), nullable=False)  # e.g., "External Tutor Agent"
    key_hash = Column(String(64), nullable=False, unique=True, index=True)  # SHA-256
    key_prefix = Column(String(8), nullable=False)  # First 8 chars for identification
    scopes = Column(JSON, default=list)  # e.g., ["mcp:generate_flashcards"]
    is_active = Column(Boolean, default=True)
    created_by = Column(String(50), ForeignKey("users.id"), nullable=True)
    last_used_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime, nullable=True)
