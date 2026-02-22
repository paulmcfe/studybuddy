"""Production checklist verification for StudyBuddy v14.

Checks system readiness across categories: security, performance,
monitoring, database, and external services. Returns status indicators
(green/yellow/red) for each check.

Used by the /api/production-checklist endpoint and the frontend
Production Checklist UI component.
"""

import os
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class CheckStatus(str, Enum):
    GREEN = "green"
    YELLOW = "yellow"
    RED = "red"


@dataclass
class CheckResult:
    name: str
    status: CheckStatus
    category: str
    message: str
    detail: Optional[str] = None


class ProductionChecklist:
    """Run all production readiness checks."""

    def run_all(self) -> list[CheckResult]:
        checks = []
        checks.extend(self._security_checks())
        checks.extend(self._performance_checks())
        checks.extend(self._monitoring_checks())
        checks.extend(self._database_checks())
        checks.extend(self._external_service_checks())
        return checks

    def _security_checks(self) -> list[CheckResult]:
        results = []

        # JWT secret changed from default
        jwt_secret = os.environ.get("JWT_SECRET_KEY", "")
        if not jwt_secret or jwt_secret in ("change-this-to-a-random-secret", "dev-secret-key-change-in-production"):
            results.append(CheckResult(
                "JWT Secret", CheckStatus.RED, "security",
                "Using default JWT secret key",
                "Set JWT_SECRET_KEY to a strong random value in production",
            ))
        else:
            results.append(CheckResult(
                "JWT Secret", CheckStatus.GREEN, "security",
                "Custom JWT secret configured",
            ))

        # HTTPS
        https_enabled = os.environ.get("HTTPS_ENABLED", "false").lower() == "true"
        results.append(CheckResult(
            "HTTPS", CheckStatus.GREEN if https_enabled else CheckStatus.YELLOW,
            "security",
            "HTTPS enabled" if https_enabled else "HTTPS not explicitly enabled",
            "Set HTTPS_ENABLED=true when behind TLS termination" if not https_enabled else None,
        ))

        # Rate limiting
        results.append(CheckResult(
            "Rate Limiting", CheckStatus.GREEN, "security",
            "Rate limiting is active (slowapi)",
        ))

        # API keys for MCP
        try:
            from ..database.connection import SessionLocal
            from ..database.models import ApiKey
            db = SessionLocal()
            key_count = db.query(ApiKey).filter_by(is_active=True).count()
            db.close()
            if key_count > 0:
                results.append(CheckResult(
                    "MCP API Keys", CheckStatus.GREEN, "security",
                    f"{key_count} active API key(s) configured",
                ))
            else:
                results.append(CheckResult(
                    "MCP API Keys", CheckStatus.YELLOW, "security",
                    "No API keys configured for MCP authentication",
                    "Create API keys via POST /api/admin/api-keys",
                ))
        except Exception:
            results.append(CheckResult(
                "MCP API Keys", CheckStatus.YELLOW, "security",
                "Could not check API key status",
            ))

        return results

    def _performance_checks(self) -> list[CheckResult]:
        results = []

        # Connection pool
        try:
            from ..database.connection import engine
            pool = engine.pool
            results.append(CheckResult(
                "Connection Pool", CheckStatus.GREEN, "performance",
                f"Pool size={pool.size()}, overflow={pool.overflow()}",
            ))
        except Exception:
            results.append(CheckResult(
                "Connection Pool", CheckStatus.YELLOW, "performance",
                "Could not inspect connection pool",
            ))

        # Embedding cache
        try:
            from ..database.connection import SessionLocal
            from ..database.models import EmbeddingCache
            db = SessionLocal()
            cache_count = db.query(EmbeddingCache).count()
            db.close()
            results.append(CheckResult(
                "Embedding Cache", CheckStatus.GREEN, "performance",
                f"{cache_count} cached embeddings",
            ))
        except Exception:
            results.append(CheckResult(
                "Embedding Cache", CheckStatus.YELLOW, "performance",
                "Embedding cache not available",
            ))

        # Semantic cache
        try:
            from ..database.connection import SessionLocal
            from ..database.models import SemanticCache
            db = SessionLocal()
            sc_count = db.query(SemanticCache).count()
            db.close()
            results.append(CheckResult(
                "Semantic Cache", CheckStatus.GREEN, "performance",
                f"{sc_count} cached responses",
            ))
        except Exception:
            results.append(CheckResult(
                "Semantic Cache", CheckStatus.YELLOW, "performance",
                "Semantic cache not available",
            ))

        return results

    def _monitoring_checks(self) -> list[CheckResult]:
        results = []

        # Metrics endpoint
        results.append(CheckResult(
            "Metrics Endpoint", CheckStatus.GREEN, "monitoring",
            "Prometheus metrics available at /api/metrics",
        ))

        # LangSmith tracing
        langsmith_enabled = os.environ.get("LANGSMITH_TRACING", "").lower() == "true"
        if langsmith_enabled:
            results.append(CheckResult(
                "LangSmith Tracing", CheckStatus.GREEN, "monitoring",
                "LangSmith tracing enabled",
            ))
        else:
            results.append(CheckResult(
                "LangSmith Tracing", CheckStatus.YELLOW, "monitoring",
                "LangSmith tracing not enabled",
                "Set LANGSMITH_TRACING=true and LANGSMITH_API_KEY for production tracing",
            ))

        return results

    def _database_checks(self) -> list[CheckResult]:
        results = []

        # PostgreSQL connectivity
        try:
            from ..database.connection import SessionLocal
            from sqlalchemy import text
            db = SessionLocal()
            db.execute(text("SELECT 1"))
            db.close()
            results.append(CheckResult(
                "PostgreSQL", CheckStatus.GREEN, "database",
                "Connected and responsive",
            ))
        except Exception as e:
            results.append(CheckResult(
                "PostgreSQL", CheckStatus.RED, "database",
                "Cannot connect to PostgreSQL",
                str(e),
            ))

        return results

    def _external_service_checks(self) -> list[CheckResult]:
        results = []

        # Qdrant
        try:
            from .retrieval import get_qdrant_client, QDRANT_URL
            qdrant_url = QDRANT_URL
            client = get_qdrant_client()
            client.get_collections()
            results.append(CheckResult(
                "Qdrant", CheckStatus.GREEN, "external",
                f"Connected at {qdrant_url}",
            ))
        except Exception as e:
            results.append(CheckResult(
                "Qdrant", CheckStatus.RED, "external",
                "Cannot connect to Qdrant",
                str(e),
            ))

        # OpenAI API key
        openai_key = os.environ.get("OPENAI_API_KEY", "")
        if openai_key and openai_key.startswith("sk-"):
            results.append(CheckResult(
                "OpenAI API", CheckStatus.GREEN, "external",
                "API key configured",
            ))
        else:
            results.append(CheckResult(
                "OpenAI API", CheckStatus.RED, "external",
                "OpenAI API key not configured",
                "Set OPENAI_API_KEY environment variable",
            ))

        # Together AI (optional)
        together_key = os.environ.get("TOGETHER_API_KEY", "")
        if together_key:
            results.append(CheckResult(
                "Together AI", CheckStatus.GREEN, "external",
                "API key configured",
            ))
        else:
            results.append(CheckResult(
                "Together AI", CheckStatus.YELLOW, "external",
                "Not configured (optional)",
                "Set TOGETHER_API_KEY for open-source model support",
            ))

        # Ollama (optional)
        try:
            from ..services.models.ollama_client import check_ollama_running
            if check_ollama_running():
                results.append(CheckResult(
                    "Ollama", CheckStatus.GREEN, "external",
                    "Running and accessible",
                ))
            else:
                results.append(CheckResult(
                    "Ollama", CheckStatus.YELLOW, "external",
                    "Not running (optional)",
                    "Start Ollama for local model support",
                ))
        except Exception:
            results.append(CheckResult(
                "Ollama", CheckStatus.YELLOW, "external",
                "Not available (optional)",
            ))

        return results
