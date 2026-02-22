"""Audit logging service for StudyBuddy v14.

Records security-relevant events to the database for compliance
and debugging. Audit log entries are immutable — they are only
ever inserted, never updated or deleted.

Events include: authentication, data access, MCP tool invocations,
guardrail triggers, and admin actions.
"""

import logging
from typing import Optional

from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


class AuditLogger:
    """Records audit events to the database."""

    def log(
        self,
        db: Session,
        *,
        action: str,
        user_id: Optional[str] = None,
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
        source_ip: Optional[str] = None,
        user_agent: Optional[str] = None,
        result: str = "success",
        details: Optional[dict] = None,
    ):
        """Record an audit event.

        Args:
            db: Database session
            action: What happened (e.g., "auth.login", "mcp.generate_flashcards")
            user_id: Who triggered it (None for system events)
            resource_type: Type of resource affected (e.g., "program", "document")
            resource_id: ID of the affected resource
            source_ip: Client IP address
            user_agent: Client user-agent string
            result: Outcome — "success", "failure", or "blocked"
            details: Additional context as JSON
        """
        from ..database.models import AuditLog

        entry = AuditLog(
            user_id=user_id,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            source_ip=source_ip,
            user_agent=user_agent,
            result=result,
            details=details,
        )
        db.add(entry)
        try:
            db.commit()
        except Exception:
            db.rollback()
            logger.error(f"Failed to write audit log: action={action}", exc_info=True)

    def log_from_request(
        self,
        db: Session,
        request,
        *,
        action: str,
        user_id: Optional[str] = None,
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
        result: str = "success",
        details: Optional[dict] = None,
    ):
        """Record an audit event, extracting IP and user-agent from a FastAPI Request."""
        source_ip = request.client.host if request.client else None
        user_agent = request.headers.get("user-agent", "")[:500]

        self.log(
            db,
            action=action,
            user_id=user_id,
            resource_type=resource_type,
            resource_id=resource_id,
            source_ip=source_ip,
            user_agent=user_agent,
            result=result,
            details=details,
        )


# Singleton instance
audit_logger = AuditLogger()
