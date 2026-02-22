"""API key management for inter-agent authentication in StudyBuddy v14.

Provides API key generation and verification for MCP tools and
agent-to-agent (A2A) communication. Keys are stored as SHA-256
hashes in the database — the full key is only shown once at creation.

Key format: sb_<48 hex chars> (e.g., sb_a1b2c3d4...)
"""

import hashlib
import secrets
import logging
from datetime import datetime
from typing import Optional

from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

KEY_PREFIX = "sb_"


def generate_api_key() -> tuple[str, str, str]:
    """Generate a new API key.

    Returns:
        Tuple of (full_key, key_hash, key_prefix):
        - full_key: The complete API key to give to the caller (shown once)
        - key_hash: SHA-256 hash for database storage
        - key_prefix: First 8 chars for identification in logs/UI
    """
    random_part = secrets.token_hex(24)
    full_key = f"{KEY_PREFIX}{random_part}"
    key_hash = hashlib.sha256(full_key.encode()).hexdigest()
    key_prefix = full_key[:8]
    return full_key, key_hash, key_prefix


def verify_api_key(provided_key: str, db: Session) -> Optional["ApiKey"]:
    """Verify an API key and return the ApiKey record if valid.

    Checks:
    - Key exists in database (by hash)
    - Key is active
    - Key has not expired

    Updates last_used_at on successful verification.

    Args:
        provided_key: The full API key string to verify
        db: Database session

    Returns:
        The ApiKey record if valid, None otherwise.
    """
    from ..database.models import ApiKey

    key_hash = hashlib.sha256(provided_key.encode()).hexdigest()
    api_key = (
        db.query(ApiKey)
        .filter_by(key_hash=key_hash, is_active=True)
        .first()
    )

    if not api_key:
        return None

    # Check expiry
    if api_key.expires_at and api_key.expires_at < datetime.utcnow():
        logger.warning(f"Expired API key used: prefix={api_key.key_prefix}")
        return None

    # Update last used
    api_key.last_used_at = datetime.utcnow()
    db.commit()

    return api_key


def has_scope(api_key, scope: str) -> bool:
    """Check if an API key has a specific permission scope.

    Args:
        api_key: ApiKey database record
        scope: Required scope (e.g., "mcp:generate_flashcards")

    Returns:
        True if the key has the scope, or if scopes is empty (full access).
    """
    scopes = api_key.scopes or []
    # Empty scopes means full access
    if not scopes:
        return True
    return scope in scopes
