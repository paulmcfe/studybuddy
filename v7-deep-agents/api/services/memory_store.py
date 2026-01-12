"""Custom SQLAlchemy-based memory store.

This provides a simple memory interface without the langmem dependency.
Memories are organized by namespace for easy categorization.

Namespace conventions:
- "preferences": Learning style, favorite topics, etc.
- "struggles": Topics the user finds difficult
- "goals": What the user is studying for
- "sessions": Summaries of past study sessions
"""

import uuid
from datetime import datetime
from sqlalchemy.orm import Session

from ..database.models import Memory


class MemoryStore:
    """Simple memory store backed by SQLAlchemy.

    Usage:
        store = MemoryStore(db_session)
        store.put("default", "preferences", "learning_style", {"type": "visual"})
        memories = store.search("default", "preferences")
    """

    def __init__(self, db: Session):
        """Initialize with database session.

        Args:
            db: SQLAlchemy database session
        """
        self.db = db

    def put(
        self,
        user_id: str,
        namespace: str,
        key: str,
        value: dict,
    ) -> Memory:
        """Store or update a memory.

        Args:
            user_id: User identifier
            namespace: Category (e.g., "preferences", "struggles")
            key: Unique key within namespace
            value: Dictionary of data to store

        Returns:
            The created or updated Memory object
        """
        # Check for existing memory
        existing = (
            self.db.query(Memory)
            .filter_by(user_id=user_id, namespace=namespace, key=key)
            .first()
        )

        if existing:
            existing.value = value
            existing.updated_at = datetime.utcnow()
            self.db.commit()
            return existing
        else:
            memory = Memory(
                id=str(uuid.uuid4()),
                user_id=user_id,
                namespace=namespace,
                key=key,
                value=value,
            )
            self.db.add(memory)
            self.db.commit()
            return memory

    def get(
        self,
        user_id: str,
        namespace: str,
        key: str,
    ) -> dict | None:
        """Get a specific memory by key.

        Args:
            user_id: User identifier
            namespace: Category
            key: Memory key

        Returns:
            The memory value dict, or None if not found
        """
        memory = (
            self.db.query(Memory)
            .filter_by(user_id=user_id, namespace=namespace, key=key)
            .first()
        )
        return memory.value if memory else None

    def search(
        self,
        user_id: str,
        namespace: str | None = None,
        limit: int = 10,
    ) -> list[dict]:
        """Search memories in a namespace.

        Args:
            user_id: User identifier
            namespace: Optional namespace to filter by
            limit: Maximum results to return

        Returns:
            List of memory dicts with key and value
        """
        query = self.db.query(Memory).filter_by(user_id=user_id)

        if namespace:
            query = query.filter_by(namespace=namespace)

        query = query.order_by(Memory.updated_at.desc()).limit(limit)

        return [
            {
                "namespace": m.namespace,
                "key": m.key,
                "value": m.value,
                "updated_at": m.updated_at.isoformat() if m.updated_at else None,
            }
            for m in query.all()
        ]

    def delete(
        self,
        user_id: str,
        namespace: str,
        key: str,
    ) -> bool:
        """Delete a specific memory.

        Args:
            user_id: User identifier
            namespace: Category
            key: Memory key

        Returns:
            True if deleted, False if not found
        """
        memory = (
            self.db.query(Memory)
            .filter_by(user_id=user_id, namespace=namespace, key=key)
            .first()
        )

        if memory:
            self.db.delete(memory)
            self.db.commit()
            return True
        return False

    def list_namespaces(self, user_id: str) -> list[str]:
        """List all namespaces with memories for a user.

        Args:
            user_id: User identifier

        Returns:
            List of namespace strings
        """
        from sqlalchemy import distinct

        namespaces = (
            self.db.query(distinct(Memory.namespace))
            .filter_by(user_id=user_id)
            .all()
        )
        return [ns[0] for ns in namespaces]


def extract_memories_from_conversation(
    conversation_text: str,
    llm,
) -> list[dict]:
    """Extract memorable information from conversation text.

    Uses LLM to identify information worth remembering:
    - User preferences
    - Learning goals
    - Struggle areas
    - Important context

    Args:
        conversation_text: The conversation to analyze
        llm: Language model for extraction

    Returns:
        List of memory dicts with namespace, key, and value
    """
    from langchain_core.messages import SystemMessage, HumanMessage
    import json

    system_prompt = """Extract memorable information from this conversation that would be useful to remember for future tutoring sessions.

Look for:
- Learning preferences (visual learner, prefers examples, etc.)
- Goals (studying for certification, learning for job, etc.)
- Struggle areas (topics they find confusing)
- Background info (experience level, related knowledge)

Return a JSON array of memories to store:
[
    {
        "namespace": "preferences|goals|struggles|background",
        "key": "short_identifier",
        "value": {"description": "...", "context": "..."}
    }
]

If nothing memorable, return an empty array: []"""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Conversation:\n{conversation_text}"),
    ]

    response = llm.invoke(messages)

    try:
        content = response.content.strip()
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
            content = content.strip()
        return json.loads(content)
    except json.JSONDecodeError:
        return []
