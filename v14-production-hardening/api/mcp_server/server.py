"""FastMCP server exposing StudyBuddy capabilities.

Provides MCP tools for external agents to:
- Generate flashcards for topics
- Get tutoring explanations
- Retrieve curriculum structure
- Search indexed knowledge

Based on FastMCP patterns from chapter13.ipynb.
"""

import json
import logging
import os
from typing import Optional

from mcp.server.fastmcp import FastMCP

logger = logging.getLogger(__name__)

# Initialize FastMCP server
mcp = FastMCP("StudyBuddy")


def _get_db_session():
    """Get a database session for MCP tools."""
    from ..database.connection import SessionLocal

    return SessionLocal()


def _get_retriever(program_id: str):
    """Get retriever for a program's knowledge base."""
    from ..services.retrieval import get_retriever

    return get_retriever(f"program_{program_id}")


def _verify_and_log(api_key: str, tool_name: str):
    """Verify an API key and log the access attempt.

    Args:
        api_key: The API key to verify
        tool_name: Name of the tool being accessed

    Returns:
        Tuple of (key_record, error_json). If valid, error_json is None.
        If invalid, key_record is None and error_json contains the error.
    """
    from ..services.api_keys import verify_api_key
    from ..services.audit import audit_logger

    db = _get_db_session()
    try:
        key_record = verify_api_key(api_key, db)
        if not key_record:
            audit_logger.log(
                db,
                action=f"mcp.{tool_name}",
                result="failure",
                details={"reason": "invalid_api_key"},
            )
            return (None, json.dumps({"error": "Invalid or expired API key"}))

        audit_logger.log(
            db,
            action=f"mcp.{tool_name}",
            result="success",
            details={"key_prefix": key_record.key_prefix},
        )
        return (key_record, None)
    finally:
        db.close()


@mcp.tool()
async def generate_flashcards(
    api_key: str,
    topic: str,
    program_id: str,
    count: int = 5,
) -> str:
    """Generate flashcards for a topic in a learning program.

    Args:
        api_key: API key for authentication
        topic: The topic to generate flashcards for
        program_id: ID of the learning program
        count: Number of flashcards to generate (default: 5)

    Returns:
        JSON string with generated flashcards
    """
    key, error = _verify_and_log(api_key, "generate_flashcards")
    if error:
        return error

    from ..services.flashcard import generate_flashcard
    from ..services.retrieval import get_retriever
    from ..database.models import LearningProgram

    db = _get_db_session()
    try:
        # Get program info
        program = db.query(LearningProgram).filter(
            LearningProgram.id == program_id
        ).first()

        if not program:
            return json.dumps({"error": f"Program {program_id} not found"})

        # Get retriever and context
        retriever = get_retriever(program.qdrant_collection)

        # Generate flashcards
        cards = []
        for i in range(count):
            # Retrieve context for the topic
            docs = retriever.invoke(f"{topic} concept {i+1}")
            context = "\n\n".join([doc.page_content for doc in docs[:3]])

            if not context:
                continue

            card = await generate_flashcard(
                topic=topic,
                context=context,
                program_id=program_id,
                db=db,
                program_name=program.name,
                program_description=program.description or "",
            )

            if card:
                cards.append({
                    "id": card.id,
                    "topic": card.topic,
                    "question": card.question,
                    "answer": card.answer,
                })

        return json.dumps({
            "flashcards": cards,
            "count": len(cards),
            "program_id": program_id,
            "topic": topic,
        })

    finally:
        db.close()


@mcp.tool()
async def explain_concept(
    api_key: str,
    concept: str,
    program_id: str,
    detail_level: str = "intermediate",
) -> str:
    """Get a tutoring explanation for a concept.

    Args:
        api_key: API key for authentication
        concept: The concept to explain
        program_id: ID of the learning program for context
        detail_level: Level of detail (beginner, intermediate, advanced)

    Returns:
        JSON string with explanation
    """
    key, error = _verify_and_log(api_key, "explain_concept")
    if error:
        return error

    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import ChatPromptTemplate
    from ..database.models import LearningProgram

    db = _get_db_session()
    try:
        # Get program info
        program = db.query(LearningProgram).filter(
            LearningProgram.id == program_id
        ).first()

        if not program:
            return json.dumps({"error": f"Program {program_id} not found"})

        # Get context from knowledge base
        retriever = _get_retriever(program_id)
        docs = retriever.invoke(concept)
        context = "\n\n".join([doc.page_content for doc in docs[:5]])

        # Create explanation prompt
        prompt = ChatPromptTemplate.from_template("""You are an expert tutor explaining concepts from the learning program: {program_name}

Context from the knowledge base:
{context}

Explain the following concept at a {detail_level} level:
{concept}

Provide a clear, educational explanation that helps the learner understand this concept.
Include examples where appropriate.
""")

        llm = ChatOpenAI(model="gpt-4o", temperature=0.7)
        chain = prompt | llm

        result = await chain.ainvoke({
            "program_name": program.name,
            "context": context,
            "detail_level": detail_level,
            "concept": concept,
        })

        return json.dumps({
            "concept": concept,
            "explanation": result.content,
            "detail_level": detail_level,
            "program_id": program_id,
        })

    finally:
        db.close()


@mcp.tool()
async def get_curriculum(api_key: str, program_id: str) -> str:
    """Retrieve the curriculum structure for a learning program.

    Args:
        api_key: API key for authentication
        program_id: ID of the learning program

    Returns:
        JSON string with curriculum structure
    """
    key, error = _verify_and_log(api_key, "get_curriculum")
    if error:
        return error

    from ..database.models import LearningProgram

    db = _get_db_session()
    try:
        program = db.query(LearningProgram).filter(
            LearningProgram.id == program_id
        ).first()

        if not program:
            return json.dumps({"error": f"Program {program_id} not found"})

        topic_list = program.topic_list or {}
        chapters = topic_list.get("chapters", [])

        # Count topics
        topic_count = 0
        for chapter in chapters:
            for topic in chapter.get("topics", []):
                topic_count += 1
                topic_count += len(topic.get("subtopics", []))

        return json.dumps({
            "program_id": program_id,
            "program_name": program.name,
            "curriculum": topic_list,
            "chapter_count": len(chapters),
            "topic_count": topic_count,
        })

    finally:
        db.close()


@mcp.tool()
async def search_knowledge(
    api_key: str,
    query: str,
    program_id: str,
    max_results: int = 5,
) -> str:
    """Search indexed learning materials for relevant content.

    Args:
        api_key: API key for authentication
        query: Search query
        program_id: ID of the learning program
        max_results: Maximum number of results (default: 5)

    Returns:
        JSON string with search results
    """
    key, error = _verify_and_log(api_key, "search_knowledge")
    if error:
        return error

    from ..database.models import LearningProgram

    db = _get_db_session()
    try:
        program = db.query(LearningProgram).filter(
            LearningProgram.id == program_id
        ).first()

        if not program:
            return json.dumps({"error": f"Program {program_id} not found"})

        # Search knowledge base
        retriever = _get_retriever(program_id)
        docs = retriever.invoke(query)[:max_results]

        results = []
        for doc in docs:
            results.append({
                "content": doc.page_content,
                "metadata": doc.metadata,
            })

        return json.dumps({
            "query": query,
            "program_id": program_id,
            "results": results,
            "count": len(results),
        })

    finally:
        db.close()


@mcp.resource("studybuddy://status")
def get_status() -> str:
    """Get StudyBuddy server status and version information."""
    from ..services.models.ollama_client import get_ollama_status
    from ..services.models.together_client import get_together_status

    return json.dumps({
        "service": "StudyBuddy",
        "version": "14.0.0",
        "capabilities": [
            "generate_flashcards",
            "explain_concept",
            "get_curriculum",
            "search_knowledge",
        ],
        "providers": {
            "ollama": get_ollama_status(),
            "together": get_together_status(),
            "openai": {"configured": bool(os.environ.get("OPENAI_API_KEY"))},
        },
    })


@mcp.resource("studybuddy://models")
def get_available_models() -> str:
    """Get list of available models and their status."""
    from ..services.models.config import AVAILABLE_MODELS
    from ..services.models.router import ModelRouter

    router = ModelRouter()
    models = router.get_available_models()

    return json.dumps({
        "models": models,
        "count": len(models),
    })


def create_mcp_app():
    """Create and return the MCP application for deployment."""
    return mcp


if __name__ == "__main__":
    # Run the MCP server standalone
    mcp.run()
