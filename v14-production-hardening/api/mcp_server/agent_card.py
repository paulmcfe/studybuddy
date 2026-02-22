"""Agent Card for A2A (Agent-to-Agent) discovery.

Provides a standardized way for other agents to discover
StudyBuddy's capabilities. Based on AgentCard pattern from chapter13.ipynb.
"""

from pydantic import BaseModel


class AgentCard(BaseModel):
    """Agent capability advertisement for A2A discovery.

    This standardized format allows other agents to:
    - Discover what capabilities StudyBuddy offers
    - Understand supported modalities
    - Find the endpoint for communication
    - Know authentication requirements
    """

    name: str
    description: str
    version: str
    capabilities: list[str]
    supported_modalities: list[str]
    endpoint: str
    authentication: dict
    metadata: dict = {}


# StudyBuddy's agent card for discovery
STUDYBUDDY_AGENT_CARD = AgentCard(
    name="StudyBuddy",
    description="AI-powered learning assistant with flashcard generation, tutoring, curriculum management, and knowledge search capabilities",
    version="14.0.0",
    capabilities=[
        "generate_flashcards",
        "explain_concept",
        "get_curriculum",
        "search_knowledge",
    ],
    supported_modalities=["text"],
    endpoint="mcp://studybuddy/v14",
    authentication={
        "type": "api_key",
        "required": True,
        "header": "X-API-Key",
        "description": "API key for inter-agent authentication. Obtain via POST /api/admin/api-keys",
    },
    metadata={
        "tags": ["education", "learning", "flashcards", "tutoring", "ai"],
        "documentation": "https://github.com/studybuddy/docs",
        "supported_providers": ["openai", "together", "ollama"],
        "features": {
            "multi_model": True,
            "fallback_chains": True,
            "cost_tracking": True,
            "spaced_repetition": True,
        },
    },
)


def get_agent_card_dict() -> dict:
    """Get agent card as dictionary for API response."""
    return STUDYBUDDY_AGENT_CARD.model_dump()
