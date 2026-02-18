"""MCP Server for StudyBuddy v13.

Exposes StudyBuddy capabilities as MCP tools for external agents.
"""

from .server import mcp, create_mcp_app
from .agent_card import STUDYBUDDY_AGENT_CARD, AgentCard

__all__ = [
    "mcp",
    "create_mcp_app",
    "STUDYBUDDY_AGENT_CARD",
    "AgentCard",
]
