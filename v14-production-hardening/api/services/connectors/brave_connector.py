"""Brave Search Connector for StudyBuddy v11.

Augments the tutoring agent with web search capability using the
Brave Search MCP server. Unlike the Fetch and GitHub connectors,
this does NOT import documents — it provides a real-time search tool
that the tutor agent can use during conversations.

When the tutor can't find sufficient context in uploaded documents,
it searches the web and includes relevant results (with citations)
in its response.

MCP Server: @brave/brave-search-mcp-server (Brave, npm/npx)
Transport: stdio
"""

import logging
from typing import Optional

from .mcp_client import get_brave_search_server_config

logger = logging.getLogger(__name__)


async def get_brave_search_tools(api_key: str) -> tuple[list, any]:
    """Get Brave Search tools as LangChain tools.

    Creates a MultiServerMCPClient connected to the Brave Search
    MCP server and returns the available tools.

    Args:
        api_key: Brave Search API key

    Returns:
        Tuple of (tools_list, mcp_client).
    """
    from langchain_mcp_adapters.client import MultiServerMCPClient

    client = MultiServerMCPClient(
        {"brave": get_brave_search_server_config(api_key)}
    )
    tools = await client.get_tools()

    # Filter for the web search tool specifically
    search_tools = [
        t for t in tools
        if "web_search" in t.name.lower() or "brave" in t.name.lower()
    ]

    if not search_tools:
        logger.warning(
            f"No search tools found. Available tools: {[t.name for t in tools]}"
        )
        # Return all tools if we can't identify the search tool
        search_tools = list(tools)

    logger.info(
        f"Loaded {len(search_tools)} Brave Search tools: "
        f"{[t.name for t in search_tools]}"
    )

    return search_tools, client
