"""MCP Client configuration for StudyBuddy v11.

Provides server configuration for each MCP connector type.
Each function returns the config dict expected by MultiServerMCPClient.

Three MCP servers, three packaging methods:
- Fetch: Python package via uvx (mcp-server-fetch)
- GitHub: Docker container (ghcr.io/github/github-mcp-server)
- Brave Search: npm package via npx (@brave/brave-search-mcp-server)
"""


def get_fetch_server_config() -> dict:
    """Config for the Fetch MCP server.

    Fetches web pages and converts HTML to markdown.
    Package: mcp-server-fetch (Anthropic, installed via pip/uvx)
    """
    return {
        "command": "uvx",
        "args": ["mcp-server-fetch"],
        "transport": "stdio",
    }


def get_github_server_config(token: str) -> dict:
    """Config for the GitHub MCP server.

    Provides repository browsing and file content retrieval.
    Package: github/github-mcp-server (GitHub official, Docker)

    Args:
        token: GitHub Personal Access Token
    """
    return {
        "command": "docker",
        "args": [
            "run", "-i", "--rm",
            "-e", f"GITHUB_PERSONAL_ACCESS_TOKEN={token}",
            "ghcr.io/github/github-mcp-server",
        ],
        "transport": "stdio",
    }


def get_brave_search_server_config(api_key: str) -> dict:
    """Config for the Brave Search MCP server.

    Provides web search, news search, and summarization.
    Package: @brave/brave-search-mcp-server (Brave, npm)

    Args:
        api_key: Brave Search API key
    """
    return {
        "command": "npx",
        "args": ["-y", "@brave/brave-search-mcp-server"],
        "transport": "stdio",
        "env": {
            "BRAVE_API_KEY": api_key,
        },
    }
