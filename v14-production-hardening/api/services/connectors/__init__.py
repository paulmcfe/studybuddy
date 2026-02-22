"""MCP Connectors package for StudyBuddy v11.

Provides three connector types:
- Fetch: Import web pages as learning materials via URL
- GitHub: Import markdown/docs from GitHub repositories
- Brave Search: Augment the tutor with web search capability
"""

from .fetch_connector import import_url
from .github_connector import list_repo_files, import_github_files
from .brave_connector import get_brave_search_tools

__all__ = [
    "import_url",
    "list_repo_files",
    "import_github_files",
    "get_brave_search_tools",
]
