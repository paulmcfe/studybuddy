"""Agent package for StudyBuddy v11.

v11 adds the tutor agent with optional Brave Search web search capability.
"""

from .tutor import create_tutor_with_search

__all__ = ["create_tutor_with_search"]
