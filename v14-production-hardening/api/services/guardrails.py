"""Content guardrails for StudyBuddy v14.

Provides input validation (prompt injection detection, inappropriate content
filtering) and output filtering (age-appropriate explanations, educational safety).

Guardrails operate at two levels:
1. API level: Input checked before reaching the LLM (fast, regex-based)
2. LangGraph level: Guardrail nodes wrap the agent graph for deeper integration
"""

import re
import logging
from enum import Enum
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


class GuardrailResult(Enum):
    PASS = "pass"
    BLOCKED = "blocked"
    MODIFIED = "modified"


@dataclass
class GuardrailCheckResult:
    result: GuardrailResult
    original_text: str
    filtered_text: Optional[str] = None
    reason: Optional[str] = None
    category: Optional[str] = None


class InputGuardrail:
    """Validates user input before it reaches the LLM.

    Checks for:
    - Prompt injection attempts (trying to override system instructions)
    - Inappropriate content for an educational context
    """

    # Prompt injection patterns — attempts to override system instructions
    INJECTION_PATTERNS = [
        r"ignore\s+(all\s+)?(previous|prior|above)\s+(instructions|prompts|rules)",
        r"forget\s+(all\s+)?(your|previous)\s+(instructions|rules|training)",
        r"system\s*prompt\s*[:=]",
        r"<\s*/?system\s*>",
        r"override\s+(your\s+)?(safety|content|system)",
        r"disregard\s+(all\s+)?(previous|prior|safety)",
        r"you\s+are\s+now\s+(a|an)\s+(?!student)",
        r"pretend\s+(that\s+)?you\s+(are|have)\s+no\s+(rules|restrictions|limits)",
        r"jailbreak",
        r"\bDAN\s+mode\b",
        r"do\s+anything\s+now",
        r"bypass\s+(your\s+)?(filters|safety|restrictions|guardrails)",
    ]

    # Content inappropriate for an educational tutoring context
    INAPPROPRIATE_PATTERNS = [
        r"\b(how\s+to\s+(make|build|create)\s+(a\s+)?(bomb|weapon|explosive))\b",
        r"\b(how\s+to\s+(hack|crack|break\s+into))\b",
        r"\b(generate\s+(explicit|pornographic|adult)\s+content)\b",
        r"\b(self[- ]harm|suicide\s+method)\b",
    ]

    def __init__(self):
        self._injection_compiled = [
            re.compile(p, re.IGNORECASE) for p in self.INJECTION_PATTERNS
        ]
        self._inappropriate_compiled = [
            re.compile(p, re.IGNORECASE) for p in self.INAPPROPRIATE_PATTERNS
        ]

    def check(self, text: str) -> GuardrailCheckResult:
        """Check user input for prompt injection and inappropriate content.

        Returns a GuardrailCheckResult indicating whether the input is safe.
        """
        # Check prompt injection
        for pattern in self._injection_compiled:
            if pattern.search(text):
                logger.warning(f"Prompt injection detected: pattern={pattern.pattern!r}")
                return GuardrailCheckResult(
                    result=GuardrailResult.BLOCKED,
                    original_text=text,
                    reason="Your message was flagged as a potential prompt injection attempt. Please rephrase your question about the study material.",
                    category="prompt_injection",
                )

        # Check inappropriate content
        for pattern in self._inappropriate_compiled:
            if pattern.search(text):
                logger.warning(f"Inappropriate content detected: pattern={pattern.pattern!r}")
                return GuardrailCheckResult(
                    result=GuardrailResult.BLOCKED,
                    original_text=text,
                    reason="Your message contains content that isn't appropriate for an educational context. Please ask a question related to your study materials.",
                    category="inappropriate_content",
                )

        return GuardrailCheckResult(
            result=GuardrailResult.PASS,
            original_text=text,
        )


class OutputGuardrail:
    """Filters LLM output to ensure age-appropriate educational content.

    Rather than blocking entire responses, this redacts specific problematic
    portions while preserving the educational content.
    """

    # Patterns that should not appear in tutoring output
    UNSAFE_OUTPUT_PATTERNS = [
        (re.compile(r"(?:instructions?\s+to\s+)?(?:build|make|create)\s+(?:a\s+)?(?:bomb|weapon|explosive)[^.]*\.", re.IGNORECASE), "[content removed for safety]"),
        (re.compile(r"(?:step[s-]*by[- ]step\s+)?(?:guide|instructions?)\s+(?:for|to|on)\s+(?:hack|crack|exploit)[^.]*\.", re.IGNORECASE), "[content removed for safety]"),
    ]

    def filter(self, text: str) -> GuardrailCheckResult:
        """Filter LLM output for safety.

        Redacts problematic portions rather than blocking the entire response.
        """
        filtered = text
        was_modified = False

        for pattern, replacement in self.UNSAFE_OUTPUT_PATTERNS:
            if pattern.search(filtered):
                filtered = pattern.sub(replacement, filtered)
                was_modified = True

        if was_modified:
            logger.warning("Output guardrail modified LLM response")
            return GuardrailCheckResult(
                result=GuardrailResult.MODIFIED,
                original_text=text,
                filtered_text=filtered,
                reason="Output contained content filtered for educational safety.",
                category="output_safety",
            )

        return GuardrailCheckResult(
            result=GuardrailResult.PASS,
            original_text=text,
            filtered_text=text,
        )


# Singleton instances
input_guardrail = InputGuardrail()
output_guardrail = OutputGuardrail()
