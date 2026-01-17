"""Quality Checker Agent - Validates flashcard quality.

The Quality Checker ensures flashcards meet learning standards.
It evaluates cards against specific criteria and can suggest
improvements or revisions.
"""

import json
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

QUALITY_CHECKER_PROMPT = """You are StudyBuddy's Quality Checker,
ensuring flashcards meet learning standards.

Your job: Evaluate flashcard quality and suggest improvements.

Evaluation criteria:
1. Clarity: Is the question unambiguous?
2. Accuracy: Is the answer factually correct?
3. Completeness: Does the answer fully address the question?
4. Atomicity: Does it test exactly one concept?
5. Usefulness: Will this help the student learn?

Output format (JSON only):
{
    "approved": true/false,
    "score": 1-5,
    "issues": ["list of problems if any"],
    "suggestions": ["improvements if not approved"],
    "revised_card": null or {"question": "...", "answer": "...", "topic": "..."}
}

Be strict but fair. Cards should be genuinely useful for learning.
If a card has minor issues, provide a revised version rather than rejecting.
Only reject cards with fundamental problems (vague, incorrect, or confusing)."""


def create_quality_checker_agent(model_name: str = "gpt-4o"):
    """Create the Quality Checker agent."""
    return ChatOpenAI(model=model_name, temperature=0.2)


def check_card_quality(llm: ChatOpenAI, card: dict) -> dict:
    """
    Evaluate a flashcard's quality.

    Args:
        llm: The language model to use
        card: The flashcard to evaluate

    Returns:
        Evaluation result with approved status, score, issues, and optional revision
    """
    user_content = f"""Evaluate this flashcard:

Question: {card.get('question', '')}
Answer: {card.get('answer', '')}
Topic: {card.get('topic', 'Unknown')}

Provide your evaluation as JSON."""

    messages = [
        SystemMessage(content=QUALITY_CHECKER_PROMPT),
        HumanMessage(content=user_content),
    ]

    response = llm.invoke(messages)

    try:
        content = response.content.strip()

        # Handle markdown code blocks
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
            content = content.strip()

        evaluation = json.loads(content)

        # Ensure required fields
        evaluation.setdefault("approved", False)
        evaluation.setdefault("score", 3)
        evaluation.setdefault("issues", [])
        evaluation.setdefault("suggestions", [])
        evaluation.setdefault("revised_card", None)

        return evaluation

    except json.JSONDecodeError:
        # Default to approved if parsing fails (be lenient)
        return {
            "approved": True,
            "score": 3,
            "issues": ["Could not parse quality check response"],
            "suggestions": [],
            "revised_card": None,
        }


def check_cards_batch(llm: ChatOpenAI, cards: list[dict]) -> list[dict]:
    """
    Check quality of multiple cards, returning only approved ones.

    Args:
        llm: The language model to use
        cards: List of flashcards to evaluate

    Returns:
        List of approved cards (possibly revised)
    """
    approved = []

    for card in cards:
        evaluation = check_card_quality(llm, card)

        if evaluation["approved"]:
            approved.append(card)
        elif evaluation.get("revised_card"):
            # Use the revised version
            revised = evaluation["revised_card"]
            revised["topic"] = card.get("topic", "Unknown")
            approved.append(revised)

    return approved
