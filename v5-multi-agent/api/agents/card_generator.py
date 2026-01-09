"""Card Generator Agent - Creates flashcards from explanations and materials.

The Card Generator is specialized in creating effective flashcards
that help students remember key concepts. It outputs structured JSON
for easy parsing.
"""

import json
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

CARD_GENERATOR_PROMPT = """You are StudyBuddy's Card Generator,
specialized in creating effective flashcards.

Your job: Create flashcards that help students remember key concepts.

Guidelines for good flashcards:
- One concept per card (atomic)
- Questions should be specific and unambiguous
- Answers should be concise but complete
- Use active recall (questions that require thinking)
- Include context when helpful
- Vary question types: definition, comparison, application
- Avoid yes/no questions

Output format - respond with ONLY a JSON array:
[
    {
        "question": "Clear, specific question",
        "answer": "Concise, complete answer",
        "topic": "Main topic or concept",
        "difficulty": "basic|intermediate|advanced"
    }
]

Generate 1-3 high-quality flashcards per request."""


def create_card_generator_agent(model_name: str = "gpt-4o-mini"):
    """Create the Card Generator agent."""
    return ChatOpenAI(model=model_name, temperature=0.3)


def generate_cards(
    llm: ChatOpenAI,
    topic: str,
    context: str = "",
    explanation: str = "",
) -> list[dict]:
    """
    Generate flashcards for a topic.

    Args:
        llm: The language model to use
        topic: The topic to create cards for
        context: Retrieved context from knowledge base
        explanation: Recent explanation from Tutor (if any)

    Returns:
        List of flashcard dictionaries
    """
    user_content = f"Create flashcards for: {topic}"

    if explanation:
        user_content += f"""

Based on this explanation:
{explanation}"""

    if context:
        user_content += f"""

Reference material:
{context}"""

    messages = [
        SystemMessage(content=CARD_GENERATOR_PROMPT),
        HumanMessage(content=user_content),
    ]

    response = llm.invoke(messages)

    # Parse the JSON response
    try:
        content = response.content.strip()

        # Handle markdown code blocks
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
            content = content.strip()

        cards = json.loads(content)

        # Ensure it's a list
        if isinstance(cards, dict):
            cards = [cards]

        return cards

    except json.JSONDecodeError:
        # Return empty list on parse failure
        return []


def generate_single_card(
    llm: ChatOpenAI,
    topic: str,
    subtopics: list[str],
    context: str = "",
) -> dict | None:
    """
    Generate a single flashcard for a topic.

    Args:
        llm: The language model to use
        topic: The main topic
        subtopics: List of subtopics to consider
        context: Retrieved context from knowledge base

    Returns:
        Single flashcard dictionary or None
    """
    prompt = f"""Create ONE high-quality flashcard for the topic: {topic}

Subtopics to consider: {', '.join(subtopics) if subtopics else 'General concepts'}

{f'Reference material:{chr(10)}{context}' if context else 'Use your knowledge of AI engineering.'}

Create ONE flashcard that:
- Tests understanding of a key concept (not just recall)
- Has a clear, unambiguous question
- Has a concise but complete answer
- Avoids yes/no questions

Respond with JSON only:
{{"question": "...", "answer": "..."}}"""

    messages = [
        SystemMessage(content="You are a flashcard generator. Output only valid JSON."),
        HumanMessage(content=prompt),
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

        card = json.loads(content)
        card["topic"] = topic
        return card

    except json.JSONDecodeError:
        return None
