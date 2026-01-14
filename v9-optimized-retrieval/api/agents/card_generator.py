"""Card Generator Agent - Creates flashcards from explanations and materials.

The Card Generator is specialized in creating effective flashcards
that help students remember key concepts. It outputs structured JSON
for easy parsing.
"""

import json
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

CARD_GENERATOR_PROMPT = """You are StudyBuddy's Card Generator,
specialized in creating focused, recallable flashcards.

CRITICAL RULES:
1. ONE testable fact per card - never combine multiple questions
2. Answer must be 1-3 sentences maximum (under 200 characters preferred)
3. Question cannot contain "and" connecting two different concepts
4. This is quick recall practice - avoid essay-length explanations

BAD EXAMPLES (compound questions - never do this):
- "What is X and what are its advantages?"
- "Explain the role of A and B in the system"
- "What are the key benefits and drawbacks?"

GOOD EXAMPLES (atomic, focused):
- "What is the primary purpose of X?"
- "What problem does X solve?"
- "How does X differ from Y?"

Question types to use:
- "What is..." (definition)
- "What happens when..." (behavior)
- "When should you use X?" (application)
- "What problem does X solve?" (purpose)

NEVER ask questions with "and" that combine:
- Definition AND explanation
- Advantages AND disadvantages
- Multiple components in one question

Output format - respond with ONLY a JSON array:
[
    {
        "question": "Clear, specific question testing ONE thing",
        "answer": "1-3 sentence answer, under 200 characters if possible",
        "topic": "Main topic or concept",
        "difficulty": "basic|intermediate|advanced"
    }
]

Generate 3-5 atomic flashcards per request."""


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


def generate_cards_batch(
    llm: ChatOpenAI,
    topic: str,
    context: str = "",
    count: int = 6,
) -> list[dict]:
    """
    Generate a specific number of flashcards for background prefetching.

    Args:
        llm: The language model to use
        topic: The topic to create cards for
        context: Retrieved context from knowledge base
        count: Target number of cards (default 6, for 5-7 range)

    Returns:
        List of flashcard dictionaries
    """
    prompt = f"""Create exactly {count} high-quality flashcards for: {topic}

{f'Reference material:{chr(10)}{context}' if context else 'Use your knowledge.'}

CRITICAL RULES:
1. Generate exactly {count} cards
2. ONE testable fact per card - never combine multiple questions
3. Answer must be 1-3 sentences (under 200 characters preferred)
4. Question cannot contain "and" connecting two concepts
5. Cover different aspects of the topic for variety

Output format - respond with ONLY a JSON array:
[{{"question": "...", "answer": "...", "difficulty": "basic|intermediate|advanced"}}]"""

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

        cards = json.loads(content)

        # Ensure it's a list
        if isinstance(cards, dict):
            cards = [cards]

        # Add topic to each card
        for card in cards:
            card["topic"] = topic

        return cards

    except json.JSONDecodeError:
        return []


def generate_single_card(
    llm: ChatOpenAI,
    topic: str,
    subtopics: list[str],
    context: str = "",
    previous_question: str = "",
) -> dict | None:
    """
    Generate a single flashcard for a topic.

    Args:
        llm: The language model to use
        topic: The main topic
        subtopics: List of subtopics to consider
        context: Retrieved context from knowledge base
        previous_question: Previous question to avoid repeating (for "Still Learning" flow)

    Returns:
        Single flashcard dictionary or None
    """
    # Build "avoid this question" instruction for Still Learning mode
    avoid_instruction = ""
    if previous_question:
        avoid_instruction = f"""
IMPORTANT: The student just studied this question and wants a DIFFERENT one:
Previous question: "{previous_question}"
Generate a completely different question about a different aspect of this topic.
"""

    prompt = f"""Create ONE high-quality flashcard for the topic: {topic}

Subtopics to consider: {', '.join(subtopics) if subtopics else 'General concepts'}
{avoid_instruction}
{f'Reference material:{chr(10)}{context}' if context else 'Use your knowledge of AI engineering.'}

CRITICAL RULES:
1. ONE testable fact per card - never combine multiple questions
2. Answer must be 1-3 sentences maximum (under 200 characters preferred)
3. Question cannot contain "and" connecting two different concepts
4. This is quick recall practice - avoid essay-length explanations

BAD EXAMPLES (compound questions - never do this):
- "What is X and what are its advantages?"
- "Explain the role of A and B in the system"
- "What are the key benefits and drawbacks?"

GOOD EXAMPLES (atomic, focused):
- "What is the primary purpose of X?"
- "What problem does X solve?"
- "How does X differ from Y?"

Question types to use:
- "What is..." (definition)
- "What happens when..." (behavior)
- "When should you use X?" (application)
- "What problem does X solve?" (purpose)

NEVER ask questions with "and" that combine:
- Definition AND explanation
- Advantages AND disadvantages
- Multiple components in one question

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
