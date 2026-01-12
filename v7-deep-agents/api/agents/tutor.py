"""Tutor Agent - Explains concepts conversationally.

The Tutor is the friendly expert who helps students understand
difficult material. It has access to the knowledge base and is
optimized for clear, educational explanations.
"""

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

TUTOR_PROMPT = """You are StudyBuddy's Tutor, an expert at explaining
AI engineering concepts clearly and engagingly.

Your job: Help students understand concepts from the reference guides.

Guidelines:
- Use clear, conversational language
- Start with the big picture, then details
- Use analogies to make abstract concepts concrete
- Check understanding by asking follow-up questions
- Reference specific sources when explaining
- Keep explanations focused and digestible

When given context from the knowledge base, use it to provide accurate,
sourced explanations. If no context is provided, use your general knowledge
but be clear about what's from the reference materials vs general knowledge."""


def create_tutor_agent(model_name: str = "gpt-4o"):
    """Create the Tutor agent."""
    return ChatOpenAI(model=model_name, temperature=0.7)


def tutor_explain(
    llm: ChatOpenAI,
    query: str,
    context: str = "",
    card_context: dict | None = None,
) -> str:
    """
    Have the Tutor explain a concept.

    Args:
        llm: The language model to use
        query: The student's question
        context: Retrieved context from knowledge base
        card_context: Current flashcard context if studying a card

    Returns:
        The Tutor's explanation
    """
    # Build the system message
    system_content = TUTOR_PROMPT

    # Add card context if present
    if card_context:
        system_content += f"""

The student is currently studying a flashcard:
- Topic: {card_context.get('topic', 'Unknown')}
- Question: {card_context.get('question', '')}
- Answer: {card_context.get('answer', '')}

Focus your explanation on helping them understand this specific concept better."""

    # Build the user message
    user_content = f"Student question: {query}"

    if context:
        user_content += f"""

Reference material:
{context}

Use this reference material to provide an accurate, sourced explanation."""

    messages = [
        SystemMessage(content=system_content),
        HumanMessage(content=user_content),
    ]

    response = llm.invoke(messages)
    return response.content
