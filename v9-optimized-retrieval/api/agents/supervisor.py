"""Supervisor Agent - Orchestrates the multi-agent team.

The Learning Coordinator (Supervisor) decides which agents to engage
based on what the student needs, managing the workflow between
tutoring, card generation, quality checking, spaced repetition,
and curriculum planning.
"""

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
import json

SUPERVISOR_PROMPT = """You are StudyBuddy's Learning Coordinator,
orchestrating a team of specialized agents.

Your team:
- TUTOR: Explains concepts conversationally. Use for questions,
  confusion, or requests to learn about topics.
- CARD_GENERATOR: Creates flashcards. Use after tutoring explains
  something the student should remember.
- QUALITY_CHECKER: Validates flashcards. Always use after generating
  cards before showing to student.
- SCHEDULER: Manages spaced repetition. Use when student wants to
  practice, review due cards, or check their study progress/stats.
- CURRICULUM_PLANNER: Creates learning paths, tracks curriculum progress,
  handles module checkpoints. Use when student wants to create a study
  plan, resume their curriculum, or check curriculum progress.

Workflow patterns:
1. Learning mode: question → TUTOR explains → CARD_GENERATOR creates →
   QUALITY_CHECKER validates → respond with explanation and cards
2. Practice mode: "practice", "review", "quiz me" → SCHEDULER gets due cards
3. Stats mode: "how am I doing", "my progress", "stats" → SCHEDULER
4. Curriculum mode: "create curriculum", "learning path", "study plan",
   "resume curriculum", "continue my curriculum" → CURRICULUM_PLANNER
5. Direct response: Simple greetings, clarifications, or meta-questions

Decision rules:
- Questions about concepts → TUTOR first
- "Create flashcards for X" → CARD_GENERATOR (then QUALITY_CHECKER)
- After TUTOR explains a key concept → consider CARD_GENERATOR
- "Practice", "review", "quiz me", "due cards" → SCHEDULER
- "Progress", "stats", "how am I doing" → SCHEDULER
- "Create a study plan for X", "learning path for X" → CURRICULUM_PLANNER
- "Resume studying", "continue my curriculum" → CURRICULUM_PLANNER
- "Curriculum progress", "how's my learning path" → CURRICULUM_PLANNER
- "Take checkpoint", "evaluate my progress on module" → CURRICULUM_PLANNER
- Never show cards that haven't passed QUALITY_CHECKER
- Simple greetings/thanks → respond directly (no agent needed)

Output format (JSON only):
{
    "next_agent": "tutor|card_generator|quality_checker|scheduler|curriculum_planner|respond",
    "reasoning": "Brief explanation of your decision",
    "task": "Specific instruction for the chosen agent (if not respond)"
}

For "respond", also include:
{
    "next_agent": "respond",
    "reasoning": "...",
    "response": "Your direct response to the student"
}"""


def create_supervisor_agent(model_name: str = "gpt-4o-mini"):
    """Create the Supervisor agent."""
    return ChatOpenAI(model=model_name, temperature=0.3)


def route_request(
    llm: ChatOpenAI,
    query: str,
    context: dict | None = None,
) -> dict:
    """
    Decide which agent should handle the request.

    Args:
        llm: The language model to use
        query: The user's message
        context: Additional context (current_mode, pending_cards, etc.)

    Returns:
        Routing decision with next_agent, reasoning, and task/response
    """
    context = context or {}

    user_content = f"Student message: {query}"

    if context.get("current_mode"):
        user_content += f"\nCurrent mode: {context['current_mode']}"

    if context.get("pending_cards"):
        user_content += f"\nPending cards awaiting quality check: {len(context['pending_cards'])}"

    if context.get("recent_explanation"):
        user_content += "\nNote: Tutor just provided an explanation. Consider generating flashcards."

    if context.get("card_context"):
        card = context["card_context"]
        user_content += f"\nStudent is studying flashcard: {card.get('question', '')[:50]}..."

    user_content += "\n\nDecide which agent should handle this, or respond directly."

    messages = [
        SystemMessage(content=SUPERVISOR_PROMPT),
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

        decision = json.loads(content)

        # Validate next_agent (v7: added curriculum_planner)
        valid_agents = ["tutor", "card_generator", "quality_checker", "scheduler", "curriculum_planner", "respond"]
        if decision.get("next_agent") not in valid_agents:
            decision["next_agent"] = "tutor"  # Default to tutor

        return decision

    except json.JSONDecodeError:
        # Default to tutor for questions
        return {
            "next_agent": "tutor",
            "reasoning": "Could not parse routing decision, defaulting to tutor",
            "task": query,
        }


def should_generate_cards(explanation: str) -> bool:
    """
    Heuristic to decide if an explanation warrants flashcard generation.

    Args:
        explanation: The tutor's explanation

    Returns:
        True if flashcards should be generated
    """
    # Generate cards for substantive explanations
    if len(explanation) < 200:
        return False

    # Look for educational content markers
    educational_markers = [
        "is a", "are ", "means", "refers to", "defined as",
        "key concept", "important", "remember", "example",
        "first", "second", "third", "steps", "types",
    ]

    lower_explanation = explanation.lower()
    return any(marker in lower_explanation for marker in educational_markers)
