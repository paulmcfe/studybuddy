# Building StudyBuddy v5

Alright, time to put all of this into practice. You're going to transform StudyBuddy from a single agentic RAG system into a full multi-agent architecture using the supervisor pattern.

## Where We Left Off

In Chapter 4, you built StudyBuddy v4: an agentic RAG system using LangGraph. It could analyze queries, decide when to search the knowledge base, reflect on its answers, and generate flashcards when explaining important concepts. All of this happened in a single agent with a state machine controlling its behavior.

That single agent is doing a lot. It's tutoring (explaining concepts), generating content (creating flashcards), and evaluating quality (deciding if flashcards are good). These are genuinely different tasks that benefit from specialization.

## What We're Adding

StudyBuddy v5 splits into a team of specialized agents, each focused on doing one thing really well:

- **Tutor Agent:** Explains concepts conversationally. This is the friendly expert who helps you understand difficult material.
- **Card Generator Agent:** Creates flashcards from explanations and source materials. Focused entirely on producing high-quality cards.
- **Quality Checker Agent:** Validates that flashcards are clear, accurate, and useful. Acts as the editor for generated content.
- **Learning Coordinator (Supervisor):** Orchestrates the team. Decides which agents to engage based on what the student needs.

This architecture focuses on the learning experience. When you ask a question, the Tutor explains. When flashcards would help, the Coordinator engages the Card Generator, then the Quality Checker validates them before showing to the student. All orchestrated seamlessly.

## Project Setup

Create a new directory for v5 and set up dependencies:

```bash
cd v5-multi-agent
uv sync
uv run uvicorn api.index:app --reload --port 800
```

The `langgraph-supervisor` package provides the `create_supervisor` function that handles much of the coordination boilerplate. It's built on top of LangGraph and gives you a working supervisor pattern with minimal code.

## Defining the Agent Team

Let's start by defining each specialist agent. First, the Tutor:

```python
from langchain.agents import create_agent
from langchain.tools import tool

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

You have access to search_materials to find relevant content."""

@tool
def search_materials(query: str) -> str:
    """Search the AI engineering reference guides for information.

    Args:
        query: Topic or concept to search for
    """
    results = vector_store.similarity_search(query, k=4)

    if not results:
        return "No relevant information found."

    formatted = []
    for doc in results:
        source = doc.metadata.get('source', 'Unknown')
        formatted.append(f"[{source}]:\n{doc.page_content}")

    return "\n\n".join(formatted)

tutor_agent = create_agent(
    model="gpt-5-nano",
    tools=[search_materials],
    system_prompt=TUTOR_PROMPT
)
```

The Tutor is optimized for explanation. It has access to the knowledge base and is prompted to be conversational and check understanding.

Now the Card Generator:

```python
CARD_GENERATOR_PROMPT = """You are StudyBuddy's Card Generator,
specialized in creating effective flashcards.

Your job: Create flashcards that help students remember key concepts.

Guidelines for good flashcards:
- One concept per card (atomic)
- Questions should be specific and unambiguous
- Answers should be concise but complete
- Use active recall (question that requires thinking)
- Include context when helpful
- Vary question types: definition, comparison, application

Output format (JSON):
{
    "question": "Clear, specific question",
    "answer": "Concise, complete answer",
    "topic": "Main topic or concept",
    "difficulty": "basic|intermediate|advanced",
    "source": "Reference guide name"
}"""

card_generator = create_agent(
    model="gpt-4o-mini",
    tools=[search_materials],
    system_prompt=CARD_GENERATOR_PROMPT
)
```

Notice we're using `gpt-4o-mini` for the Card Generator. Card generation benefits from structured output capabilities, and 4o-mini handles JSON generation well. The Tutor uses `gpt-5-nano` for conversational quality. Different agents can use different models based on their needs.

The Quality Checker validates generated cards:

```python
QUALITY_CHECKER_PROMPT = """You are StudyBuddy's Quality Checker,
ensuring flashcards meet learning standards.

Your job: Evaluate flashcard quality and suggest improvements.

Evaluation criteria:
1. Clarity: Is the question unambiguous?
2. Accuracy: Is the answer factually correct?
3. Completeness: Does the answer fully address the question?
4. Atomicity: Does it test exactly one concept?
5. Usefulness: Will this help the student learn?

Output format (JSON):
{
    "approved": true/false,
    "score": 1-5,
    "issues": ["list of problems if any"],
    "suggestions": ["improvements if not approved"],
    "revised_card": null or improved card object
}

Be strict but fair. Cards should be genuinely useful for learning."""

quality_checker = create_agent(
    model="gpt-5-nano",
    tools=[],
    system_prompt=QUALITY_CHECKER_PROMPT
)
```

The Quality Checker has no tools—it just evaluates content. Its job is critique, not research.

## Creating the Supervisor

Now we wire up the team with `langgraph-supervisor`:

```python
from langgraph_supervisor import create_supervisor

COORDINATOR_PROMPT = """You are StudyBuddy's Learning Coordinator,
orchestrating a team of specialized agents.

Your team:
- TUTOR: Explains concepts conversationally. Use for questions,
  confusion, or requests to learn about topics.
- CARD_GENERATOR: Creates flashcards. Use after tutoring explains
  something the student should remember.
- QUALITY_CHECKER: Validates flashcards. Always use after generating
  cards before showing to student.

Workflow patterns:
1. Learning mode: Tutor explains -> Card Generator creates ->
   Quality Checker validates -> Show cards to student
2. Direct response: Simple greetings, clarifications, or meta-questions

Guidelines:
- Always route questions to Tutor first
- Generate cards for key concepts explained by Tutor
- Never show cards that haven't passed Quality Checker
- Keep the student experience seamless and natural"""

# Create the supervisor
supervisor = create_supervisor(
    agents={
        "tutor": tutor_agent,
        "card_generator": card_generator,
        "quality_checker": quality_checker
    },
    model="gpt-5-nano",
    system_prompt=COORDINATOR_PROMPT
)
```

The `create_supervisor` function handles the orchestration mechanics. It creates a graph where the supervisor can route to any worker, workers return to the supervisor, and the supervisor decides what happens next. You get all the coordination logic from earlier in this chapter, wrapped up in a clean API.

## The Complete Graph

Let's look at the full implementation with shared state:

```python
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

class StudyBuddyState(TypedDict):
    messages: Annotated[list, add_messages]
    current_mode: str  # "learning"
    pending_cards: list[dict]
    approved_cards: list[dict]
    current_topic: str

def build_studybuddy_graph():
    """Build the complete multi-agent graph."""

    graph = StateGraph(StudyBuddyState)

    # Add supervisor node (handles routing)
    graph.add_node("supervisor", supervisor_node)

    # Add worker nodes
    graph.add_node("tutor", tutor_node)
    graph.add_node("card_generator", card_generator_node)
    graph.add_node("quality_checker", quality_checker_node)
    graph.add_node("respond", respond_node)

    # Entry point
    graph.set_entry_point("supervisor")

    # Supervisor routes to workers or responds
    graph.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        {
            "tutor": "tutor",
            "card_generator": "card_generator",
            "quality_checker": "quality_checker",
            "respond": "respond"
        }
    )

    # Workers return to supervisor for next decision
    graph.add_edge("tutor", "supervisor")
    graph.add_edge("card_generator", "supervisor")
    graph.add_edge("quality_checker", "supervisor")

    # Respond ends the graph
    graph.add_edge("respond", END)

    return graph.compile()
```

The graph structure mirrors what we discussed earlier: supervisor at the hub, workers around the edges, communication flowing through the supervisor.

## Node Implementations

Each node wraps its agent and handles state updates:

```python
def tutor_node(state: StudyBuddyState) -> dict:
    """Tutor explains concepts."""

    # Get the task from supervisor
    task = get_current_task(state)

    # Run tutor agent
    result = tutor_agent.invoke({
        "messages": [{"role": "user", "content": task}]
    })

    # Extract the explanation
    explanation = result["messages"][-1].content

    return {
        "messages": [{"role": "assistant", "content": explanation}],
        "current_topic": extract_topic(task)
    }

def card_generator_node(state: StudyBuddyState) -> dict:
    """Generate flashcards from recent explanation."""

    # Get context from recent tutoring
    recent_explanation = get_recent_explanation(state)
    topic = state.get("current_topic", "General")

    # Generate cards
    result = card_generator.invoke({
        "messages": [{
            "role": "user",
            "content": f"Create flashcards for: {topic}\n\n"
                      f"Based on: {recent_explanation}"
        }]
    })

    # Parse generated cards
    cards = parse_cards(result["messages"][-1].content)

    return {"pending_cards": cards}

def quality_checker_node(state: StudyBuddyState) -> dict:
    """Validate pending cards."""

    pending = state.get("pending_cards", [])
    approved = []

    for card in pending:
        result = quality_checker.invoke({
            "messages": [{
                "role": "user",
                "content": f"Evaluate this flashcard:\n{json.dumps(card)}"
            }]
        })

        evaluation = parse_evaluation(result["messages"][-1].content)

        if evaluation["approved"]:
            approved.append(card)
        elif evaluation.get("revised_card"):
            approved.append(evaluation["revised_card"])

    return {
        "pending_cards": [],
        "approved_cards": state.get("approved_cards", []) + approved
    }
```

## Background Card Generation Pipeline

One powerful pattern is generating cards in the background while the tutoring conversation continues. After the Tutor explains something, kick off card generation and quality checking asynchronously:

```python
import asyncio

async def background_card_pipeline(explanation: str, topic: str):
    """Generate and validate cards in background."""

    # Generate cards
    cards = await card_generator.ainvoke({
        "messages": [{
            "role": "user",
            "content": f"Create flashcards for: {topic}\n\n{explanation}"
        }]
    })

    parsed_cards = parse_cards(cards["messages"][-1].content)

    # Validate each card
    approved = []
    for card in parsed_cards:
        evaluation = await quality_checker.ainvoke({
            "messages": [{
                "role": "user",
                "content": f"Evaluate: {json.dumps(card)}"
            }]
        })

        result = parse_evaluation(evaluation["messages"][-1].content)
        if result["approved"]:
            approved.append(card)

    return approved
```

The student gets immediate responses from the Tutor while flashcards are being prepared behind the scenes. When cards are ready, they appear seamlessly.

## Error Recovery

Multi-agent systems need robust error handling. What if the Card Generator produces invalid JSON? What if the Quality Checker rejects everything?

```python
def card_generator_node_with_recovery(state: StudyBuddyState) -> dict:
    """Generate cards with error recovery."""

    max_attempts = 3

    for attempt in range(max_attempts):
        try:
            result = card_generator.invoke({
                "messages": [{
                    "role": "user",
                    "content": build_card_prompt(state)
                }]
            })

            cards = parse_cards(result["messages"][-1].content)

            if cards:  # Successfully parsed at least one card
                return {"pending_cards": cards}

        except json.JSONDecodeError:
            # Invalid JSON, retry with explicit format reminder
            continue
        except Exception as e:
            logger.error(f"Card generation failed: {e}")
            if attempt == max_attempts - 1:
                raise

    # All attempts failed, return empty
    return {"pending_cards": []}
```

Similar recovery logic protects each agent. The system degrades gracefully rather than crashing on the first error.

## Putting It All Together

Here's the complete FastAPI backend:

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

# Initialize the graph
studybuddy = build_studybuddy_graph()

class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"

class ChatResponse(BaseModel):
    reply: str
    cards: list[dict] = []

@app.post("/api/chat")
async def chat(request: ChatRequest) -> ChatResponse:
    """Main chat endpoint."""

    try:
        # Load session state
        state = load_session_state(request.session_id)

        # Add user message
        state["messages"].append({
            "role": "user",
            "content": request.message
        })

        # Run the multi-agent graph
        result = await studybuddy.ainvoke(state)

        # Save updated state
        save_session_state(request.session_id, result)

        # Extract response
        reply = result["messages"][-1]["content"]
        new_cards = result.get("approved_cards", [])

        return ChatResponse(
            reply=reply,
            cards=new_cards
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

## Testing Multi-Agent Coordination

Testing multi-agent systems requires checking both individual agents and their coordination:

```python
import pytest

class TestStudyBuddyV5:

    def test_tutor_explains_concept(self):
        """Tutor should search and explain."""
        result = tutor_agent.invoke({
            "messages": [{"role": "user", "content": "Explain RAG"}]
        })

        response = result["messages"][-1].content
        assert "retrieval" in response.lower()
        assert len(response) > 100  # Substantive explanation

    def test_card_generator_produces_valid_json(self):
        """Card generator should output parseable cards."""
        result = card_generator.invoke({
            "messages": [{
                "role": "user",
                "content": "Create flashcards about embeddings"
            }]
        })

        cards = parse_cards(result["messages"][-1].content)
        assert len(cards) > 0
        assert all("question" in c for c in cards)
        assert all("answer" in c for c in cards)

    def test_quality_checker_rejects_bad_cards(self):
        """Quality checker should catch problems."""
        bad_card = {
            "question": "What?",  # Too vague
            "answer": "Yes",      # Not helpful
            "topic": "Unknown"
        }

        result = quality_checker.invoke({
            "messages": [{
                "role": "user",
                "content": f"Evaluate: {json.dumps(bad_card)}"
            }]
        })

        evaluation = parse_evaluation(result["messages"][-1].content)
        assert not evaluation["approved"]

    def test_full_learning_workflow(self):
        """Complete workflow: question -> explanation -> cards."""
        state = {
            "messages": [{"role": "user", "content": "Teach me about vector databases"}],
            "current_mode": "learning",
            "pending_cards": [],
            "approved_cards": [],
            "current_topic": ""
        }

        result = studybuddy.invoke(state)

        # Should have explanation in messages
        assert len(result["messages"]) > 1

        # Should have generated and approved cards
        assert len(result["approved_cards"]) > 0
```

## Using LangSmith Studio

With multi-agent systems, LangSmith Studio becomes invaluable. Enable tracing and explore your agent interactions:

```python
import os

os.environ["LANGSMITH_API_KEY"] = "your-api-key"
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_PROJECT"] = "studybuddy-v5"
```

Now run some requests and open Studio. You'll see the supervisor receiving queries, routing to specialists, collecting results, and making decisions. Click any node to inspect its inputs and outputs. Watch the coordination unfold.

When something goes wrong—a card gets rejected, an explanation misses the point—you can trace exactly what happened. Studio shows you the full execution graph, letting you identify which agent misbehaved and why.

## What's Next

StudyBuddy v5 demonstrates the power of multi-agent coordination. You have specialists doing what they're good at, a supervisor keeping everything organized, and a system that's more maintainable than a monolithic single agent.

But there's something missing: memory. Right now, StudyBuddy forgets everything between sessions. It doesn't remember what you've studied, what you struggle with, or how you like to learn. And without memory, we can't implement spaced repetition—scheduling reviews based on your learning history requires remembering that history.

In Chapter 6, you'll add persistent memory that makes StudyBuddy truly personal. You'll also add a Scheduler Agent that uses the SM-2 spaced repetition algorithm to determine what cards to review and when. Spaced repetition only makes sense with persistent memory, because tracking review history and calculating optimal intervals requires knowing what happened in previous sessions.

You've leveled up from single agent to team coordination. That's a big deal. Take a moment to appreciate what you've built: a system where multiple AI specialists collaborate to help students learn. That's real AI engineering.

Let's keep building.
