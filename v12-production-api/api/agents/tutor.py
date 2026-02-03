"""LangGraph-based tutor agent for StudyBuddy v12.

Extends the v10 chat endpoint with optional Brave Search capability.
When a program has a Brave Search connector configured, the tutor
uses a LangGraph agent to reason about when to search the web.

Without Brave Search, the chat endpoint falls back to the existing
v10 direct-LLM flow (no agent needed).

The create_tutor_graph() factory is used by LangGraph Cloud (langgraph.json)
to deploy the agent as a production API endpoint.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


async def create_tutor_with_search(
    program_name: str,
    context: str,
    brave_api_key: Optional[str] = None,
):
    """Create a tutor agent, optionally with Brave Search capability.

    When brave_api_key is provided, returns a LangGraph agent that can
    search the web when the local document context is insufficient.

    When brave_api_key is None, returns None (caller should use the
    existing v10 direct-LLM flow).

    Args:
        program_name: Name of the learning program
        context: Retrieved document context for the current question
        brave_api_key: Optional Brave Search API key

    Returns:
        Tuple of (agent, mcp_client) or (None, None) if no search configured.
    """
    if not brave_api_key:
        return None, None

    from ..services.connectors.brave_connector import get_brave_search_tools

    tools, mcp_client = await get_brave_search_tools(brave_api_key)

    if not tools:
        logger.warning("No Brave Search tools available, falling back to no-agent flow")
        return None, None

    # Build the system prompt
    has_context = bool(context and context.strip())

    if has_context:
        system_prompt = (
            f"You are a helpful tutor for the learning program: {program_name}\n\n"
            f"Use the following context from the student's documents to answer "
            f"questions accurately:\n\n{context}\n\n"
            f"You also have access to web search. Use it ONLY when the document "
            f"context above does not contain sufficient information to answer the "
            f"question fully. Prefer document context over web search results.\n\n"
            f"When you use web search results, always cite the source URL so the "
            f"student can read more."
        )
    else:
        system_prompt = (
            f"You are a helpful tutor for the learning program: {program_name}\n\n"
            f"The student hasn't uploaded documents for this topic yet. "
            f"You have access to web search to find relevant information.\n\n"
            f"When you use web search results, always cite the source URL so the "
            f"student can read more."
        )

    agent = _build_langgraph_agent(system_prompt, tools)

    logger.info(
        f"Created tutor agent with {len(tools)} search tools "
        f"for program '{program_name}'"
    )

    return agent, mcp_client


def _build_langgraph_agent(system_prompt: str, tools: list):
    """Build a LangGraph agent with the given system prompt and tools.

    Uses a StateGraph with an agent node (LLM with tools bound) and
    a ToolNode for executing tool calls. The agent decides whether
    to call tools or return a final response.
    """
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import SystemMessage
    from langgraph.graph import StateGraph, MessagesState, START
    from langgraph.prebuilt import ToolNode, tools_condition

    llm = ChatOpenAI(model="gpt-4o", streaming=True)
    llm_with_tools = llm.bind_tools(tools)

    async def call_model(state: MessagesState):
        messages = [SystemMessage(content=system_prompt)] + state["messages"]
        response = await llm_with_tools.ainvoke(messages)
        return {"messages": [response]}

    builder = StateGraph(MessagesState)
    builder.add_node("agent", call_model)
    builder.add_node("tools", ToolNode(tools))
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", tools_condition)
    builder.add_edge("tools", "agent")

    return builder.compile()


def create_tutor_graph():
    """Factory function for LangGraph Cloud deployment.

    Returns a compiled StateGraph that can be served as an API endpoint
    via `langgraph serve`. Referenced in langgraph.json.
    """
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import SystemMessage
    from langgraph.graph import StateGraph, MessagesState, START
    from langgraph.prebuilt import ToolNode, tools_condition

    llm = ChatOpenAI(model="gpt-4o", streaming=True)

    system_prompt = (
        "You are StudyBuddy, a helpful AI tutor. "
        "Help the student learn by answering questions clearly and "
        "encouraging active recall. Use examples and analogies when helpful."
    )

    async def call_model(state: MessagesState):
        messages = [SystemMessage(content=system_prompt)] + state["messages"]
        response = await llm.ainvoke(messages)
        return {"messages": [response]}

    builder = StateGraph(MessagesState)
    builder.add_node("agent", call_model)
    builder.add_edge(START, "agent")

    return builder.compile()
