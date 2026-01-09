# Memory Systems

## The Memory Problem

LLMs are stateless. Each request starts fresh with no memory of previous interactions. The context window provides temporary memory within a conversation, but it has limits—typically 8K to 200K tokens. Once a conversation exceeds this, older content is lost.

For agents that need to maintain state across sessions, remember user preferences, or learn from past interactions, explicit memory systems are required.

## Types of Memory

Memory in AI systems mirrors human memory in useful ways.

### Semantic Memory

Stores facts and knowledge. "The user prefers detailed explanations." "They're studying for AWS certification." "Their company uses Python 3.11."

Semantic memory drives personalization. It captures what you know about a user or domain that should inform future interactions.

### Episodic Memory

Records specific interactions and experiences. "Last Tuesday, we discussed transformer architectures." "The user struggled with the attention mechanism explanation."

Episodic memory enables learning from experience. You can reference past conversations, avoid repeating mistakes, and build on previous work.

### Procedural Memory

Encodes behaviors and patterns. "Always cite sources when answering technical questions." "Ask clarifying questions when queries are ambiguous."

Procedural memory shapes how the agent behaves rather than what it knows.

## Memory Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      AGENT                                   │
├─────────────────────────────────────────────────────────────┤
│  Context Window (Short-Term)                                │
│  - Current conversation                                      │
│  - Retrieved memories                                        │
│  - Active task state                                        │
├─────────────────────────────────────────────────────────────┤
│  Memory Store (Long-Term)                                   │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │  Semantic   │  │  Episodic   │  │ Procedural  │         │
│  │   Facts     │  │   Events    │  │  Behaviors  │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
└─────────────────────────────────────────────────────────────┘
```

## Memory Formation

### Hot Path Memory

Memory created during the conversation while the agent is actively responding.

```python
# During conversation, agent decides to remember something
@tool
def remember(fact: str) -> str:
    """
    Store an important fact about the user or conversation.
    Use when the user shares preferences, goals, or important context.
    """
    memory_store.add(
        content=fact,
        type="semantic",
        timestamp=datetime.now()
    )
    return f"I'll remember that: {fact}"
```

**Pros:** Immediate, agent can decide what's important in context.
**Cons:** Adds latency, uses tokens for memory operations.

### Background Memory

Memory extracted from conversations after they complete.

```python
def extract_memories_background(conversation: list[Message]):
    """Run after conversation ends to extract memories."""
    
    # Use LLM to identify important information
    prompt = """Analyze this conversation and extract:
    1. User preferences mentioned
    2. Important facts shared
    3. Topics discussed
    4. Any commitments or follow-ups needed
    
    Conversation:
    {conversation}
    """
    
    extraction = llm.invoke(prompt.format(conversation=conversation))
    
    for memory in parse_memories(extraction):
        memory_store.add(memory)
```

**Pros:** Doesn't add latency to conversations, can do deeper analysis.
**Cons:** May miss nuance, extraction quality varies.

### Hybrid Approach

Combine both: hot path for explicit memory requests, background for automatic extraction.

## LangGraph Memory (langmem)

LangGraph provides native memory capabilities through the `langmem` library.

### Setup

```python
from langgraph.store.memory import InMemoryStore
from langmem import create_manage_memory_tool, create_search_memory_tool

# Create memory store
memory_store = InMemoryStore()

# Create memory tools
manage_memory = create_manage_memory_tool(memory_store)
search_memory = create_search_memory_tool(memory_store)
```

### Using Memory Tools

```python
from langchain.agents import create_agent

agent = create_agent(
    model="gpt-5-nano",
    tools=[manage_memory, search_memory, other_tools...],
    system_prompt="""You have access to a memory system.

Use manage_memory to store important information about the user:
- Preferences and interests
- Goals they're working toward
- Facts they share about themselves
- Topics they want to revisit

Use search_memory to recall relevant information from past conversations.

Always check memory at the start of conversations for relevant context."""
)
```

### Memory Namespaces

Organize memories by user, topic, or any dimension:

```python
# Store memory with namespace
memory_store.put(
    namespace=("user", user_id, "preferences"),
    key="learning_style",
    value={"preference": "detailed_explanations", "examples": True}
)

# Retrieve by namespace
preferences = memory_store.search(
    namespace=("user", user_id, "preferences")
)
```

## Memory Storage Backends

### InMemoryStore

Good for development and testing. Data lost when process stops.

```python
from langgraph.store.memory import InMemoryStore
store = InMemoryStore()
```

### PostgreSQL

Production-grade persistent storage.

```python
from langgraph.store.postgres import PostgresStore

store = PostgresStore(
    connection_string="postgresql://user:pass@localhost/memorydb"
)
```

### Redis

Fast, good for session-based memory with TTL.

```python
from langgraph.store.redis import RedisStore

store = RedisStore(
    url="redis://localhost:6379",
    ttl=86400  # Memories expire after 24 hours
)
```

## Memory Retrieval

### Semantic Search

Find memories related to current context:

```python
def get_relevant_memories(query: str, user_id: str, k: int = 5):
    """Retrieve memories semantically related to the query."""
    
    results = memory_store.search(
        namespace=("user", user_id),
        query=query,
        limit=k
    )
    
    return [r.value for r in results]
```

### Recency Weighting

Combine relevance with recency:

```python
def get_memories_with_recency(query: str, user_id: str):
    """Get memories weighted by both relevance and recency."""
    
    results = memory_store.search(
        namespace=("user", user_id),
        query=query,
        limit=20
    )
    
    # Score by relevance * recency
    scored = []
    now = datetime.now()
    for r in results:
        age_hours = (now - r.metadata["timestamp"]).total_seconds() / 3600
        recency_weight = 1 / (1 + age_hours / 24)  # Decay over days
        combined_score = r.score * recency_weight
        scored.append((combined_score, r))
    
    scored.sort(reverse=True)
    return [r for _, r in scored[:5]]
```

## Memory Integration Patterns

### Injection at Start

Load relevant memories into system prompt:

```python
def build_system_prompt(user_id: str, current_query: str):
    memories = get_relevant_memories(current_query, user_id)
    
    memory_context = "\n".join([
        f"- {m['content']}" for m in memories
    ])
    
    return f"""You are a helpful assistant.

Known information about this user:
{memory_context}

Use this context to personalize your responses."""
```

### On-Demand Retrieval

Agent searches memory when needed:

```python
@tool
def recall(topic: str) -> str:
    """
    Search memory for information about a topic.
    Use when you need to remember something about the user or past conversations.
    """
    memories = memory_store.search(
        query=topic,
        limit=5
    )
    
    if not memories:
        return "No relevant memories found."
    
    return "\n".join([f"- {m.value['content']}" for m in memories])
```

### Proactive Memory

System suggests relevant memories:

```python
def get_proactive_memories(conversation_context: str, user_id: str):
    """Find memories the user might not ask about but would be helpful."""
    
    # Look for related past topics
    related = memory_store.search(
        namespace=("user", user_id, "topics"),
        query=conversation_context,
        limit=3
    )
    
    suggestions = []
    for memory in related:
        if memory.score > 0.7:  # High relevance threshold
            suggestions.append(memory.value)
    
    return suggestions
```

## Privacy and Data Governance

Memory systems store personal data. Consider:

### User Control

```python
@tool
def forget(topic: str) -> str:
    """
    Remove memories about a specific topic.
    Use when the user asks you to forget something.
    """
    deleted = memory_store.delete(
        namespace=("user", current_user_id),
        filter={"topic": topic}
    )
    return f"Removed {deleted} memories about {topic}"

@tool  
def show_memories() -> str:
    """Show all stored memories for the current user."""
    memories = memory_store.list(
        namespace=("user", current_user_id)
    )
    return format_memories_for_display(memories)
```

### Data Retention

```python
# Automatic cleanup of old memories
def cleanup_old_memories(max_age_days: int = 90):
    cutoff = datetime.now() - timedelta(days=max_age_days)
    
    memory_store.delete(
        filter={"timestamp": {"$lt": cutoff}}
    )
```

### Access Control

```python
# Ensure users can only access their own memories
def get_user_memories(user_id: str, requesting_user_id: str):
    if user_id != requesting_user_id:
        raise PermissionError("Cannot access other users' memories")
    
    return memory_store.search(namespace=("user", user_id))
```

## Graceful Degradation

Memory systems can fail. Handle gracefully:

```python
def get_memories_safe(user_id: str, query: str):
    """Get memories with fallback on failure."""
    try:
        return memory_store.search(
            namespace=("user", user_id),
            query=query,
            timeout=2.0  # Don't wait forever
        )
    except MemoryStoreError as e:
        logger.warning(f"Memory retrieval failed: {e}")
        return []  # Continue without memories
    except TimeoutError:
        logger.warning("Memory retrieval timed out")
        return []
```

## Memory vs. RAG

Memory and RAG both provide context, but serve different purposes:

| Aspect | Memory | RAG |
|--------|--------|-----|
| Content | User-specific, interaction history | Document corpus, knowledge base |
| Scope | Per-user, personal | Shared across users |
| Updates | Continuous, from interactions | Batch, from document changes |
| Query type | "What do I know about this user?" | "What information exists about X?" |

Often used together: RAG for knowledge, memory for personalization.

## Best Practices

1. **Be selective** about what to remember. Not everything is worth storing.

2. **Structure memories** consistently. Use schemas for different memory types.

3. **Provide user transparency**. Let users see and manage their memories.

4. **Set retention policies**. Old memories should expire or be consolidated.

5. **Handle failures gracefully**. Agents should work (less well) without memory.

6. **Respect privacy**. Don't store sensitive information unnecessarily.

## Related Concepts

- **Agents**: Primary users of memory systems
- **RAG**: Complementary knowledge retrieval
- **Context Engineering**: Managing what goes in the context window
- **LangGraph**: Framework providing native memory support
