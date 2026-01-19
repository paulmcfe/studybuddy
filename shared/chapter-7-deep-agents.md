# Chapter 7: Deep Agents

Here's the thing about the agents we've built so far: they're reactive. User asks a question, agent thinks for a moment, maybe calls a tool or two, and responds. That's powerful stuff, and it handles a huge range of tasks. But some problems don't fit neatly into a single conversation turn.

Think about planning a two-week trip through Europe. You need to research destinations, check flight availability, compare hotel prices, build an itinerary that makes geographic sense, book everything in the right sequence, and handle the inevitable conflicts when your preferred hotel is full on the dates you need. A simple agent might help you search for flights. A *deep* agent could orchestrate the entire planning process over multiple sessions, remembering your preferences, adjusting plans when constraints change, and coordinating all the moving pieces.

That's what this chapter is about: building agents that can tackle complex, multi-step projects that unfold over time. We're moving from agents that respond to agents that *plan, delegate, reflect, and adapt*. This is where AI engineering gets seriously interesting.

We'll explore the four key elements that transform a basic agent into a deep agent: planning and task decomposition, context management over long time horizons, subagent spawning and delegation, and long-term memory integration. You've already built memory systems in Chapter 6, so we'll focus on how memory integrates with the other three capabilities to enable truly sophisticated agent behavior.

By the end of this chapter, you'll understand when deep agents make sense (and when they're overkill), how to implement each of the four key elements, and how to combine them into agents that can handle projects spanning hours, days, or even weeks. Let's get into it.

## What Makes an Agent "Deep"?

The term "deep agent" comes from recent work in the AI engineering community, particularly from teams at Anthropic and LangChain who've been pushing the boundaries of what agents can accomplish. But what actually makes an agent "deep" versus "shallow"?

A *shallow* agent operates in what we might call stimulus-response mode. User provides input, agent generates output. The agent might use tools and reason through problems, but each interaction is essentially self-contained. The ReAct pattern we covered in Chapter 3 is a perfect example. The agent thinks, acts, observes, and repeats until it has an answer. Powerful, but bounded.

A *deep* agent breaks out of this single-turn paradigm. It maintains goals across multiple interactions. It decomposes big objectives into smaller, manageable subtasks. It spawns helper agents to handle specific pieces of work. It reflects on their own progress and adjust strategies when things aren't working. And crucially, the deep agent manages context intelligently by knowing what to remember, what to summarize, and what to let go.

Here's a concrete way to think about the distinction. Imagine you're building a research assistant. A shallow agent helps you search for papers and summarize what it finds. You ask, it delivers. A deep agent, on the other hand, could take a research question such "What are the emerging best practices for fine-tuning large language models?" and autonomously create a research plan, identify the key sub-questions to investigate, search multiple sources for each sub-question, synthesize findings across sources, identify gaps and contradictions, and produce a comprehensive report, all while keeping you informed of progress and asking for guidance only when genuinely needed.

So when do you actually need deep agent capabilities? Here's a quick decision framework:

**Consider deep agents when:**
- The task naturally spans multiple sessions or extended time periods.
- Success requires coordinating multiple distinct subtasks.
- The problem benefits from planning before acting.
- You need the agent to recover from dead ends autonomously.
- Context management becomes a bottleneck with simpler approaches.

**Stick with simpler agents when:**
- Tasks complete in a single conversation turn.
- The workflow is predictable and can be hard-coded.
- Latency is critical (deep agents add overhead).
- You're still figuring out the problem space.

Deep agents aren't always better. They're more complex to build, harder to debug, and slower to execute. But for the right problems? They're absolute game-changers.

## The Four Key Elements of Deep Agents

The deep agents framework rests on four interconnected capabilities. Each element addresses a specific challenge that emerges when agents tackle complex, long-running tasks:

- **Planning and Task Decomposition**: Breaking big goals into achievable subtasks with clear dependencies. Without this, agents flail. They might work hard but not make meaningful progress toward the actual objective.

- **Context Management**: Strategically handling information over long time horizons. Context windows are finite, and deep agents need to decide what's worth keeping, what to summarize, and what to forget.

- **Subagent Spawning and Delegation**: Creating specialized helper agents for specific subtasks. Just like a manager delegates to team members, a deep agent orchestrates a workforce of focused specialists.

- **Long-Term Memory Integration**: Connecting persistent memory (which you built in Chapter 6) to enable learning across sessions. Deep agents remember, but they use those memories to inform planning and improve over time.

These four elements work together as a system. Planning generates the subtasks that get delegated to subagents. Context management ensures the agent doesn't lose track of its goals or progress. Memory provides continuity across sessions and informs future planning based on past experience. When all four click together, you get agents that feel genuinely intelligent, not just responsive, but purposeful.

Let's dig into each element.

## Planning and Task Decomposition

Every complex project starts with a plan. This is true for humans, and it's equally true for deep agents. Without explicit planning, agents tend to tackle whatever seems most immediate, which often isn't what's most important. Planning gives agents direction.

The core idea is straightforward: take a high-level goal and break it down into a sequence of smaller, achievable tasks. But the execution has real nuance. How do you decide what tasks are needed? How do you handle dependencies between tasks? How detailed should each task be?

### Breaking Down Complex Problems

Let's say you're building an agent to help organize a company offsite. The high-level goal is "Plan and execute a successful team offsite for 30 people." That's way too vague for an agent to act on directly. We need decomposition.

A good decomposition might look like this:

```python
from pydantic import BaseModel

class Task(BaseModel):
    id: str
    description: str
    depends_on: list[str] = []  # Task IDs this depends on
    estimated_hours: float
    status: str = "pending"

class Plan(BaseModel):
    goal: str
    tasks: list[Task]

offsite_plan = Plan(
    goal="Plan team offsite for 30 people",
    tasks=[
        Task(id="t1", description="Determine budget and dates", depends_on=[], estimated_hours=1),
        Task(id="t2", description="Survey team for location preferences", depends_on=["t1"], estimated_hours=0.5),
        Task(id="t3", description="Research venue options matching budget", depends_on=["t1", "t2"], estimated_hours=3),
        Task(id="t4", description="Get quotes from top 3 venues", depends_on=["t3"], estimated_hours=2),
        Task(id="t5", description="Book selected venue", depends_on=["t4"], estimated_hours=0.5),
        Task(id="t6", description="Plan agenda and activities", depends_on=["t5"], estimated_hours=4),
        Task(id="t7", description="Arrange transportation and logistics", depends_on=["t5"], estimated_hours=2),
        Task(id="t8", description="Send invitations with details", depends_on=["t6", "t7"], estimated_hours=1),
    ]
)
```

Notice what's happening here. Each task is specific enough to act on. Dependencies are explicit so, for example, you can't book a venue before you know the budget. And the decomposition reveals the natural structure of the problem.

### Forward Planning vs. Backward Chaining

There are two main strategies for generating plans: forward planning and backward chaining.

Forward planning starts from the current state and asks, "What's the first thing I need to do? Then what? Then what?" It's intuitive and works well when the path forward is reasonably clear.

Backward chaining starts from the goal and works backwards: "To achieve X, what needs to be true? To make that true, what needs to happen first?" This approach excels when you have a clear end state but the path to get there is murky.

For learning-related goals, backward chaining often works beautifully. Say someone wants to pass the AWS Solutions Architect certification exam. Working backwards:

- To pass the exam, I need to understand all domains covered
- To understand those domains, I need to study each one systematically
- To study effectively, I need quality learning materials and practice questions
- To get practice questions, I need to either find them or generate them

Each step backwards reveals prerequisites, eventually reaching tasks the agent can execute immediately.

Here's a simplified backward chaining implementation:

```python
def backward_chain(goal: str, llm) -> list[str]:
    """Generate prerequisites by working backward from goal."""
    
    prompt = f"""Given this goal: {goal}
    
What must be true or completed IMMEDIATELY BEFORE this goal can be achieved?
List only the direct prerequisites, not earlier steps.
Be specific and actionable.

Format: One prerequisite per line, no numbering."""
    
    prerequisites = llm.invoke(prompt).content.strip().split("\n")
    
    all_tasks = []
    for prereq in prerequisites:
        prereq = prereq.strip()
        if prereq:
            # Recursively find prerequisites of prerequisites
            sub_prereqs = backward_chain(prereq, llm)
            all_tasks.extend(sub_prereqs)
            all_tasks.append(prereq)
    
    all_tasks.append(goal)
    return all_tasks
```

In practice, you'd add depth limits to prevent infinite recursion and caching to avoid recomputing the same prerequisites. But the core pattern—start at the goal, work backwards—is incredibly powerful for educational and project planning agents.

### State Space Search

For really complex planning problems, you might need more sophisticated approaches. State space search treats planning as navigating from a start state to a goal state through a space of possible intermediate states.

Think of it like GPS navigation. You have a starting point (current knowledge), a destination (learning goal), and many possible routes. Some routes are faster, some are more scenic, some have construction delays. The planner needs to find a good path considering constraints like time, prerequisites, and the student's preferences.

```python
from dataclasses import dataclass
from heapq import heappush, heappop

@dataclass
class LearningState:
    known_topics: frozenset[str]
    time_spent: float
    path: list[str]
    
    def __lt__(self, other):
        return len(self.known_topics) > len(other.known_topics)

def search_learning_path(
    start_knowledge: set[str],
    goal_topics: set[str],
    topic_graph: dict,  # topic -> {prerequisites, time_hours}
    max_time: float = 100
) -> list[str]:
    """Find optimal learning path using A* search."""
    
    initial = LearningState(
        known_topics=frozenset(start_knowledge),
        time_spent=0,
        path=[]
    )
    
    frontier = [initial]
    visited = set()
    
    while frontier:
        current = heappop(frontier)
        
        # Check if we've reached the goal
        if goal_topics.issubset(current.known_topics):
            return current.path
        
        state_key = current.known_topics
        if state_key in visited:
            continue
        visited.add(state_key)
        
        # Find learnable topics (prerequisites satisfied)
        for topic, info in topic_graph.items():
            if topic in current.known_topics:
                continue
            if not set(info["prerequisites"]).issubset(current.known_topics):
                continue
            
            new_time = current.time_spent + info["time_hours"]
            if new_time > max_time:
                continue
            
            new_state = LearningState(
                known_topics=current.known_topics | {topic},
                time_spent=new_time,
                path=current.path + [topic]
            )
            heappush(frontier, new_state)
    
    return []  # No path found
```

This is admittedly getting into advanced territory, but it's worth knowing these techniques exist. For most curriculum planning, backward chaining with a reasonable depth limit works great. But when you have complex interdependencies and hard constraints, state space search can find solutions that simpler approaches miss.

### Hierarchical Planning

Sometimes a single level of decomposition isn't enough. The task "Research venue options" might itself need breaking down: search event venues in the area, filter by capacity, check availability for target dates, read reviews, and so on.

Hierarchical planning handles this by allowing tasks to have their own subtask decompositions. You end up with a tree structure where high-level objectives branch into progressively more concrete actions.

```python
class HierarchicalTask(BaseModel):
    id: str
    description: str
    subtasks: list["HierarchicalTask"] = []
    depends_on: list[str] = []
    is_leaf: bool = True  # True if directly executable
```

The key insight is that leaf tasks—the ones at the bottom of the tree—should be directly executable by an agent or tool. Everything above is organizational structure that guides execution but doesn't execute itself.

## Context Management in Deep Agents

Here's a problem that doesn't exist for simple agents but becomes critical for deep ones: context window exhaustion.

When an agent operates over extended periods, information accumulates. Conversation history, tool outputs, intermediate results, plans, progress updates, it all adds up. Eventually, you hit the context window limit. But even before that, performance degrades. Models get confused when context is cluttered with irrelevant details. The important stuff gets lost in the noise.

Deep agents need explicit strategies for managing context over time.

### The Context Window as Scarce Resource

Think of the context window like working memory. Humans can hold roughly seven items in working memory at once. Exceed that, and we start forgetting things or making errors. LLMs work similarly: there's a soft limit where performance drops even if you're technically within the token limit.

This means context management is fundamentally about prioritization. What information is essential right now? What can be summarized? What can be safely forgotten?

### Selective Information Retention

Not all information is equally important. A deep agent working on a research project probably needs to retain:

- The original research question (always relevant)
- Current plan and progress status (need to know where we are)
- Key findings discovered so far (the actual results)
- Recent conversation context (immediate continuity)

But it probably doesn't need:

- Full text of every document searched (it can re-fetch if needed)
- Detailed reasoning traces from completed tasks (the conclusion matters, not the journey)
- Failed attempts that led nowhere (unless they inform current strategy)

Here's a pattern for selective retention:

```python
def prioritize_context(context_items: list[dict], current_task: str) -> list[dict]:
    """Rank context items by relevance to current task."""
    
    categories = {
        "goal": 10,          # Original objective - always keep
        "plan": 9,           # Current plan - critical
        "progress": 8,       # What's done/pending - important
        "findings": 7,       # Key results - valuable
        "recent": 6,         # Last few messages - continuity
        "reference": 3,      # Supporting details - nice to have
        "historical": 1,     # Old interactions - low priority
    }
    
    scored = []
    for item in context_items:
        base_score = categories.get(item.get("type"), 2)
        
        # Boost if relevant to current task
        if is_relevant(item, current_task):
            base_score *= 1.5
            
        scored.append((base_score, item))
    
    # Sort by score descending, return items
    scored.sort(key=lambda x: x[0], reverse=True)
    return [item for _, item in scored]
```

### Context Compression and Summarization

When you can't keep everything, summarize. The trick is summarizing at the right level of detail: enough to be useful, concise enough to save space.

```python
async def compress_context(full_context: str, max_tokens: int, llm) -> str:
    """Compress context while preserving essential information."""
    
    prompt = f"""Summarize the following context, preserving:
1. The main goal or objective
2. Key decisions made and their rationale
3. Important findings or results
4. Current status and next steps
5. Any critical constraints or requirements

Remove:
- Redundant information
- Detailed intermediate reasoning
- Verbose explanations where a summary suffices

Context to compress:
{full_context}

Provide a compressed summary in under {max_tokens} tokens:"""
    
    summary = await llm.ainvoke(prompt)
    return summary.content
```

For long-running agents, periodic compression keeps context manageable. You might compress after completing each major task, or when context size exceeds a threshold, or at natural breakpoints in the workflow.

### When to Summarize vs. Retain Details

This is a judgment call, but here are some guidelines:

**Retain full details when:**
- Information is needed for immediate next steps.
- Exact wording matters (legal text, specific quotes).
- You're in active problem-solving mode.
- The information hasn't been acted on yet.

**Summarize when:**
- A task is complete and you only need the conclusion.
- Information is supporting context, not primary content.
- You're transitioning between major phases of work.
- Multiple items cover similar ground and can be consolidated.

The best deep agents develop intuitions about this through reflection: noticing when they've summarized too aggressively and lost needed detail, or kept too much and gotten confused.

### Progressive Summarization
One effective pattern is *progressive summarization*: maintaining multiple levels of detail simultaneously. You keep the most recent information at full fidelity, slightly older information at medium compression, and historical information as high-level summaries.

```python
class ProgressiveContext:
    def __init__(self, max_recent: int = 10, max_medium: int = 20):
        self.recent = []  # Full detail
        self.medium = []  # Summarized
        self.historical = ""  # Highly compressed
        self.max_recent = max_recent
        self.max_medium = max_medium
    
    def add(self, item: dict, llm):
        """Add new item, promoting older items to higher compression."""
        self.recent.append(item)
        
        # Promote recent to medium when threshold hit
        if len(self.recent) > self.max_recent:
            to_summarize = self.recent[:-self.max_recent]
            self.recent = self.recent[-self.max_recent:]
            
            summary = summarize_items(to_summarize, llm)
            self.medium.append(summary)
        
        # Promote medium to historical when threshold hit
        if len(self.medium) > self.max_medium:
            to_compress = self.medium[:-self.max_medium]
            self.medium = self.medium[-self.max_medium:]
            
            compressed = compress_to_summary(to_compress, self.historical, llm)
            self.historical = compressed
    
    def get_context(self) -> str:
        """Get context at appropriate detail levels."""
        parts = []
        
        if self.historical:
            parts.append(f"Historical context: {self.historical}")
        
        if self.medium:
            parts.append(f"Recent history: {format_medium(self.medium)}")
        
        parts.append(f"Current: {format_recent(self.recent)}")
        
        return "\n\n".join(parts)
```

This pattern is particularly useful for agents that operate over days or weeks. You never lose the historical context entirely, but you're not drowning in details from two weeks ago either.

## Subagent Spawning and Delegation

A single agent trying to do everything is like a one-person company trying to scale. At some point, you need to delegate.

Subagent spawning is the pattern where a coordinating agent creates specialized helper agents to handle specific subtasks. Each subagent has a focused role, relevant tools, and just the context it needs for its job. The coordinator orchestrates the overall workflow, delegates tasks, collects results, and synthesizes the final output.

### When to Spawn Subagents

Not every task needs delegation. Here's when subagents make sense:

**Good candidates for subagents:**
- Tasks requiring specialized tools or knowledge
- Work that can proceed independently of other tasks
- Subtasks with clearly defined inputs and outputs
- Tasks where context isolation helps (don't need full project context)

**Keep in the main agent:**
- Tasks requiring holistic project understanding
- Decision points that affect overall strategy
- Synthesis and integration work
- Tasks where context switching overhead exceeds benefit

### Managing Subagent Lifecycles

Spawning a subagent is just the beginning. You also need to:

1. **Initialize** with appropriate context and instructions
2. **Monitor** progress and handle failures
3. **Collect** results when complete
4. **Terminate** cleanly, releasing resources

```python
from dataclasses import dataclass
from typing import Any
from enum import Enum

class SubagentStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class Subagent:
    id: str
    role: str
    task: str
    status: SubagentStatus
    result: Any = None
    error: str = None

class SubagentManager:
    def __init__(self):
        self.subagents: dict[str, Subagent] = {}
    
    def spawn(self, role: str, task: str, context: dict) -> str:
        """Create and start a new subagent."""
        agent_id = f"{role}_{len(self.subagents)}"
        
        subagent = Subagent(
            id=agent_id,
            role=role,
            task=task,
            status=SubagentStatus.PENDING
        )
        self.subagents[agent_id] = subagent
        
        # In practice, you'd actually create and run the agent here
        self._execute_subagent(subagent, context)
        
        return agent_id
    
    def get_status(self, agent_id: str) -> SubagentStatus:
        return self.subagents[agent_id].status
    
    def get_result(self, agent_id: str) -> Any:
        subagent = self.subagents[agent_id]
        if subagent.status != SubagentStatus.COMPLETED:
            raise ValueError(f"Subagent {agent_id} not yet complete")
        return subagent.result
    
    def collect_all(self) -> dict[str, Any]:
        """Collect results from all completed subagents."""
        return {
            agent_id: sa.result 
            for agent_id, sa in self.subagents.items()
            if sa.status == SubagentStatus.COMPLETED
        }
```

### Parallel vs. Sequential Delegation

Sometimes subtasks can run in parallel because they don't depend on each other's outputs. Other times, they must run sequentially because each builds on the previous. Deep agents need to handle both patterns.

```python
import asyncio

async def parallel_delegation(tasks: list[dict], manager: SubagentManager) -> list[Any]:
    """Execute independent tasks in parallel."""
    
    async def run_task(task):
        agent_id = manager.spawn(
            role=task["role"],
            task=task["description"],
            context=task.get("context", {})
        )
        # Wait for completion (simplified)
        while manager.get_status(agent_id) == SubagentStatus.RUNNING:
            await asyncio.sleep(0.1)
        return manager.get_result(agent_id)
    
    results = await asyncio.gather(*[run_task(t) for t in tasks])
    return results

def sequential_delegation(tasks: list[dict], manager: SubagentManager) -> list[Any]:
    """Execute dependent tasks sequentially, passing results forward."""
    
    results = []
    accumulated_context = {}
    
    for task in tasks:
        # Include results from previous tasks in context
        task_context = {**task.get("context", {}), **accumulated_context}
        
        agent_id = manager.spawn(
            role=task["role"],
            task=task["description"],
            context=task_context
        )
        
        # Wait and get result
        result = manager.get_result(agent_id)
        results.append(result)
        
        # Add to accumulated context for next task
        accumulated_context[f"{task['role']}_result"] = result
    
    return results
```

### Hybrid Delegation Patterns

Real workflows often mix parallel and sequential execution. Maybe your research phase can run three searches in parallel, but the writing phase must wait for all research to complete, and editing must wait for writing. A smart coordinator handles this complexity.

```python
async def hybrid_delegation(workflow: dict, manager: SubagentManager) -> dict:
    """Execute a workflow with both parallel and sequential phases."""
    
    results = {}
    
    for phase in workflow["phases"]:
        phase_tasks = phase["tasks"]
        
        if phase.get("parallel", False):
            # Run all tasks in this phase concurrently
            phase_results = await parallel_delegation(phase_tasks, manager)
        else:
            # Run tasks sequentially
            phase_results = sequential_delegation(phase_tasks, manager)
        
        results[phase["name"]] = phase_results
        
        # Add phase results to context for next phase
        for task, result in zip(phase_tasks, phase_results):
            manager.update_shared_context(task["role"], result)
    
    return results
```

### Error Handling in Delegation

Subagents can fail. Networks time out, APIs return errors, and sometimes the subagent just gets confused. Robust delegation handles these failures gracefully.

```python
async def resilient_delegation(
    task: dict, 
    manager: SubagentManager,
    max_retries: int = 3,
    timeout_seconds: float = 60
) -> dict:
    """Delegate with retry logic and timeout handling."""
    
    for attempt in range(max_retries):
        try:
            agent_id = manager.spawn(
                role=task["role"],
                task=task["description"],
                context=task.get("context", {})
            )
            
            # Wait with timeout
            start_time = time.time()
            while manager.get_status(agent_id) == SubagentStatus.RUNNING:
                if time.time() - start_time > timeout_seconds:
                    manager.terminate(agent_id)
                    raise TimeoutError(f"Subagent {agent_id} timed out")
                await asyncio.sleep(0.5)
            
            status = manager.get_status(agent_id)
            if status == SubagentStatus.COMPLETED:
                return {"success": True, "result": manager.get_result(agent_id)}
            elif status == SubagentStatus.FAILED:
                error = manager.get_error(agent_id)
                if attempt < max_retries - 1:
                    continue  # Retry
                return {"success": False, "error": error}
        
        except TimeoutError as e:
            if attempt < max_retries - 1:
                continue
            return {"success": False, "error": str(e)}
    
    return {"success": False, "error": "Max retries exceeded"}
```

### Collecting and Integrating Results

The coordinator's job is to both delegate tasks *and* to make sense of what comes back. Subagent results often need to be synthesized, conflicts need to be resolved, and gaps need to be identified.

```python
def integrate_research_results(results: list[dict], llm) -> dict:
    """Synthesize results from multiple research subagents."""
    
    prompt = f"""You have research results from multiple specialists:

{format_results(results)}

Synthesize these into a coherent summary:
1. Identify key themes that appear across multiple sources
2. Note any contradictions or disagreements between sources
3. Highlight gaps where more research may be needed
4. Provide an overall conclusion

Be specific and cite which source each finding came from."""
    
    synthesis = llm.invoke(prompt)
    
    return {
        "synthesis": synthesis.content,
        "source_count": len(results),
        "sources": [r.get("source") for r in results]
    }
```

## Multi-Step Reasoning

Deep agents reason through complex problems step by step, exploring multiple angles when needed.

### Sequential Reasoning Chains

Sometimes you need to think through a problem one step at a time, with each step building on the previous. This is classic chain-of-thought reasoning extended across a longer horizon.

```python
def sequential_reasoning(problem: str, max_steps: int, llm) -> dict:
    """Reason through a problem step by step."""
    
    steps = []
    current_state = f"Problem: {problem}\n\nLet me think through this step by step."
    
    for i in range(max_steps):
        prompt = f"""{current_state}

Step {i + 1}: What's the next logical step in solving this problem?
Consider what we know, what we need to find out, and what follows logically.

If we have enough information to give a final answer, say "CONCLUSION: [answer]"
Otherwise, describe your reasoning for this step."""
        
        response = llm.invoke(prompt).content
        steps.append({"step": i + 1, "reasoning": response})
        
        if response.startswith("CONCLUSION:"):
            return {
                "conclusion": response.replace("CONCLUSION:", "").strip(),
                "steps": steps,
                "total_steps": i + 1
            }
        
        current_state += f"\n\nStep {i + 1}: {response}"
    
    return {
        "conclusion": None,
        "steps": steps,
        "total_steps": max_steps,
        "status": "max_steps_reached"
    }
```

### Parallel Reasoning Paths

For some problems, multiple reasoning approaches might lead to an answer. Rather than committing to one path, deep agents can explore several in parallel and compare results.

```python
async def parallel_reasoning(problem: str, approaches: list[str], llm) -> dict:
    """Try multiple reasoning approaches in parallel."""
    
    async def try_approach(approach: str):
        prompt = f"""Problem: {problem}

Approach this problem using: {approach}

Think through the problem using this approach and provide your conclusion.
Explain your reasoning clearly."""
        
        response = await llm.ainvoke(prompt)
        return {"approach": approach, "reasoning": response.content}
    
    results = await asyncio.gather(*[try_approach(a) for a in approaches])
    
    # Compare results
    comparison_prompt = f"""Problem: {problem}

Multiple reasoning approaches were tried:

{format_reasoning_results(results)}

Compare these approaches:
1. Do they reach the same conclusion?
2. Which approach seems most rigorous?
3. Are there any errors in reasoning?
4. What is the best final answer?"""
    
    comparison = llm.invoke(comparison_prompt)
    
    return {
        "approaches": results,
        "comparison": comparison.content
    }
```

### When to Stop Going Deeper

Deep reasoning is powerful but expensive. Every additional reasoning step costs time and tokens. Deep agents need heuristics for when to stop.

Some reasonable stopping conditions:

- **Confidence threshold**: The agent believes it has a reliable answer
- **Diminishing returns**: Recent steps aren't adding new insights
- **Step limit**: Hard cap to prevent runaway reasoning
- **Resource budget**: Time or cost constraints reached
- **Convergence**: Multiple approaches reach the same conclusion

Here's a practical implementation of adaptive stopping:

```python
def adaptive_reasoning(problem: str, llm, max_steps: int = 10) -> dict:
    """Reason with adaptive stopping based on confidence and progress."""
    
    steps = []
    confidence_history = []
    
    for i in range(max_steps):
        # Generate next reasoning step
        context = format_reasoning_history(steps)
        
        prompt = f"""Problem: {problem}

Reasoning so far:
{context if context else "None yet - this is step 1"}

Continue reasoning. After your analysis, rate your confidence (0.0-1.0)
that you can now give a correct final answer.

Format:
REASONING: Your next step of analysis
CONFIDENCE: 0.X
FINAL_ANSWER: [only if confidence >= 0.8] Your answer"""
        
        response = llm.invoke(prompt).content
        reasoning, confidence, answer = parse_reasoning_response(response)
        
        steps.append({"step": i + 1, "reasoning": reasoning})
        confidence_history.append(confidence)
        
        # Check stopping conditions
        if answer:  # Model provided final answer
            return {"answer": answer, "steps": steps, "confidence": confidence}
        
        if confidence >= 0.85:  # High confidence - prompt for answer
            final = get_final_answer(problem, steps, llm)
            return {"answer": final, "steps": steps, "confidence": confidence}
        
        # Check for plateau (no confidence improvement in 3 steps)
        if len(confidence_history) >= 3:
            recent = confidence_history[-3:]
            if max(recent) - min(recent) < 0.05:
                final = get_final_answer(problem, steps, llm)
                return {"answer": final, "steps": steps, "confidence": confidence, 
                        "note": "Stopped due to confidence plateau"}
    
    # Max steps reached
    final = get_final_answer(problem, steps, llm)
    return {"answer": final, "steps": steps, "confidence": confidence_history[-1],
            "note": "Max steps reached"}
```

The key insight is that reasoning should be goal-directed. You're reasoning, that is, to reach a reliable conclusion. Once you're confident enough, or once more reasoning isn't helping, it's time to commit to an answer.

## Self-Reflection and Metacognition

Here's where things get really interesting. Deep agents can think about their own thinking. They can evaluate whether they're making progress, critique their own outputs, and adjust strategies when something isn't working.

### Evaluating Progress Toward Goals

A deep agent working on a long project should periodically ask itself: "Am I actually making progress?" This prevents the agent from spinning its wheels on unproductive paths.

```python
def evaluate_progress(goal: str, plan: Plan, completed_tasks: list[str], llm) -> dict:
    """Assess progress toward the overall goal."""
    
    total_tasks = len(plan.tasks)
    completed_count = len(completed_tasks)
    
    prompt = f"""Goal: {goal}

Original plan:
{format_plan(plan)}

Completed tasks:
{format_completed(completed_tasks)}

Progress: {completed_count}/{total_tasks} tasks complete

Evaluate:
1. Are we on track to achieve the goal?
2. Have completed tasks actually moved us closer to the goal?
3. Are there any warning signs or blockers?
4. Should we adjust the remaining plan?

Be honest and specific."""
    
    evaluation = llm.invoke(prompt)
    
    return {
        "progress_fraction": completed_count / total_tasks,
        "evaluation": evaluation.content,
        "status": "on_track" if completed_count / total_tasks > 0.3 else "needs_review"
    }
```

### Strategy Adjustment Based on Results

Sometimes the original plan isn't working. Maybe research is turning up less useful information than expected. Maybe a venue fell through and you need to pivot. Deep agents adjust.

```python
def adjust_strategy(current_plan: Plan, evaluation: dict, llm) -> Plan:
    """Modify plan based on progress evaluation."""
    
    if "on_track" in evaluation.get("status", ""):
        return current_plan  # No adjustment needed
    
    prompt = f"""Current plan:
{format_plan(current_plan)}

Progress evaluation:
{evaluation['evaluation']}

The current approach isn't working well. Propose adjustments:
1. Which remaining tasks should be modified or replaced?
2. Are there new tasks that should be added?
3. Should any tasks be removed or deprioritized?
4. What's the revised strategy?

Provide a revised plan."""
    
    revised = llm.invoke(prompt)
    
    # In practice, you'd parse this into a new Plan object
    return parse_plan(revised.content)
```

### The Reflection Loop

Reflection is most powerful when built into the agent's core loop. After completing significant work, the agent evaluates and potentially revises before moving on.

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict

class ReflectiveState(TypedDict):
    goal: str
    plan: dict
    current_task_idx: int
    task_results: dict
    reflection: str
    needs_replanning: bool

def execute_task_node(state: ReflectiveState) -> ReflectiveState:
    """Execute the current task."""
    task = state["plan"]["tasks"][state["current_task_idx"]]
    result = execute_task(task)
    
    new_results = {**state["task_results"], task["id"]: result}
    return {"task_results": new_results}

def reflect_node(state: ReflectiveState) -> ReflectiveState:
    """Reflect on progress after task completion."""
    
    reflection_prompt = f"""Goal: {state['goal']}
    
Completed task: {state['plan']['tasks'][state['current_task_idx']]}
Result: {state['task_results']}

Reflect:
1. Did this task achieve what we expected?
2. Are we still on track for the overall goal?
3. Should we adjust the plan?

If the plan needs significant changes, say "REPLAN NEEDED".
Otherwise, say "CONTINUE"."""
    
    reflection = llm.invoke(reflection_prompt).content
    
    needs_replan = "REPLAN NEEDED" in reflection
    return {"reflection": reflection, "needs_replanning": needs_replan}

def replan_node(state: ReflectiveState) -> ReflectiveState:
    """Create a new plan based on reflection."""
    new_plan = adjust_strategy(state["plan"], {"evaluation": state["reflection"]}, llm)
    return {"plan": new_plan, "needs_replanning": False, "current_task_idx": 0}

def route_after_reflection(state: ReflectiveState) -> str:
    """Decide whether to continue, replan, or finish."""
    if state["needs_replanning"]:
        return "replan"
    if state["current_task_idx"] >= len(state["plan"]["tasks"]) - 1:
        return "finish"
    return "continue"

# Build the graph
graph = StateGraph(ReflectiveState)
graph.add_node("execute", execute_task_node)
graph.add_node("reflect", reflect_node)
graph.add_node("replan", replan_node)

graph.set_entry_point("execute")
graph.add_edge("execute", "reflect")
graph.add_conditional_edges("reflect", route_after_reflection, {
    "continue": "execute",
    "replan": "replan",
    "finish": END
})
graph.add_edge("replan", "execute")

reflective_agent = graph.compile()
```

## Backtracking and Recovery

Even with good planning and reflection, agents sometimes go down wrong paths. Deep agents need the ability to recognize dead ends and backtrack to try alternative approaches.

### Detecting Dead Ends

A dead end might look like:

- Repeated failures on the same task
- Circular dependencies that can't be resolved
- External constraints that make the current path impossible
- Diminishing confidence in the current approach

```python
def detect_dead_end(state: dict, history: list[dict]) -> dict:
    """Check if the agent is stuck."""
    
    # Check for repeated failures
    recent_failures = [h for h in history[-5:] if h.get("status") == "failed"]
    if len(recent_failures) >= 3:
        return {"is_stuck": True, "reason": "repeated_failures"}
    
    # Check for cycling (same task attempted multiple times)
    recent_tasks = [h.get("task_id") for h in history[-10:]]
    for task_id in set(recent_tasks):
        if recent_tasks.count(task_id) >= 3:
            return {"is_stuck": True, "reason": "task_cycling", "task": task_id}
    
    # Check for lack of progress
    if len(history) > 10:
        recent_progress = sum(1 for h in history[-10:] if h.get("progress_made"))
        if recent_progress < 2:
            return {"is_stuck": True, "reason": "no_progress"}
    
    return {"is_stuck": False}
```

### Rolling Back Decisions

When a dead end is detected, the agent needs to roll back to a known good state and try something different. This requires maintaining exploration history.

```python
@dataclass
class ExplorationState:
    id: str
    task_state: dict
    decision_made: str
    alternatives: list[str]
    timestamp: float

class ExplorationHistory:
    def __init__(self):
        self.states: list[ExplorationState] = []
        self.current_idx: int = -1
    
    def checkpoint(self, state: dict, decision: str, alternatives: list[str]):
        """Save a decision point we might want to return to."""
        exploration_state = ExplorationState(
            id=f"checkpoint_{len(self.states)}",
            task_state=state.copy(),
            decision_made=decision,
            alternatives=alternatives,
            timestamp=time.time()
        )
        self.states.append(exploration_state)
        self.current_idx = len(self.states) - 1
    
    def backtrack(self) -> tuple[dict, str] | None:
        """Return to most recent checkpoint with untried alternatives."""
        for i in range(self.current_idx, -1, -1):
            state = self.states[i]
            if state.alternatives:
                # Try next alternative
                next_alternative = state.alternatives.pop(0)
                self.current_idx = i
                return state.task_state, next_alternative
        
        return None  # No alternatives left
```

### Exploring Alternative Paths

When backtracking, the agent doesn't merely retry the same thing, it tries a different approach. The checkpoint system tracks which alternatives exist and which have been tried.

```python
def handle_dead_end(state: dict, history: ExplorationHistory, llm) -> dict:
    """Recover from a dead end by backtracking and trying alternatives."""
    
    backtrack_result = history.backtrack()
    
    if backtrack_result is None:
        # No alternatives left - report failure
        return {
            "status": "failed",
            "reason": "All alternatives exhausted"
        }
    
    restored_state, alternative = backtrack_result
    
    # Log the backtrack
    print(f"Backtracking to try alternative: {alternative}")
    
    # Continue from restored state with new approach
    return {
        "status": "backtracked",
        "restored_state": restored_state,
        "new_approach": alternative
    }
```

## The Claude Agent SDK

Before we move to implementation, it's worth mentioning Anthropic's Claude Agent SDK. While we've been building with LangGraph throughout this book, Anthropic has released their own framework for building agents with Claude models. It embodies many of the same deep agent principles we've discussed.

The Claude Agent SDK emphasizes a few key patterns:

**Agentic loops with explicit reasoning**: Claude agents are encouraged to think step-by-step, with reasoning traces that explain decisions. This aligns perfectly with the metacognition we discussed earlier. Every action is preceded by explicit thought, making the agent's decision-making process transparent.

**Tool use as first-class citizen**: The SDK provides clean abstractions for defining and using tools, with automatic schema generation and validation. You define what your tools do, and the SDK handles the plumbing: marshaling parameters, handling errors, and presenting results back to the model.

**Conversation memory management**: Built-in support for managing context across long conversations, including summarization when needed. The SDK tracks conversation history and can automatically compress older messages to stay within context limits.

**Computer use capabilities**: For agents that need to interact with GUIs, the SDK provides abstractions for screen interaction. This enables agents that can browse the web, fill out forms, and use desktop applications, capabilities that were previously the domain of specialized automation tools.

### Key Concepts in the Claude SDK

The SDK introduces several concepts worth understanding:

**Agents**: The core abstraction. An agent has a system prompt defining its role, a set of available tools, and configuration for memory and reasoning behavior.

**Tools**: Functions the agent can call. The SDK uses Python type hints to automatically generate schemas, making tool definition straightforward.

**Messages**: The conversation history. Messages flow between human, assistant, and tool roles, with the SDK managing the complexities of multi-turn conversations.

**Runs**: A single execution of the agent loop. A run takes a user message, processes it through potentially many agent iterations, and returns a final response.

Here's a taste of what Claude Agent SDK code looks like:

```python
# Conceptual example - check Anthropic docs for current API
from anthropic import Claude
from anthropic.tools import tool

@tool
def search_documents(query: str) -> str:
    """Search the knowledge base for relevant documents."""
    results = vector_store.similarity_search(query, k=3)
    return format_results(results)

@tool
def create_task(title: str, description: str, due_date: str) -> str:
    """Create a new task in the task management system."""
    task = task_manager.create(title=title, description=description, due=due_date)
    return f"Created task: {task.id}"

agent = Claude.agent(
    model="claude-sonnet-4-20250514",
    system="You are a helpful research assistant. Think step by step before taking actions.",
    tools=[search_documents, create_task],
    max_iterations=10
)

response = agent.run("Find information about RAG systems and create a task to study them")
```

### When to Choose Which Framework

When should you consider Claude's SDK versus LangGraph?

**Consider Claude's SDK when:**
- You're building exclusively with Claude models and want the tightest integration
- You want Anthropic's opinionated best practices baked in from the start
- Computer use (GUI interaction) is a primary use case for your agent
- You prefer a more batteries-included approach with less configuration
- You're prototyping quickly and want to minimize boilerplate

**Stick with LangGraph when:**
- You need model flexibility and might switch between OpenAI, Claude, and others
- You want fine-grained control over every aspect of agent architecture
- You're building complex multi-agent systems with custom coordination patterns
- You need deep integration with LangSmith for observability and debugging
- Your team already has LangChain expertise and tooling in place
- You need features like human-in-the-loop that LangGraph handles well

The good news is that the concepts are transferable. Deep agent patterns—planning, context management, delegation, reflection—work regardless of which framework you choose. Learn the principles well, and you can apply them anywhere. The framework is just the vehicle; the architecture is what matters.

## Building StudyBuddy v7

Alright, let's put all of this into practice. We're adding deep agent capabilities to StudyBuddy, and the star of the show is the new **Curriculum Planner Agent**.

### Where We Left Off

In v6, StudyBuddy gained persistence and memory. It remembers your learning history across sessions, tracks which concepts you've mastered versus struggled with, and uses the SM-2 algorithm for spaced repetition. The multi-agent team—Tutor, Card Generator, Quality Checker, and Scheduler—works together smoothly under the Learning Coordinator's supervision.

But here's what's missing: StudyBuddy is reactive. It helps you study AI engineering, but it doesn't help you figure out *what* to study or *in what order*. If you want to master AI engineering, you're on your own to decide the path.

### What We're Adding

The Curriculum Planner Agent transforms StudyBuddy from a tutoring assistant into a full learning system. Tell it your goal—"I want to be job-ready as an AI engineer in 12 weeks"—and it will:

1. Decompose that goal into a structured curriculum
2. Identify prerequisite chains (what you need to learn first)
3. Create a week-by-week study schedule
4. Delegate flashcard generation to the Card Generator for each topic
5. Set progress checkpoints with knowledge assessments
6. Track your actual progress and adjust the plan when needed
7. Handle backtracking when you struggle with a concept

This is deep agent territory. Planning, task decomposition, subagent delegation, context management across a multi-week project, and adaptive replanning based on your performance. Let's build it.

### The Curriculum Planner Agent

First, we define the planner's role and capabilities:

```python
# api/agents/curriculum_planner.py

CURRICULUM_PLANNER_PROMPT = """You are StudyBuddy's Curriculum Planner,
an expert at designing effective learning paths.

Your job: Take a learning goal and create a structured curriculum that
guides the student from where they are to where they want to be.

When creating a curriculum:
1. Break the goal into major topics or modules
2. Identify prerequisites - what must be learned first
3. Sequence topics logically based on dependencies
4. Estimate time needed for each topic
5. Define checkpoints to assess understanding

Use backward chaining: Start from the goal, identify what's needed to
achieve it, then what's needed for that, until you reach foundational
concepts the student likely already knows.

You have access to the AI Engineering reference materials for domain
knowledge about what topics exist and how they relate.

Output format - respond with JSON:
{
    "goal": "The learning objective",
    "estimated_duration_weeks": 8,
    "modules": [
        {
            "id": "m1",
            "title": "Module title",
            "description": "What this covers",
            "prerequisites": [],
            "topics": ["topic1", "topic2"],
            "estimated_hours": 10,
            "checkpoint": "How to assess completion"
        }
    ],
    "learning_path": ["m1", "m2", "m3"]
}"""

def create_curriculum_planner(model_name: str = "gpt-4o"):
    """Create the Curriculum Planner agent."""
    return ChatOpenAI(model=model_name, temperature=0.4)
```

### Task Decomposition for Learning Goals

When a user provides a goal like "Master RAG systems," the planner uses backward chaining to build a curriculum:

```python
def generate_curriculum(
    llm: ChatOpenAI,
    goal: str,
    current_knowledge: list[str] = None,
    available_time_hours: int = None,
    context: str = "",
) -> dict:
    """
    Generate a complete curriculum for a learning goal.
    
    Args:
        llm: The language model
        goal: What the user wants to learn
        current_knowledge: Topics user already knows
        available_time_hours: Weekly time commitment
        context: Retrieved knowledge base content
    
    Returns:
        Curriculum dictionary with modules and learning path
    """
    current_knowledge = current_knowledge or []
    
    user_content = f"""Create a curriculum for this learning goal: {goal}

Student's current knowledge: {', '.join(current_knowledge) if current_knowledge else 'Assume foundational programming knowledge'}

{'Weekly time available: ' + str(available_time_hours) + ' hours' if available_time_hours else ''}

Use backward chaining:
1. What does the student need to know to achieve this goal?
2. For each of those, what prerequisites are needed?
3. Continue until you reach topics the student already knows
4. Sequence everything based on dependencies

Reference material for AI engineering topics:
{context if context else 'Use your knowledge of AI engineering fundamentals.'}

Provide the curriculum as JSON."""

    messages = [
        SystemMessage(content=CURRICULUM_PLANNER_PROMPT),
        HumanMessage(content=user_content),
    ]
    
    response = llm.invoke(messages)
    
    try:
        content = response.content.strip()
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
            content = content.strip()
        
        return json.loads(content)
    
    except json.JSONDecodeError:
        return {"error": "Failed to parse curriculum", "raw": response.content}
```

### Subagent Delegation for Flashcard Generation

Once we have a curriculum, the planner delegates flashcard generation to our existing Card Generator agent. Each topic in each module gets its own set of cards:

```python
async def generate_curriculum_flashcards(
    curriculum: dict,
    card_generator_llm: ChatOpenAI,
    search_func,
    db,
) -> dict:
    """
    Generate flashcards for all topics in a curriculum.
    
    Uses the Card Generator agent as a subagent, delegating card
    creation for each topic.
    """
    from .card_generator import generate_cards_batch
    from ..services.flashcard_cache import cache_flashcards
    
    results = {"modules": {}, "total_cards": 0}
    
    for module in curriculum.get("modules", []):
        module_id = module["id"]
        module_cards = []
        
        for topic in module.get("topics", []):
            # Get context from knowledge base
            context = search_func(topic, k=4)
            
            # Delegate to Card Generator subagent
            cards = generate_cards_batch(
                card_generator_llm,
                topic=topic,
                context=context,
                count=5  # 5 cards per topic
            )
            
            if cards:
                # Cache the generated cards
                cached = cache_flashcards(
                    topic=topic,
                    source_context=context,
                    cards_data=cards,
                    db=db,
                    chapter_id=None  # Curriculum cards aren't chapter-bound
                )
                module_cards.extend(cached)
        
        results["modules"][module_id] = {
            "title": module["title"],
            "card_count": len(module_cards),
            "card_ids": [c.id for c in module_cards]
        }
        results["total_cards"] += len(module_cards)
    
    return results
```

### Progress Checkpoints and Assessments

The curriculum includes checkpoints, which are moments to assess whether the student has actually learned the material before moving on:

```python
def evaluate_checkpoint(
    llm: ChatOpenAI,
    module: dict,
    user_performance: dict,
    db,
) -> dict:
    """
    Evaluate whether user has passed a module checkpoint.
    
    Args:
        llm: Language model for evaluation
        module: The module being evaluated
        user_performance: Spaced repetition stats for this module's topics
        db: Database session
    
    Returns:
        Evaluation with pass/fail and recommendations
    """
    # Calculate performance metrics
    topics = module.get("topics", [])
    avg_ease = user_performance.get("average_ease_factor", 2.5)
    accuracy = user_performance.get("accuracy", 0)
    cards_studied = user_performance.get("cards_reviewed", 0)
    
    # Heuristic checkpoint evaluation
    passed = (
        accuracy >= 0.7 and  # 70% accuracy threshold
        avg_ease >= 2.3 and  # Cards aren't too difficult
        cards_studied >= len(topics) * 3  # Studied enough cards
    )
    
    # Get LLM assessment for nuance
    prompt = f"""Evaluate this student's checkpoint for module: {module['title']}

Checkpoint criteria: {module.get('checkpoint', 'Demonstrate understanding of key concepts')}

Performance data:
- Topics covered: {', '.join(topics)}
- Flashcard accuracy: {accuracy:.0%}
- Average ease factor: {avg_ease:.2f}
- Cards reviewed: {cards_studied}

Based on this data, has the student sufficiently mastered this module?
What specific areas might need more attention?

Respond with JSON:
{{"passed": true/false, "confidence": 0.0-1.0, "feedback": "...", "recommendations": ["..."]}}"""
    
    response = llm.invoke(prompt)
    
    try:
        evaluation = json.loads(response.content)
        evaluation["metrics"] = {
            "accuracy": accuracy,
            "ease_factor": avg_ease,
            "cards_studied": cards_studied
        }
        return evaluation
    except:
        return {
            "passed": passed,
            "confidence": 0.7,
            "feedback": "Checkpoint evaluated based on performance metrics.",
            "metrics": {"accuracy": accuracy, "ease_factor": avg_ease}
        }
```

### Context Summarization for Long Study Sessions

When a user has been studying for weeks, we need to keep context manageable. The planner periodically summarizes progress:

```python
def summarize_learning_progress(
    curriculum: dict,
    completed_modules: list[str],
    current_module: str,
    struggle_areas: list[str],
    llm: ChatOpenAI,
) -> str:
    """
    Create a compressed summary of learning progress.
    
    Used to maintain context without overwhelming the context window
    with full history.
    """
    progress_pct = len(completed_modules) / len(curriculum.get("modules", [])) * 100
    
    prompt = f"""Summarize this student's learning progress concisely:

Goal: {curriculum['goal']}
Progress: {progress_pct:.0f}% complete

Completed modules:
{chr(10).join(f'- {m}' for m in completed_modules) if completed_modules else '- None yet'}

Currently studying: {current_module}

Struggle areas: {', '.join(struggle_areas) if struggle_areas else 'None identified'}

Provide a 2-3 sentence summary capturing:
1. Overall progress status
2. Current focus area
3. Any concerns or adjustments needed"""
    
    summary = llm.invoke(prompt)
    return summary.content
```

### Managing Learning Context Across Sessions

A student might study for 30 minutes today, skip two days, then come back for an hour. The Curriculum Planner needs to maintain coherent context across these fragmented sessions. We do this by storing session summaries in the database and loading relevant context when the student returns.

```python
def get_session_context(user_id: str, curriculum_id: str, db) -> dict:
    """
    Load context for a returning student.
    
    Combines curriculum state, recent performance, and session history
    into a coherent context object.
    """
    # Get curriculum progress
    progress = db.query(CurriculumProgress).filter_by(
        user_id=user_id, 
        curriculum_id=curriculum_id
    ).first()
    
    # Get recent study sessions (last 5)
    recent_sessions = db.query(StudySession).filter_by(
        user_id=user_id
    ).order_by(StudySession.ended_at.desc()).limit(5).all()
    
    # Get performance metrics for current module
    current_module = progress.current_module if progress else None
    performance = get_module_performance(user_id, current_module, db) if current_module else {}
    
    # Build context
    return {
        "curriculum_goal": progress.goal if progress else None,
        "current_module": current_module,
        "modules_completed": progress.completed_modules if progress else [],
        "overall_progress": progress.progress_percentage if progress else 0,
        "recent_sessions": [
            {
                "date": s.ended_at.isoformat(),
                "duration_minutes": s.duration_minutes,
                "topics_studied": s.topics,
                "cards_reviewed": s.cards_reviewed
            }
            for s in recent_sessions
        ],
        "current_performance": performance,
        "days_since_last_session": calculate_days_since(recent_sessions[0].ended_at) if recent_sessions else None
    }

def resume_study_session(user_id: str, curriculum_id: str, db, llm) -> dict:
    """
    Resume a student's study session with appropriate context.
    
    Handles the case where a student returns after some time away.
    """
    context = get_session_context(user_id, curriculum_id, db)
    
    days_away = context.get("days_since_last_session")
    
    if days_away is None:
        # First session ever
        return {
            "message": "Welcome! Let's start your learning journey.",
            "action": "start_curriculum",
            "context": context
        }
    elif days_away > 7:
        # Been away a while - need review
        return {
            "message": f"Welcome back! It's been {days_away} days. Let's do a quick review before continuing.",
            "action": "review_then_continue",
            "review_topics": context["modules_completed"][-2:],  # Review recent modules
            "context": context
        }
    elif days_away > 2:
        # Brief break - gentle re-engagement
        summary = summarize_learning_progress(
            curriculum={"goal": context["curriculum_goal"], "modules": []},
            completed_modules=context["modules_completed"],
            current_module=context["current_module"],
            struggle_areas=[],
            llm=llm
        )
        return {
            "message": f"Welcome back! Here's where we left off: {summary}",
            "action": "continue",
            "context": context
        }
    else:
        # Recent session - just continue
        return {
            "message": f"Ready to continue with {context['current_module']}?",
            "action": "continue",
            "context": context
        }
```

This context management ensures that students get a personalized, coherent experience regardless of their study patterns. The agent adapts its behavior based on how long it's been since the last session and what the student was working on.

### Backtracking When Learning Doesn't Work

If a student keeps struggling with a module, the planner can backtrack and try a different approach:

```python
def handle_learning_struggle(
    curriculum: dict,
    struggling_module: dict,
    attempts: int,
    llm: ChatOpenAI,
) -> dict:
    """
    Handle when a student is stuck on a module.
    
    Returns adjusted plan with alternative approaches.
    """
    if attempts < 2:
        # First struggle: suggest review and more practice
        return {
            "action": "reinforce",
            "recommendation": "Review foundational concepts and practice more flashcards",
            "additional_cards_needed": 10
        }
    
    elif attempts < 4:
        # Continued struggle: try alternative learning approach
        prompt = f"""A student is struggling with this module despite multiple attempts:

Module: {struggling_module['title']}
Topics: {', '.join(struggling_module.get('topics', []))}
Attempts: {attempts}

Suggest alternative approaches:
1. Different ways to explain these concepts
2. Prerequisite topics that might need review
3. Simpler stepping stones to build up to this material

Provide JSON:
{{"alternative_approach": "...", "prerequisite_review": ["..."], "simplified_path": ["..."]}}"""
        
        response = llm.invoke(prompt)
        try:
            alternatives = json.loads(response.content)
            return {"action": "alternative_approach", **alternatives}
        except:
            return {"action": "escalate", "recommendation": "Consider 1-on-1 tutoring"}
    
    else:
        # Persistent struggle: backtrack to prerequisites
        return {
            "action": "backtrack",
            "recommendation": "Return to prerequisite modules",
            "target_modules": struggling_module.get("prerequisites", []),
            "reason": "Foundational gaps detected"
        }
```

### Memory-Connected Learning Modes

StudyBuddy v7 connects the tutoring and practice modes through persistent memory. When you struggle with flashcards, that information flows to the tutor. When the tutor identifies concepts you find challenging, those become focus areas for practice.

This bidirectional feedback loop is one of the hallmarks of a deep agent—it maintains context and learns from interactions over time, using that knowledge to personalize future responses.

#### Struggle Areas Flow to Tutoring

The scheduler tracks which topics you struggle with (less than 60% accuracy after 3+ reviews). These struggle areas are persisted to the memory store:

```python
# In scheduler.py - after calculating struggle_topics
memory_store = MemoryStore(db)
for topic in struggle_topics:
    stats = topic_stats[topic]
    accuracy = stats["correct"] / stats["total"]
    memory_store.put(
        user_id=user_id,
        namespace="struggles",
        key=topic.lower().replace(" ", "_"),
        value={
            "topic": topic,
            "accuracy": round(accuracy, 2),
            "total_reviews": stats["total"],
            "identified_at": datetime.utcnow().isoformat(),
        },
    )
```

When you ask the tutor a question, it loads your struggle areas to personalize explanations:

```python
# In tutor_node - before calling tutor_explain
with get_db() as db:
    memory_store = MemoryStore(db)
    struggles = memory_store.search(user_id, namespace="struggles", limit=5)
    preferences = memory_store.search(user_id, namespace="preferences", limit=3)

memory_context = format_memories_for_tutor(struggles, preferences)
explanation = tutor_explain(tutor_llm, query, context, card_context, memory_context)
```

The tutor system prompt includes these struggle areas, so it can be extra clear when explaining related concepts. If you've been struggling with "vector databases" in practice, and you ask the tutor about RAG systems, it knows to be especially thorough when explaining the vector search component.

#### Learning Insights Flow to Practice

After tutoring sessions, we extract memorable insights synchronously:

```python
# After generating an explanation
conversation_text = f"Student asked: {query}\n\nTutor explained: {explanation}"
memories = extract_memories_from_conversation(conversation_text, tutor_llm)
for memory in memories:
    memory_store.put(
        user_id=user_id,
        namespace=memory.get("namespace", "background"),
        key=memory.get("key", "unknown"),
        value=memory.get("value", {}),
    )
```

The extraction identifies:

- **Preferences**: How the student likes things explained (visual, step-by-step, etc.)
- **Goals**: What they're studying for (certification, job interview, etc.)
- **Struggles**: Concepts they expressed confusion about during tutoring
- **Background**: Relevant prior knowledge that helps contextualize explanations

This creates a true feedback loop: practice performance informs tutoring, and tutoring insights inform practice recommendations.

#### The Frontend Shows Focus Areas

The sidebar displays your current struggle areas (labeled "Focus Areas") so you can see what the system has identified. These update after each review, so you can watch topics move on and off the list as your performance improves.

The v7 frontend uses Next.js (consistent with v5 and v6) and introduces two new React components for curriculum functionality:

- **FocusAreas.tsx**: Displays struggle topics in the sidebar with "Chat about this" links that auto-open the chat panel with a pre-filled question about the topic.
- **CurriculumModal.tsx**: A modal dialog for creating learning paths. It collects the user's goal and weekly time commitment, shows a loading state while the curriculum is generated, and displays the resulting modules with progress indicators.

The curriculum state is persisted to localStorage so students can resume their learning path across browser sessions. The main page component (`page.tsx`) manages two study modes—`'chapter'` for traditional chapter-based studying and `'curriculum'` for goal-based learning paths.

This transparency helps students understand why the tutor is being especially thorough on certain topics—and gives them a clear sense of progress as they master difficult material.

### Putting It Together

The Curriculum Planner integrates with the existing StudyBuddy system through a new API endpoint and updated supervisor routing:

```python
# In api/index.py - new endpoint

class CurriculumRequest(BaseModel):
    goal: str
    current_knowledge: list[str] = []
    weekly_hours: int = 10

class CurriculumResponse(BaseModel):
    curriculum: dict
    flashcards_generated: int
    estimated_duration_weeks: int

@app.post("/api/curriculum", response_model=CurriculumResponse)
async def create_curriculum(request: CurriculumRequest):
    """Generate a personalized learning curriculum."""
    
    # Search for relevant context
    context = search_materials(request.goal, k=6)
    
    # Generate curriculum
    curriculum = generate_curriculum(
        curriculum_planner_llm,
        goal=request.goal,
        current_knowledge=request.current_knowledge,
        available_time_hours=request.weekly_hours,
        context=context,
    )
    
    if "error" in curriculum:
        raise HTTPException(status_code=500, detail=curriculum["error"])
    
    # Generate flashcards for all topics (async)
    with get_db() as db:
        card_results = await generate_curriculum_flashcards(
            curriculum=curriculum,
            card_generator_llm=card_generator_llm,
            search_func=search_materials,
            db=db,
        )
    
    return CurriculumResponse(
        curriculum=curriculum,
        flashcards_generated=card_results["total_cards"],
        estimated_duration_weeks=curriculum.get("estimated_duration_weeks", 8)
    )
```

### Testing Curriculum Generation

To verify everything works, we test the full flow:

```python
# Test curriculum generation
def test_curriculum_generation():
    response = client.post("/api/curriculum", json={
        "goal": "Understand RAG systems well enough to build production applications",
        "current_knowledge": ["Python", "APIs", "Basic machine learning"],
        "weekly_hours": 8
    })
    
    assert response.status_code == 200
    data = response.json()
    
    # Verify curriculum structure
    assert "curriculum" in data
    assert "modules" in data["curriculum"]
    assert len(data["curriculum"]["modules"]) > 0
    
    # Verify flashcards were generated
    assert data["flashcards_generated"] > 0
    
    # Verify learning path
    assert "learning_path" in data["curriculum"]
    
    print(f"Generated curriculum with {len(data['curriculum']['modules'])} modules")
    print(f"Total flashcards: {data['flashcards_generated']}")
    
    for module in data["curriculum"]["modules"]:
        print(f"  - {module['title']}: {module.get('estimated_hours', '?')} hours")
```

### Curriculum Study Integration

With the curriculum created and flashcards generated, we need a way to fetch cards specifically for the current module the student is studying. This requires a new endpoint and updates to the scheduler's topic filtering:

```python
# In api/index.py - curriculum flashcard endpoint

class CurriculumFlashcardRequest(BaseModel):
    curriculum_id: str
    module_id: str | None = None  # None = use current module
    current_topic: str | None = None  # For "Still Learning" flow
    previous_question: str | None = None

@app.post("/api/curriculum/flashcard", response_model=FlashcardResponse)
def get_curriculum_flashcard(
    request: CurriculumFlashcardRequest,
    db: Session = Depends(get_db),
):
    """Get a flashcard from the current curriculum module.

    Prioritizes spaced repetition (due cards first), then new cards,
    falling back to on-demand generation if no cards exist.
    """
    # 1. Get curriculum and progress
    curriculum = get_curriculum_by_id(db, request.curriculum_id)
    progress = get_curriculum_progress(db, request.curriculum_id, "default")

    # 2. Determine which module to use
    module_id = request.module_id or (progress.current_module_id if progress else None)
    current_module = find_module(curriculum, module_id)

    # 3. Get module's topics
    module_topics = current_module.get("topics", [])

    # 4. Try due cards first (spaced repetition priority)
    due_cards = get_due_cards(db, "default", limit=20, topic_filter=module_topics)
    if due_cards:
        selected = random.choice(due_cards)
        return FlashcardResponse(
            question=selected["question"],
            answer=selected["answer"],
            topic=selected["topic"],
            source="curriculum",
            flashcard_id=selected["flashcard_id"],
        )

    # 5. Then try new cards (not yet reviewed)
    new_cards = get_new_cards(db, "default", limit=20, topic_filter=module_topics)
    if new_cards:
        selected = random.choice(new_cards)
        return FlashcardResponse(...)

    # 6. No cards found - generate on-demand
    selected_topic = random.choice(module_topics)
    context = search_materials(selected_topic, k=4)
    card = generate_single_card(card_generator_llm, selected_topic, [], context)

    # Cache and return
    cache_flashcards(topic=selected_topic, cards_data=[card], db=db)
    return FlashcardResponse(...)
```

The key insight is that curriculum modules contain multiple topics, so we need the scheduler's `get_due_cards()` and `get_new_cards()` functions to accept a list of topics:

```python
# In api/agents/scheduler.py - updated topic filtering

def get_due_cards(
    db: Session,
    user_id: str = "default",
    limit: int = 10,
    topic_filter: str | list[str] | None = None,  # Now accepts list
) -> list[dict]:
    """Get flashcards due for review, optionally filtered by topic(s)."""

    query = (
        db.query(CardReview, Flashcard)
        .join(Flashcard, CardReview.flashcard_id == Flashcard.id)
        .filter(CardReview.user_id == user_id, CardReview.next_review <= now)
    )

    if topic_filter:
        if isinstance(topic_filter, list):
            # Multiple topics - use OR with case-insensitive matching
            from sqlalchemy import or_
            conditions = [Flashcard.topic.ilike(f"%{t}%") for t in topic_filter]
            query = query.filter(or_(*conditions))
        else:
            # Single topic
            query = query.filter(Flashcard.topic.ilike(f"%{topic_filter}%"))

    # ... rest of function unchanged
```

This multi-topic filtering enables curriculum study mode to draw from all topics within a module while still respecting spaced repetition priorities. The same update applies to `get_new_cards()`.

## A Note on the UI

The code examples above focus on the core multi-agent architecture. The actual StudyBuddy v7 implementation extends this with a flashcard-first user interface, where students see flashcards as their primary study mode, with chat available as a secondary feature for deeper exploration.

This required a few additions to the API layer:

- /api/chapters endpoint that parses a topic-list.md file for chapter/topic navigation.
- /api/flashcard endpoint that generates scoped flashcards using the Card Generator and Quality Checker agents.
- /api/curriculum endpoint that generates personalized learning paths with the Curriculum Planner agent.
- /api/curriculum/flashcard endpoint that fetches flashcards scoped to the current curriculum module.
- card_context parameter in the chat endpoint so the tutor knows what flashcard you're studying.

The chat feature focuses on tutoring—answering questions and explaining concepts—while flashcard generation is handled separately by the dedicated flashcard endpoints. This separation keeps chat responses clean and focused.

The v7 frontend uses a three-button review system (No / Took a sec / Yes) that maps to SM-2 spaced repetition quality ratings. Reviews are recorded via the /api/review endpoint, and the frontend refreshes focus areas after each review to show updated struggle topics.

The frontend also supports two study modes: chapter-based studying (select a chapter and scope) and curriculum-based studying (follow a personalized learning path). The CurriculumModal component handles learning path creation and display, while the Sidebar shows curriculum progress and focus areas during study sessions.

## Running locally

StudyBuddy v7 runs with two terminals—one for the backend, one for the frontend.

**Terminal 1 - Backend:**
```bash
cd v7-deep-agents
uv run uvicorn api.index:app --reload --port 8000
```

**Terminal 2 - Frontend:**
```bash
cd v7-deep-agents/frontend
npm run dev
```

Visit http://localhost:3000. The Next.js dev server proxies `/api/*` requests to the FastAPI backend on port 8000 (configured in `next.config.ts`).

## Deploying to Vercel

To deploy v7 to Vercel, you need to update two files in the repo root.

First, update vercel.json to point to the v7-deep-agents directory:

```json
{
    "version": 2,
    "builds": [
        {
            "src": "v7-deep-agents/api/index.py",
            "use": "@vercel/python"
        },
        {
            "src": "v7-deep-agents/frontend/package.json",
            "use": "@vercel/next"
        }
    ],
    "routes": [
        { "src": "/api/(.*)", "dest": "/v7-deep-agents/api/index.py" },
        { "src": "/(.*)", "dest": "/v7-deep-agents/frontend/$1" }
    ],
    "git": {
        "deploymentEnabled": {
            "main": true
        }
    },
    "ignoreCommand": "bash scripts/should-deploy.sh"
}
```

Second, update scripts/should-deploy.sh to trigger on v7 changes:

```bash
#!/bin/bash
# Deploy if changes are in v7-deep-agents, vercel config, or scripts
# Exit 1 (true) = proceed with build, Exit 0 (false) = skip build
git diff HEAD^ HEAD --name-only | grep -qE "^v7-deep-agents/|^vercel\.json|^scripts/" && exit 1 || exit 0
```

If you haven't done so already, set your OpenAI API key environment variable in Vercel:

```
vercel env add OPENAI_API_KEY
```

Deploy:

```
vercel --prod
```

Or simply push to GitHub. Vercel will automatically deploy when changes are detected in the v7-deep-agents directory.

### What's Next

We've built a curriculum planner that can take a high-level learning goal and turn it into a structured study plan with automatically generated flashcards. That's a major upgrade from reactive tutoring.

In Chapter 8, we'll shift focus to evaluation. How do we know if our agents are actually helping students learn? How do we test at scale? We'll dive into synthetic data generation for building evaluation datasets and set up the infrastructure for systematic testing and improvement.

You now have a deep agent that plans, delegates, tracks progress, and adapts. That's a serious capability to add to your toolkit. Let's keep building.
