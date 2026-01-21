# Chapter 10: Full Stack Applications

You've built something real. Over the last nine chapters, you've gone from basic LLM calls to sophisticated multi-agent systems with memory, evaluation infrastructure, and production-grade retrieval. That's serious engineering. But here's the thing: right now, your users interact with all of that power through a terminal or API endpoint. That's fine for developers, but it's not how you ship products.

This chapter is about crossing the finish line. We're transforming backend agent systems into full-stack applications that real people can actually use. Dashboards that visualize learning progress. Chat interfaces that stream responses in real-time. File upload workflows that let users bring their own content. The whole experience.

The patterns here apply far beyond any single project. Whether you're building a customer support assistant, a content generation tool, or a knowledge management system, the architecture stays the same. Frontend talks to backend, backend orchestrates agents, agents do the heavy lifting. Clean separation. Clear responsibilities. That's the move.

By the end of this chapter, you'll understand how production LLM applications are structured, how to build responsive user interfaces that handle streaming AI responses, and how to design databases that scale with your application. You'll also learn the user experience patterns that separate frustrating AI tools from delightful ones. Let's get after it.

## Industry Use Cases

Before we dive into architecture, let's talk about what's actually working in production. The LLM application landscape has matured significantly, and certain patterns have emerged as clear winners.

Customer support automation leads the pack in terms of deployed applications. Companies route incoming tickets through AI agents that can handle routine inquiries, gather information for complex cases, and escalate appropriately. The best implementations don't try to replace human agents entirely. Instead, they augment the team by handling the repetitive stuff and providing context when humans need to step in. A support agent seeing a summary of the customer's issue, relevant account history, and suggested solutions can resolve tickets faster than starting from scratch every time.

Content generation tools have found their groove in specific niches rather than general-purpose writing. Marketing teams use them for product descriptions, social media posts, and email variations. Documentation teams use them to draft technical content that subject matter experts then refine. The key insight is that AI-generated content works best when it's part of a workflow, not a replacement for human judgment. First draft from AI, refinement from humans, final review before publishing.

Knowledge management represents perhaps the most valuable enterprise use case. Organizations have decades of documentation scattered across wikis, shared drives, and individual hard drives. RAG-powered search lets employees find relevant information using natural language rather than guessing the exact keywords someone used five years ago. The applications that nail this combine semantic search with proper access controls, so people only see documents they're authorized to view.

Learning and training applications have exploded in both corporate and consumer contexts. Adaptive tutoring systems adjust difficulty based on performance. Onboarding assistants help new employees navigate company processes. Language learning apps provide conversational practice at any hour. These applications share a common pattern: they maintain state about the learner across sessions, personalize content based on observed behavior, and provide feedback that helps users improve.

The patterns that generalize across all these use cases are remarkably consistent:

- **Stream responses** to keep users engaged during generation
- **Provide transparency** about what the AI is doing and why
- **Handle failures gracefully** rather than showing cryptic error messages
- **Remember context** from previous interactions
- **Collect feedback** that helps improve the system over time

What doesn't work is equally instructive. Applications that try to hide the AI tend to backfire when users discover the deception. Overpromising capabilities leads to disappointment and churn. Ignoring latency makes experiences feel sluggish even when the underlying system is powerful. And shipping without proper evaluation means you're finding bugs through user complaints rather than systematic testing.

The lessons from successful deployments cluster around a few themes:

- **Start narrow and expand.** Companies that succeeded focused on a specific use case, nailed it, and then broadened scope. Those that tried to build general-purpose AI assistants from day one often ended up with mediocre performance across the board.
- **Invest in feedback loops early.** The applications that improve fastest have mechanisms for collecting user corrections and incorporating them into system improvements.
- **Set expectations clearly.** Users who understand what an AI can and can't do have dramatically higher satisfaction than those who discover limitations through failure.
- **Measure everything.** Teams that track latency, accuracy, user satisfaction, and cost per query make better decisions than those flying blind.

## Architecture Overview

Full-stack AI applications follow a three-layer architecture that keeps concerns cleanly separated. The frontend handles user interaction and display. The backend manages business logic, authentication, and orchestration. The agent layer performs AI operations. Each layer has distinct responsibilities, and the interfaces between them should be explicit and stable.

The frontend is everything the user sees and interacts with. In a web application, this means the React components, CSS styling, and client-side JavaScript that renders the interface. The frontend knows nothing about how agents work internally. It sends requests, receives responses, and displays results. This ignorance is intentional. When you upgrade your agent architecture, the frontend shouldn't need to change. When you redesign the UI, the agent logic stays untouched.

The backend sits between frontend and agents, handling concerns that neither should worry about:

- **Authentication** verifies that requests come from legitimate users
- **Rate limiting** protects your API budget from runaway usage
- **Request validation** ensures that malformed inputs don't reach your agents
- **Background job scheduling** manages long-running tasks

The backend exposes a clean REST or WebSocket API to the frontend while orchestrating potentially complex agent interactions behind the scenes.

The agent layer is where AI operations happen. Retrieval, generation, tool use, multi-step reasoning—all of that lives here. Agents receive structured inputs and return structured outputs. They don't know about HTTP status codes or session cookies. They don't handle authentication. They just do their job: take a question and context, return an answer. This isolation makes agents easier to test, easier to swap out, and easier to reason about.

State management spans all three layers, and getting it right is critical:

- **Frontend state** covers UI concerns: which panel is open, what the user typed in the draft field, whether a request is in flight. This state is ephemeral and can be reconstructed if lost.
- **Backend state** includes session and orchestration data: which user is authenticated, what background jobs are running, what conversations are in progress.
- **Agent state** maintains conversation memory and retrieved context: what was discussed previously, what documents are relevant.

Each layer owns its state and synchronizes with others through well-defined interfaces.

The pattern of clean separation pays dividends as your application grows. When you need to support mobile apps alongside your web interface, you add a new frontend without touching backend or agents. When you want to swap OpenAI for an open-source model, you modify the agent layer without touching frontend or backend. When you need to add team features with shared workspaces, you modify the backend without touching agents. Modularity enables evolution.

Communication between layers happens through well-defined APIs. The frontend talks to the backend via REST endpoints for standard operations and WebSockets for real-time streaming. The backend talks to agents through Python function calls or service classes. These boundaries should be explicit in your codebase—a `services/` directory for agent wrappers, an `api/` directory for endpoint definitions, clean separation between concerns.

One common mistake is letting agent concerns leak into the backend. When your route handlers start containing prompt templates or retrieval logic, you've violated the separation. The fix is ruthless: extract agent logic into dedicated modules, expose clean interfaces, and keep your route handlers thin. Another mistake is putting business logic in the frontend. Validation, authorization, rate limiting—these belong in the backend where they can't be bypassed by clever users inspecting your JavaScript.

## Frontend Components

Modern AI applications demand interfaces that go far beyond the terminal REPL you've been using for development. Users expect responsive, polished experiences that feel like the other software they use daily. That means proper loading states, streaming text, intuitive navigation, and mobile support.

Chat interfaces are the bread and butter of conversational AI applications. The basic pattern involves a message list showing the conversation history and an input field for new messages. Key elements to include:

- **Visual distinction** between user inputs and AI responses through different background colors, alignment, or avatars
- **Timestamps** to help users orient themselves in long conversations
- **Markdown rendering** so AI responses can include formatted code, lists, and emphasis without displaying raw syntax
- **Auto-scroll** that follows new messages but allows users to scroll up through history

```javascript
function ChatMessage({ message, isUser }) {
    const containerClass = isUser 
        ? 'chat-message user-message' 
        : 'chat-message assistant-message';
    
    return (
        <div className={containerClass}>
            <div className="message-content">
                {isUser ? (
                    <p>{message.content}</p>
                ) : (
                    <MarkdownRenderer content={message.content} />
                )}
            </div>
            <span className="message-timestamp">
                {formatTimestamp(message.createdAt)}
            </span>
        </div>
    );
}
```

Streaming responses deserve special attention because they fundamentally change how users perceive AI interactions. A response that takes ten seconds to generate feels much faster when users see it appear word by word than when they stare at a spinner for ten seconds and then see everything at once. The implementation requires consuming a stream from your backend and progressively updating the displayed content.

```javascript
function useStreamingChat() {
    const [messages, setMessages] = useState([]);
    const [isStreaming, setIsStreaming] = useState(false);
    const [currentResponse, setCurrentResponse] = useState('');
    
    const sendMessage = async (userMessage) => {
        // Add user message immediately
        setMessages(prev => [...prev, { 
            role: 'user', 
            content: userMessage 
        }]);
        setIsStreaming(true);
        setCurrentResponse('');
        
        try {
            const response = await fetch('/api/chat/stream', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ message: userMessage })
            });
            
            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let accumulated = '';
            
            while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                
                const chunk = decoder.decode(value);
                const lines = chunk.split('\n');
                
                for (const line of lines) {
                    if (line.startsWith('data: ') && line.slice(6) !== '[DONE]') {
                        accumulated += line.slice(6);
                        setCurrentResponse(accumulated);
                    }
                }
            }
            
            // Finalize the assistant message
            setMessages(prev => [...prev, { 
                role: 'assistant', 
                content: accumulated 
            }]);
        } finally {
            setIsStreaming(false);
            setCurrentResponse('');
        }
    };
    
    return { messages, currentResponse, isStreaming, sendMessage };
}
```

Dashboards and analytics views help users understand their data and progress over time. Charts showing activity trends, performance metrics, or content statistics provide value that pure chat interfaces can't match. The key is choosing the right visualization for each metric. Line charts work for trends over time. Bar charts compare discrete categories. Progress bars show completion toward goals. Pie charts should generally be avoided—they're harder to read than bar charts and rarely the right choice.

When designing dashboards for AI applications, focus on actionable insights rather than vanity metrics. Users don't need to know how many tokens they've consumed—they need to know whether they're making progress toward their goals. Frame metrics in terms of user outcomes: topics mastered, questions answered correctly, time saved. Include trend indicators that show whether things are getting better or worse. And provide clear calls to action: if accuracy is declining, suggest review sessions; if a topic is overdue, surface it prominently.

The data pipeline for dashboards often catches teams off guard. Calculating metrics in real-time from raw event data is expensive at scale. The pattern is to compute aggregates on write or in batch jobs, storing pre-calculated metrics that dashboards can query cheaply. For a learning application, you might update daily statistics whenever a user completes a review session rather than scanning all historical sessions on every dashboard load.

User feedback mechanisms let your users tell you when things go wrong or right. The simplest version is thumbs up and thumbs down buttons on AI responses. More sophisticated implementations let users edit AI responses to show what should have been said, flag responses as inappropriate, or rate responses on multiple dimensions. This feedback becomes training data for improving your system and helps you identify failure modes you might not catch in testing.

Mobile-responsive design is non-negotiable for consumer applications. Users expect to access your app from their phones, and experiences that require horizontal scrolling or have touch targets too small to tap reliably will drive them away. Use responsive CSS that adapts layouts to screen size. Ensure touch targets are at least 44 pixels in each dimension. Test on actual devices, not just browser dev tools, because the feel of an interface changes when you're holding it in your hand.

## Backend Architecture

FastAPI has become the framework of choice for AI application backends, and for good reason. It's fast, it handles async operations naturally, and its automatic OpenAPI documentation makes frontend integration smoother. The patterns you establish here determine how well your application scales and how pleasant it is to maintain.

Request handling in an AI backend differs from traditional CRUD applications because of response times. A typical database query returns in milliseconds. An LLM call might take several seconds. That mismatch means you need to think carefully about timeouts, streaming, and background processing.

```python
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import asyncio

app = FastAPI()

class ChatRequest(BaseModel):
    message: str
    conversation_id: str | None = None

@app.post("/api/chat/stream")
async def stream_chat(request: ChatRequest):
    """Stream a chat response using Server-Sent Events."""
    
    async def generate():
        try:
            # Initialize or retrieve conversation
            conversation = await get_or_create_conversation(
                request.conversation_id
            )
            
            # Stream from the agent
            async for chunk in agent.astream(
                message=request.message,
                conversation_id=conversation.id
            ):
                if chunk.content:
                    yield f"data: {chunk.content}\n\n"
            
            yield "data: [DONE]\n\n"
            
        except Exception as e:
            yield f"data: {{\"error\": \"{str(e)}\"}}\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive"
        }
    )
```

WebSocket integration enables bidirectional real-time communication that Server-Sent Events can't match. When you need the client to send messages while the server is still streaming a response, or when you want to push notifications to clients without them polling, WebSockets are the answer. The tradeoff is complexity: WebSocket connections require more careful state management, and handling reconnection gracefully takes effort.

```python
from fastapi import WebSocket, WebSocketDisconnect
from typing import Dict
import json

class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
    
    async def connect(self, websocket: WebSocket, user_id: str):
        await websocket.accept()
        self.active_connections[user_id] = websocket
    
    def disconnect(self, user_id: str):
        if user_id in self.active_connections:
            del self.active_connections[user_id]
    
    async def send_message(self, user_id: str, message: dict):
        if user_id in self.active_connections:
            await self.active_connections[user_id].send_json(message)

manager = ConnectionManager()

@app.websocket("/ws/{user_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: str):
    await manager.connect(websocket, user_id)
    
    try:
        while True:
            data = await websocket.receive_json()
            
            if data["type"] == "chat":
                # Stream response back through WebSocket
                async for chunk in agent.astream(data["message"]):
                    await manager.send_message(user_id, {
                        "type": "chunk",
                        "content": chunk.content
                    })
                
                await manager.send_message(user_id, {"type": "done"})
                
    except WebSocketDisconnect:
        manager.disconnect(user_id)
```

Background job processing handles tasks that take too long for a synchronous request-response cycle. Indexing uploaded documents, generating comprehensive reports, running batch evaluations—these operations might take minutes rather than seconds. The pattern is to accept the request, queue the job, return a job ID immediately, and let clients poll for completion or receive a notification when it's done.

```python
from fastapi import BackgroundTasks
import uuid

# In-memory job store (use Redis in production)
jobs = {}

class JobStatus(BaseModel):
    id: str
    status: str  # pending, processing, completed, failed
    result: dict | None = None
    error: str | None = None

@app.post("/api/documents/upload")
async def upload_document(
    file: UploadFile,
    background_tasks: BackgroundTasks
):
    """Accept document upload and queue indexing."""
    
    job_id = str(uuid.uuid4())
    jobs[job_id] = JobStatus(id=job_id, status="pending")
    
    # Save file
    content = await file.read()
    file_path = save_uploaded_file(content, file.filename)
    
    # Queue background processing
    background_tasks.add_task(
        index_document_task,
        job_id=job_id,
        file_path=file_path
    )
    
    return {"job_id": job_id, "status": "pending"}

async def index_document_task(job_id: str, file_path: str):
    """Background task to index a document."""
    
    jobs[job_id].status = "processing"
    
    try:
        # Parse document
        documents = await parse_document(file_path)
        
        # Chunk and embed
        chunks = await chunk_documents(documents)
        
        # Store in vector database
        await vector_store.add_documents(chunks)
        
        jobs[job_id].status = "completed"
        jobs[job_id].result = {
            "chunks_indexed": len(chunks),
            "file_path": file_path
        }
        
    except Exception as e:
        jobs[job_id].status = "failed"
        jobs[job_id].error = str(e)

@app.get("/api/jobs/{job_id}")
async def get_job_status(job_id: str):
    """Check the status of a background job."""
    
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return jobs[job_id]
```

Queue management becomes important as your application scales. The simple in-memory approach works for development, but production systems need persistent queues that survive server restarts and can distribute work across multiple workers. Redis with RQ or Celery are popular choices. The key insight is that your queue is a commitment to your users: if they upload a document, you need to process it even if your server restarts before the job completes.

For AI applications specifically, queue design needs to account for variable job duration. An LLM call might take one second or thirty seconds depending on output length and model load. Long-running jobs shouldn't block shorter ones. The solution is often separate queues for different job types:

- **Fast queue** for chat responses and quick lookups
- **Slow queue** for document indexing and content generation
- **Batch queue** for nightly processing and bulk operations

Workers can be allocated to queues based on expected load patterns.

Timeout handling requires special attention. HTTP requests typically timeout after 30-60 seconds, but generating a comprehensive curriculum might take longer. The pattern is to accept the initial request synchronously, return a job ID immediately, and switch to async polling or WebSocket updates for results. Never leave users hanging on a request that might timeout—always provide feedback and a path to completion.

Health checks and graceful shutdown round out production backend concerns. Your load balancer needs to know when instances are ready to receive traffic and when they're draining in preparation for shutdown. Kubernetes or your deployment platform uses these signals to route requests appropriately. A health endpoint that checks database connectivity and agent availability prevents routing traffic to broken instances.

## State Management

State management in full-stack AI applications spans multiple layers, each with different persistence requirements and synchronization challenges. Getting this right determines whether your app feels responsive and reliable or sluggish and buggy.

Client-side state covers everything that lives in the user's browser. UI state like which tabs are selected, whether a modal is open, or what the user has typed in a text field. This state should be managed with whatever framework you're using—React's useState and useReducer, Vue's reactive system, or vanilla JavaScript if you're keeping it simple. The key principle is that client state should be reconstructable. If the user refreshes the page, some state loss is acceptable. They might need to reopen a panel or retype an unsent message, but that's a minor inconvenience.

For state that should survive page refreshes but doesn't need server persistence, localStorage provides a simple solution. Draft messages that users haven't sent yet, user preferences like dark mode, recently accessed items—all good candidates for localStorage. The pattern is to sync with localStorage on changes and restore from localStorage on page load.

```javascript
function usePersistentState(key, initialValue) {
    const [value, setValue] = useState(() => {
        const stored = localStorage.getItem(key);
        return stored ? JSON.parse(stored) : initialValue;
    });
    
    useEffect(() => {
        localStorage.setItem(key, JSON.stringify(value));
    }, [key, value]);
    
    return [value, setValue];
}

// Usage
const [draftMessage, setDraftMessage] = usePersistentState(
    'draft-message', 
    ''
);
```

Server-side state is the source of truth for anything that matters. User profiles, conversation history, generated content, progress data—all of this lives in your database. PostgreSQL remains the right choice for most applications. It's reliable, well-understood, and capable of handling both structured data and JSON blobs when you need flexibility. The discipline of treating the server as the source of truth simplifies debugging and enables features like multi-device sync automatically.

State synchronization between client and server requires careful thought. The naive approach is to fetch fresh data from the server on every interaction, but that creates unnecessary latency and load. A better approach uses optimistic updates: when the user takes an action, update the UI immediately as if it succeeded, send the request to the server, and roll back only if the server reports failure.

```javascript
async function sendMessage(content) {
    // Optimistic update: add message to UI immediately
    const tempId = `temp-${Date.now()}`;
    const optimisticMessage = {
        id: tempId,
        content,
        status: 'sending'
    };
    
    setMessages(prev => [...prev, optimisticMessage]);
    
    try {
        // Actually send to server
        const response = await api.sendMessage(content);
        
        // Replace optimistic message with real one
        setMessages(prev => prev.map(msg => 
            msg.id === tempId 
                ? { ...response.data, status: 'sent' }
                : msg
        ));
        
    } catch (error) {
        // Mark as failed, let user retry
        setMessages(prev => prev.map(msg =>
            msg.id === tempId
                ? { ...msg, status: 'failed', error: error.message }
                : msg
        ));
    }
}
```

Handling offline scenarios gracefully improves user experience significantly, especially on mobile where connections can be flaky. The minimal version detects when the network is unavailable and shows a banner. A more sophisticated approach queues actions taken while offline and replays them when connectivity returns. For AI applications specifically, consider caching recent responses so users can review previous conversations even without a connection.

The boundary between client and server state can blur in complex applications. A useful heuristic: if losing the data would be merely inconvenient, it can live on the client. If losing it would be a problem—user work, progress data, payment information—it belongs on the server with proper persistence. When in doubt, persist to the server. Disk space is cheap; user frustration is expensive.

## Agent Integration

Connecting your frontend and backend to your agent layer requires thoughtful interface design. The goal is clean abstraction: your backend shouldn't need to know the internal structure of your agents, and your agents shouldn't know they're being called from a web application.

The service layer pattern wraps your agents in classes that expose simple, typed interfaces. This abstraction insulates the rest of your application from changes in agent implementation and makes testing straightforward.

```python
from typing import AsyncIterator
from dataclasses import dataclass

@dataclass
class ChatResponse:
    content: str
    sources: list[str]
    confidence: float

class AgentService:
    """Service layer wrapping agent operations."""
    
    def __init__(self, agent, retriever):
        self.agent = agent
        self.retriever = retriever
    
    async def chat(
        self, 
        message: str, 
        conversation_id: str
    ) -> ChatResponse:
        """Synchronous chat returning complete response."""
        
        result = await self.agent.ainvoke({
            "messages": [("user", message)],
            "conversation_id": conversation_id
        })
        
        return ChatResponse(
            content=result["messages"][-1].content,
            sources=result.get("sources", []),
            confidence=result.get("confidence", 1.0)
        )
    
    async def chat_stream(
        self,
        message: str,
        conversation_id: str
    ) -> AsyncIterator[str]:
        """Streaming chat yielding chunks."""
        
        async for event in self.agent.astream(
            {"messages": [("user", message)], 
             "conversation_id": conversation_id},
            stream_mode="updates"
        ):
            for node_name, update in event.items():
                messages = update.get("messages", [])
                for msg in messages:
                    if hasattr(msg, "content") and msg.content:
                        yield msg.content
```

Long-running operations need special handling. When a user asks for something that takes thirty seconds—generating a comprehensive study plan, analyzing a large document, running a multi-step research task—you can't just leave them staring at a spinner. Progress updates keep users engaged and reassured.

```python
@dataclass
class ProgressUpdate:
    stage: str
    message: str
    progress: float  # 0.0 to 1.0

async def generate_curriculum_with_progress(
    topic: str,
    depth: str
) -> AsyncIterator[ProgressUpdate | dict]:
    """Generate curriculum with progress updates."""
    
    yield ProgressUpdate(
        stage="analyzing",
        message="Analyzing topic requirements...",
        progress=0.1
    )
    
    # Analyze the topic
    analysis = await analyze_topic(topic)
    
    yield ProgressUpdate(
        stage="structuring",
        message="Creating curriculum structure...",
        progress=0.3
    )
    
    # Generate structure
    structure = await generate_structure(analysis, depth)
    
    yield ProgressUpdate(
        stage="generating",
        message="Generating detailed content...",
        progress=0.5
    )
    
    # Generate content for each section
    sections = []
    for i, section in enumerate(structure.sections):
        content = await generate_section_content(section)
        sections.append(content)
        
        progress = 0.5 + (0.4 * (i + 1) / len(structure.sections))
        yield ProgressUpdate(
            stage="generating",
            message=f"Generated {section.title}",
            progress=progress
        )
    
    yield ProgressUpdate(
        stage="complete",
        message="Curriculum generation complete!",
        progress=1.0
    )
    
    # Final result
    yield {
        "type": "result",
        "curriculum": {
            "topic": topic,
            "sections": sections
        }
    }
```

Error handling in agent integrations needs to distinguish between different failure modes. Network errors are transient and should trigger retries. Rate limit errors need backoff. Authentication errors need user action. And agent errors—cases where the AI genuinely doesn't know how to help—need graceful degradation and helpful messages.

The retry logic deserves careful calibration. Retrying immediately after a rate limit doesn't help—you need to wait. Retrying a request that failed due to bad input is pointless—you need to inform the user. Retrying indefinitely wastes resources and leaves users waiting forever. The typical pattern combines three strategies:

1. **Exponential backoff**: Wait longer after each consecutive failure
2. **Maximum retry count**: Give up after N attempts rather than retrying forever
3. **Circuit breaking**: If too many requests fail in a window, stop trying temporarily to let the service recover

```python
from tenacity import (
    retry, 
    stop_after_attempt, 
    wait_exponential,
    retry_if_exception_type
)

class RetryableError(Exception):
    """Error that should trigger a retry."""
    pass

class UserActionRequired(Exception):
    """Error requiring user intervention."""
    pass

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(RetryableError)
)
async def call_agent_with_retry(agent, message: str):
    """Call agent with automatic retry on transient failures."""
    
    try:
        return await agent.ainvoke({"messages": [("user", message)]})
        
    except RateLimitError:
        raise RetryableError("Rate limited, retrying...")
        
    except AuthenticationError:
        raise UserActionRequired("Please check your API key")
        
    except Exception as e:
        # Log for debugging, return graceful fallback
        logger.error(f"Agent error: {e}")
        return {
            "messages": [("assistant", 
                "I encountered an issue processing your request. "
                "Could you try rephrasing or breaking it into smaller parts?"
            )]
        }
```

## User Experience Considerations

The difference between an AI application that feels magical and one that feels frustrating often comes down to user experience details. Loading states, feedback mechanisms, and accessibility aren't afterthoughts—they're core to whether people actually want to use what you've built.

Loading states deserve careful design because AI operations take longer than users expect from modern software. The worst experience is a frozen interface with no indication that anything is happening. Slightly better is a generic spinner. Better still is a skeleton screen showing the shape of the content that will appear. Best is a progress indicator with stage information so users know what's happening.

```javascript
function LoadingState({ stage, progress }) {
    const stages = [
        { key: 'analyzing', label: 'Analyzing your request' },
        { key: 'retrieving', label: 'Finding relevant information' },
        { key: 'generating', label: 'Generating response' }
    ];
    
    return (
        <div className="loading-state" role="status" aria-live="polite">
            <div className="progress-bar">
                <div 
                    className="progress-fill" 
                    style={{ width: `${progress * 100}%` }}
                />
            </div>
            <ul className="stage-list">
                {stages.map((s, i) => (
                    <li 
                        key={s.key}
                        className={getStageClass(s.key, stage, stages, i)}
                    >
                        {s.label}
                    </li>
                ))}
            </ul>
        </div>
    );
}

function getStageClass(stageKey, currentStage, stages, index) {
    const currentIndex = stages.findIndex(s => s.key === currentStage);
    if (index < currentIndex) return 'stage-complete';
    if (index === currentIndex) return 'stage-active';
    return 'stage-pending';
}
```

Transparency about AI behavior builds trust. Users should understand that they're interacting with an AI, what the AI is capable of, and where its information comes from. Showing source citations for RAG applications lets users verify claims. Explaining when the AI is uncertain helps set appropriate expectations. And indicating when a response was generated versus retrieved from cache helps users understand what they're seeing.

Progressive disclosure manages complexity by showing users only what they need at each moment. The initial interface might show just a chat input and response. Users who want more can expand panels showing source documents, confidence scores, or intermediate reasoning steps. Power users get access to advanced configuration without overwhelming newcomers. The principle is to optimize for the common case while supporting edge cases.

Accessibility isn't optional—it's required for any application serving a broad audience and often required by law for enterprise software. The fundamentals include:

- **Semantic HTML elements** that screen readers understand
- **ARIA labels** for interactive components that lack visible text
- **Keyboard navigation** for everything a mouse can do
- **Sufficient color contrast** for users with visual impairments

```javascript
function AccessibleChatInput({ onSend, disabled }) {
    const [message, setMessage] = useState('');
    const inputRef = useRef(null);
    
    const handleSubmit = (e) => {
        e.preventDefault();
        if (message.trim() && !disabled) {
            onSend(message);
            setMessage('');
        }
    };
    
    const handleKeyDown = (e) => {
        // Submit on Enter, but allow Shift+Enter for newlines
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSubmit(e);
        }
    };
    
    return (
        <form onSubmit={handleSubmit} className="chat-input-form">
            <label htmlFor="chat-input" className="visually-hidden">
                Type your message
            </label>
            <textarea
                id="chat-input"
                ref={inputRef}
                value={message}
                onChange={(e) => setMessage(e.target.value)}
                onKeyDown={handleKeyDown}
                disabled={disabled}
                placeholder="Type your message..."
                aria-describedby="chat-input-hint"
                rows={1}
            />
            <span id="chat-input-hint" className="visually-hidden">
                Press Enter to send, Shift+Enter for new line
            </span>
            <button 
                type="submit" 
                disabled={disabled || !message.trim()}
                aria-label="Send message"
            >
                <SendIcon aria-hidden="true" />
            </button>
        </form>
    );
}
```

Feedback mechanisms close the loop between users and your system. Thumbs up and thumbs down buttons on responses provide signal about quality. A text field for explaining what went wrong helps diagnose issues. And making it easy to report problems—without requiring users to describe the entire context—increases the likelihood that they'll actually report rather than silently churning.

Error messages deserve special attention in AI applications because failures are more common and more confusing than in traditional software. When a database query fails, you can tell users exactly what went wrong. When an LLM generates a nonsensical response, the failure mode is less clear. Good error handling distinguishes between infrastructure errors (API rate limits, network failures) and capability errors (the AI genuinely doesn't know how to help). Infrastructure errors should trigger retries or clear user guidance. Capability errors should prompt graceful degradation—alternative suggestions, offers to rephrase, or escalation to human support.

The tone of error messages matters more than developers often realize. "Error: request failed" tells users nothing. "I'm having trouble connecting right now. Want to try again?" maintains the conversational feel while providing actionable guidance. Match error message tone to the rest of your interface. If your AI speaks warmly and helpfully, your errors shouldn't suddenly turn cold and technical.

Consider how errors affect user trust. A single catastrophic failure—losing a user's work, generating offensive content, exposing private information—can permanently damage the relationship. Defensive programming that prevents the worst outcomes is worth significant investment even if those outcomes are rare. Autosave user inputs before sending to the AI. Filter outputs before displaying. Validate that responses make sense before presenting them as answers.

## Database Design for Agent Apps

Database schema design for AI applications follows the same principles as traditional applications, with a few AI-specific considerations. You're storing user data, tracking progress, persisting generated content, and enabling analytics. The schema you design now determines how easily you can add features later.

User and session management form the foundation. Even before you implement authentication, designing your schema with user isolation in mind pays dividends. Every piece of user-generated or user-specific data should have a user_id foreign key. This makes multi-tenancy a schema addition rather than a rewrite.

```python
from sqlalchemy import (
    Column, String, Integer, Float, DateTime, 
    ForeignKey, Text, JSON, create_engine
)
from sqlalchemy.orm import declarative_base, relationship
from datetime import datetime
import uuid

Base = declarative_base()

def generate_uuid():
    return str(uuid.uuid4())

class User(Base):
    __tablename__ = 'users'
    
    id = Column(String, primary_key=True, default=generate_uuid)
    email = Column(String, unique=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    workspaces = relationship("Workspace", back_populates="user")
    conversations = relationship("Conversation", back_populates="user")


class Workspace(Base):
    """A workspace containing documents and configuration."""
    __tablename__ = 'workspaces'
    
    id = Column(String, primary_key=True, default=generate_uuid)
    user_id = Column(String, ForeignKey('users.id'), nullable=False)
    name = Column(String, nullable=False)
    description = Column(Text)
    settings = Column(JSON)  # Workspace-specific configuration
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="workspaces")
    documents = relationship("Document", back_populates="workspace")
```

Document storage tracks uploaded files and their indexed content. The actual vector embeddings live in your vector database, but metadata about documents belongs in your relational database where you can query and join it easily.

```python
class Document(Base):
    """Uploaded document metadata."""
    __tablename__ = 'documents'
    
    id = Column(String, primary_key=True, default=generate_uuid)
    workspace_id = Column(String, ForeignKey('workspaces.id'), nullable=False)
    filename = Column(String, nullable=False)
    file_type = Column(String)  # pdf, md, txt
    file_size = Column(Integer)
    file_path = Column(String)  # Storage location
    
    # Indexing status
    status = Column(String, default='pending')  # pending, processing, indexed, failed
    chunks_count = Column(Integer)
    indexed_at = Column(DateTime)
    error_message = Column(Text)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    workspace = relationship("Workspace", back_populates="documents")
```

Conversation history enables continuity across interactions. Storing the full conversation lets you resume where users left off and provides training data for improving your system.

```python
class Conversation(Base):
    """Chat conversation with message history."""
    __tablename__ = 'conversations'
    
    id = Column(String, primary_key=True, default=generate_uuid)
    user_id = Column(String, ForeignKey('users.id'), nullable=False)
    workspace_id = Column(String, ForeignKey('workspaces.id'))
    title = Column(String)
    
    # Conversation metadata
    started_at = Column(DateTime, default=datetime.utcnow)
    last_activity = Column(DateTime, default=datetime.utcnow)
    message_count = Column(Integer, default=0)
    
    # Relationships
    user = relationship("User", back_populates="conversations")
    messages = relationship("Message", back_populates="conversation")


class Message(Base):
    """Individual message in a conversation."""
    __tablename__ = 'messages'
    
    id = Column(String, primary_key=True, default=generate_uuid)
    conversation_id = Column(String, ForeignKey('conversations.id'), nullable=False)
    role = Column(String, nullable=False)  # user, assistant
    content = Column(Text, nullable=False)
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    tokens_used = Column(Integer)
    model = Column(String)
    
    # For feedback tracking
    feedback = Column(String)  # positive, negative, null
    feedback_text = Column(Text)
    
    # Relationships
    conversation = relationship("Conversation", back_populates="messages")
```

Generated content storage is important for AI applications that create artifacts—reports, summaries, analyses, or other outputs that users want to keep. Storing these in the database enables retrieval, versioning, and caching.

```python
class GeneratedContent(Base):
    """AI-generated content that users want to keep."""
    __tablename__ = 'generated_content'
    
    id = Column(String, primary_key=True, default=generate_uuid)
    user_id = Column(String, ForeignKey('users.id'), nullable=False)
    workspace_id = Column(String, ForeignKey('workspaces.id'))
    content_type = Column(String, nullable=False)  # report, summary, analysis
    title = Column(String)
    content = Column(Text, nullable=False)
    
    # Caching metadata
    input_hash = Column(String, index=True)  # For cache lookups
    
    # Generation metadata
    model = Column(String)
    tokens_used = Column(Integer)
    generation_time_ms = Column(Integer)
    
    created_at = Column(DateTime, default=datetime.utcnow)
```

This schema design anticipates growth. Adding authentication means adding password hashes and tokens to the User model, not restructuring everything. Adding team features means adding a Team model and team_id foreign keys. The foundation supports evolution.

## Building StudyBuddy v10

Here's where it all comes together. We're transforming StudyBuddy from a backend system with a basic interface into a full-stack application that users actually want to use. More importantly, we're generalizing it so users can learn any subject—not just AI engineering.

Let's take stock of where we left off. StudyBuddy v9 delivered production-grade retrieval with semantic chunking, hybrid search, reranking, and RAG-Fusion. The evaluation infrastructure from v8 proved these improvements with actual metrics. But users still interact through a basic web interface designed for testing, not for learning. And the system only works with the AI engineering reference guides we've been using all along.

Version 10 changes both of those things. The major feature is generalizing StudyBuddy to learn any subject. Users can upload their own documents, create custom topic lists, and manage multiple learning programs simultaneously. Want to study Spanish vocabulary in the morning and AWS certification in the afternoon? You can do that. The second major change is building a proper full-stack UI with dashboards, progress tracking, and polished interactions.

### Learning Program Architecture

The core concept is a **Learning Program**—a container that holds everything related to studying a particular subject. Each program has its own knowledge base (uploaded documents indexed in their own Qdrant collection), its own topic list defining what to study, its own flashcards, and its own progress tracking. Users can create as many programs as they want and switch between them freely.

```python
from dataclasses import dataclass
from typing import Optional
from datetime import datetime

@dataclass
class LearningProgram:
    id: str
    user_id: str
    name: str
    description: Optional[str]
    topic_list: dict  # Structured topic hierarchy
    collection_name: str  # Qdrant collection for this program
    created_at: datetime
    
    @property
    def qdrant_collection(self) -> str:
        """Collection name for this program's vectors."""
        return f"program_{self.id}"
```

The retrieval infrastructure from Chapter 9—hybrid search, reranking, RAG-Fusion—adapts naturally to program-scoped collections. Each program gets its own retriever instance configured with the program's Qdrant collection:

```python
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

def get_program_retriever(program: LearningProgram):
    """Create a retriever for a program's knowledge base.

    This reuses the v9 retrieval infrastructure with a
    program-specific Qdrant collection.
    """
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    client = QdrantClient(url="http://localhost:6333")

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=program.qdrant_collection,
        embedding=embeddings
    )

    # Use adaptive retrieval from Chapter 9
    # Simple queries get hybrid search, complex queries get RAG-Fusion
    return AdaptiveRetriever(
        vector_store=vector_store,
        embeddings=embeddings,
        collection_name=program.qdrant_collection
    )
```

When a user creates a new program, they have two options. They can upload their own topic list in markdown format following the same structure we've used throughout the book. Or they can let the AI generate a curriculum for them. The curriculum generator from Chapter 7 gets repurposed here—give it a topic and depth level, and it creates a comprehensive learning plan.

The curriculum generation flow demonstrates how AI can bootstrap content that users then customize. The generated curriculum is a starting point, not a final product. Users can edit topics, add their own materials, and adjust the structure. This human-in-the-loop approach produces better results than either pure AI generation or purely manual creation. The AI handles the tedious work of structuring a comprehensive curriculum; the human provides domain expertise and personal learning preferences.

For complex topics, curriculum generation benefits from additional context. When a user asks to learn "machine learning," the results are better if we ask clarifying questions: What's your math background? Are you interested in research or application? Do you have a specific project in mind? These questions let us tailor the curriculum to the learner rather than generating generic content.

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

curriculum_prompt = ChatPromptTemplate.from_template("""
You are an expert curriculum designer. Create a comprehensive 
learning curriculum for the topic: {topic}

Depth level: {depth} (beginner, intermediate, or advanced)

Structure your response as a markdown document with:
- Chapters using # Chapter N: Title format
- Topics using ## Topic Name format  
- Subtopics using - Subtopic format

Include {chapter_count} chapters covering the subject systematically, 
from foundational concepts to advanced applications.

Focus on practical, actionable learning objectives.
""")

async def generate_curriculum(
    topic: str, 
    depth: str = "intermediate",
    chapter_count: int = 8
) -> str:
    """Generate a curriculum for any topic."""
    
    llm = ChatOpenAI(model="gpt-4o", temperature=0.7)
    
    chain = curriculum_prompt | llm
    
    result = await chain.ainvoke({
        "topic": topic,
        "depth": depth,
        "chapter_count": chapter_count
    })
    
    return result.content
```

### Document Upload and Indexing

Users upload documents to build their program's knowledge base. The system accepts PDF, Markdown, and plain text files. Each upload triggers a background job that parses the document, chunks it using the semantic chunking from Chapter 9, generates embeddings, and stores everything in the program's dedicated Qdrant collection.

The upload interface uses drag-and-drop for a modern feel. Users can drop multiple files at once, and each file shows its indexing status: uploading, processing, indexed, or failed. A progress indicator shows how indexing is progressing for large documents. And notifications alert users when indexing completes so they can start studying immediately.

File size limits protect your infrastructure from abuse. A reasonable limit for most applications is 10-50MB per file. Documents larger than that are often better split into chapters or sections anyway. The limit should be communicated clearly in the UI so users don't waste time uploading files that will be rejected.

Duplicate detection saves users from accidentally indexing the same document twice. Compute a hash of the file content on upload and check against existing documents in the program. If a duplicate is detected, inform the user and offer options: skip the duplicate, replace the existing document, or index anyway with a different name.

```python
from fastapi import UploadFile, BackgroundTasks
from pathlib import Path
import aiofiles
import hashlib

UPLOAD_DIR = Path("uploads")
ALLOWED_TYPES = {"application/pdf", "text/markdown", "text/plain"}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB

@app.post("/api/programs/{program_id}/documents")
async def upload_document(
    program_id: str,
    file: UploadFile,
    background_tasks: BackgroundTasks,
    user: User = Depends(get_current_user)
):
    """Upload a document to a learning program."""

    # Verify program belongs to user
    program = await get_program(program_id)
    if program.user_id != user.id:
        raise HTTPException(status_code=403, detail="Not your program")

    # Validate file type
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed: PDF, Markdown, plain text"
        )

    # Read and validate file size
    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"File exceeds {MAX_FILE_SIZE // 1024 // 1024}MB limit"
        )

    # Check for duplicates
    content_hash = hashlib.sha256(content).hexdigest()
    existing = await get_document_by_hash(program_id, content_hash)
    if existing:
        raise HTTPException(
            status_code=409,
            detail=f"Document already uploaded as '{existing.filename}'"
        )

    # Save uploaded file
    file_path = UPLOAD_DIR / program_id / file.filename
    file_path.parent.mkdir(parents=True, exist_ok=True)

    async with aiofiles.open(file_path, 'wb') as f:
        await f.write(content)

    # Create document record
    document = Document(
        program_id=program_id,
        filename=file.filename,
        file_type=file.content_type,
        file_size=len(content),
        file_path=str(file_path),
        content_hash=content_hash,
        status="pending"
    )
    await save_document(document)

    # Queue indexing
    background_tasks.add_task(
        index_document,
        document_id=document.id,
        program_id=program_id
    )

    return {
        "document_id": document.id,
        "status": "pending",
        "message": "Document uploaded. Indexing in progress."
    }
```

The indexing job uses the advanced retrieval pipeline from v9. Semantic chunking respects document structure. The chunks get embedded and stored in a program-specific Qdrant collection, keeping each program's knowledge base isolated.

```python
from langchain_community.document_loaders import (
    PyPDFLoader, 
    UnstructuredMarkdownLoader,
    TextLoader
)
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from langchain_openai import OpenAIEmbeddings

async def index_document(document_id: str, program_id: str):
    """Index a document into the program's vector store."""
    
    document = await get_document(document_id)
    program = await get_program(program_id)
    
    try:
        # Update status
        document.status = "processing"
        await save_document(document)
        
        # Load document based on type
        file_path = document.file_path
        if document.file_type == "application/pdf":
            loader = PyPDFLoader(file_path)
        elif file_path.endswith(".md"):
            loader = UnstructuredMarkdownLoader(file_path)
        else:
            loader = TextLoader(file_path)
        
        raw_docs = loader.load()
        
        # Semantic chunking from v9
        chunks = semantic_chunk(raw_docs)
        
        # Add metadata
        for chunk in chunks:
            chunk.metadata["document_id"] = document_id
            chunk.metadata["program_id"] = program_id
            chunk.metadata["filename"] = document.filename
        
        # Store in program-specific collection
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        
        vector_store = QdrantVectorStore.from_documents(
            documents=chunks,
            embedding=embeddings,
            collection_name=program.qdrant_collection,
            url="http://localhost:6333"
        )
        
        # Update document status
        document.status = "indexed"
        document.chunks_count = len(chunks)
        document.indexed_at = datetime.utcnow()
        await save_document(document)
        
    except Exception as e:
        document.status = "failed"
        document.error_message = str(e)
        await save_document(document)
        raise
```

### Flashcards and Spaced Repetition

Each learning program maintains its own flashcard collection. The flashcard model extends the spaced repetition system from earlier chapters with program scoping:

```python
class Flashcard(Base):
    """Flashcard scoped to a learning program."""
    __tablename__ = 'flashcards'

    id = Column(String, primary_key=True, default=generate_uuid)
    program_id = Column(String, ForeignKey('learning_programs.id'), nullable=False)
    topic = Column(String, nullable=False)
    question = Column(Text, nullable=False)
    answer = Column(Text, nullable=False)
    source_context = Column(Text)
    content_hash = Column(String, index=True)  # For deduplication

    # Spaced repetition fields (SM-2 algorithm)
    ease_factor = Column(Float, default=2.5)
    interval = Column(Integer, default=0)  # Days until next review
    repetitions = Column(Integer, default=0)
    next_review = Column(DateTime)

    created_at = Column(DateTime, default=datetime.utcnow)

    program = relationship("LearningProgram", back_populates="flashcards")
```

The due cards endpoint returns flashcards ready for review, filtered by program:

```python
@app.get("/api/programs/{program_id}/due-cards")
async def get_due_cards(
    program_id: str,
    limit: int = 10,
    user: User = Depends(get_current_user)
):
    """Get flashcards due for review in a program."""
    program = await get_program(program_id, user.id)

    async with get_session() as session:
        result = await session.execute(
            select(Flashcard)
            .where(
                Flashcard.program_id == program_id,
                Flashcard.next_review <= datetime.utcnow()
            )
            .order_by(Flashcard.next_review)
            .limit(limit)
        )
        due_cards = result.scalars().all()

    return {"cards": due_cards, "count": len(due_cards)}


@app.post("/api/programs/{program_id}/flashcards/{card_id}/review")
async def review_flashcard(
    program_id: str,
    card_id: str,
    quality: int,  # 0-5 rating from user
    user: User = Depends(get_current_user)
):
    """Record a flashcard review and update spaced repetition schedule."""
    await get_program(program_id, user.id)  # Verify ownership

    card = await get_flashcard(card_id, program_id)

    # SM-2 algorithm update
    if quality >= 3:  # Correct response
        if card.repetitions == 0:
            card.interval = 1
        elif card.repetitions == 1:
            card.interval = 6
        else:
            card.interval = int(card.interval * card.ease_factor)
        card.repetitions += 1
    else:  # Incorrect response
        card.repetitions = 0
        card.interval = 1

    # Update ease factor
    card.ease_factor = max(1.3, card.ease_factor + 0.1 - (5 - quality) * 0.08)
    card.next_review = datetime.utcnow() + timedelta(days=card.interval)

    await save_flashcard(card)
    return {"next_review": card.next_review, "interval": card.interval}
```

### The Full-Stack Interface

The frontend provides a complete learning experience. The main dashboard shows all learning programs with progress summaries. Each program has its own view with detailed analytics, document management, and study tools.

The program selector lets users switch between programs with a single click. Progress charts show performance over time. The document panel displays uploaded materials with indexing status. And the study interface—chat and flashcards—adapts to whichever program is currently selected.

The frontend uses React with Next.js for server-side capabilities and optimized development. Initialize a new project with `npx create-next-app frontend`, then install dependencies with `npm install`. The examples below assume this setup, using React's hooks for state management and the Fetch API for backend communication.

```javascript
function Dashboard() {
    const [programs, setPrograms] = useState([]);
    const [selectedProgram, setSelectedProgram] = useState(null);
    const [view, setView] = useState('overview'); // overview, study, documents
    
    useEffect(() => {
        loadPrograms();
    }, []);
    
    const loadPrograms = async () => {
        const response = await fetch('/api/programs');
        const data = await response.json();
        setPrograms(data.programs);
        
        if (data.programs.length > 0 && !selectedProgram) {
            setSelectedProgram(data.programs[0]);
        }
    };
    
    return (
        <div className="dashboard">
            <Sidebar 
                programs={programs}
                selectedProgram={selectedProgram}
                onSelectProgram={setSelectedProgram}
                onCreateProgram={handleCreateProgram}
            />
            
            <main className="main-content">
                <Header 
                    program={selectedProgram}
                    view={view}
                    onViewChange={setView}
                />
                
                {view === 'overview' && (
                    <ProgramOverview program={selectedProgram} />
                )}
                
                {view === 'study' && (
                    <StudyInterface program={selectedProgram} />
                )}
                
                {view === 'documents' && (
                    <DocumentManager program={selectedProgram} />
                )}
            </main>
        </div>
    );
}
```

The progress dashboard shows learning analytics per program. Charts display review accuracy over time, topic mastery levels, and upcoming reviews. Users can see at a glance which subjects need attention and which they've mastered.

```javascript
function ProgramOverview({ program }) {
    const [stats, setStats] = useState(null);
    
    useEffect(() => {
        if (program) {
            loadStats(program.id);
        }
    }, [program]);
    
    const loadStats = async (programId) => {
        const response = await fetch(`/api/programs/${programId}/stats`);
        const data = await response.json();
        setStats(data);
    };
    
    if (!stats) return <LoadingSkeleton />;
    
    return (
        <div className="program-overview">
            <div className="stats-grid">
                <StatCard 
                    title="Topics Studied"
                    value={stats.topicsStudied}
                    total={stats.totalTopics}
                />
                <StatCard 
                    title="Flashcards Reviewed"
                    value={stats.cardsReviewed}
                    subtitle="this week"
                />
                <StatCard 
                    title="Accuracy"
                    value={`${stats.accuracy}%`}
                    trend={stats.accuracyTrend}
                />
                <StatCard 
                    title="Due for Review"
                    value={stats.dueForReview}
                    urgent={stats.dueForReview > 20}
                />
            </div>
            
            <div className="charts-row">
                <ProgressChart data={stats.progressHistory} />
                <TopicMasteryChart data={stats.topicMastery} />
            </div>
            
            <UpcomingReviews reviews={stats.upcomingReviews} />
        </div>
    );
}
```

### Streaming Responses

Long generation tasks—creating flashcards for a new topic, generating a curriculum, answering complex questions—use WebSocket streaming to keep users engaged. The interface shows what's happening in real-time rather than making users wait for completion.

The tutor agent wraps the program's retriever in a LangGraph workflow that retrieves context, generates responses, and streams output:

```python
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage

def create_tutor_agent(retriever):
    """Create a tutor agent for a specific program's knowledge base.

    This simplified version shows the core pattern. The full implementation
    in v9 includes multiple specialized agents (card generator, quality
    checker, scheduler) coordinated by a supervisor.
    """
    llm = ChatOpenAI(model="gpt-4o", streaming=True)

    async def retrieve_context(state):
        """Retrieve relevant documents for the question."""
        last_message = state["messages"][-1]
        docs = await retriever.ainvoke(last_message.content)
        context = "\n\n".join(doc.page_content for doc in docs)
        return {"context": context}

    async def generate_response(state):
        """Generate a response using retrieved context."""
        context = state.get("context", "")
        messages = state["messages"]

        system_prompt = f"""You are a helpful tutor. Use the following context
to answer questions accurately and helpfully.

Context:
{context}

If the context doesn't contain relevant information, say so honestly."""

        response = await llm.ainvoke([
            ("system", system_prompt),
            *messages
        ])
        return {"messages": [response]}

    # Build the graph
    workflow = StateGraph(dict)
    workflow.add_node("retrieve", retrieve_context)
    workflow.add_node("generate", generate_response)

    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)

    return workflow.compile()
```

The WebSocket endpoint uses this agent to stream responses:

```python
@app.websocket("/ws/study/{program_id}")
async def study_websocket(
    websocket: WebSocket, 
    program_id: str
):
    """WebSocket endpoint for real-time study interactions."""
    
    await websocket.accept()
    
    # Get program and initialize retriever
    program = await get_program(program_id)
    retriever = get_program_retriever(program)
    agent = create_tutor_agent(retriever)
    
    try:
        while True:
            data = await websocket.receive_json()
            
            if data["type"] == "chat":
                # Stream chat response
                async for chunk in agent.astream(
                    {"messages": [("user", data["message"])]},
                    stream_mode="updates"
                ):
                    for node_name, update in chunk.items():
                        messages = update.get("messages", [])
                        for msg in messages:
                            if hasattr(msg, "content") and msg.content:
                                await websocket.send_json({
                                    "type": "chunk",
                                    "content": msg.content
                                })
                
                await websocket.send_json({"type": "done"})
            
            elif data["type"] == "generate_card":
                # Stream flashcard generation
                await websocket.send_json({
                    "type": "status",
                    "message": "Generating flashcard..."
                })
                
                card = await generate_flashcard(
                    program_id=program_id,
                    topic=data["topic"]
                )
                
                await websocket.send_json({
                    "type": "flashcard",
                    "card": card
                })
                
    except WebSocketDisconnect:
        pass
```

### User-Scoped Data Isolation

Even though full authentication comes in Chapter 12, we design for user isolation now. Every database query filters by user_id. Every Qdrant collection is namespaced by program. This means adding authentication later is a configuration change, not a rewrite.

```python
async def get_user_programs(user_id: str) -> list[LearningProgram]:
    """Get all programs for a user."""
    
    async with get_session() as session:
        result = await session.execute(
            select(LearningProgram)
            .where(LearningProgram.user_id == user_id)
            .order_by(LearningProgram.updated_at.desc())
        )
        return result.scalars().all()

async def get_program(
    program_id: str, 
    user_id: str
) -> LearningProgram:
    """Get a specific program, verifying ownership."""
    
    async with get_session() as session:
        result = await session.execute(
            select(LearningProgram)
            .where(
                LearningProgram.id == program_id,
                LearningProgram.user_id == user_id
            )
        )
        program = result.scalar_one_or_none()
        
        if not program:
            raise HTTPException(
                status_code=404, 
                detail="Program not found"
            )
        
        return program
```

### Running the Application

With all the pieces in place, here's how to run StudyBuddy v10. The setup assumes you have PostgreSQL and Qdrant running locally.

First, install dependencies:

```bash
cd v10-full-stack

# Install Python dependencies
uv sync

# Install frontend dependencies
cd frontend
npm install
cd ..
```

Configure your environment by creating a `.env` file:

```bash
# .env
OPENAI_API_KEY=sk-your-key-here
POSTGRES_URL=postgresql://localhost/studybuddy
QDRANT_URL=http://localhost:6333
```

Start Qdrant if it's not already running:

```bash
docker run -p 6333:6333 qdrant/qdrant
```

The database tables are created automatically on startup. Open a second terminal and run the following:

```bash
# Check if database exists
psql -l | grep studybuddy

# If no output, create it
createdb studybuddy
```

Now start the API server on the second terminal you just opened:

```bash
# Terminal 2: Start the API server
cd v10-full-stack
uv run uvicorn api.index:app --reload --port 8000
```

Finally, open a third terminal and start the frontend dev server:

```bash
# Terminal 3: Start the frontend dev server
cd v10-full-stack/frontend
npm run dev
```

Open http://localhost:3000 in your browser. You should see the StudyBuddy interface ready to create your first learning program.

### What We've Built

StudyBuddy v10 is a transformation. Users can now learn any subject by uploading their own materials and generating custom curricula. The full-stack interface provides real progress tracking, document management, and a polished study experience. Streaming keeps users engaged during long operations. And the architecture is ready for multi-user deployment in Chapter 12.

Let's enumerate the key features:

- **Learning programs**: Users create separate programs for different subjects, each with its own knowledge base, topic list, and progress tracking.
- **Document upload**: PDF, Markdown, and text files can be uploaded and automatically indexed into program-specific vector collections.
- **AI curriculum generation**: Users can generate comprehensive curricula for any topic by describing what they want to learn.
- **Progress dashboard**: Charts and statistics show learning progress over time, upcoming reviews, and topic mastery levels.
- **Streaming chat**: Real-time streaming of AI responses keeps users engaged during generation.
- **Background processing**: Long-running tasks like document indexing happen asynchronously without blocking the UI.
- **User-scoped isolation**: All data is associated with user IDs, preparing for multi-user deployment.

The patterns here generalize completely. Document upload and indexing works for any knowledge base application. Program management works for any multi-workspace scenario. The streaming infrastructure works for any long-running AI operation. The progress tracking schema works for any application that needs analytics.

### What's Next

We've got a complete application that users can actually use to learn any subject. But it's still a single-user system running locally. In Chapter 11, we'll add MCP connectors that let users import content from external sources—GitHub repositories, Notion workspaces, Google Drive. That expands what users can learn from without manual file uploads. Then in Chapter 12, we deploy to production with proper authentication and multi-user support.

You've now built a full-stack AI application from the ground up. The agent layer from previous chapters, the retrieval infrastructure from Chapter 9, the UI patterns from this chapter—they all come together into something you could actually ship. That's the whole point. Build, ship, share.

Let's keep building!
