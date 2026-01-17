# Building StudyBuddy v6

Alright, here's where it all comes together. We've covered the theory of memory systems, explored formation strategies, and discussed caching for performance. Now let's put it into practice by adding memory and persistence to StudyBuddy.

## Where We Left Off

In Chapter 5, we built a multi-agent system with a Tutor Agent, Card Generator Agent, Quality Checker Agent, and a Learning Coordinator (supervisor) that orchestrates them. The system can explain concepts and generate flashcards, but it has a significant limitation: it forgets everything the moment you restart the server. Ask it what you studied yesterday? No idea. Start a new session? Complete blank slate.

Version 6 fixes that. We're adding persistent memory that survives across sessions, flashcard caching that avoids redundant generation, and a new Scheduler Agent that implements spaced repetition. When we're done, StudyBuddy will remember what you've studied, how well you've performed, and when you should review each topic next.

## What We're Adding

This version introduces several new capabilities that work together to create a personalized learning experience:

- **Persistence with PostgreSQL:** All data—memories, flashcards, learning history—persists in a real database. Restart the server, come back tomorrow, everything is still there.

- **LangGraph native memory:** Using langmem for user-specific memories about learning preferences, struggle areas, and progress.

- **Content-addressed flashcard caching:** Generated flashcards are cached by content hash. Same material, instant results, with no regeneration needed.

- **Scheduler Agent:** A new agent that tracks review performance and implements the SM-2 spaced repetition algorithm to optimize learning.

- **Learning history tracking:** Complete record of what concepts you've studied, how you performed, and what needs attention.

## Persistence Strategy

We're using PostgreSQL for all persistent storage, with the same database engine in development and production. Locally, you'll run PostgreSQL via Homebrew. On Vercel, you'll use their managed PostgreSQL service. Your code doesn't change between environments; only the connection string differs.

This approach eliminates the "works on my machine" problem. When something works locally, it works in production. When you debug an issue locally, you're debugging against the same database engine you're running in production. No surprises.

Your database contains everything: user profiles, memories, flashcard cache, and learning history. Locally, it's your Homebrew PostgreSQL instance. In production, it's your Vercel Postgres database. The application code is identical.

## Database Configuration

Our database setup requires the POSTGRES_URL environment variable, failing fast if it's missing rather than silently using a wrong database:

```python
# api/database/connection.py
import os
from contextlib import contextmanager
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from .models import Base, User

# PostgreSQL connection (required)
POSTGRES_URL = os.environ.get("POSTGRES_URL")
if not POSTGRES_URL:
    raise RuntimeError(
        "POSTGRES_URL environment variable is required. "
        "Set it in your .env file, e.g.: POSTGRES_URL=postgresql://localhost/studybuddy"
    )

# Fix URL scheme for SQLAlchemy 2.0 (Vercel uses postgres://)
DATABASE_URL = POSTGRES_URL.replace("postgres://", "postgresql://", 1)

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    pool_size=5,
    max_overflow=10,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

@contextmanager
def get_db():
    """Context manager for database sessions."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_db_dependency():
    """FastAPI dependency for database sessions."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
```

## Custom Memory Store

For user-specific memories about learning preferences and progress, we use a custom SQLAlchemy-based memory store. This gives us full control without external dependencies:

```python
# api/services/memory_store.py
import uuid
from datetime import datetime
from sqlalchemy.orm import Session
from ..database.models import Memory

class MemoryStore:
    """Simple memory store backed by SQLAlchemy.

    Usage:
        store = MemoryStore(db_session)
        store.put("default", "preferences", "learning_style", {"type": "visual"})
        memories = store.search("default", "preferences")
    """

    def __init__(self, db: Session):
        self.db = db

    def put(self, user_id: str, namespace: str, key: str, value: dict) -> Memory:
        """Store or update a memory."""
        existing = (
            self.db.query(Memory)
            .filter_by(user_id=user_id, namespace=namespace, key=key)
            .first()
        )

        if existing:
            existing.value = value
            existing.updated_at = datetime.utcnow()
            self.db.commit()
            return existing
        else:
            memory = Memory(
                id=str(uuid.uuid4()),
                user_id=user_id,
                namespace=namespace,
                key=key,
                value=value,
            )
            self.db.add(memory)
            self.db.commit()
            return memory

    def get(self, user_id: str, namespace: str, key: str) -> dict | None:
        """Get a specific memory by key."""
        memory = (
            self.db.query(Memory)
            .filter_by(user_id=user_id, namespace=namespace, key=key)
            .first()
        )
        return memory.value if memory else None

    def search(self, user_id: str, namespace: str | None = None, limit: int = 10) -> list[dict]:
        """Search memories in a namespace, ordered by recency."""
        query = self.db.query(Memory).filter_by(user_id=user_id)

        if namespace:
            query = query.filter_by(namespace=namespace)

        query = query.order_by(Memory.updated_at.desc()).limit(limit)

        return [
            {
                "namespace": m.namespace,
                "key": m.key,
                "value": m.value,
                "updated_at": m.updated_at.isoformat() if m.updated_at else None,
            }
            for m in query.all()
        ]
```

The namespace conventions organize memories into categories:

- preferences: Learning style, favorite topics, preferred explanation depth
- struggles: Topics the user finds difficult
- goals: What the user is studying for (certification, job, etc.)
- sessions: Summaries of past study sessions

When the Tutor Agent learns something important about a user, it stores that information. On subsequent sessions—whether running locally or in production—it retrieves relevant context to personalize the experience.

## Content-Addressed Flashcard Caching

Here's where we get serious performance wins. Instead of regenerating flashcards every time someone studies a topic, we cache generated cards by content hash. Same topic plus same source material equals same hash, leading to instant retrieval instead of expensive LLM generation.

```python
def get_cached_flashcards(
    topic: str,
    source_context: str,
    db: Session
) -> list[Flashcard] | None:
    """Check cache for existing flashcards."""
    content_hash = compute_content_hash(topic, source_context)
    return db.query(Flashcard).filter_by(content_hash=content_hash).all()

def generate_and_cache_flashcards(
    topic: str,
    source_context: str,
    db: Session
) -> list[Flashcard]:
    """Generate new flashcards and store in cache."""
    # Check cache first
    cached = get_cached_flashcards(topic, source_context, db)
    if cached:
        logger.info(f"Cache hit for topic '{topic}' - returning {len(cached)} cards")
        return cached

    # Cache miss - generate with LLM
    logger.info(f"Cache miss for topic '{topic}' - generating new cards")
    cards_data = card_generator.generate(topic, source_context)

    # Store in cache for future requests
    content_hash = compute_content_hash(topic, source_context)
    flashcards = []
    for card in cards_data:
        flashcard = Flashcard(
            id=str(uuid.uuid4()),
            content_hash=content_hash,
            topic=topic,
            question=card["question"],
            answer=card["answer"],
            source_context=source_context[:500]  # Store summary
        )
        db.add(flashcard)
        flashcards.append(flashcard)

    db.commit()
    return flashcards
```

The first user to study a topic pays the generation cost. Every subsequent user gets instant results. For a curriculum with well-defined topics, cache hit rates can exceed 80%, a massive savings in both cost and latency.

## The Scheduler Agent and SM-2 Algorithm

Our new Scheduler Agent implements spaced repetition using the SM-2 algorithm. The idea is simple: cards you know well get reviewed less frequently, while cards you struggle with come back sooner. This optimizes learning by focusing practice where it's needed most.

The SM-2 algorithm tracks two key values for each card: an ease factor (how easy this card is for the user) and an interval (days until next review). After each review, these get updated based on how well the user answered:

1. If the user struggled (quality 0-2), reset the interval to 1 day
2. If the user got it right (quality 3-5), extend the interval based on ease factor
3. Adjust the ease factor based on performance: easier cards get higher factors, harder cards get lower

```python
def calculate_next_review(
    current_interval: int,
    ease_factor: float,
    quality: int  # 0-5 rating from user
) -> tuple[int, float, datetime]:
    """Calculate next review using SM-2 algorithm."""
    if quality < 3:
        # Failed - reset to beginning
        new_interval = 1
        new_ease = max(1.3, ease_factor - 0.2)
    else:
        # Passed - extend interval
        if current_interval == 1:
            new_interval = 6
        else:
            new_interval = int(current_interval * ease_factor)

        # Adjust ease factor based on quality
        new_ease = ease_factor + (0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02))
        new_ease = max(1.3, new_ease)  # Never go below 1.3

    next_review = datetime.utcnow() + timedelta(days=new_interval)
    return new_interval, new_ease, next_review
```

The Scheduler Agent uses this algorithm to determine which cards are due for review and in what order. It prioritizes cards based on their due date and how overdue they are.

```python
SCHEDULER_PROMPT = """You are StudyBuddy's Scheduler, responsible for
optimizing the user's learning through spaced repetition.

You track which cards need review and when. When asked for cards to study:
1. Find cards that are due (next_review <= now)
2. Prioritize overdue cards
3. Mix in cards from topics the user is currently focusing on

After each review, record the user's performance and calculate
the next review date using the SM-2 algorithm."""

scheduler_agent = create_agent(
    model="gpt-4o-mini",
    tools=[get_due_cards, record_review, get_study_stats],
    system_prompt=SCHEDULER_PROMPT
)
```

## Tracking Learning History

With persistent storage, we can track complete learning history across sessions. What concepts has this user studied? How did they perform? Which topics need more attention? This data powers both the spaced repetition system and personalized learning recommendations.

```python
def get_user_stats(user_id: str, db: Session) -> dict:
    """Get comprehensive learning statistics for a user."""
    reviews = db.query(CardReview).filter_by(user_id=user_id).all()

    # Cards by topic
    topic_stats = defaultdict(lambda: {"total": 0, "correct": 0})
    for review in reviews:
        card = db.query(Flashcard).get(review.flashcard_id)
        topic_stats[card.topic]["total"] += 1
        if review.quality >= 3:
            topic_stats[card.topic]["correct"] += 1

    # Struggle areas (topics with < 60% accuracy)
    struggle_topics = [
        topic for topic, stats in topic_stats.items()
        if stats["total"] >= 3 and stats["correct"] / stats["total"] < 0.6
    ]

    # Cards due for review
    now = datetime.utcnow()
    due_count = db.query(CardReview).filter(
        CardReview.user_id == user_id,
        CardReview.next_review <= now
    ).count()

    return {
        "total_reviews": len(reviews),
        "topics_studied": list(topic_stats.keys()),
        "struggle_areas": struggle_topics,
        "cards_due": due_count,
        "topic_performance": dict(topic_stats)
    }
```

## Memory Connecting Both Modes

Here's something powerful: memory connects tutoring and practice modes. When you're getting tutoring about embeddings and struggle with the concept, that gets stored. Later, when reviewing flashcards about embeddings, the system knows you need extra attention on this topic.

This bidirectional connection works through shared memory namespaces. Both the Tutor Agent and Scheduler Agent can read and write to the user's learning profile. Struggle areas identified during tutoring inform flashcard prioritization. Performance during practice informs tutoring recommendations.

```python
# After tutoring session identifies a struggle area
memory_store.put(
    namespace=("user", user_id, "struggles"),
    key="embeddings_difficulty",
    value={
        "topic": "embeddings",
        "noted_at": datetime.utcnow().isoformat(),
        "context": "User needed multiple explanations of cosine similarity"
    }
)

# Scheduler uses this when prioritizing cards
struggles = memory_store.search(
    namespace=("user", user_id, "struggles"),
    query=current_topic
)
if struggles:
    # Prioritize cards from struggle areas
    boost_priority(struggles)
```

## Testing Persistence

With PostgreSQL running locally, testing persistence is straightforward. Start the server, have a tutoring session, generate some flashcards, review a few cards. Then stop the server completely and restart it. Everything should still be there.

Before testing, make sure your local PostgreSQL is running:

```bash
# Check PostgreSQL service status
brew services list | grep postgresql

# If not running, start it
brew services start postgresql@16
```

Here's your testing checklist:

1. Check that memories survive restart: the memories table should retain data across server restarts.
2. Verify flashcard cache hits: studying the same topic should return instant results
3. Confirm learning history: stats endpoint should show previous review performance
4. Test spaced repetition: cards reviewed yesterday shouldn't be due again yet

You can inspect your local database directly using psql or a GUI tool like pgAdmin, DBeaver, or TablePlus:

```bash
# Connect and explore
psql studybuddy

# List tables
\dt

# Check flashcard cache
SELECT topic, COUNT(*) FROM flashcards GROUP BY topic;
```

For production testing on Vercel, the process is identical: deploy, test the same checklist, verify data persists. Vercel's dashboard includes a Data Browser for exploring your production tables, or you can connect with any PostgreSQL client using the credentials from your project settings.

If you need to reset your local database and start fresh:

```bash
# Drop and recreate the database
dropdb studybuddy
createdb studybuddy
```

This clean-slate approach takes seconds. When your local data gets messy from testing, just drop and recreate.

## A Note on the UI

The code examples above focus on the core multi-agent architecture. The actual StudyBuddy v6 implementation extends this with a flashcard-first user interface, where students see flashcards as their primary study mode, with chat available as a secondary feature for deeper exploration.

This required a few additions to the API layer:

- /api/chapters endpoint that parses a topic-list.md file for chapter/topic navigation.
- /api/flashcard endpoint that generates scoped flashcards using the Card Generator and Quality Checker agents.
- card_context parameter in the chat endpoint so the tutor knows what flashcard you're studying.

The chat feature focuses on tutoring—answering questions and explaining concepts—while flashcard generation is handled separately by the dedicated /api/flashcard endpoint. This separation keeps chat responses clean and focused.

The v6 frontend uses a three-button review system (No / Took a sec / Yes) instead of the two-button system in v5 (Got It / Study More). This maps to spaced repetition quality ratings, though the current implementation uses these buttons primarily to control card flow: "No" fetches another card on the same topic, while "Took a sec" or "Yes" moves to a new random topic.

## Running locally

StudyBuddy v6 runs with two terminals—one for the backend, one for the frontend.

**Terminal 1 - Backend:**
```bash
cd v6-agent-memory
uv run uvicorn api.index:app --reload --port 8000
```

**Terminal 2 - Frontend:**
```bash
cd v6-agent-memory/frontend
npm run dev
```

Visit http://localhost:3000. The Next.js dev server proxies `/api/*` requests to the FastAPI backend on port 8000 (configured in `next.config.ts`).

## Deploying to Vercel

To deploy v6 to Vercel, you need to update two files in the repo root.

First, update vercel.json to point to the v6-agent-memory directory:

```json
{
    "version": 2,
    "builds": [
        {
            "src": "v6-agent-memory/api/index.py",
            "use": "@vercel/python"
        },
        {
            "src": "v6-agent-memory/frontend/package.json",
            "use": "@vercel/next"
        }
    ],
    "routes": [
        { "src": "/api/(.*)", "dest": "/v6-agent-memory/api/index.py" },
        { "src": "/(.*)", "dest": "/v6-agent-memory/frontend/$1" }
    ],
    "git": {
        "deploymentEnabled": {
            "main": true
        }
    },
    "ignoreCommand": "bash scripts/should-deploy.sh"
}
```

Second, update scripts/should-deploy.sh to trigger on v6 changes:

```bash
#!/bin/bash
# Deploy if changes are in v6-agent-memory, vercel config, or scripts
# Exit 1 (true) = proceed with build, Exit 0 (false) = skip build
git diff HEAD^ HEAD --name-only | grep -qE "^v6-agent-memory/|^vercel\.json|^scripts/" && exit 1 || exit 0
```

If you haven't done so already, set your OpenAI API key environment variable in Vercel:

```
vercel env add OPENAI_API_KEY
```

Deploy:

```
vercel --prod
```

Or simply push to GitHub. Vercel will automatically deploy when changes are detected in the v6-agent-memory directory.

## What's Next

StudyBuddy v6 is a major upgrade. We've added persistence that survives restarts, intelligent caching that cuts costs, and spaced repetition that optimizes learning. The system now genuinely learns about users and adapts to their needs.

In Chapter 7, we'll go even deeper with Deep Agents that plan over long time horizons, spawn subagents for complex tasks, and manage context across extended operations. The memory foundation we've built here will be essential for agents that need to maintain state across days or weeks of work.

Let's keep building.
