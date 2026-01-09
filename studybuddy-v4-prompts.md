# StudyBuddy v4 - Claude Code Implementation Prompts

## Overview

StudyBuddy v4 is a flashcard-first learning app with chat as a secondary feature. The user selects a chapter (or cumulative chapters), then studies via AI-generated flashcards. Chat is available for deeper explanations when a concept isn't clicking.

---

## Prompt 1: Backend Modifications

### Context

I'm building StudyBuddy v4, a flashcard-based learning app. The current backend (index.py) has a simple chat endpoint. I need to extend it to support:

1. Chapter/topic selection from a topic-list.md file
2. Scoped flashcard generation (cards only from selected chapters)
3. Chat with optional card context
4. "Study More" functionality (more cards on same topic)

### Requirements

**New Endpoints Needed:**

1. `GET /api/chapters` - Parse topic-list.md and return chapter list
   - Response: `{ chapters: [{ id: 1, title: "Chapter 1: Vibe-Coding Your First LLM Application", sections: [...] }, ...] }`
   - Parse H1 headings (`# Chapter X: Title`) as chapters
   - Parse H2 headings as sections within each chapter
   - Parse bullet points as subtopics within sections

2. `POST /api/flashcard` - Generate a single flashcard
   - Request: `{ chapter_id: number, scope: "single" | "cumulative", current_topic?: string }`
   - If `current_topic` is provided, generate another card on that topic ("Study More" mode)
   - If not provided, pick a topic from the scoped chapters
   - Scope the RAG retrieval to only search documents relevant to selected chapters
   - Response: `{ question: string, answer: string, topic: string, source: "rag" | "llm" }`
   - If RAG returns insufficient content, generate from LLM knowledge and set source to "llm"

3. `POST /api/chat` - Modified chat endpoint
   - Request: `{ message: string, chapter_id?: number, scope?: "single" | "cumulative", card_context?: { question: string, answer: string, topic: string } }`
   - If card_context is provided, the agent knows what card the user is asking about
   - If chapter_id is provided, scope retrieval to those chapters
   - Response: `{ reply: string, confidence: float }`

**Parsing topic-list.md:**

The file structure is:
```markdown
# Chapter 1: Title Here

## Section Name
- Subtopic 1
- Subtopic 2

## Another Section
- More subtopics

---

# Chapter 2: Next Title
...
```

Parse this into a structured format. The `---` horizontal rules separate chapters.

**Scoped Retrieval:**

When generating flashcards or answering questions, only retrieve from documents that match the selected scope. You'll need to:
- Map chapter topics to document sources
- Filter the vector search by chapter scope
- The topic-list.md sections/subtopics give hints about what content to search for

**Flashcard Generation Prompt:**

The flashcard generation should:
- Pick a topic from the scoped chapters (randomly or systematically)
- Search RAG for content on that topic
- Generate a Q&A flashcard testing understanding of a key concept
- If RAG content is thin, fall back to LLM knowledge
- Return the topic name so "Study More" can request more cards on it

**Files:**
- Backend: `api/index.py`
- Topic list: `documents/topic-list.md`
- Reference guides: `documents/ref-*.md`

### Current Backend Code

```python
[The full index.py content will be provided]
```

### Deliverables

1. Modified index.py with:
   - New `/api/chapters` endpoint
   - New `/api/flashcard` endpoint  
   - Modified `/api/chat` endpoint with card context support
   - Topic list parsing logic
   - Scoped retrieval logic
   - Flashcard generation node/logic

2. Keep the existing LangGraph structure but add new nodes/edges as needed

3. Maintain backwards compatibility - existing functionality should still work

---

## Prompt 2: Frontend Implementation

### Context

I'm building the frontend for StudyBuddy v4, a mobile-first flashcard learning app. I have wireframes (studybuddy-wireframes.html) showing the complete UI design.

### Tech Stack

- Vanilla JavaScript (no frameworks)
- Single HTML file with separate CSS and JS files
- Mobile-first responsive design
- Fetch API for backend calls

### Screens

**1. Home Screen**
- App logo/title
- Chapter dropdown (populated from /api/chapters)
- Scope toggle: "This chapter only" vs "Chapters 1-X"
- "Start Studying" button
- "Just want to chat?" link

**2. Study Mode**
- Full-screen flashcard
- Card shows topic label in corner
- Tap anywhere to flip between question and answer
- Question side: shows question + "tap to reveal" hint
- Answer side: shows answer + two action buttons
- Action buttons: "Got it" (green) and "Study More" (amber)
- "Ask StudyBuddy" link to open chat
- Loading state while generating cards

**3. Chat Panel**
- Mobile: slides up from bottom, covers most of screen
- Desktop: appears as right sidebar
- Header with title and close button
- Message history (assistant and user messages)
- Input field with send button
- Context-aware: knows current card if opened from Study Mode

### Responsive Breakpoints

- Mobile: < 640px - Single column, full-screen cards
- Tablet: 640px - 1024px - Two columns (sidebar + card)
- Desktop: > 1024px - Three columns when chat is open

### API Integration

```javascript
// Get chapters for dropdown
GET /api/chapters
Response: { chapters: [{ id, title, sections }, ...] }

// Generate flashcard
POST /api/flashcard
Body: { chapter_id, scope, current_topic? }
Response: { question, answer, topic, source }

// Chat message
POST /api/chat  
Body: { message, chapter_id?, scope?, card_context? }
Response: { reply, confidence }

// Check if backend is ready
GET /api/status
Response: { indexing_complete, documents_indexed, ... }
```

### Interaction Details

**Card Flip Animation:**
- CSS transform for flip effect
- Tap/click anywhere on card to flip
- Disable flip while loading

**Study Flow:**
1. User selects chapter + scope, clicks "Start Studying"
2. App calls /api/flashcard to get first card
3. User sees question, taps to flip, sees answer
4. "Got it" → calls /api/flashcard (new topic)
5. "Study More" → calls /api/flashcard with current_topic

**Chat Flow:**
1. User clicks "Ask StudyBuddy" from Study Mode
2. Panel slides up/in with context message
3. User types question, hits send
4. POST /api/chat with card_context included
5. Display response in chat history

**Scope Toggle Behavior:**
- When chapter changes, update the cumulative label
- If Chapter 4 selected, cumulative reads "Chapters 1-4"
- Persist selection in sessionStorage

### Design Tokens

```css
/* Colors from wireframes */
--color-primary: #4a7c59;      /* Green - primary actions */
--color-primary-light: #e8f5e9; /* Light green - selected states */
--color-primary-dark: #2e5339;  /* Dark green - text on light green */
--color-secondary: #f0ad4e;     /* Amber - "Study More" */
--color-bg: #fafafa;            /* Page background */
--color-card: #ffffff;          /* Card background */
--color-card-answer: #f8fdf9;   /* Card back (slight green tint) */
--color-border: #ddd;           /* Borders */
--color-text: #333;             /* Primary text */
--color-text-muted: #666;       /* Secondary text */
--color-text-hint: #999;        /* Hints, labels */

/* Spacing */
--radius-sm: 8px;
--radius-md: 12px;
--radius-lg: 16px;

/* Shadows */
--shadow-card: 0 2px 10px rgba(0,0,0,0.1);
```

### State Management

```javascript
const state = {
    // Configuration
    chapters: [],           // From /api/chapters
    selectedChapter: null,  // Current chapter object
    scope: 'single',        // 'single' or 'cumulative'
    
    // Study session
    currentCard: null,      // { question, answer, topic, source }
    isFlipped: false,
    isLoading: false,
    
    // Chat
    chatOpen: false,
    chatMessages: [],       // [{ role: 'user'|'assistant', content }]
    
    // UI
    currentScreen: 'home'   // 'home' | 'study' | 'chat-only'
};
```

### File Structure

```
frontend/
├── index.html      # Single file with all HTML/CSS/JS
└── (no other files needed)
```

### Accessibility

- Semantic HTML (main, nav, article, button)
- ARIA labels for interactive elements
- Focus management when panels open/close
- Keyboard support (Enter to flip card, Escape to close chat)

### Error Handling

- Show loading spinner while waiting for API
- Display friendly error if API fails
- "Retry" button on errors
- Handle case where backend is still indexing

### Deliverables

1. Complete `frontend/index.html` with:
   - All HTML structure
   - Embedded CSS (mobile-first, responsive)
   - Embedded JavaScript (state management, API calls, DOM updates)
   
2. Match the wireframe designs closely

3. Smooth animations for:
   - Card flip
   - Chat panel slide
   - Button interactions

4. Works on mobile, tablet, and desktop

---

## Additional Notes

### Topic-to-Document Mapping

The topic-list.md sections should help guide retrieval. For example, if the user is studying Chapter 4 (Agentic RAG), and the topic is "Query Planning and Analysis", the search queries should include terms like "query planning", "query decomposition", "user intent".

The ref-*.md files in the documents folder cover these topics:
- ref-rag-fundamentals.md
- ref-agentic-rag-pattern.md
- ref-embeddings.md
- ref-chunking-strategies.md
- ref-agents-and-agency.md
- ref-tool-use.md
- ref-memory-systems.md
- etc.

### Flashcard Quality

Good flashcards:
- Test one concept at a time
- Have clear, unambiguous questions
- Have concise but complete answers
- Avoid yes/no questions
- Test understanding, not just recall

Example:
```
Topic: Dynamic Retrieval Strategies
Question: What are the three key decisions an agentic RAG system makes that a basic RAG system doesn't?
Answer: 1) When to retrieve (not every query needs it), 2) What to retrieve (query reformulation), 3) How much to retrieve (adaptive k value)
```

### Source Attribution

When a flashcard is generated from RAG content, the source should be "rag". When falling back to LLM knowledge (because RAG was insufficient), the source should be "llm". The UI can optionally display this:
- RAG: "From your study materials"
- LLM: "Based on general AI knowledge"

This transparency helps users understand the system's capabilities and limitations.
