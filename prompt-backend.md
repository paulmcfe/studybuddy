# Backend Prompt for Claude Code

Modify the StudyBuddy v4 backend (api/index.py) to support a flashcard-first learning app with these new capabilities:

## New Endpoints

### 1. GET /api/chapters
Parse `documents/topic-list.md` and return structured chapter data.

**Response format:**
```json
{
  "chapters": [
    {
      "id": 1,
      "title": "Vibe-Coding Your First LLM Application",
      "sections": [
        {
          "name": "Understanding the Tools",
          "subtopics": ["uv for Python package management", "FastAPI for backend APIs", ...]
        }
      ]
    }
  ]
}
```

**Parsing rules:**
- `# Chapter X: Title` → chapter (extract number and title)
- `## Section Name` → section within current chapter
- `- Subtopic` → subtopic within current section
- `---` → chapter boundary (ignore, use H1 to detect new chapters)
- Skip the Introduction section (before Chapter 1)

### 2. POST /api/flashcard
Generate a single flashcard scoped to selected chapters.

**Request:**
```json
{
  "chapter_id": 4,
  "scope": "single" | "cumulative",
  "current_topic": "Query Planning and Analysis"  // optional, for "Study More"
}
```

**Response:**
```json
{
  "question": "What are the three key decisions an agentic RAG system makes?",
  "answer": "1) When to retrieve, 2) What to retrieve, 3) How much to retrieve",
  "topic": "Dynamic Retrieval Strategies",
  "source": "rag" | "llm"
}
```

**Logic:**
1. Load chapter data from topic-list.md
2. Determine scope: if "cumulative", include chapters 1 through chapter_id; if "single", just chapter_id
3. If `current_topic` provided, generate another card on that topic
4. Otherwise, pick a random section/topic from the scoped chapters
5. Search RAG with queries derived from the topic name and subtopics
6. If RAG returns good content (confidence >= 0.6), generate card from it (source: "rag")
7. If RAG is insufficient, generate card from LLM knowledge (source: "llm")
8. Return the topic so frontend can request "Study More" on same topic

**Flashcard generation prompt guidance:**
- Test one concept per card
- Clear, unambiguous questions
- Concise but complete answers
- Avoid yes/no questions
- Test understanding, not just recall

### 3. POST /api/chat (Modified)
Add support for card context and chapter scoping.

**Request:**
```json
{
  "message": "What does 'how much to retrieve' mean exactly?",
  "chapter_id": 4,
  "scope": "cumulative",
  "card_context": {
    "question": "What are the three key decisions...",
    "answer": "1) When to retrieve...",
    "topic": "Dynamic Retrieval Strategies"
  }
}
```

**Behavior:**
- If `card_context` provided, include it in the prompt so agent knows what the user is asking about
- If `chapter_id` provided, scope RAG retrieval to relevant chapters
- Keep existing response format: `{ reply, confidence }`

## Implementation Notes

**Topic list parsing:** Create a helper function `parse_topic_list()` that returns the structured chapter data. Cache it after first parse (file won't change during runtime).

**Scoped retrieval:** When searching the vector store, use the topic/section names from the selected chapters to build better search queries. The section names in topic-list.md directly correspond to concepts covered in the ref-*.md files.

**Random topic selection:** For flashcard generation without current_topic, randomly select a section from the scoped chapters, then use its subtopics to guide the search queries.

**Fallback to LLM:** If similarity_search returns no documents or evaluation confidence is low, generate the flashcard from the LLM's training knowledge about AI engineering. Set source to "llm" so the frontend can indicate this.

## Keep Existing Functionality

- Keep the existing LangGraph structure for the chat flow
- Keep /api/status endpoint as-is
- Keep document indexing on startup
- Keep CORS and static file serving

## Files to Modify

- `api/index.py` - Add new endpoints and modify existing

## Files to Read

- `documents/topic-list.md` - Parse for chapter structure
- `documents/ref-*.md` - These are indexed in the vector store
