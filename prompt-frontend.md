# Frontend Prompt for Claude Code

Build the StudyBuddy v4 frontend as a single `frontend/index.html` file. This is a mobile-first flashcard learning app with chat as a secondary feature.

## Reference

See `studybuddy-wireframes.html` for the complete visual design. Match it closely.

## Tech Stack

- Single HTML file with separate CSS and JS files
- Vanilla JavaScript (no frameworks)
- Mobile-first responsive CSS
- Fetch API for backend calls

## Screens

### 1. Home Screen
- Header with app logo: "📚 StudyBuddy"
- Chapter dropdown (populated from GET /api/chapters)
- Scope toggle: two buttons, "This chapter only" and "Chapters 1-X"
- When chapter selection changes, update the cumulative label (e.g., Chapter 4 → "Chapters 1-4")
- Large "Start Studying" button
- "💬 Just want to chat?" link below

### 2. Study Mode
- Full-screen flashcard that fills available space
- Topic label in top-left corner of card (e.g., "Agentic RAG")
- Question text centered on card front
- "tap to reveal answer" hint at bottom of card front
- Tap anywhere on card to flip (with CSS flip animation)
- Answer text on card back (slightly green-tinted background)
- Two action buttons below card: "✓ Got it" (green) and "↻ Study More" (amber)
- "💬 Ask StudyBuddy" link at bottom
- Loading state: show spinner with "Generating flashcard..." while waiting for API

### 3. Chat Panel
- **Mobile:** Slides up from bottom, covers ~90% of screen
- **Tablet/Desktop:** Slides in from right as a sidebar
- Header: "💬 Ask StudyBuddy" title + "✕ Close" button
- Scrollable message area
- Messages styled differently: assistant (gray, left-aligned) vs user (green, right-aligned)
- Input area: text input + circular send button
- When opened from Study Mode, first message should be context-aware: "I see you're studying [topic]. What would you like to know?"

## Responsive Layout

```
Mobile (< 640px):
┌─────────────────┐
│     Header      │
├─────────────────┤
│                 │
│   Full-width    │
│   content       │
│                 │
└─────────────────┘

Tablet (640-1024px):
┌────────┬────────────────┐
│Sidebar │                │
│        │   Main Card    │
│        │     Area       │
│        │                │
└────────┴────────────────┘

Desktop with Chat (> 1024px):
┌────────┬────────────┬────────┐
│Sidebar │   Card     │  Chat  │
│        │   Area     │ Panel  │
└────────┴────────────┴────────┘
```

## API Integration

```javascript
// On page load: fetch chapters
const response = await fetch('/api/chapters');
const { chapters } = await response.json();
// Populate dropdown with chapters

// Start studying: get first flashcard
const response = await fetch('/api/flashcard', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
        chapter_id: selectedChapter.id,
        scope: scope  // 'single' or 'cumulative'
    })
});
const card = await response.json();
// card = { question, answer, topic, source }

// "Got it" button: get new card (different topic)
// Same as above, no current_topic

// "Study More" button: get card on same topic
const response = await fetch('/api/flashcard', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
        chapter_id: selectedChapter.id,
        scope: scope,
        current_topic: currentCard.topic  // Request same topic
    })
});

// Chat message
const response = await fetch('/api/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
        message: userMessage,
        chapter_id: selectedChapter?.id,
        scope: scope,
        card_context: currentCard ? {
            question: currentCard.question,
            answer: currentCard.answer,
            topic: currentCard.topic
        } : undefined
    })
});
const { reply } = await response.json();
```

## CSS Design Tokens

```css
:root {
    --color-primary: #4a7c59;
    --color-primary-light: #e8f5e9;
    --color-primary-dark: #2e5339;
    --color-secondary: #f0ad4e;
    --color-bg: #fafafa;
    --color-card: #ffffff;
    --color-card-answer: #f8fdf9;
    --color-border: #ddd;
    --color-text: #333;
    --color-text-muted: #666;
    --color-text-hint: #999;
    --radius-sm: 8px;
    --radius-md: 12px;
    --radius-lg: 16px;
}
```

## State Management

```javascript
const state = {
    chapters: [],
    selectedChapter: null,
    scope: 'single',
    currentCard: null,
    isFlipped: false,
    isLoading: false,
    chatOpen: false,
    chatMessages: [],
    currentScreen: 'home'  // 'home' | 'study'
};

function render() {
    // Re-render UI based on state
    // Called after any state change
}
```

## Key Interactions

**Card Flip:**
```css
.card {
    transition: transform 0.6s;
    transform-style: preserve-3d;
}
.card.flipped {
    transform: rotateY(180deg);
}
.card-front, .card-back {
    backface-visibility: hidden;
}
.card-back {
    transform: rotateY(180deg);
}
```

**Chat Panel Slide:**
```css
/* Mobile: slide up */
@media (max-width: 639px) {
    .chat-panel {
        transform: translateY(100%);
        transition: transform 0.3s ease;
    }
    .chat-panel.open {
        transform: translateY(0);
    }
}

/* Desktop: slide in from right */
@media (min-width: 1024px) {
    .chat-panel {
        transform: translateX(100%);
        transition: transform 0.3s ease;
    }
    .chat-panel.open {
        transform: translateX(0);
    }
}
```

## Error Handling

- If /api/chapters fails, show error message with retry button
- If /api/flashcard fails, show error on card with retry button
- If backend is still indexing (503 from /api/status), show "Setting up... please wait"
- Disable buttons while loading to prevent double-clicks

## Accessibility

- Use semantic HTML: `<main>`, `<nav>`, `<article>`, `<button>`
- Add `aria-label` to icon buttons
- Keyboard support: Enter to send chat, Escape to close panels
- Focus trap in chat panel when open
- Visible focus indicators

## Extra Polish

- Subtle hover effects on buttons
- Smooth transitions on all state changes
- Empty state on Home: welcoming message in main area
- Source indicator on cards: small "From study materials" or "Based on AI knowledge" text

## Deliverable

Single file: `frontend/index.html` containing all HTML, CSS, and JavaScript.

The app should:
1. Load and display chapter dropdown
2. Allow starting a study session
3. Show flashcards with flip animation
4. Support "Got it" and "Study More" actions
5. Open chat panel with card context
6. Work on mobile, tablet, and desktop
7. Handle loading and error states gracefully
