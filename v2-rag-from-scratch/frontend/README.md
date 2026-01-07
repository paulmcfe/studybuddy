### Front End

A clean, modern chat interface for StudyBuddy.

#### Features

- Markdown rendering for formatted AI responses
- Mobile-responsive design
- Keyboard shortcuts (Enter to send, Shift+Enter for new line)
- Error handling for backend connectivity issues

#### Running Locally

The frontend is served directly by the FastAPI backend. From the `v1-basic-chatbot` directory:

```bash
uv run uvicorn api.index:app --reload --port 8000
```

Then visit `http://localhost:8000` in your browser.

#### Configuration

The frontend uses a relative URL (`/api/chat`) so it works both locally and when deployed.
