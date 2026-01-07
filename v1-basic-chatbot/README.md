# StudyBuddy v1 - Basic Tutoring Chatbot

Your first AI-powered study assistant. This is the foundation we'll build on throughout the book.

## What This Is

StudyBuddy v1 is a conversational AI tutor built with FastAPI and vanilla JavaScript. It can explain concepts, answer questions, and help you understand difficult topics. Right now it's pulling purely from GPT's knowledge - no custom data yet. That's coming in v2.

This version demonstrates:
- Clean FastAPI backend with OpenAI integration
- Simple, modern chat interface
- Markdown rendering for formatted responses
- Deployment-ready architecture
- The "build, ship, share" workflow

## Prerequisites

- Python 3.12+
- An OpenAI API key (grab one at platform.openai.com)
- Git and GitHub account
- Vercel account (free tier works great)

## Local Setup

### 1. Clone the repo

```bash
git clone https://github.com/yourusername/v1-basic-chatbot.git
cd v1-basic-chatbot
```

### 2. Set up the project

```bash
# Create .env file
echo "OPENAI_API_KEY=your-key-here" > .env

# Create virtual environment and install dependencies
uv sync
```

### 3. Run the app

```bash
uv run uvicorn api.index:app --reload --port 8000
```

You should see: `Uvicorn running on http://localhost:8000`

Visit `http://localhost:8000` in your browser.

### 4. Test it out

Ask StudyBuddy something like:
- "Explain how neural networks learn"
- "What's the difference between supervised and unsupervised learning?"
- "Help me understand gradient descent"

## Project Structure

```
v1-basic-chatbot/
├── api/
│   ├── index.py         # FastAPI app with chat endpoint
│   └── requirements.txt # Vercel dependencies
├── frontend/
│   ├── index.html       # Chat interface
│   ├── styles.css       # Styling
│   └── app.js           # Frontend logic
├── .env                 # Your API keys (never commit!)
├── .gitignore           # Keeps secrets out of git
├── pyproject.toml       # Python dependencies
└── README.md            # You are here
```

## Deploying to Vercel

The app deploys as a single project - the backend serves both the API and frontend.

```bash
# From the repo root
vercel

# Set your environment variable
vercel env add OPENAI_API_KEY
# Paste your OpenAI API key when prompted
# Choose "Production" environment

# Deploy to production
vercel --prod
```

You'll get a URL like: `https://studybuddy.vercel.app`

## Customizing Your StudyBuddy

Want to make it your own? Here's what to tweak:

### Change the personality

Edit `api/index.py`, find the system prompt:

```python
system_prompt = """You are StudyBuddy, a friendly AI tutor..."""
```

Make it more formal, more casual, more encouraging - whatever fits your vibe.

### Adjust the styling

Edit `frontend/styles.css`:
- Change colors (line 6-10 for the main color scheme)
- Adjust font sizes (line 45 for message text)
- Modify spacing and layout

### Add features

Some ideas for extending v1:
- Add a "Clear conversation" button
- Show timestamp on messages
- Add syntax highlighting for code blocks
- Implement dark mode toggle

## What's Next?

In Chapter 2, we'll add RAG (Retrieval-Augmented Generation) so StudyBuddy can answer questions based on your actual study materials. That's when things get interesting.

## Troubleshooting

**Backend won't start:**
- Check your OpenAI API key is set in `.env`
- Make sure you're in the virtual environment (`which python` should show `.venv`)
- Verify dependencies installed: `pip list | grep fastapi`

**Frontend can't reach backend:**
- Check backend is running on port 8000
- Check for CORS errors in browser console
- Verify the API_URL in `app.js` matches your backend

**Deployment issues:**
- Make sure `.env` is in `.gitignore` (never commit API keys!)
- Verify environment variables are set in Vercel dashboard
- Check logs: `vercel logs` for error details

## Cost Considerations

StudyBuddy v1 uses GPT-5-nano by default, which is cheap:
- ~$0.15 per million input tokens
- ~$0.60 per million output tokens

A typical conversation (10 exchanges) costs less than $0.01. Monitor usage at platform.openai.com/usage.

## Contributing

This is your learning project! Fork it, modify it, make it your own. If you build something cool, share it with the community. That's the move.

## License

MIT - do whatever you want with this.

## Questions?

Hit up the AI Makerspace community or open an issue. We're here to help.

---

Built as part of the AI Engineering Bootcamp. Keep building. 🚀
