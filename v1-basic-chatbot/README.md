# StudyBuddy v1 - Basic Tutoring Chatbot

Your first AI-powered study assistant. This is the foundation we'll build on throughout the book.

## What This Is

StudyBuddy v1 is a conversational AI tutor built with a **Next.js frontend** and **FastAPI backend**. It can explain concepts, answer questions, and help you understand difficult topics. Right now it's pulling purely from GPT's knowledge - no custom data yet. That's coming in v2.

This version demonstrates:
- Next.js App Router with React components
- FastAPI backend with OpenAI integration
- Markdown rendering for formatted responses
- Deployment-ready architecture for Vercel
- The "build, ship, share" workflow

## Prerequisites

### Accounts You'll Need

- **GitHub account:** Sign up at github.com if you don't have one. This is where your code lives and where Vercel pulls from for deployment.
- **OpenAI API account:** You need API access to OpenAI's models. Refer to Appendix A for setup details. Add some credits to your account - $50 will last you a good while for the exercises in this book.
- **Vercel account:** Sign up using your GitHub account for seamless deployment integration.

### Software You'll Need

- **Git:** Version control system for tracking code changes (see Appendix A for installation)
- **Node.js and npm:** Required for the Next.js frontend. Download from nodejs.org (LTS version recommended).
- **uv:** Modern Python package manager for the FastAPI backend
- **A code editor (Cursor recommended):** Cursor is a VS Code fork with AI capabilities built in, perfect for vibe coding. You can also use VS Code with Claude Code or GitHub Copilot.
- **Terminal access:** Terminal on Mac/Linux, Command Prompt/PowerShell/Windows Terminal on Windows

## Local Setup

### 1. Fork and Clone the Repository

First, fork the AI Engineer Challenge repository:

1. Navigate to https://github.com/AI-Maker-Space/The-AI-Engineer-Challenge
2. Click **Fork** and create your own copy
3. Rename it to match your project (e.g., "studybuddy" or "my-llm-app")

Then clone your fork:

```bash
# Using HTTPS
git clone https://github.com/USERNAME/REPO-NAME.git

# Or using SSH
git clone git@github.com:USERNAME/REPO-NAME.git

cd REPO-NAME
```

### 2. Set up the Backend

The backend uses Python with FastAPI. From the project root:

```bash
cd v1-basic-chatbot
uv sync
```

This command:
- Checks for Python 3.12 (downloads it if needed)
- Creates a virtual environment in `.venv`
- Installs all Python dependencies

### 3. Set up the Frontend

The frontend uses Next.js with React. From the `v1-basic-chatbot` directory:

```bash
cd frontend
npm install
```

### 4. Set your API key

Create a `.env` file in the `v1-basic-chatbot` directory:

```bash
echo "OPENAI_API_KEY=your-key-here" > .env
```

Or set it as an environment variable:

**Mac/Linux:**
```bash
export OPENAI_API_KEY=sk-your-actual-api-key-here
```

**Windows (PowerShell):**
```powershell
$env:OPENAI_API_KEY="sk-your-actual-api-key-here"
```

### 5. Run the app

You'll need **two terminal windows** - one for the backend, one for the frontend.

**Terminal 1 - Backend (from v1-basic-chatbot/):**
```bash
uv run uvicorn api.index:app --reload --port 8000
```

**Terminal 2 - Frontend (from v1-basic-chatbot/frontend/):**
```bash
npm run dev
```

Visit `http://localhost:3000` in your browser.

### 6. Test it out

Ask StudyBuddy something like:
- "Explain how neural networks learn"
- "What's the difference between supervised and unsupervised learning?"
- "Help me understand gradient descent"

## Project Structure

```
v1-basic-chatbot/
├── api/
│   ├── index.py              # FastAPI app with chat endpoint
│   └── requirements.txt      # Vercel Python dependencies
├── frontend/
│   ├── src/
│   │   ├── app/
│   │   │   ├── layout.tsx    # Root layout
│   │   │   ├── page.tsx      # Main chat page
│   │   │   └── globals.css   # Styles and animations
│   │   └── components/
│   │       ├── Message.tsx       # Message bubble component
│   │       ├── MessageList.tsx   # Scrollable message container
│   │       ├── MessageInput.tsx  # Input textarea + send button
│   │       └── LoadingDots.tsx   # Loading animation
│   ├── package.json          # Node.js dependencies
│   ├── next.config.ts        # Next.js configuration (API proxy)
│   └── tsconfig.json         # TypeScript configuration
├── .env                      # Your API keys (never commit!)
├── .gitignore                # Keeps secrets out of git
├── pyproject.toml            # Python dependencies for local dev
└── README.md                 # You are here
```

## Understanding the Architecture

### Backend (FastAPI)

The backend in `api/index.py`:
- Sets up a FastAPI server with CORS enabled
- Loads environment variables from `.env`
- Initializes an OpenAI client
- Exposes a POST endpoint at `/api/chat` that sends messages to OpenAI's API using gpt-4o-mini

The chat endpoint uses OpenAI's Responses API:

```python
response = client.responses.create(
    model="gpt-4o-mini",
    instructions="Your prompt goes here",
    input=user_message
)
return {"reply": response.output_text}
```

### Frontend (Next.js)

The frontend uses React with TypeScript:
- **App Router:** Modern Next.js routing in `src/app/`
- **Components:** Reusable React components in `src/components/`
- **State Management:** React hooks (useState) for chat messages
- **Markdown:** react-markdown for rendering AI responses
- **Styling:** Tailwind CSS for utility-first styling

### Local Development

During development, the Next.js dev server (port 3000) proxies `/api/*` requests to the FastAPI backend (port 8000). This is configured in `next.config.ts`.

## Deploying to Vercel

### Install the Vercel CLI

```bash
npm install -g vercel
```

### Deploy your application

```bash
vercel
```

First time setup:
1. Log in with your GitHub account
2. Create a new project when prompted
3. Accept default settings
4. Say **No** when asked to link to the original repo, then run `vercel git connect` and choose your own repo

### Set environment variables

1. Go to vercel.com and open your project
2. Click **Settings** > **Environment Variables**
3. Add `OPENAI_API_KEY` with your API key
4. Click **Save**, then **Redeploy** when prompted

### Deploy to production

```bash
vercel --prod
```

You'll get a URL like: `https://your-project.vercel.app`

## Customizing Your StudyBuddy

### Change the personality

Edit `api/index.py` and modify the `instructions` parameter in the `responses.create()` call. The default StudyBuddy prompt is:

```
You are StudyBuddy, a helpful AI tutoring assistant. Your job is to help students learn by:
- Explaining concepts clearly and at the right level for the student
- Breaking down complex ideas into simpler pieces
- Providing examples to illustrate your explanations
- Encouraging questions and curiosity
- Being patient and supportive
```

Check out Appendix B for more project ideas and prompts.

### Adjust the styling

Edit `frontend/src/app/globals.css` or modify the Tailwind classes in the React components.

### Add features

Some ideas for extending v1:
- Add a "Clear conversation" button
- Show timestamps on messages
- Add syntax highlighting for code blocks
- Implement dark mode toggle

## Vibe Coding the Frontend

This project uses "vibe coding" - describing what you want to an AI and letting it generate the code. If you're using Cursor:

1. Open Cursor's chat panel (Cmd+L or Ctrl+L)
2. Switch to Agent mode
3. Describe what you want your frontend to look like
4. Let Cursor generate the files
5. Iterate by asking for changes and fixes

The `.cursor/rules/` folder contains configuration files that guide Cursor's code generation.

## What's Next?

In Chapter 2, we'll add RAG (Retrieval-Augmented Generation) so StudyBuddy can answer questions based on your actual study materials. That's when things get interesting.

## Troubleshooting

**Backend won't start:**
- Check your OpenAI API key is set in `.env` or as an environment variable
- Make sure Python dependencies are installed: `uv sync`
- Verify you're in the v1-basic-chatbot directory

**Frontend won't start:**
- Make sure Node.js dependencies are installed: `cd frontend && npm install`
- Check that you're running `npm run dev` from the `frontend/` directory

**Frontend can't reach backend:**
- Make sure the backend is running on port 8000
- Check for CORS errors in browser console
- Verify the Next.js proxy is configured in `next.config.ts`

**Deployment issues:**
- Make sure `.env` is in `.gitignore` (never commit API keys!)
- Verify environment variables are set in Vercel dashboard
- Check logs: `vercel logs` for error details

**Testing the backend directly:**
```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello"}'
```

## Cost Considerations

StudyBuddy v1 uses gpt-4o-mini by default, which is cost-effective. A typical conversation (10 exchanges) costs less than $0.01. Monitor usage at platform.openai.com/usage.

## Sharing Your Work

You built and deployed an LLM-powered application - that's a real accomplishment! Share it:
- Post on LinkedIn (tag @AIMakerspace)
- Share the URL with friends and family
- Show off your work on social media

## Contributing

This is your learning project! Fork it, modify it, make it your own. If you build something cool, share it with the community.

## License

MIT - do whatever you want with this.

## Questions?

Hit up the AI Makerspace community or open an issue. We're here to help.

---

Built as part of the AI Engineering Bootcamp. Keep building.
