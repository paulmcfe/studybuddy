# StudyBuddy v2 - RAG from Scratch

Your AI tutor now has a memory. This version adds Retrieval-Augmented Generation (RAG) so StudyBuddy can answer questions based on actual study materials.

## What This Is

StudyBuddy v2 builds on v1 by adding RAG capabilities. Instead of relying solely on the model's training data, it retrieves relevant context from study materials before answering. This version includes a complete RAG implementation built from scratch—no frameworks, just pure Python.

This version demonstrates:
- **Next.js frontend** with React components
- **FastAPI backend** with custom RAG pipeline
- Custom vector database implementation using NumPy
- Text chunking with overlap for better context
- OpenAI embeddings for semantic search
- Cosine similarity for finding relevant content
- Context-augmented prompting

## How RAG Works

1. **Chunking**: Study materials are split into overlapping chunks
2. **Embedding**: Each chunk is converted to a vector using OpenAI's embedding model
3. **Storage**: Vectors are stored in a simple in-memory database
4. **Retrieval**: When you ask a question, it's embedded and compared to stored chunks
5. **Augmentation**: The most relevant chunks are added to the prompt as context
6. **Generation**: The model answers using both its knowledge and the retrieved context

## Prerequisites

- **Node.js and npm**: Required for the Next.js frontend. Download from nodejs.org (LTS version recommended).
- **Python 3.12+**: For the FastAPI backend
- **uv**: Modern Python package manager
- An OpenAI API key (grab one at platform.openai.com)
- Git and GitHub account
- Vercel account (free tier works great)

## Local Setup

### 1. Clone the repo

```bash
git clone https://github.com/yourusername/studybuddy.git
cd studybuddy/v2-rag-from-scratch
```

### 2. Set up the Backend

```bash
# Create .env file
echo "OPENAI_API_KEY=your-key-here" > .env

# Create virtual environment and install dependencies
uv sync
```

### 3. Set up the Frontend

```bash
cd frontend
npm install
```

### 4. Run the app

You'll need **two terminal windows** - one for the backend, one for the frontend.

**Terminal 1 - Backend (from v2-rag-from-scratch/):**
```bash
uv run uvicorn api.index:app --reload --port 8000
```

**Terminal 2 - Frontend (from v2-rag-from-scratch/frontend/):**
```bash
npm run dev
```

Visit `http://localhost:3000` in your browser.

### 5. Test it out

The built-in study materials cover biology topics. Try:
- "What are the phases of mitosis?"
- "Explain the Calvin Cycle"
- "How does the water cycle work?"

Watch the footer - it shows "Indexing..." initially, then "RAG enabled" once the vector database is ready.

## Project Structure

```
v2-rag-from-scratch/
├── api/
│   ├── index.py              # FastAPI app with RAG implementation
│   └── requirements.txt      # Vercel Python dependencies
├── frontend/
│   ├── src/
│   │   ├── app/
│   │   │   ├── layout.tsx    # Root layout
│   │   │   ├── page.tsx      # Main chat page with RAG status
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
├── pyproject.toml            # Python dependencies
└── README.md                 # You are here
```

## Deploying to Vercel

The app deploys as a hybrid project - Next.js frontend + Python serverless backend. For Vercel's serverless environment, the study materials are embedded directly in the code since in-memory storage doesn't persist between requests.

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

## Customizing Your StudyBuddy

### Change the study material

Edit `api/index.py`, find the `STUDY_MATERIAL` variable:

```python
STUDY_MATERIAL = """
Your study content here...
"""
```

Replace it with your own study materials—course notes, textbook excerpts, whatever you're studying.

### Adjust RAG parameters

In `api/index.py`:
- `chunk_size`: How many words per chunk (default: 500)
- `overlap`: How many words overlap between chunks (default: 50)
- `k`: How many chunks to retrieve (default: 3)

### Change the personality

Edit the prompt in `answer_question()`:

```python
prompt = f'''You are StudyBuddy, a helpful tutoring assistant...'''
```

### Adjust the frontend styling

Edit `frontend/src/app/globals.css` or modify the Tailwind classes in the React components.

## Understanding the RAG Implementation

The `SimpleVectorDB` class is a minimal vector database:

```python
class SimpleVectorDB:
    def __init__(self):
        self.vectors = []  # Embeddings
        self.texts = []    # Original text chunks
        self.metadata = [] # Source info

    def search(self, query_vector, k=5):
        # Cosine similarity search
        ...
```

This is intentionally simple to show how RAG works under the hood. Production systems use dedicated vector databases like Qdrant, Pinecone, or Weaviate.

## What's Next?

In Chapter 3, we'll rebuild StudyBuddy as a proper agent using LangChain 1.0 and Qdrant. The agent will reason about when to search and when to answer directly, making it much smarter about tool usage.

## Troubleshooting

**Backend won't start:**
- Check your OpenAI API key is set in `.env`
- Make sure Python dependencies are installed: `uv sync`
- Verify you're in the v2-rag-from-scratch directory

**Frontend won't start:**
- Make sure Node.js dependencies are installed: `cd frontend && npm install`
- Check that you're running `npm run dev` from the `frontend/` directory

**RAG not finding relevant content:**
- Check the study material is being indexed (see RAG status in footer)
- Try adjusting chunk size for your content type
- Ensure your questions relate to the indexed material

**Frontend can't reach backend:**
- Make sure the backend is running on port 8000
- Check for CORS errors in browser console
- Verify the Next.js proxy is configured in `next.config.ts`

**Deployment issues:**
- Make sure `.env` is in `.gitignore` (never commit API keys!)
- Verify environment variables are set in Vercel dashboard
- Check logs: `vercel logs` for error details

## Cost Considerations

StudyBuddy v2 uses:
- GPT-5-nano for generation (~$0.15/M input, ~$0.60/M output tokens)
- text-embedding-3-small for embeddings (~$0.02/M tokens)

Indexing happens once per session. A typical conversation costs less than $0.01. Monitor usage at platform.openai.com/usage.

## Contributing

This is your learning project! Fork it, modify it, make it your own. If you build something cool, share it with the community.

## License

MIT - do whatever you want with this.

## Questions?

Hit up the AI Makerspace community or open an issue. We're here to help.

---

Built as part of the AI Engineering Bootcamp. Keep building.
