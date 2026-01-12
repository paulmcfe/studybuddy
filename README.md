# StudyBuddy - AI-Powered Learning Assistant

An intelligent learning system that combines conversational tutoring with spaced repetition flashcard practice. Built progressively across 14 versions in the **AI Engineering Bootcamp** book.

## What is StudyBuddy?

StudyBuddy is a complete AI learning platform that helps you master any subject through two complementary modes:

- **Tutoring Mode:** Ask questions and get clear explanations adapted to your level
- **Practice Mode:** AI-generated flashcards with spaced repetition scheduling

The system tracks your learning history, identifies weak areas, and adjusts to your preferred learning style—all powered by modern AI engineering patterns including RAG, multi-agent systems, memory, and production deployment strategies.

## Repository Structure

This repository contains all 14 versions of StudyBuddy, each corresponding to a chapter in the book:

```
studybuddy/
├── v1-basic-chatbot/
├── v2-rag-from-scratch/
├── v3-the-agentic-loop/
├── v4-agentic-rag/
├── v5-multi-agent/
├── v6-memory-enhanced/
├── v7-deep-agents/
├── v8-evaluation/
├── v9-optimized-retrieval/
├── v10-full-stack/
├── v11-external-integrations/
├── v12-production-api/
├── v13-open-source/
├── v14-production-ready/
├── shared/              # Shared utilities and data
├── scripts/             # Deployment and utility scripts
├── .gitignore
└── README.md           # This file
```

## Versions by Chapter

### Part 1: Foundations

**[v1-basic-chatbot](v1-basic-chatbot/)** - Chapter 1
- Simple tutoring chatbot deployed to Vercel
- First OpenAI API integration
- Basic prompt engineering

**[v2-rag-from-scratch](v2-rag-from-scratch/)** - Chapter 2  
- RAG pipeline built from scratch
- Document upload and indexing
- Vector similarity search
- Grounded explanations from study materials

**[v3-agentic-rag](v3-agentic-rag/)** - Chapter 3
- LangChain agent with ReAct loop
- Tool use for searching materials
- Qdrant vector database integration
- Agent decides when to retrieve vs. answer directly

**[v4-agentic-rag](v4-agentic-rag/)** - Chapter 4
- LangGraph for fine-grained control
- Query analysis and planning
- Dynamic retrieval strategies
- Reflection and confidence scoring
- LangSmith observability

### Part 2: Advanced Agents

**[v5-multi-agent](v5-multi-agent/)** - Chapter 5
- Multi-agent architecture with supervisor pattern
- Tutor Agent, Card Generator, Quality Checker, Scheduler
- **Both modes unified:** tutoring + flashcard practice
- Spaced repetition with SM-2 algorithm
- LangSmith Studio visualization

**[v6-memory-enhanced](v6-memory-enhanced/)** - Chapter 6
- LangGraph native memory with langmem
- Tracks learning history across sessions
- Remembers struggle areas vs. mastery
- Learning style preferences
- Memory connects tutoring and practice modes

**[v7-learning-planner](v7-learning-planner/)** - Chapter 7
- Deep agent capabilities
- Multi-week study schedule planning
- Prerequisite chain decomposition
- Subagent spawning for research
- Long-horizon task management

### Part 3: Evaluation & Optimization

**[v8-evaluation](v8-evaluation/)** - Chapter 8
- Synthetic test data generation with RAGAS
- Evaluation datasets for tutoring quality
- Flashcard validation
- LangSmith evaluation pipelines
- Baseline performance metrics

**[v9-optimized-retrieval](v9-optimized-retrieval/)** - Chapter 9
- Hybrid search (dense + BM25)
- Cohere reranking
- Semantic chunking
- RAG-Fusion for comprehensive coverage
- Performance benchmarking with RAGAS

### Part 4: Production Systems

**[v10-full-stack](v10-full-stack/)** - Chapter 10
- Complete web application
- Learning dashboard with progress tracking
- Flashcard review interface (swipe to answer)
- Material upload with drag-and-drop
- Study calendar and analytics
- Mobile-friendly design

**[v11-external-integrations](v11-external-integrations/)** - Chapter 11
- MCP connectors for external resources
- Notion integration for study notes
- Google Drive for documents
- GitHub for code repositories
- Calendar for session scheduling

**[v12-production-api](v12-production-api/)** - Chapter 12
- Deployed to LangSmith Deployment
- Production API endpoints
- Authentication and multi-user support
- Isolated memory per user
- Health checks and monitoring

**[v13-open-source](v13-open-source/)** - Chapter 13
- Open-source LLM support (Llama, Mixtral)
- Together AI deployment
- Open embedding models
- MCP server implementation
- Agent-to-agent communication (A2A)
- Cost optimization benchmarks

**[v14-production-ready](v14-production-ready/)** - Chapter 14
- **Final production-ready system**
- Guardrails (input validation, content filtering)
- Comprehensive caching (semantic, prompt, embeddings, responses)
- Rate limiting and throttling
- Security hardening
- Cost optimization
- Complete monitoring and alerting

## Quick Start

### Prerequisites

Before diving in, make sure you have:

- Python 3.12 or later
- Node.js 18 or later (for frontend in later versions)
- Git
- OpenAI API key
- Accounts: GitHub, Vercel (for deployment)

See **Appendix A** in the book for complete environment setup.

### Running a Specific Version

Each version is self-contained. To run version 6, for example:

```bash
# Clone the repository
git clone https://github.com/yourusername/studybuddy.git
cd studybuddy

# Navigate to the version you want
cd v6-memory-enhanced

# Follow instructions in that directory's README
cat README.md
```

Each version directory contains:
- Its own `README.md` with specific setup instructions
- Complete, working code for that chapter
- `.env.example` template for environment variables
- Deployment configuration (where applicable)

### Following Along with the Book

Reading Chapter 6? Jump straight to:
```bash
cd v6-memory-enhanced
```

Want to see how memory was added from v5 to v6? Compare:
```bash
# Look at both directories side by side
ls -la v5-multi-agent/
ls -la v6-memory-enhanced/
```

## Deployment

### Development (All Versions)
Each version can be run locally for development. See the README in each version directory.

### Production (v14 Only)
The final production-ready version (v14) is designed for deployment to:
- **LangSmith Deployment** for the agent backend
- **Vercel** for the frontend

#### Preventing Unnecessary Rebuilds

Since the entire repository is version-controlled, you'll want to prevent Vercel from rebuilding when you add earlier versions. Use this `vercel.json` configuration:

```json
{
  "git": {
    "deploymentEnabled": {
      "main": true
    }
  },
  "ignoreCommand": "bash scripts/should-deploy.sh"
}
```

And create `scripts/should-deploy.sh`:
```bash
#!/bin/bash
# Only deploy if changes are in v14-production-ready
git diff HEAD^ HEAD --name-only | grep -q "^v14-production-ready/"
```

This ensures Vercel only redeploys when v14 changes, not when you're adding earlier versions.

## Repository Organization

### Shared Resources
The `shared/` directory contains utilities and data used across multiple versions:
- Common configuration files
- Shared datasets (Vapor Labs documents, example study materials)
- Utility functions
- Reusable components

### Scripts
The `scripts/` directory contains:
- Deployment helpers (`should-deploy.sh`)
- Data preprocessing scripts
- Setup automation
- Testing utilities

### Version Independence
While there's code duplication across versions, this is intentional. Each version is:
- Complete and working on its own
- A snapshot of what StudyBuddy looks like at that chapter
- Independently deployable (though we only deploy v14 in production)
- Easy to understand without cross-referencing other versions

## Learning Path

**New to AI Engineering?** Start with v1 and work through sequentially.

**Specific interests?**
- **RAG fundamentals:** v2-v4
- **Multi-agent systems:** v5
- **Memory in agents:** v6
- **Production deployment:** v12-v14
- **Evaluation:** v8-v9

**Just want the final product?** Jump to v14 and deploy it, then work backward through the book to understand how it was built.

## Contributing

Found an issue or have an improvement? We welcome contributions!

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Make your changes
4. Test thoroughly
5. Submit a pull request

Please target your PR to the appropriate version directory.

## Project History

StudyBuddy was built as the primary teaching example for the **AI Engineering Bootcamp** book. The progression from v1 (simple chatbot) to v14 (production-ready learning platform) mirrors the learning journey from basic LLM integration to sophisticated agentic systems.

The project demonstrates:
- RAG (Retrieval Augmented Generation)
- Agentic AI systems
- Multi-agent coordination
- Memory and personalization
- Production deployment patterns
- Cost optimization
- Evaluation and testing
- Full-stack AI application development

## Need Help?

- **Book Content:** Refer to the chapter corresponding to the version you're working with
- **Code Issues:** Check the README in the specific version directory
- **Deployment Problems:** See the deployment section above
- **General Questions:** Open an issue in this repository

## License

[Your license here - probably MIT or similar for an educational project]

---

**Built with:** LangChain, LangGraph, OpenAI, Qdrant, FastAPI, and lots of learning.

**Remember:** Build, ship, share. 🚀