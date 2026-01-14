# StudyBuddy v8 - Evaluation Infrastructure

An intelligent AI Engineering tutor with comprehensive evaluation infrastructure for measuring tutoring quality, flashcard effectiveness, and retrieval accuracy.

## What's New in v8

StudyBuddy v8 adds systematic evaluation capabilities to measure and improve the learning assistant:

- **RAGAS Integration**: Synthetic test data generation using knowledge graph approach
- **LangSmith Datasets**: Managed evaluation datasets for consistent testing
- **Custom Evaluators**: Domain-specific evaluators for educational content quality
- **Baseline Metrics**: Systematic measurement of tutoring, flashcard, and retrieval quality
- **Dashboard Metrics**: Aggregated scores, trends, and actionable recommendations

## Features

- **All v7 Features**: Deep research, planning, focus areas, markdown support
- **Synthetic Test Generation**: RAGAS-based test case creation from reference documents
- **Tutoring Quality Evaluation**: Measures accuracy, clarity, completeness, pedagogy, engagement
- **Flashcard Quality Evaluation**: Measures focus, clarity, testability, answer quality, learning value
- **Retrieval Precision Evaluation**: Measures context relevance and coverage
- **Evaluation Dashboard**: Track improvements over time

## Architecture

```
                    ┌─────────────┐
                    │  Supervisor │
                    └──────┬──────┘
                           │
         ┌─────────────────┼─────────────────┐
         │         │       │       │         │
         ▼         ▼       ▼       ▼         ▼
    ┌────────┐ ┌───────┐ ┌───────┐ ┌─────────┐
    │ Tutor  │ │ Card  │ │Quality│ │Scheduler│
    └────────┘ │  Gen  │ │Checker│ └─────────┘
               └───────┘ └───────┘
                           │
              ┌────────────┴────────────┐
              │    Evaluation Module    │
              │  ┌─────────────────────┐│
              │  │ Testset Generator   ││
              │  │ Dataset Builder     ││
              │  │ Custom Evaluators   ││
              │  │ Baseline Runner     ││
              │  │ Dashboard Metrics   ││
              │  └─────────────────────┘│
              └────────────┬────────────┘
                           │
                    ┌──────┴──────┐
                    │  LangSmith  │
                    └─────────────┘
```

## Prerequisites

- Python 3.12+
- OpenAI API key
- LangSmith API key (required for evaluation)

## Setup

### 1. Navigate to the directory

```bash
cd studybuddy/v8-evaluation
```

### 2. Create virtual environment

```bash
uv sync
```

### 3. Configure environment variables

Create or update `.env`:

```
OPENAI_API_KEY=sk-your-key-here
LANGSMITH_API_KEY=lsv2_your-key-here
LANGSMITH_PROJECT=studybuddy-v8
```

### 4. Run the server

```bash
uv run uvicorn api.index:app --reload --port 8000
```

### 5. Open the app

Visit `http://localhost:8000` in your browser.

## Evaluation API Endpoints (NEW in v8)

### Generate Synthetic Test Data

```bash
curl -X POST http://localhost:8000/api/evaluation/generate-testset \
  -H "Content-Type: application/json" \
  -d '{"test_size": 50}'
```

Creates a LangSmith dataset with synthetic questions generated from reference documents using RAGAS. This takes several minutes due to knowledge graph extraction.

### Run Evaluation

```bash
# Evaluate all components
curl -X POST http://localhost:8000/api/evaluation/run \
  -H "Content-Type: application/json" \
  -d '{"dataset_name": "tutoring-eval-20250113", "eval_type": "all"}'

# Evaluate specific component
curl -X POST http://localhost:8000/api/evaluation/run \
  -H "Content-Type: application/json" \
  -d '{"dataset_name": "tutoring-eval-20250113", "eval_type": "tutoring"}'
```

### Get Dashboard Metrics

```bash
curl http://localhost:8000/api/evaluation/dashboard
```

Returns current scores, trends, weak areas, and recommendations.

## Existing API Endpoints

All v7 endpoints remain available:

- `POST /api/chat` - Chat with the tutor
- `GET/POST /api/focus-areas` - Manage focus areas
- `POST /api/flashcard` - Generate flashcards
- `GET /api/due-cards` - Get cards due for review
- `POST /api/review` - Record flashcard review
- `GET /api/stats` - Get learning statistics

## Project Structure

```
v8-evaluation/
├── api/
│   ├── __init__.py
│   ├── index.py              # FastAPI app + evaluation endpoints
│   ├── state.py
│   ├── database/
│   │   ├── models.py         # Includes EvaluationRun model
│   │   └── connection.py
│   ├── agents/
│   │   └── ...               # Same as v7
│   ├── services/
│   │   └── ...               # Same as v7
│   └── evaluation/           # NEW: Evaluation module
│       ├── __init__.py
│       ├── testset_generator.py   # RAGAS synthetic data
│       ├── dataset_builder.py     # LangSmith datasets
│       ├── evaluators.py          # Custom evaluators
│       ├── run_baseline.py        # Evaluation runner
│       └── dashboard.py           # Metrics aggregation
├── tests/
│   └── test_evaluation.py    # Evaluation tests
├── documents/
├── frontend/
├── .env
├── pyproject.toml
└── README.md
```

## Differences from v7

| Aspect | v7 | v8 |
|--------|----|----|
| Evaluation | None | Full infrastructure |
| Test data | Manual | Synthetic (RAGAS) |
| Metrics | None | Tutoring, flashcard, retrieval |
| Dashboard | None | Scores, trends, recommendations |
| Dependencies | Base | + ragas, langsmith, pandas |

## Evaluation Workflow

1. **Generate testset**: Create synthetic test cases from your documents
2. **Run evaluation**: Execute evaluators against the test cases
3. **Review dashboard**: See scores, identify weak areas
4. **Iterate**: Make improvements, re-evaluate to measure impact

## Important Notes

- Evaluation endpoints are disabled on Vercel (too compute-heavy)
- Run evaluations locally during development
- Generate testsets once, reuse for multiple evaluation runs
- RAGAS uses gpt-4o-mini to avoid rate limits during knowledge graph extraction

## LangSmith Integration

- View evaluation results at [smith.langchain.com](https://smith.langchain.com)
- Datasets stored under your LangSmith project
- Experiments tracked with timestamps for comparison

## Cost Considerations

Evaluation adds costs beyond normal usage:
- Testset generation: Many LLM calls for knowledge graph extraction
- Evaluation runs: LLM-as-judge calls for each test case
- Recommendation: Generate testsets sparingly, reuse them

Monitor usage at platform.openai.com/usage.
