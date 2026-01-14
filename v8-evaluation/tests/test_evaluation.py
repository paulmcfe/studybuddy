"""Tests for evaluation infrastructure.

These tests verify the evaluation components work correctly,
including testset generation, evaluators, and baseline evaluation.
"""

import pytest
import pandas as pd


def test_testset_generation():
    """Verify we can generate synthetic test data."""
    from api.evaluation.testset_generator import (
        load_reference_documents,
        generate_tutoring_testset,
        validate_testset,
    )

    # Load documents
    docs = load_reference_documents("documents")
    assert len(docs) > 0, "Should load documents from documents directory"

    # Generate small testset for speed
    testset = generate_tutoring_testset(docs, test_size=5)
    assert len(testset) == 5, "Should generate requested number of test cases"
    assert "question" in testset.columns, "Should have question column"
    assert "ground_truth" in testset.columns, "Should have ground_truth column"

    # Validate testset
    validated = validate_testset(testset)
    assert len(validated) <= len(testset), "Validation should filter, not add"


def test_tutoring_evaluator():
    """Verify tutoring evaluator produces valid scores."""
    from api.evaluation.evaluators import tutoring_quality_evaluator

    # Create mock objects that mimic LangSmith run/example
    class MockRun:
        outputs = {
            "response": "Machine learning is a subset of AI that enables computers to learn from data without being explicitly programmed. It works by finding patterns in training data and using those patterns to make predictions on new data."
        }

    class MockExample:
        inputs = {"question": "What is machine learning?"}
        outputs = {
            "reference_answer": "Machine learning is a field of artificial intelligence that uses algorithms to learn patterns from data and make predictions or decisions without being explicitly programmed."
        }

    result = tutoring_quality_evaluator(MockRun(), MockExample())

    assert "score" in result, "Should return score"
    assert 0 <= result["score"] <= 1, "Score should be between 0 and 1"
    assert result["key"] == "tutoring_quality", "Should have correct key"


def test_flashcard_evaluator():
    """Verify flashcard evaluator produces valid scores."""
    from api.evaluation.evaluators import flashcard_quality_evaluator

    class MockRun:
        outputs = {
            "question": "What is RAG?",
            "answer": "RAG (Retrieval-Augmented Generation) is a technique that combines information retrieval with text generation, allowing LLMs to access external knowledge when generating responses.",
        }

    class MockExample:
        inputs = {"topic": "RAG"}

    result = flashcard_quality_evaluator(MockRun(), MockExample())

    assert 0 <= result["score"] <= 1, "Score should be between 0 and 1"
    assert result["key"] == "flashcard_quality", "Should have correct key"


def test_retrieval_precision_evaluator():
    """Verify retrieval precision calculation."""
    from api.evaluation.evaluators import retrieval_precision_evaluator

    class MockRun:
        outputs = {
            "contexts": ["The ReAct pattern combines reasoning and acting in a loop..."]
        }

    class MockExample:
        outputs = {"expected_chunks": ["ReAct", "reasoning", "acting"]}

    result = retrieval_precision_evaluator(MockRun(), MockExample())

    assert result["score"] == 1.0, "All expected chunks should be found"
    assert result["key"] == "retrieval_precision", "Should have correct key"


def test_retrieval_precision_partial():
    """Verify partial retrieval precision."""
    from api.evaluation.evaluators import retrieval_precision_evaluator

    class MockRun:
        outputs = {"contexts": ["The ReAct pattern is important..."]}

    class MockExample:
        outputs = {"expected_chunks": ["ReAct", "reasoning", "acting", "loop"]}

    result = retrieval_precision_evaluator(MockRun(), MockExample())

    # Only "ReAct" should be found
    assert result["score"] == 0.25, "Should find 1 of 4 expected chunks"


def test_extract_topic():
    """Verify topic extraction from questions."""
    from api.evaluation.dataset_builder import extract_topic

    assert extract_topic("What is RAG?") == "RAG"
    assert extract_topic("How do embeddings work?") == "embeddings"
    assert extract_topic("What's the weather today?") == "general"


def test_extract_key_concepts():
    """Verify key concept extraction from answers."""
    from api.evaluation.dataset_builder import extract_key_concepts

    answer = "Machine Learning uses Neural Networks and Deep Learning to process data with Natural Language Processing."
    concepts = extract_key_concepts(answer)

    assert len(concepts) <= 5, "Should limit to 5 concepts"
    assert any("Machine" in c or "Learning" in c for c in concepts), "Should extract ML concepts"


def test_validate_testset_filters_empty():
    """Verify testset validation filters empty entries."""
    from api.evaluation.testset_generator import validate_testset

    df = pd.DataFrame(
        {
            "question": ["Valid question?", "", "Another valid question?"],
            "ground_truth": ["Valid answer.", "Empty question", ""],
        }
    )

    validated = validate_testset(df)

    # Should filter out empty question and empty answer rows
    assert len(validated) <= 1, "Should filter empty entries"


def test_validate_testset_filters_short():
    """Verify testset validation filters short entries."""
    from api.evaluation.testset_generator import validate_testset

    df = pd.DataFrame(
        {
            "question": [
                "What is machine learning and how does it work?",
                "Q?",  # Too short
            ],
            "ground_truth": [
                "Machine learning is a subset of AI that enables computers to learn from data.",
                "Answer",  # Too short
            ],
        }
    )

    validated = validate_testset(df)

    assert len(validated) == 1, "Should filter short entries"


def test_dashboard_metrics_model():
    """Verify dashboard metrics model structure."""
    from api.evaluation.dashboard import DashboardMetrics

    metrics = DashboardMetrics(
        current_scores={"tutoring_quality": 0.85},
        score_trends={"tutoring_quality": [{"version": "v1", "value": 0.85}]},
        weak_areas=[{"topic": "RAG", "average_score": 0.65}],
        recent_experiments=[{"name": "test", "date": "2024-01-01"}],
        recommendations=["Improve RAG coverage"],
    )

    assert metrics.current_scores["tutoring_quality"] == 0.85
    assert len(metrics.recommendations) == 1


def test_generate_recommendations_low_scores():
    """Verify recommendations are generated for low scores."""
    from api.evaluation.dashboard import generate_recommendations

    scores = {"tutoring_quality": 0.5, "retrieval_precision": 0.6}
    weak_areas = [{"topic": "RAG", "average_score": 0.5}]

    recommendations = generate_recommendations(scores, weak_areas)

    assert len(recommendations) > 0, "Should generate recommendations"
    assert any("tutoring" in r.lower() for r in recommendations), "Should recommend tutoring improvements"


def test_generate_recommendations_good_scores():
    """Verify positive message for good scores."""
    from api.evaluation.dashboard import generate_recommendations

    scores = {"tutoring_quality": 0.95, "retrieval_precision": 0.9}
    weak_areas = []

    recommendations = generate_recommendations(scores, weak_areas)

    assert len(recommendations) > 0, "Should have at least one recommendation"
    assert any("meeting targets" in r.lower() for r in recommendations), "Should acknowledge good performance"


# Integration tests (require LangSmith and API keys)
@pytest.mark.integration
def test_langsmith_dataset_creation():
    """Verify LangSmith dataset creation works."""
    from api.evaluation.dataset_builder import (
        get_langsmith_client,
        create_tutoring_evaluation_dataset,
    )

    client = get_langsmith_client()
    if not client:
        pytest.skip("LangSmith not available")

    # Create test dataset
    test_df = pd.DataFrame(
        {
            "question": ["Test question?"],
            "ground_truth": ["Test answer for evaluation."],
            "evolution_type": ["simple"],
        }
    )

    dataset = create_tutoring_evaluation_dataset(test_df, "test-dataset-delete-me")
    assert dataset is not None, "Should create dataset"
    assert dataset["id"] is not None, "Should have dataset ID"

    # Cleanup
    try:
        client.delete_dataset(dataset_id=dataset["id"])
    except Exception:
        pass  # Best effort cleanup


@pytest.mark.integration
def test_baseline_evaluation_runs():
    """Verify full baseline evaluation completes."""
    from api.evaluation.run_baseline import run_baseline_evaluation

    # Define mock functions
    def mock_tutor(inputs):
        return {"response": "Mock tutoring response."}

    def mock_flashcard(inputs):
        return {"question": "What is X?", "answer": "X is Y."}

    def mock_retrieval(inputs):
        return {"contexts": ["Mock context"]}

    results = run_baseline_evaluation(mock_tutor, mock_flashcard, mock_retrieval)

    assert "tutoring" in results, "Should have tutoring results"
    assert "flashcards" in results, "Should have flashcard results"
    assert "retrieval" in results, "Should have retrieval results"
