"""Baseline evaluation runner for StudyBuddy.

This module runs evaluation against LangSmith datasets to establish
baseline metrics for tutoring, flashcards, and retrieval quality.
"""

import logging
import uuid
from datetime import datetime

from langsmith import Client
from langsmith.evaluation import evaluate
from sqlalchemy.orm import Session

from .evaluators import (
    tutoring_quality_evaluator,
    flashcard_quality_evaluator,
    retrieval_precision_evaluator,
)

# Configure logging to output to stdout for debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def _get_langsmith_client() -> Client | None:
    """Get LangSmith client with error handling."""
    try:
        client = Client()
        list(client.list_datasets(limit=1))
        return client
    except Exception as e:
        logger.warning(f"LangSmith unavailable: {e}")
        return None


def run_tutoring_evaluation(
    tutor_func,
    dataset_name: str = "tutoring-evaluation-v1",
    experiment_prefix: str = "tutor-baseline",
) -> dict:
    """Run tutoring evaluation against LangSmith dataset.

    Args:
        tutor_func: Function that takes inputs dict and returns {"response": str}
        dataset_name: Name of the LangSmith dataset
        experiment_prefix: Prefix for the experiment name

    Returns:
        Dict with tutoring_quality score and metadata
    """
    client = _get_langsmith_client()
    if not client:
        return {"error": "LangSmith unavailable", "tutoring_quality": 0.0}

    try:
        results = evaluate(
            tutor_func,
            data=dataset_name,
            evaluators=[tutoring_quality_evaluator],
            experiment_prefix=experiment_prefix,
        )

        # Extract aggregate score from ExperimentResults
        # Each result is a dict with 'run', 'example', 'evaluation_results' keys
        # evaluation_results['results'] contains EvaluationResult objects with .key and .score
        scores = []
        result_list = list(results)  # Consume the iterator

        for r in result_list:
            eval_results = r.get("evaluation_results", {})
            results_list = eval_results.get("results", [])

            for eval_result in results_list:
                key = getattr(eval_result, "key", None)
                score = getattr(eval_result, "score", None)
                if key == "tutoring_quality" and score is not None:
                    scores.append(score)

        avg_score = sum(scores) / len(scores) if scores else 0.0
        logger.info(f"Tutoring evaluation complete: {len(scores)} samples, avg={avg_score:.3f}")

        return {
            "tutoring_quality": avg_score,
            "sample_size": len(scores),
            "experiment": experiment_prefix,
        }

    except Exception as e:
        logger.error(f"Tutoring evaluation failed: {e}")
        return {"error": str(e), "tutoring_quality": 0.0}


def run_flashcard_evaluation(
    flashcard_func,
    dataset_name: str = "flashcard-quality-v1",
    experiment_prefix: str = "flashcard-baseline",
) -> dict:
    """Run flashcard evaluation against LangSmith dataset.

    Args:
        flashcard_func: Function that takes inputs dict and returns {"question": str, "answer": str}
        dataset_name: Name of the LangSmith dataset
        experiment_prefix: Prefix for the experiment name

    Returns:
        Dict with flashcard_quality score and metadata
    """
    client = _get_langsmith_client()
    if not client:
        return {"error": "LangSmith unavailable", "flashcard_quality": 0.0}

    try:
        results = evaluate(
            flashcard_func,
            data=dataset_name,
            evaluators=[flashcard_quality_evaluator],
            experiment_prefix=experiment_prefix,
        )

        # Extract aggregate score from ExperimentResults
        scores = []
        result_list = list(results)

        for r in result_list:
            eval_results = r.get("evaluation_results", {})
            results_list = eval_results.get("results", [])

            for eval_result in results_list:
                if hasattr(eval_result, "key") and eval_result.key == "flashcard_quality":
                    if hasattr(eval_result, "score") and eval_result.score is not None:
                        scores.append(eval_result.score)

        avg_score = sum(scores) / len(scores) if scores else 0.0

        return {
            "flashcard_quality": avg_score,
            "sample_size": len(scores),
            "experiment": experiment_prefix,
        }

    except Exception as e:
        logger.error(f"Flashcard evaluation failed: {e}")
        return {"error": str(e), "flashcard_quality": 0.0}


def run_retrieval_evaluation(
    retrieval_func,
    dataset_name: str = "retrieval-quality-v1",
    experiment_prefix: str = "retrieval-baseline",
) -> dict:
    """Run retrieval evaluation against LangSmith dataset.

    Args:
        retrieval_func: Function that takes inputs dict and returns {"contexts": list[str]}
        dataset_name: Name of the LangSmith dataset
        experiment_prefix: Prefix for the experiment name

    Returns:
        Dict with retrieval_precision score and metadata
    """
    client = _get_langsmith_client()
    if not client:
        return {"error": "LangSmith unavailable", "retrieval_precision": 0.0}

    try:
        results = evaluate(
            retrieval_func,
            data=dataset_name,
            evaluators=[retrieval_precision_evaluator],
            experiment_prefix=experiment_prefix,
        )

        # Extract aggregate score from ExperimentResults
        scores = []
        result_list = list(results)

        for r in result_list:
            eval_results = r.get("evaluation_results", {})
            results_list = eval_results.get("results", [])

            for eval_result in results_list:
                if hasattr(eval_result, "key") and eval_result.key == "retrieval_precision":
                    if hasattr(eval_result, "score") and eval_result.score is not None:
                        scores.append(eval_result.score)

        avg_score = sum(scores) / len(scores) if scores else 0.0

        return {
            "retrieval_precision": avg_score,
            "sample_size": len(scores),
            "experiment": experiment_prefix,
        }

    except Exception as e:
        logger.error(f"Retrieval evaluation failed: {e}")
        return {"error": str(e), "retrieval_precision": 0.0}


def run_baseline_evaluation(
    tutor_func,
    flashcard_func,
    retrieval_func,
) -> dict:
    """Run full baseline evaluation for all components.

    Args:
        tutor_func: Tutoring system function
        flashcard_func: Flashcard generation function
        retrieval_func: Retrieval function

    Returns:
        Dict with results for tutoring, flashcards, and retrieval
    """
    timestamp = datetime.now().strftime("%Y%m%d-%H%M")

    results = {
        "tutoring": run_tutoring_evaluation(
            tutor_func,
            experiment_prefix=f"tutor-baseline-{timestamp}",
        ),
        "flashcards": run_flashcard_evaluation(
            flashcard_func,
            experiment_prefix=f"flashcard-baseline-{timestamp}",
        ),
        "retrieval": run_retrieval_evaluation(
            retrieval_func,
            experiment_prefix=f"retrieval-baseline-{timestamp}",
        ),
        "timestamp": timestamp,
    }

    return results


def save_evaluation_run(
    db: Session,
    experiment_name: str,
    dataset_type: str,
    metrics: dict,
    config: dict | None = None,
    notes: str | None = None,
):
    """Persist evaluation results to database.

    Args:
        db: Database session
        experiment_name: Name of the experiment
        dataset_type: Type of dataset (tutoring, flashcard, retrieval)
        metrics: Dict of metric scores
        config: Optional configuration used
        notes: Optional notes about the run

    Returns:
        Created EvaluationRun record
    """
    from ..database.models import EvaluationRun

    run = EvaluationRun(
        id=str(uuid.uuid4()),
        experiment_name=experiment_name,
        dataset_type=dataset_type,
        metrics=metrics,
        config=config or {},
        sample_size=metrics.get("sample_size", 0),
        notes=notes,
    )

    db.add(run)
    db.commit()
    db.refresh(run)

    return run


def print_baseline_report(results: dict) -> None:
    """Print formatted baseline report to console.

    Args:
        results: Dict with tutoring, flashcards, retrieval results
    """
    print("\n" + "=" * 50)
    print("STUDYBUDDY V8 BASELINE METRICS")
    print("=" * 50)

    print("\nTutoring Quality:")
    tutoring = results.get("tutoring", {})
    print(f"  Overall Score: {tutoring.get('tutoring_quality', 0):.3f}")
    print(f"  Sample Size: {tutoring.get('sample_size', 0)}")
    if tutoring.get("error"):
        print(f"  Error: {tutoring['error']}")

    print("\nFlashcard Quality:")
    flashcards = results.get("flashcards", {})
    print(f"  Overall Score: {flashcards.get('flashcard_quality', 0):.3f}")
    print(f"  Sample Size: {flashcards.get('sample_size', 0)}")
    if flashcards.get("error"):
        print(f"  Error: {flashcards['error']}")

    print("\nRetrieval Precision:")
    retrieval = results.get("retrieval", {})
    print(f"  Overall Score: {retrieval.get('retrieval_precision', 0):.3f}")
    print(f"  Sample Size: {retrieval.get('sample_size', 0)}")
    if retrieval.get("error"):
        print(f"  Error: {retrieval['error']}")

    print("\n" + "=" * 50)
    print(f"Evaluation completed at: {results.get('timestamp', 'unknown')}")
    print("=" * 50 + "\n")
