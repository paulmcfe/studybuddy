"""Evaluation dashboard metrics aggregation.

This module provides functions for aggregating evaluation metrics,
building score trends, identifying weak areas, and generating
actionable recommendations.
"""

import logging
from collections import defaultdict
from datetime import datetime

from langsmith import Client
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class DashboardMetrics(BaseModel):
    """Dashboard metrics response model."""

    current_scores: dict[str, float]
    score_trends: dict[str, list[dict]]
    weak_areas: list[dict]
    recent_experiments: list[dict]
    recommendations: list[str]


def _get_langsmith_client() -> Client | None:
    """Get LangSmith client with error handling."""
    try:
        client = Client()
        list(client.list_datasets(limit=1))
        return client
    except Exception as e:
        logger.warning(f"LangSmith unavailable: {e}")
        return None


def get_experiment_scores(project_name: str) -> dict[str, float]:
    """Get aggregate scores from a LangSmith project/experiment.

    Args:
        project_name: Name of the LangSmith project

    Returns:
        Dict mapping metric names to average scores
    """
    client = _get_langsmith_client()
    if not client:
        return {}

    try:
        runs = list(client.list_runs(project_name=project_name, limit=1000))

        if not runs:
            return {}

        # Aggregate feedback scores
        metric_scores = defaultdict(list)

        for run in runs:
            feedback = run.feedback_stats or {}
            for metric_name, stats in feedback.items():
                if isinstance(stats, dict) and "avg" in stats:
                    metric_scores[metric_name].append(stats["avg"])

        # Calculate averages
        return {
            metric: sum(scores) / len(scores)
            for metric, scores in metric_scores.items()
            if scores
        }

    except Exception as e:
        logger.error(f"Failed to get experiment scores: {e}")
        return {}


def build_score_trends(project_prefix: str, limit: int = 10) -> dict[str, list[dict]]:
    """Build score trends over time from experiments.

    Args:
        project_prefix: Prefix to filter experiments (e.g., "tutor-baseline")
        limit: Maximum number of experiments to include

    Returns:
        Dict mapping metric names to lists of {version, value, timestamp} dicts
    """
    client = _get_langsmith_client()
    if not client:
        return {}

    try:
        # List projects matching the prefix
        projects = list(client.list_projects())
        matching = [p for p in projects if p.name.startswith(project_prefix)]
        matching = sorted(matching, key=lambda p: p.created_at or datetime.min)[-limit:]

        trends = defaultdict(list)

        for project in matching:
            scores = get_experiment_scores(project.name)
            timestamp = (
                project.created_at.isoformat() if project.created_at else "unknown"
            )

            for metric, value in scores.items():
                trends[metric].append(
                    {
                        "version": project.name,
                        "value": value,
                        "timestamp": timestamp,
                    }
                )

        return dict(trends)

    except Exception as e:
        logger.error(f"Failed to build score trends: {e}")
        return {}


def identify_weak_areas(project_name: str) -> list[dict]:
    """Identify content areas with low scores.

    Groups scores by topic and identifies topics with consistently
    low performance.

    Args:
        project_name: Name of the LangSmith project to analyze

    Returns:
        List of dicts with topic, average_score, sample_count, suggestion
    """
    client = _get_langsmith_client()
    if not client:
        return []

    try:
        runs = list(client.list_runs(project_name=project_name, limit=1000))

        if not runs:
            return []

        # Group scores by topic
        topic_scores = defaultdict(list)

        for run in runs:
            topic = run.inputs.get("topic", "unknown") if run.inputs else "unknown"
            feedback = run.feedback_stats or {}

            # Get the primary quality metric
            for metric_name, stats in feedback.items():
                if isinstance(stats, dict) and "avg" in stats:
                    topic_scores[topic].append(stats["avg"])
                    break  # Use first metric found

        # Find topics with consistently low scores
        weak_areas = []
        for topic, scores in topic_scores.items():
            if not scores:
                continue

            avg = sum(scores) / len(scores)
            if avg < 0.7:  # Below threshold
                weak_areas.append(
                    {
                        "topic": topic,
                        "average_score": round(avg, 3),
                        "sample_count": len(scores),
                        "suggestion": f"Review and improve content coverage for {topic}",
                    }
                )

        return sorted(weak_areas, key=lambda x: x["average_score"])

    except Exception as e:
        logger.error(f"Failed to identify weak areas: {e}")
        return []


def generate_recommendations(
    scores: dict[str, float], weak_areas: list[dict]
) -> list[str]:
    """Generate actionable recommendations from evaluation data.

    Args:
        scores: Dict of current metric scores
        weak_areas: List of identified weak areas

    Returns:
        List of recommendation strings
    """
    recommendations = []

    # Check each metric against thresholds
    if scores.get("faithfulness", 1) < 0.85:
        recommendations.append(
            "Faithfulness is below target. Consider strengthening grounding "
            "instructions in your prompts or reducing temperature."
        )

    if scores.get("retrieval_precision", 1) < 0.75:
        recommendations.append(
            "Retrieval precision needs improvement. Consider implementing "
            "reranking or hybrid search techniques."
        )

    if scores.get("tutoring_quality", 1) < 0.8:
        recommendations.append(
            "Tutoring quality has room for improvement. Review the weak areas "
            "identified below and consider adding more examples to prompts."
        )

    if scores.get("flashcard_quality", 1) < 0.8:
        recommendations.append(
            "Flashcard quality could be improved. Focus on generating more "
            "focused, single-concept cards with clear testability."
        )

    # Add recommendations based on weak areas
    if weak_areas:
        topics = [w["topic"] for w in weak_areas[:3]]
        recommendations.append(
            f"Content quality is low for: {', '.join(topics)}. "
            "Consider adding more reference materials for these topics."
        )

    if not recommendations:
        recommendations.append(
            "All metrics are meeting targets. Consider running adversarial "
            "test cases to identify edge case failures."
        )

    return recommendations


def get_dashboard_metrics(
    tutoring_project: str = "tutor-baseline",
    flashcard_project: str = "flashcard-baseline",
    retrieval_project: str = "retrieval-baseline",
) -> DashboardMetrics:
    """Get complete dashboard metrics.

    Args:
        tutoring_project: Project prefix for tutoring experiments
        flashcard_project: Project prefix for flashcard experiments
        retrieval_project: Project prefix for retrieval experiments

    Returns:
        DashboardMetrics with all aggregated data
    """
    client = _get_langsmith_client()

    # Get current scores from most recent experiments
    current_scores = {}

    if client:
        try:
            # Find most recent projects for each type
            projects = list(client.list_projects())

            for prefix, metric_key in [
                (tutoring_project, "tutoring_quality"),
                (flashcard_project, "flashcard_quality"),
                (retrieval_project, "retrieval_precision"),
            ]:
                matching = [p for p in projects if p.name.startswith(prefix)]
                if matching:
                    latest = max(matching, key=lambda p: p.created_at or datetime.min)
                    scores = get_experiment_scores(latest.name)
                    if scores:
                        current_scores[metric_key] = scores.get(metric_key, 0)
        except Exception as e:
            logger.error(f"Failed to get current scores: {e}")

    # Build score trends
    score_trends = {}
    for prefix in [tutoring_project, flashcard_project, retrieval_project]:
        trends = build_score_trends(prefix)
        score_trends.update(trends)

    # Identify weak areas from most recent tutoring evaluation
    weak_areas = []
    if client:
        try:
            projects = list(client.list_projects())
            tutoring_projects = [p for p in projects if p.name.startswith(tutoring_project)]
            if tutoring_projects:
                latest = max(tutoring_projects, key=lambda p: p.created_at or datetime.min)
                weak_areas = identify_weak_areas(latest.name)
        except Exception as e:
            logger.error(f"Failed to identify weak areas: {e}")

    # Get recent experiments
    recent_experiments = []
    if client:
        try:
            projects = list(client.list_projects())
            # Filter to evaluation projects
            eval_projects = [
                p
                for p in projects
                if any(
                    p.name.startswith(prefix)
                    for prefix in [tutoring_project, flashcard_project, retrieval_project]
                )
            ]
            # Sort by date and take most recent
            eval_projects = sorted(
                eval_projects, key=lambda p: p.created_at or datetime.min, reverse=True
            )[:5]

            recent_experiments = [
                {
                    "name": p.name,
                    "date": p.created_at.isoformat() if p.created_at else "unknown",
                }
                for p in eval_projects
            ]
        except Exception as e:
            logger.error(f"Failed to get recent experiments: {e}")

    # Generate recommendations
    recommendations = generate_recommendations(current_scores, weak_areas)

    return DashboardMetrics(
        current_scores=current_scores,
        score_trends=score_trends,
        weak_areas=weak_areas,
        recent_experiments=recent_experiments,
        recommendations=recommendations,
    )
