"""StudyBuddy v8 Evaluation Infrastructure.

This module provides comprehensive evaluation infrastructure for measuring
tutoring quality, flashcard effectiveness, and retrieval accuracy.

Key components:
- testset_generator: RAGAS-based synthetic test data generation
- dataset_builder: LangSmith dataset management
- evaluators: Custom evaluators for educational content
- run_baseline: Baseline evaluation runner
- dashboard: Metrics aggregation and recommendations
"""

from .testset_generator import (
    load_reference_documents,
    create_testset_generator,
    generate_tutoring_testset,
    validate_testset,
)

from .dataset_builder import (
    get_langsmith_client,
    create_tutoring_evaluation_dataset,
    create_flashcard_evaluation_dataset,
    create_retrieval_evaluation_dataset,
    extract_topic,
    extract_key_concepts,
)

from .evaluators import (
    tutoring_quality_evaluator,
    flashcard_quality_evaluator,
    retrieval_precision_evaluator,
    combined_rag_evaluator,
)

from .run_baseline import (
    run_tutoring_evaluation,
    run_flashcard_evaluation,
    run_retrieval_evaluation,
    run_baseline_evaluation,
    save_evaluation_run,
    print_baseline_report,
)

from .dashboard import (
    DashboardMetrics,
    get_experiment_scores,
    build_score_trends,
    identify_weak_areas,
    generate_recommendations,
    get_dashboard_metrics,
)

__all__ = [
    # Testset generation
    "load_reference_documents",
    "create_testset_generator",
    "generate_tutoring_testset",
    "validate_testset",
    # Dataset building
    "get_langsmith_client",
    "create_tutoring_evaluation_dataset",
    "create_flashcard_evaluation_dataset",
    "create_retrieval_evaluation_dataset",
    "extract_topic",
    "extract_key_concepts",
    # Evaluators
    "tutoring_quality_evaluator",
    "flashcard_quality_evaluator",
    "retrieval_precision_evaluator",
    "combined_rag_evaluator",
    # Baseline evaluation
    "run_tutoring_evaluation",
    "run_flashcard_evaluation",
    "run_retrieval_evaluation",
    "run_baseline_evaluation",
    "save_evaluation_run",
    "print_baseline_report",
    # Dashboard
    "DashboardMetrics",
    "get_experiment_scores",
    "build_score_trends",
    "identify_weak_areas",
    "generate_recommendations",
    "get_dashboard_metrics",
]
