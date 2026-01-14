"""Retrieval comparison tool for StudyBuddy v9.

Provides utilities for comparing different retrieval strategies
and measuring improvements with RAGAS metrics.
"""

import time
from dataclasses import dataclass
from typing import Callable

from ragas import evaluate
from ragas.metrics import context_precision, context_recall
from datasets import Dataset
from langchain_core.documents import Document


@dataclass
class RetrieverConfig:
    """Configuration for a retriever to compare."""

    name: str
    retrieve_fn: Callable[[str, int], list[Document]]
    description: str


def compare_retrieval_versions(
    old_retriever_fn: Callable[[str, int], list[Document]],
    new_retriever_fn: Callable[[str, int], list[Document]],
    test_cases: list[dict],
    k: int = 5,
) -> dict:
    """
    Compare old and new retrieval systems using RAGAS metrics.

    Args:
        old_retriever_fn: Function that takes (query, k) and returns documents
        new_retriever_fn: Function that takes (query, k) and returns documents
        test_cases: List of dicts with "question" and "ground_truth" keys
        k: Number of results to retrieve

    Returns:
        Dict with old scores, new scores, and improvements
    """
    results = {"old": [], "new": []}

    for case in test_cases:
        query = case["question"]
        ground_truth = case["ground_truth"]

        # Get contexts from both retrievers
        old_docs = old_retriever_fn(query, k)
        new_docs = new_retriever_fn(query, k)

        old_contexts = [doc.page_content for doc in old_docs]
        new_contexts = [doc.page_content for doc in new_docs]

        results["old"].append(
            {
                "question": query,
                "contexts": old_contexts,
                "ground_truth": ground_truth,
            }
        )
        results["new"].append(
            {
                "question": query,
                "contexts": new_contexts,
                "ground_truth": ground_truth,
            }
        )

    # Evaluate both with RAGAS
    old_dataset = Dataset.from_dict(
        {
            "question": [r["question"] for r in results["old"]],
            "contexts": [r["contexts"] for r in results["old"]],
            "ground_truth": [r["ground_truth"] for r in results["old"]],
        }
    )

    new_dataset = Dataset.from_dict(
        {
            "question": [r["question"] for r in results["new"]],
            "contexts": [r["contexts"] for r in results["new"]],
            "ground_truth": [r["ground_truth"] for r in results["new"]],
        }
    )

    old_scores = evaluate(old_dataset, metrics=[context_precision, context_recall])
    new_scores = evaluate(new_dataset, metrics=[context_precision, context_recall])

    return {
        "old": {
            "context_precision": old_scores["context_precision"],
            "context_recall": old_scores["context_recall"],
        },
        "new": {
            "context_precision": new_scores["context_precision"],
            "context_recall": new_scores["context_recall"],
        },
        "improvement": {
            "context_precision": new_scores["context_precision"]
            - old_scores["context_precision"],
            "context_recall": new_scores["context_recall"]
            - old_scores["context_recall"],
        },
    }


def side_by_side_comparison(
    query: str,
    old_retriever_fn: Callable[[str, int], list[Document]],
    new_retriever_fn: Callable[[str, int], list[Document]],
    k: int = 5,
) -> dict:
    """
    Generate a side-by-side comparison for manual inspection.

    Useful for debugging and understanding retriever behavior.

    Args:
        query: Search query
        old_retriever_fn: Old retriever function
        new_retriever_fn: New retriever function
        k: Number of results

    Returns:
        Comparison dict with results from both retrievers
    """
    # Time both retrievers
    start = time.time()
    old_results = old_retriever_fn(query, k)
    old_latency = time.time() - start

    start = time.time()
    new_results = new_retriever_fn(query, k)
    new_latency = time.time() - start

    comparison = {
        "query": query,
        "old_retriever": {
            "latency_ms": round(old_latency * 1000, 2),
            "results": [
                {
                    "rank": i + 1,
                    "content_preview": doc.page_content[:200] + "..."
                    if len(doc.page_content) > 200
                    else doc.page_content,
                    "source": doc.metadata.get("source", "unknown"),
                    "score": doc.metadata.get("score"),
                }
                for i, doc in enumerate(old_results)
            ],
        },
        "new_retriever": {
            "latency_ms": round(new_latency * 1000, 2),
            "results": [
                {
                    "rank": i + 1,
                    "content_preview": doc.page_content[:200] + "..."
                    if len(doc.page_content) > 200
                    else doc.page_content,
                    "source": doc.metadata.get("source", "unknown"),
                    "score": doc.metadata.get("rerank_score"),
                }
                for i, doc in enumerate(new_results)
            ],
        },
        "overlap": len(
            set(hash(d.page_content) for d in old_results)
            & set(hash(d.page_content) for d in new_results)
        ),
    }

    return comparison


def generate_comparison_report(
    test_queries: list[str],
    old_retriever_fn: Callable[[str, int], list[Document]],
    new_retriever_fn: Callable[[str, int], list[Document]],
    k: int = 5,
) -> str:
    """
    Generate a markdown report comparing retrievers.

    Args:
        test_queries: List of queries to test
        old_retriever_fn: Old retriever function
        new_retriever_fn: New retriever function
        k: Number of results per query

    Returns:
        Markdown formatted report
    """
    report = "# Retrieval Comparison Report\n\n"
    total_old_latency = 0
    total_new_latency = 0

    for query in test_queries:
        comparison = side_by_side_comparison(
            query, old_retriever_fn, new_retriever_fn, k
        )

        total_old_latency += comparison["old_retriever"]["latency_ms"]
        total_new_latency += comparison["new_retriever"]["latency_ms"]

        report += f"## Query: {query}\n\n"
        report += f"**Overlap:** {comparison['overlap']}/{k} results in common\n"
        report += f"**Latency:** Old: {comparison['old_retriever']['latency_ms']}ms, "
        report += f"New: {comparison['new_retriever']['latency_ms']}ms\n\n"

        report += "### Old Retriever\n"
        for r in comparison["old_retriever"]["results"]:
            report += f"{r['rank']}. [{r['source']}] {r['content_preview']}\n"

        report += "\n### New Retriever\n"
        for r in comparison["new_retriever"]["results"]:
            score_str = f" (score: {r['score']:.3f})" if r["score"] else ""
            report += f"{r['rank']}. [{r['source']}]{score_str} {r['content_preview']}\n"

        report += "\n---\n\n"

    # Summary
    report += "## Summary\n\n"
    report += f"- Total queries: {len(test_queries)}\n"
    report += f"- Avg old latency: {total_old_latency / len(test_queries):.1f}ms\n"
    report += f"- Avg new latency: {total_new_latency / len(test_queries):.1f}ms\n"

    return report


def compare_retrievers(
    configs: list[RetrieverConfig],
    test_queries: list[dict],
    evaluate_fn: Callable | None = None,
) -> dict:
    """
    Systematically compare multiple retrieval configurations.

    Args:
        configs: List of retriever configurations to test
        test_queries: List of {query, ground_truth} dicts
        evaluate_fn: Optional custom evaluation function

    Returns:
        Comparison results with metrics for each config
    """
    results = {}

    for config in configs:
        config_results = {
            "scores": [],
            "latencies": [],
            "name": config.name,
            "description": config.description,
        }

        for test_case in test_queries:
            query = test_case["query"] if "query" in test_case else test_case["question"]
            ground_truth = test_case.get("ground_truth", "")

            # Measure retrieval
            start = time.time()
            retrieved = config.retrieve_fn(query, 5)
            latency = time.time() - start

            # Score results if evaluation function provided
            if evaluate_fn:
                score = evaluate_fn(retrieved, ground_truth, query)
                config_results["scores"].append(score)

            config_results["latencies"].append(latency)

        # Aggregate metrics
        if config_results["scores"]:
            config_results["mean_score"] = sum(config_results["scores"]) / len(
                config_results["scores"]
            )
        config_results["mean_latency"] = sum(config_results["latencies"]) / len(
            config_results["latencies"]
        )
        sorted_latencies = sorted(config_results["latencies"])
        p95_index = int(len(sorted_latencies) * 0.95)
        config_results["p95_latency"] = sorted_latencies[min(p95_index, len(sorted_latencies) - 1)]

        results[config.name] = config_results

    return results


def segment_analysis(
    results: dict,
    test_queries: list[dict],
    segment_fn: Callable[[dict], str],
) -> dict:
    """
    Analyze retriever performance across query segments.

    Args:
        results: Raw results from compare_retrievers
        test_queries: Original test queries with metadata
        segment_fn: Function that returns segment name for a query

    Returns:
        Segmented analysis results
    """
    segmented = {}

    for config_name, config_results in results.items():
        segmented[config_name] = {}

        for i, query in enumerate(test_queries):
            segment = segment_fn(query)

            if segment not in segmented[config_name]:
                segmented[config_name][segment] = {"scores": [], "latencies": []}

            if config_results.get("scores") and i < len(config_results["scores"]):
                segmented[config_name][segment]["scores"].append(
                    config_results["scores"][i]
                )
            segmented[config_name][segment]["latencies"].append(
                config_results["latencies"][i]
            )

        # Calculate segment averages
        for segment in segmented[config_name]:
            latencies = segmented[config_name][segment]["latencies"]
            segmented[config_name][segment]["mean_latency"] = sum(latencies) / len(
                latencies
            )

            scores = segmented[config_name][segment]["scores"]
            if scores:
                segmented[config_name][segment]["mean_score"] = sum(scores) / len(
                    scores
                )

    return segmented


def query_complexity_segment(query: dict) -> str:
    """Segment queries by complexity."""
    query_text = query.get("query", query.get("question", ""))
    words = query_text.split()

    if len(words) <= 5:
        return "simple"
    elif len(words) <= 15:
        return "moderate"
    else:
        return "complex"
