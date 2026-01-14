"""LangSmith dataset management for evaluation.

This module handles creating and managing evaluation datasets in LangSmith,
including tutoring, flashcard, and retrieval quality datasets.
"""

import json
import re
import logging
from datetime import datetime

from langsmith import Client
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
import pandas as pd

logger = logging.getLogger(__name__)


def get_langsmith_client() -> Client | None:
    """Get or create LangSmith client.

    Returns:
        Configured LangSmith Client instance, or None if unavailable
    """
    try:
        client = Client()
        # Verify connection with a simple operation
        list(client.list_datasets(limit=1))
        return client
    except Exception as e:
        logger.warning(f"LangSmith unavailable: {e}. Evaluation will run locally only.")
        return None


def extract_topic(question: str) -> str:
    """Extract topic from question for categorization.

    Uses keyword matching to identify the main topic of a question.

    Args:
        question: The question text

    Returns:
        Extracted topic name or "general"
    """
    topic_keywords = [
        "RAG",
        "retrieval",
        "embeddings",
        "agents",
        "memory",
        "LLM",
        "prompt",
        "chunking",
        "vector",
        "semantic",
        "evaluation",
        "fine-tuning",
        "context",
        "generation",
    ]

    question_lower = question.lower()
    for keyword in topic_keywords:
        if keyword.lower() in question_lower:
            return keyword

    return "general"


def extract_key_concepts(answer: str) -> list[str]:
    """Extract key concepts from reference answer.

    Identifies important terms and phrases from the answer text.

    Args:
        answer: The reference answer text

    Returns:
        List of key concept strings (max 5)
    """
    # Extract capitalized multi-word phrases (likely proper nouns/terms)
    concepts = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", answer)

    # Also extract technical terms in all-caps
    concepts.extend(re.findall(r"\b[A-Z]{2,}\b", answer))

    # Deduplicate and limit
    unique_concepts = list(set(concepts))
    return unique_concepts[:5]


def create_tutoring_evaluation_dataset(
    testset_df: pd.DataFrame,
    dataset_name: str,
) -> dict | None:
    """Create LangSmith dataset for tutoring evaluation.

    Args:
        testset_df: DataFrame from testset generation with question/ground_truth
        dataset_name: Name for the dataset in LangSmith

    Returns:
        Created LangSmith Dataset info dict, or None if creation failed
    """
    client = get_langsmith_client()
    if not client:
        logger.error("Cannot create dataset: LangSmith unavailable")
        return None

    try:
        # Try to create dataset, handle conflict by adding timestamp
        try:
            dataset = client.create_dataset(
                dataset_name=dataset_name,
                description="Synthetic test cases for tutoring quality evaluation",
            )
        except Exception as e:
            if "409" in str(e) or "already exists" in str(e).lower():
                # Dataset exists, add timestamp to make unique
                timestamp = datetime.now().strftime("%H%M%S")
                dataset_name = f"{dataset_name}-{timestamp}"
                logger.info(f"Dataset name conflict, using: {dataset_name}")
                dataset = client.create_dataset(
                    dataset_name=dataset_name,
                    description="Synthetic test cases for tutoring quality evaluation",
                )
            else:
                raise

        examples_created = 0
        for _, row in testset_df.iterrows():
            question = row.get("question", "")
            ground_truth = row.get("ground_truth", "")

            if not question or not ground_truth:
                continue

            client.create_example(
                inputs={
                    "question": question,
                    "topic": extract_topic(question),
                    "difficulty": row.get("evolution_type", "simple"),
                },
                outputs={
                    "reference_answer": ground_truth,
                    "key_concepts": extract_key_concepts(ground_truth),
                },
                dataset_id=dataset.id,
            )
            examples_created += 1

        logger.info(f"Created dataset '{dataset_name}' with {examples_created} examples")
        return {
            "id": dataset.id,
            "name": dataset.name,
            "examples_count": examples_created,
        }

    except Exception as e:
        logger.error(f"Failed to create tutoring dataset: {e}")
        return None


def create_flashcard_evaluation_dataset(
    topics: list[str],
    dataset_name: str = "flashcard-quality-v1",
) -> dict | None:
    """Create dataset for flashcard quality evaluation.

    Generates examples of good and bad flashcards for each topic.

    Args:
        topics: List of topics to generate examples for
        dataset_name: Name for the dataset

    Returns:
        Created LangSmith Dataset info dict, or None if creation failed
    """
    client = get_langsmith_client()
    if not client:
        logger.error("Cannot create dataset: LangSmith unavailable")
        return None

    llm = ChatOpenAI(model="gpt-4o")

    try:
        dataset = client.create_dataset(
            dataset_name=dataset_name,
            description="Test cases for flashcard clarity and effectiveness",
        )

        examples_created = 0
        for topic in topics:
            prompt = f"""For the topic "{topic}", generate examples for flashcard evaluation:

1. A GOOD flashcard (clear, focused, testable)
2. A BAD flashcard - too vague
3. A BAD flashcard - too complex (multiple concepts)
4. A BAD flashcard - unclear wording

For each, provide:
- question: The flashcard question
- answer: The expected answer
- quality: "good" or "bad"
- issue: If bad, what's wrong with it (empty string if good)

Format as JSON array."""

            response = llm.invoke(prompt)

            try:
                examples = json.loads(response.content)
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse flashcard examples for topic: {topic}")
                continue

            for ex in examples:
                client.create_example(
                    inputs={
                        "topic": topic,
                        "flashcard_question": ex.get("question", ""),
                        "flashcard_answer": ex.get("answer", ""),
                    },
                    outputs={
                        "expected_quality": ex.get("quality", "unknown"),
                        "expected_issue": ex.get("issue", "none"),
                    },
                    dataset_id=dataset.id,
                )
                examples_created += 1

        logger.info(f"Created flashcard dataset '{dataset_name}' with {examples_created} examples")
        return {
            "id": dataset.id,
            "name": dataset.name,
            "examples_count": examples_created,
        }

    except Exception as e:
        logger.error(f"Failed to create flashcard dataset: {e}")
        return None


def create_retrieval_evaluation_dataset(
    documents: list[Document],
    dataset_name: str = "retrieval-quality-v1",
) -> dict | None:
    """Create dataset for retrieval quality evaluation.

    Generates questions that should retrieve specific documents.

    Args:
        documents: List of documents to generate retrieval tests from
        dataset_name: Name for the dataset

    Returns:
        Created LangSmith Dataset info dict, or None if creation failed
    """
    client = get_langsmith_client()
    if not client:
        logger.error("Cannot create dataset: LangSmith unavailable")
        return None

    llm = ChatOpenAI(model="gpt-4o")

    try:
        dataset = client.create_dataset(
            dataset_name=dataset_name,
            description="Test cases for measuring retrieval accuracy",
        )

        examples_created = 0
        # Sample documents to avoid too many API calls
        sample_docs = documents[:50] if len(documents) > 50 else documents

        for doc in sample_docs:
            content = doc.page_content[:1500]  # Limit content length
            source = doc.metadata.get("source", "unknown")

            prompt = f"""Based on this content, generate 3 questions that should retrieve this document:

Content:
{content}

For each question:
- question: The query
- expected_chunks: Key phrases (2-4 words each) that should appear in retrieved content

Format as JSON array: [{{"question": "...", "expected_chunks": ["...", "..."]}}]"""

            response = llm.invoke(prompt)

            try:
                questions = json.loads(response.content)
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse retrieval questions for source: {source}")
                continue

            for q in questions:
                client.create_example(
                    inputs={"query": q.get("question", "")},
                    outputs={
                        "expected_chunks": q.get("expected_chunks", []),
                        "source_file": source,
                    },
                    dataset_id=dataset.id,
                )
                examples_created += 1

        logger.info(f"Created retrieval dataset '{dataset_name}' with {examples_created} examples")
        return {
            "id": dataset.id,
            "name": dataset.name,
            "examples_count": examples_created,
        }

    except Exception as e:
        logger.error(f"Failed to create retrieval dataset: {e}")
        return None
