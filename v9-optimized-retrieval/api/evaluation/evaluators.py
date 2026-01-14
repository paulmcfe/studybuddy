"""Custom evaluators for educational content quality.

This module provides evaluators specifically designed for measuring
the quality of tutoring responses, flashcards, and retrieval results
in an educational context.
"""

import json
import logging
import re

from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)


def _extract_json(text: str) -> dict:
    """Extract JSON from LLM response, handling markdown code blocks.

    Args:
        text: Raw LLM response that may contain markdown formatting

    Returns:
        Parsed JSON dict

    Raises:
        json.JSONDecodeError: If no valid JSON found
    """
    # Try to extract JSON from markdown code block
    json_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if json_match:
        return json.loads(json_match.group(1).strip())

    # Try parsing the whole response as JSON
    return json.loads(text)


def tutoring_quality_evaluator(run, example) -> dict:
    """Evaluate tutoring response quality for learning effectiveness.

    Criteria evaluated (1-5 scale each):
    - ACCURACY: Is the information factually correct?
    - CLARITY: Is the explanation clear and easy to understand?
    - COMPLETENESS: Does it cover the key concepts needed to answer?
    - PEDAGOGY: Does it teach effectively (examples, analogies, building blocks)?
    - ENGAGEMENT: Is it engaging and encouraging for a learner?

    Args:
        run: LangSmith run with outputs["response"]
        example: LangSmith example with inputs["question"], outputs["reference_answer"]

    Returns:
        Dict with key="tutoring_quality", score=0-1, comment=justification
    """
    llm = ChatOpenAI(model="gpt-4o-mini")

    question = example.inputs.get("question", "")
    response = run.outputs.get("response", "")
    reference = example.outputs.get("reference_answer", "")

    prompt = f"""Evaluate this tutoring response for educational effectiveness.

Student Question: {question}

Tutor Response: {response}

Reference Answer: {reference}

Evaluate on these criteria (1-5 scale):

1. ACCURACY: Is the information factually correct?
2. CLARITY: Is the explanation clear and easy to understand?
3. COMPLETENESS: Does it cover the key concepts needed to answer?
4. PEDAGOGY: Does it teach effectively (examples, analogies, building blocks)?
5. ENGAGEMENT: Is it engaging and encouraging for a learner?

Provide scores and brief justification for each.
Format as JSON: {{"accuracy": N, "clarity": N, "completeness": N, "pedagogy": N, "engagement": N, "overall": N, "justification": "..."}}

The overall score should be the average of the five criteria, rounded to nearest integer."""

    try:
        result = _extract_json(llm.invoke(prompt).content)
        overall = result.get("overall", 3)

        return {
            "key": "tutoring_quality",
            "score": overall / 5.0,  # Normalize to 0-1
            "comment": result.get("justification", ""),
        }
    except (json.JSONDecodeError, Exception) as e:
        logger.warning(f"Tutoring evaluator failed: {e}")
        return {
            "key": "tutoring_quality",
            "score": 0.5,  # Default middle score on failure
            "comment": f"Evaluation failed: {e}",
        }


def flashcard_quality_evaluator(run, example) -> dict:
    """Evaluate generated flashcard quality.

    Criteria evaluated (1-5 scale each):
    - FOCUS: Does it test exactly ONE concept? (not too broad or narrow)
    - CLARITY: Is the question unambiguous?
    - TESTABILITY: Can someone clearly know if they got it right?
    - ANSWER_QUALITY: Is the answer accurate and appropriately detailed?
    - LEARNING_VALUE: Will this help someone learn the topic?

    Args:
        run: LangSmith run with outputs["question"], outputs["answer"]
        example: LangSmith example with inputs["topic"]

    Returns:
        Dict with key="flashcard_quality", score=0-1, comment=issues
    """
    llm = ChatOpenAI(model="gpt-4o-mini")

    topic = example.inputs.get("topic", "")
    generated_q = run.outputs.get("question", "")
    generated_a = run.outputs.get("answer", "")

    prompt = f"""Evaluate this flashcard for learning effectiveness.

Topic: {topic}
Question: {generated_q}
Answer: {generated_a}

Evaluate:
1. FOCUS: Does it test exactly ONE concept? (not too broad or narrow)
2. CLARITY: Is the question unambiguous?
3. TESTABILITY: Can someone clearly know if they got it right?
4. ANSWER_QUALITY: Is the answer accurate and appropriately detailed?
5. LEARNING_VALUE: Will this help someone learn the topic?

Score each 1-5, then overall 1-5.
Identify any specific issues.

Format as JSON: {{"focus": N, "clarity": N, "testability": N, "answer_quality": N, "learning_value": N, "overall": N, "issues": ["..."]}}"""

    try:
        result = _extract_json(llm.invoke(prompt).content)
        overall = result.get("overall", 3)
        issues = result.get("issues", [])

        return {
            "key": "flashcard_quality",
            "score": overall / 5.0,  # Normalize to 0-1
            "comment": "; ".join(issues) if issues else "No issues found",
        }
    except (json.JSONDecodeError, Exception) as e:
        logger.warning(f"Flashcard evaluator failed: {e}")
        return {
            "key": "flashcard_quality",
            "score": 0.5,
            "comment": f"Evaluation failed: {e}",
        }


def retrieval_precision_evaluator(run, example) -> dict:
    """Evaluate whether retrieval found the right content.

    Checks what proportion of expected key phrases appear in the
    retrieved contexts.

    Args:
        run: LangSmith run with outputs["contexts"]
        example: LangSmith example with outputs["expected_chunks"]

    Returns:
        Dict with key="retrieval_precision", score=0-1
    """
    expected_chunks = example.outputs.get("expected_chunks", [])
    retrieved = run.outputs.get("contexts", [])

    if not expected_chunks:
        return {
            "key": "retrieval_precision",
            "score": 1.0,  # No expectations = automatic pass
            "comment": "No expected chunks to verify",
        }

    # Combine all retrieved contexts into one searchable string
    if isinstance(retrieved, list):
        retrieved_text = " ".join(str(c) for c in retrieved).lower()
    else:
        retrieved_text = str(retrieved).lower()

    # Check how many expected phrases appear in retrieved content
    found = sum(1 for chunk in expected_chunks if chunk.lower() in retrieved_text)

    precision = found / len(expected_chunks)

    return {
        "key": "retrieval_precision",
        "score": precision,
        "comment": f"Found {found}/{len(expected_chunks)} expected chunks",
    }


def combined_rag_evaluator(run, example) -> list[dict]:
    """Run all evaluators and return combined results.

    Useful for comprehensive evaluation in a single pass.

    Args:
        run: LangSmith run with full RAG outputs
        example: LangSmith example with full expectations

    Returns:
        List of evaluator result dicts
    """
    results = []

    # Only run evaluators where relevant data exists
    if run.outputs.get("response"):
        results.append(tutoring_quality_evaluator(run, example))

    if run.outputs.get("question") and run.outputs.get("answer"):
        results.append(flashcard_quality_evaluator(run, example))

    if run.outputs.get("contexts"):
        results.append(retrieval_precision_evaluator(run, example))

    return results
