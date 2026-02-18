"""Model configuration and catalog for StudyBuddy v13.

Defines available models, their pricing, and default task assignments.
Based on patterns from chapter13.ipynb.
"""

from enum import Enum
from typing import Optional
from pydantic import BaseModel


class Provider(str, Enum):
    """Supported LLM providers."""

    OPENAI = "openai"
    TOGETHER = "together"
    OLLAMA = "ollama"


class TaskType(str, Enum):
    """Task types for model routing."""

    FLASHCARD = "flashcard_generation"
    TUTORING = "tutoring"
    CURRICULUM = "curriculum"
    EMBEDDING = "embedding"


class ModelInfo(BaseModel):
    """Information about an available model."""

    provider: Provider
    model_id: str
    display_name: str
    context_length: int
    input_cost_per_1m: float  # USD per 1M input tokens
    output_cost_per_1m: float  # USD per 1M output tokens
    best_for: list[str]
    supports_streaming: bool = True
    is_embedding: bool = False


# Model catalog with February 2026 pricing
AVAILABLE_MODELS: dict[str, ModelInfo] = {
    # OpenAI models
    "gpt-4o": ModelInfo(
        provider=Provider.OPENAI,
        model_id="gpt-4o",
        display_name="GPT-4o",
        context_length=128_000,
        input_cost_per_1m=2.50,
        output_cost_per_1m=10.00,
        best_for=["complex_reasoning", "tutoring", "analysis"],
    ),
    "gpt-4o-mini": ModelInfo(
        provider=Provider.OPENAI,
        model_id="gpt-4o-mini",
        display_name="GPT-4o Mini",
        context_length=128_000,
        input_cost_per_1m=0.15,
        output_cost_per_1m=0.60,
        best_for=["flashcards", "simple_tasks", "quick_responses"],
    ),
    "text-embedding-3-small": ModelInfo(
        provider=Provider.OPENAI,
        model_id="text-embedding-3-small",
        display_name="Embedding 3 Small",
        context_length=8_191,
        input_cost_per_1m=0.02,
        output_cost_per_1m=0.0,
        best_for=["embedding"],
        supports_streaming=False,
        is_embedding=True,
    ),
    "text-embedding-3-large": ModelInfo(
        provider=Provider.OPENAI,
        model_id="text-embedding-3-large",
        display_name="Embedding 3 Large",
        context_length=8_191,
        input_cost_per_1m=0.13,
        output_cost_per_1m=0.0,
        best_for=["embedding", "high_quality"],
        supports_streaming=False,
        is_embedding=True,
    ),
    # Together AI models (open-source hosted)
    "llama-3.3-70b": ModelInfo(
        provider=Provider.TOGETHER,
        model_id="meta-llama/Llama-3.3-70B-Instruct-Turbo",
        display_name="Llama 3.3 70B Turbo",
        context_length=128_000,
        input_cost_per_1m=0.88,
        output_cost_per_1m=0.88,
        best_for=["flashcards", "general", "cost_effective"],
    ),
    "llama-4-maverick": ModelInfo(
        provider=Provider.TOGETHER,
        model_id="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
        display_name="Llama 4 Maverick",
        context_length=128_000,
        input_cost_per_1m=0.27,
        output_cost_per_1m=0.85,
        best_for=["complex_reasoning", "tutoring", "high_quality"],
    ),
    "mixtral-8x7b": ModelInfo(
        provider=Provider.TOGETHER,
        model_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
        display_name="Mixtral 8x7B",
        context_length=32_000,
        input_cost_per_1m=0.20,
        output_cost_per_1m=0.60,
        best_for=["multilingual", "general", "coding"],
    ),
    "qwen3-235b": ModelInfo(
        provider=Provider.TOGETHER,
        model_id="Qwen/Qwen3-235B-A22B-Instruct-2507-tput",
        display_name="Qwen 3 235B",
        context_length=128_000,
        input_cost_per_1m=0.30,
        output_cost_per_1m=0.90,
        best_for=["math", "coding", "asian_languages"],
    ),
    "togethercomputer/m2-bert-80M-8k-retrieval": ModelInfo(
        provider=Provider.TOGETHER,
        model_id="togethercomputer/m2-bert-80M-8k-retrieval",
        display_name="M2-BERT Embedding",
        context_length=8_192,
        input_cost_per_1m=0.008,
        output_cost_per_1m=0.0,
        best_for=["embedding", "cost_effective"],
        supports_streaming=False,
        is_embedding=True,
    ),
    # Ollama models (local, free)
    "llama3.2": ModelInfo(
        provider=Provider.OLLAMA,
        model_id="llama3.2:3b",
        display_name="Llama 3.2 3B (Local)",
        context_length=128_000,
        input_cost_per_1m=0.0,
        output_cost_per_1m=0.0,
        best_for=["development", "privacy", "offline"],
    ),
    "nomic-embed-text": ModelInfo(
        provider=Provider.OLLAMA,
        model_id="nomic-embed-text",
        display_name="Nomic Embed (Local)",
        context_length=8_192,
        input_cost_per_1m=0.0,
        output_cost_per_1m=0.0,
        best_for=["embedding", "privacy", "offline"],
        supports_streaming=False,
        is_embedding=True,
    ),
}


# Default task -> model mappings
DEFAULT_TASK_MODELS: dict[TaskType, dict[str, Optional[str]]] = {
    TaskType.FLASHCARD: {
        "primary": "llama-3.3-70b",  # Cost-effective for flashcard generation
        "fallback": "gpt-4o-mini",
    },
    TaskType.TUTORING: {
        "primary": "gpt-4o",  # Best quality for tutoring
        "fallback": "llama-4-maverick",
    },
    TaskType.CURRICULUM: {
        "primary": "gpt-4o",  # Complex reasoning for curriculum design
        "fallback": "llama-4-maverick",
    },
    TaskType.EMBEDDING: {
        "primary": "text-embedding-3-small",
        "fallback": None,
    },
}


def get_model_info(model_key: str) -> Optional[ModelInfo]:
    """Get model info by key."""
    return AVAILABLE_MODELS.get(model_key)


def calculate_cost(
    model_key: str,
    input_tokens: int,
    output_tokens: int,
) -> float:
    """Calculate cost in USD for a model call."""
    model = AVAILABLE_MODELS.get(model_key)
    if not model:
        return 0.0

    input_cost = (input_tokens / 1_000_000) * model.input_cost_per_1m
    output_cost = (output_tokens / 1_000_000) * model.output_cost_per_1m
    return input_cost + output_cost


def get_models_for_provider(provider: Provider) -> list[ModelInfo]:
    """Get all models for a specific provider."""
    return [m for m in AVAILABLE_MODELS.values() if m.provider == provider]


def get_chat_models() -> dict[str, ModelInfo]:
    """Get all non-embedding models."""
    return {k: v for k, v in AVAILABLE_MODELS.items() if not v.is_embedding}


def get_embedding_models() -> dict[str, ModelInfo]:
    """Get all embedding models."""
    return {k: v for k, v in AVAILABLE_MODELS.items() if v.is_embedding}
