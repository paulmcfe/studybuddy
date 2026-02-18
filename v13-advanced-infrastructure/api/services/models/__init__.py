"""Multi-model infrastructure for StudyBuddy v13.

This module provides:
- Model configuration and catalog (config.py)
- Ollama client for local development (ollama_client.py)
- Together AI client for hosted open-source models (together_client.py)
- Fallback chain for resilient inference (fallback.py)
- Task-based model routing (router.py)
- Performance benchmarking utilities (benchmark.py)
"""

from .config import (
    Provider,
    TaskType,
    ModelInfo,
    AVAILABLE_MODELS,
    DEFAULT_TASK_MODELS,
    get_model_info,
    calculate_cost,
)
from .ollama_client import (
    OLLAMA_AVAILABLE,
    OLLAMA_RUNNING,
    OllamaClient,
    check_ollama_running,
    get_ollama_status,
)
from .together_client import (
    TOGETHER_AVAILABLE,
    TogetherAIClient,
    get_together_status,
)
from .fallback import ModelFallbackChain, FallbackStats
from .router import ModelRouter, select_model_for_task
from .benchmark import BenchmarkResultData, benchmark_model, run_benchmark_suite

__all__ = [
    # Config
    "Provider",
    "TaskType",
    "ModelInfo",
    "AVAILABLE_MODELS",
    "DEFAULT_TASK_MODELS",
    "get_model_info",
    "calculate_cost",
    # Ollama
    "OLLAMA_AVAILABLE",
    "OLLAMA_RUNNING",
    "OllamaClient",
    "check_ollama_running",
    "get_ollama_status",
    # Together
    "TOGETHER_AVAILABLE",
    "TogetherAIClient",
    "get_together_status",
    # Fallback
    "ModelFallbackChain",
    "FallbackStats",
    # Router
    "ModelRouter",
    "select_model_for_task",
    # Benchmark
    "BenchmarkResultData",
    "benchmark_model",
    "run_benchmark_suite",
]
