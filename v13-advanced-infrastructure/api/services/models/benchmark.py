"""Performance benchmarking utilities for model comparison.

Based on benchmark_model pattern from chapter13.ipynb.
"""

import time
import logging
from dataclasses import dataclass
from typing import Optional

from .config import Provider, AVAILABLE_MODELS
from .ollama_client import OllamaClient, OLLAMA_AVAILABLE, check_ollama_running
from .together_client import TogetherAIClient, TOGETHER_AVAILABLE, _get_api_key as _get_together_key

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResultData:
    """Result from a single benchmark run."""

    provider: str
    model: str
    latency_ms: float
    response_length: int
    tokens_per_second: float
    success: bool = True
    error: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for API response."""
        return {
            "provider": self.provider,
            "model": self.model,
            "latency_ms": self.latency_ms,
            "response_length": self.response_length,
            "tokens_per_second": self.tokens_per_second,
            "success": self.success,
            "error": self.error,
        }


async def benchmark_model(
    model_key: str,
    prompt: str,
    max_tokens: int = 200,
) -> BenchmarkResultData:
    """Benchmark a single model invocation.

    Args:
        model_key: Key from AVAILABLE_MODELS
        prompt: Test prompt to send
        max_tokens: Maximum tokens to generate

    Returns:
        BenchmarkResultData with timing and success info
    """
    model_info = AVAILABLE_MODELS.get(model_key)
    if not model_info:
        return BenchmarkResultData(
            provider="unknown",
            model=model_key,
            latency_ms=0,
            response_length=0,
            tokens_per_second=0,
            success=False,
            error=f"Unknown model: {model_key}",
        )

    provider = model_info.provider
    messages = [{"role": "user", "content": prompt}]

    start = time.perf_counter()

    try:
        if provider == Provider.OPENAI:
            from langchain_openai import ChatOpenAI
            from langchain_core.messages import HumanMessage

            llm = ChatOpenAI(model=model_info.model_id, max_tokens=max_tokens)
            result = await llm.ainvoke([HumanMessage(content=prompt)])
            response_text = result.content

        elif provider == Provider.TOGETHER:
            if not TOGETHER_AVAILABLE or not _get_together_key():
                raise RuntimeError("Together AI not configured")
            client = TogetherAIClient()
            response_text, _, _ = await client.chat(
                model=model_info.model_id,
                messages=messages,
                max_tokens=max_tokens,
            )

        elif provider == Provider.OLLAMA:
            if not OLLAMA_AVAILABLE or not check_ollama_running():
                raise RuntimeError("Ollama not running")
            client = OllamaClient(model=model_info.model_id)
            response_text, _, _ = await client.chat(
                messages=messages,
                max_tokens=max_tokens,
            )

        else:
            raise ValueError(f"Unknown provider: {provider}")

        elapsed = time.perf_counter() - start
        elapsed_ms = elapsed * 1000

        # Estimate tokens (rough: 4 chars per token)
        tokens = len(response_text) // 4
        tokens_per_second = tokens / elapsed if elapsed > 0 else 0

        return BenchmarkResultData(
            provider=provider.value,
            model=model_key,
            latency_ms=elapsed_ms,
            response_length=len(response_text),
            tokens_per_second=tokens_per_second,
            success=True,
        )

    except Exception as e:
        elapsed = time.perf_counter() - start
        logger.error(f"Benchmark failed for {model_key}: {e}")
        return BenchmarkResultData(
            provider=provider.value,
            model=model_key,
            latency_ms=elapsed * 1000,
            response_length=0,
            tokens_per_second=0,
            success=False,
            error=str(e),
        )


async def run_benchmark_suite(
    prompt: str = "Explain spaced repetition in 3 sentences.",
    max_tokens: int = 200,
    include_ollama: bool = True,
    include_together: bool = True,
) -> list[BenchmarkResultData]:
    """Run benchmarks across all available models.

    Args:
        prompt: Test prompt to use for all models
        max_tokens: Maximum tokens to generate
        include_ollama: Include Ollama models if available
        include_together: Include Together AI models if configured

    Returns:
        List of benchmark results
    """
    results = []

    # Determine which providers to test
    test_ollama = include_ollama and OLLAMA_AVAILABLE and check_ollama_running()
    test_together = include_together and TOGETHER_AVAILABLE and bool(_get_together_key())

    # Models to benchmark (excluding embeddings)
    models_to_test = []

    for key, info in AVAILABLE_MODELS.items():
        if info.is_embedding:
            continue

        if info.provider == Provider.OLLAMA and not test_ollama:
            continue
        if info.provider == Provider.TOGETHER and not test_together:
            continue

        models_to_test.append(key)

    # Run benchmarks sequentially to avoid rate limits
    for model_key in models_to_test:
        logger.info(f"Benchmarking {model_key}...")
        result = await benchmark_model(model_key, prompt, max_tokens)
        results.append(result)

    return results


def compare_costs(
    input_tokens: int = 10_000_000,
    output_tokens: int = 5_000_000,
) -> list[dict]:
    """Compare monthly costs across providers for given usage.

    Args:
        input_tokens: Estimated monthly input tokens
        output_tokens: Estimated monthly output tokens

    Returns:
        List of cost comparisons sorted by total cost
    """
    comparisons = []

    for key, info in AVAILABLE_MODELS.items():
        if info.is_embedding:
            continue

        input_cost = (input_tokens / 1_000_000) * info.input_cost_per_1m
        output_cost = (output_tokens / 1_000_000) * info.output_cost_per_1m
        total_cost = input_cost + output_cost

        comparisons.append(
            {
                "model": key,
                "display_name": info.display_name,
                "provider": info.provider.value,
                "input_cost": input_cost,
                "output_cost": output_cost,
                "total_cost": total_cost,
                "input_cost_per_1m": info.input_cost_per_1m,
                "output_cost_per_1m": info.output_cost_per_1m,
            }
        )

    # Sort by total cost
    comparisons.sort(key=lambda x: x["total_cost"])
    return comparisons
