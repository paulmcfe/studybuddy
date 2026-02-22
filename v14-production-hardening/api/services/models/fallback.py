"""Model fallback chain for resilient inference.

Implements graceful degradation from primary model to fallback
on failures. Based on ModelFallbackChain pattern from chapter13.ipynb.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional, Callable, Any

from langchain_openai import ChatOpenAI

from .config import Provider, AVAILABLE_MODELS, calculate_cost
from .ollama_client import OllamaClient, OLLAMA_AVAILABLE, check_ollama_running
from .together_client import TogetherAIClient, TOGETHER_AVAILABLE

logger = logging.getLogger(__name__)


@dataclass
class FallbackStats:
    """Statistics for fallback chain usage."""

    primary_calls: int = 0
    primary_successes: int = 0
    primary_failures: int = 0
    fallback_calls: int = 0
    fallback_successes: int = 0
    fallback_failures: int = 0
    total_latency_ms: float = 0.0

    @property
    def primary_success_rate(self) -> float:
        """Success rate for primary model."""
        total = self.primary_successes + self.primary_failures
        return self.primary_successes / total if total > 0 else 0.0

    @property
    def fallback_rate(self) -> float:
        """Rate at which fallback is used."""
        total = self.primary_successes + self.fallback_calls
        return self.fallback_calls / total if total > 0 else 0.0

    def to_dict(self) -> dict:
        """Convert to dictionary for API response."""
        return {
            "primary_calls": self.primary_calls,
            "primary_successes": self.primary_successes,
            "primary_failures": self.primary_failures,
            "fallback_calls": self.fallback_calls,
            "fallback_successes": self.fallback_successes,
            "fallback_failures": self.fallback_failures,
            "primary_success_rate": self.primary_success_rate,
            "fallback_rate": self.fallback_rate,
            "total_latency_ms": self.total_latency_ms,
        }


class ModelFallbackChain:
    """Implement fallback from primary model to backup on failures.

    Usage:
        chain = ModelFallbackChain(
            primary_model="llama-4-scout",
            fallback_model="gpt-4o-mini",
            on_usage=lambda **kwargs: record_usage(**kwargs),
        )
        response, model_used, was_fallback = await chain.invoke(messages)
    """

    def __init__(
        self,
        primary_model: str,
        fallback_model: Optional[str] = None,
        on_usage: Optional[Callable[..., Any]] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ):
        self.primary_model = primary_model
        self.fallback_model = fallback_model
        self.on_usage = on_usage
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.stats = FallbackStats()

        # Initialize clients based on model providers
        self._primary_client = self._create_client(primary_model)
        self._fallback_client = (
            self._create_client(fallback_model) if fallback_model else None
        )

    def _create_client(self, model_key: str) -> Any:
        """Create appropriate client for a model."""
        model_info = AVAILABLE_MODELS.get(model_key)
        if not model_info:
            raise ValueError(f"Unknown model: {model_key}")

        provider = model_info.provider

        if provider == Provider.OPENAI:
            return ChatOpenAI(
                model=model_info.model_id,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        elif provider == Provider.TOGETHER:
            if not TOGETHER_AVAILABLE:
                raise RuntimeError("Together AI SDK not available")
            return TogetherAIClient()
        elif provider == Provider.OLLAMA:
            if not OLLAMA_AVAILABLE or not check_ollama_running():
                raise RuntimeError("Ollama not available")
            return OllamaClient(model=model_info.model_id)
        else:
            raise ValueError(f"Unknown provider: {provider}")

    def _get_model_info(self, model_key: str):
        """Get model info for a model key."""
        return AVAILABLE_MODELS.get(model_key)

    async def _call_model(
        self,
        client: Any,
        model_key: str,
        messages: list[dict],
    ) -> tuple[str, int, int]:
        """Call a model and return (response, input_tokens, output_tokens)."""
        model_info = self._get_model_info(model_key)
        provider = model_info.provider

        if provider == Provider.OPENAI:
            # LangChain ChatOpenAI
            from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

            lc_messages = []
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role == "system":
                    lc_messages.append(SystemMessage(content=content))
                elif role == "assistant":
                    lc_messages.append(AIMessage(content=content))
                else:
                    lc_messages.append(HumanMessage(content=content))

            result = await client.ainvoke(lc_messages)
            response_text = result.content

            # Get token usage from response metadata if available
            usage = getattr(result, "response_metadata", {}).get("token_usage", {})
            input_tokens = usage.get("prompt_tokens", len(str(messages)) // 4)
            output_tokens = usage.get("completion_tokens", len(response_text) // 4)

            return response_text, input_tokens, output_tokens

        elif provider == Provider.TOGETHER:
            return await client.chat(
                model=model_info.model_id,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

        elif provider == Provider.OLLAMA:
            return await client.chat(
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

        raise ValueError(f"Unknown provider: {provider}")

    async def invoke(
        self,
        messages: list[dict],
    ) -> tuple[str, str, bool]:
        """Call primary model, fall back on failure.

        Returns: (response_text, model_used, was_fallback)
        """
        import time

        start_time = time.perf_counter()

        # Try primary model
        self.stats.primary_calls += 1
        try:
            response, input_tokens, output_tokens = await self._call_model(
                self._primary_client, self.primary_model, messages
            )
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            self.stats.primary_successes += 1
            self.stats.total_latency_ms += elapsed_ms

            # Record usage
            if self.on_usage:
                model_info = self._get_model_info(self.primary_model)
                cost = calculate_cost(self.primary_model, input_tokens, output_tokens)
                self.on_usage(
                    provider=model_info.provider.value,
                    model=self.primary_model,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cost_cents=cost * 100,
                    latency_ms=elapsed_ms,
                    was_fallback=False,
                )

            return response, self.primary_model, False

        except Exception as e:
            self.stats.primary_failures += 1
            logger.warning(f"Primary model {self.primary_model} failed: {e}")

            # Try fallback if available
            if self._fallback_client and self.fallback_model:
                self.stats.fallback_calls += 1
                try:
                    response, input_tokens, output_tokens = await self._call_model(
                        self._fallback_client, self.fallback_model, messages
                    )
                    elapsed_ms = (time.perf_counter() - start_time) * 1000
                    self.stats.fallback_successes += 1
                    self.stats.total_latency_ms += elapsed_ms

                    # Record usage
                    if self.on_usage:
                        model_info = self._get_model_info(self.fallback_model)
                        cost = calculate_cost(
                            self.fallback_model, input_tokens, output_tokens
                        )
                        self.on_usage(
                            provider=model_info.provider.value,
                            model=self.fallback_model,
                            input_tokens=input_tokens,
                            output_tokens=output_tokens,
                            cost_cents=cost * 100,
                            latency_ms=elapsed_ms,
                            was_fallback=True,
                        )

                    return response, self.fallback_model, True

                except Exception as fallback_error:
                    self.stats.fallback_failures += 1
                    logger.error(
                        f"Fallback model {self.fallback_model} also failed: {fallback_error}"
                    )
                    raise RuntimeError(
                        f"Both primary ({self.primary_model}) and fallback ({self.fallback_model}) models failed"
                    ) from fallback_error

            # No fallback available
            raise RuntimeError(
                f"Primary model {self.primary_model} failed with no fallback"
            ) from e

    def get_stats(self) -> dict:
        """Get usage statistics."""
        return self.stats.to_dict()

    def reset_stats(self):
        """Reset usage statistics."""
        self.stats = FallbackStats()
