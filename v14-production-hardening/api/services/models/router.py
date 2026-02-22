"""Task-based model routing for StudyBuddy v13.

Routes tasks to appropriate models based on user configuration
or system defaults.
"""

import logging
from typing import Optional, Callable, Any

from sqlalchemy.orm import Session

from .config import (
    Provider,
    TaskType,
    ModelInfo,
    AVAILABLE_MODELS,
    DEFAULT_TASK_MODELS,
)
from .fallback import ModelFallbackChain
from .ollama_client import OLLAMA_AVAILABLE, check_ollama_running
from .together_client import TOGETHER_AVAILABLE, _get_api_key as _get_together_key

logger = logging.getLogger(__name__)


def select_model_for_task(
    task_type: TaskType,
    context_needed: int = 4096,
    prefer_local: bool = False,
    prefer_open_source: bool = False,
) -> str:
    """Recommend a model based on task requirements.

    Args:
        task_type: The type of task to perform
        context_needed: Required context window size
        prefer_local: Prefer Ollama if available
        prefer_open_source: Prefer Together AI over OpenAI

    Returns:
        Model key from AVAILABLE_MODELS
    """
    # Check for local preference
    if prefer_local and OLLAMA_AVAILABLE and check_ollama_running():
        return "llama3.2"

    # Get default for task type
    defaults = DEFAULT_TASK_MODELS.get(task_type, {})
    primary = defaults.get("primary", "gpt-4o-mini")

    # Check if primary is available
    model_info = AVAILABLE_MODELS.get(primary)
    if model_info:
        # Check provider availability
        if model_info.provider == Provider.TOGETHER:
            if not TOGETHER_AVAILABLE or not _get_together_key():
                # Fall back to OpenAI
                return defaults.get("fallback", "gpt-4o-mini")
        elif model_info.provider == Provider.OLLAMA:
            if not OLLAMA_AVAILABLE or not check_ollama_running():
                return defaults.get("fallback", "gpt-4o-mini")

        # Check context requirement
        if model_info.context_length < context_needed:
            # Find a model with larger context
            if context_needed > 100_000:
                return "llama-4-scout"  # 10M context
            return "gpt-4o"  # 128K context

        return primary

    return "gpt-4o-mini"  # Safe default


class ModelRouter:
    """Route tasks to appropriate models based on configuration.

    The router checks:
    1. User-specific configuration (from database)
    2. System defaults (from DEFAULT_TASK_MODELS)
    3. Provider availability (Ollama running, Together AI configured)
    """

    def __init__(
        self,
        db: Optional[Session] = None,
        user_id: Optional[str] = None,
        on_usage: Optional[Callable[..., Any]] = None,
    ):
        self.db = db
        self.user_id = user_id
        self.on_usage = on_usage
        self._chains: dict[TaskType, ModelFallbackChain] = {}

    def _get_user_config(self, task_type: TaskType) -> Optional[dict]:
        """Get user-specific model configuration from database."""
        if not self.db or not self.user_id:
            return None

        from ...database.models import ModelConfig

        config = (
            self.db.query(ModelConfig)
            .filter(
                ModelConfig.user_id == self.user_id,
                ModelConfig.task_type == task_type.value,
            )
            .first()
        )

        if config:
            return {
                "primary_provider": config.primary_provider,
                "primary_model": config.primary_model,
                "fallback_provider": config.fallback_provider,
                "fallback_model": config.fallback_model,
                "temperature": config.temperature,
                "max_tokens": config.max_tokens,
            }

        return None

    def _find_model_key(self, provider: str, model_id: str) -> Optional[str]:
        """Find model key by provider and model_id."""
        for key, info in AVAILABLE_MODELS.items():
            if info.provider.value == provider and info.model_id == model_id:
                return key
        return None

    def _check_model_available(self, model_key: str) -> bool:
        """Check if a model is currently available."""
        model_info = AVAILABLE_MODELS.get(model_key)
        if not model_info:
            return False

        if model_info.provider == Provider.OLLAMA:
            return OLLAMA_AVAILABLE and check_ollama_running()
        elif model_info.provider == Provider.TOGETHER:
            return TOGETHER_AVAILABLE and bool(_get_together_key())

        return True  # OpenAI is always available

    def get_chain_for_task(
        self,
        task_type: TaskType,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> ModelFallbackChain:
        """Get or create a fallback chain for a task type."""
        # Check cache first
        if task_type in self._chains:
            return self._chains[task_type]

        # Get configuration
        user_config = self._get_user_config(task_type)

        if user_config:
            # Use user configuration
            primary = self._find_model_key(
                user_config["primary_provider"],
                user_config["primary_model"],
            )
            fallback = None
            if user_config.get("fallback_provider") and user_config.get(
                "fallback_model"
            ):
                fallback = self._find_model_key(
                    user_config["fallback_provider"],
                    user_config["fallback_model"],
                )
            temperature = user_config.get("temperature", temperature)
            max_tokens = user_config.get("max_tokens", max_tokens)
        else:
            # Use system defaults
            defaults = DEFAULT_TASK_MODELS.get(task_type, {})
            primary = defaults.get("primary", "gpt-4o-mini")
            fallback = defaults.get("fallback")

        # Check availability and adjust
        if not self._check_model_available(primary):
            logger.warning(f"Primary model {primary} not available, using fallback")
            if fallback and self._check_model_available(fallback):
                primary = fallback
                fallback = None
            else:
                primary = "gpt-4o-mini"  # Safe default
                fallback = None

        if fallback and not self._check_model_available(fallback):
            logger.warning(f"Fallback model {fallback} not available")
            fallback = None

        # Create chain
        chain = ModelFallbackChain(
            primary_model=primary,
            fallback_model=fallback,
            on_usage=self.on_usage,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        self._chains[task_type] = chain
        return chain

    async def invoke_for_task(
        self,
        task_type: TaskType,
        messages: list[dict],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> tuple[str, str, bool]:
        """Invoke the appropriate model for a task.

        Returns: (response_text, model_used, was_fallback)
        """
        chain = self.get_chain_for_task(
            task_type,
            temperature=temperature or 0.7,
            max_tokens=max_tokens,
        )
        return await chain.invoke(messages)

    def get_recommended_model(
        self,
        task_type: TaskType,
        context_needed: int = 4096,
    ) -> str:
        """Get recommended model for a task without invoking."""
        return select_model_for_task(task_type, context_needed)

    def get_available_models(self) -> dict[str, dict]:
        """Get all available models with their current status."""
        result = {}
        for key, info in AVAILABLE_MODELS.items():
            result[key] = {
                "provider": info.provider.value,
                "display_name": info.display_name,
                "model_id": info.model_id,
                "context_length": info.context_length,
                "input_cost_per_1m": info.input_cost_per_1m,
                "output_cost_per_1m": info.output_cost_per_1m,
                "best_for": info.best_for,
                "is_available": self._check_model_available(key),
                "is_embedding": info.is_embedding,
            }
        return result

    def clear_cache(self):
        """Clear cached fallback chains."""
        self._chains.clear()
