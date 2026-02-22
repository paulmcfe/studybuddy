"""Together AI client for hosted open-source models.

Provides integration with Together AI for running Llama, Mixtral,
and other open-source models in the cloud.
Based on patterns from chapter13.ipynb.
"""

import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Conditional import pattern
try:
    from together import Together

    TOGETHER_AVAILABLE = True
except ImportError:
    TOGETHER_AVAILABLE = False
    Together = None
    logger.warning("together SDK not installed. Together AI support disabled.")

def _get_api_key() -> str | None:
    """Get Together AI API key fresh from environment."""
    return os.environ.get("TOGETHER_API_KEY")

# For backwards compatibility
TOGETHER_API_KEY = _get_api_key()


def get_together_status() -> dict:
    """Get Together AI availability status for API response."""
    return {
        "installed": TOGETHER_AVAILABLE,
        "configured": bool(_get_api_key()),
    }


class TogetherAIClient:
    """Wrapper for Together AI API.

    Provides async chat completion and embedding that matches the
    interface expected by ModelFallbackChain.
    """

    def __init__(self, api_key: Optional[str] = None):
        if not TOGETHER_AVAILABLE:
            raise RuntimeError(
                "together SDK not installed. Install with: pip install together"
            )

        self.api_key = api_key or _get_api_key()
        if not self.api_key:
            raise ValueError(
                "TOGETHER_API_KEY required. Set it in environment or pass to constructor."
            )

        self.client = Together(api_key=self.api_key)

    async def chat(
        self,
        model: str,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> tuple[str, int, int]:
        """Send chat completion request.

        Returns: (response_text, input_tokens, output_tokens)
        """
        kwargs = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens

        # Together SDK is sync, but we wrap it for consistency
        response = self.client.chat.completions.create(**kwargs)

        response_text = response.choices[0].message.content

        # Extract token counts from response
        input_tokens = response.usage.prompt_tokens if response.usage else 0
        output_tokens = response.usage.completion_tokens if response.usage else 0

        return response_text, input_tokens, output_tokens

    async def astream(
        self,
        model: str,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ):
        """Stream chat completion response.

        Yields chunks of the response as they arrive.
        """
        kwargs = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "stream": True,
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens

        stream = self.client.chat.completions.create(**kwargs)

        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    async def embed(
        self,
        model: str,
        texts: list[str],
    ) -> list[list[float]]:
        """Generate embeddings for a list of texts.

        Returns: List of embedding vectors
        """
        response = self.client.embeddings.create(
            model=model,
            input=texts,
        )
        return [item.embedding for item in response.data]

    def chat_sync(
        self,
        model: str,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> tuple[str, int, int]:
        """Synchronous version of chat for non-async contexts."""
        kwargs = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens

        response = self.client.chat.completions.create(**kwargs)

        response_text = response.choices[0].message.content
        input_tokens = response.usage.prompt_tokens if response.usage else 0
        output_tokens = response.usage.completion_tokens if response.usage else 0

        return response_text, input_tokens, output_tokens

    def embed_sync(
        self,
        model: str,
        texts: list[str],
    ) -> list[list[float]]:
        """Synchronous version of embed."""
        response = self.client.embeddings.create(
            model=model,
            input=texts,
        )
        return [item.embedding for item in response.data]
