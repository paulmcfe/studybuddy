"""Ollama client for local model development.

Provides integration with Ollama for running open-source models locally.
Based on patterns from chapter13.ipynb.
"""

import os
import urllib.request
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Conditional import pattern from chapter13.ipynb
try:
    from langchain_ollama import ChatOllama
    from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    ChatOllama = None
    HumanMessage = None
    SystemMessage = None
    AIMessage = None
    logger.warning("langchain_ollama not installed. Ollama support disabled.")

OLLAMA_BASE_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")


def check_ollama_running() -> bool:
    """Check if Ollama server is running locally."""
    try:
        urllib.request.urlopen(f"{OLLAMA_BASE_URL}/api/tags", timeout=2)
        return True
    except Exception:
        return False


# Check at module load time
OLLAMA_RUNNING = check_ollama_running() if OLLAMA_AVAILABLE else False


def get_ollama_status() -> dict:
    """Get Ollama availability status for API response."""
    running = check_ollama_running() if OLLAMA_AVAILABLE else False
    return {
        "installed": OLLAMA_AVAILABLE,
        "running": running,
        "base_url": OLLAMA_BASE_URL,
    }


class OllamaClient:
    """Wrapper for Ollama with standardized interface.

    Provides async chat completion that matches the interface
    expected by ModelFallbackChain.
    """

    def __init__(self, model: str = "llama3.2"):
        if not OLLAMA_AVAILABLE:
            raise RuntimeError(
                "langchain_ollama not installed. Install with: pip install langchain-ollama"
            )
        if not check_ollama_running():
            raise RuntimeError(
                f"Ollama server not running at {OLLAMA_BASE_URL}. Start with: ollama serve"
            )

        self.model = model
        self.llm = ChatOllama(model=model, base_url=OLLAMA_BASE_URL)

    def _convert_messages(self, messages: list[dict]) -> list:
        """Convert OpenAI-style messages to LangChain format."""
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
        return lc_messages

    async def chat(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> tuple[str, int, int]:
        """Send chat completion request.

        Returns: (response_text, input_tokens, output_tokens)

        Note: Ollama doesn't provide token counts directly, so we estimate.
        """
        lc_messages = self._convert_messages(messages)

        # Create configured LLM instance (temperature/num_predict are constructor params)
        llm_kwargs = {"model": self.model, "base_url": OLLAMA_BASE_URL}
        if temperature is not None:
            llm_kwargs["temperature"] = temperature
        if max_tokens is not None:
            llm_kwargs["num_predict"] = max_tokens

        llm = ChatOllama(**llm_kwargs)
        result = await llm.ainvoke(lc_messages)

        response_text = result.content

        # Estimate tokens (rough approximation: 4 chars per token)
        input_text = " ".join(m.get("content", "") for m in messages)
        input_tokens = len(input_text) // 4
        output_tokens = len(response_text) // 4

        return response_text, input_tokens, output_tokens

    async def astream(self, messages: list[dict], **kwargs):
        """Stream chat completion response.

        Yields chunks of the response as they arrive.
        """
        lc_messages = self._convert_messages(messages)

        async for chunk in self.llm.astream(lc_messages):
            if hasattr(chunk, "content"):
                yield chunk.content

    def chat_sync(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> tuple[str, int, int]:
        """Synchronous version of chat for non-async contexts."""
        lc_messages = self._convert_messages(messages)

        llm_kwargs = {"model": self.model, "base_url": OLLAMA_BASE_URL}
        if temperature is not None:
            llm_kwargs["temperature"] = temperature
        if max_tokens is not None:
            llm_kwargs["num_predict"] = max_tokens

        llm = ChatOllama(**llm_kwargs)
        result = llm.invoke(lc_messages)

        response_text = result.content

        input_text = " ".join(m.get("content", "") for m in messages)
        input_tokens = len(input_text) // 4
        output_tokens = len(response_text) // 4

        return response_text, input_tokens, output_tokens
