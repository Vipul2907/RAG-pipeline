"""
LLM provider abstraction for RAG generation.

Provides a unified interface for generating responses from LLMs
with support for Anthropic, OpenAI, and other providers.
"""

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

from ..config import config, logger


@dataclass
class GenerationResult:
    """Result of LLM generation."""
    text: str
    model: str
    usage: Optional[dict] = None


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model name."""
        pass

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.1,
        **kwargs,
    ) -> GenerationResult:
        """
        Generate a response from the LLM.

        Args:
            prompt: The user prompt/message.
            system_prompt: Optional system prompt for context.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            **kwargs: Additional provider-specific parameters.

        Returns:
            GenerationResult with the generated text and metadata.
        """
        pass


class AnthropicProvider(LLMProvider):
    """
    Anthropic Claude API provider.

    Uses Anthropic's Messages API for generation.
    Requires ANTHROPIC_API_KEY environment variable.
    """

    # Model configurations
    DEFAULT_MODEL = "claude-3-5-haiku-20241022"
    MODEL_ALIASES = {
        "haiku": "claude-3-5-haiku-20241022",
        "sonnet": "claude-3-5-sonnet-20241022",
        "opus": "claude-3-opus-20240229",
        "haiku-3": "claude-3-haiku-20240307",
        "sonnet-3": "claude-3-sonnet-20240229",
    }

    def __init__(
        self,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        max_tokens: int = 1024,
        timeout: float = 60.0,
        max_retries: int = 3,
    ):
        """
        Initialize the Anthropic provider.

        Args:
            model_name: Model name or alias (e.g., "haiku", "sonnet", "opus").
            api_key: Anthropic API key (defaults to ANTHROPIC_API_KEY env var).
            max_tokens: Default max tokens for generation.
            timeout: Request timeout in seconds.
            max_retries: Maximum number of retries.
        """
        try:
            import anthropic
        except ImportError as e:
            raise ImportError(
                "anthropic package is required for AnthropicProvider. "
                "Install with: pip install anthropic"
            ) from e

        # Resolve model alias
        resolved_model = self.MODEL_ALIASES.get(model_name, model_name or self.DEFAULT_MODEL)

        self._model_name = resolved_model
        self._default_max_tokens = max_tokens

        # Get API key
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Anthropic API key not found. Set ANTHROPIC_API_KEY environment variable "
                "or pass api_key parameter."
            )

        self._client = anthropic.Anthropic(
            api_key=self.api_key,
            timeout=timeout,
            max_retries=max_retries,
        )

        logger.info(f"Initialized AnthropicProvider: model={self._model_name}")

    @property
    def model_name(self) -> str:
        return self._model_name

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = None,
        temperature: float = 0.1,
        **kwargs,
    ) -> GenerationResult:
        """Generate response using Anthropic API."""
        max_tokens = max_tokens or self._default_max_tokens

        messages = [{"role": "user", "content": prompt}]

        try:
            response = self._client.messages.create(
                model=self._model_name,
                messages=messages,
                system=system_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs,
            )

            # Extract text content
            text = ""
            for block in response.content:
                if block.type == "text":
                    text += block.text

            usage = {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            }

            return GenerationResult(
                text=text,
                model=self._model_name,
                usage=usage,
            )

        except Exception as e:
            logger.error(f"Anthropic generation failed: {e}")
            raise


class OpenAIProvider(LLMProvider):
    """
    OpenAI API provider.

    Uses OpenAI's Chat Completions API for generation.
    Requires OPENAI_API_KEY environment variable.
    """

    DEFAULT_MODEL = "gpt-4o-mini"
    MODEL_ALIASES = {
        "gpt-4o-mini": "gpt-4o-mini",
        "gpt-4o": "gpt-4o",
        "gpt-4-turbo": "gpt-4-turbo",
        "gpt-3.5-turbo": "gpt-3.5-turbo",
    }

    def __init__(
        self,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        max_tokens: int = 1024,
        timeout: float = 60.0,
        max_retries: int = 3,
        base_url: Optional[str] = None,
    ):
        """
        Initialize the OpenAI provider.

        Args:
            model_name: Model name or alias.
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var).
            max_tokens: Default max tokens for generation.
            timeout: Request timeout in seconds.
            max_retries: Maximum number of retries.
            base_url: Custom base URL for OpenAI-compatible APIs (e.g., Groq).
        """
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError(
                "openai package is required for OpenAIProvider. "
                "Install with: pip install openai"
            ) from e

        resolved_model = self.MODEL_ALIASES.get(model_name, model_name or self.DEFAULT_MODEL)

        self._model_name = resolved_model
        self._default_max_tokens = max_tokens

        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key not found. Set OPENAI_API_KEY environment variable "
                "or pass api_key parameter."
            )

        self._client = OpenAI(
            api_key=self.api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
        )

        logger.info(f"Initialized OpenAIProvider: model={self._model_name}")

    @property
    def model_name(self) -> str:
        return self._model_name

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = None,
        temperature: float = 0.1,
        **kwargs,
    ) -> GenerationResult:
        """Generate response using OpenAI API."""
        max_tokens = max_tokens or self._default_max_tokens

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            response = self._client.chat.completions.create(
                model=self._model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs,
            )

            text = response.choices[0].message.content or ""

            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }

            return GenerationResult(
                text=text,
                model=self._model_name,
                usage=usage,
            )

        except Exception as e:
            logger.error(f"OpenAI generation failed: {e}")
            raise


class GroqProvider(LLMProvider):
    """
    Groq API provider.

    Uses Groq's OpenAI-compatible API for fast inference.
    Requires GROQ_API_KEY environment variable.

    Popular models:
    - "openai/gpt-oss-120b" (120B parameters, open weights)
    - "llama-3.1-70b-versatile" (70B Llama 3.1)
    - "llama-3.1-8b-instant" (8B Llama 3.1, fastest)
    - "mixtral-8x7b-32768" (Mixtral 8x7B)
    - "gemma2-9b-it" (Gemma 2 9B)
    """

    DEFAULT_MODEL = "openai/gpt-oss-120b"
    GROQ_BASE_URL = "https://api.groq.com/openai/v1"

    MODEL_ALIASES = {
        "gpt-oss-120b": "openai/gpt-oss-120b",
        "llama-70b": "llama-3.1-70b-versatile",
        "llama-8b": "llama-3.1-8b-instant",
        "mixtral": "mixtral-8x7b-32768",
        "gemma2": "gemma2-9b-it",
    }

    def __init__(
        self,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        max_tokens: int = 1024,
        timeout: float = 60.0,
        max_retries: int = 3,
    ):
        """
        Initialize the Groq provider.

        Args:
            model_name: Model name or alias (e.g., "gpt-oss-120b", "llama-70b").
            api_key: Groq API key (defaults to GROQ_API_KEY env var).
            max_tokens: Default max tokens for generation.
            timeout: Request timeout in seconds.
            max_retries: Maximum number of retries.
        """
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError(
                "openai package is required for GroqProvider (uses OpenAI-compatible API). "
                "Install with: pip install openai"
            ) from e

        # Resolve model alias
        resolved_model = self.MODEL_ALIASES.get(model_name, model_name or self.DEFAULT_MODEL)

        self._model_name = resolved_model
        self._default_max_tokens = max_tokens

        # Get API key
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Groq API key not found. Set GROQ_API_KEY environment variable "
                "or pass api_key parameter."
            )

        # Use OpenAI client with Groq's base URL
        self._client = OpenAI(
            api_key=self.api_key,
            base_url=self.GROQ_BASE_URL,
            timeout=timeout,
            max_retries=max_retries,
        )

        logger.info(f"Initialized GroqProvider: model={self._model_name}")

    @property
    def model_name(self) -> str:
        return self._model_name

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = None,
        temperature: float = 0.1,
        **kwargs,
    ) -> GenerationResult:
        """Generate response using Groq API (OpenAI-compatible)."""
        max_tokens = max_tokens or self._default_max_tokens

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            response = self._client.chat.completions.create(
                model=self._model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs,
            )

            text = response.choices[0].message.content or ""

            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }

            return GenerationResult(
                text=text,
                model=self._model_name,
                usage=usage,
            )

        except Exception as e:
            logger.error(f"Groq generation failed: {e}")
            raise


def get_llm_provider(provider_name: Optional[str] = None, **kwargs) -> LLMProvider:
    """
    Factory function to get an LLM provider by name.

    Args:
        provider_name: Name of provider ("anthropic", "openai", "groq", or from config).
        **kwargs: Additional arguments passed to the provider constructor.

    Returns:
        LLMProvider instance.

    Raises:
        ValueError: If provider name is unknown.
    """
    name = (provider_name or getattr(config, "llm_provider", "anthropic")).lower()

    if name in ("anthropic", "claude"):
        return AnthropicProvider(**kwargs)
    elif name in ("openai", "gpt"):
        return OpenAIProvider(**kwargs)
    elif name in ("groq",):
        return GroqProvider(**kwargs)
    else:
        raise ValueError(
            f"Unknown LLM provider: {name}. "
            f"Supported: 'anthropic', 'openai', 'groq'. "
            f"Set LLM_PROVIDER env var or pass provider_name."
        )