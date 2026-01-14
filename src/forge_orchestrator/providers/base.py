"""Abstract base class for LLM providers."""

from __future__ import annotations

from abc import ABC, abstractmethod

from pydantic import SecretStr

from forge_orchestrator.models.provider import ModelConfig, ModelSuggestion


class BaseProvider(ABC):
    """Abstract base class for LLM providers.

    Each provider implements model fetching (if API available) and
    provides suggestion chips for manual model addition.
    """

    # Provider identifier (e.g., "openai", "anthropic")
    provider_id: str

    # Human-readable name (e.g., "OpenAI", "Anthropic")
    display_name: str

    # Whether this provider has an API to fetch models
    has_api: bool = False

    def __init__(self, api_key: SecretStr | None = None) -> None:
        """Initialize the provider.

        Args:
            api_key: Optional API key for this provider.
        """
        self.api_key = api_key

    @property
    def is_configured(self) -> bool:
        """Check if the provider is configured (has API key)."""
        return self.api_key is not None

    @abstractmethod
    async def fetch_models(self) -> list[ModelConfig]:
        """Fetch available models from the provider's API.

        Returns:
            List of ModelConfig for available models.

        Raises:
            NotImplementedError: If provider has no API (has_api=False).
            ProviderError: If API call fails.
        """

    @abstractmethod
    def get_suggestions(self) -> list[ModelSuggestion]:
        """Get suggested models for manual addition.

        Returns:
            List of ModelSuggestion with recommended models.
        """


class ProviderError(Exception):
    """Error from a provider operation."""

    def __init__(self, provider_id: str, message: str, retryable: bool = False) -> None:
        self.provider_id = provider_id
        self.retryable = retryable
        super().__init__(f"{provider_id}: {message}")
