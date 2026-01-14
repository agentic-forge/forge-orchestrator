"""Registry for LLM providers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import SecretStr

from forge_orchestrator.logging import get_logger

if TYPE_CHECKING:
    from forge_orchestrator.providers.base import BaseProvider

logger = get_logger(__name__)


class ProviderRegistry:
    """Registry for managing LLM providers.

    Provides factory methods to get provider instances and check
    which providers are available/configured.
    """

    def __init__(self) -> None:
        """Initialize the registry."""
        self._provider_classes: dict[str, type[BaseProvider]] = {}
        self._api_keys: dict[str, SecretStr | None] = {}

    def register(self, provider_class: type[BaseProvider]) -> type[BaseProvider]:
        """Register a provider class.

        Can be used as a decorator:
            @provider_registry.register
            class OpenAIProvider(BaseProvider):
                ...

        Args:
            provider_class: The provider class to register.

        Returns:
            The provider class (for decorator use).
        """
        provider_id = provider_class.provider_id
        self._provider_classes[provider_id] = provider_class
        logger.debug("Registered provider", provider_id=provider_id)
        return provider_class

    def set_api_key(self, provider_id: str, api_key: SecretStr | None) -> None:
        """Set the API key for a provider.

        Args:
            provider_id: Provider identifier.
            api_key: The API key (or None if not configured).
        """
        self._api_keys[provider_id] = api_key

    def configure_from_settings(self, settings) -> None:
        """Configure API keys from settings object.

        Args:
            settings: Settings object with API key attributes.
        """
        key_mappings = {
            "openai": "openai_api_key",
            "anthropic": "anthropic_api_key",
            "google": "gemini_api_key",
            "openrouter": "openrouter_api_key",
        }

        for provider_id, attr_name in key_mappings.items():
            api_key = getattr(settings, attr_name, None)
            self.set_api_key(provider_id, api_key)
            if api_key:
                logger.debug("Configured API key", provider=provider_id)

    def get(self, provider_id: str) -> BaseProvider | None:
        """Get a provider instance by ID.

        Args:
            provider_id: Provider identifier.

        Returns:
            Provider instance or None if not registered.
        """
        provider_class = self._provider_classes.get(provider_id)
        if provider_class is None:
            return None

        api_key = self._api_keys.get(provider_id)
        return provider_class(api_key=api_key)

    def get_all(self) -> list[BaseProvider]:
        """Get all registered provider instances.

        Returns:
            List of provider instances.
        """
        providers = []
        for provider_id in self._provider_classes:
            provider = self.get(provider_id)
            if provider:
                providers.append(provider)
        return providers

    def get_provider_ids(self) -> list[str]:
        """Get all registered provider IDs.

        Returns:
            List of provider IDs.
        """
        return list(self._provider_classes.keys())

    def is_configured(self, provider_id: str) -> bool:
        """Check if a provider has an API key configured.

        Args:
            provider_id: Provider identifier.

        Returns:
            True if provider has API key.
        """
        api_key = self._api_keys.get(provider_id)
        return api_key is not None

    def get_configured_providers(self) -> list[str]:
        """Get IDs of providers with API keys configured.

        Returns:
            List of configured provider IDs.
        """
        return [
            provider_id
            for provider_id, api_key in self._api_keys.items()
            if api_key is not None
        ]

    def has_api(self, provider_id: str) -> bool:
        """Check if a provider supports API model fetching.

        Args:
            provider_id: Provider identifier.

        Returns:
            True if provider has models API.
        """
        provider_class = self._provider_classes.get(provider_id)
        if provider_class is None:
            return False
        return provider_class.has_api


# Global registry instance
provider_registry = ProviderRegistry()
