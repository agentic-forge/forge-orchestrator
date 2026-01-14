"""Provider abstraction layer for multi-provider model management."""

from forge_orchestrator.providers.base import BaseProvider, ProviderError
from forge_orchestrator.providers.registry import ProviderRegistry, provider_registry

# Import providers to trigger registration
from forge_orchestrator.providers.anthropic import AnthropicProvider
from forge_orchestrator.providers.google import GoogleProvider
from forge_orchestrator.providers.openai import OpenAIProvider
from forge_orchestrator.providers.openrouter import OpenRouterProvider

# Register all providers
provider_registry.register(OpenAIProvider)
provider_registry.register(AnthropicProvider)
provider_registry.register(GoogleProvider)
provider_registry.register(OpenRouterProvider)

__all__ = [
    "BaseProvider",
    "ProviderError",
    "ProviderRegistry",
    "provider_registry",
    "OpenAIProvider",
    "AnthropicProvider",
    "GoogleProvider",
    "OpenRouterProvider",
]
