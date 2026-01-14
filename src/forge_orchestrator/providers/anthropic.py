"""Anthropic provider implementation."""

from __future__ import annotations

from forge_orchestrator.logging import get_logger
from forge_orchestrator.models.provider import ModelCapabilities, ModelConfig, ModelSuggestion
from forge_orchestrator.providers.base import BaseProvider, ProviderError

logger = get_logger(__name__)


class AnthropicProvider(BaseProvider):
    """Anthropic provider (manual model addition only)."""

    provider_id = "anthropic"
    display_name = "Anthropic"
    has_api = False

    async def fetch_models(self) -> list[ModelConfig]:
        """Anthropic does not have a public models API.

        Raises:
            ProviderError: Always, as this provider has no API.
        """
        raise ProviderError(
            self.provider_id,
            "Anthropic does not have a public models API. Use suggestions to add models manually.",
        )

    def get_suggestions(self) -> list[ModelSuggestion]:
        """Get suggested Anthropic models for manual addition."""
        return [
            ModelSuggestion(
                id="claude-sonnet-4-20250514",
                display_name="Claude Sonnet 4",
                recommended=True,
                capabilities=ModelCapabilities(tools=True, vision=True, reasoning=False),
            ),
            ModelSuggestion(
                id="claude-opus-4-20250514",
                display_name="Claude Opus 4",
                capabilities=ModelCapabilities(tools=True, vision=True, reasoning=False),
            ),
            ModelSuggestion(
                id="claude-3-5-sonnet-20241022",
                display_name="Claude 3.5 Sonnet",
                capabilities=ModelCapabilities(tools=True, vision=True, reasoning=False),
            ),
            ModelSuggestion(
                id="claude-3-5-haiku-20241022",
                display_name="Claude 3.5 Haiku",
                capabilities=ModelCapabilities(tools=True, vision=True, reasoning=False),
            ),
            ModelSuggestion(
                id="claude-3-7-sonnet-20250219",
                display_name="Claude 3.7 Sonnet",
                capabilities=ModelCapabilities(tools=True, vision=True, reasoning=False),
            ),
            ModelSuggestion(
                id="claude-3-7-sonnet-20250219:thinking",
                display_name="Claude 3.7 Sonnet (Thinking)",
                capabilities=ModelCapabilities(tools=True, vision=True, reasoning=True),
            ),
        ]
