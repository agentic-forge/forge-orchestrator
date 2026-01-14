"""Google provider implementation."""

from __future__ import annotations

from forge_orchestrator.logging import get_logger
from forge_orchestrator.models.provider import ModelCapabilities, ModelConfig, ModelSuggestion
from forge_orchestrator.providers.base import BaseProvider, ProviderError

logger = get_logger(__name__)


class GoogleProvider(BaseProvider):
    """Google provider (manual model addition only).

    Google has a models API but it's complex (requires project auth).
    For simplicity, we use manual addition with suggestions.
    """

    provider_id = "google"
    display_name = "Google"
    has_api = False

    async def fetch_models(self) -> list[ModelConfig]:
        """Google models API is not supported.

        Raises:
            ProviderError: Always, as this provider has no API support.
        """
        raise ProviderError(
            self.provider_id,
            "Google models API is not supported. Use suggestions to add models manually.",
        )

    def get_suggestions(self) -> list[ModelSuggestion]:
        """Get suggested Google models for manual addition."""
        return [
            ModelSuggestion(
                id="gemini-2.0-flash",
                display_name="Gemini 2.0 Flash",
                recommended=True,
                capabilities=ModelCapabilities(tools=True, vision=True, reasoning=False),
            ),
            ModelSuggestion(
                id="gemini-2.0-flash-thinking-exp",
                display_name="Gemini 2.0 Flash Thinking",
                capabilities=ModelCapabilities(tools=True, vision=True, reasoning=True),
            ),
            ModelSuggestion(
                id="gemini-1.5-pro",
                display_name="Gemini 1.5 Pro",
                capabilities=ModelCapabilities(tools=True, vision=True, reasoning=False),
            ),
            ModelSuggestion(
                id="gemini-1.5-flash",
                display_name="Gemini 1.5 Flash",
                capabilities=ModelCapabilities(tools=True, vision=True, reasoning=False),
            ),
            ModelSuggestion(
                id="gemini-exp-1206",
                display_name="Gemini Experimental 1206",
                capabilities=ModelCapabilities(tools=True, vision=True, reasoning=False),
            ),
        ]
