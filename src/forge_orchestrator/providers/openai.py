"""OpenAI provider implementation."""

from __future__ import annotations

import httpx

from forge_orchestrator.logging import get_logger
from forge_orchestrator.models.provider import ModelCapabilities, ModelConfig, ModelSuggestion
from forge_orchestrator.providers.base import BaseProvider, ProviderError

logger = get_logger(__name__)


class OpenAIProvider(BaseProvider):
    """OpenAI provider with API support for fetching models."""

    provider_id = "openai"
    display_name = "OpenAI"
    has_api = True

    # Patterns for chat-capable models
    CHAT_PATTERNS = ("gpt-", "o1", "o3", "chatgpt-")

    # Patterns to exclude (not chat models)
    EXCLUDE_PATTERNS = (
        "-instruct",
        "whisper",
        "dall-e",
        "tts-",
        "text-embedding",
        "embedding",
        "babbage",
        "davinci",
        "ada",
        "curie",
    )

    # Known reasoning models
    REASONING_MODELS = {"o1", "o1-mini", "o1-preview", "o1-pro", "o3", "o3-mini"}

    async def fetch_models(self) -> list[ModelConfig]:
        """Fetch available models from OpenAI API.

        Returns:
            List of chat-capable models.

        Raises:
            ProviderError: If API call fails.
        """
        if not self.api_key:
            raise ProviderError(self.provider_id, "API key not configured")

        url = "https://api.openai.com/v1/models"

        logger.info("Fetching models from OpenAI", url=url)

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(
                    url,
                    headers={"Authorization": f"Bearer {self.api_key.get_secret_value()}"},
                )
                response.raise_for_status()
                data = response.json()
        except httpx.HTTPStatusError as e:
            raise ProviderError(
                self.provider_id,
                f"HTTP error: {e.response.status_code}",
                retryable=e.response.status_code >= 500,
            ) from e
        except httpx.RequestError as e:
            raise ProviderError(self.provider_id, f"Request failed: {e}", retryable=True) from e

        raw_models = data.get("data", [])
        logger.info("Received models from OpenAI", count=len(raw_models))

        # Filter and transform models
        models = []
        for raw in raw_models:
            model_id = raw.get("id", "")

            # Check if it's a chat-capable model
            if not self._is_chat_model(model_id):
                continue

            # Determine capabilities
            capabilities = self._detect_capabilities(model_id)

            model = ModelConfig(
                id=model_id,
                display_name=self._format_display_name(model_id),
                category="chat",
                source="api",
                capabilities=capabilities,
            )
            models.append(model)

        # Sort by name
        models.sort(key=lambda m: m.id)

        logger.info("Processed OpenAI models", count=len(models))
        return models

    def _is_chat_model(self, model_id: str) -> bool:
        """Check if a model ID represents a chat-capable model."""
        model_lower = model_id.lower()

        # Check exclusions first
        for pattern in self.EXCLUDE_PATTERNS:
            if pattern in model_lower:
                return False

        # Check if it matches chat patterns
        for pattern in self.CHAT_PATTERNS:
            if model_lower.startswith(pattern):
                return True

        return False

    def _detect_capabilities(self, model_id: str) -> ModelCapabilities:
        """Detect model capabilities based on model ID."""
        model_lower = model_id.lower()

        # Reasoning models
        is_reasoning = any(
            model_lower == r or model_lower.startswith(f"{r}-")
            for r in self.REASONING_MODELS
        )

        # Vision support (gpt-4o, gpt-4-turbo, etc.)
        has_vision = "gpt-4o" in model_lower or "gpt-4-turbo" in model_lower

        # Tool support (most GPT-4 and GPT-3.5-turbo models)
        has_tools = (
            "gpt-4" in model_lower
            or "gpt-3.5-turbo" in model_lower
            or model_lower.startswith("o1")
            or model_lower.startswith("o3")
        )

        return ModelCapabilities(
            tools=has_tools,
            vision=has_vision,
            reasoning=is_reasoning,
        )

    def _format_display_name(self, model_id: str) -> str:
        """Format a display name from model ID."""
        # Simple formatting: capitalize and replace hyphens
        parts = model_id.split("-")
        if parts[0].lower() == "gpt":
            parts[0] = "GPT"
        return "-".join(parts)

    def get_suggestions(self) -> list[ModelSuggestion]:
        """Get suggested OpenAI models for manual addition."""
        return [
            ModelSuggestion(
                id="gpt-4o",
                display_name="GPT-4o",
                recommended=True,
                capabilities=ModelCapabilities(tools=True, vision=True, reasoning=False),
            ),
            ModelSuggestion(
                id="gpt-4o-mini",
                display_name="GPT-4o Mini",
                capabilities=ModelCapabilities(tools=True, vision=True, reasoning=False),
            ),
            ModelSuggestion(
                id="o1",
                display_name="o1",
                capabilities=ModelCapabilities(tools=True, vision=False, reasoning=True),
            ),
            ModelSuggestion(
                id="o1-mini",
                display_name="o1-mini",
                capabilities=ModelCapabilities(tools=True, vision=False, reasoning=True),
            ),
            ModelSuggestion(
                id="gpt-4-turbo",
                display_name="GPT-4 Turbo",
                capabilities=ModelCapabilities(tools=True, vision=True, reasoning=False),
            ),
        ]
