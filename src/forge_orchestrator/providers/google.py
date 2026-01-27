"""Google provider implementation."""

from __future__ import annotations

import httpx

from forge_orchestrator.logging import get_logger
from forge_orchestrator.models.provider import ModelCapabilities, ModelConfig, ModelSuggestion
from forge_orchestrator.providers.base import BaseProvider, ProviderError

logger = get_logger(__name__)


class GoogleProvider(BaseProvider):
    """Google provider with API support for fetching Gemini models."""

    provider_id = "google"
    display_name = "Google"
    has_api = True

    # Patterns to identify chat-capable models
    CHAT_PATTERNS = ("gemini",)

    # Patterns to exclude (not chat models or deprecated)
    EXCLUDE_PATTERNS = (
        "embedding",
        "aqa",
        "text-bison",
        "chat-bison",
        "codechat-bison",
        "code-bison",
    )

    # Known reasoning models (thinking models)
    REASONING_PATTERNS = ("thinking", "think")

    async def fetch_models(self) -> list[ModelConfig]:
        """Fetch available models from Google Generative AI API.

        Returns:
            List of chat-capable Gemini models.

        Raises:
            ProviderError: If API call fails.
        """
        if not self.api_key:
            raise ProviderError(self.provider_id, "API key not configured")

        url = "https://generativelanguage.googleapis.com/v1beta/models"

        logger.info("Fetching models from Google", url=url)

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(
                    url,
                    params={"key": self.api_key.get_secret_value()},
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

        raw_models = data.get("models", [])
        logger.info("Received models from Google", count=len(raw_models))

        # Filter and transform models
        models = []
        for raw in raw_models:
            # Model name comes as "models/gemini-1.5-pro" - extract just the ID
            full_name = raw.get("name", "")
            model_id = full_name.replace("models/", "")

            # Check if it's a chat-capable model
            if not self._is_chat_model(model_id, raw):
                continue

            # Determine capabilities from API data
            capabilities = self._detect_capabilities(model_id, raw)

            model = ModelConfig(
                id=model_id,
                display_name=raw.get("displayName", self._format_display_name(model_id)),
                category="chat",
                source="api",
                capabilities=capabilities,
            )
            models.append(model)

        # Sort by name
        models.sort(key=lambda m: m.id)

        logger.info("Processed Google models", count=len(models))
        return models

    def _is_chat_model(self, model_id: str, raw: dict) -> bool:
        """Check if a model is a chat-capable model."""
        model_lower = model_id.lower()

        # Check exclusions first
        for pattern in self.EXCLUDE_PATTERNS:
            if pattern in model_lower:
                return False

        # Check supported generation methods from API
        supported_methods = raw.get("supportedGenerationMethods", [])
        if "generateContent" not in supported_methods:
            return False

        # Check if it matches chat patterns
        for pattern in self.CHAT_PATTERNS:
            if pattern in model_lower:
                return True

        return False

    def _detect_capabilities(self, model_id: str, raw: dict) -> ModelCapabilities:
        """Detect model capabilities from model ID and API data."""
        model_lower = model_id.lower()

        # Reasoning models (thinking models)
        is_reasoning = any(pattern in model_lower for pattern in self.REASONING_PATTERNS)

        # Vision support - check input token limit or model name patterns
        # Gemini models generally support vision (multimodal)
        has_vision = "gemini" in model_lower

        # Tool support - most Gemini models support function calling
        has_tools = "gemini" in model_lower

        return ModelCapabilities(
            tools=has_tools,
            vision=has_vision,
            reasoning=is_reasoning,
        )

    def _format_display_name(self, model_id: str) -> str:
        """Format a display name from model ID."""
        # Capitalize Gemini and format nicely
        parts = model_id.split("-")
        if parts[0].lower() == "gemini":
            parts[0] = "Gemini"
        return " ".join(parts).replace("  ", " ")

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
