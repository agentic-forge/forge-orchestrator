"""OpenRouter provider implementation."""

from __future__ import annotations

import httpx

from forge_orchestrator.logging import get_logger
from forge_orchestrator.models.provider import ModelCapabilities, ModelConfig, ModelSuggestion
from forge_orchestrator.providers.base import BaseProvider, ProviderError

logger = get_logger(__name__)


class OpenRouterProvider(BaseProvider):
    """OpenRouter provider with API support for fetching models.

    OpenRouter aggregates models from many providers. We fetch all
    and display them as a single "OpenRouter" provider, with the
    underlying provider shown as metadata.
    """

    provider_id = "openrouter"
    display_name = "OpenRouter"
    has_api = True

    BASE_URL = "https://openrouter.ai/api/v1"

    # Known reasoning model patterns
    REASONING_PATTERNS = (
        "o1",
        "o3",
        "deepseek-r1",
        "qwq",
        "thinking",
        "reasoner",
    )

    async def fetch_models(self) -> list[ModelConfig]:
        """Fetch available models from OpenRouter API.

        Returns:
            List of all available models.

        Raises:
            ProviderError: If API call fails.
        """
        if not self.api_key:
            raise ProviderError(self.provider_id, "API key not configured")

        url = f"{self.BASE_URL}/models"

        logger.info("Fetching models from OpenRouter", url=url)

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(url)
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
        logger.info("Received models from OpenRouter", count=len(raw_models))

        # Transform models
        models = []
        for raw in raw_models:
            model_id = raw.get("id", "")
            if not model_id:
                continue

            # Detect capabilities
            capabilities = self._detect_capabilities(raw)

            # Determine category
            category = self._detect_category(model_id, raw)

            model = ModelConfig(
                id=model_id,
                display_name=raw.get("name", model_id),
                category=category,
                source="api",
                capabilities=capabilities,
            )
            models.append(model)

        # Sort by name
        models.sort(key=lambda m: m.display_name.lower())

        logger.info("Processed OpenRouter models", count=len(models))
        return models

    def _detect_capabilities(self, raw: dict) -> ModelCapabilities:
        """Detect model capabilities from API response."""
        # Tool support from supported_parameters
        supported_params = raw.get("supported_parameters", []) or []
        has_tools = "tools" in supported_params

        # Vision support from modality
        architecture = raw.get("architecture", {}) or {}
        modality = architecture.get("modality", "") or ""
        has_vision = "image" in modality.lower()

        # Reasoning from model name patterns
        model_id = raw.get("id", "").lower()
        model_name = raw.get("name", "").lower()
        is_reasoning = any(
            pattern in model_id or pattern in model_name
            for pattern in self.REASONING_PATTERNS
        )

        return ModelCapabilities(
            tools=has_tools,
            vision=has_vision,
            reasoning=is_reasoning,
        )

    def _detect_category(self, model_id: str, raw: dict) -> str:
        """Detect model category (chat, embedding, other)."""
        model_lower = model_id.lower()

        # Embedding models
        if "embed" in model_lower:
            return "embedding"

        # Image generation models
        if "dall-e" in model_lower or "stable-diffusion" in model_lower:
            return "other"

        # Audio models
        if "whisper" in model_lower or "tts" in model_lower:
            return "other"

        # Default to chat
        return "chat"

    def get_suggestions(self) -> list[ModelSuggestion]:
        """Get suggested OpenRouter models for manual addition."""
        # OpenRouter has API, so suggestions are just popular models
        return [
            ModelSuggestion(
                id="anthropic/claude-sonnet-4",
                display_name="Claude Sonnet 4 (via OpenRouter)",
                recommended=True,
                capabilities=ModelCapabilities(tools=True, vision=True, reasoning=False),
            ),
            ModelSuggestion(
                id="openai/gpt-4o",
                display_name="GPT-4o (via OpenRouter)",
                capabilities=ModelCapabilities(tools=True, vision=True, reasoning=False),
            ),
            ModelSuggestion(
                id="google/gemini-2.0-flash-001",
                display_name="Gemini 2.0 Flash (via OpenRouter)",
                capabilities=ModelCapabilities(tools=True, vision=True, reasoning=False),
            ),
            ModelSuggestion(
                id="deepseek/deepseek-r1",
                display_name="DeepSeek R1 (via OpenRouter)",
                capabilities=ModelCapabilities(tools=True, vision=False, reasoning=True),
            ),
        ]
