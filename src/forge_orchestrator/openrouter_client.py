"""Client for OpenRouter API."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime

import httpx

from forge_orchestrator.logging import get_logger
from forge_orchestrator.models.openrouter import ModelInfo, ModelPricing, ModelsData

logger = get_logger(__name__)


class OpenRouterError(Exception):
    """Error from OpenRouter API."""


class OpenRouterClient:
    """Client for fetching models from OpenRouter API."""

    def __init__(
        self,
        base_url: str = "https://openrouter.ai/api/v1",
        provider_whitelist: list[str] | None = None,
        models_per_provider: int = 3,
        model_include_list: list[str] | None = None,
    ):
        """Initialize the OpenRouter client.

        Args:
            base_url: OpenRouter API base URL.
            provider_whitelist: List of provider IDs to include. If None, include all.
            models_per_provider: Maximum number of models to keep per provider.
            model_include_list: List of model IDs to always include (e.g., thinking models).
        """
        self.base_url = base_url.rstrip("/")
        self.provider_whitelist = provider_whitelist
        self.models_per_provider = models_per_provider
        self.model_include_list = set(model_include_list) if model_include_list else set()

    async def fetch_models(self) -> ModelsData:
        """Fetch models from OpenRouter API.

        Returns:
            ModelsData with filtered and sorted models.

        Raises:
            OpenRouterError: If the API request fails.
        """
        url = f"{self.base_url}/models"

        logger.info("Fetching models from OpenRouter", url=url)

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(url)
                response.raise_for_status()
                data = response.json()
        except httpx.HTTPStatusError as e:
            raise OpenRouterError(f"HTTP error: {e.response.status_code}") from e
        except httpx.RequestError as e:
            raise OpenRouterError(f"Request failed: {e}") from e

        raw_models = data.get("data", [])
        logger.info("Received models from OpenRouter", count=len(raw_models))

        # Transform and filter models
        models = self._process_models(raw_models)

        # Extract unique providers
        providers = sorted({m.provider for m in models})

        logger.info(
            "Processed models",
            model_count=len(models),
            provider_count=len(providers),
            providers=providers,
        )

        return ModelsData(
            models=models,
            providers=providers,
            fetched_at=datetime.utcnow(),
        )

    def _process_models(self, raw_models: list[dict]) -> list[ModelInfo]:
        """Process raw models from API response.

        1. Filter by provider whitelist
        2. Transform to ModelInfo objects
        3. Group by provider
        4. Sort each group by created timestamp (newest first)
        5. Keep only top N per provider
        6. Always include models from model_include_list

        Args:
            raw_models: Raw model data from OpenRouter API.

        Returns:
            List of processed ModelInfo objects.
        """
        # Build a lookup of all models by ID (for include_list)
        all_models_by_id: dict[str, ModelInfo] = {}

        # Group models by provider
        by_provider: dict[str, list[ModelInfo]] = defaultdict(list)

        for raw in raw_models:
            model_id = raw.get("id", "")
            if "/" not in model_id:
                continue

            provider = model_id.split("/")[0]

            # Extract pricing
            pricing_data = raw.get("pricing", {})
            pricing = ModelPricing(
                prompt=float(pricing_data.get("prompt", 0) or 0),
                completion=float(pricing_data.get("completion", 0) or 0),
            )

            # Extract capabilities
            supported_params = raw.get("supported_parameters", []) or []
            supports_tools = "tools" in supported_params

            # Check for vision support from modality
            architecture = raw.get("architecture", {}) or {}
            modality = architecture.get("modality", "") or ""
            supports_vision = "image" in modality.lower()

            model = ModelInfo(
                id=model_id,
                name=raw.get("name", model_id),
                provider=provider,
                context_length=raw.get("context_length", 0) or 0,
                pricing=pricing,
                supports_tools=supports_tools,
                supports_vision=supports_vision,
                modality=modality if modality else None,
                created=raw.get("created"),
            )

            # Store all models for include_list lookup
            all_models_by_id[model_id] = model

            # Filter by whitelist if configured (for top N selection)
            if self.provider_whitelist and provider not in self.provider_whitelist:
                continue

            by_provider[provider].append(model)

        # Sort each provider's models by created timestamp (newest first)
        # and keep only top N
        result: list[ModelInfo] = []
        included_ids: set[str] = set()

        for provider in sorted(by_provider.keys()):
            models = by_provider[provider]

            # Sort by created timestamp (newest first), None values last
            models.sort(key=lambda m: m.created or 0, reverse=True)

            # Keep only top N
            top_models = models[: self.models_per_provider]
            result.extend(top_models)
            included_ids.update(m.id for m in top_models)

            logger.debug(
                "Provider models",
                provider=provider,
                total=len(models),
                kept=len(top_models),
                models=[m.id for m in top_models],
            )

        # Add any include_list models that weren't already included
        for model_id in self.model_include_list:
            if model_id not in included_ids and model_id in all_models_by_id:
                result.append(all_models_by_id[model_id])
                logger.debug("Added include_list model", model_id=model_id)

        # Sort final result by provider, then by name
        result.sort(key=lambda m: (m.provider, m.name))

        return result
