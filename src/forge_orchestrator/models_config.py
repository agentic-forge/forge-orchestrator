"""JSON file storage for multi-provider model configuration."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import aiofiles
import aiofiles.os

from forge_orchestrator.logging import get_logger
from forge_orchestrator.models.provider import (
    LastUsedModel,
    ModelCapabilities,
    ModelConfig,
    ModelsConfigData,
    ProviderConfig,
    RecentModel,
)

if TYPE_CHECKING:
    from forge_orchestrator.models.openrouter import ModelsData

logger = get_logger(__name__)


class ModelsConfigError(Exception):
    """Error with models config operations."""


class ModelsConfig:
    """JSON file storage for multi-provider model configuration.

    Stores model configuration at the specified file path.
    Uses atomic writes (write to temp, rename) for safety.
    """

    def __init__(self, config_file: Path) -> None:
        """Initialize the config storage.

        Args:
            config_file: Path to the config JSON file.
        """
        self.config_file = Path(config_file).expanduser()
        self._data: ModelsConfigData | None = None

    async def ensure_dir(self) -> None:
        """Ensure the config directory exists."""
        config_dir = self.config_file.parent
        if not config_dir.exists():
            config_dir.mkdir(parents=True, exist_ok=True)

    async def load(self) -> ModelsConfigData:
        """Load or initialize config data.

        Returns:
            ModelsConfigData with loaded or empty data.
        """
        if self._data is not None:
            return self._data

        if not self.config_file.exists():
            logger.debug("Config file does not exist, creating empty config", path=str(self.config_file))
            self._data = ModelsConfigData()
            return self._data

        try:
            async with aiofiles.open(self.config_file, encoding="utf-8") as f:
                content = await f.read()
                data = json.loads(content)
                self._data = ModelsConfigData.model_validate(data)
                logger.info(
                    "Loaded models config",
                    provider_count=len(self._data.providers),
                    recent_count=len(self._data.recent_models),
                )
                return self._data
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning("Failed to load config, creating empty config", error=str(e))
            self._data = ModelsConfigData()
            return self._data

    async def save(self) -> None:
        """Save config data to file.

        Uses atomic writes by writing to a temp file first, then renaming.
        """
        if self._data is None:
            return

        await self.ensure_dir()

        temp_path = self.config_file.with_suffix(".json.tmp")

        # Write to temp file with human-readable formatting
        content = json.dumps(self._data.model_dump(mode="json"), indent=2, default=str)

        try:
            async with aiofiles.open(temp_path, "w", encoding="utf-8") as f:
                await f.write(content)

            # Atomic rename
            await aiofiles.os.rename(temp_path, self.config_file)

            logger.debug(
                "Saved models config",
                path=str(self.config_file),
                provider_count=len(self._data.providers),
            )
        except Exception as e:
            # Clean up temp file on error
            if temp_path.exists():
                await aiofiles.os.remove(temp_path)
            msg = f"Failed to save config: {e}"
            raise ModelsConfigError(msg) from e

    def get_provider(self, provider_id: str) -> ProviderConfig | None:
        """Get a provider's configuration.

        Args:
            provider_id: Provider identifier (e.g., "openai", "anthropic")

        Returns:
            ProviderConfig if exists, None otherwise.
        """
        if self._data is None:
            return None
        return self._data.providers.get(provider_id)

    def ensure_provider(self, provider_id: str) -> ProviderConfig:
        """Ensure a provider exists in config, creating if needed.

        Args:
            provider_id: Provider identifier.

        Returns:
            The provider's configuration.
        """
        if self._data is None:
            self._data = ModelsConfigData()

        if provider_id not in self._data.providers:
            self._data.providers[provider_id] = ProviderConfig()

        return self._data.providers[provider_id]

    async def add_model(
        self,
        provider_id: str,
        model_id: str,
        display_name: str | None = None,
        category: str = "chat",
        source: str = "manual",
        capabilities: ModelCapabilities | None = None,
    ) -> ModelConfig:
        """Add a model to a provider.

        Args:
            provider_id: Provider identifier.
            model_id: Model identifier.
            display_name: Optional display name (defaults to model_id).
            category: Model category (chat, embedding, other).
            source: How model was added (api, manual, legacy).
            capabilities: Model capabilities.

        Returns:
            The created ModelConfig.
        """
        await self.load()
        provider = self.ensure_provider(provider_id)

        model = ModelConfig(
            id=model_id,
            display_name=display_name or model_id,
            category=category,  # type: ignore
            source=source,  # type: ignore
            capabilities=capabilities or ModelCapabilities(),
        )

        provider.models[model_id] = model
        await self.save()

        logger.info("Added model", provider=provider_id, model=model_id)
        return model

    async def remove_model(self, provider_id: str, model_id: str) -> bool:
        """Remove a model from a provider.

        Args:
            provider_id: Provider identifier.
            model_id: Model identifier.

        Returns:
            True if model was removed, False if not found.
        """
        await self.load()
        provider = self.get_provider(provider_id)

        if provider is None or model_id not in provider.models:
            return False

        del provider.models[model_id]
        await self.save()

        logger.info("Removed model", provider=provider_id, model=model_id)
        return True

    async def update_model(
        self,
        provider_id: str,
        model_id: str,
        *,
        favorited: bool | None = None,
        display_name: str | None = None,
        capabilities: ModelCapabilities | None = None,
    ) -> ModelConfig | None:
        """Update a model's configuration.

        Args:
            provider_id: Provider identifier.
            model_id: Model identifier.
            favorited: Optional new favorite status.
            display_name: Optional new display name.
            capabilities: Optional new capabilities.

        Returns:
            Updated ModelConfig if found, None otherwise.
        """
        await self.load()
        provider = self.get_provider(provider_id)

        if provider is None or model_id not in provider.models:
            return None

        model = provider.models[model_id]

        if favorited is not None:
            model.favorited = favorited
        if display_name is not None:
            model.display_name = display_name
        if capabilities is not None:
            model.capabilities = capabilities

        await self.save()

        logger.debug("Updated model", provider=provider_id, model=model_id)
        return model

    async def toggle_favorite(self, provider_id: str, model_id: str) -> bool | None:
        """Toggle a model's favorite status.

        Args:
            provider_id: Provider identifier.
            model_id: Model identifier.

        Returns:
            New favorite status, or None if model not found.
        """
        await self.load()
        provider = self.get_provider(provider_id)

        if provider is None or model_id not in provider.models:
            return None

        model = provider.models[model_id]
        model.favorited = not model.favorited
        await self.save()

        return model.favorited

    async def add_to_recent(self, provider_id: str, model_id: str) -> None:
        """Add a model to the recent list.

        Args:
            provider_id: Provider identifier.
            model_id: Model identifier.
        """
        await self.load()

        # Remove if already in list
        self._data.recent_models = [
            r for r in self._data.recent_models
            if not (r.provider == provider_id and r.model_id == model_id)
        ]

        # Add to front
        self._data.recent_models.insert(0, RecentModel(
            provider=provider_id,
            model_id=model_id,
        ))

        # Trim to limit
        limit = self._data.settings.recent_models_limit
        self._data.recent_models = self._data.recent_models[:limit]

        # Update last used model
        self._data.settings.last_used_model = LastUsedModel(
            provider=provider_id,
            model_id=model_id,
        )

        # Update model's last_used_at timestamp
        provider = self.get_provider(provider_id)
        if provider and model_id in provider.models:
            provider.models[model_id].last_used_at = datetime.utcnow()

        await self.save()

    def get_favorites(self) -> list[tuple[str, ModelConfig]]:
        """Get all favorited models.

        Returns:
            List of (provider_id, model_config) tuples.
        """
        if self._data is None:
            return []

        favorites = []
        for provider_id, provider in self._data.providers.items():
            for model in provider.models.values():
                if model.favorited:
                    favorites.append((provider_id, model))
        return favorites

    def get_recent(self) -> list[RecentModel]:
        """Get recent models list.

        Returns:
            List of RecentModel entries.
        """
        if self._data is None:
            return []
        return self._data.recent_models

    def get_default_model(self) -> LastUsedModel | None:
        """Get the default model (last used).

        Returns:
            LastUsedModel or None if not set.
        """
        if self._data is None:
            return None
        return self._data.settings.last_used_model

    async def set_provider_enabled(self, provider_id: str, enabled: bool) -> bool:
        """Enable or disable a provider.

        Args:
            provider_id: Provider identifier.
            enabled: Whether to enable.

        Returns:
            True if provider exists, False otherwise.
        """
        await self.load()
        provider = self.get_provider(provider_id)

        if provider is None:
            return False

        provider.enabled = enabled
        await self.save()

        logger.info("Updated provider", provider=provider_id, enabled=enabled)
        return True

    async def set_models_from_api(
        self,
        provider_id: str,
        models: list[ModelConfig],
    ) -> tuple[int, int, list[str]]:
        """Set models from API fetch, tracking changes.

        Preserves user settings (favorited) for existing models.

        Args:
            provider_id: Provider identifier.
            models: List of models from API.

        Returns:
            Tuple of (added_count, updated_count, deprecated_model_ids).
        """
        await self.load()
        provider = self.ensure_provider(provider_id)

        existing_ids = set(provider.models.keys())
        new_ids = {m.id for m in models}

        added = 0
        updated = 0
        deprecated = []

        # Add/update models
        for model in models:
            if model.id in existing_ids:
                # Preserve user settings
                existing = provider.models[model.id]
                model.favorited = existing.favorited
                model.added_at = existing.added_at
                model.last_used_at = existing.last_used_at
                updated += 1
            else:
                added += 1
            provider.models[model.id] = model

        # Find deprecated (in config but not in API response)
        for model_id in existing_ids:
            if model_id not in new_ids:
                # Only mark as deprecated if it came from API
                if provider.models[model_id].source == "api":
                    deprecated.append(model_id)

        provider.last_fetched_at = datetime.utcnow()
        await self.save()

        logger.info(
            "Updated models from API",
            provider=provider_id,
            added=added,
            updated=updated,
            deprecated=len(deprecated),
        )

        return added, updated, deprecated

    async def exists(self) -> bool:
        """Check if config file exists.

        Returns:
            True if config file exists.
        """
        return self.config_file.exists()

    async def migrate_from_legacy_cache(self, legacy_cache_file: Path) -> int:
        """Migrate models from legacy models_cache.json.

        Imports OpenRouter models from the old cache format as 'legacy' source.

        Args:
            legacy_cache_file: Path to the legacy cache file.

        Returns:
            Number of models imported.
        """
        if not legacy_cache_file.exists():
            logger.debug("No legacy cache file to migrate", path=str(legacy_cache_file))
            return 0

        try:
            async with aiofiles.open(legacy_cache_file, encoding="utf-8") as f:
                content = await f.read()
                legacy_data = json.loads(content)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Failed to read legacy cache", error=str(e))
            return 0

        await self.load()

        # Import models grouped by provider
        models_list = legacy_data.get("models", [])
        imported = 0

        for model_data in models_list:
            model_id = model_data.get("id", "")
            if "/" not in model_id:
                continue

            # Parse provider from OpenRouter format (e.g., "anthropic/claude-3")
            provider_id = model_id.split("/")[0]
            local_model_id = model_id  # Keep full ID for OpenRouter

            # Skip if already exists
            provider = self.get_provider(provider_id)
            if provider and local_model_id in provider.models:
                continue

            # Detect capabilities from legacy format
            capabilities = ModelCapabilities(
                tools=model_data.get("supports_tools", False),
                vision=model_data.get("supports_vision", False),
                reasoning=False,  # Not tracked in legacy format
            )

            # Add to config
            await self.add_model(
                provider_id=provider_id,
                model_id=local_model_id,
                display_name=model_data.get("name", local_model_id),
                category="chat",
                source="legacy",
                capabilities=capabilities,
            )
            imported += 1

        logger.info(
            "Migrated models from legacy cache",
            imported=imported,
            source=str(legacy_cache_file),
        )

        return imported
