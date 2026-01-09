"""JSON file cache for OpenRouter models."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import aiofiles
import aiofiles.os

from forge_orchestrator.logging import get_logger
from forge_orchestrator.models.openrouter import ModelsData

logger = get_logger(__name__)


class CacheError(Exception):
    """Error with cache operations."""


class ModelsCache:
    """JSON file cache for OpenRouter models list.

    Stores models data at the specified cache file path.
    Uses atomic writes (write to temp, rename) for safety.
    """

    def __init__(self, cache_file: Path) -> None:
        """Initialize the cache.

        Args:
            cache_file: Path to the cache JSON file.
        """
        self.cache_file = Path(cache_file).expanduser()

    async def ensure_dir(self) -> None:
        """Ensure the cache directory exists."""
        cache_dir = self.cache_file.parent
        if not cache_dir.exists():
            cache_dir.mkdir(parents=True, exist_ok=True)

    async def load(self) -> ModelsData | None:
        """Load cached models data.

        Returns:
            ModelsData if cache exists and is valid, None otherwise.
        """
        if not self.cache_file.exists():
            logger.debug("Cache file does not exist", path=str(self.cache_file))
            return None

        try:
            async with aiofiles.open(self.cache_file, encoding="utf-8") as f:
                content = await f.read()
                data = json.loads(content)
                models_data = ModelsData.model_validate(data)
                logger.info(
                    "Loaded models from cache",
                    model_count=len(models_data.models),
                    provider_count=len(models_data.providers),
                    fetched_at=str(models_data.fetched_at),
                )
                return models_data
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning("Failed to load cache, will refresh", error=str(e))
            return None

    async def save(self, data: ModelsData) -> None:
        """Save models data to cache.

        Uses atomic writes by writing to a temp file first, then renaming.

        Args:
            data: The ModelsData to save.
        """
        await self.ensure_dir()

        temp_path = self.cache_file.with_suffix(".json.tmp")

        # Write to temp file with human-readable formatting
        content = json.dumps(data.model_dump(mode="json"), indent=2, default=str)

        try:
            async with aiofiles.open(temp_path, "w", encoding="utf-8") as f:
                await f.write(content)

            # Atomic rename
            await aiofiles.os.rename(temp_path, self.cache_file)

            logger.info(
                "Saved models to cache",
                path=str(self.cache_file),
                model_count=len(data.models),
                provider_count=len(data.providers),
            )
        except Exception as e:
            # Clean up temp file on error
            if temp_path.exists():
                await aiofiles.os.remove(temp_path)
            msg = f"Failed to save cache: {e}"
            raise CacheError(msg) from e

    async def get_age_seconds(self) -> float | None:
        """Get the age of the cache in seconds.

        Returns:
            Age in seconds if cache exists, None otherwise.
        """
        if not self.cache_file.exists():
            return None

        try:
            stat = self.cache_file.stat()
            age = datetime.utcnow().timestamp() - stat.st_mtime
            return age
        except OSError:
            return None

    async def exists(self) -> bool:
        """Check if cache file exists.

        Returns:
            True if cache file exists.
        """
        return self.cache_file.exists()
