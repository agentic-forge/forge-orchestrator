"""JSON file storage for conversations."""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import aiofiles
import aiofiles.os

if TYPE_CHECKING:
    from forge_orchestrator.models import Conversation, ConversationMetadata


class StorageError(Exception):
    """Base exception for storage errors."""


class ConversationNotFoundError(StorageError):
    """Conversation not found in storage."""

    def __init__(self, conv_id: str) -> None:
        self.conv_id = conv_id
        super().__init__(f"Conversation not found: {conv_id}")


class ConversationStorage:
    """JSON file storage for conversations.

    Stores conversations at {storage_dir}/{conversation_id}.json
    Uses atomic writes (write to temp, rename) for safety.
    """

    def __init__(self, storage_dir: Path) -> None:
        """Initialize storage.

        Args:
            storage_dir: Directory to store conversation JSON files.
        """
        self.storage_dir = Path(storage_dir).expanduser()

    async def ensure_dir(self) -> None:
        """Ensure storage directory exists."""
        if not self.storage_dir.exists():
            self.storage_dir.mkdir(parents=True, exist_ok=True)

    def _get_path(self, conv_id: str) -> Path:
        """Get the file path for a conversation."""
        return self.storage_dir / f"{conv_id}.json"

    async def create(
        self,
        model: str,
        system_prompt: str = "",
    ) -> Conversation:
        """Create a new conversation.

        Args:
            model: The LLM model to use for this conversation.
            system_prompt: Optional system prompt for the conversation.

        Returns:
            The newly created Conversation.
        """
        from forge_orchestrator.models import Conversation, ConversationMetadata

        await self.ensure_dir()

        conv_id = str(uuid.uuid4())
        now = datetime.utcnow()

        conversation = Conversation(
            metadata=ConversationMetadata(
                id=conv_id,
                created_at=now,
                updated_at=now,
                model=model,
                system_prompt=system_prompt,
            ),
            messages=[],
        )

        await self.save(conversation)
        return conversation

    async def get(self, conv_id: str) -> Conversation | None:
        """Get a conversation by ID.

        Args:
            conv_id: The conversation ID.

        Returns:
            The Conversation if found, None otherwise.
        """
        from forge_orchestrator.models import Conversation

        file_path = self._get_path(conv_id)

        if not file_path.exists():
            return None

        try:
            async with aiofiles.open(file_path, encoding="utf-8") as f:
                content = await f.read()
                data = json.loads(content)
                return Conversation.model_validate(data)
        except (json.JSONDecodeError, ValueError) as e:
            msg = f"Failed to load conversation {conv_id}: {e}"
            raise StorageError(msg) from e

    async def save(self, conversation: Conversation) -> None:
        """Save a conversation to storage.

        Uses atomic writes by writing to a temp file first, then renaming.

        Args:
            conversation: The Conversation to save.
        """
        await self.ensure_dir()

        file_path = self._get_path(conversation.metadata.id)
        temp_path = file_path.with_suffix(".json.tmp")

        # Update the updated_at timestamp
        conversation.metadata.updated_at = datetime.utcnow()

        # Write to temp file with human-readable formatting
        data = conversation.model_dump(mode="json")
        content = json.dumps(data, indent=2, default=str)

        try:
            async with aiofiles.open(temp_path, "w", encoding="utf-8") as f:
                await f.write(content)

            # Atomic rename
            await aiofiles.os.rename(temp_path, file_path)
        except Exception as e:
            # Clean up temp file on error
            if temp_path.exists():
                await aiofiles.os.remove(temp_path)
            msg = f"Failed to save conversation {conversation.metadata.id}: {e}"
            raise StorageError(msg) from e

    async def delete(self, conv_id: str) -> bool:
        """Delete a conversation.

        Args:
            conv_id: The conversation ID to delete.

        Returns:
            True if deleted, False if not found.
        """
        file_path = self._get_path(conv_id)

        if not file_path.exists():
            return False

        try:
            await aiofiles.os.remove(file_path)
            return True
        except OSError as e:
            msg = f"Failed to delete conversation {conv_id}: {e}"
            raise StorageError(msg) from e

    async def list_all(self) -> list[str]:
        """List all conversation IDs.

        Returns:
            List of conversation IDs, sorted by modification time (newest first).
        """
        await self.ensure_dir()

        # Get all JSON files
        json_files = list(self.storage_dir.glob("*.json"))

        # Sort by modification time (newest first)
        json_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)

        # Extract conversation IDs (filename without .json)
        return [f.stem for f in json_files]

    async def list_metadata(self) -> list[ConversationMetadata]:
        """List metadata for all conversations.

        Returns:
            List of ConversationMetadata, sorted by modification time (newest first).
        """
        from forge_orchestrator.models import ConversationMetadata

        await self.ensure_dir()

        # Get all JSON files
        json_files = list(self.storage_dir.glob("*.json"))

        # Sort by modification time (newest first)
        json_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)

        metadata_list = []
        for file_path in json_files:
            try:
                async with aiofiles.open(file_path, encoding="utf-8") as f:
                    content = await f.read()
                    data = json.loads(content)
                    metadata = ConversationMetadata.model_validate(data["metadata"])
                    metadata_list.append(metadata)
            except (json.JSONDecodeError, ValueError, KeyError):
                # Skip invalid files
                continue

        return metadata_list
