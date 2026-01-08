"""Tests for conversation storage."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from forge_orchestrator.storage import ConversationStorage


class TestConversationStorage:
    """Tests for ConversationStorage."""

    async def test_create_conversation(self, temp_storage: ConversationStorage) -> None:
        """Test creating a conversation."""
        conv = await temp_storage.create(
            model="test-model",
            system_prompt="You are helpful.",
        )

        assert conv.metadata.id is not None
        assert conv.metadata.model == "test-model"
        assert conv.metadata.system_prompt == "You are helpful."
        assert len(conv.messages) == 0

    async def test_get_conversation(self, temp_storage: ConversationStorage) -> None:
        """Test getting a conversation."""
        created = await temp_storage.create(model="test-model")

        loaded = await temp_storage.get(created.metadata.id)

        assert loaded is not None
        assert loaded.metadata.id == created.metadata.id
        assert loaded.metadata.model == created.metadata.model

    async def test_get_nonexistent_conversation(
        self, temp_storage: ConversationStorage
    ) -> None:
        """Test getting a non-existent conversation returns None."""
        result = await temp_storage.get("nonexistent-id")
        assert result is None

    async def test_save_conversation(self, temp_storage: ConversationStorage) -> None:
        """Test saving a conversation."""
        from forge_orchestrator.models import Message

        conv = await temp_storage.create(model="test-model")

        # Add a message
        conv.add_message(Message(id="msg_001", role="user", content="Hello"))

        # Save
        await temp_storage.save(conv)

        # Reload and verify
        loaded = await temp_storage.get(conv.metadata.id)
        assert loaded is not None
        assert len(loaded.messages) == 1
        assert loaded.messages[0].content == "Hello"

    async def test_delete_conversation(self, temp_storage: ConversationStorage) -> None:
        """Test deleting a conversation."""
        conv = await temp_storage.create(model="test-model")

        # Delete
        result = await temp_storage.delete(conv.metadata.id)
        assert result is True

        # Verify gone
        loaded = await temp_storage.get(conv.metadata.id)
        assert loaded is None

    async def test_delete_nonexistent_conversation(
        self, temp_storage: ConversationStorage
    ) -> None:
        """Test deleting a non-existent conversation returns False."""
        result = await temp_storage.delete("nonexistent-id")
        assert result is False

    async def test_list_all_conversations(
        self, temp_storage: ConversationStorage
    ) -> None:
        """Test listing all conversation IDs."""
        # Create multiple conversations
        conv1 = await temp_storage.create(model="model-1")
        conv2 = await temp_storage.create(model="model-2")
        conv3 = await temp_storage.create(model="model-3")

        # List all
        ids = await temp_storage.list_all()

        assert len(ids) == 3
        assert conv1.metadata.id in ids
        assert conv2.metadata.id in ids
        assert conv3.metadata.id in ids

    async def test_list_metadata(self, temp_storage: ConversationStorage) -> None:
        """Test listing metadata for all conversations."""
        await temp_storage.create(model="model-1", system_prompt="Prompt 1")
        await temp_storage.create(model="model-2", system_prompt="Prompt 2")

        metadata_list = await temp_storage.list_metadata()

        assert len(metadata_list) == 2
        models = {m.model for m in metadata_list}
        assert "model-1" in models
        assert "model-2" in models

    async def test_storage_creates_directory(self, tmp_path: Path) -> None:
        """Test that storage creates the directory if it doesn't exist."""
        from forge_orchestrator.storage import ConversationStorage

        storage_dir = tmp_path / "nested" / "path" / "conversations"
        storage = ConversationStorage(storage_dir)

        # Directory shouldn't exist yet
        assert not storage_dir.exists()

        # Create a conversation (triggers directory creation)
        await storage.create(model="test-model")

        # Directory should now exist
        assert storage_dir.exists()

    async def test_atomic_save(self, temp_storage: ConversationStorage) -> None:
        """Test that save is atomic (no temp files left behind)."""
        conv = await temp_storage.create(model="test-model")

        # Save multiple times
        for i in range(5):
            conv.metadata.message_count = i
            await temp_storage.save(conv)

        # Check no temp files exist
        await temp_storage.ensure_dir()
        temp_files = list(temp_storage.storage_dir.glob("*.tmp"))
        assert len(temp_files) == 0
