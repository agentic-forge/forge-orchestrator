"""Shared test fixtures for forge-orchestrator."""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from forge_orchestrator.conversation import ConversationManager
    from forge_orchestrator.orchestrator import AgentOrchestrator
    from forge_orchestrator.settings import Settings
    from forge_orchestrator.storage import ConversationStorage


@pytest.fixture
def mock_settings(tmp_path: Path) -> Settings:
    """Create settings configured for testing."""
    from forge_orchestrator.settings import Settings

    class TestSettings(Settings):
        """Test settings with overrides."""

        model_config = {"env_prefix": "TEST_ORCHESTRATOR_"}

    return TestSettings(
        armory_url="http://localhost:8080/mcp",
        default_model="test-model",
        host="127.0.0.1",
        port=8001,
        conversations_dir=tmp_path / "conversations",
        mock_llm=True,
        show_thinking=True,
        heartbeat_interval=15,
        tool_timeout_warning=30,
    )


@pytest.fixture
def temp_storage(tmp_path: Path) -> ConversationStorage:
    """Create a temporary storage directory."""
    from forge_orchestrator.storage import ConversationStorage

    storage_dir = tmp_path / "conversations"
    return ConversationStorage(storage_dir)


@pytest.fixture
def mock_orchestrator(mock_settings: Settings) -> AgentOrchestrator:
    """Create an orchestrator in mock mode."""
    from forge_orchestrator.orchestrator import AgentOrchestrator

    return AgentOrchestrator(mock_settings)


@pytest.fixture
async def initialized_orchestrator(
    mock_orchestrator: AgentOrchestrator,
) -> AsyncIterator[AgentOrchestrator]:
    """Create and initialize an orchestrator."""
    await mock_orchestrator.initialize()
    yield mock_orchestrator
    await mock_orchestrator.shutdown()


@pytest.fixture
async def conversation_manager(
    temp_storage: ConversationStorage,
    initialized_orchestrator: AgentOrchestrator,
) -> ConversationManager:
    """Create a conversation manager with mock orchestrator."""
    from forge_orchestrator.conversation import ConversationManager

    return ConversationManager(temp_storage, initialized_orchestrator)
