"""Shared test fixtures for forge-orchestrator."""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from forge_orchestrator.orchestrator import AgentOrchestrator
    from forge_orchestrator.settings import Settings


@pytest.fixture
def mock_settings(tmp_path: Path) -> Settings:
    """Create settings configured for testing."""
    from forge_orchestrator.settings import Settings

    class TestSettings(Settings):
        """Test settings with overrides."""

        model_config = {"env_prefix": "TEST_ORCHESTRATOR_", "extra": "ignore"}

    return TestSettings(
        armory_url="http://localhost:8080/mcp",
        default_model="test-model",
        host="127.0.0.1",
        port=8001,
        models_cache_file=tmp_path / "models_cache.json",
        mock_llm=True,
        show_thinking=True,
        heartbeat_interval=15,
        tool_timeout_warning=30,
    )


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
