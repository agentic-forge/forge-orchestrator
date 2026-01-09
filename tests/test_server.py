"""Tests for the FastAPI server."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from fastapi.testclient import TestClient

if TYPE_CHECKING:
    from forge_orchestrator.orchestrator import AgentOrchestrator


@pytest.fixture
def test_app(initialized_orchestrator: AgentOrchestrator) -> TestClient:
    """Create a test client with mock dependencies."""
    from forge_orchestrator.server import app

    # Override app state
    app.state.orchestrator = initialized_orchestrator

    return TestClient(app, raise_server_exceptions=False)


class TestHealthEndpoint:
    """Tests for the health endpoint."""

    def test_health_check(self, test_app: TestClient) -> None:
        """Test health check returns 200."""
        response = test_app.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "armory_available" in data


class TestToolsEndpoints:
    """Tests for the tools endpoints."""

    def test_list_tools(self, test_app: TestClient) -> None:
        """Test listing tools (empty in mock mode)."""
        response = test_app.get("/tools")
        assert response.status_code == 200
        assert response.json() == []

    def test_refresh_tools(self, test_app: TestClient) -> None:
        """Test refreshing tools."""
        response = test_app.post("/tools/refresh")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "refreshed"
        assert data["tool_count"] == 0


class TestModelsEndpoints:
    """Tests for the models endpoints."""

    def test_list_models_empty_cache(self, test_app: TestClient) -> None:
        """Test listing models with empty cache."""
        response = test_app.get("/models")
        assert response.status_code == 200
        data = response.json()
        assert "models" in data
        assert "providers" in data


class TestChatEndpoint:
    """Tests for stateless chat endpoint."""

    def test_chat_stream_basic(self, test_app: TestClient) -> None:
        """Test basic chat streaming returns SSE response."""
        response = test_app.post(
            "/chat/stream",
            json={
                "user_message": "Hello",
                "messages": [],
                "system_prompt": None,
                "model": "test-model",
            },
        )
        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]

    def test_chat_stream_with_history(self, test_app: TestClient) -> None:
        """Test chat streaming with message history."""
        response = test_app.post(
            "/chat/stream",
            json={
                "user_message": "How are you?",
                "messages": [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi there!"},
                ],
                "system_prompt": "You are a helpful assistant.",
                "model": "test-model",
            },
        )
        assert response.status_code == 200

    def test_chat_stream_minimal_request(self, test_app: TestClient) -> None:
        """Test chat streaming with minimal request (only user_message)."""
        response = test_app.post(
            "/chat/stream",
            json={"user_message": "Hello"},
        )
        assert response.status_code == 200

    def test_chat_stream_contains_events(self, test_app: TestClient) -> None:
        """Test that chat stream contains expected SSE events."""
        response = test_app.post(
            "/chat/stream",
            json={"user_message": "Hello"},
        )
        assert response.status_code == 200

        # Check response contains SSE event format
        content = response.text
        assert "event:" in content
        assert "data:" in content
        # Should contain thinking, token, and complete events
        assert "thinking" in content or "token" in content
        assert "complete" in content
