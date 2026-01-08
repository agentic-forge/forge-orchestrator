"""Tests for the FastAPI server."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from fastapi.testclient import TestClient

if TYPE_CHECKING:
    from forge_orchestrator.orchestrator import AgentOrchestrator
    from forge_orchestrator.storage import ConversationStorage


@pytest.fixture
def test_app(
    temp_storage: ConversationStorage,
    initialized_orchestrator: AgentOrchestrator,
) -> TestClient:
    """Create a test client with mock dependencies."""
    from forge_orchestrator.conversation import ConversationManager
    from forge_orchestrator.server import app

    # Create manager
    manager = ConversationManager(temp_storage, initialized_orchestrator)

    # Override app state
    app.state.storage = temp_storage
    app.state.orchestrator = initialized_orchestrator
    app.state.manager = manager

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


class TestConversationEndpoints:
    """Tests for conversation CRUD endpoints."""

    def test_create_conversation(self, test_app: TestClient) -> None:
        """Test creating a conversation."""
        response = test_app.post(
            "/conversations",
            json={
                "model": "test-model",
                "system_prompt": "You are helpful.",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "id" in data
        assert data["model"] == "test-model"
        assert data["system_prompt"] == "You are helpful."

    def test_create_conversation_defaults(self, test_app: TestClient) -> None:
        """Test creating a conversation with defaults."""
        response = test_app.post("/conversations", json={})
        assert response.status_code == 200
        data = response.json()
        assert "id" in data
        assert data["model"] == "test-model"  # From mock settings

    def test_get_conversation(self, test_app: TestClient) -> None:
        """Test getting a conversation."""
        # Create first
        create_response = test_app.post("/conversations", json={})
        conv_id = create_response.json()["id"]

        # Get
        response = test_app.get(f"/conversations/{conv_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["metadata"]["id"] == conv_id

    def test_get_nonexistent_conversation(self, test_app: TestClient) -> None:
        """Test getting a non-existent conversation returns 404."""
        response = test_app.get("/conversations/nonexistent-id")
        assert response.status_code == 404

    def test_delete_conversation(self, test_app: TestClient) -> None:
        """Test deleting a conversation."""
        # Create first
        create_response = test_app.post("/conversations", json={})
        conv_id = create_response.json()["id"]

        # Delete
        response = test_app.delete(f"/conversations/{conv_id}")
        assert response.status_code == 200
        assert response.json()["deleted"] is True

        # Verify gone
        get_response = test_app.get(f"/conversations/{conv_id}")
        assert get_response.status_code == 404

    def test_delete_nonexistent_conversation(self, test_app: TestClient) -> None:
        """Test deleting a non-existent conversation returns 404."""
        response = test_app.delete("/conversations/nonexistent-id")
        assert response.status_code == 404


class TestMessageEndpoints:
    """Tests for message-related endpoints."""

    def test_send_message_returns_stream_url(self, test_app: TestClient) -> None:
        """Test sending a message returns stream URL."""
        # Create conversation
        create_response = test_app.post("/conversations", json={})
        conv_id = create_response.json()["id"]

        # Send message
        response = test_app.post(
            f"/conversations/{conv_id}/messages",
            json={"content": "Hello!"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "accepted"
        assert "stream_url" in data
        assert conv_id in data["stream_url"]

    def test_send_message_nonexistent_conversation(
        self, test_app: TestClient
    ) -> None:
        """Test sending a message to non-existent conversation returns 404."""
        response = test_app.post(
            "/conversations/nonexistent-id/messages",
            json={"content": "Hello!"},
        )
        assert response.status_code == 404

    def test_cancel_generation(self, test_app: TestClient) -> None:
        """Test cancelling a generation."""
        # Create conversation
        create_response = test_app.post("/conversations", json={})
        conv_id = create_response.json()["id"]

        # Cancel (no active run, so cancelled=False)
        response = test_app.post(f"/conversations/{conv_id}/cancel")
        assert response.status_code == 200
        # Note: cancelled=False because there's no active run
        assert "cancelled" in response.json()

    def test_delete_messages_from(self, test_app: TestClient) -> None:
        """Test deleting messages from an index."""
        # Create conversation
        create_response = test_app.post("/conversations", json={})
        conv_id = create_response.json()["id"]

        # Delete from index 0 (no messages, so no change)
        response = test_app.delete(f"/conversations/{conv_id}/messages/0")
        assert response.status_code == 200
        data = response.json()
        assert len(data["messages"]) == 0


class TestSystemPromptEndpoint:
    """Tests for system prompt endpoint."""

    def test_update_system_prompt(self, test_app: TestClient) -> None:
        """Test updating system prompt."""
        # Create conversation
        create_response = test_app.post(
            "/conversations",
            json={"system_prompt": "Original prompt"},
        )
        conv_id = create_response.json()["id"]

        # Update system prompt
        response = test_app.patch(
            f"/conversations/{conv_id}/system-prompt",
            json={"content": "New prompt"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["metadata"]["system_prompt"] == "New prompt"
        assert len(data["metadata"]["system_prompt_history"]) == 1
        assert (
            data["metadata"]["system_prompt_history"][0]["content"]
            == "Original prompt"
        )

    def test_update_system_prompt_nonexistent(self, test_app: TestClient) -> None:
        """Test updating system prompt for non-existent conversation."""
        response = test_app.patch(
            "/conversations/nonexistent-id/system-prompt",
            json={"content": "New prompt"},
        )
        assert response.status_code == 404
