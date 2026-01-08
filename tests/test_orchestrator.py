"""Tests for the agent orchestrator."""

from __future__ import annotations

from typing import TYPE_CHECKING

from forge_orchestrator.models import (
    CompleteEvent,
    Conversation,
    ConversationMetadata,
    ErrorEvent,
    ToolCallEvent,
    ToolResultEvent,
)

if TYPE_CHECKING:
    from forge_orchestrator.orchestrator import AgentOrchestrator


class TestAgentOrchestrator:
    """Tests for AgentOrchestrator in mock mode."""

    async def test_initialize_mock_mode(
        self, initialized_orchestrator: AgentOrchestrator
    ) -> None:
        """Test initialization in mock mode."""
        assert initialized_orchestrator.settings.mock_llm is True
        # No MCP server in mock mode
        assert initialized_orchestrator._mcp_server is None

    async def test_run_stream_mock_basic(
        self, initialized_orchestrator: AgentOrchestrator
    ) -> None:
        """Test basic mock streaming."""
        conv = Conversation(
            metadata=ConversationMetadata(
                id="test_conv",
                model="test-model",
            ),
            messages=[],
        )

        events = []
        async for event in initialized_orchestrator.run_stream(conv, "Hello"):
            events.append(event)

        # Should have thinking, tokens, and complete
        event_types = [type(e).__name__ for e in events]
        assert "ThinkingEvent" in event_types
        assert "TokenEvent" in event_types
        assert "CompleteEvent" in event_types

    async def test_run_stream_mock_with_weather(
        self, initialized_orchestrator: AgentOrchestrator
    ) -> None:
        """Test mock streaming with tool call trigger."""
        conv = Conversation(
            metadata=ConversationMetadata(
                id="test_conv",
                model="test-model",
            ),
            messages=[],
        )

        events = []
        async for event in initialized_orchestrator.run_stream(
            conv, "What's the weather?"
        ):
            events.append(event)

        # Should include tool call and result
        event_types = [type(e).__name__ for e in events]
        assert "ToolCallEvent" in event_types
        assert "ToolResultEvent" in event_types
        assert "CompleteEvent" in event_types

        # Check tool call details
        tool_calls = [e for e in events if isinstance(e, ToolCallEvent)]
        assert len(tool_calls) >= 2  # pending and executing
        assert any(e.status == "pending" for e in tool_calls)
        assert any(e.status == "executing" for e in tool_calls)

        # Check tool result
        tool_results = [e for e in events if isinstance(e, ToolResultEvent)]
        assert len(tool_results) == 1
        assert tool_results[0].is_error is False

    async def test_run_stream_complete_event_has_usage(
        self, initialized_orchestrator: AgentOrchestrator
    ) -> None:
        """Test that complete event includes usage stats."""
        conv = Conversation(
            metadata=ConversationMetadata(
                id="test_conv",
                model="test-model",
            ),
            messages=[],
        )

        events = []
        async for event in initialized_orchestrator.run_stream(conv, "Hi"):
            events.append(event)

        complete_events = [e for e in events if isinstance(e, CompleteEvent)]
        assert len(complete_events) == 1
        assert complete_events[0].usage is not None
        assert complete_events[0].usage.prompt_tokens > 0

    async def test_cancel_run(
        self, initialized_orchestrator: AgentOrchestrator
    ) -> None:
        """Test cancellation."""
        conv = Conversation(
            metadata=ConversationMetadata(
                id="test_cancel",
                model="test-model",
            ),
            messages=[],
        )

        # Start a run and cancel it
        initialized_orchestrator._cancelled.add("test_cancel")

        events = []
        async for event in initialized_orchestrator.run_stream(conv, "Hello"):
            events.append(event)

        # Should get a cancelled error
        error_events = [e for e in events if isinstance(e, ErrorEvent)]
        assert len(error_events) == 1
        assert error_events[0].code == "CANCELLED"

    async def test_list_tools_empty_in_mock(
        self, initialized_orchestrator: AgentOrchestrator
    ) -> None:
        """Test that list_tools returns empty in mock mode."""
        tools = await initialized_orchestrator.list_tools()
        assert tools == []

    async def test_get_model_string(
        self, initialized_orchestrator: AgentOrchestrator
    ) -> None:
        """Test model string conversion."""
        # Already in Pydantic AI format
        assert (
            initialized_orchestrator._get_model_string("openai:gpt-4")
            == "openai:gpt-4"
        )

        # OpenRouter format
        assert (
            initialized_orchestrator._get_model_string("anthropic/claude-sonnet-4")
            == "openrouter:anthropic/claude-sonnet-4"
        )
