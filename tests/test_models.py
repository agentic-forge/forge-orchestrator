"""Tests for Pydantic models."""

from __future__ import annotations

from forge_orchestrator.models import (
    CompleteEvent,
    Conversation,
    ConversationMetadata,
    ErrorEvent,
    Message,
    PingEvent,
    ThinkingEvent,
    TokenEvent,
    TokenUsage,
    ToolCallEvent,
    ToolResultEvent,
)
from forge_orchestrator.models.messages import get_event_type


class TestSSEEventModels:
    """Tests for SSE event models."""

    def test_token_event(self) -> None:
        """Test TokenEvent model."""
        event = TokenEvent(token="Hello", cumulative="Hello")
        assert event.token == "Hello"
        assert event.cumulative == "Hello"

    def test_thinking_event(self) -> None:
        """Test ThinkingEvent model."""
        event = ThinkingEvent(content="Thinking...", cumulative="Thinking...")
        assert event.content == "Thinking..."

    def test_tool_call_event(self) -> None:
        """Test ToolCallEvent model."""
        event = ToolCallEvent(
            id="tc_123",
            tool_name="weather__get_forecast",
            arguments={"city": "Berlin"},
            status="pending",
        )
        assert event.id == "tc_123"
        assert event.tool_name == "weather__get_forecast"
        assert event.status == "pending"

    def test_tool_result_event(self) -> None:
        """Test ToolResultEvent model."""
        event = ToolResultEvent(
            tool_call_id="tc_123",
            result={"temperature": 20},
            is_error=False,
            latency_ms=150,
        )
        assert event.tool_call_id == "tc_123"
        assert event.result == {"temperature": 20}
        assert event.latency_ms == 150

    def test_complete_event(self) -> None:
        """Test CompleteEvent model."""
        event = CompleteEvent(
            response="The weather is nice.",
            usage=TokenUsage(prompt_tokens=100, completion_tokens=50),
        )
        assert event.response == "The weather is nice."
        assert event.usage is not None
        assert event.usage.prompt_tokens == 100

    def test_error_event(self) -> None:
        """Test ErrorEvent model."""
        event = ErrorEvent(
            code="MODEL_ERROR",
            message="Invalid model",
            retryable=True,
        )
        assert event.code == "MODEL_ERROR"
        assert event.retryable is True

    def test_ping_event(self) -> None:
        """Test PingEvent model."""
        event = PingEvent(timestamp=1704067200)
        assert event.timestamp == 1704067200

    def test_get_event_type(self) -> None:
        """Test get_event_type function."""
        assert get_event_type(TokenEvent(token="", cumulative="")) == "token"
        assert get_event_type(ThinkingEvent(content="", cumulative="")) == "thinking"
        assert get_event_type(
            ToolCallEvent(id="", tool_name="", arguments={}, status="pending")
        ) == "tool_call"
        assert get_event_type(
            ToolResultEvent(tool_call_id="", result=None, is_error=False, latency_ms=0)
        ) == "tool_result"
        assert get_event_type(CompleteEvent(response="")) == "complete"
        assert get_event_type(ErrorEvent(code="", message="", retryable=False)) == "error"
        assert get_event_type(PingEvent(timestamp=0)) == "ping"


class TestConversationModels:
    """Tests for conversation models."""

    def test_message_user(self) -> None:
        """Test user message model."""
        msg = Message(
            id="msg_001",
            role="user",
            content="Hello!",
        )
        assert msg.role == "user"
        assert msg.content == "Hello!"
        assert msg.status == "complete"

    def test_message_assistant(self) -> None:
        """Test assistant message with usage."""
        msg = Message(
            id="msg_002",
            role="assistant",
            content="Hi there!",
            model="claude-sonnet-4",
            usage=TokenUsage(prompt_tokens=10, completion_tokens=5),
        )
        assert msg.role == "assistant"
        assert msg.model == "claude-sonnet-4"
        assert msg.usage is not None

    def test_message_tool_call(self) -> None:
        """Test tool call message."""
        msg = Message(
            id="msg_003",
            role="tool_call",
            content="",
            tool_name="weather__get_forecast",
            tool_arguments={"city": "Berlin"},
            tool_call_id="tc_001",
        )
        assert msg.role == "tool_call"
        assert msg.tool_name == "weather__get_forecast"

    def test_message_tool_result(self) -> None:
        """Test tool result message."""
        msg = Message(
            id="msg_004",
            role="tool_result",
            content="",
            tool_call_id="tc_001",
            tool_result={"temperature": 20},
            is_error=False,
            latency_ms=150,
        )
        assert msg.role == "tool_result"
        assert msg.tool_result == {"temperature": 20}
        assert msg.latency_ms == 150

    def test_conversation_metadata(self) -> None:
        """Test conversation metadata."""
        meta = ConversationMetadata(
            id="conv_123",
            model="claude-sonnet-4",
            system_prompt="You are helpful.",
        )
        assert meta.id == "conv_123"
        assert meta.model == "claude-sonnet-4"
        assert meta.message_count == 0
        assert meta.total_tokens == 0

    def test_conversation_add_message(self) -> None:
        """Test adding messages to conversation."""
        conv = Conversation(
            metadata=ConversationMetadata(
                id="conv_123",
                model="claude-sonnet-4",
            ),
            messages=[],
        )

        msg = Message(id="msg_001", role="user", content="Hello")
        conv.add_message(msg)

        assert len(conv.messages) == 1
        assert conv.metadata.message_count == 1

    def test_conversation_update_system_prompt(self) -> None:
        """Test updating system prompt with versioning."""
        conv = Conversation(
            metadata=ConversationMetadata(
                id="conv_123",
                model="claude-sonnet-4",
                system_prompt="Original prompt",
            ),
            messages=[],
        )

        conv.update_system_prompt("New prompt")

        assert conv.metadata.system_prompt == "New prompt"
        assert len(conv.metadata.system_prompt_history) == 1
        assert conv.metadata.system_prompt_history[0].content == "Original prompt"

    def test_conversation_truncate(self) -> None:
        """Test truncating messages."""
        conv = Conversation(
            metadata=ConversationMetadata(
                id="conv_123",
                model="claude-sonnet-4",
            ),
            messages=[
                Message(id="1", role="user", content="First"),
                Message(id="2", role="assistant", content="Response 1"),
                Message(id="3", role="user", content="Second"),
                Message(id="4", role="assistant", content="Response 2"),
            ],
        )

        conv.truncate_from(2)

        assert len(conv.messages) == 2
        assert conv.messages[-1].content == "Response 1"
