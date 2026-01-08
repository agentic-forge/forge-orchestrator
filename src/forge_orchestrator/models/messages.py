"""SSE event models for streaming responses."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel

from forge_orchestrator.models.conversation import TokenUsage


class TokenEvent(BaseModel):
    """Streaming text token event."""

    token: str
    cumulative: str


class ThinkingEvent(BaseModel):
    """Model thinking/reasoning event (for models that support it)."""

    content: str
    cumulative: str


class ToolCallEvent(BaseModel):
    """Tool call event with status tracking."""

    id: str
    tool_name: str
    arguments: dict[str, Any]
    status: Literal["pending", "executing", "complete", "error"]


class ToolResultEvent(BaseModel):
    """Tool execution result event."""

    tool_call_id: str
    result: Any
    is_error: bool
    latency_ms: int


class CompleteEvent(BaseModel):
    """Response completion event."""

    response: str
    usage: TokenUsage | None = None


class ErrorEvent(BaseModel):
    """Error event."""

    code: str
    message: str
    retryable: bool


class PingEvent(BaseModel):
    """Heartbeat ping event."""

    timestamp: int


# Union type for all SSE events
SSEEvent = (
    TokenEvent
    | ThinkingEvent
    | ToolCallEvent
    | ToolResultEvent
    | CompleteEvent
    | ErrorEvent
    | PingEvent
)


def get_event_type(event: SSEEvent) -> str:
    """Get the SSE event type string for an event."""
    if isinstance(event, TokenEvent):
        return "token"
    if isinstance(event, ThinkingEvent):
        return "thinking"
    if isinstance(event, ToolCallEvent):
        return "tool_call"
    if isinstance(event, ToolResultEvent):
        return "tool_result"
    if isinstance(event, CompleteEvent):
        return "complete"
    if isinstance(event, ErrorEvent):
        return "error"
    if isinstance(event, PingEvent):
        return "ping"
    msg = f"Unknown event type: {type(event)}"
    raise ValueError(msg)
