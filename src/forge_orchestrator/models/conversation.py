"""Conversation models for JSON persistence."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class TokenUsage(BaseModel):
    """Token usage statistics for a message."""

    prompt_tokens: int
    completion_tokens: int


class Message(BaseModel):
    """A message in a conversation.

    Messages can be of different roles:
    - user: User input
    - assistant: LLM response
    - system: System message (not typically stored)
    - tool_call: A tool invocation request
    - tool_result: The result of a tool invocation
    """

    id: str
    role: Literal["user", "assistant", "system", "tool_call", "tool_result"]
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    # For assistant messages
    model: str | None = None
    usage: TokenUsage | None = None

    # For tool_call messages
    tool_name: str | None = None
    tool_arguments: dict[str, Any] | None = None
    tool_call_id: str | None = None

    # For tool_result messages
    tool_result: Any | None = None
    is_error: bool = False
    latency_ms: int | None = None

    # Status
    status: Literal["complete", "cancelled", "error"] = "complete"


class SystemPromptHistory(BaseModel):
    """Historical system prompt entry for versioning."""

    content: str
    set_at: datetime


class ConversationMetadata(BaseModel):
    """Metadata for a conversation."""

    id: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    model: str
    system_prompt: str = ""
    system_prompt_history: list[SystemPromptHistory] = Field(default_factory=list)
    total_tokens: int = 0
    message_count: int = 0


class Conversation(BaseModel):
    """Full conversation format for JSON files.

    This is the complete state of a conversation that gets persisted
    to ~/.forge/conversations/{id}.json
    """

    metadata: ConversationMetadata
    messages: list[Message] = Field(default_factory=list)

    def add_message(self, message: Message) -> None:
        """Add a message to the conversation and update metadata."""
        self.messages.append(message)
        self.metadata.message_count = len(self.messages)
        self.metadata.updated_at = datetime.utcnow()

        # Update token count if usage is available
        if message.usage:
            self.metadata.total_tokens += (
                message.usage.prompt_tokens + message.usage.completion_tokens
            )

    def update_system_prompt(self, new_prompt: str) -> None:
        """Update the system prompt with versioning."""
        if self.metadata.system_prompt:
            # Save current prompt to history
            self.metadata.system_prompt_history.append(
                SystemPromptHistory(
                    content=self.metadata.system_prompt,
                    set_at=self.metadata.updated_at,
                )
            )
        self.metadata.system_prompt = new_prompt
        self.metadata.updated_at = datetime.utcnow()

    def truncate_from(self, index: int) -> None:
        """Delete message at index and all following messages."""
        if 0 <= index < len(self.messages):
            self.messages = self.messages[:index]
            self.metadata.message_count = len(self.messages)
            self.metadata.updated_at = datetime.utcnow()
