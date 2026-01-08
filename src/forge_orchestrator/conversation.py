"""Conversation manager - connects storage and orchestrator."""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from datetime import datetime
from typing import TYPE_CHECKING

from forge_orchestrator.logging import get_logger
from forge_orchestrator.models import (
    CompleteEvent,
    ErrorEvent,
    Message,
    SSEEvent,
    TokenEvent,
    ToolCallEvent,
    ToolResultEvent,
)
from forge_orchestrator.storage import ConversationNotFoundError

if TYPE_CHECKING:
    from forge_orchestrator.models import Conversation, ConversationMetadata
    from forge_orchestrator.orchestrator import AgentOrchestrator
    from forge_orchestrator.storage import ConversationStorage

logger = get_logger(__name__)


class ConversationManager:
    """Manages conversation lifecycle and state.

    Coordinates between storage (persistence) and orchestrator (LLM execution).
    """

    def __init__(
        self,
        storage: ConversationStorage,
        orchestrator: AgentOrchestrator,
    ) -> None:
        """Initialize the manager.

        Args:
            storage: Conversation storage instance.
            orchestrator: Agent orchestrator instance.
        """
        self.storage = storage
        self.orchestrator = orchestrator

    async def create(
        self,
        model: str | None = None,
        system_prompt: str | None = None,
    ) -> Conversation:
        """Create a new conversation.

        Args:
            model: Model to use (defaults to orchestrator's default).
            system_prompt: Optional system prompt.

        Returns:
            The newly created conversation.
        """
        actual_model = model or self.orchestrator.settings.default_model
        actual_prompt = system_prompt or ""

        conversation = await self.storage.create(
            model=actual_model,
            system_prompt=actual_prompt,
        )

        logger.info(
            "Created conversation",
            conversation_id=conversation.metadata.id,
            model=actual_model,
        )

        return conversation

    async def get(self, conv_id: str) -> Conversation | None:
        """Get a conversation by ID.

        Args:
            conv_id: The conversation ID.

        Returns:
            The conversation if found, None otherwise.
        """
        return await self.storage.get(conv_id)

    async def delete(self, conv_id: str) -> bool:
        """Delete a conversation.

        Args:
            conv_id: The conversation ID to delete.

        Returns:
            True if deleted, False if not found.
        """
        result = await self.storage.delete(conv_id)
        if result:
            logger.info("Deleted conversation", conversation_id=conv_id)
        return result

    async def list_all(self) -> list[str]:
        """List all conversation IDs.

        Returns:
            List of conversation IDs.
        """
        return await self.storage.list_all()

    async def list_metadata(self) -> list[ConversationMetadata]:
        """List metadata for all conversations.

        Returns:
            List of conversation metadata.
        """
        return await self.storage.list_metadata()

    async def send_message(
        self,
        conv_id: str,
        content: str,
        model: str | None = None,
    ) -> AsyncIterator[SSEEvent]:
        """Send a user message and stream the response.

        Args:
            conv_id: The conversation ID.
            content: The message content.
            model: Optional model override for this message.

        Yields:
            SSE events for the response.
        """
        # Load conversation
        conversation = await self.storage.get(conv_id)
        if conversation is None:
            raise ConversationNotFoundError(conv_id)

        # Override model if specified
        if model:
            conversation.metadata.model = model

        # Add user message
        user_msg = Message(
            id=str(uuid.uuid4()),
            role="user",
            content=content,
            timestamp=datetime.utcnow(),
            status="complete",
        )
        conversation.add_message(user_msg)

        # Save with user message
        await self.storage.save(conversation)

        logger.info(
            "Sending message",
            conversation_id=conv_id,
            model=conversation.metadata.model,
        )

        # Track response for saving
        assistant_content: list[str] = []
        tool_calls: list[Message] = []
        tool_results: list[Message] = []
        response_complete = False
        usage = None

        try:
            # Stream response
            async for event in self.orchestrator.run_stream(conversation, content):
                # Track tokens for final message
                if isinstance(event, TokenEvent):
                    assistant_content.append(event.token)

                # Track tool calls
                elif isinstance(event, ToolCallEvent):
                    if event.status == "pending":
                        tool_calls.append(
                            Message(
                                id=str(uuid.uuid4()),
                                role="tool_call",
                                content="",
                                timestamp=datetime.utcnow(),
                                tool_name=event.tool_name,
                                tool_arguments=event.arguments,
                                tool_call_id=event.id,
                                status="complete",
                            )
                        )

                # Track tool results
                elif isinstance(event, ToolResultEvent):
                    tool_results.append(
                        Message(
                            id=str(uuid.uuid4()),
                            role="tool_result",
                            content="",
                            timestamp=datetime.utcnow(),
                            tool_call_id=event.tool_call_id,
                            tool_result=event.result,
                            is_error=event.is_error,
                            latency_ms=event.latency_ms,
                            status="complete",
                        )
                    )

                # Track completion
                elif isinstance(event, CompleteEvent):
                    response_complete = True
                    usage = event.usage

                # Track errors
                elif isinstance(event, ErrorEvent):
                    if event.code == "CANCELLED":
                        # Don't save partial response on cancellation
                        logger.info("Generation cancelled", conversation_id=conv_id)
                        yield event
                        return

                yield event

            # Save completed response
            if response_complete:
                # Add tool call/result messages
                for tool_call in tool_calls:
                    conversation.add_message(tool_call)
                for tool_result in tool_results:
                    conversation.add_message(tool_result)

                # Add assistant message
                assistant_msg = Message(
                    id=str(uuid.uuid4()),
                    role="assistant",
                    content="".join(assistant_content),
                    timestamp=datetime.utcnow(),
                    model=conversation.metadata.model,
                    usage=usage,
                    status="complete",
                )
                conversation.add_message(assistant_msg)

                await self.storage.save(conversation)

                logger.info(
                    "Message complete",
                    conversation_id=conv_id,
                    tokens=usage.prompt_tokens + usage.completion_tokens if usage else None,
                )

        except Exception as e:
            logger.exception("Error during message streaming", conversation_id=conv_id)
            yield ErrorEvent(
                code="STREAM_ERROR",
                message=str(e),
                retryable=True,
            )

    async def cancel(self, conv_id: str) -> bool:
        """Cancel an active generation.

        Args:
            conv_id: The conversation ID.

        Returns:
            True if cancelled, False if no active run.
        """
        return await self.orchestrator.cancel(conv_id)

    async def delete_messages_from(self, conv_id: str, index: int) -> Conversation:
        """Delete message at index and all following messages.

        Args:
            conv_id: The conversation ID.
            index: The message index to delete from.

        Returns:
            The updated conversation.

        Raises:
            ConversationNotFoundError: If conversation not found.
        """
        conversation = await self.storage.get(conv_id)
        if conversation is None:
            raise ConversationNotFoundError(conv_id)

        conversation.truncate_from(index)
        await self.storage.save(conversation)

        logger.info(
            "Truncated messages",
            conversation_id=conv_id,
            from_index=index,
            remaining=len(conversation.messages),
        )

        return conversation

    async def update_system_prompt(
        self,
        conv_id: str,
        content: str,
    ) -> Conversation:
        """Update the system prompt with versioning.

        Args:
            conv_id: The conversation ID.
            content: The new system prompt content.

        Returns:
            The updated conversation.

        Raises:
            ConversationNotFoundError: If conversation not found.
        """
        conversation = await self.storage.get(conv_id)
        if conversation is None:
            raise ConversationNotFoundError(conv_id)

        conversation.update_system_prompt(content)
        await self.storage.save(conversation)

        logger.info(
            "Updated system prompt",
            conversation_id=conv_id,
            version=len(conversation.metadata.system_prompt_history) + 1,
        )

        return conversation

    async def update_model(self, conv_id: str, model: str) -> Conversation:
        """Update the model for a conversation.

        Args:
            conv_id: The conversation ID.
            model: The new model name.

        Returns:
            The updated conversation.

        Raises:
            ConversationNotFoundError: If conversation not found.
        """
        conversation = await self.storage.get(conv_id)
        if conversation is None:
            raise ConversationNotFoundError(conv_id)

        conversation.metadata.model = model
        await self.storage.save(conversation)

        logger.info(
            "Updated model",
            conversation_id=conv_id,
            model=model,
        )

        return conversation
