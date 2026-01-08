"""Agent orchestrator using Pydantic AI."""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerStreamableHTTP
from pydantic_ai.messages import ModelMessage

from forge_orchestrator.logging import get_logger
from forge_orchestrator.mcp_client import ArmoryClient
from forge_orchestrator.models import (
    CompleteEvent,
    ErrorEvent,
    PingEvent,
    SSEEvent,
    ThinkingEvent,
    TokenEvent,
    TokenUsage,
    ToolCallEvent,
    ToolResultEvent,
)

if TYPE_CHECKING:
    from forge_orchestrator.models import Conversation
    from forge_orchestrator.settings import Settings

logger = get_logger(__name__)


class OrchestratorError(Exception):
    """Base exception for orchestrator errors."""


class ModelError(OrchestratorError):
    """Error from the LLM model."""


class CancellationError(OrchestratorError):
    """Generation was cancelled."""


class AgentOrchestrator:
    """Orchestrates LLM agent interactions with streaming.

    Uses Pydantic AI for model abstraction and MCP for tool access via Armory.
    """

    def __init__(self, settings: Settings) -> None:
        """Initialize the orchestrator.

        Args:
            settings: Application settings.
        """
        self.settings = settings
        self._mcp_server: MCPServerStreamableHTTP | None = None
        self._armory_client: ArmoryClient | None = None
        self._active_runs: dict[str, asyncio.Task[None]] = {}
        self._cancelled: set[str] = set()
        self._armory_available = False

    async def initialize(self) -> None:
        """Initialize the orchestrator and connect to Armory."""
        if self.settings.mock_llm:
            logger.info("Running in mock LLM mode")
            return

        # Try to connect to Armory
        self._armory_client = ArmoryClient(
            self.settings.armory_url,
            timeout=self.settings.tool_timeout_warning,
        )

        try:
            if await self._armory_client.ping():
                self._mcp_server = MCPServerStreamableHTTP(self.settings.armory_url)
                self._armory_available = True
                logger.info("Connected to Armory", armory_url=self.settings.armory_url)
            else:
                logger.warning(
                    "Armory not available, starting without tools",
                    armory_url=self.settings.armory_url,
                )
        except Exception as e:
            logger.warning(
                "Failed to connect to Armory, starting without tools",
                armory_url=self.settings.armory_url,
                error=str(e),
            )

    async def shutdown(self) -> None:
        """Shutdown the orchestrator and cancel active runs."""
        # Cancel all active runs
        for conv_id, task in self._active_runs.items():
            task.cancel()
            logger.info("Cancelled active run", conversation_id=conv_id)

        self._active_runs.clear()
        self._cancelled.clear()

    async def refresh_tools(self) -> list[dict[str, Any]]:
        """Refresh tools from Armory.

        Returns:
            List of available tools.
        """
        if self._armory_client is None:
            return []

        try:
            if await self._armory_client.ping():
                self._mcp_server = MCPServerStreamableHTTP(self.settings.armory_url)
                self._armory_available = True
                return await self._armory_client.list_tools()
        except Exception as e:
            logger.warning("Failed to refresh tools", error=str(e))

        return []

    async def list_tools(self) -> list[dict[str, Any]]:
        """List available tools from Armory.

        Returns:
            List of available tools.
        """
        if self._armory_client is None:
            return []

        try:
            return await self._armory_client.list_tools()
        except Exception as e:
            logger.warning("Failed to list tools", error=str(e))
            return []

    def _build_message_history(
        self, conversation: Conversation
    ) -> list[ModelMessage]:
        """Build Pydantic AI message history from conversation.

        Args:
            conversation: The conversation to build history from.

        Returns:
            List of ModelMessage for Pydantic AI.
        """
        from pydantic_ai.messages import (
            ModelRequest,
            ModelResponse,
            SystemPromptPart,
            TextPart,
            UserPromptPart,
        )

        messages: list[ModelMessage] = []

        # Add system prompt as first message if present
        if conversation.metadata.system_prompt:
            messages.append(
                ModelRequest(
                    parts=[SystemPromptPart(content=conversation.metadata.system_prompt)]
                )
            )

        # Convert conversation messages
        for msg in conversation.messages:
            if msg.role == "user":
                messages.append(
                    ModelRequest(parts=[UserPromptPart(content=msg.content)])
                )
            elif msg.role == "assistant":
                messages.append(
                    ModelResponse(parts=[TextPart(content=msg.content)])
                )
            # Skip tool_call and tool_result for now - they're handled internally

        return messages

    async def run_stream(
        self,
        conversation: Conversation,
        user_message: str,
    ) -> AsyncIterator[SSEEvent]:
        """Execute agent loop and yield SSE events.

        Args:
            conversation: The conversation to continue.
            user_message: The user's message.

        Yields:
            SSE events for the response.
        """
        conv_id = conversation.metadata.id

        # Check if cancelled
        if conv_id in self._cancelled:
            self._cancelled.discard(conv_id)
            yield ErrorEvent(code="CANCELLED", message="Generation cancelled", retryable=False)
            return

        # Use mock mode if enabled
        if self.settings.mock_llm:
            async for event in self._run_mock_stream(user_message):
                yield event
            return

        # Get model from conversation or use default
        model = conversation.metadata.model or self.settings.default_model

        # Build toolsets
        toolsets = []
        if self._mcp_server and self._armory_available:
            toolsets.append(self._mcp_server)
        elif not self._armory_available:
            yield ErrorEvent(
                code="ARMORY_UNAVAILABLE",
                message="Tools unavailable - Armory connection failed",
                retryable=True,
            )

        # Create agent
        agent = Agent(
            model=self._get_model_string(model),
            toolsets=toolsets,
        )

        # Build message history
        message_history = self._build_message_history(conversation)

        # Track state for streaming
        cumulative_text = ""
        last_ping = time.time()

        try:
            async with agent:
                # Use run_stream for text streaming
                async with agent.run_stream(
                    user_message,
                    message_history=message_history if message_history else None,
                ) as result:
                    # Stream text tokens
                    async for delta in result.stream_text(delta=True):
                        # Check for cancellation
                        if conv_id in self._cancelled:
                            self._cancelled.discard(conv_id)
                            yield ErrorEvent(
                                code="CANCELLED",
                                message="Generation cancelled",
                                retryable=False,
                            )
                            return

                        cumulative_text += delta
                        yield TokenEvent(token=delta, cumulative=cumulative_text)

                        # Send ping if needed
                        now = time.time()
                        if now - last_ping >= self.settings.heartbeat_interval:
                            yield PingEvent(timestamp=int(now))
                            last_ping = now

                    # Get final result and usage
                    usage = None
                    if hasattr(result, "usage") and result.usage:
                        usage_data = result.usage()
                        if usage_data:
                            usage = TokenUsage(
                                prompt_tokens=usage_data.request_tokens or 0,
                                completion_tokens=usage_data.response_tokens or 0,
                            )

                    yield CompleteEvent(response=cumulative_text, usage=usage)

        except asyncio.CancelledError:
            yield ErrorEvent(code="CANCELLED", message="Generation cancelled", retryable=False)
        except Exception as e:
            logger.exception("Error during agent run", conversation_id=conv_id)
            yield ErrorEvent(
                code="MODEL_ERROR",
                message=str(e),
                retryable=True,
            )

    async def _run_mock_stream(self, user_message: str) -> AsyncIterator[SSEEvent]:
        """Run a mock stream for testing.

        Args:
            user_message: The user's message.

        Yields:
            Mock SSE events.
        """
        # Simulate thinking
        thinking_text = "Let me think about this..."
        yield ThinkingEvent(content=thinking_text, cumulative=thinking_text)
        await asyncio.sleep(0.1)

        # Simulate a tool call if the message mentions weather
        if "weather" in user_message.lower():
            tool_id = f"tc_{uuid.uuid4().hex[:8]}"

            yield ToolCallEvent(
                id=tool_id,
                tool_name="weather__get_current_weather",
                arguments={"city": "Berlin"},
                status="pending",
            )
            await asyncio.sleep(0.1)

            yield ToolCallEvent(
                id=tool_id,
                tool_name="weather__get_current_weather",
                arguments={"city": "Berlin"},
                status="executing",
            )
            await asyncio.sleep(0.2)

            yield ToolResultEvent(
                tool_call_id=tool_id,
                result={"temperature": 20, "condition": "sunny"},
                is_error=False,
                latency_ms=200,
            )

        # Simulate streaming response
        response = f"This is a mock response to: {user_message}"
        cumulative = ""

        for word in response.split():
            cumulative += word + " "
            yield TokenEvent(token=word + " ", cumulative=cumulative)
            await asyncio.sleep(0.05)

        yield CompleteEvent(
            response=cumulative.strip(),
            usage=TokenUsage(prompt_tokens=50, completion_tokens=len(response.split())),
        )

    async def cancel(self, conv_id: str) -> bool:
        """Cancel an active generation.

        Args:
            conv_id: The conversation ID to cancel.

        Returns:
            True if cancelled, False if no active run.
        """
        self._cancelled.add(conv_id)

        if conv_id in self._active_runs:
            self._active_runs[conv_id].cancel()
            del self._active_runs[conv_id]
            logger.info("Cancelled generation", conversation_id=conv_id)
            return True

        return False

    def _get_model_string(self, model: str) -> str:
        """Convert model name to Pydantic AI format.

        Pydantic AI uses format like 'openai:gpt-4' or 'anthropic:claude-sonnet-4-0'.
        OpenRouter models come as 'anthropic/claude-sonnet-4' and need conversion.

        Args:
            model: The model name (OpenRouter format).

        Returns:
            Model string in Pydantic AI format.
        """
        # If already in Pydantic AI format, return as-is
        if ":" in model:
            return model

        # Convert OpenRouter format to Pydantic AI format with OpenRouter provider
        # e.g., "anthropic/claude-sonnet-4" -> "openrouter:anthropic/claude-sonnet-4"
        return f"openrouter:{model}"
