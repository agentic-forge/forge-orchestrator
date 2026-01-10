"""Agent orchestrator using Pydantic AI."""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerStreamableHTTP
from pydantic_ai.messages import (
    ModelMessage,
    ToolCallPart,
    ToolReturnPart,
)

from forge_orchestrator.logging import get_logger
from forge_orchestrator.mcp_client import ArmoryClient
from forge_orchestrator.models import (
    CompleteEvent,
    ErrorEvent,
    ModelsData,
    PingEvent,
    SSEEvent,
    ThinkingEvent,
    TokenEvent,
    TokenUsage,
    ToolCallEvent,
    ToolResultEvent,
)
from forge_orchestrator.models_cache import ModelsCache
from forge_orchestrator.openrouter_client import OpenRouterClient

if TYPE_CHECKING:
    from forge_orchestrator.settings import Settings

logger = get_logger(__name__)


class OrchestratorError(Exception):
    """Base exception for orchestrator errors."""


class ModelError(OrchestratorError):
    """Error from the LLM model."""


class MessageInput:
    """Simple message input for building history."""

    def __init__(self, role: str, content: str) -> None:
        self.role = role
        self.content = content


class AgentOrchestrator:
    """Orchestrates LLM agent interactions with streaming.

    Stateless orchestrator - each request provides its own message history.
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
        self._armory_available = False

        # Models cache and OpenRouter client
        self._models_cache = ModelsCache(settings.models_cache_file)
        self._openrouter_client = OpenRouterClient(
            base_url=settings.openrouter_base_url,
            provider_whitelist=settings.provider_whitelist,
            models_per_provider=settings.models_per_provider,
            model_include_list=settings.model_include_list,
        )

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
        """Shutdown the orchestrator."""
        logger.info("Orchestrator shutdown")

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

    async def get_models(self) -> ModelsData | None:
        """Get cached models data.

        Returns:
            ModelsData if cache exists, None otherwise.
        """
        return await self._models_cache.load()

    async def refresh_models(self) -> ModelsData:
        """Fetch models from OpenRouter and update cache.

        Returns:
            The refreshed ModelsData.
        """
        logger.info("Refreshing models from OpenRouter")

        # Fetch from OpenRouter
        models_data = await self._openrouter_client.fetch_models()

        # Save to cache
        await self._models_cache.save(models_data)

        logger.info(
            "Models cache refreshed",
            model_count=len(models_data.models),
            provider_count=len(models_data.providers),
        )

        return models_data

    def _build_message_history(
        self,
        messages: list[Any],
        system_prompt: str | None = None,
    ) -> list[ModelMessage]:
        """Build Pydantic AI message history from message list.

        Args:
            messages: List of messages with role and content.
            system_prompt: Optional system prompt.

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

        result: list[ModelMessage] = []

        # Add system prompt as first message if present
        if system_prompt:
            result.append(
                ModelRequest(parts=[SystemPromptPart(content=system_prompt)])
            )

        # Convert messages
        for msg in messages:
            # Handle both dict and object with attributes
            role = msg.role if hasattr(msg, "role") else msg.get("role")
            content = msg.content if hasattr(msg, "content") else msg.get("content", "")

            if role == "user":
                result.append(
                    ModelRequest(parts=[UserPromptPart(content=content)])
                )
            elif role == "assistant":
                result.append(
                    ModelResponse(parts=[TextPart(content=content)])
                )
            # Skip tool_call and tool_result - they're handled internally by Pydantic AI

        return result

    async def run_stream(
        self,
        user_message: str,
        messages: list[Any] | None = None,
        system_prompt: str | None = None,
        model: str | None = None,
    ) -> AsyncIterator[SSEEvent]:
        """Execute agent loop and yield SSE events.

        Stateless - all context is provided in the request.

        Note: When tools are available, uses non-streaming mode due to a pydantic-ai
        bug where streaming methods don't properly execute MCP tools. The response
        is chunked into token events to simulate streaming feel.

        Args:
            user_message: The user's message.
            messages: Previous conversation history.
            system_prompt: Optional system prompt.
            model: Optional model override.

        Yields:
            SSE events for the response.
        """
        # Use mock mode if enabled
        if self.settings.mock_llm:
            async for event in self._run_mock_stream(user_message):
                yield event
            return

        # Get model or use default
        actual_model = model or self.settings.default_model

        # Build toolsets
        toolsets = []
        if self._mcp_server and self._armory_available:
            toolsets.append(self._mcp_server)
            logger.info("MCP toolset added to agent")
        elif not self._armory_available:
            yield ErrorEvent(
                code="ARMORY_UNAVAILABLE",
                message="Tools unavailable - Armory connection failed",
                retryable=True,
            )

        # Create agent
        model_string = self._get_model_string(actual_model)
        logger.info("Creating agent", model=model_string, has_toolsets=bool(toolsets))
        agent = Agent(
            model=model_string,
            toolsets=toolsets,
        )

        # Build message history
        message_history = self._build_message_history(
            messages or [],
            system_prompt,
        )

        # Use non-streaming when tools are available (pydantic-ai streaming + MCP bug)
        if toolsets:
            logger.info("Using non-streaming mode (tools available)")
            async for event in self._run_non_streaming(
                agent, user_message, message_history
            ):
                yield event
            return

        # Use streaming mode when no tools (pure text generation)
        async for event in self._run_streaming(
            agent, user_message, message_history
        ):
            yield event

    async def _run_streaming(
        self,
        agent: Agent,
        user_message: str,
        message_history: list[ModelMessage],
    ) -> AsyncIterator[SSEEvent]:
        """Run agent in streaming mode (no tools).

        Args:
            agent: The Pydantic AI agent.
            user_message: The user's message.
            message_history: Conversation history.

        Yields:
            SSE events for the response.
        """
        cumulative_text = ""
        last_ping = time.time()

        try:
            async with agent:
                logger.info("Agent context entered, starting streaming run")

                async with agent.run_stream(
                    user_message,
                    message_history=message_history if message_history else None,
                ) as result:
                    logger.info("run_stream started")

                    # Stream text tokens
                    async for delta in result.stream_text(delta=True):
                        if delta:
                            cumulative_text += delta
                            yield TokenEvent(token=delta, cumulative=cumulative_text)

                        # Send ping if needed
                        now = time.time()
                        if now - last_ping >= self.settings.heartbeat_interval:
                            yield PingEvent(timestamp=int(now))
                            last_ping = now

                    # Extract thinking from message history
                    cumulative_thinking = ""
                    for msg in result.new_messages():
                        thinking = getattr(msg, 'thinking', None)
                        if thinking:
                            cumulative_thinking = str(thinking)
                            yield ThinkingEvent(content=cumulative_thinking, cumulative=cumulative_thinking)
                            break

                    logger.info(
                        "Streaming complete",
                        cumulative_text_len=len(cumulative_text),
                        cumulative_thinking_len=len(cumulative_thinking),
                    )

                    # Get usage stats
                    usage = None
                    if hasattr(result, "usage") and result.usage:
                        usage_data = result.usage()
                        if usage_data:
                            usage = TokenUsage(
                                prompt_tokens=usage_data.input_tokens or 0,
                                completion_tokens=usage_data.output_tokens or 0,
                            )

                    yield CompleteEvent(response=cumulative_text, usage=usage)

        except asyncio.CancelledError:
            yield ErrorEvent(code="CANCELLED", message="Generation cancelled", retryable=False)
        except Exception as e:
            logger.exception("Error during streaming run")
            yield ErrorEvent(
                code="MODEL_ERROR",
                message=str(e),
                retryable=True,
            )

    async def _run_non_streaming(
        self,
        agent: Agent,
        user_message: str,
        message_history: list[ModelMessage],
    ) -> AsyncIterator[SSEEvent]:
        """Run agent in non-streaming mode (with tools).

        This is a workaround for pydantic-ai bug where streaming doesn't
        properly execute MCP tools. Non-streaming mode works correctly.

        Args:
            agent: The Pydantic AI agent.
            user_message: The user's message.
            message_history: Conversation history.

        Yields:
            SSE events for the response.
        """
        try:
            async with agent:
                logger.info("Agent context entered, starting non-streaming run")

                # Run synchronously - this properly executes tool calls
                result = await agent.run(
                    user_message,
                    message_history=message_history if message_history else None,
                )

                logger.info("Non-streaming run complete")

                # Extract thinking from new messages
                cumulative_thinking = ""
                for msg in result.new_messages():
                    thinking = getattr(msg, 'thinking', None)
                    if thinking:
                        cumulative_thinking = str(thinking)
                        yield ThinkingEvent(content=cumulative_thinking, cumulative=cumulative_thinking)
                        break

                # Extract tool calls and results from message history
                for msg in result.new_messages():
                    if hasattr(msg, 'parts'):
                        for part in msg.parts:
                            if isinstance(part, ToolCallPart):
                                logger.info(
                                    "Tool call executed",
                                    tool_name=part.tool_name,
                                    tool_call_id=part.tool_call_id,
                                )
                                try:
                                    args = json.loads(part.args) if isinstance(part.args, str) else part.args
                                except Exception:
                                    args = {"raw": part.args}

                                yield ToolCallEvent(
                                    id=part.tool_call_id or f"tc_{uuid.uuid4().hex[:8]}",
                                    tool_name=part.tool_name,
                                    arguments=args if isinstance(args, dict) else {},
                                    status="complete",
                                )

                            elif isinstance(part, ToolReturnPart):
                                logger.info(
                                    "Tool result received",
                                    tool_name=part.tool_name,
                                    tool_call_id=part.tool_call_id,
                                    content_preview=str(part.content)[:100],
                                )
                                try:
                                    result_content = json.loads(part.content) if isinstance(part.content, str) else part.content
                                except Exception:
                                    result_content = part.content

                                # Check for error - look for is_error attribute on the content object
                                is_error = bool(getattr(part.content, 'is_error', False))

                                yield ToolResultEvent(
                                    tool_call_id=part.tool_call_id or "unknown",
                                    result=result_content,
                                    is_error=is_error,
                                    latency_ms=0,
                                )

                # Get the final response text
                response_text = str(result.output) if result.output else ""

                # Chunk response into tokens to simulate streaming feel
                cumulative_text = ""
                words = response_text.split()
                for i, word in enumerate(words):
                    # Add space before word (except first)
                    token = f" {word}" if i > 0 else word
                    cumulative_text += token
                    yield TokenEvent(token=token, cumulative=cumulative_text)
                    # Small delay to simulate streaming
                    await asyncio.sleep(0.01)

                logger.info(
                    "Response chunking complete",
                    response_len=len(response_text),
                    thinking_len=len(cumulative_thinking),
                )

                # Get usage stats
                usage = None
                if hasattr(result, "usage") and result.usage:
                    usage_data = result.usage()
                    if usage_data:
                        usage = TokenUsage(
                            prompt_tokens=usage_data.input_tokens or 0,
                            completion_tokens=usage_data.output_tokens or 0,
                        )

                yield CompleteEvent(response=response_text, usage=usage)

        except asyncio.CancelledError:
            yield ErrorEvent(code="CANCELLED", message="Generation cancelled", retryable=False)
        except Exception as e:
            logger.exception("Error during non-streaming run")
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

    def _get_model_string(self, model: str) -> str:
        """Convert model name to Pydantic AI format.

        Pydantic AI uses format like 'openai:gpt-4' or 'anthropic:claude-sonnet-4-0'.
        OpenRouter models come as 'anthropic/claude-sonnet-4' and need conversion.
        Some OpenRouter models have suffixes like ':thinking' or ':free'.

        Args:
            model: The model name (OpenRouter format).

        Returns:
            Model string in Pydantic AI format.
        """
        # Known pydantic-ai provider prefixes
        pydantic_providers = ["openrouter:", "openai:", "anthropic:", "google:", "groq:", "mistral:"]

        # If already in Pydantic AI format (starts with known provider), return as-is
        if any(model.startswith(p) for p in pydantic_providers):
            return model

        # Convert OpenRouter format to Pydantic AI format with OpenRouter provider
        # e.g., "anthropic/claude-sonnet-4" -> "openrouter:anthropic/claude-sonnet-4"
        # e.g., "anthropic/claude-3.7-sonnet:thinking" -> "openrouter:anthropic/claude-3.7-sonnet:thinking"
        return f"openrouter:{model}"
