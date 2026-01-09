"""FastAPI server with REST + SSE endpoints."""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import Annotated, Any

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from forge_orchestrator.conversation import ConversationManager
from forge_orchestrator.logging import configure_logging, get_logger
from forge_orchestrator.models.messages import get_event_type
from forge_orchestrator.orchestrator import AgentOrchestrator
from forge_orchestrator.settings import settings
from forge_orchestrator.storage import ConversationNotFoundError, ConversationStorage

logger = get_logger(__name__)


# ============================================================================
# Request/Response Models
# ============================================================================


class CreateConversationRequest(BaseModel):
    """Request body for creating a conversation."""

    system_prompt: str | None = None
    model: str | None = None


class CreateConversationResponse(BaseModel):
    """Response body for creating a conversation."""

    id: str
    model: str
    system_prompt: str


class SendMessageRequest(BaseModel):
    """Request body for sending a message."""

    content: str
    model: str | None = None  # Override model for this message


class SendMessageResponse(BaseModel):
    """Response body for send message (returns immediately)."""

    status: str = "accepted"
    stream_url: str


class UpdateSystemPromptRequest(BaseModel):
    """Request body for updating system prompt."""

    content: str


class CancelResponse(BaseModel):
    """Response body for cancel endpoint."""

    cancelled: bool


class DeleteResponse(BaseModel):
    """Response body for delete endpoints."""

    deleted: bool


class HealthResponse(BaseModel):
    """Response body for health check."""

    status: str
    armory_available: bool


class ToolsRefreshResponse(BaseModel):
    """Response body for tools refresh."""

    status: str
    tool_count: int


# ============================================================================
# Application Lifespan
# ============================================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Configure logging
    configure_logging(json_output=True)

    logger.info("Starting Forge Orchestrator...")

    # Initialize storage
    storage = ConversationStorage(settings.conversations_dir)
    await storage.ensure_dir()
    logger.info("Storage initialized", path=str(settings.conversations_dir))

    # Initialize orchestrator
    orchestrator = AgentOrchestrator(settings)
    await orchestrator.initialize()
    logger.info("Orchestrator initialized", mock_mode=settings.mock_llm)

    # Create manager
    manager = ConversationManager(storage, orchestrator)

    # Store in app state
    app.state.storage = storage
    app.state.orchestrator = orchestrator
    app.state.manager = manager

    logger.info(
        "Forge Orchestrator started",
        host=settings.host,
        port=settings.port,
        model=settings.default_model,
    )

    yield

    # Shutdown
    logger.info("Shutting down Forge Orchestrator...")
    await orchestrator.shutdown()
    logger.info("Forge Orchestrator shutdown complete")


# ============================================================================
# FastAPI Application
# ============================================================================

app = FastAPI(
    title="Forge Orchestrator",
    description="LLM agent loop for Agentic Forge - connects UI to language models and MCP tools",
    version="0.1.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Health Check
# ============================================================================


@app.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    """Health check endpoint."""
    orchestrator: AgentOrchestrator = request.app.state.orchestrator
    return HealthResponse(
        status="ok",
        armory_available=orchestrator._armory_available,
    )


# ============================================================================
# Tools Endpoints
# ============================================================================


@app.get("/tools")
async def list_tools(request: Request) -> list[dict[str, Any]]:
    """List available tools from Armory."""
    orchestrator: AgentOrchestrator = request.app.state.orchestrator
    return await orchestrator.list_tools()


@app.post("/tools/refresh", response_model=ToolsRefreshResponse)
async def refresh_tools(request: Request) -> ToolsRefreshResponse:
    """Refresh tools from Armory."""
    orchestrator: AgentOrchestrator = request.app.state.orchestrator
    tools = await orchestrator.refresh_tools()
    return ToolsRefreshResponse(status="refreshed", tool_count=len(tools))


# ============================================================================
# Models Endpoints
# ============================================================================


@app.get("/models")
async def list_models(
    request: Request,
    provider: Annotated[str | None, Query(description="Filter by provider")] = None,
    supports_tools: Annotated[
        bool | None, Query(description="Filter by tool calling support")
    ] = None,
    supports_vision: Annotated[
        bool | None, Query(description="Filter by vision support")
    ] = None,
) -> dict[str, Any]:
    """List available models from cache.

    Returns cached models with optional filtering. If cache is empty,
    returns empty list (use POST /models/refresh to populate).
    """
    from forge_orchestrator.models import ModelsResponse

    orchestrator: AgentOrchestrator = request.app.state.orchestrator

    models_data = await orchestrator.get_models()

    if models_data is None:
        return ModelsResponse(
            providers=[],
            models=[],
            cached_at=None,
        ).model_dump(mode="json")

    # Apply filters
    filtered_models = models_data.models

    if provider is not None:
        filtered_models = [m for m in filtered_models if m.provider == provider]

    if supports_tools is not None:
        filtered_models = [m for m in filtered_models if m.supports_tools == supports_tools]

    if supports_vision is not None:
        filtered_models = [m for m in filtered_models if m.supports_vision == supports_vision]

    # Get unique providers from filtered models
    providers = sorted({m.provider for m in filtered_models})

    return ModelsResponse(
        providers=providers,
        models=filtered_models,
        cached_at=models_data.fetched_at,
    ).model_dump(mode="json")


@app.post("/models/refresh")
async def refresh_models(request: Request) -> dict[str, Any]:
    """Refresh models cache from OpenRouter.

    Fetches the latest models from OpenRouter API and updates the cache.
    """
    from forge_orchestrator.models import ModelsRefreshResponse
    from forge_orchestrator.openrouter_client import OpenRouterError

    orchestrator: AgentOrchestrator = request.app.state.orchestrator

    try:
        models_data = await orchestrator.refresh_models()
        return ModelsRefreshResponse(
            status="refreshed",
            model_count=len(models_data.models),
            provider_count=len(models_data.providers),
        ).model_dump(mode="json")
    except OpenRouterError as e:
        logger.error("Failed to refresh models", error=str(e))
        raise HTTPException(
            status_code=502,
            detail=f"Failed to fetch models from OpenRouter: {e}",
        ) from e


# ============================================================================
# Conversation Endpoints
# ============================================================================


@app.post("/conversations", response_model=CreateConversationResponse)
async def create_conversation(
    request: Request,
    body: CreateConversationRequest,
) -> CreateConversationResponse:
    """Create a new conversation."""
    manager: ConversationManager = request.app.state.manager

    conversation = await manager.create(
        model=body.model,
        system_prompt=body.system_prompt,
    )

    return CreateConversationResponse(
        id=conversation.metadata.id,
        model=conversation.metadata.model,
        system_prompt=conversation.metadata.system_prompt,
    )


@app.get("/conversations/{conv_id}")
async def get_conversation(request: Request, conv_id: str) -> dict[str, Any]:
    """Get conversation state."""
    manager: ConversationManager = request.app.state.manager

    conversation = await manager.get(conv_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found")

    return conversation.model_dump(mode="json")


@app.delete("/conversations/{conv_id}", response_model=DeleteResponse)
async def delete_conversation(request: Request, conv_id: str) -> DeleteResponse:
    """Delete a conversation."""
    manager: ConversationManager = request.app.state.manager

    deleted = await manager.delete(conv_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Conversation not found")

    return DeleteResponse(deleted=True)


@app.post("/conversations/{conv_id}/messages", response_model=SendMessageResponse)
async def send_message(
    request: Request,
    conv_id: str,
    body: SendMessageRequest,
) -> SendMessageResponse:
    """Send a message (returns immediately, use SSE for response).

    The actual response is streamed via the /conversations/{id}/stream endpoint.
    """
    manager: ConversationManager = request.app.state.manager

    # Validate conversation exists
    conversation = await manager.get(conv_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found")

    return SendMessageResponse(
        status="accepted",
        stream_url=f"/conversations/{conv_id}/stream?message={body.content}",
    )


@app.get("/conversations/{conv_id}/stream")
async def stream_response(
    request: Request,
    conv_id: str,
    message: Annotated[str, Query(description="The message to send")],
    model: Annotated[str | None, Query(description="Optional model override")] = None,
) -> EventSourceResponse:
    """SSE stream for conversation events.

    Streams:
    - token: Streaming text tokens
    - thinking: Model thinking content
    - tool_call: Tool execution status
    - tool_result: Tool execution result
    - complete: Response complete
    - error: Error occurred
    - ping: Heartbeat
    """
    manager: ConversationManager = request.app.state.manager

    async def event_generator():
        """Generate SSE events from the manager."""
        last_ping = time.time()

        try:
            async for event in manager.send_message(conv_id, message, model):
                # Get event type and serialize
                event_type = get_event_type(event)
                yield {
                    "event": event_type,
                    "data": event.model_dump_json(),
                }

                # Check if we need a ping
                now = time.time()
                if now - last_ping >= settings.heartbeat_interval:
                    from forge_orchestrator.models import PingEvent

                    yield {
                        "event": "ping",
                        "data": PingEvent(timestamp=int(now)).model_dump_json(),
                    }
                    last_ping = now

        except ConversationNotFoundError:
            from forge_orchestrator.models import ErrorEvent

            yield {
                "event": "error",
                "data": ErrorEvent(
                    code="NOT_FOUND",
                    message="Conversation not found",
                    retryable=False,
                ).model_dump_json(),
            }
        except Exception as e:
            from forge_orchestrator.models import ErrorEvent

            logger.exception("Error in SSE stream", conversation_id=conv_id)
            yield {
                "event": "error",
                "data": ErrorEvent(
                    code="STREAM_ERROR",
                    message=str(e),
                    retryable=True,
                ).model_dump_json(),
            }

    return EventSourceResponse(
        event_generator(),
        headers={
            "X-Accel-Buffering": "no",  # Disable Nginx buffering
            "Cache-Control": "no-cache",
        },
    )


@app.post("/conversations/{conv_id}/cancel", response_model=CancelResponse)
async def cancel_generation(request: Request, conv_id: str) -> CancelResponse:
    """Cancel an ongoing generation."""
    manager: ConversationManager = request.app.state.manager

    # Validate conversation exists
    conversation = await manager.get(conv_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found")

    cancelled = await manager.cancel(conv_id)
    return CancelResponse(cancelled=cancelled)


@app.delete("/conversations/{conv_id}/messages/{index}", response_model=dict[str, Any])
async def delete_messages(
    request: Request,
    conv_id: str,
    index: int,
) -> dict[str, Any]:
    """Delete message at index and all following messages."""
    manager: ConversationManager = request.app.state.manager

    try:
        conversation = await manager.delete_messages_from(conv_id, index)
        return conversation.model_dump(mode="json")
    except ConversationNotFoundError:
        raise HTTPException(status_code=404, detail="Conversation not found") from None


@app.patch("/conversations/{conv_id}/system-prompt", response_model=dict[str, Any])
async def update_system_prompt(
    request: Request,
    conv_id: str,
    body: UpdateSystemPromptRequest,
) -> dict[str, Any]:
    """Update the system prompt for a conversation."""
    manager: ConversationManager = request.app.state.manager

    try:
        conversation = await manager.update_system_prompt(conv_id, body.content)
        return conversation.model_dump(mode="json")
    except ConversationNotFoundError:
        raise HTTPException(status_code=404, detail="Conversation not found") from None
