"""FastAPI server with REST + SSE endpoints."""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import Annotated, Any

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from forge_orchestrator.logging import configure_logging, get_logger
from forge_orchestrator.models.messages import get_event_type
from forge_orchestrator.orchestrator import AgentOrchestrator
from forge_orchestrator.settings import settings

logger = get_logger(__name__)


# ============================================================================
# Request/Response Models
# ============================================================================


class MessageInput(BaseModel):
    """A message in the conversation history."""

    role: str  # "user" | "assistant" | "system"
    content: str


class ChatRequest(BaseModel):
    """Request body for stateless chat."""

    user_message: str = Field(description="The new user message to send")
    messages: list[MessageInput] = Field(
        default_factory=list,
        description="Previous conversation history",
    )
    system_prompt: str | None = Field(
        default=None,
        description="System prompt for the conversation",
    )
    model: str | None = Field(
        default=None,
        description="Model to use (defaults to server default)",
    )


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

    # Initialize orchestrator
    orchestrator = AgentOrchestrator(settings)
    await orchestrator.initialize()
    logger.info("Orchestrator initialized", mock_mode=settings.mock_llm)

    # Store in app state
    app.state.orchestrator = orchestrator

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
    description="Stateless LLM agent loop for Agentic Forge - connects UI to language models and MCP tools",
    version="0.2.0",
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
# Stateless Chat Endpoint
# ============================================================================


@app.post("/chat/stream")
async def chat_stream(
    request: Request,
    body: ChatRequest,
) -> EventSourceResponse:
    """Stateless chat endpoint with SSE streaming.

    Sends a message with conversation history and streams the response.
    The client is responsible for maintaining conversation state.

    Streams:
    - token: Streaming text tokens
    - thinking: Model thinking content (if supported)
    - tool_call: Tool execution status
    - tool_result: Tool execution result
    - complete: Response complete with usage stats
    - error: Error occurred
    - ping: Heartbeat
    """
    orchestrator: AgentOrchestrator = request.app.state.orchestrator

    async def event_generator():
        """Generate SSE events from the orchestrator."""
        last_ping = time.time()

        try:
            async for event in orchestrator.run_stream(
                user_message=body.user_message,
                messages=body.messages,
                system_prompt=body.system_prompt,
                model=body.model,
            ):
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

        except Exception as e:
            from forge_orchestrator.models import ErrorEvent

            logger.exception("Error in SSE stream")
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
