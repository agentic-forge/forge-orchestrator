"""FastAPI server with REST + SSE endpoints."""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import Annotated, Any

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from forge_orchestrator.keys import KeyProvider
from forge_orchestrator.logging import configure_logging, get_logger
from forge_orchestrator.models.api import (
    AddModelRequest,
    FetchModelsRequest,
    UpdateModelRequest,
    UpdateProviderRequest,
)
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
        description="Model to use (e.g., 'anthropic/claude-sonnet-4' or 'gpt-4o')",
    )
    provider: str | None = Field(
        default=None,
        description="LLM provider (openrouter, openai, anthropic, google). If not specified, auto-detected from model.",
    )
    enable_tools: bool = Field(
        default=True,
        description="Whether to enable tool calling (default: True)",
    )
    use_toon_format: bool = Field(
        default=False,
        description="Enable TOON format for tool results (reduces tokens via local transformation)",
    )
    use_tool_rag_mode: bool | None = Field(
        default=None,
        description="Use Tool RAG mode for semantic tool search. If None, uses server default.",
    )
    extra_mcp_servers: list[dict[str, Any]] | None = Field(
        default=None,
        description="User's custom MCP servers to connect directly (for future use).",
    )


class HealthResponse(BaseModel):
    """Response body for health check."""

    status: str
    armory_available: bool


class ToolsRefreshResponse(BaseModel):
    """Response body for tools refresh."""

    status: str
    tool_count: int


class ConfigResponse(BaseModel):
    """Response body for client configuration."""

    server_providers: dict[str, bool] = Field(
        description="Which providers have keys configured server-side"
    )
    allow_byok: bool = Field(
        description="Whether BYOK (header keys) is enabled"
    )


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

    # Initialize key provider for BYOK support
    key_provider = KeyProvider(settings)

    # Store in app state
    app.state.orchestrator = orchestrator
    app.state.key_provider = key_provider

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
# Configuration Endpoint (for BYOK)
# ============================================================================


@app.get("/config", response_model=ConfigResponse)
async def get_config(request: Request) -> ConfigResponse:
    """Return client-relevant configuration.

    Used by forge-ui to know which providers are available server-side
    and whether BYOK (Bring Your Own Key) is enabled.
    """
    key_provider: KeyProvider = request.app.state.key_provider
    return ConfigResponse(
        server_providers=key_provider.get_configured_providers(),
        allow_byok=key_provider.allow_header_keys,
    )


# ============================================================================
# Tools Endpoints
# ============================================================================


@app.get("/tools")
async def list_tools(
    request: Request,
    mode: Annotated[str | None, Query(description="Tool mode: 'rag' for semantic search")] = None,
) -> list[dict[str, Any]]:
    """List available tools from Armory.

    With mode=rag, returns only the search_tools meta-tool for semantic search.
    """
    orchestrator: AgentOrchestrator = request.app.state.orchestrator
    use_rag_mode = mode == "rag"
    return await orchestrator.list_tools(use_rag_mode=use_rag_mode)


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
# New Model Management Endpoints
# ============================================================================


@app.get("/providers")
async def list_providers(request: Request) -> dict[str, Any]:
    """List available providers and their configuration status."""
    from forge_orchestrator.models import ProviderResponse, ProvidersResponse
    from forge_orchestrator.providers import provider_registry

    orchestrator: AgentOrchestrator = request.app.state.orchestrator

    # Configure registry with API keys from settings
    provider_registry.configure_from_settings(orchestrator._settings)

    providers = []
    for provider_id in provider_registry.get_provider_ids():
        provider = provider_registry.get(provider_id)
        if not provider:
            continue

        # Get model count from config
        config = await orchestrator.get_models_config()
        provider_config = config.get_provider(provider_id)
        model_count = len(provider_config.models) if provider_config else 0

        providers.append(ProviderResponse(
            id=provider_id,
            name=provider.display_name,
            configured=provider.is_configured,
            has_api=provider.has_api,
            model_count=model_count,
            enabled=provider_config.enabled if provider_config else True,
        ))

    return ProvidersResponse(providers=providers).model_dump(mode="json")


@app.get("/models/grouped")
async def list_models_grouped(request: Request) -> dict[str, Any]:
    """List all models grouped by provider and category.

    Returns models organized for the management modal UI.
    """
    from forge_orchestrator.models import GroupedModelsResponse, ModelReference

    orchestrator: AgentOrchestrator = request.app.state.orchestrator

    config = await orchestrator.get_models_config()

    # Build grouped structure
    providers_data: dict[str, dict[str, list]] = {}

    for provider_id, provider_config in config._data.providers.items():
        if not provider_config.enabled:
            continue

        categories: dict[str, list] = {"chat": [], "embedding": [], "other": []}

        for model in provider_config.models.values():
            category = model.category
            if category not in categories:
                category = "other"
            categories[category].append(model)

        # Sort each category by display name
        for cat_models in categories.values():
            cat_models.sort(key=lambda m: m.display_name.lower())

        providers_data[provider_id] = categories

    # Build favorites list
    favorites = [
        ModelReference(provider=provider_id, model_id=model.id)
        for provider_id, model in config.get_favorites()
    ]

    # Build recent list
    recent = [
        ModelReference(provider=r.provider, model_id=r.model_id)
        for r in config.get_recent()
    ]

    # Get default model
    default = config.get_default_model()
    default_ref = ModelReference(provider=default.provider, model_id=default.model_id) if default else None

    return GroupedModelsResponse(
        providers=providers_data,
        favorites=favorites,
        recent=recent,
        default_model=default_ref,
    ).model_dump(mode="json")


@app.post("/models/fetch")
async def fetch_models_from_provider(
    request: Request,
    body: FetchModelsRequest,
) -> dict[str, Any]:
    """Fetch models from a provider's API.

    Only works for providers with has_api=True (OpenAI, OpenRouter).
    """
    from forge_orchestrator.models import DeprecatedModel, FetchModelsRequest, FetchModelsResponse
    from forge_orchestrator.providers import ProviderError, provider_registry

    orchestrator: AgentOrchestrator = request.app.state.orchestrator

    # Configure registry
    provider_registry.configure_from_settings(orchestrator._settings)

    provider = provider_registry.get(body.provider)
    if not provider:
        raise HTTPException(status_code=404, detail=f"Provider not found: {body.provider}")

    if not provider.has_api:
        raise HTTPException(
            status_code=400,
            detail=f"Provider {body.provider} does not support API fetching. Use suggestions for manual addition.",
        )

    if not provider.is_configured:
        raise HTTPException(
            status_code=400,
            detail=f"Provider {body.provider} is not configured. Add API key to .env file.",
        )

    try:
        models = await provider.fetch_models()
    except ProviderError as e:
        logger.error("Failed to fetch models", provider=body.provider, error=str(e))
        raise HTTPException(
            status_code=502 if e.retryable else 400,
            detail=str(e),
        ) from e

    # Save to config
    config = await orchestrator.get_models_config()
    added, updated, deprecated_ids = await config.set_models_from_api(body.provider, models)

    deprecated = [DeprecatedModel(id=mid) for mid in deprecated_ids]

    return FetchModelsResponse(
        provider=body.provider,
        models_added=added,
        models_updated=updated,
        deprecated=deprecated,
    ).model_dump(mode="json")


@app.post("/models")
async def add_model(
    request: Request,
    body: AddModelRequest,
) -> dict[str, Any]:
    """Add a model manually."""
    from forge_orchestrator.models import AddModelResponse

    orchestrator: AgentOrchestrator = request.app.state.orchestrator

    config = await orchestrator.get_models_config()
    model = await config.add_model(
        provider_id=body.provider,
        model_id=body.model_id,
        display_name=body.display_name,
        source="manual",
        capabilities=body.capabilities,
    )

    return AddModelResponse(model=model).model_dump(mode="json")


@app.put("/models/{provider}/{model_id:path}")
async def update_model(
    request: Request,
    provider: str,
    model_id: str,
    body: UpdateModelRequest,
) -> dict[str, Any]:
    """Update a model's properties (favorite, display name, capabilities)."""
    orchestrator: AgentOrchestrator = request.app.state.orchestrator

    config = await orchestrator.get_models_config()
    model = await config.update_model(
        provider_id=provider,
        model_id=model_id,
        favorited=body.favorited,
        display_name=body.display_name,
        capabilities=body.capabilities,
    )

    if model is None:
        raise HTTPException(status_code=404, detail=f"Model not found: {provider}/{model_id}")

    return {"status": "updated", "model": model.model_dump(mode="json")}


@app.delete("/models/{provider}/{model_id:path}")
async def delete_model(
    request: Request,
    provider: str,
    model_id: str,
) -> dict[str, Any]:
    """Remove a model from configuration."""
    orchestrator: AgentOrchestrator = request.app.state.orchestrator

    config = await orchestrator.get_models_config()
    removed = await config.remove_model(provider, model_id)

    if not removed:
        raise HTTPException(status_code=404, detail=f"Model not found: {provider}/{model_id}")

    return {"status": "deleted", "provider": provider, "model_id": model_id}


@app.put("/providers/{provider_id}")
async def update_provider(
    request: Request,
    provider_id: str,
    body: UpdateProviderRequest,
) -> dict[str, Any]:
    """Enable or disable a provider."""
    orchestrator: AgentOrchestrator = request.app.state.orchestrator

    config = await orchestrator.get_models_config()
    success = await config.set_provider_enabled(provider_id, body.enabled)

    if not success:
        raise HTTPException(status_code=404, detail=f"Provider not found: {provider_id}")

    return {"status": "updated", "provider": provider_id, "enabled": body.enabled}


@app.get("/models/suggestions/{provider_id}")
async def get_suggestions(
    request: Request,
    provider_id: str,
) -> dict[str, Any]:
    """Get model suggestions for a provider (for manual addition)."""
    from forge_orchestrator.models import ModelSuggestionResponse, SuggestionsResponse
    from forge_orchestrator.providers import provider_registry

    orchestrator: AgentOrchestrator = request.app.state.orchestrator

    # Configure registry
    provider_registry.configure_from_settings(orchestrator._settings)

    provider = provider_registry.get(provider_id)
    if not provider:
        raise HTTPException(status_code=404, detail=f"Provider not found: {provider_id}")

    suggestions = provider.get_suggestions()

    return SuggestionsResponse(
        provider=provider_id,
        suggestions=[
            ModelSuggestionResponse(
                id=s.id,
                display_name=s.display_name,
                recommended=s.recommended,
            )
            for s in suggestions
        ],
    ).model_dump(mode="json")


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

    BYOK Headers (optional):
    - X-LLM-Key: API key for the LLM provider
    - X-LLM-Provider: Provider name (openrouter, openai, anthropic, google)
    - X-MCP-Keys: JSON object of MCP server keys

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
    key_provider: KeyProvider = request.app.state.key_provider

    # Get API key from headers or environment (header takes priority)
    try:
        api_key, resolved_provider = key_provider.get_llm_key(request, body.provider)
    except HTTPException:
        # Re-raise HTTP exceptions (like 401 for missing key)
        raise

    # Get MCP keys from headers (if any)
    mcp_keys = key_provider.get_mcp_keys(request)

    async def event_generator():
        """Generate SSE events from the orchestrator."""
        last_ping = time.time()

        try:
            async for event in orchestrator.run_stream(
                user_message=body.user_message,
                messages=body.messages,
                system_prompt=body.system_prompt,
                model=body.model,
                provider=resolved_provider,
                api_key=api_key,
                mcp_keys=mcp_keys,
                enable_tools=body.enable_tools,
                use_toon_format=body.use_toon_format,
                use_tool_rag_mode=body.use_tool_rag_mode,
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
