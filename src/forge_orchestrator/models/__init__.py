"""Pydantic models for Forge Orchestrator."""

from forge_orchestrator.models.conversation import (
    Conversation,
    ConversationMetadata,
    Message,
    SystemPromptHistory,
    TokenUsage,
)
from forge_orchestrator.models.messages import (
    CompleteEvent,
    ErrorEvent,
    PingEvent,
    SSEEvent,
    ThinkingEvent,
    TokenEvent,
    ToolCallEvent,
    ToolResultEvent,
)
from forge_orchestrator.models.openrouter import (
    ModelInfo,
    ModelPricing,
    ModelsData,
    ModelsRefreshResponse,
    ModelsResponse,
)
from forge_orchestrator.models.provider import (
    LastUsedModel,
    ModelCapabilities,
    ModelConfig,
    ModelsConfigData,
    ModelsConfigSettings,
    ModelSuggestion,
    ProviderConfig,
    RecentModel,
)
from forge_orchestrator.models.api import (
    AddModelRequest,
    AddModelResponse,
    DeprecatedModel,
    FetchModelsRequest,
    FetchModelsResponse,
    GroupedModelsResponse,
    ModelReference,
    ModelSuggestionResponse,
    ProviderResponse,
    ProvidersResponse,
    SuggestionsResponse,
    UpdateModelRequest,
    UpdateProviderRequest,
)

__all__ = [
    # Conversation models
    "Conversation",
    "ConversationMetadata",
    "Message",
    "SystemPromptHistory",
    "TokenUsage",
    # SSE event models
    "TokenEvent",
    "ThinkingEvent",
    "ToolCallEvent",
    "ToolResultEvent",
    "CompleteEvent",
    "ErrorEvent",
    "PingEvent",
    "SSEEvent",
    # OpenRouter models (legacy)
    "ModelInfo",
    "ModelPricing",
    "ModelsData",
    "ModelsResponse",
    "ModelsRefreshResponse",
    # Provider models (new)
    "ModelCapabilities",
    "ModelConfig",
    "ProviderConfig",
    "RecentModel",
    "LastUsedModel",
    "ModelsConfigSettings",
    "ModelsConfigData",
    "ModelSuggestion",
    # API models (new)
    "AddModelRequest",
    "AddModelResponse",
    "DeprecatedModel",
    "FetchModelsRequest",
    "FetchModelsResponse",
    "GroupedModelsResponse",
    "ModelReference",
    "ModelSuggestionResponse",
    "ProviderResponse",
    "ProvidersResponse",
    "SuggestionsResponse",
    "UpdateModelRequest",
    "UpdateProviderRequest",
]
