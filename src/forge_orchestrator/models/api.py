"""Pydantic models for API request/response payloads."""

from __future__ import annotations

from pydantic import BaseModel, Field

from forge_orchestrator.models.provider import ModelCapabilities, ModelConfig


class ProviderResponse(BaseModel):
    """Information about a provider for API responses."""

    id: str
    name: str
    configured: bool  # Has API key
    has_api: bool  # Supports fetching models
    model_count: int
    enabled: bool


class ProvidersResponse(BaseModel):
    """Response for GET /providers."""

    providers: list[ProviderResponse]


class ModelReference(BaseModel):
    """Reference to a model (provider + model_id)."""

    provider: str
    model_id: str


class GroupedModelsResponse(BaseModel):
    """Response for GET /models (new format with grouping)."""

    providers: dict[str, dict[str, list[ModelConfig]]]  # provider -> category -> models
    favorites: list[ModelReference] = Field(default_factory=list)
    recent: list[ModelReference] = Field(default_factory=list)
    default_model: ModelReference | None = None


class AddModelRequest(BaseModel):
    """Request to add a model manually."""

    provider: str
    model_id: str
    display_name: str | None = None
    capabilities: ModelCapabilities | None = None


class AddModelResponse(BaseModel):
    """Response after adding a model."""

    status: str = "created"
    model: ModelConfig


class UpdateModelRequest(BaseModel):
    """Request to update a model's properties."""

    favorited: bool | None = None
    display_name: str | None = None
    capabilities: ModelCapabilities | None = None


class FetchModelsRequest(BaseModel):
    """Request to fetch models from a provider's API."""

    provider: str


class DeprecatedModel(BaseModel):
    """A deprecated model (no longer in API)."""

    id: str
    reason: str = "No longer in API response"


class FetchModelsResponse(BaseModel):
    """Response after fetching models from provider API."""

    status: str = "success"
    provider: str
    models_added: int
    models_updated: int
    deprecated: list[DeprecatedModel] = Field(default_factory=list)


class UpdateProviderRequest(BaseModel):
    """Request to update a provider's settings."""

    enabled: bool


class ModelSuggestionResponse(BaseModel):
    """A model suggestion for manual addition."""

    id: str
    display_name: str
    recommended: bool = False


class SuggestionsResponse(BaseModel):
    """Response for GET /models/suggestions/{provider}."""

    provider: str
    suggestions: list[ModelSuggestionResponse]
