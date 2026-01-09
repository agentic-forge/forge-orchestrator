"""Pydantic models for OpenRouter API responses."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class ModelPricing(BaseModel):
    """Pricing information for a model (per token costs)."""

    prompt: float = 0.0  # Cost per prompt token
    completion: float = 0.0  # Cost per completion token


class ModelInfo(BaseModel):
    """Information about an available model."""

    id: str  # e.g., "anthropic/claude-sonnet-4"
    name: str  # Display name
    provider: str  # Extracted from id prefix
    context_length: int
    pricing: ModelPricing
    supports_tools: bool = False  # Derived from supported_parameters
    supports_vision: bool = False  # Derived from modality
    modality: str | None = None  # e.g., "text+image->text"
    created: int | None = None  # Unix timestamp


class ModelsData(BaseModel):
    """Cached models data structure."""

    models: list[ModelInfo] = Field(default_factory=list)
    providers: list[str] = Field(default_factory=list)  # Unique sorted providers
    fetched_at: datetime = Field(default_factory=datetime.utcnow)


class ModelsResponse(BaseModel):
    """API response for GET /models."""

    providers: list[str]
    models: list[ModelInfo]
    cached_at: datetime | None = None


class ModelsRefreshResponse(BaseModel):
    """API response for POST /models/refresh."""

    status: str = "refreshed"
    model_count: int
    provider_count: int
