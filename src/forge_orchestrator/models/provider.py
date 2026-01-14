"""Pydantic models for multi-provider model management."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class ModelCapabilities(BaseModel):
    """Capabilities of a model."""

    tools: bool = False  # Supports tool/function calling
    vision: bool = False  # Supports image input
    reasoning: bool = False  # Thinking/reasoning model (o1, claude-3.7-thinking, etc.)


class ModelConfig(BaseModel):
    """Configuration for a single model within a provider."""

    id: str  # Model identifier (e.g., "gpt-4o", "claude-sonnet-4-20250514")
    display_name: str  # Human-readable name
    category: Literal["chat", "embedding", "other"] = "chat"
    source: Literal["api", "manual", "legacy"] = "manual"  # How model was added
    capabilities: ModelCapabilities = Field(default_factory=ModelCapabilities)
    favorited: bool = False
    added_at: datetime = Field(default_factory=datetime.utcnow)
    last_used_at: datetime | None = None


class ProviderConfig(BaseModel):
    """Configuration for a provider."""

    enabled: bool = True
    models: dict[str, ModelConfig] = Field(default_factory=dict)
    last_fetched_at: datetime | None = None  # For API-capable providers


class RecentModel(BaseModel):
    """A recently used model reference."""

    provider: str
    model_id: str
    used_at: datetime = Field(default_factory=datetime.utcnow)


class LastUsedModel(BaseModel):
    """Reference to the last used model (for default selection)."""

    provider: str
    model_id: str


class ModelsConfigSettings(BaseModel):
    """Settings for model management."""

    recent_models_limit: int = 5
    last_used_model: LastUsedModel | None = None


class ModelsConfigData(BaseModel):
    """Root data structure for models_config.json."""

    version: str = "1.0"
    providers: dict[str, ProviderConfig] = Field(default_factory=dict)
    recent_models: list[RecentModel] = Field(default_factory=list)
    settings: ModelsConfigSettings = Field(default_factory=ModelsConfigSettings)


class ModelSuggestion(BaseModel):
    """A suggested model for manual addition (for providers without API)."""

    id: str
    display_name: str
    recommended: bool = False
    capabilities: ModelCapabilities = Field(default_factory=ModelCapabilities)
