"""API key management for BYOK (Bring Your Own Key) support.

Provides key resolution with priority: header > environment.
This allows deployed instances to have default keys while users can override.
"""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING

from fastapi import HTTPException, Request

if TYPE_CHECKING:
    from forge_orchestrator.settings import Settings


class KeyProvider:
    """Provides API keys with priority: header > environment.

    This allows deployed instances to have default keys while
    users can still override with their own keys via the UI.
    """

    # Map provider names to environment variable names
    PROVIDER_ENV_VARS = {
        "openrouter": "OPENROUTER_API_KEY",
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "google": "GEMINI_API_KEY",
    }

    def __init__(self, settings: Settings) -> None:
        """Initialize the key provider.

        Args:
            settings: Application settings instance.
        """
        self._settings = settings
        self.allow_header_keys = settings.allow_header_keys

    def get_llm_key(
        self,
        request: Request,
        provider: str | None = None,
    ) -> tuple[str, str]:
        """Get LLM API key with priority: header > environment.

        Args:
            request: The HTTP request (for header extraction).
            provider: Requested provider (from request body/header).

        Returns:
            Tuple of (api_key, provider_name).

        Raises:
            HTTPException: If no API key is available.

        Priority:
        1. X-LLM-Key header for requested provider (if allow_header_keys)
        2. Environment variable for requested provider
        3. Any available environment variable (fallback)
        """
        # First: check header (user-provided key takes priority)
        if self.allow_header_keys:
            header_key = request.headers.get("X-LLM-Key")
            header_provider = request.headers.get("X-LLM-Provider", provider)
            if header_key:
                return header_key, header_provider or "openrouter"

        # Second: try requested provider's env var via settings
        if provider:
            env_key = self._get_key_from_settings(provider)
            if env_key:
                return env_key, provider

        # Third: fall back to any available key from settings
        for prov in ["openrouter", "openai", "anthropic", "google"]:
            env_key = self._get_key_from_settings(prov)
            if env_key:
                return env_key, prov

        # No key found anywhere
        raise HTTPException(
            status_code=401,
            detail="No API key available. Please configure your API key in Settings, "
            "or contact the administrator.",
        )

    def _get_key_from_settings(self, provider: str) -> str | None:
        """Get API key from settings for a provider.

        Args:
            provider: Provider name.

        Returns:
            API key string or None if not configured.
        """
        key_attr_map = {
            "openrouter": "openrouter_api_key",
            "openai": "openai_api_key",
            "anthropic": "anthropic_api_key",
            "google": "gemini_api_key",
        }

        attr = key_attr_map.get(provider)
        if not attr:
            return None

        secret = getattr(self._settings, attr, None)
        if secret is None:
            return None

        # SecretStr has get_secret_value()
        if hasattr(secret, "get_secret_value"):
            return secret.get_secret_value()
        return str(secret) if secret else None

    def get_mcp_keys(self, request: Request) -> dict[str, str]:
        """Get MCP server API keys from header.

        These are merged with any server-side configured keys,
        with header keys taking priority.

        Args:
            request: The HTTP request.

        Returns:
            Dict of server-name -> api-key.
        """
        header_keys: dict[str, str] = {}
        if self.allow_header_keys:
            keys_json = request.headers.get("X-MCP-Keys", "{}")
            try:
                header_keys = json.loads(keys_json)
            except json.JSONDecodeError:
                pass

        return header_keys

    def get_configured_providers(self) -> dict[str, bool]:
        """Check which providers have keys configured in environment.

        Returns:
            Dict of provider -> has_key.
        """
        return {
            "openrouter": bool(self._get_key_from_settings("openrouter")),
            "openai": bool(self._get_key_from_settings("openai")),
            "anthropic": bool(self._get_key_from_settings("anthropic")),
            "google": bool(self._get_key_from_settings("google")),
        }

    def has_any_env_key(self) -> bool:
        """Check if any LLM key is configured in environment.

        Returns:
            True if at least one provider has a key configured.
        """
        return any(self.get_configured_providers().values())
