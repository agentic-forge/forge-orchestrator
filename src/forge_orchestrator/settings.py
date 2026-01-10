"""Application settings using pydantic-settings."""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


def _load_env_file(env_path: Path) -> None:
    """Load .env file into os.environ."""
    if not env_path.exists():
        return
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key not in os.environ:  # Don't override existing env vars
                    os.environ[key] = value


# Load .env file into os.environ before settings class is instantiated
# This ensures OPENROUTER_API_KEY (without prefix) is available
# Check both CWD and package root directory
_load_env_file(Path(".env"))
_load_env_file(Path(__file__).parent.parent.parent / ".env")


class Settings(BaseSettings):
    """Application settings loaded from environment variables.

    All settings can be overridden via environment variables with the
    ORCHESTRATOR_ prefix (except OPENROUTER_API_KEY which has no prefix).
    """

    model_config = SettingsConfigDict(
        env_prefix="ORCHESTRATOR_",
        env_file=".env",
        env_file_encoding="utf-8",
    )

    # Armory connection
    armory_url: str = "http://localhost:8080/mcp"

    # LLM settings
    default_model: str = "anthropic/claude-sonnet-4"
    openrouter_api_key: SecretStr | None = None

    # Server settings
    host: str = "0.0.0.0"
    port: int = 8001

    # Storage (only for caching, not conversation persistence)
    models_cache_file: Path = Path.home() / ".forge" / "models_cache.json"

    # Model selection
    provider_whitelist: list[str] = [
        "anthropic",
        "openai",
        "google",
        "deepseek",
        "moonshotai",
        "qwen",
    ]
    models_per_provider: int = 3

    # Models to always include (thinking/reasoning models from each provider)
    model_include_list: list[str] = [
        "anthropic/claude-3.7-sonnet:thinking",  # Claude with thinking mode
        "openai/o1",  # OpenAI reasoning model
        "openai/o1-pro",  # OpenAI reasoning model (advanced)
        "deepseek/deepseek-r1",  # DeepSeek R1 reasoning model
        "qwen/qwq-32b",  # Qwen QwQ reasoning model
        "moonshotai/kimi-k2-thinking",  # Moonshot thinking model
    ]

    # Features
    mock_llm: bool = False  # For testing without real LLM
    show_thinking: bool = True  # Stream thinking tokens if model supports

    # SSE settings
    heartbeat_interval: int = 15  # Seconds between ping events
    tool_timeout_warning: int = 30  # Seconds before showing "still working"

    @property
    def openrouter_base_url(self) -> str:
        """OpenRouter API base URL."""
        return "https://openrouter.ai/api/v1"


# Override the env_prefix for openrouter_api_key to have no prefix
class SettingsWithApiKey(Settings):
    """Settings with API key loaded from OPENROUTER_API_KEY (no prefix)."""

    model_config = SettingsConfigDict(
        env_prefix="ORCHESTRATOR_",
        env_file=".env",
        env_file_encoding="utf-8",
        # Allow extra fields from environment
        extra="ignore",
    )

    # Override to use OPENROUTER_API_KEY directly (no ORCHESTRATOR_ prefix)
    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings,
        env_settings,
        dotenv_settings,
        file_secret_settings,
    ):
        """Customize settings sources to handle OPENROUTER_API_KEY without prefix."""
        from pydantic_settings import EnvSettingsSource

        class CustomEnvSettings(EnvSettingsSource):
            def prepare_field_value(
                self, field_name: str, field, value, value_is_complex
            ):
                # Handle openrouter_api_key specially - look for OPENROUTER_API_KEY
                if field_name == "openrouter_api_key" and value is None:
                    import os

                    return os.environ.get("OPENROUTER_API_KEY")
                return super().prepare_field_value(field_name, field, value, value_is_complex)

        return (
            init_settings,
            CustomEnvSettings(settings_cls),
            dotenv_settings,
            file_secret_settings,
        )


# Global settings instance
settings = SettingsWithApiKey()
