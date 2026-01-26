"""Tests for model provider routing."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from forge_orchestrator.orchestrator import AgentOrchestrator
    from forge_orchestrator.settings import Settings


@pytest.fixture
def settings_with_all_keys(tmp_path: Path) -> Settings:
    """Create settings with all API keys configured."""
    from forge_orchestrator.settings import Settings

    class TestSettings(Settings):
        model_config = {"env_prefix": "TEST_ROUTING_", "extra": "ignore"}

    return TestSettings(
        armory_url="http://localhost:8080/mcp",
        default_model="test-model",
        host="127.0.0.1",
        port=8001,
        models_cache_file=tmp_path / "models_cache.json",
        models_config_file=tmp_path / "models_config.json",
        mock_llm=True,
        openai_api_key="sk-test-openai-key",
        anthropic_api_key="sk-test-anthropic-key",
        gemini_api_key="test-gemini-key",
        openrouter_api_key="sk-test-openrouter-key",
    )


@pytest.fixture
def settings_openrouter_only(tmp_path: Path) -> Settings:
    """Create settings with only OpenRouter API key."""
    from forge_orchestrator.settings import Settings

    class TestSettings(Settings):
        model_config = {"env_prefix": "TEST_ROUTING_OR_", "extra": "ignore"}

    return TestSettings(
        armory_url="http://localhost:8080/mcp",
        default_model="test-model",
        host="127.0.0.1",
        port=8001,
        models_cache_file=tmp_path / "models_cache.json",
        models_config_file=tmp_path / "models_config.json",
        mock_llm=True,
        openrouter_api_key="sk-test-openrouter-key",
        # No native API keys
    )


@pytest.fixture
def settings_google_only(tmp_path: Path) -> Settings:
    """Create settings with only Google API key."""
    from forge_orchestrator.settings import Settings

    class TestSettings(Settings):
        model_config = {"env_prefix": "TEST_ROUTING_G_", "extra": "ignore"}

    return TestSettings(
        armory_url="http://localhost:8080/mcp",
        default_model="test-model",
        host="127.0.0.1",
        port=8001,
        models_cache_file=tmp_path / "models_cache.json",
        models_config_file=tmp_path / "models_config.json",
        mock_llm=True,
        gemini_api_key="test-gemini-key",
        # No other API keys
    )


@pytest.fixture
def orchestrator_all_keys(settings_with_all_keys: Settings) -> AgentOrchestrator:
    """Create orchestrator with all API keys."""
    from forge_orchestrator.orchestrator import AgentOrchestrator

    return AgentOrchestrator(settings_with_all_keys)


@pytest.fixture
def orchestrator_openrouter_only(settings_openrouter_only: Settings) -> AgentOrchestrator:
    """Create orchestrator with only OpenRouter key."""
    from forge_orchestrator.orchestrator import AgentOrchestrator

    return AgentOrchestrator(settings_openrouter_only)


@pytest.fixture
def orchestrator_google_only(settings_google_only: Settings) -> AgentOrchestrator:
    """Create orchestrator with only Google key."""
    from forge_orchestrator.orchestrator import AgentOrchestrator

    return AgentOrchestrator(settings_google_only)


class TestGetModelString:
    """Tests for _get_model_string method."""

    def test_already_prefixed_openai(self, orchestrator_all_keys: AgentOrchestrator) -> None:
        """Already prefixed OpenAI model should pass through."""
        result = orchestrator_all_keys._get_model_string("openai:gpt-4o")
        assert result == "openai:gpt-4o"

    def test_already_prefixed_anthropic(self, orchestrator_all_keys: AgentOrchestrator) -> None:
        """Already prefixed Anthropic model should pass through."""
        result = orchestrator_all_keys._get_model_string("anthropic:claude-sonnet-4")
        assert result == "anthropic:claude-sonnet-4"

    def test_already_prefixed_google(self, orchestrator_all_keys: AgentOrchestrator) -> None:
        """Already prefixed Google model should pass through."""
        result = orchestrator_all_keys._get_model_string("google-gla:gemini-2.0-flash")
        assert result == "google-gla:gemini-2.0-flash"

    def test_already_prefixed_openrouter(self, orchestrator_all_keys: AgentOrchestrator) -> None:
        """Already prefixed OpenRouter model should pass through."""
        result = orchestrator_all_keys._get_model_string("openrouter:meta-llama/llama-3-70b")
        assert result == "openrouter:meta-llama/llama-3-70b"


class TestNativeProviderRouting:
    """Tests for routing OpenRouter format models to native providers."""

    def test_google_model_routes_to_native_when_key_available(
        self, orchestrator_all_keys: AgentOrchestrator
    ) -> None:
        """Google model should route to native API when GEMINI_API_KEY is set."""
        result = orchestrator_all_keys._get_model_string("google/gemini-3-flash-preview")
        assert result == "google-gla:gemini-3-flash-preview"

    def test_anthropic_model_routes_to_native_when_key_available(
        self, orchestrator_all_keys: AgentOrchestrator
    ) -> None:
        """Anthropic model should route to native API when ANTHROPIC_API_KEY is set."""
        result = orchestrator_all_keys._get_model_string("anthropic/claude-sonnet-4")
        assert result == "anthropic:claude-sonnet-4"

    def test_openai_model_routes_to_native_when_key_available(
        self, orchestrator_all_keys: AgentOrchestrator
    ) -> None:
        """OpenAI model should route to native API when OPENAI_API_KEY is set."""
        result = orchestrator_all_keys._get_model_string("openai/gpt-4o")
        assert result == "openai:gpt-4o"

    def test_meta_llama_always_routes_to_openrouter(
        self, orchestrator_all_keys: AgentOrchestrator
    ) -> None:
        """Meta-Llama models should always route to OpenRouter (no native API)."""
        result = orchestrator_all_keys._get_model_string("meta-llama/llama-3.3-70b-instruct")
        assert result == "openrouter:meta-llama/llama-3.3-70b-instruct"


class TestOpenRouterFallback:
    """Tests for fallback to OpenRouter when native keys not available."""

    def test_google_model_falls_back_to_openrouter(
        self, orchestrator_openrouter_only: AgentOrchestrator
    ) -> None:
        """Google model should fall back to OpenRouter when no GEMINI_API_KEY."""
        result = orchestrator_openrouter_only._get_model_string("google/gemini-3-flash-preview")
        assert result == "openrouter:google/gemini-3-flash-preview"

    def test_anthropic_model_falls_back_to_openrouter(
        self, orchestrator_openrouter_only: AgentOrchestrator
    ) -> None:
        """Anthropic model should fall back to OpenRouter when no ANTHROPIC_API_KEY."""
        result = orchestrator_openrouter_only._get_model_string("anthropic/claude-sonnet-4")
        assert result == "openrouter:anthropic/claude-sonnet-4"

    def test_openai_model_falls_back_to_openrouter(
        self, orchestrator_openrouter_only: AgentOrchestrator
    ) -> None:
        """OpenAI model should fall back to OpenRouter when no OPENAI_API_KEY."""
        result = orchestrator_openrouter_only._get_model_string("openai/gpt-4o")
        assert result == "openrouter:openai/gpt-4o"


class TestTryNativeProviderFromOpenRouterFormat:
    """Tests for _try_native_provider_from_openrouter_format method."""

    def test_returns_none_for_non_slash_model(
        self, orchestrator_all_keys: AgentOrchestrator
    ) -> None:
        """Should return None for models without slash."""
        result = orchestrator_all_keys._try_native_provider_from_openrouter_format("gpt-4o")
        assert result is None

    def test_extracts_model_name_correctly(
        self, orchestrator_all_keys: AgentOrchestrator
    ) -> None:
        """Should correctly extract model name after provider prefix."""
        result = orchestrator_all_keys._try_native_provider_from_openrouter_format(
            "google/gemini-2.0-flash-exp"
        )
        assert result == "google-gla:gemini-2.0-flash-exp"

    def test_handles_model_with_multiple_slashes(
        self, orchestrator_all_keys: AgentOrchestrator
    ) -> None:
        """Should handle model names with multiple path components."""
        # Some models might have paths like "org/repo/model"
        result = orchestrator_all_keys._try_native_provider_from_openrouter_format(
            "google/some/nested/model"
        )
        # Should split only on first slash
        assert result == "google-gla:some/nested/model"

    def test_case_insensitive_provider_matching(
        self, orchestrator_all_keys: AgentOrchestrator
    ) -> None:
        """Provider matching should be case insensitive."""
        result = orchestrator_all_keys._try_native_provider_from_openrouter_format(
            "Google/gemini-2.0-flash"
        )
        assert result == "google-gla:gemini-2.0-flash"


class TestAutoDetectProvider:
    """Tests for _detect_provider_from_model method."""

    def test_detects_openai_models(
        self, orchestrator_all_keys: AgentOrchestrator
    ) -> None:
        """Should detect OpenAI models by name pattern."""
        assert orchestrator_all_keys._detect_provider_from_model("gpt-4o") == "openai"
        assert orchestrator_all_keys._detect_provider_from_model("gpt-4-turbo") == "openai"
        assert orchestrator_all_keys._detect_provider_from_model("o1-preview") == "openai"

    def test_detects_anthropic_models(
        self, orchestrator_all_keys: AgentOrchestrator
    ) -> None:
        """Should detect Anthropic models by name pattern."""
        assert orchestrator_all_keys._detect_provider_from_model("claude-sonnet-4") == "anthropic"
        assert orchestrator_all_keys._detect_provider_from_model("claude-3-opus") == "anthropic"

    def test_detects_google_models(
        self, orchestrator_all_keys: AgentOrchestrator
    ) -> None:
        """Should detect Google models by name pattern."""
        assert orchestrator_all_keys._detect_provider_from_model("gemini-2.0-flash") == "google-gla"
        assert orchestrator_all_keys._detect_provider_from_model("gemini-pro") == "google-gla"

    def test_returns_none_for_unknown_model(
        self, orchestrator_all_keys: AgentOrchestrator
    ) -> None:
        """Should return None for models that don't match any pattern."""
        result = orchestrator_all_keys._detect_provider_from_model("unknown-model-xyz")
        assert result is None

    def test_falls_back_to_openrouter_when_no_native_key(
        self, orchestrator_openrouter_only: AgentOrchestrator
    ) -> None:
        """Should return None (OpenRouter fallback) when native key not available."""
        # OpenAI model detected but no OPENAI_API_KEY, has OPENROUTER_API_KEY
        result = orchestrator_openrouter_only._detect_provider_from_model("gpt-4o")
        assert result is None  # Falls back to OpenRouter

    def test_returns_provider_when_no_fallback_available(
        self, orchestrator_google_only: AgentOrchestrator
    ) -> None:
        """Should return provider even without key (will fail with clear error)."""
        # OpenAI model detected, no OPENAI_API_KEY, no OPENROUTER_API_KEY
        # Returns "openai" anyway so it fails with a clear error message
        result = orchestrator_google_only._detect_provider_from_model("gpt-4o")
        assert result == "openai"
