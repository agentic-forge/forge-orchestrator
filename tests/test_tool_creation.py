"""Tests for tool creation and schema handling."""

from __future__ import annotations

import pytest

from forge_orchestrator.orchestrator import (
    _make_tool_wrapper,
    create_discovered_tools,
    format_tool_result_for_llm,
)
from forge_orchestrator.toon import is_toon_available


class TestFormatToolResultForLLM:
    """Tests for format_tool_result_for_llm function."""

    def test_returns_original_when_toon_disabled(self) -> None:
        """Should return original result when TOON is disabled."""
        data = {"temperature": 20, "city": "London"}
        result = format_tool_result_for_llm(data, use_toon=False)
        assert result == data

    def test_returns_toon_string_when_enabled(self) -> None:
        """Should return TOON string when enabled and available."""
        if not is_toon_available():
            pytest.skip("toon_python not available")

        data = [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]
        result = format_tool_result_for_llm(data, use_toon=True)
        # Result should be a string (TOON format)
        assert isinstance(result, str)
        assert "Alice" in result
        assert "Bob" in result

    def test_preserves_string_result(self) -> None:
        """Should preserve string results."""
        data = "Already a string"
        result = format_tool_result_for_llm(data, use_toon=True)
        # Strings go through TOON but come back as strings
        assert isinstance(result, str)


class TestCreateDiscoveredTools:
    """Tests for create_discovered_tools function."""

    def test_creates_tools_from_definitions(self) -> None:
        """Should create Tool objects from tool definitions."""
        tool_defs = [
            {
                "name": "test_tool",
                "description": "A test tool",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"}
                    },
                    "required": ["query"],
                },
            }
        ]

        tools = create_discovered_tools(
            tool_defs, "http://localhost:4042/mcp", {}, use_toon=False
        )

        assert len(tools) == 1
        assert tools[0].name == "test_tool"
        assert tools[0].description == "A test tool"

    def test_preserves_input_schema(self) -> None:
        """Should preserve inputSchema in function_schema."""
        tool_defs = [
            {
                "name": "search_tools",
                "description": "Search for tools",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Natural language query",
                        },
                        "threshold": {
                            "type": "number",
                            "description": "Similarity threshold",
                            "minimum": 0,
                            "maximum": 1,
                        },
                    },
                    "required": ["query"],
                },
            }
        ]

        tools = create_discovered_tools(
            tool_defs, "http://localhost:4042/mcp", {}, use_toon=False
        )

        assert len(tools) == 1
        tool = tools[0]

        # Check that function_schema has the correct json_schema
        assert hasattr(tool, "function_schema")
        assert tool.function_schema is not None
        schema = tool.function_schema.json_schema
        assert schema["type"] == "object"
        assert "query" in schema["properties"]
        assert schema["properties"]["query"]["type"] == "string"
        assert "required" in schema
        assert "query" in schema["required"]

    def test_skips_tools_without_name(self) -> None:
        """Should skip tool definitions without a name."""
        tool_defs = [
            {"description": "No name tool", "inputSchema": {}},
            {"name": "valid_tool", "description": "Has name", "inputSchema": {}},
        ]

        tools = create_discovered_tools(
            tool_defs, "http://localhost:4042/mcp", {}, use_toon=False
        )

        assert len(tools) == 1
        assert tools[0].name == "valid_tool"

    def test_handles_empty_input_schema(self) -> None:
        """Should handle tools with empty or missing inputSchema."""
        tool_defs = [
            {"name": "no_schema", "description": "Tool without schema"},
            {"name": "empty_schema", "description": "Tool with empty schema", "inputSchema": {}},
        ]

        tools = create_discovered_tools(
            tool_defs, "http://localhost:4042/mcp", {}, use_toon=False
        )

        assert len(tools) == 2

    def test_multiple_tools(self) -> None:
        """Should create multiple tools from definitions."""
        tool_defs = [
            {
                "name": "weather__get_current_weather",
                "description": "Get current weather",
                "inputSchema": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                },
            },
            {
                "name": "weather__get_forecast",
                "description": "Get weather forecast",
                "inputSchema": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}, "days": {"type": "integer"}},
                },
            },
        ]

        tools = create_discovered_tools(
            tool_defs, "http://localhost:4042/mcp", {}, use_toon=True
        )

        assert len(tools) == 2
        names = {t.name for t in tools}
        assert "weather__get_current_weather" in names
        assert "weather__get_forecast" in names


class TestMakeToolWrapper:
    """Tests for _make_tool_wrapper function."""

    def test_wrapper_has_correct_name(self) -> None:
        """Wrapper function should have __name__ set to tool name."""
        wrapper = _make_tool_wrapper(
            "http://localhost:4042/mcp", "test_tool", {}, use_toon=False
        )
        assert wrapper.__name__ == "test_tool"

    def test_search_tools_wrapper_skips_toon(self) -> None:
        """search_tools wrapper should not apply TOON transformation."""
        # This is tested indirectly - the wrapper for search_tools
        # should return raw dict, not TOON string
        wrapper = _make_tool_wrapper(
            "http://localhost:4042/mcp", "search_tools", {}, use_toon=True
        )
        assert wrapper.__name__ == "search_tools"
        # The actual TOON skipping is tested in integration tests
        # since it requires calling the wrapper with real data
