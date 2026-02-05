"""Tests for the interactive tool pause mechanism."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock

from forge_orchestrator.models import (
    CompleteEvent,
    ToolCallEvent,
    ToolResultEvent,
    UiMetadata,
)

if TYPE_CHECKING:
    from forge_orchestrator.orchestrator import AgentOrchestrator


class TestExtractUiMetadataInteraction:
    """Tests for _extract_ui_metadata with requiresInteraction flag."""

    def test_extracts_requires_interaction_true(
        self, mock_orchestrator: AgentOrchestrator
    ) -> None:
        """Should propagate requiresInteraction=true from tool result."""
        content = {
            "_meta": {
                "ui": {
                    "resourceUri": "ui://location-picker",
                    "permissions": ["geolocation"],
                    "requiresInteraction": True,
                }
            },
            "_content": "some content",
        }
        result = mock_orchestrator._extract_ui_metadata(content, "weather__pick_location")
        assert result is not None
        assert result.requiresInteraction is True

    def test_extracts_requires_interaction_false(
        self, mock_orchestrator: AgentOrchestrator
    ) -> None:
        """Should default requiresInteraction to false when not present."""
        content = {
            "_meta": {
                "ui": {
                    "resourceUri": "ui://chart",
                }
            },
            "_content": "some content",
        }
        result = mock_orchestrator._extract_ui_metadata(content, "weather__show_chart")
        assert result is not None
        assert result.requiresInteraction is False

    def test_no_ui_metadata_returns_none(
        self, mock_orchestrator: AgentOrchestrator
    ) -> None:
        """Should return None when no _meta.ui present."""
        content = {"temperature": 20, "condition": "sunny"}
        result = mock_orchestrator._extract_ui_metadata(content, "weather__get_weather")
        assert result is None


class TestCollectEventsWithInteractionCheck:
    """Tests for _collect_events_with_interaction_check helper."""

    def _make_tool_call_part(
        self, tool_name: str, tool_call_id: str, args: dict | None = None,
    ) -> MagicMock:
        """Create a mock ToolCallPart."""
        from pydantic_ai.messages import ToolCallPart

        part = MagicMock(spec=ToolCallPart)
        part.tool_name = tool_name
        part.tool_call_id = tool_call_id
        part.args = args or {}
        return part

    def _make_tool_return_part(
        self,
        tool_name: str,
        tool_call_id: str,
        content: dict,
        timestamp: object = None,
    ) -> MagicMock:
        """Create a mock ToolReturnPart."""
        from pydantic_ai.messages import ToolReturnPart

        part = MagicMock(spec=ToolReturnPart)
        part.tool_name = tool_name
        part.tool_call_id = tool_call_id
        part.content = content
        part.timestamp = timestamp
        return part

    def _make_message(self, parts: list, timestamp: object = None) -> MagicMock:
        """Create a mock message with parts."""
        msg = MagicMock()
        msg.parts = parts
        msg.timestamp = timestamp
        return msg

    def test_no_interactive_tools(
        self, mock_orchestrator: AgentOrchestrator
    ) -> None:
        """Should return all events and None when no interactive tools."""
        call_part = self._make_tool_call_part("weather__get_weather", "tc_001")
        return_part = self._make_tool_return_part(
            "weather__get_weather", "tc_001",
            {"temperature": 20},
        )
        msg1 = self._make_message([call_part])
        msg2 = self._make_message([return_part])

        events, interactive_id = mock_orchestrator._collect_events_with_interaction_check(
            [msg1, msg2]
        )

        assert interactive_id is None
        assert len(events) == 2
        assert isinstance(events[0], ToolCallEvent)
        assert isinstance(events[1], ToolResultEvent)

    def test_truncates_at_interactive_tool(
        self, mock_orchestrator: AgentOrchestrator
    ) -> None:
        """Should stop collecting after an interactive tool and return its ID."""
        # First tool: pick_location (interactive)
        pick_call = self._make_tool_call_part("weather__pick_location", "tc_001")
        pick_return = self._make_tool_return_part(
            "weather__pick_location", "tc_001",
            {
                "_meta": {
                    "ui": {
                        "resourceUri": "ui://location-picker",
                        "requiresInteraction": True,
                    }
                },
                "_content": {"instructions": "Click the map"},
            },
        )

        # Second tool: get_weather (should be truncated)
        weather_call = self._make_tool_call_part("weather__get_weather", "tc_002")
        weather_return = self._make_tool_return_part(
            "weather__get_weather", "tc_002",
            {"temperature": 20},
        )

        msg1 = self._make_message([pick_call])
        msg2 = self._make_message([pick_return])
        msg3 = self._make_message([weather_call])
        msg4 = self._make_message([weather_return])

        events, interactive_id = mock_orchestrator._collect_events_with_interaction_check(
            [msg1, msg2, msg3, msg4]
        )

        assert interactive_id == "tc_001"
        # Should only have the pick_location call + result, not get_weather
        assert len(events) == 2
        assert isinstance(events[0], ToolCallEvent)
        assert events[0].tool_name == "weather__pick_location"
        assert isinstance(events[1], ToolResultEvent)
        assert events[1].ui_metadata is not None
        assert events[1].ui_metadata.requiresInteraction is True

    def test_non_interactive_ui_tool_passes_through(
        self, mock_orchestrator: AgentOrchestrator
    ) -> None:
        """Tools with UI but no requiresInteraction should not trigger pause."""
        call_part = self._make_tool_call_part("weather__show_chart", "tc_001")
        return_part = self._make_tool_return_part(
            "weather__show_chart", "tc_001",
            {
                "_meta": {
                    "ui": {
                        "resourceUri": "ui://chart",
                        # requiresInteraction not set (defaults to False)
                    }
                },
                "_content": {"data": [1, 2, 3]},
            },
        )

        # Another tool after the chart
        call2 = self._make_tool_call_part("weather__get_weather", "tc_002")
        return2 = self._make_tool_return_part(
            "weather__get_weather", "tc_002",
            {"temperature": 20},
        )

        msg1 = self._make_message([call_part])
        msg2 = self._make_message([return_part])
        msg3 = self._make_message([call2])
        msg4 = self._make_message([return2])

        events, interactive_id = mock_orchestrator._collect_events_with_interaction_check(
            [msg1, msg2, msg3, msg4]
        )

        # No pause - all 4 events collected
        assert interactive_id is None
        assert len(events) == 4

    def test_empty_messages(
        self, mock_orchestrator: AgentOrchestrator
    ) -> None:
        """Should handle empty message list."""
        events, interactive_id = mock_orchestrator._collect_events_with_interaction_check([])
        assert interactive_id is None
        assert len(events) == 0

    def test_messages_without_parts(
        self, mock_orchestrator: AgentOrchestrator
    ) -> None:
        """Should handle messages that don't have parts attribute."""
        msg = MagicMock(spec=[])  # No attributes
        events, interactive_id = mock_orchestrator._collect_events_with_interaction_check([msg])
        assert interactive_id is None
        assert len(events) == 0
