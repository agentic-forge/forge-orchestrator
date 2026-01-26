"""Tests for TOON (Token-Oriented Object Notation) format support."""

from __future__ import annotations

import pytest

from forge_orchestrator.toon import (
    get_size_comparison,
    is_toon_available,
    should_use_toon,
    to_json,
    to_toon,
    transform_result,
)


class TestToonAvailability:
    """Tests for TOON availability check."""

    def test_is_toon_available(self) -> None:
        """toon_python should be available in test environment."""
        assert is_toon_available() is True


class TestShouldUseToon:
    """Tests for should_use_toon() heuristic."""

    def test_uniform_array_returns_true(self) -> None:
        """Uniform array of objects should use TOON."""
        data = [
            {"id": 1, "name": "Alice"},
            {"id": 2, "name": "Bob"},
        ]
        assert should_use_toon(data) is True

    def test_uniform_array_three_items(self) -> None:
        """Larger uniform array should use TOON."""
        data = [
            {"id": 1, "name": "Alice", "email": "alice@test.com"},
            {"id": 2, "name": "Bob", "email": "bob@test.com"},
            {"id": 3, "name": "Charlie", "email": "charlie@test.com"},
        ]
        assert should_use_toon(data) is True

    def test_heterogeneous_array_returns_false(self) -> None:
        """Non-uniform array should not use TOON."""
        data = [
            {"id": 1, "name": "Alice"},
            {"id": 2, "email": "bob@test.com"},  # Different keys
        ]
        assert should_use_toon(data) is False

    def test_nested_uniform_array_returns_true(self) -> None:
        """Dict with uniform array value should use TOON."""
        data = {
            "users": [
                {"id": 1, "name": "Alice"},
                {"id": 2, "name": "Bob"},
            ]
        }
        assert should_use_toon(data) is True

    def test_simple_dict_returns_false(self) -> None:
        """Simple flat dict should not use TOON."""
        data = {"temperature": 20, "city": "London"}
        assert should_use_toon(data) is False

    def test_string_returns_false(self) -> None:
        """Strings should not use TOON."""
        assert should_use_toon("hello") is False

    def test_number_returns_false(self) -> None:
        """Numbers should not use TOON."""
        assert should_use_toon(42) is False

    def test_none_returns_false(self) -> None:
        """None should not use TOON."""
        assert should_use_toon(None) is False

    def test_empty_array_returns_false(self) -> None:
        """Empty array should not use TOON."""
        assert should_use_toon([]) is False

    def test_single_item_array_returns_false(self) -> None:
        """Single item array should not use TOON (below min_array_size)."""
        data = [{"id": 1, "name": "Alice"}]
        assert should_use_toon(data) is False

    def test_min_array_size_parameter(self) -> None:
        """min_array_size parameter should be respected."""
        data = [
            {"id": 1, "name": "Alice"},
            {"id": 2, "name": "Bob"},
        ]
        assert should_use_toon(data, min_array_size=2) is True
        assert should_use_toon(data, min_array_size=3) is False


class TestToToon:
    """Tests for to_toon() conversion."""

    def test_converts_uniform_array(self) -> None:
        """to_toon should convert uniform array."""
        data = [
            {"id": 1, "name": "Alice"},
            {"id": 2, "name": "Bob"},
        ]
        result = to_toon(data)
        assert result is not None
        assert "Alice" in result
        assert "Bob" in result

    def test_converts_simple_dict(self) -> None:
        """to_toon should convert simple dict."""
        data = {"temperature": 20, "city": "London"}
        result = to_toon(data)
        assert result is not None
        assert "temperature" in result
        assert "20" in result


class TestToJson:
    """Tests for to_json() conversion."""

    def test_converts_dict(self) -> None:
        """to_json should convert dict."""
        data = {"temperature": 20, "city": "London"}
        result = to_json(data)
        assert result == '{"temperature": 20, "city": "London"}'

    def test_converts_array(self) -> None:
        """to_json should convert array."""
        data = [1, 2, 3]
        result = to_json(data)
        assert result == "[1, 2, 3]"

    def test_indent_parameter(self) -> None:
        """to_json should respect indent parameter."""
        data = {"a": 1}
        result = to_json(data, indent=2)
        assert "\n" in result


class TestTransformResult:
    """Tests for transform_result() main function."""

    def test_returns_toon_when_preferred(self) -> None:
        """Returns TOON format when requested."""
        data = [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]
        formatted, content_type = transform_result(data, prefer_toon=True)
        assert content_type == "text/toon"
        assert "Alice" in formatted

    def test_returns_toon_for_simple_data_when_preferred(self) -> None:
        """Returns TOON format for any data when explicitly requested."""
        data = {"simple": "object"}
        formatted, content_type = transform_result(data, prefer_toon=True)
        assert content_type == "text/toon"
        assert "simple" in formatted

    def test_returns_json_when_not_preferred(self) -> None:
        """Returns JSON when client doesn't prefer TOON."""
        data = [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]
        formatted, content_type = transform_result(data, prefer_toon=False)
        assert content_type == "application/json"

    def test_default_is_json(self) -> None:
        """Default format should be JSON."""
        data = [{"id": 1}, {"id": 2}]
        formatted, content_type = transform_result(data)
        assert content_type == "application/json"


class TestGetSizeComparison:
    """Tests for get_size_comparison() metrics function."""

    def test_returns_all_fields(self) -> None:
        """Should return all expected fields."""
        data = [{"id": 1}, {"id": 2}]
        result = get_size_comparison(data)
        assert "json_size" in result
        assert "toon_size" in result
        assert "savings_percent" in result
        assert "savings_bytes" in result
        assert "recommendation" in result
        assert "toon_beneficial" in result

    def test_recommends_toon_for_uniform_array(self) -> None:
        """Should recommend TOON for uniform arrays."""
        data = [
            {"id": 1, "name": "Alice", "email": "alice@test.com"},
            {"id": 2, "name": "Bob", "email": "bob@test.com"},
            {"id": 3, "name": "Charlie", "email": "charlie@test.com"},
        ]
        result = get_size_comparison(data)
        assert result["recommendation"] == "toon"
        assert result["toon_beneficial"] is True
        assert result["savings_percent"] > 0

    def test_recommends_json_for_simple_data(self) -> None:
        """Should recommend JSON for non-beneficial data."""
        data = {"temperature": 20}
        result = get_size_comparison(data)
        assert result["recommendation"] == "json"
        assert result["toon_beneficial"] is False


class TestRealWorldData:
    """Tests with real-world-like data structures."""

    def test_weather_response(self) -> None:
        """Test with weather-like response data."""
        data = {
            "location": "Islamabad, Pakistan",
            "temperature": 9.7,
            "feels_like": 6.7,
            "humidity": 54,
            "weather_description": "Overcast",
            "wind_speed": 6.8,
        }
        # Simple dict - TOON won't provide much benefit but should still work
        result = to_toon(data)
        assert result is not None
        assert "Islamabad" in result

    def test_forecast_response(self) -> None:
        """Test with forecast-like response data (nested uniform array)."""
        data = {
            "location": "Berlin, Germany",
            "timezone": "Europe/Berlin",
            "daily": [
                {"date": "2026-01-19", "temp_max": 5, "temp_min": -2},
                {"date": "2026-01-20", "temp_max": 7, "temp_min": 0},
                {"date": "2026-01-21", "temp_max": 4, "temp_min": -3},
            ],
        }
        assert should_use_toon(data) is True
        comparison = get_size_comparison(data)
        assert comparison["savings_percent"] > 10
