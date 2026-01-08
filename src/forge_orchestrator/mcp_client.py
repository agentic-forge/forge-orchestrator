"""MCP client wrapper for connecting to Armory."""

from __future__ import annotations

from typing import Any

from fastmcp import Client


class OrchestratorError(Exception):
    """Base exception for Orchestrator errors."""


class ArmoryConnectionError(OrchestratorError):
    """Failed to connect to Armory."""


class ToolCallError(OrchestratorError):
    """Failed to call a tool."""


class ArmoryClient:
    """Wrapper around fastmcp.Client for connecting to Armory.

    Provides a simpler interface for:
    - Health checks / ping
    - Listing available tools
    - Direct tool calls (for refresh operations)

    For Pydantic AI integration, the orchestrator uses MCPServerStreamableHTTP
    directly as a toolset.
    """

    def __init__(self, armory_url: str, timeout: float = 30.0) -> None:
        """Initialize the client.

        Args:
            armory_url: URL of the Armory MCP endpoint (e.g., http://localhost:8080/mcp)
            timeout: Request timeout in seconds.
        """
        self.armory_url = armory_url
        self.timeout = timeout

    async def ping(self) -> bool:
        """Check if Armory is responsive.

        Returns:
            True if Armory responds, False otherwise.
        """
        try:
            async with Client(self.armory_url) as client:
                await client.ping()
                return True
        except Exception:
            return False

    async def get_server_info(self) -> dict[str, Any]:
        """Get server capabilities and info.

        Returns:
            Dictionary with server info:
            - name: Server name
            - version: Server version
            - protocol_version: MCP protocol version
            - capabilities: Server capabilities
        """
        try:
            async with Client(self.armory_url) as client:
                init_result = client.initialize_result

                if init_result is not None:
                    info = getattr(init_result, "serverInfo", None)
                    name = getattr(info, "name", "Unknown") if info else "Unknown"
                    version = getattr(info, "version", "Unknown") if info else "Unknown"
                    return {
                        "name": name,
                        "version": version,
                        "protocol_version": getattr(init_result, "protocolVersion", "Unknown"),
                        "capabilities": _capabilities_to_dict(
                            getattr(init_result, "capabilities", None)
                        ),
                    }

                return {
                    "name": "Unknown",
                    "version": "Unknown",
                    "protocol_version": "Unknown",
                    "capabilities": {},
                }
        except Exception as e:
            raise ArmoryConnectionError(f"Failed to connect to {self.armory_url}: {e}") from e

    async def list_tools(self) -> list[dict[str, Any]]:
        """List available tools from Armory.

        Returns:
            List of tools with name, description, and input_schema.
            Tool names are prefixed (e.g., weather__get_forecast).
        """
        try:
            async with Client(self.armory_url) as client:
                tools = await client.list_tools()
                return [
                    {
                        "name": tool.name,
                        "description": tool.description,
                        "input_schema": tool.inputSchema,
                    }
                    for tool in tools
                ]
        except Exception as e:
            raise ArmoryConnectionError(f"Failed to list tools: {e}") from e

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Call a tool and return the result.

        Args:
            name: Tool name (prefixed, e.g., weather__get_forecast)
            arguments: Tool arguments.

        Returns:
            Dictionary with content and is_error flag.
        """
        try:
            async with Client(self.armory_url) as client:
                result = await client.call_tool(name, arguments or {})
                return {
                    "content": _content_to_list(result),
                    "is_error": getattr(result, "isError", False),
                }
        except Exception as e:
            raise ToolCallError(f"Failed to call tool '{name}': {e}") from e


def _capabilities_to_dict(capabilities: Any) -> dict[str, Any]:
    """Convert server capabilities to a dictionary."""
    if capabilities is None:
        return {}

    result: dict[str, Any] = {}

    if hasattr(capabilities, "tools") and capabilities.tools:
        result["tools"] = True
    if hasattr(capabilities, "resources") and capabilities.resources:
        result["resources"] = True
    if hasattr(capabilities, "prompts") and capabilities.prompts:
        result["prompts"] = True
    if hasattr(capabilities, "logging") and capabilities.logging:
        result["logging"] = True

    return result


def _content_to_list(result: Any) -> list[dict[str, Any]]:
    """Convert tool result content to a list of dictionaries."""
    content_list = []

    # Handle different result types
    if hasattr(result, "content"):
        items = result.content if isinstance(result.content, list) else [result.content]
    elif isinstance(result, list):
        items = result
    else:
        items = [result]

    for item in items:
        if hasattr(item, "text"):
            content_list.append({"type": "text", "text": item.text})
        elif hasattr(item, "data"):
            content_list.append(
                {
                    "type": "image",
                    "data": item.data,
                    "mime_type": getattr(item, "mimeType", "image/png"),
                }
            )
        elif isinstance(item, str):
            content_list.append({"type": "text", "text": item})
        elif isinstance(item, dict):
            content_list.append(item)
        else:
            content_list.append({"type": "text", "text": str(item)})

    return content_list
