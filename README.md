# Forge Orchestrator

LLM agent loop for Agentic Forge - connects the UI to language models and MCP tools via Armory.

## Overview

Forge Orchestrator is the core agent loop that:
- Receives user messages from forge-ui via REST + SSE
- Calls language models via OpenRouter using Pydantic AI
- Executes tools via forge-armory MCP gateway
- Streams responses back to the UI in real-time
- Persists conversations to JSON files

## Installation

```bash
# Install dependencies
uv sync

# Install pre-commit hooks
uv run pre-commit install
```

## Configuration

Set environment variables or create a `.env` file:

```bash
# Required
OPENROUTER_API_KEY=sk-or-...

# Optional (with defaults)
ORCHESTRATOR_ARMORY_URL=http://localhost:8080/mcp
ORCHESTRATOR_DEFAULT_MODEL=anthropic/claude-sonnet-4
ORCHESTRATOR_HOST=0.0.0.0
ORCHESTRATOR_PORT=8001
ORCHESTRATOR_CONVERSATIONS_DIR=~/.forge/conversations
ORCHESTRATOR_MOCK_LLM=false
ORCHESTRATOR_SHOW_THINKING=true
ORCHESTRATOR_HEARTBEAT_INTERVAL=15
ORCHESTRATOR_TOOL_TIMEOUT_WARNING=30
```

## Usage

```bash
# Show configuration
uv run orchestrator info

# Start the server
uv run orchestrator serve

# Start with auto-reload for development
uv run orchestrator serve --reload
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | /conversations | Create new conversation |
| GET | /conversations/{id} | Get conversation state |
| DELETE | /conversations/{id} | Delete conversation |
| POST | /conversations/{id}/messages | Send message |
| GET | /conversations/{id}/stream | SSE stream for events |
| POST | /conversations/{id}/cancel | Cancel generation |
| DELETE | /conversations/{id}/messages/{n} | Delete from message N |
| PATCH | /conversations/{id}/system-prompt | Update system prompt |
| GET | /health | Health check |
| POST | /tools/refresh | Refresh tools from Armory |
| GET | /tools | List available tools |

## SSE Events

The `/conversations/{id}/stream` endpoint streams Server-Sent Events:

- `token` - Streaming text tokens
- `thinking` - Model thinking content (if enabled)
- `tool_call` - Tool execution status (pending/executing/complete/error)
- `tool_result` - Tool execution result
- `complete` - Response complete with usage stats
- `error` - Error occurred
- `ping` - Heartbeat (every 15 seconds)

## Development

```bash
# Run tests
uv run pytest

# Type checking
uv run basedpyright

# Linting
uv run ruff check .

# Format code
uv run ruff format .

# Run all pre-commit checks
uv run pre-commit run --all-files
```

## Architecture

```
forge-ui (Vue.js)
    ↓ SSE + REST
forge-orchestrator (this project)
    ↓ MCP (via fastmcp.Client)
forge-armory (MCP gateway)
    ↓ MCP
MCP servers (weather, search, etc.)
```

## License

MIT
