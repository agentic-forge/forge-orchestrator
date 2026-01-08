# forge-orchestrator Testing Guide

This document lists all implemented features and how to manually test them.

## What Was Implemented

### Core Components

| Component | File | Description |
|-----------|------|-------------|
| Settings | `src/forge_orchestrator/settings.py` | Configuration via environment variables |
| CLI | `src/forge_orchestrator/cli.py` | Typer CLI with `info` and `serve` commands |
| Logging | `src/forge_orchestrator/logging.py` | Structlog JSON logging |
| SSE Models | `src/forge_orchestrator/models/messages.py` | TokenEvent, ThinkingEvent, ToolCallEvent, etc. |
| Conversation Models | `src/forge_orchestrator/models/conversation.py` | Message, Conversation, TokenUsage |
| Storage | `src/forge_orchestrator/storage.py` | JSON file persistence for conversations |
| MCP Client | `src/forge_orchestrator/mcp_client.py` | Armory connection wrapper |
| Orchestrator | `src/forge_orchestrator/orchestrator.py` | Pydantic AI agent loop with streaming |
| Conversation Manager | `src/forge_orchestrator/conversation.py` | High-level conversation operations |
| Server | `src/forge_orchestrator/server.py` | FastAPI + SSE endpoints |

### API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| GET | `/tools` | List available MCP tools |
| POST | `/tools/refresh` | Refresh tools from Armory |
| POST | `/conversations` | Create new conversation |
| GET | `/conversations/{id}` | Get conversation state |
| DELETE | `/conversations/{id}` | Delete conversation |
| POST | `/conversations/{id}/messages` | Send message (returns 202) |
| GET | `/conversations/{id}/stream` | SSE stream for responses |
| POST | `/conversations/{id}/cancel` | Cancel active generation |
| DELETE | `/conversations/{id}/messages/{n}` | Delete messages from index N |
| PATCH | `/conversations/{id}/system-prompt` | Update system prompt |
| PATCH | `/conversations/{id}/model` | Update model |

---

## Prerequisites

```bash
cd /MyWork/Projects/agentic-forge/forge-orchestrator

# Install dependencies
uv sync

# Run automated tests first (should all pass)
uv run pytest -v
```

---

## Manual Testing

### 1. CLI Commands

#### Test: Show version and configuration
```bash
uv run orchestrator info
```

**Expected output:**
- Version: 0.1.0
- Configuration settings displayed in a table
- Conversations directory path shown

#### Test: Show help
```bash
uv run orchestrator --help
uv run orchestrator serve --help
```

### 2. Server in Mock Mode

Start the server in mock mode (no API keys needed):

```bash
ORCHESTRATOR_MOCK_LLM=true uv run orchestrator serve
```

**Expected:** Server starts on http://0.0.0.0:8001

#### Test: Health Check
```bash
curl http://localhost:8001/health
```

**Expected:**
```json
{"status":"healthy","armory_connected":false,"active_runs":0}
```

#### Test: List Tools (empty in mock mode)
```bash
curl http://localhost:8001/tools
```

**Expected:**
```json
{"tools":[]}
```

### 3. Conversation CRUD

#### Test: Create Conversation
```bash
curl -X POST http://localhost:8001/conversations \
  -H "Content-Type: application/json" \
  -d '{"model": "test-model", "system_prompt": "You are a helpful assistant."}'
```

**Expected:** Returns conversation object with `metadata.id`

Save the conversation ID for subsequent tests:
```bash
CONV_ID="<id from response>"
```

#### Test: Get Conversation
```bash
curl http://localhost:8001/conversations/$CONV_ID
```

**Expected:** Returns full conversation with metadata and empty messages array

#### Test: Update System Prompt
```bash
curl -X PATCH "http://localhost:8001/conversations/$CONV_ID/system-prompt" \
  -H "Content-Type: application/json" \
  -d '{"content": "You are a pirate assistant."}'
```

**Expected:** Returns updated conversation, `system_prompt_history` should have 1 entry

#### Test: Update Model
```bash
curl -X PATCH "http://localhost:8001/conversations/$CONV_ID/model" \
  -H "Content-Type: application/json" \
  -d '{"model": "anthropic/claude-sonnet-4"}'
```

**Expected:** Returns updated conversation with new model

### 4. SSE Streaming (Mock Mode)

#### Test: Basic Message Stream
```bash
curl -N "http://localhost:8001/conversations/$CONV_ID/stream?message=Hello"
```

**Expected SSE events:**
1. `event: thinking` - Thinking event
2. Multiple `event: token` - Token events with cumulative text
3. `event: complete` - Complete event with usage stats

#### Test: Weather Tool Call (Mock)
```bash
curl -N "http://localhost:8001/conversations/$CONV_ID/stream?message=What%27s%20the%20weather%3F"
```

**Expected SSE events:**
1. `event: thinking`
2. `event: tool_call` with status "pending"
3. `event: tool_call` with status "executing"
4. `event: tool_result` with mock weather data
5. Multiple `event: token`
6. `event: complete`

### 5. Message Management

#### Test: Send Message via POST
```bash
curl -X POST "http://localhost:8001/conversations/$CONV_ID/messages" \
  -H "Content-Type: application/json" \
  -d '{"content": "Remember this test message."}'
```

**Expected:** 202 Accepted, then stream available at `/stream`

#### Test: Get Conversation with Messages
```bash
curl http://localhost:8001/conversations/$CONV_ID
```

**Expected:** Messages array should have user and assistant messages

#### Test: Delete Messages from Index
```bash
# First, check current message count
curl http://localhost:8001/conversations/$CONV_ID | jq '.metadata.message_count'

# Delete from message 2 onwards (keeps first 2 messages)
curl -X DELETE "http://localhost:8001/conversations/$CONV_ID/messages/2"
```

**Expected:** Returns truncated conversation

### 6. Cancellation

#### Test: Cancel Generation
```bash
# In terminal 1: Start a stream
curl -N "http://localhost:8001/conversations/$CONV_ID/stream?message=Tell%20me%20a%20long%20story"

# In terminal 2: Cancel it
curl -X POST "http://localhost:8001/conversations/$CONV_ID/cancel"
```

**Expected:** Stream should receive `event: error` with code "CANCELLED"

### 7. Delete Conversation

```bash
curl -X DELETE http://localhost:8001/conversations/$CONV_ID
```

**Expected:**
```json
{"deleted":true}
```

### 8. Error Handling

#### Test: Get Non-existent Conversation
```bash
curl http://localhost:8001/conversations/non-existent-id
```

**Expected:** 404 Not Found

#### Test: Stream Non-existent Conversation
```bash
curl -N "http://localhost:8001/conversations/non-existent-id/stream?message=Hello"
```

**Expected:** SSE error event with code "CONVERSATION_NOT_FOUND"

---

## Testing with Real LLM (OpenRouter)

### Prerequisites

1. Get an API key from [OpenRouter](https://openrouter.ai/)
2. Start Armory (optional, for tool access):
   ```bash
   cd /MyWork/Projects/agentic-forge/forge-armory
   uv run armory serve
   ```

### Start Server with OpenRouter

```bash
export OPENROUTER_API_KEY="your-api-key-here"
uv run orchestrator serve
```

### Test Real Streaming

```bash
# Create conversation
CONV_ID=$(curl -s -X POST http://localhost:8001/conversations \
  -H "Content-Type: application/json" \
  -d '{"model": "anthropic/claude-sonnet-4"}' | jq -r '.metadata.id')

# Stream a response
curl -N "http://localhost:8001/conversations/$CONV_ID/stream?message=Hello%2C%20who%20are%20you%3F"
```

**Expected:** Real streaming tokens from Claude

---

## Testing with Armory (MCP Tools)

### Prerequisites

1. Start Armory with MCP servers configured
2. Set `ORCHESTRATOR_ARMORY_URL` if not default

```bash
export ORCHESTRATOR_ARMORY_URL="http://localhost:8080/mcp"
export OPENROUTER_API_KEY="your-api-key"
uv run orchestrator serve
```

### Test Tool Listing

```bash
curl http://localhost:8001/tools
```

**Expected:** List of tools from connected MCP servers

### Test Tool Refresh

```bash
curl -X POST http://localhost:8001/tools/refresh
```

**Expected:** Updated tool list

---

## Automated Test Suite

```bash
# Run all tests
uv run pytest -v

# Run with coverage
uv run pytest --cov=forge_orchestrator --cov-report=term-missing

# Run specific test file
uv run pytest tests/test_server.py -v

# Run specific test
uv run pytest tests/test_orchestrator.py::TestAgentOrchestrator::test_run_stream_mock_basic -v
```

### Current Test Coverage

| Module | Tests |
|--------|-------|
| test_models.py | 16 tests (SSE events, conversation models) |
| test_storage.py | 10 tests (CRUD, atomic writes, metadata) |
| test_orchestrator.py | 7 tests (mock streaming, cancellation) |
| test_server.py | 15 tests (REST endpoints) |
| **Total** | **48 tests** |

---

## Known Limitations

1. **No WebSocket support** - Uses SSE (by design per spec)
2. **No database** - Uses JSON files (by design for simplicity)
3. **Tool results not persisted** - Tool calls shown in stream but not saved to conversation messages yet
4. **No authentication** - Add auth middleware for production
5. **Single node only** - No distributed session support

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ORCHESTRATOR_HOST` | 0.0.0.0 | Server host |
| `ORCHESTRATOR_PORT` | 8001 | Server port |
| `ORCHESTRATOR_ARMORY_URL` | http://localhost:8080/mcp | Armory MCP endpoint |
| `ORCHESTRATOR_DEFAULT_MODEL` | anthropic/claude-sonnet-4 | Default LLM model |
| `ORCHESTRATOR_CONVERSATIONS_DIR` | ~/.forge/conversations | Storage directory |
| `ORCHESTRATOR_MOCK_LLM` | false | Enable mock mode |
| `ORCHESTRATOR_SHOW_THINKING` | true | Show thinking events |
| `ORCHESTRATOR_HEARTBEAT_INTERVAL` | 15 | Ping interval (seconds) |
| `ORCHESTRATOR_TOOL_TIMEOUT_WARNING` | 30 | Tool timeout warning (seconds) |
| `OPENROUTER_API_KEY` | (required) | OpenRouter API key |

---

## Quick Test Script

Create a file `test_manual.sh`:

```bash
#!/bin/bash
set -e

BASE_URL="http://localhost:8001"

echo "=== Health Check ==="
curl -s $BASE_URL/health | jq

echo -e "\n=== Create Conversation ==="
CONV=$(curl -s -X POST $BASE_URL/conversations \
  -H "Content-Type: application/json" \
  -d '{"model": "test-model", "system_prompt": "You are helpful."}')
echo $CONV | jq
CONV_ID=$(echo $CONV | jq -r '.metadata.id')

echo -e "\n=== Get Conversation ==="
curl -s $BASE_URL/conversations/$CONV_ID | jq

echo -e "\n=== Update System Prompt ==="
curl -s -X PATCH "$BASE_URL/conversations/$CONV_ID/system-prompt" \
  -H "Content-Type: application/json" \
  -d '{"content": "You are a pirate."}' | jq '.metadata.system_prompt'

echo -e "\n=== Stream Message ==="
echo "Streaming response (Ctrl+C to stop)..."
curl -N "$BASE_URL/conversations/$CONV_ID/stream?message=Hello"

echo -e "\n\n=== Delete Conversation ==="
curl -s -X DELETE $BASE_URL/conversations/$CONV_ID | jq

echo -e "\n=== Done ==="
```

Run with:
```bash
chmod +x test_manual.sh
./test_manual.sh
```

---

# forge-ui Testing Guide

## What Was Implemented

### UI Components

| Component | File | Description |
|-----------|------|-------------|
| App | `src/App.vue` | Main application wrapper |
| HeaderBar | `src/components/HeaderBar.vue` | Logo, connection status, toggles |
| WelcomeScreen | `src/components/WelcomeScreen.vue` | New chat, model select, import |
| ChatView | `src/views/ChatView.vue` | Main chat interface |
| MessageList | `src/components/MessageList.vue` | Scrollable message history |
| MessageBubble | `src/components/MessageBubble.vue` | Individual message with markdown |
| StreamingMessage | `src/components/StreamingMessage.vue` | Live streaming response |
| ToolCallCard | `src/components/ToolCallCard.vue` | Collapsible tool execution display |
| ChatInput | `src/components/ChatInput.vue` | Multi-line input with send/stop |
| ModelSelector | `src/components/ModelSelector.vue` | Model dropdown |
| SystemPromptEditor | `src/components/SystemPromptEditor.vue` | System prompt dialog |
| DebugPanel | `src/components/DebugPanel.vue` | Raw SSE event viewer |

### Composables

| Composable | File | Description |
|------------|------|-------------|
| useTheme | `src/composables/useTheme.ts` | Dark/light mode toggle |
| useSSE | `src/composables/useSSE.ts` | SSE connection management |
| useConversation | `src/composables/useConversation.ts` | Global conversation state |

---

## Prerequisites

```bash
cd /MyWork/Projects/agentic-forge/forge-ui

# Install dependencies (using Bun)
bun install

# Type check
bun run type-check

# Build (optional - for production)
bun run build
```

---

## Running the UI

### Development Mode

```bash
cd /MyWork/Projects/agentic-forge/forge-ui
bun run dev
```

**Expected:** Vite dev server starts on http://localhost:5173

### With Orchestrator (Mock Mode)

In separate terminals:

```bash
# Terminal 1: Start orchestrator in mock mode
cd /MyWork/Projects/agentic-forge/forge-orchestrator
ORCHESTRATOR_MOCK_LLM=true uv run orchestrator serve

# Terminal 2: Start UI
cd /MyWork/Projects/agentic-forge/forge-ui
bun run dev
```

---

## Manual UI Testing

### 1. Welcome Screen

1. Open http://localhost:5173
2. **Expected:** Welcome screen with:
   - "New Chat" button
   - Model selector dropdown
   - System prompt textarea (optional)
   - "Import Conversation" button

### 2. Create New Conversation

1. Select a model from dropdown
2. Optionally enter a system prompt
3. Click "New Chat"

**Expected:** Redirects to chat view with empty message list

### 3. Send a Message

1. Type a message in the input area
2. Click the Send button (or wait for orchestrator mock response)

**Expected:**
- User message appears on the right (blue bubble)
- Typing indicator shows
- Streaming response appears
- Assistant message appears on the left

### 4. Streaming Response

With mock mode:
1. Type "Hello" and send
2. Watch the streaming tokens appear

**Expected:**
- "Generating..." indicator in the assistant bubble
- Text streams in word by word
- Complete event finalizes the message

### 5. Tool Call Display

With mock mode:
1. Type "What's the weather?" and send

**Expected:**
- ToolCallCard appears showing `weather__get_current_weather`
- Card shows "pending" then "executing" status
- Card shows result with mock weather data
- Response mentions the weather

### 6. Dark/Light Mode Toggle

1. Click the sun/moon icon in the header

**Expected:** UI switches between dark and light themes

### 7. Basic/Advanced View Toggle

1. Click "Basic" / "Advanced" toggle in header

**In Advanced View, you should see:**
- Token usage per message
- Tool call latency (e.g., "150ms")
- Full tool names with server prefix
- Model name per message
- Debug panel toggle button
- Refresh tools button

### 8. Debug Panel (Advanced View)

1. Enable Advanced view
2. Click the code icon to show debug panel

**Expected:**
- Side panel opens on the right
- Shows raw SSE events as JSON
- Events color-coded by type
- Copy button for each event

### 9. System Prompt Editor

1. Click the edit icon in the chat toolbar
2. Modify the system prompt
3. Click Save

**Expected:**
- Dialog shows current prompt
- Shows version history (if any)
- Saving updates the conversation

### 10. Model Switching

1. Click the model dropdown in the header
2. Select a different model

**Expected:** Model updates immediately

### 11. Message Deletion (Advanced View)

1. Enable Advanced view
2. Hover over a message
3. Click the trash icon

**Expected:** Confirmation dialog, then message and all following deleted

### 12. Conversation Export

1. Click the download icon in chat toolbar

**Expected:** Downloads `conversation-{id}.json` file

### 13. Stop Generation

1. Send a message
2. Wait 2-3 seconds for stop button to appear
3. Click the stop button (or press Escape)

**Expected:** Generation stops, partial response discarded

### 14. Connection Status

1. Look at the header bar

**Expected:**
- Green dot when connected to orchestrator
- Red dot when disconnected
- Status text shows "Connected" or "Disconnected"

### 15. Draft Preservation

1. Type something in the input (don't send)
2. Refresh the page

**Expected:** Draft text is preserved in localStorage

---

## Environment Variables (UI)

Create `.env` file in `forge-ui/`:

```bash
VITE_API_URL=http://localhost:8001
```

---

## Full Integration Test

1. Start all services:

```bash
# Terminal 1: Armory (optional, for real tools)
cd /MyWork/Projects/agentic-forge/forge-armory
uv run armory serve

# Terminal 2: Orchestrator
cd /MyWork/Projects/agentic-forge/forge-orchestrator
export OPENROUTER_API_KEY="your-key"
uv run orchestrator serve

# Terminal 3: UI
cd /MyWork/Projects/agentic-forge/forge-ui
bun run dev
```

2. Open http://localhost:5173
3. Create a new conversation
4. Send messages and observe real LLM responses
5. Test tool calls if Armory is connected

---

## Known UI Limitations

1. **No conversation persistence** - Conversations not loaded from storage on refresh
2. **No conversation list** - Only one conversation visible at a time
3. **No responsive mobile layout** - Optimized for desktop
4. **Large bundle size** - Could benefit from code splitting
5. **No keyboard navigation** - Accessibility improvements needed
