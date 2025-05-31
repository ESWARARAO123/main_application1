# MCP Integration Detailed Flow

This document provides a detailed view of the data flow between components in the MCP chat integration.

## Component Interaction Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           Frontend (React)                               │
│                                                                         │
│  ┌───────────┐     ┌───────────┐      ┌──────────────┐     ┌─────────┐  │
│  │           │     │           │      │              │     │         │  │
│  │  Chatbot  │◄───►│ ChatInput │      │MCPNotification◄────┤MCPToggle│  │
│  │           │     │           │      │              │     │         │  │
│  └─────┬─────┘     └───────────┘      └──────┬───────┘     └─────────┘  │
│        │                                     │                          │
│        │                                     │                          │
│  ┌─────▼─────┐                        ┌─────▼────────┐                  │
│  │           │                        │              │                  │
│  │mcpChatSvc │                        │ MCPAgentCtx  │                  │
│  │           │                        │              │                  │
│  └─────┬─────┘                        └──────┬───────┘                  │
│        │                                     │                          │
│        └─────────────────┬───────────────────┘                          │
│                          │                                              │
│                    ┌─────▼─────┐                                        │
│                    │           │                                        │
│                    │ MCPContext│                                        │
│                    │           │                                        │
│                    └─────┬─────┘                                        │
│                          │                                              │
└──────────────────────────┼──────────────────────────────────────────────┘
                           │
                           │  REST API / WebSocket
                           │
┌──────────────────────────┼──────────────────────────────────────────────┐
│                          │                                              │
│                    ┌─────▼─────┐                                        │
│                    │           │                                        │
│                    │  mcp.js   │                                        │
│                    │           │                                        │
│                    └─────┬─────┘                                        │
│                          │                                              │
│               ┌──────────┴──────────┐                                   │
│               │                     │                                   │
│        ┌──────▼─────┐        ┌─────▼──────┐                            │
│        │            │        │            │                            │
│        │ollamaService│        │ mcpService │                            │
│        │            │        │            │                            │
│        └──────┬─────┘        └──────┬─────┘                            │
│               │                     │                                   │
│               │                     │                                   │
│        ┌──────▼─────────────┐ ┌────▼───────────────┐                   │
│        │                    │ │                    │                    │
│        │    Ollama API      │ │     MCP Server     │                    │
│        │                    │ │                    │                    │
│        └────────────────────┘ └────────────────────┘                    │
│                         Backend (Express)                               │
└─────────────────────────────────────────────────────────────────────────┘
```

## Data Flow Sequence

### 1. MCP Connection Initialization

```
┌─────────┐          ┌──────────┐          ┌─────────┐          ┌────────┐
│ Chatbot │          │MCPContext│          │ mcp.js  │          │MCP Srv │
└────┬────┘          └────┬─────┘          └────┬────┘          └────┬───┘
     │                     │                     │                    │
     │  Toggle MCP ON      │                     │                    │
     │ ─────────────────► │                     │                    │
     │                     │                     │                    │
     │                     │    Connect WebSocket │                    │
     │                     │ ──────────────────► │                    │
     │                     │                     │                    │
     │                     │                     │  WebSocket Connect │
     │                     │                     │ ──────────────────►│
     │                     │                     │                    │
     │                     │                     │ ◄─────────────────┐│
     │                     │                     │  Connection Accept ││
     │                     │                     │                    ││
     │                     │ ◄───────────────── │                    ││
     │                     │   ClientID Received │                    ││
     │                     │                     │                    │
     │ ◄─────────────────┐│                     │                    │
     │  MCP Ready Status  ││                     │                    │
     │                    ││                     │                    │
     │                    │                     │                    │
```

### 2. Sending a Message with MCP Enabled

```
┌───────┐     ┌────────────┐     ┌─────────┐     ┌──────┐     ┌───────┐
│ User  │     │ Chatbot    │     │mcpChatSvc│     │mcp.js│     │Ollama │
└───┬───┘     └─────┬──────┘     └────┬────┘     └───┬──┘     └───┬───┘
    │                │                 │              │            │
    │ Enter Message  │                 │              │            │
    │ ───────────────►                 │              │            │
    │                │                 │              │            │
    │                │ Send with MCP ID│              │            │
    │                │ ──────────────► │              │            │
    │                │                 │              │            │
    │                │                 │ POST /mcp/chat             │
    │                │                 │ ─────────────►             │
    │                │                 │              │            │
    │                │                 │              │ Process AI │
    │                │                 │              │ ──────────►│
    │                │                 │              │            │
    │                │                 │              │ ◄──────────│
    │                │                 │              │ Stream Data│
    │                │                 │              │            │
    │                │                 │ ◄───────────┐│            │
    │                │                 │ Stream Chunks││            │
    │                │                 │             ││            │
    │                │ ◄───────────────│             ││            │
    │                │ Update UI       │             ││            │
    │                │                 │             ││            │
    │ ◄──────────────│                 │             ││            │
    │ See Response   │                 │             ││            │
    │                │                 │             ││            │
```

### 3. Notification System Flow

```
┌───────────┐     ┌──────────┐     ┌────────────┐
│MCPContext │     │MCPAgentCtx│     │MCPNotification│
└─────┬─────┘     └─────┬────┘     └──────┬─────┘
      │                 │                 │
      │ Status Change   │                 │
      │ ──────────────► │                 │
      │                 │                 │
      │                 │ Update Notif.   │
      │                 │ ───────────────►│
      │                 │                 │
      │                 │                 │ Update UI
      │                 │                 │ ─────────▶
      │                 │                 │
```

## Key Data Structures

### MCP Client ID Flow

The MCP Client ID is a critical piece of information that flows through the system:

1. **Generation**: Created by the MCP Server upon WebSocket connection
2. **Storage**: Stored in MCPContext in React
3. **Usage**: Included in every chat request through mcpChatService
4. **Display**: Shown in the notification panel through MCPNotifications

### Message Structure

When sending a message with MCP enabled:

```json
{
  "modelId": "selected-model-id",
  "messages": [
    { "role": "user", "content": "user message" }
  ],
  "mcpClientId": "1234567890",
  "mcpServer": {
    "host": "mcp-server-host",
    "port": 8080
  },
  "options": {
    "stream": true
  }
}
```

### Response Structure

The streaming response follows this format:

```json
{
  "id": "chat-1234567890",
  "created": 1234567890,
  "model": "model-name",
  "choices": [
    {
      "index": 0,
      "delta": {
        "content": "partial response content"
      },
      "finish_reason": null
    }
  ]
}
```

## Integration Benefits

This architecture provides several benefits:

1. **Separation of Concerns**: MCP connection management is separated from chat functionality
2. **Fallback Capability**: If MCP fails, the system can fall back to regular chat
3. **Unified User Experience**: Chat works the same whether MCP is enabled or not
4. **Scalability**: New MCP features can be added without changing the core chat functionality

The modular design allows for future expansion and makes the system more maintainable. 