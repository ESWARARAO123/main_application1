# MCP Chat Integration

This document explains how the Model Context Protocol (MCP) is integrated with the chat system. The MCP integration allows the chat application to maintain a connection to an MCP server while still using normal AI chat functionality.

## Overview

The MCP integration allows users to:
1. Connect to an MCP server via WebSocket
2. Maintain an MCP client ID for server communication
3. Send and receive AI chat messages through the MCP proxy
4. View connection status and notifications in a dedicated panel

## Architecture

```
┌─────────────────┐      ┌──────────────────┐      ┌────────────────┐
│                 │      │                  │      │                │
│  React Frontend │◄────►│  Express Backend │◄────►│  Ollama Server │
│                 │      │                  │      │                │
└────────┬────────┘      └────────┬─────────┘      └────────────────┘
         │                        │                         
         │                        │                         
         ▼                        ▼                         
┌─────────────────┐      ┌──────────────────┐      ┌────────────────┐
│                 │      │                  │      │                │
│   MCP Context   │◄────►│   MCP Services   │◄────►│   MCP Server   │
│                 │      │                  │      │                │
└─────────────────┘      └──────────────────┘      └────────────────┘
```

## Key Components

### Frontend Components

1. **MCPContext** (`client/src/contexts/MCPContext.tsx`)
   - Manages MCP connection state
   - Stores client ID and server information
   - Provides connection functions to components

2. **MCPAgentContext** (`client/src/contexts/MCPAgentContext.tsx`)
   - Manages MCP notification system
   - Tracks client ID status and connection information
   - Handles notification storage and display

3. **MCPNotifications** (`client/src/components/mcp/MCPNotifications.tsx`)
   - Displays connection status and notification panel
   - Shows client ID information
   - Offers connection management options

4. **mcpChatService** (`client/src/services/mcpChatService.ts`)
   - Handles communication with the backend MCP chat endpoint
   - Formats requests with MCP client ID information
   - Manages streaming responses from the AI model

5. **Chatbot** (`client/src/pages/Chatbot.tsx`)
   - Integrates MCP with the main chat interface
   - Routes messages through MCP when enabled
   - Handles errors and fallbacks

### Backend Components

1. **mcp.js** (`src/routes/mcp.js`)
   - Provides MCP API endpoints
   - Handles chat proxying to Ollama
   - Manages MCP server connections

2. **ollamaService.js** (`src/services/ollamaService.js`)
   - Connects to the Ollama API
   - Processes chat messages and streams responses
   - Manages model selection and settings

## Message Flow

When MCP is enabled and a user sends a message:

```
User Message → Chatbot Component → mcpChatService → Backend MCP Chat Endpoint → 
Ollama Service → Ollama API → Streamed Response → Frontend Display
```

With more detail:

1. User enters a message in the chat input field
2. The Chatbot component processes the message
3. If MCP is enabled:
   - The message is routed through mcpChatService
   - MCP client ID and server info are included
4. The backend `/mcp/chat` endpoint:
   - Receives the message with MCP information
   - Formats it for the Ollama service
   - Processes it through the selected AI model
5. Responses are streamed back through the same path
6. The content is displayed in the chat interface

## Connection Management

The MCP connection cycle works as follows:

```
┌─────────────────┐     ┌───────────────────┐     ┌─────────────────┐
│  Toggle MCP On  │────►│ WebSocket Connect │────►│ Get Client ID   │
└─────────────────┘     └───────────────────┘     └────────┬────────┘
                                                           │
┌─────────────────┐     ┌───────────────────┐     ┌────────▼────────┐
│  Use in Chat    │◄────┤ Store Client ID   │◄────┤ Show in Notifications │
└─────────────────┘     └───────────────────┘     └─────────────────┘
```

## Notification System

The notification system provides:
1. Connection status updates
2. Client ID information
3. Error messages and recovery options

The notification panel is separate from the chat flow, allowing chat to function normally while MCP connection information is maintained independently.

## Error Handling

The system includes robust error handling:
1. Connection failures trigger notifications
2. Client ID recovery mechanisms
3. Fallback to standard chat if MCP fails
4. Detailed error logging for troubleshooting

## Implementation Details

### MCP Toggle Functionality

When a user toggles MCP on:
1. The WebSocket connection is established
2. An MCP client ID is generated and stored
3. The notification panel shows connection status
4. Chat messages are routed through the MCP proxy

When MCP is off, the standard chat flow is used without MCP client ID information.

### Chat Proxying

The MCP chat proxy (`/mcp/chat` endpoint) works by:
1. Accepting messages with an MCP client ID
2. Using the same Ollama service as regular chat
3. Supporting both streaming and non-streaming responses
4. Maintaining proper error handling

This allows chat to work identically with or without MCP enabled, with the only difference being the inclusion of MCP client ID information in requests.

## Configuration

The MCP integration can be configured in the following ways:
1. Server connections can be managed through the UI
2. Default connections can be stored between sessions
3. Streaming behavior matches regular chat settings
4. Error recovery mechanisms can be controlled through UI buttons

## Conclusion

The MCP chat integration provides a seamless way to maintain MCP connections while using the chat system normally. The separation of concerns between chat functionality and MCP connection management allows for a flexible system that can handle different use cases and failure modes. 