# MCP Integration Status Documentation

## Overview

This document provides a detailed analysis of the current state of Model Context Protocol (MCP) integration in our AI chatbot system. It addresses specific aspects of the implementation, focusing on the frontend components, backend integration, and Python capabilities.

## Frontend Implementation

### MCP Toggle & Configuration

#### Current Status

The MCP toggle functionality is implemented in the `Chatbot.tsx` component through the `useMCP` context hook. The relevant code sections show:

```jsx
// From Chatbot.tsx
const {
  isConnected: isMCPConnected,
  isMCPEnabled,
  toggleMCPEnabled,
  showServerSelector,
  setShowServerSelector,
  selectServer
} = useMCP();
```

The UI exposes this toggle in the `ChatInput` component:

```jsx
<ChatInput
  onSendMessage={handleSendMessage}
  isLoading={isLoading}
  isEmpty={isEmpty}
  isStreaming={isStreaming}
  isUploading={isUploading}
  uploadProgress={uploadProgress}
  onStopGeneration={handleStopGeneration}
  isRagAvailable={isRagAvailable}
  isRagEnabled={isRagEnabled}
  onToggleRag={toggleRagMode}
  isMCPAvailable={isMCPConnected}
  isMCPEnabled={isMCPEnabled}
  onToggleMCP={handleToggleMCP}
/>
```

The `MCPServerSelector` component provides a dialog for connecting to an MCP server:

```jsx
{/* MCP Server Selector */}
<MCPServerSelector
  isOpen={showServerSelector}
  onClose={() => setShowServerSelector(false)}
  onServerSelect={selectServer}
/>
```

When the MCP toggle is activated, it shows the server selector dialog, allowing users to enter the MCP server address. Upon successful connection, the system establishes an SSE connection and obtains a client ID, as shown in the output:

```
MCP Agent initialized. Connected to 172.16.16.54 at 172.16.16.54:8080. 16 tools available. ClientId: 1747821164970
```

Currently, toggling MCP enables the connection to the MCP server, but the integration with the AI chat functionality is still a work in progress. The system can successfully obtain a client ID but is not yet fully integrated with the AI chat flow.

### Normal Chat vs. Agentic Chat Mode

The system differentiates between normal chat and agentic chat through the `isAgentEnabled` state in the `MCPAgentContext`:

```jsx
// From Chatbot.tsx
const { isAgentEnabled, toggleAgent, processUserRequest } = useMCPAgent();

// Toggle handling
const handleToggleMCP = () => {
  // Toggle MCP in MCPContext
  toggleMCPEnabled();

  // Toggle the AI agent to match MCP state
  if (!isMCPEnabled) {
    // If we're enabling MCP, enable the agent too
    if (!isAgentEnabled) {
      toggleAgent();
    }
  } else {
    // If we're disabling MCP, disable the agent too
    if (isAgentEnabled) {
      toggleAgent();
    }
  }
};
```

When both `isMCPEnabled` and `isAgentEnabled` are true, the system will process user messages through the MCP agent rather than the regular AI flow:

```jsx
// From handleSendMessage in Chatbot.tsx
// If MCP Agent is enabled and there's no file, process the request with the agent
if (isMCPEnabled && isAgentEnabled && !file && content.trim() !== '') {
  try {
    // Process the request with the MCP Agent
    await processUserRequest(content.trim());
    return; // Exit early as the agent will handle the request
  } catch (error) {
    console.error('Error processing request with MCP Agent:', error);
    // Continue with normal processing if agent fails
  }
}
```

However, the MCP agent is currently not connected to the AI model in Ollama. This means that while MCP can execute tools, there is no AI assistance driving those tools. The system can toggle between normal chat mode (which uses Ollama) and agentic mode (which uses MCP tools), but these are currently separate systems.

### UI for Approve/Decline Actions

The system includes a component for displaying MCP agent commands in the `MessageList.tsx` component:

```jsx
{/* MCP Agent Commands and Results */}
<MCPAgentCommands />
```

The `MCPAgentCommands` component is responsible for rendering command proposals and their results in the chat interface. However, the specific UI for approve/decline actions is not fully implemented yet.

The current chat message rendering is handled by the `ChatMessage.tsx` component, which supports various content types including markdown, code blocks, file attachments, and RAG sources. The component structure suggests it can be extended to include action buttons:

```jsx
// From ChatMessage.tsx
<div style={isAI ? messageBubbleStyles.ai.content : messageBubbleStyles.user.content}>
  {/* Various content renderings */}
  {/* Potential location for rendering action buttons */}
</div>
```

Currently, there's no explicit implementation for rendering approve/decline buttons within chat messages. To add this functionality, you would need to:

1. Extend the message type to include a property for available actions
2. Add UI components in `ChatMessage.tsx` to render buttons conditionally
3. Implement callback handling in `Chatbot.tsx` to process approved or declined actions

## Backend Flow

### Endpoint Structure

The backend architecture includes separate routes for AI chat interactions and tool invocations:

1. **AI Chat Endpoints**: Handled primarily by `ollama.js` routes for Ollama integration
   - `/api/ollama/chat` for regular AI chat interactions
   - `/api/ollama/rag-chat` for RAG-enhanced conversations

2. **Chat Session Management**: Implemented in `chatbot.js`
   - `/api/chatbot/sessions/*` for managing chat sessions
   - `/api/chatbot/message` for storing messages in the database

However, there is no explicit endpoint yet for proxying tool invocations to MCP. This would typically be implemented as:
- A new endpoint like `/api/mcp/invoke` that would accept tool invocation requests
- Backend logic to manage MCP sessions and forward requests to the MCP server

Currently, it appears that MCP communication is happening directly from the frontend to the MCP server without going through your backend API.

### Expected Response Format

Based on the `MCPClient` implementation in the MCP protocol guide, the frontend expects tool invocation results in JSON format:

```javascript
// Example expected response format
{
  "id": "msg-1",
  "type": "tool_result",
  "content": {
    "content": [
      {
        "type": "text",
        "text": "Result content here"
      }
    ]
  }
}
```

This format matches the responses seen in the console output where MCP tools are invoked:

```
Command Result
3:22:55 PM
{
  "type": "text",
  "text": "MCP Agent initialized. Connected to 172.16.16.54 at 172.16.16.54:8080. 16 tools available. ClientId: 1747821164970"
}
```

The frontend is designed to handle SSE connections for streaming responses from the MCP server, but this is managed directly by the frontend rather than proxied through your backend.

## Session Management

### SSE Connection Lifecycle

The SSE connection to the MCP server is established when the user toggles on the MCP feature and selects a server. Based on the provided code samples and output, the connection is initialized as follows:

1. User toggles on MCP in the UI
2. Server selector dialog appears
3. User enters MCP server address (e.g., 172.16.16.54:8080)
4. Frontend initiates an SSE connection to `http://<server>/sse`
5. MCP server responds with a client ID
6. Frontend stores this client ID for subsequent tool invocations

The SSE connection appears to be maintained for the duration of the user's session or until MCP is toggled off. However, there is no explicit indication of implemented reconnection logic or heartbeats in the current code.

### Client ID Management

Based on the available information, the client ID is currently managed in frontend state rather than being stored in the database:

1. When the SSE connection is established, the client ID is stored in the `MCPContext`
2. This client ID is then used for all subsequent tool invocations
3. There is no apparent mapping of user IDs to client IDs in the database

To enhance session persistence across page reloads, you would need to:
1. Store the client ID in the database, associated with the user account
2. Implement reconnection logic that retrieves the stored client ID on page load
3. Add heartbeat mechanisms to detect and recover from broken connections

## Python Integration

### Current Python Implementation

The Python environment (venv) in your system is currently primarily used for document processing in the RAG pipeline, particularly for extracting text from PDFs using `pdfplumber`. The Python-related files in the `/python` directory include:

- `extract_text.py` - Basic text extraction from documents
- `extract_text_with_tables.py` - Enhanced text extraction that handles tables
- `requirements.txt` - Python dependencies
- `installvenv.sh` - Script for setting up the Python virtual environment
- `test_extraction.sh` - Testing script for the extraction functionality

The current architecture spawns Python processes for document processing:
```javascript
// Pseudocode based on typical implementation
const processDocument = async (documentId) => {
  // Spawn Python process to extract text
  const result = await spawnPythonProcess('python/extract_text.py', [documentPath]);
  // Process the extracted text for RAG
};
```

### Potential MCP Python Integration

The MCP server already includes a `runPythonFile` tool that could execute Python scripts:

```json
{
  "name": "runPythonFile",
  "description": "Execute a Python file and return output",
  "parameters": {
    "filePath": {"type": "ZodString", "description": "Path to the Python file to execute", "required": true},
    "args": {"type": "ZodOptional", "description": "Optional arguments to pass to the Python script", "required": true}
  }
}
```

This tool could be leveraged to run your existing Python scripts through MCP, providing a more standardized interface. For example:

```javascript
// Using MCP to run Python script
const result = await mcpClient.invoke_tool("runPythonFile", {
  filePath: "python/extract_text.py",
  args: ["--input", documentPath, "--output", outputPath]
});
```

To fully integrate Python tools with MCP, you would need to:
1. Ensure the Python virtual environment is accessible to the MCP server
2. Adapt your Python scripts to work as MCP tools (proper input/output handling)
3. Update your frontend to invoke these tools through the MCP interface

## Next Steps for Full MCP Integration

Based on the current implementation status, here are the recommended next steps to fully integrate MCP with your AI chatbot system:

1. **Complete the UI for Command Approval**:
   - Implement approve/decline buttons in chat messages
   - Add callback handlers for user decisions

2. **Develop Backend MCP Proxy**:
   - Create API endpoints to proxy MCP tool invocations
   - Implement session management for MCP connections

3. **Connect MCP with AI Model**:
   - Integrate the Ollama AI model with MCP for intelligent tool selection
   - Implement a workflow where AI suggests tools and user approves/declines

4. **Enhance Session Persistence**:
   - Store MCP client IDs in the database
   - Implement reconnection logic and heartbeats
   - Add error handling for connection issues

5. **Integrate Python Tools**:
   - Adapt existing Python scripts to work as MCP tools
   - Develop new specialized Python tools for data analysis

## Conclusion

The MCP integration in your AI chatbot system has made significant progress, with the ability to toggle MCP, connect to a server, and obtain a client ID. However, several key components still need to be implemented to achieve a fully functional integration.

The current architecture provides a solid foundation for these enhancements, with clean separation between the Ollama AI chat functionality and the emerging MCP agent capabilities. By building on this foundation and following the recommended next steps, you can create a powerful agentic chat experience that combines AI intelligence with tool execution capabilities. 