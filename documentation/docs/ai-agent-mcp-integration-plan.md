# AI Agent MCP Integration Plan

## Detailed User Flow

1. **User Sends a Message**
   - User types a message in the chat interface
   - Message is sent to the AI model (Ollama)

2. **AI Analyzes and Responds**
   - AI processes the message and generates a response
   - AI identifies if the task requires using an MCP tool
   - AI includes a suggested command in its response

3. **Command Suggestion Display**
   - The suggested command appears in the chat as part of the AI's message
   - Command is displayed in a terminal-like format
   - "Run" and "Cancel" buttons are embedded in the message

4. **User Approves Command**
   - User reviews the suggested command
   - User clicks "Run" to approve or "Cancel" to reject

5. **Command Execution**
   - Approved command is executed via Python orchestrator
   - Execution status is shown in the same chat message
   - Results are streamed back and displayed in the same message

6. **AI Analyzes Results**
   - AI receives the command execution results
   - AI analyzes the results and provides further clarification
   - AI may suggest additional commands based on the results

## Current System Analysis

### Existing Components

1. **React Chat Application**
   - Uses Ollama AI models for chat
   - Has MCP toggle functionality (`MCPContext.tsx`)
   - Has agent toggle functionality (`MCPAgentContext.tsx`)
   - Supports streaming responses

2. **MCP Integration**
   - Connection to MCP server via WebSocket/SSE
   - Client ID management
   - Basic tool execution capabilities

3. **Python Orchestrator**
   - Command-line tool for executing MCP server tools
   - Located in `python/terminal-mcp-orchestrator/`
   - Uses `mcp_client.py` to connect to MCP server

4. **Backend Services**
   - Node.js/Express backend
   - Already has capability to spawn Python processes
   - Configuration for Python interpreter in `conf/config.ini`

## Implementation Approach

### 1. AI Command Suggestion Integration

The key is to have the AI model (Ollama) suggest commands as part of its response. This requires:

1. **Prompt Engineering**:
   - Modify the system prompt sent to Ollama to include instructions about suggesting commands
   - Include available tools and their parameters in the context

2. **Response Parsing**:
   - Parse AI responses to identify command suggestions
   - Extract tool name and parameters from the suggestion
   - Format the command for display in the chat

```javascript
// Example system prompt addition
const systemPrompt = `
... [existing prompt] ...

When a task requires using a tool, suggest a command in the following format:

<command>
tool: [tool_name]
parameters: {
  "param1": "value1",
  "param2": "value2"
}
description: Brief description of what this command will do
</command>

Available tools:
${JSON.stringify(availableTools, null, 2)}
`;
```

### 2. Command Execution Flow

#### Frontend Flow
1. User sends a message with MCP and Agent enabled
2. AI responds with text that includes command suggestions
3. Frontend parses the response and renders command UI with buttons
4. User clicks "Run" to approve the command
5. Command is sent to backend for execution
6. Results are streamed back to the same chat message
7. AI receives the results and provides further analysis

#### Backend Flow
1. Receive approved command from frontend
2. Spawn Python orchestrator process
3. Execute command via MCP server
4. Stream results back to frontend
5. Send results to AI for analysis

```javascript
// Example backend endpoint for command execution
app.post('/api/mcp/agent/execute-command', async (req, res) => {
  try {
    const { messageId, toolName, parameters, mcpServer } = req.body;

    // Get Python interpreter path from config
    const { pythonInterpreter } = getPythonConfig();

    // Prepare orchestrator command
    const orchestratorPath = path.resolve(process.cwd(),
      './python/terminal-mcp-orchestrator/orchestrator.py');

    // Format parameters as JSON string
    const paramsJson = JSON.stringify(parameters);

    // Set up SSE for streaming results
    res.setHeader('Content-Type', 'text/event-stream');
    res.setHeader('Cache-Control', 'no-cache');
    res.setHeader('Connection', 'keep-alive');

    // Spawn Python process
    const pythonProcess = spawn(pythonInterpreter, [
      orchestratorPath,
      '--server', mcpServer.host + ':' + mcpServer.port,
      toolName,
      paramsJson
    ]);

    // Stream stdout in real-time
    pythonProcess.stdout.on('data', (data) => {
      res.write(`data: ${JSON.stringify({type: 'output', content: data.toString()})}\n\n`);
    });

    // Stream stderr in real-time
    pythonProcess.stderr.on('data', (data) => {
      res.write(`data: ${JSON.stringify({type: 'error', content: data.toString()})}\n\n`);
    });

    // Handle process completion
    pythonProcess.on('close', (code) => {
      res.write(`data: ${JSON.stringify({type: 'complete', exitCode: code})}\n\n`);
      res.end();
    });

  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});
```

### 3. Command UI in Chat Messages

The key innovation is embedding the command UI directly in the chat message:

```jsx
// Example React component for rendering AI messages with embedded commands
const AIMessage = ({ message }) => {
  // Parse message content to find command suggestions
  const { textContent, commands } = parseMessageForCommands(message.content);

  return (
    <div className="ai-message">
      {/* Render regular text content */}
      <div className="message-text">{textContent}</div>

      {/* Render command suggestions if any */}
      {commands.map((command, index) => (
        <CommandWidget
          key={index}
          command={command}
          messageId={message.id}
          status={command.status || 'pending'}
          result={command.result}
        />
      ))}
    </div>
  );
};

// Command widget with approval buttons and result display
const CommandWidget = ({ command, messageId, status, result, onApprove, onReject }) => {
  return (
    <div className="command-widget">
      {/* Command preview */}
      <div className="command-preview">
        <div className="command-header">
          <span className="tool-name">{command.toolName}</span>
        </div>
        <pre className="command-code">{JSON.stringify(command.parameters, null, 2)}</pre>
        <div className="command-description">{command.description}</div>
      </div>

      {/* Action buttons - only show if status is pending */}
      {status === 'pending' && (
        <div className="command-actions">
          <button
            className="run-button"
            onClick={() => onApprove(messageId, command)}
          >
            Run
          </button>
          <button
            className="cancel-button"
            onClick={() => onReject(messageId, command)}
          >
            Cancel
          </button>
        </div>
      )}

      {/* Execution status */}
      {status === 'running' && (
        <div className="execution-status">
          <span className="spinner"></span>
          <span>Executing command...</span>
        </div>
      )}

      {/* Result display */}
      {status === 'completed' && result && (
        <div className="command-result">
          <div className="result-header">Result:</div>
          <pre className="result-content">{
            typeof result === 'object'
              ? JSON.stringify(result, null, 2)
              : result
          }</pre>
        </div>
      )}

      {/* Error display */}
      {status === 'error' && result && (
        <div className="command-error">
          <div className="error-header">Error:</div>
          <pre className="error-content">{result.error || result}</pre>
        </div>
      )}
    </div>
  );
};
```

### 4. Result Analysis and AI Follow-up

After command execution, the results need to be sent back to the AI for analysis:

```javascript
// Example function to send results to AI for analysis
const analyzeCommandResult = async (originalMessage, command, result) => {
  try {
    // Prepare the prompt for AI analysis
    const analysisPrompt = `
The user asked: "${originalMessage}"

I suggested running this command:
Tool: ${command.toolName}
Parameters: ${JSON.stringify(command.parameters, null, 2)}

The command execution resulted in:
${typeof result === 'object' ? JSON.stringify(result, null, 2) : result}

Please analyze these results and provide a helpful explanation to the user.
If further actions are needed, suggest additional commands in the proper format.
`;

    // Send to Ollama for analysis
    const analysisResponse = await ollamaService.generateCompletion({
      model: selectedModel,
      prompt: analysisPrompt,
      options: { temperature: 0.2 }
    });

    return analysisResponse.text;
  } catch (error) {
    console.error('Error analyzing command result:', error);
    return 'I encountered an error while analyzing the command results.';
  }
};
```

## Implementation Plan

### Phase 1: Backend Integration

1. **Create Python Orchestrator Wrapper**
   - Create a Node.js service to spawn Python orchestrator processes
   - Implement streaming response handling
   - Add error handling and timeout management

2. **Implement Command Execution Endpoint**
   - Create `/api/mcp/agent/execute-command` endpoint
   - Configure SSE for streaming results
   - Spawn Python orchestrator process with parameters from config.ini

3. **Implement Result Analysis**
   - Create function to send command results to AI
   - Format results for AI analysis
   - Parse AI response for follow-up suggestions

### Phase 2: Frontend Enhancement

1. **Enhance Message Parsing**
   - Implement parser to detect command suggestions in AI responses
   - Extract tool name, parameters, and description
   - Create command objects for UI rendering

2. **Create Command UI Components**
   - Implement `CommandWidget` component for displaying commands
   - Add approval/rejection buttons
   - Create status indicators and result display
   - Style components to match chat interface

3. **Integrate with Chat Flow**
   - Update message rendering to include command widgets
   - Implement command approval/rejection handlers
   - Add result display in the same message
   - Handle AI follow-up responses

### Phase 3: AI Integration

1. **Modify System Prompt**
   - Update system prompt to include command suggestion format
   - Add available tools and their parameters
   - Include examples of proper command suggestions

2. **Implement Result Analysis Prompting**
   - Create prompts for AI to analyze command results
   - Include original request and command context
   - Format results for AI consumption

## Technical Considerations

### 1. Security

- **Parameter Validation**: Validate all parameters before execution
  ```javascript
  // Example validation function
  const validateParameters = (toolName, parameters) => {
    // Get tool schema
    const toolSchema = availableTools.find(t => t.name === toolName);
    if (!toolSchema) return { valid: false, error: 'Unknown tool' };

    // Validate required parameters
    for (const param of toolSchema.requiredParams || []) {
      if (parameters[param] === undefined) {
        return { valid: false, error: `Missing required parameter: ${param}` };
      }
    }

    return { valid: true };
  };
  ```

- **Command Approval**: Require explicit user approval before execution
- **Timeout Limits**: Implement timeouts for long-running commands

### 2. Error Handling

- **Process Failures**: Handle Python process crashes gracefully
  ```javascript
  pythonProcess.on('error', (error) => {
    res.write(`data: ${JSON.stringify({
      type: 'error',
      content: `Failed to start process: ${error.message}`
    })}\n\n`);
    res.end();
  });
  ```

- **Clear Error Messages**: Format errors for user understanding
- **Retry Mechanism**: Allow retrying failed commands

### 3. User Experience

- **Real-time Feedback**: Show command execution progress
- **Formatted Results**: Present results in a readable format
- **Context Preservation**: Maintain conversation context after command execution

## Implementation Steps

1. **Create Backend Endpoints**
   - Implement `/api/mcp/agent/execute-command` endpoint
   - Configure Python process spawning
   - Set up SSE for streaming results

2. **Enhance Frontend Components**
   - Create command parsing utilities
   - Implement command UI components
   - Update message rendering

3. **Modify AI Integration**
   - Update system prompt
   - Implement result analysis

4. **Test and Refine**
   - Test with various command scenarios
   - Refine UI based on user feedback
   - Optimize performance

## Next Steps

1. Create detailed technical specifications for each component
2. Implement Python orchestrator wrapper service
3. Create command generation and execution endpoints
4. Enhance frontend with command approval UI
5. Integrate AI for command generation and result analysis
6. Test with sample user scenarios
7. Refine based on user feedback
