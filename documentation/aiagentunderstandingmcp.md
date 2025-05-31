# AI Agent Integration with MCP Server

## Overview

This document outlines our approach for integrating AI agents with the Model Context Protocol (MCP) server. Instead of using cloud-based AI APIs (like OpenAI), we'll leverage the existing Ollama models available in our application.

## Architecture

The architecture involves the following components:

1. **User Interface (Chat)**: Where users submit queries and interact with the agent
2. **Ollama LLM**: The local AI model that interprets user requests and generates responses
3. **MCP Client**: A middleware that connects to the MCP server and executes tools
4. **MCP Server**: Hosts various tools that can be executed via HTTP requests
5. **Command Approval System**: UI elements that allow users to approve/decline proposed commands

## Flow

1. User submits a query to the chatbot (with AI agent toggle enabled)
2. Ollama model processes the query and determines which MCP tool(s) might be needed
3. The model generates a "proposed command" that would invoke the appropriate MCP tool
4. UI displays this proposed command to the user with approval buttons
5. If approved, the frontend sends the command to the MCP client
6. MCP client:
   - Establishes SSE connection (if not already connected)
   - Obtains a client ID
   - Invokes the requested tool
   - Returns results back to the UI
7. Results are displayed in the chat
8. Ollama model interprets the results and provides a final response

## Current Codebase Analysis

After reviewing the codebase, we've discovered that most of the infrastructure we need is already in place:

1. **MCPContext (client/src/contexts/MCPContext.tsx)**
   - Handles MCP server configuration and connection
   - Provides methods to execute commands and manage tools
   - Already implements SSE connection handling via WebSocket

2. **MCPAgentContext (client/src/contexts/MCPAgentContext.tsx)**
   - Manages the agent state (enabled/disabled)
   - Handles pending commands and command results
   - Has approval workflow for commands
   - Has analysis mechanism for results

3. **UI Components**
   - MCPServerSelector.tsx - Allows users to select and connect to MCP servers
   - MCPCommandResult.tsx - Displays the results of executed commands
   - MCPAgentCommands.tsx - Component for showing agent-related commands

## What's Missing

1. **Integration with Ollama Model**
   - We need to create a system prompt that teaches the Ollama model how to use MCP tools
   - The model needs to format its responses in a way that can be parsed to extract tool calls

2. **Command Generation and Extraction**
   - Need logic to extract tool calls from AI responses
   - Parse parameters from the extracted tool calls

3. **Command Approval UI**
   - Enhance the UI to show tool calls awaiting approval
   - Allow users to modify parameters before execution

## Implementation Plan

### 1. Enhance MCPAgentContext

We need to modify the `processUserRequest` method in MCPAgentContext to:
1. Generate a system prompt that explains available tools to the Ollama model
2. Extract tool calls from the model response
3. Create pending commands for user approval

```typescript
// New methods to add to MCPAgentContext:

const createToolSystemPrompt = () => {
  // Create a system prompt that teaches the model about available tools
  const toolDescriptions = availableTools.map(tool => {
    const paramDesc = Object.entries(tool.parameters || {})
      .map(([key, param]) => {
        const isRequired = param.required ? ' (Required)' : '';
        return `    - ${key}${isRequired}: ${param.description || 'No description'}`;
      })
      .join('\n');
    
    return `${tool.name}: ${tool.description}
Parameters:
${paramDesc}`;
  }).join('\n\n');

  return `
You are a helpful AI assistant that can use tools to help users. When you need to use a tool, follow this format:

THINK: [Your reasoning about what tool to use]
TOOL: {
  "tool": "toolName",
  "parameters": {
    "param1": "value1",
    "param2": "value2"
  }
}

After you receive the tool result, provide your final answer.

Available tools:
${toolDescriptions}

Example:
User: What files are in the current directory?
Assistant:
THINK: I need to check the contents of the current directory to answer this question.
TOOL: {
  "tool": "readDirectory",
  "parameters": {
    "dirPath": "."
  }
}

[After receiving tool result]
The current directory contains the following files and directories: [list based on result]
`;
};

const extractToolCall = (text) => {
  try {
    // Look for TOOL: marker followed by JSON
    const toolMatch = text.match(/TOOL:\s*(\{[\s\S]*?\})/);
    if (toolMatch && toolMatch[1]) {
      const jsonStr = toolMatch[1];
      const toolCall = JSON.parse(jsonStr);
      
      if (toolCall.tool && toolCall.parameters) {
        return {
          toolName: toolCall.tool,
          parameters: toolCall.parameters
        };
      }
    }
    return null;
  } catch (e) {
    console.error('Error extracting tool call:', e);
    return null;
  }
};
```

### 2. Modify the existing processUserRequest method

```typescript
const processUserRequest = async (request: string, chatCallback?: (response: string) => void) => {
  if (!isAgentEnabled || isProcessing) {
    console.log('Agent is disabled or already processing');
    return;
  }

  setIsProcessing(true);
  
  try {
    // Check for a valid MCP connection
    if (!mcpConnection.clientId) {
      const connectionSuccess = await checkAndEstablishMCPConnection();
      if (!connectionSuccess) {
        if (chatCallback) {
          chatCallback("I couldn't connect to the MCP server. Please check your connection and try again.");
        }
        setIsProcessing(false);
        return;
      }
    }
    
    // Generate the system prompt with available tools
    const systemPrompt = createToolSystemPrompt();
    
    // Query the Ollama model
    const response = await axios.post('/api/ollama/chat', {
      prompt: request,
      systemPrompt,
      model: activeModel || 'llama3'
    });
    
    const aiResponse = response.data.response;
    
    // Extract any tool calls
    const toolCall = extractToolCall(aiResponse);
    
    if (toolCall) {
      // Create a pending command
      const commandId = `cmd-${Date.now()}`;
      const pendingCommand = {
        id: commandId,
        command: request,
        toolName: toolCall.toolName,
        parameters: toolCall.parameters,
        description: `The AI wants to use the ${toolCall.toolName} tool to help answer your question.`,
        timestamp: Date.now()
      };
      
      // Add to pending commands
      setPendingCommands(prev => [...prev, pendingCommand]);
      
      // Extract the thinking part to show to the user
      const thinking = aiResponse.match(/THINK:\s*([\s\S]*?)(?=TOOL:)/);
      const thinkingText = thinking ? thinking[1].trim() : '';
      
      // Notify the chat
      if (chatCallback) {
        let userMessage = `I need to use a tool to help answer your question.\n\n`;
        if (thinkingText) {
          userMessage += `${thinkingText}\n\n`;
        }
        userMessage += `[Waiting for your approval to run the ${toolCall.toolName} tool]`;
        
        chatCallback(userMessage);
      }
    } else {
      // No tool call, just return the response
      if (chatCallback) {
        chatCallback(aiResponse);
      }
    }
  } catch (error) {
    console.error('Error processing user request:', error);
    
    if (chatCallback) {
      chatCallback("I encountered an error while processing your request. Please try again.");
    }
  } finally {
    setIsProcessing(false);
  }
};
```

### 3. Create a Component for Pending Commands

Let's create a new component to display pending commands that need user approval:

```tsx
// client/src/components/chat/MCPPendingCommand.tsx
import React, { useState } from 'react';
import { useMCPAgent } from '../../contexts/MCPAgentContext';
import {
  CheckCircleIcon,
  XCircleIcon,
  PencilIcon,
  CodeBracketIcon
} from '@heroicons/react/24/outline';

interface MCPPendingCommandProps {
  commandId: string;
}

const MCPPendingCommand: React.FC<MCPPendingCommandProps> = ({ commandId }) => {
  const { pendingCommands, approveCommand, rejectCommand, modifyCommand } = useMCPAgent();
  const [isEditing, setIsEditing] = useState(false);
  const [editedParams, setEditedParams] = useState<any>({});
  
  // Find the command
  const command = pendingCommands.find(cmd => cmd.id === commandId);
  
  if (!command) {
    return null;
  }
  
  const handleApprove = () => {
    approveCommand(commandId);
  };
  
  const handleReject = () => {
    rejectCommand(commandId);
  };
  
  const handleEdit = () => {
    setEditedParams(command.parameters);
    setIsEditing(true);
  };
  
  const handleSave = () => {
    modifyCommand(commandId, editedParams);
    setIsEditing(false);
  };
  
  const handleCancel = () => {
    setIsEditing(false);
  };
  
  return (
    <div 
      className="rounded-lg border p-4 my-3 transition-all"
      style={{ 
        backgroundColor: 'var(--color-surface-light)', 
        borderColor: 'var(--color-warning-light)',
        borderLeft: '3px solid var(--color-warning)'
      }}
    >
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center">
          <CodeBracketIcon className="h-5 w-5 mr-2" style={{ color: 'var(--color-warning)' }} />
          <h4 className="font-medium" style={{ color: 'var(--color-text)' }}>
            Pending Command: {command.toolName}
          </h4>
        </div>
      </div>
      
      <div className="mb-4">
        <p className="text-sm mb-2" style={{ color: 'var(--color-text)' }}>
          {command.description}
        </p>
        
        {isEditing ? (
          <div className="mb-4">
            <textarea
              value={JSON.stringify(editedParams, null, 2)}
              onChange={(e) => {
                try {
                  setEditedParams(JSON.parse(e.target.value));
                } catch (error) {
                  // Invalid JSON, don't update
                }
              }}
              className="w-full p-2 font-mono text-sm rounded"
              style={{
                backgroundColor: 'var(--color-surface-dark)',
                color: 'var(--color-text)',
                border: '1px solid var(--color-border)'
              }}
              rows={Object.keys(editedParams).length + 2}
            />
            
            <div className="flex mt-2">
              <button
                onClick={handleSave}
                className="px-3 py-1 rounded mr-2 flex items-center text-xs"
                style={{
                  backgroundColor: 'var(--color-success-bg)',
                  color: 'var(--color-success)'
                }}
              >
                <CheckCircleIcon className="h-4 w-4 mr-1" />
                Save Changes
              </button>
              
              <button
                onClick={handleCancel}
                className="px-3 py-1 rounded flex items-center text-xs"
                style={{
                  backgroundColor: 'var(--color-surface-dark)',
                  color: 'var(--color-text-muted)'
                }}
              >
                <XCircleIcon className="h-4 w-4 mr-1" />
                Cancel
              </button>
            </div>
          </div>
        ) : (
          <div
            className="p-3 rounded font-mono text-sm overflow-auto mb-3"
            style={{
              backgroundColor: 'var(--color-surface-dark)',
              color: 'var(--color-text)'
            }}
          >
            <pre className="whitespace-pre-wrap break-words"><code>{JSON.stringify(command.parameters, null, 2)}</code></pre>
          </div>
        )}
      </div>
      
      <div className="flex items-center space-x-2">
        <button
          onClick={handleApprove}
          className="px-4 py-2 rounded flex items-center text-sm"
          style={{
            backgroundColor: 'var(--color-success-bg)',
            color: 'var(--color-success)'
          }}
        >
          <CheckCircleIcon className="h-4 w-4 mr-1" />
          Approve
        </button>
        
        <button
          onClick={handleReject}
          className="px-4 py-2 rounded flex items-center text-sm"
          style={{
            backgroundColor: 'var(--color-error-bg)',
            color: 'var(--color-error)'
          }}
        >
          <XCircleIcon className="h-4 w-4 mr-1" />
          Reject
        </button>
        
        {!isEditing && (
          <button
            onClick={handleEdit}
            className="px-4 py-2 rounded flex items-center text-sm"
            style={{
              backgroundColor: 'var(--color-surface-dark)',
              color: 'var(--color-text-muted)'
            }}
          >
            <PencilIcon className="h-4 w-4 mr-1" />
            Edit Parameters
          </button>
        )}
      </div>
    </div>
  );
};

export default MCPPendingCommand;
```

### 4. Update MCPAgentCommands Component

Now let's update the existing MCPAgentCommands.tsx to show pending commands:

```tsx
// Update in client/src/components/chat/MCPAgentCommands.tsx
import React, { useEffect, useState } from 'react';
import { useMCPAgent } from '../../contexts/MCPAgentContext';
import { useMCP } from '../../contexts/MCPContext';
import MCPCommandResult from './MCPCommandResult';
import MCPPendingCommand from './MCPPendingCommand';
import axios from 'axios';

// Add interface for AI models
interface AIModel {
  id: string;
  name: string;
  model_id: string;
  ollama_model_id: string;
  is_active: boolean;
  description: string;
}

const MCPAgentCommands: React.FC = () => {
  const { 
    pendingCommands, 
    commandResults, 
    isAgentEnabled,
    isProcessing,
    sseStatus,
    addCommandResult,
    isAnalyzing,
    isStopped
  } = useMCPAgent();
  
  const { 
    availableTools,
    mcpConnection
  } = useMCP();

  // Add state for active models
  const [activeModels, setActiveModels] = useState<AIModel[]>([]);
  const [modelsLoading, setModelsLoading] = useState<boolean>(false);
  const [modelsError, setModelsError] = useState<string | null>(null);

  // Fetch active models when component mounts
  useEffect(() => {
    const fetchActiveModels = async () => {
      try {
        setModelsLoading(true);
        setModelsError(null);
        const response = await axios.get('/api/ollama/models/active');
        if (response.data && response.data.models) {
          setActiveModels(response.data.models);
        }
      } catch (error) {
        console.error('Error fetching active models:', error);
        setModelsError('Failed to fetch active AI models. Some features may be limited.');
      } finally {
        setModelsLoading(false);
      }
    };

    if (isAgentEnabled) {
      fetchActiveModels();
    }
  }, [isAgentEnabled]);

  // Don't render anything if the agent is not enabled
  if (!isAgentEnabled) {
    return null;
  }
  
  // Determine if there's any active processing happening
  const isActive = isProcessing || isAnalyzing;

  return (
    <div className="mcp-agent-commands">
      {/* Model error warning */}
      {modelsError && (
        <div className="mb-4 p-3 rounded-lg border border-warning-300"
          style={{ 
            backgroundColor: 'var(--color-warning-bg)',
            color: 'var(--color-warning)'
          }}>
          <p className="text-sm">{modelsError}</p>
        </div>
      )}

      {/* Show pending commands waiting for approval */}
      <div className="pending-commands">
        {pendingCommands.map(command => (
          <MCPPendingCommand
            key={command.id}
            commandId={command.id}
          />
        ))}
      </div>

      {/* Show tool results */}
      <div className="command-results">
        {commandResults.map(result => (
          <MCPCommandResult
            key={result.id}
            resultId={result.id}
          />
        ))}
      </div>
    </div>
  );
};

export default MCPAgentCommands;
```

## Testing Strategy

1. **Unit Testing**
   - Test the `extractToolCall` function with various inputs
   - Test the system prompt generation

2. **Component Testing**
   - Test MCPPendingCommand component in isolation
   - Test MCPCommandResult component with different result types

3. **Integration Testing**
   - Test the full flow from user query to command approval to result
   - Test error handling for various scenarios
   - Test reconnection logic

## Challenges to Address

1. **Ollama Model Compatibility**: Ensure the system prompt works with different models
2. **Error Handling**: Robust error handling for network issues and MCP server errors
3. **User Experience**: Ensure clear feedback during command approval and execution
4. **Context Management**: Keep the conversation context across multiple tool uses
5. **Tool Selection Logic**: Help models make appropriate tool selections

## Next Steps

1. Implement the MCPPendingCommand component
2. Add the extractToolCall function to MCPAgentContext
3. Update processUserRequest to work with Ollama models
4. Enhance approveCommand to handle tool results for analysis
5. Test with various query types and models
6. Refine based on testing feedback 