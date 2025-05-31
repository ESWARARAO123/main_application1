# AI Agents Architecture & Integration Guide

## Overview

AI agents in the Platform Dashboard are autonomous or semi-autonomous entities that can interpret user instructions, generate and execute system commands, interact with external tools (via MCP), and provide intelligent feedback. This document outlines the architecture, use cases, integration plan, and UI/UX flow for AI agents, with a focus on leveraging the Model Context Protocol (MCP) for secure and extensible system operations.

---

## What is an AI Agent?

An AI agent is a software entity that:
- Understands user intent (via natural language or structured input)
- Plans and generates actions (commands, tool invocations, queries)
- Executes actions (directly or via user approval)
- Observes results and adapts its behavior
- Communicates with users in a conversational or task-oriented manner

In this platform, AI agents are powered by LLMs (Ollama, GPT, etc.) and can be extended to interact with the system through the MCP server.

---

## Core Architecture

### Components

1. **AI Model**: The LLM that interprets user input and generates actions
2. **Agent Orchestrator**: Manages agent state, plans, and tool selection
3. **MCP Client**: Connects to the MCP server, discovers tools, and invokes them
4. **Command Approval UI**: Allows users to review, approve, or modify agent-generated commands
5. **Result Feedback Loop**: Feeds tool results back to the agent for further reasoning or user display
6. **Session & State Management**: Maintains context, history, and agent memory

### Data Flow

1. **User Input** → 2. **AI Agent (LLM)** → 3. **Action/Command Proposal** → 4. **User Approval** → 5. **MCP Tool Execution** → 6. **Result Feedback** → 7. **AI Agent/LLM** → 8. **User Output**

---

## MCP Orchestrator Agent

### Purpose and Function

The Orchestrator Agent serves as the primary mediator between users and the MCP server, providing AI-assisted command suggestions and execution capabilities. It enables a copilot-like experience within the terminal interface.

### Key Capabilities

- Observes user context and command history
- Suggests appropriate commands based on current terminal state
- Requests user approval before execution
- Executes approved commands through MCP server
- Analyzes results and suggests follow-up actions
- Maintains conversation context for multi-step tasks

### Architecture

1. **Agent Controller**: Central coordination component that manages the agent lifecycle and communication
2. **Context Manager**: Maintains terminal state, command history, and conversation context
3. **Command Generator**: Uses LLM to generate appropriate command suggestions
4. **Approval Interface**: UI components for user review and approval of suggested commands
5. **Execution Engine**: Handles command execution via MCP server
6. **Result Analyzer**: Processes command output and updates suggestions

### User Interaction Flow

1. **Activation**: User toggles the agent on in the terminal input interface
2. **Observation**: Agent analyzes current terminal state and conversation history
3. **Suggestion**: Agent proposes relevant commands with explanations
4. **Review**: User reviews, may modify, and approves/rejects suggestions
5. **Execution**: Approved commands are sent to MCP server for execution
6. **Analysis**: Results are analyzed by the agent for further suggestions
7. **Iteration**: The cycle continues with new suggestions based on results

### Implementation Details

#### Frontend Component

The Orchestrator Agent UI includes:
- Toggle button in the terminal input interface
- Suggestion display with command preview and rationale
- Approval/Modify/Reject controls
- Execution status indicator
- Result display with agent analysis

#### Backend Services

- **Agent Service**: Manages agent state and coordinates with LLM and MCP server
- **LLM Connector**: Interfaces with the language model for command generation
- **MCP Client**: Handles tool discovery and invocation via MCP protocol
- **Context Store**: Maintains conversation history and terminal state

#### Communication Protocol

1. **User → Agent**: User activates agent and provides context/commands
2. **Agent → LLM**: Agent formulates prompts based on context and user input
3. **LLM → Agent**: LLM generates command suggestions and rationales
4. **Agent → User**: Agent presents suggestions for approval
5. **User → Agent**: User approves, modifies, or rejects suggestions
6. **Agent → MCP**: Agent sends approved commands to MCP server
7. **MCP → Agent**: MCP returns command execution results
8. **Agent → LLM**: Agent sends results for analysis
9. **Agent → User**: Agent presents results and new suggestions

---

## Use Cases for AI Agents

- **System Automation**: File management, process monitoring, log analysis
- **DevOps Tasks**: Running scripts, deploying code, checking system health
- **Data Extraction**: Searching files, extracting structured data, summarizing logs
- **Knowledge Retrieval**: Using RAG to answer questions from documentation or uploaded files
- **Interactive Troubleshooting**: Diagnosing issues, suggesting fixes, running diagnostics
- **Custom Workflows**: Multi-step operations, chaining tool invocations, conditional logic

---

## Example: MCP AI Agent

### Agent Workflow

1. **Connect to MCP Server**: The agent establishes an SSE connection, receives a client ID, and discovers available tools.
2. **Interpret User Request**: The agent uses the LLM to parse user intent and determine which MCP tool(s) to invoke.
3. **Generate Command(s)**: The agent formulates tool invocation requests (e.g., runShellCommand, readFile, etc.) with parameters.
4. **User Approval**: The UI presents the proposed command(s) to the user for review, modification, or approval.
5. **Execute via MCP**: Upon approval, the agent (via the MCP client) sends the command to the MCP server.
6. **Receive Results**: The agent receives tool results (via HTTP and/or SSE), parses them, and may use them for further reasoning.
7. **Iterate or Respond**: The agent may propose follow-up actions or present the final result to the user.

### UI/UX Flow

- **Chat Interface**: User interacts with the agent via chat or task forms
- **Command Approval Modal**: When the agent proposes a command, a modal or sidebar displays the command, parameters, and rationale
- **User Actions**: Approve, modify, or reject the command
- **Execution Feedback**: Results are streamed back to the UI, with clear indication of success, errors, and next steps
- **History/Trace**: Users can view the sequence of agent actions, approvals, and results

## MCP Terminal Executor Agent

### Purpose

The MCP Terminal Executor Agent is a specialized AI agent designed to interact directly with the MCP server, generating and executing commands based on user requests. It creates a continuous feedback loop between the user, AI, and MCP server, allowing for complex task execution with user oversight.

### Architecture

#### Components

1. **Executor Service (`src/services/executorAgentService.js`)**:
   - Manages the lifecycle of the executor agent
   - Establishes and maintains SSE connection with MCP server
   - Handles tool discovery and invocation
   - Processes command results and generates follow-up actions

2. **Command Generator (`src/services/commandGeneratorService.js`)**:
   - Uses LLM to generate appropriate MCP commands
   - Formats commands according to MCP protocol
   - Provides explanations for suggested commands
   - Handles error correction and alternative suggestions

3. **MCP Connection Manager (`src/services/mcpConnectionManager.js`)**:
   - Manages connections to MCP servers
   - Retrieves available tools from MCP server
   - Handles authentication and session management
   - Monitors connection health and reconnects when necessary

4. **Result Analyzer (`src/services/resultAnalyzerService.js`)**:
   - Processes command execution results
   - Extracts relevant information for follow-up actions
   - Generates summaries and insights from results
   - Identifies errors and suggests corrections

#### Frontend Components

1. **Executor Agent UI (`client/src/components/executorAgent/ExecutorAgentPanel.tsx`)**:
   - Displays agent status and connection information
   - Shows command suggestions with explanations
   - Provides approve/modify/reject controls
   - Displays execution results and follow-up suggestions

2. **Command Approval UI (`client/src/components/executorAgent/CommandApproval.tsx`)**:
   - Displays command details and parameters
   - Allows modification of command parameters
   - Provides approve/reject buttons
   - Shows command history and context

3. **Result Display UI (`client/src/components/executorAgent/ResultDisplay.tsx`)**:
   - Renders command execution results
   - Highlights important information
   - Shows agent analysis and insights
   - Displays follow-up suggestions

### Implementation Plan

#### Phase 1: Core Infrastructure

1. **MCP Connection Layer**:
   - Implement SSE connection to MCP server
   - Create tool discovery and invocation functions
   - Develop connection management and health monitoring
   - Add authentication and session handling

2. **Command Generation**:
   - Create specialized LLM prompts for command generation
   - Implement command formatting according to MCP protocol
   - Develop explanation generation for commands
   - Add validation and error handling

3. **Result Processing**:
   - Implement result parsing and analysis
   - Create follow-up suggestion generation
   - Develop error detection and correction
   - Add context maintenance across commands

#### Phase 2: Frontend Integration

1. **Agent UI Components**:
   - Create executor agent panel component
   - Implement command approval interface
   - Develop result display component
   - Add agent status indicators

2. **State Management**:
   - Create executor agent context provider
   - Implement state management for agent lifecycle
   - Add command history and context tracking
   - Develop user preference management

3. **User Interaction**:
   - Implement agent activation toggle
   - Create command approval workflow
   - Develop result feedback mechanism
   - Add error handling and recovery UI

#### Phase 3: Integration and Testing

1. **MCP Server Integration**:
   - Connect to default MCP server (mcp-terminal_executor)
   - Test with various MCP tools and commands
   - Implement error handling and recovery
   - Add performance monitoring and optimization

2. **AI Integration**:
   - Fine-tune LLM prompts for command generation
   - Implement context-aware command suggestions
   - Develop result analysis capabilities
   - Add learning from user feedback

3. **User Experience Refinement**:
   - Optimize command suggestion relevance
   - Improve explanation clarity and usefulness
   - Enhance result display and analysis
   - Add user customization options

### Implementation Details

#### MCP Connection Establishment

```javascript
// src/services/mcpConnectionManager.js
async function connectToMCPServer(serverConfig) {
  try {
    // Establish SSE connection
    const eventSource = new EventSource(`http://${serverConfig.host}:${serverConfig.port}/sse`);

    // Handle connection events
    eventSource.onopen = () => {
      console.log('SSE connection established');
    };

    eventSource.onmessage = (event) => {
      const data = JSON.parse(event.data);
      handleMCPMessage(data);
    };

    eventSource.onerror = (error) => {
      console.error('SSE connection error:', error);
      reconnectWithBackoff();
    };

    // Discover available tools
    const toolsResponse = await fetch(`http://${serverConfig.host}:${serverConfig.port}/tools`);
    const toolsData = await toolsResponse.json();

    return {
      connection: eventSource,
      clientId: null, // Will be set when connected message is received
      tools: toolsData.tools
    };
  } catch (error) {
    console.error('Failed to connect to MCP server:', error);
    throw error;
  }
}
```

#### Command Generation with LLM

```javascript
// src/services/commandGeneratorService.js
async function generateCommand(userRequest, terminalContext, availableTools) {
  try {
    // Create prompt for LLM
    const prompt = `
You are an AI assistant that generates commands for the MCP Terminal.
Available tools: ${JSON.stringify(availableTools.map(t => t.name))}

Current directory: ${terminalContext.currentDirectory}
Recent commands: ${terminalContext.recentCommands.join(', ')}

User request: "${userRequest}"

Generate a command to fulfill this request. Format your response as JSON:
{
  "tool": "toolName",
  "parameters": { param1: "value1", param2: "value2" },
  "explanation": "Detailed explanation of what this command does"
}
`;

    // Get response from LLM
    const response = await ollamaService.generateCompletion(
      'llama3', // or other model ID
      prompt
    );

    // Parse and validate the response
    const parsedResponse = JSON.parse(response);

    // Validate that the tool exists
    const toolExists = availableTools.some(t => t.name === parsedResponse.tool);
    if (!toolExists) {
      throw new Error(`Tool "${parsedResponse.tool}" not found in available tools`);
    }

    return parsedResponse;
  } catch (error) {
    console.error('Error generating command:', error);
    throw error;
  }
}
```

#### Command Execution

```javascript
// src/services/executorAgentService.js
async function executeCommand(command, mcpConnection) {
  try {
    const { tool, parameters } = command;
    const messageId = `msg-${Date.now()}`;

    // Format the request according to MCP protocol
    const request = {
      id: messageId,
      type: 'invoke_tool',
      content: {
        name: tool,
        parameters: parameters
      },
      clientId: mcpConnection.clientId
    };

    // Send the request to the MCP server
    const response = await fetch(`http://${mcpConnection.host}:${mcpConnection.port}/messages`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(request)
    });

    // Parse the response
    const result = await response.json();

    // Return the result
    return {
      success: true,
      messageId,
      result
    };
  } catch (error) {
    console.error('Error executing command:', error);
    return {
      success: false,
      error: error.message
    };
  }
}
```

#### Result Analysis

```javascript
// src/services/resultAnalyzerService.js
async function analyzeResult(result, command, userRequest) {
  try {
    // Create prompt for LLM
    const prompt = `
You are an AI assistant analyzing the result of a command execution.

User request: "${userRequest}"
Command executed: ${command.tool} with parameters ${JSON.stringify(command.parameters)}
Result: ${JSON.stringify(result)}

Analyze this result and provide:
1. A summary of what was found or done
2. Any important information or insights
3. Potential follow-up actions
4. If there were errors, suggest corrections

Format your response as JSON:
{
  "summary": "Brief summary of the result",
  "insights": ["Key insight 1", "Key insight 2"],
  "followUpActions": [
    {
      "tool": "suggestedTool",
      "parameters": { param1: "value1" },
      "explanation": "Why this follow-up is suggested"
    }
  ],
  "hasErrors": boolean,
  "errorCorrection": "Suggestion to fix error if any"
}
`;

    // Get response from LLM
    const response = await ollamaService.generateCompletion(
      'llama3', // or other model ID
      prompt
    );

    // Parse and validate the response
    const analysis = JSON.parse(response);

    return analysis;
  } catch (error) {
    console.error('Error analyzing result:', error);
    return {
      summary: 'Failed to analyze result',
      insights: [],
      followUpActions: [],
      hasErrors: true,
      errorCorrection: 'The analysis service encountered an error'
    };
  }
}
```

#### Frontend Implementation

The frontend implementation of the Executor Agent includes several React components and hooks:

```typescript
// client/src/contexts/ExecutorAgentContext.tsx
import React, { createContext, useContext, useState, useEffect } from 'react';
import { useMCP } from './MCPContext';

interface ExecutorAgentContextType {
  isAgentEnabled: boolean;
  isConnecting: boolean;
  isConnected: boolean;
  currentCommand: any | null;
  commandResult: any | null;
  analysis: any | null;
  toggleAgent: () => void;
  generateCommand: (userRequest: string) => Promise<void>;
  approveCommand: () => Promise<void>;
  modifyCommand: (modifiedCommand: any) => void;
  rejectCommand: () => void;
}

export const ExecutorAgentContext = createContext<ExecutorAgentContextType | null>(null);

export const useExecutorAgent = () => {
  const context = useContext(ExecutorAgentContext);
  if (!context) {
    throw new Error('useExecutorAgent must be used within an ExecutorAgentProvider');
  }
  return context;
};

export const ExecutorAgentProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const { defaultMCPServer } = useMCP();
  const [isAgentEnabled, setIsAgentEnabled] = useState(false);
  const [isConnecting, setIsConnecting] = useState(false);
  const [isConnected, setIsConnected] = useState(false);
  const [connection, setConnection] = useState<any>(null);
  const [currentCommand, setCurrentCommand] = useState<any>(null);
  const [commandResult, setCommandResult] = useState<any>(null);
  const [analysis, setAnalysis] = useState<any>(null);

  // Toggle agent on/off
  const toggleAgent = () => {
    if (isAgentEnabled) {
      // Disconnect from MCP server
      if (connection) {
        connection.close();
        setConnection(null);
      }
      setIsConnected(false);
      setIsAgentEnabled(false);
    } else {
      // Connect to MCP server
      setIsAgentEnabled(true);
      setIsConnecting(true);
      connectToMCPServer();
    }
  };

  // Connect to MCP server
  const connectToMCPServer = async () => {
    try {
      if (!defaultMCPServer) {
        throw new Error('No default MCP server configured');
      }

      const response = await fetch('/api/agent/connect', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          host: defaultMCPServer.mcp_host,
          port: defaultMCPServer.mcp_port
        })
      });

      const data = await response.json();

      if (data.success) {
        setConnection(data.connection);
        setIsConnected(true);
      } else {
        throw new Error(data.error || 'Failed to connect to MCP server');
      }
    } catch (error) {
      console.error('Error connecting to MCP server:', error);
      setIsAgentEnabled(false);
    } finally {
      setIsConnecting(false);
    }
  };

  // Generate command based on user request
  const generateCommand = async (userRequest: string) => {
    try {
      if (!isConnected || !connection) {
        throw new Error('Not connected to MCP server');
      }

      const response = await fetch('/api/agent/generate-command', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          userRequest,
          connectionId: connection.id
        })
      });

      const data = await response.json();

      if (data.success) {
        setCurrentCommand(data.command);
        setCommandResult(null);
        setAnalysis(null);
      } else {
        throw new Error(data.error || 'Failed to generate command');
      }
    } catch (error) {
      console.error('Error generating command:', error);
    }
  };

  // Approve and execute current command
  const approveCommand = async () => {
    try {
      if (!isConnected || !connection || !currentCommand) {
        throw new Error('No command to execute');
      }

      const response = await fetch('/api/agent/execute-command', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          command: currentCommand,
          connectionId: connection.id
        })
      });

      const data = await response.json();

      if (data.success) {
        setCommandResult(data.result);

        // Analyze the result
        const analysisResponse = await fetch('/api/agent/analyze-result', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json'
          },
          body: JSON.stringify({
            result: data.result,
            command: currentCommand,
            connectionId: connection.id
          })
        });

        const analysisData = await analysisResponse.json();

        if (analysisData.success) {
          setAnalysis(analysisData.analysis);
        }
      } else {
        throw new Error(data.error || 'Failed to execute command');
      }
    } catch (error) {
      console.error('Error executing command:', error);
    }
  };

  // Modify current command
  const modifyCommand = (modifiedCommand: any) => {
    setCurrentCommand(modifiedCommand);
  };

  // Reject current command
  const rejectCommand = () => {
    setCurrentCommand(null);
    setCommandResult(null);
    setAnalysis(null);
  };

  // Clean up on unmount
  useEffect(() => {
    return () => {
      if (connection) {
        connection.close();
      }
    };
  }, [connection]);

  const value = {
    isAgentEnabled,
    isConnecting,
    isConnected,
    currentCommand,
    commandResult,
    analysis,
    toggleAgent,
    generateCommand,
    approveCommand,
    modifyCommand,
    rejectCommand
  };

  return (
    <ExecutorAgentContext.Provider value={value}>
      {children}
    </ExecutorAgentContext.Provider>
  );
};
```

The Command Approval component:

```typescript
// client/src/components/executorAgent/CommandApproval.tsx
import React, { useState } from 'react';
import { useExecutorAgent } from '../../contexts/ExecutorAgentContext';

interface CommandApprovalProps {
  className?: string;
}

export const CommandApproval: React.FC<CommandApprovalProps> = ({ className = '' }) => {
  const {
    currentCommand,
    approveCommand,
    modifyCommand,
    rejectCommand
  } = useExecutorAgent();

  const [isEditing, setIsEditing] = useState(false);
  const [editedParameters, setEditedParameters] = useState<any>(null);

  if (!currentCommand) {
    return null;
  }

  const handleEdit = () => {
    setEditedParameters(currentCommand.parameters);
    setIsEditing(true);
  };

  const handleSave = () => {
    modifyCommand({
      ...currentCommand,
      parameters: editedParameters
    });
    setIsEditing(false);
  };

  const handleCancel = () => {
    setEditedParameters(null);
    setIsEditing(false);
  };

  const handleParameterChange = (key: string, value: any) => {
    setEditedParameters({
      ...editedParameters,
      [key]: value
    });
  };

  return (
    <div className={`bg-gray-800 rounded-lg p-4 ${className}`}>
      <h3 className="text-lg font-semibold text-white mb-2">Command Suggestion</h3>

      <div className="mb-4">
        <div className="text-sm text-gray-400 mb-1">Tool</div>
        <div className="bg-gray-700 rounded p-2 text-white font-mono">
          {currentCommand.tool}
        </div>
      </div>

      <div className="mb-4">
        <div className="text-sm text-gray-400 mb-1">Parameters</div>
        {isEditing ? (
          <div className="bg-gray-700 rounded p-2">
            {Object.entries(editedParameters).map(([key, value]: [string, any]) => (
              <div key={key} className="mb-2">
                <label className="block text-sm text-gray-300 mb-1">{key}</label>
                <input
                  type="text"
                  value={value as string}
                  onChange={(e) => handleParameterChange(key, e.target.value)}
                  className="w-full bg-gray-600 text-white p-1 rounded font-mono"
                />
              </div>
            ))}
            <div className="flex mt-2">
              <button
                onClick={handleSave}
                className="bg-blue-600 text-white px-3 py-1 rounded mr-2"
              >
                Save
              </button>
              <button
                onClick={handleCancel}
                className="bg-gray-600 text-white px-3 py-1 rounded"
              >
                Cancel
              </button>
            </div>
          </div>
        ) : (
          <div className="bg-gray-700 rounded p-2 text-white font-mono">
            {JSON.stringify(currentCommand.parameters, null, 2)}
          </div>
        )}
      </div>

      <div className="mb-4">
        <div className="text-sm text-gray-400 mb-1">Explanation</div>
        <div className="bg-gray-700 rounded p-2 text-white">
          {currentCommand.explanation}
        </div>
      </div>

      <div className="flex justify-between">
        <button
          onClick={approveCommand}
          className="bg-green-600 hover:bg-green-700 text-white px-4 py-2 rounded"
        >
          Approve
        </button>
        {!isEditing && (
          <button
            onClick={handleEdit}
            className="bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded mx-2"
          >
            Modify
          </button>
        )}
        <button
          onClick={rejectCommand}
          className="bg-red-600 hover:bg-red-700 text-white px-4 py-2 rounded"
        >
          Reject
        </button>
      </div>
    </div>
  );
};
```

The Result Display component:

```typescript
// client/src/components/executorAgent/ResultDisplay.tsx
import React from 'react';
import { useExecutorAgent } from '../../contexts/ExecutorAgentContext';

interface ResultDisplayProps {
  className?: string;
}

export const ResultDisplay: React.FC<ResultDisplayProps> = ({ className = '' }) => {
  const { commandResult, analysis, generateCommand } = useExecutorAgent();

  if (!commandResult) {
    return null;
  }

  const handleFollowUp = (followUpAction: any) => {
    // Use the follow-up action as a new command
    generateCommand(followUpAction.explanation);
  };

  return (
    <div className={`bg-gray-800 rounded-lg p-4 ${className}`}>
      <h3 className="text-lg font-semibold text-white mb-2">Command Result</h3>

      <div className="mb-4">
        <div className="text-sm text-gray-400 mb-1">Output</div>
        <div className="bg-gray-700 rounded p-2 text-white font-mono whitespace-pre-wrap max-h-60 overflow-y-auto">
          {commandResult.content?.[0]?.text || JSON.stringify(commandResult, null, 2)}
        </div>
      </div>

      {analysis && (
        <>
          <div className="mb-4">
            <div className="text-sm text-gray-400 mb-1">Summary</div>
            <div className="bg-gray-700 rounded p-2 text-white">
              {analysis.summary}
            </div>
          </div>

          {analysis.insights.length > 0 && (
            <div className="mb-4">
              <div className="text-sm text-gray-400 mb-1">Insights</div>
              <ul className="bg-gray-700 rounded p-2 text-white">
                {analysis.insights.map((insight: string, index: number) => (
                  <li key={index} className="mb-1">• {insight}</li>
                ))}
              </ul>
            </div>
          )}

          {analysis.followUpActions.length > 0 && (
            <div className="mb-4">
              <div className="text-sm text-gray-400 mb-1">Follow-up Actions</div>
              <div className="bg-gray-700 rounded p-2">
                {analysis.followUpActions.map((action: any, index: number) => (
                  <div key={index} className="mb-2 p-2 bg-gray-600 rounded">
                    <div className="text-white mb-1">{action.explanation}</div>
                    <div className="text-gray-300 text-sm mb-2">
                      Tool: <span className="font-mono">{action.tool}</span>
                    </div>
                    <button
                      onClick={() => handleFollowUp(action)}
                      className="bg-blue-600 hover:bg-blue-700 text-white px-3 py-1 rounded text-sm"
                    >
                      Use This Suggestion
                    </button>
                  </div>
                ))}
              </div>
            </div>
          )}

          {analysis.hasErrors && (
            <div className="mb-4">
              <div className="text-sm text-red-400 mb-1">Error Correction</div>
              <div className="bg-gray-700 rounded p-2 text-white">
                {analysis.errorCorrection}
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
};
```

Integration with the Chat Interface:

```typescript
// client/src/components/chat/ChatInput.tsx
// Add to existing ChatInput component

import { useExecutorAgent } from '../../contexts/ExecutorAgentContext';

// Inside the ChatInput component
const {
  isAgentEnabled,
  isConnecting,
  isConnected,
  toggleAgent,
  generateCommand
} = useExecutorAgent();

// Add to the buttons row
<button
  type="button"
  onClick={toggleAgent}
  className={`flex items-center px-3 py-1 rounded ${
    isAgentEnabled
      ? 'bg-green-600 hover:bg-green-700'
      : 'bg-gray-600 hover:bg-gray-700'
  } text-white text-sm transition-colors`}
  disabled={isConnecting}
>
  {isConnecting ? (
    <span className="flex items-center">
      <svg className="animate-spin -ml-1 mr-2 h-4 w-4 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
      </svg>
      Connecting...
    </span>
  ) : (
    <span className="flex items-center">
      <svg className="h-4 w-4 mr-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
      </svg>
      {isAgentEnabled ? 'Executor Active' : 'Activate Executor'}
    </span>
  )}
</button>

// Modify the form submission handler
const handleSubmit = async (e: React.FormEvent) => {
  e.preventDefault();

  if (!message.trim()) return;

  const userMessage = message;
  setMessage('');

  if (isAgentEnabled && isConnected) {
    // Use the executor agent to generate a command
    await generateCommand(userMessage);
  } else {
    // Regular chat message handling
    // ...existing code...
  }
};
```
```

---

## Integration Plan

### 1. Agent Orchestrator Implementation

#### Phase 1: Core Infrastructure
- Build central Agent Service in the backend
  - Create `src/services/agentService.js` for command suggestion generation
  - Implement `src/services/agentContextManager.js` for context tracking
  - Add API routes in `src/routes/agent.js` for agent functionality
- Implement context management system
  - Track terminal state (current directory, command history)
  - Maintain conversation history for context-aware suggestions
  - Store user preferences for agent behavior
- Create LLM connector for command generation
  - Develop specialized prompts for terminal command generation
  - Implement parsing and validation of LLM responses
  - Handle error cases and fallbacks

#### Phase 2: Frontend Integration
- Develop terminal input toggle functionality
  - Add agent toggle button to `ChatInput` component
  - Create `AgentContext` provider for state management
  - Implement agent status indicators
- Build suggestion and approval UI components
  - Create `CommandSuggestions` component for displaying suggestions
  - Implement command selection and modification UI
  - Add explanation display for suggested commands
- Implement result display and feedback UI
  - Enhance `MCPToolResult` component to show agent analysis
  - Add follow-up suggestion display
  - Implement feedback collection for suggestion quality

#### Phase 3: MCP Integration
- Connect Agent Service to MCP client
  - Extend MCP service with agent-specific functionality
  - Implement secure command execution with proper context
  - Add terminal state retrieval capabilities
- Implement tool discovery and invocation
  - Dynamically discover available MCP tools
  - Match user intent to appropriate tools
  - Handle tool parameters and validation
- Develop result parsing and feedback loop
  - Parse command execution results
  - Generate follow-up suggestions based on results
  - Maintain context across multiple commands

### 2. MCP Client Integration
- Use the MCP client to connect to the MCP server (see mcp-protocol-guide.md, mcp-integration-guide.md)
- Discover available tools and schemas dynamically
- Format tool invocation requests as per MCP protocol
- Handle SSE for real-time result updates

### 3. Command Approval Layer
- UI component to display agent-generated commands
- Allow user to edit parameters or reject commands
- Log all approvals and rejections for auditability

### 4. Result Feedback Loop
- Feed tool results back to the agent for further reasoning
- Allow the agent to chain actions or ask clarifying questions
- Display results and next steps in the UI

### 5. Security & Permissions
- Enforce user approval for all system-affecting actions
- Limit available tools based on user roles and context
- Log all tool invocations and results

---

## UI Integration Suggestions

- **Terminal Toggle**: Add toggle button in the terminal input area
- **Suggestion Panel**: Display command suggestions with explanations
- **Inline Approval**: Allow approve/modify/reject actions directly in terminal
- **Result Analysis**: Show agent's interpretation of command results
- **Agent Status Indicator**: Show when agent is thinking, waiting for approval, or executing
- **History Timeline**: Visualize the sequence of agent actions and user approvals

---

## Example Agent Cycle for Orchestrator

1. **User**: Activates agent toggle and types "find large log files"
2. **Agent**: (Analyzes terminal context and user intent)
   - Proposes: `find /var/log -type f -size +10M -exec ls -lh {} \; | sort -rh`
   - Provides explanation: "This command finds log files larger than 10MB and sorts them by size"
   - Waits for user approval
3. **User**: Approves or modifies command
4. **Agent**: Executes command via MCP server, displays results
   - Analyzes large log files found
   - Proposes follow-up: `du -h /var/log | sort -rh | head -10`
   - Explains: "Let's check the top 10 largest directories in /var/log"
5. **User**: Approves
6. **Agent**: Executes new command, provides summary of storage usage
   - Suggests potential cleanup options based on findings

## Implementation Details

### Agent Service Architecture

The AI agent orchestration system is implemented with the following components:

1. **Agent Service (`src/services/agentService.js`)**:
   - Manages agent state and lifecycle
   - Generates command suggestions using LLM
   - Processes user input and terminal context
   - Communicates with MCP server for command execution

2. **Agent Context Manager (`src/services/agentContextManager.js`)**:
   - Maintains terminal state (current directory, command history)
   - Tracks conversation history for context-aware suggestions
   - Provides context for LLM prompt generation

3. **Agent API Routes (`src/routes/agent.js`)**:
   - Endpoints for generating command suggestions
   - Endpoints for executing commands with agent context
   - Endpoints for managing agent state and preferences

### Frontend Components

1. **Agent Context Provider (`client/src/contexts/AgentContext.tsx`)**:
   - Manages agent state and settings
   - Provides agent functionality to components
   - Handles communication with backend API

2. **Agent Toggle in Chat Input (`client/src/components/chat/ChatInput.tsx`)**:
   - Toggle button to activate/deactivate the AI agent
   - Visual indicator for agent status
   - Integration with existing chat interface

3. **Command Suggestions Component (`client/src/components/chat/CommandSuggestions.tsx`)**:
   - Displays AI-generated command suggestions
   - Allows selection and modification of commands
   - Provides explanations for suggested commands

### Integration with MCP

The agent integrates with the MCP server through:

1. **Enhanced MCP Service**:
   - Extended to support agent-specific functionality
   - Provides terminal state information to the agent
   - Executes commands with proper context

2. **Command Execution Flow**:
   - Agent generates command suggestions based on user input and context
   - User reviews and approves/modifies suggestions
   - Agent executes approved commands via MCP server
   - Agent analyzes results and suggests follow-up actions

---

## Future Directions

### Specialized Agents

The platform supports specialized agents beyond the basic Orchestrator:

#### MCP Terminal Executor Agent

The **MCP Terminal Executor Agent** is a specialized agent that directly interacts with the MCP server to execute commands based on user requests. It creates a continuous feedback loop between the user, AI, and MCP server.

**Key Features:**
- Automatically connects to the default MCP server (mcp-terminal_executor)
- Discovers available tools from the MCP server
- Generates commands using AI based on user requests
- Presents commands to the user for approval
- Executes approved commands and returns results
- Analyzes results and suggests follow-up actions
- Maintains context across multiple command executions

**Implementation Details:**
- Uses SSE connection to establish real-time communication with MCP server
- Dynamically adapts to available tools on the connected MCP server
- Generates commands using specialized LLM prompts for terminal operations
- Provides detailed explanations for suggested commands
- Handles errors and suggests corrections when commands fail

**Usage Flow:**
1. User activates the Executor Agent in the chat interface
2. Agent establishes connection to the default MCP server
3. User describes a task in natural language
4. AI generates a plan with specific MCP commands
5. Agent presents commands with explanations for user approval
6. User approves or modifies the suggested commands
7. Agent executes approved commands via MCP server
8. Agent receives and analyzes results
9. Agent suggests follow-up commands based on results
10. Process continues until task is complete

**Example Interaction:**
```
User: "Find all log files larger than 100MB and compress them"

AI: I'll help you find and compress large log files. Here's my plan:
1. First, I'll locate all log files larger than 100MB
2. Then I'll compress each file using gzip
3. Finally, I'll verify the compression was successful

Executor Agent: I'll execute this using the MCP server. Here's the first command:
Command: find / -name "*.log" -size +100M -type f
Explanation: This searches the entire filesystem for files ending in .log that are larger than 100MB
[Approve] [Modify] [Reject]

User: [Approve]

Executor Agent: Executing command...
Results:
/var/log/application.log (150MB)
/var/log/system.log (120MB)
/home/user/logs/debug.log (200MB)

Based on these results, I'll now compress each file. Here's the next command:
Command: for file in /var/log/application.log /var/log/system.log /home/user/logs/debug.log; do gzip -v "$file"; done
Explanation: This will compress each of the found log files using gzip with verbose output
[Approve] [Modify] [Reject]
```

#### Other Specialized Agents

Future implementation will extend to include additional specialized agents:

- **DevOps Agent**: Specialized for infrastructure tasks and deployment
- **Data Agent**: Focused on data processing, analysis, and extraction
- **Security Agent**: Monitors and suggests security improvements
- **Documentation Agent**: Manages and retrieves information from docs

### Advanced Capabilities

- **Autonomous Mode**: Allow trusted agents to execute certain actions without approval (with audit logging)
- **Multi-Agent Collaboration**: Agents specialized for different domains (DevOps, Data, Docs)
- **Advanced Planning**: Agents that can plan multi-step workflows and adapt based on results
- **Natural Language Programming**: Users describe goals, agent generates and executes plans
- **Integration with RAG**: Agents use RAG to ground actions in documentation and provide citations
- **Plugin Ecosystem**: Allow third-party tools to be registered with MCP and used by agents

---

## References
- See `mcp-protocol-guide.md`, `mcp-integration-guide.md`, `implementation-summary.md`, `ai-integration-guide.md`, `RAG.md`, `ai_integration.md`, and `documentation.md` for technical details and implementation examples.