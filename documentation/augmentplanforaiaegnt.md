# AI Agent Orchestration System Implementation Plan

## Overview

This document outlines the implementation plan for creating an AI agent orchestration system for the MCP-Terminal project. The orchestrator agent will serve as a mediator between the MCP server and users, providing AI-assisted command suggestions and execution capabilities.

## Current Architecture Analysis

### MCP Integration

The codebase already has:
- `MCPContext` for managing MCP server connections and state
- MCP service for communicating with MCP servers
- Command execution functionality via the MCP protocol
- UI components for command approval and display

### Chat Interface

The chat interface includes:
- `ChatInput` component with support for toggles (e.g., RAG)
- Message display components
- Model selection functionality

## Implementation Plan

### 1. Agent Architecture

#### 1.1 Create Agent Service

Create a new service `src/services/agentService.js` that will:
- Manage agent state and lifecycle
- Generate command suggestions using LLM
- Process user input and context
- Communicate with MCP server

```javascript
// src/services/agentService.js
const { logger } = require('../utils/logger');
const ollamaService = require('./ollamaService');
const mcpService = require('./mcpService');

/**
 * Generate command suggestions based on user input and terminal context
 */
async function generateCommandSuggestions(input, context, modelId) {
  try {
    // Create prompt for the LLM
    const prompt = createAgentPrompt(input, context);
    
    // Generate suggestions using Ollama
    const response = await ollamaService.generateCompletion(modelId, prompt);
    
    // Parse and validate suggestions
    return parseCommandSuggestions(response);
  } catch (error) {
    logger.error(`Error generating command suggestions: ${error.message}`);
    throw error;
  }
}

// Additional functions for agent service
```

#### 1.2 Create Agent Context Manager

Create a context manager to maintain terminal state and conversation history:

```javascript
// src/services/agentContextManager.js
class AgentContextManager {
  constructor(userId) {
    this.userId = userId;
    this.terminalState = {
      currentDirectory: '~',
      lastCommands: [],
      lastOutputs: [],
      environmentVariables: {}
    };
    this.conversationHistory = [];
  }
  
  // Methods to update and retrieve context
}
```

### 2. Frontend Integration

#### 2.1 Add Agent Toggle to ChatInput

Modify `client/src/components/chat/ChatInput.tsx` to add an agent toggle button:

```typescript
// Add to ChatInputProps interface
interface ChatInputProps {
  // Existing props
  isAgentAvailable?: boolean;
  isAgentEnabled?: boolean;
  onToggleAgent?: () => void;
}

// Add to the buttons row in the component
{/* Agent toggle button */}
<button
  type="button"
  onClick={onToggleAgent}
  disabled={!isAgentAvailable || isLoading || isUploading || isStreaming}
  style={{
    ...chatInputStyles.toggleButton,
    ...(isAgentEnabled ? chatInputStyles.toggleEnabled : chatInputStyles.toggleDisabled),
    opacity: (!isAgentAvailable || isLoading) ? 0.5 : 1,
  }}
  className="hover:bg-opacity-90 transition-all"
  aria-label={isAgentEnabled ? "Disable AI agent" : "Enable AI agent"}
  title={!isAgentAvailable ? "MCP connection required" : (isAgentEnabled ? "Disable AI agent" : "Enable AI agent")}
>
  <LightBulbIcon className="h-4 w-4 mr-1" />
  Agent
</button>
```

#### 2.2 Create Agent Context Provider

Create a new context provider for the agent:

```typescript
// client/src/contexts/AgentContext.tsx
import React, { createContext, useContext, useState, useEffect } from 'react';
import { useMCP } from './MCPContext';
import axios from 'axios';

interface AgentContextType {
  isAgentEnabled: boolean;
  isAgentAvailable: boolean;
  toggleAgent: () => void;
  commandSuggestions: string[];
  isGeneratingSuggestions: boolean;
  generateSuggestions: (input: string, context: any) => Promise<void>;
  executeCommand: (command: string) => Promise<any>;
}

const AgentContext = createContext<AgentContextType | null>(null);

export const useAgent = () => {
  const context = useContext(AgentContext);
  if (!context) {
    throw new Error('useAgent must be used within an AgentProvider');
  }
  return context;
};

export const AgentProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  // Implementation details
};
```

#### 2.3 Create Command Suggestion Component

Create a component to display command suggestions:

```typescript
// client/src/components/chat/CommandSuggestions.tsx
import React from 'react';
import { LightBulbIcon, CheckIcon } from '@heroicons/react/24/outline';

interface CommandSuggestionsProps {
  suggestions: string[];
  onSelectCommand: (command: string) => void;
  isLoading: boolean;
}

const CommandSuggestions: React.FC<CommandSuggestionsProps> = ({
  suggestions,
  onSelectCommand,
  isLoading
}) => {
  // Implementation details
};
```

### 3. Backend Implementation

#### 3.1 Create Agent API Routes

Create new routes for agent functionality:

```javascript
// src/routes/agent.js
const express = require('express');
const router = express.Router();
const { requireAuth } = require('../middleware/auth');
const agentService = require('../services/agentService');
const { logger } = require('../utils/logger');

// Generate command suggestions
router.post('/suggest', requireAuth, async (req, res) => {
  try {
    const { input, context, modelId } = req.body;
    
    if (!input) {
      return res.status(400).json({ error: 'Input is required' });
    }
    
    const suggestions = await agentService.generateCommandSuggestions(
      input,
      context || {},
      modelId
    );
    
    res.json({ success: true, suggestions });
  } catch (err) {
    logger.error(`Error generating suggestions: ${err.message}`);
    res.status(500).json({ error: err.message || 'Failed to generate suggestions' });
  }
});

// Additional routes for agent functionality

module.exports = router;
```

#### 3.2 Integrate with MCP Service

Enhance the MCP service to support agent-specific functionality:

```javascript
// Add to src/services/mcpService.js

/**
 * Execute a command with agent context
 */
async function executeCommandWithContext(serverId, command, context) {
  // Implementation details
}

/**
 * Get terminal state from MCP server
 */
async function getTerminalState(serverId) {
  // Implementation details
}
```

### 4. Documentation Updates

Update the `aiagents.md` file with implementation details and usage instructions.

## Implementation Timeline

### Phase 1: Core Infrastructure (Week 1)
- Create agent service and context manager
- Implement command suggestion generation
- Set up API routes for agent functionality

### Phase 2: Frontend Integration (Week 2)
- Add agent toggle to chat input
- Create agent context provider
- Implement command suggestion UI
- Connect frontend to backend API

### Phase 3: Testing and Refinement (Week 3)
- Test agent functionality with various commands
- Refine suggestion generation
- Improve user experience
- Update documentation

## Technical Considerations

### LLM Integration
- Use Ollama for command suggestion generation
- Create specialized prompts for terminal command generation
- Implement proper error handling for LLM failures

### MCP Integration
- Ensure proper authentication and authorization
- Handle connection failures gracefully
- Implement timeout handling for long-running commands

### User Experience
- Provide clear visual feedback for agent status
- Make command suggestions easily accessible
- Allow for easy modification of suggested commands
- Implement proper error handling and recovery

## Next Steps

1. Implement the agent service and context manager
2. Add the agent toggle to the chat input
3. Create the command suggestion UI
4. Connect the frontend to the backend API
5. Test and refine the implementation
6. Update documentation
