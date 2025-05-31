# AI Agent Context Memory System

## Overview

Instead of embedding all tool information in a large system prompt, we can create a context memory system that allows the AI to:

1. Access tool information on-demand using a `read_context` tool
2. Refresh its understanding of available tools and their usage
3. Maintain a more concise system prompt while having access to detailed tool documentation

This approach has several advantages:
- Reduces system prompt size
- Allows for dynamic updates to tool documentation
- Enables the AI to selectively access only the information it needs
- Provides a more natural way for the AI to "remember" how to use tools

## Implementation Approach

### 1. Context Memory Store

Create a structured storage system for tool documentation and usage examples:

```javascript
// Example context memory store structure
const contextMemoryStore = {
  tools: {
    // Tool definitions with detailed documentation
    runShellCommand: {
      name: "runShellCommand",
      description: "Execute a shell command on the server",
      parameters: {
        command: {
          type: "string",
          description: "The command to execute",
          required: true
        },
        workingDir: {
          type: "string",
          description: "Working directory for the command",
          required: false
        }
      },
      examples: [
        {
          description: "List files in current directory",
          command: "runShellCommand",
          parameters: { command: "ls -la" }
        },
        {
          description: "Check system information",
          command: "runShellCommand",
          parameters: { command: "systeminfo" }
        }
      ]
    },
    // Additional tools...
  },
  
  usage_patterns: {
    // Common usage patterns and workflows
    file_operations: {
      description: "Common file operation patterns",
      examples: [
        "To list files, use runShellCommand with 'ls -la'",
        "To search for text in files, use grep with appropriate parameters"
      ]
    }
  }
};
```

### 2. `read_context` Tool Implementation

Create a tool that allows the AI to query the context memory:

```javascript
// Backend implementation of read_context tool
app.post('/api/mcp/agent/read-context', async (req, res) => {
  try {
    const { contextType, contextName, query } = req.body;
    
    // Handle different types of context queries
    if (contextType === 'tool' && contextName) {
      // Return specific tool documentation
      const toolInfo = contextMemoryStore.tools[contextName];
      if (toolInfo) {
        return res.json({ context: toolInfo });
      } else {
        return res.status(404).json({ error: `Tool '${contextName}' not found` });
      }
    } 
    else if (contextType === 'tools') {
      // Return list of available tools with brief descriptions
      const toolList = Object.keys(contextMemoryStore.tools).map(key => ({
        name: contextMemoryStore.tools[key].name,
        description: contextMemoryStore.tools[key].description
      }));
      return res.json({ context: toolList });
    }
    else if (contextType === 'usage_pattern' && contextName) {
      // Return specific usage pattern
      const patternInfo = contextMemoryStore.usage_patterns[contextName];
      if (patternInfo) {
        return res.json({ context: patternInfo });
      } else {
        return res.status(404).json({ error: `Usage pattern '${contextName}' not found` });
      }
    }
    else if (contextType === 'search' && query) {
      // Search across all context information
      const results = searchContextMemory(query);
      return res.json({ context: results });
    }
    
    return res.status(400).json({ error: 'Invalid context request' });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// Helper function to search context memory
function searchContextMemory(query) {
  const results = {
    tools: [],
    usage_patterns: []
  };
  
  // Search tools
  for (const [key, tool] of Object.entries(contextMemoryStore.tools)) {
    if (
      tool.name.includes(query) || 
      tool.description.includes(query) ||
      JSON.stringify(tool.parameters).includes(query)
    ) {
      results.tools.push({
        name: tool.name,
        description: tool.description
      });
    }
  }
  
  // Search usage patterns
  for (const [key, pattern] of Object.entries(contextMemoryStore.usage_patterns)) {
    if (
      pattern.description.includes(query) ||
      pattern.examples.some(ex => ex.includes(query))
    ) {
      results.usage_patterns.push({
        name: key,
        description: pattern.description
      });
    }
  }
  
  return results;
}
```

### 3. AI Integration

Modify the AI system prompt to include instructions about using the context memory:

```javascript
const systemPrompt = `
You are an AI assistant with access to a context memory system.
When you need information about available tools or how to use them, you can use the read_context tool.

Examples of using read_context:
1. To get a list of all available tools:
   read_context(contextType: "tools")

2. To get detailed information about a specific tool:
   read_context(contextType: "tool", contextName: "runShellCommand")

3. To search for tools or usage patterns related to a topic:
   read_context(contextType: "search", query: "file operations")

4. To get common usage patterns:
   read_context(contextType: "usage_pattern", contextName: "file_operations")

Always check the context memory before suggesting commands to ensure you're using the correct syntax and parameters.
`;
```

### 4. Frontend Integration

Update the chat interface to handle context memory queries:

```jsx
// Example function to handle context memory queries in the chat flow
const handleContextMemoryQuery = async (contextType, contextName, query) => {
  try {
    const response = await api.post('/api/mcp/agent/read-context', {
      contextType,
      contextName,
      query
    });
    
    return response.data.context;
  } catch (error) {
    console.error('Error querying context memory:', error);
    return { error: 'Failed to access context memory' };
  }
};

// Example of how to integrate with AI message processing
const processAIMessage = async (message) => {
  // Check if message contains a context memory query
  const contextQueries = extractContextQueries(message);
  
  if (contextQueries.length > 0) {
    // Process each context query
    for (const query of contextQueries) {
      const contextResult = await handleContextMemoryQuery(
        query.contextType,
        query.contextName,
        query.query
      );
      
      // Replace the query placeholder with the actual context information
      message = replaceContextQueryWithResult(message, query, contextResult);
    }
  }
  
  return message;
};
```

## Usage Flow

1. **AI Needs Tool Information**:
   - AI recognizes it needs information about a specific tool
   - AI uses the `read_context` tool to query the context memory
   - System returns the requested information
   - AI uses this information to formulate a command suggestion

2. **Continuous Learning**:
   - As the AI interacts with tools, it can update its understanding
   - AI can query for usage patterns or examples when needed
   - System can dynamically update the context memory with new tools or improved documentation

3. **Selective Context Loading**:
   - Instead of loading all tool documentation upfront, AI loads only what it needs
   - Reduces token usage and allows for more efficient conversations
   - Enables more detailed documentation for each tool

## Benefits

1. **Reduced System Prompt Size**: The system prompt can be much smaller, focusing on core instructions rather than tool documentation.

2. **Dynamic Updates**: Tool documentation can be updated without changing the system prompt.

3. **Selective Access**: AI can access only the information it needs for the current task.

4. **Better Organization**: Tool documentation can be structured and organized more effectively.

5. **Improved Learning**: AI can learn from usage patterns and examples over time.

## Next Steps

1. **Create Context Memory Store**: Implement the structured storage for tool documentation.

2. **Implement `read_context` Tool**: Create the backend endpoint for accessing context memory.

3. **Update AI System Prompt**: Modify the system prompt to include instructions for using context memory.

4. **Integrate with Chat Flow**: Update the chat interface to handle context memory queries.

5. **Populate Documentation**: Create comprehensive documentation for all available tools.
