# MCP AI Agent Implementation Tracker

## Overview
This document tracks the implementation progress of the MCP AI Agent feature. The agent will allow users to interact with MCP servers through the chat interface, with command approval workflows and result display.

## Core Objectives
1. Create an intelligent AI agent that integrates seamlessly with the user's configured MCP servers
2. Analyze user requests to determine when MCP tools would be helpful
3. Generate appropriate MCP commands based on user intent
4. Provide a transparent approval workflow for commands
5. Display results in a user-friendly manner

## Implementation Roadmap

### Phase 1: Eliminate Hardcoded Server Details & Set Up Toggle Activation ✅
- [x] **Remove all hardcoded MCP server references**
  - Remove any hardcoded IP addresses (e.g., 172.16.16.54)
  - Remove any hardcoded port numbers (e.g., 8080)
  - Remove any hardcoded server names or nicknames
  - Implement proper error handling for missing server configurations

- [x] **Implement MCP toggle activation flow**
  - Add toggle button to chat interface to activate agent mode
  - Ensure toggle is visually consistent with other UI elements
  - When toggled on, initiate the server retrieval process
  - When toggled off, gracefully disconnect from any active MCP server

### Phase 2: Server Retrieval and Selection ✅
- [x] **Implement dynamic server retrieval**
  - When toggle is activated, fetch user's MCP servers from the database
  - Use existing `/api/mcp/server/config` endpoint
  - Display loading indicator during server retrieval
  - Handle any errors during the retrieval process

- [x] **Create server selection interface**
  - Design a clean, simple interface for selecting from existing MCP servers
  - Show connection status for each server
  - Allow user to test connections before selecting
  - Remember the last used server as default for future sessions
  - Provide helpful messages if no servers are configured (direct user to settings page)

### Phase 3: MCP Connection Establishment ✅
- [x] **Implement connection to MCP endpoints**
  - Create functions to connect to key MCP endpoints:
    - `/info` - Get general server information
    - `/tools` - Discover available tools
    - `/messages` - Send and receive messages
    - `/sse` - Establish Server-Sent Events connection
  - Handle connection errors for each endpoint
  - Implement reconnection logic for dropped connections

- [x] **Develop connection status reporting**
  - Show connection status in the chat interface
  - Provide informative messages about connection state
  - Display helpful recovery suggestions for connection issues
  - Log connection events for debugging purposes

### Phase 4: Tool Discovery and AI Integration ✅
- [x] **Implement dynamic tool discovery**
  - Query MCP server's `/tools` endpoint
  - Parse and store tool definitions
  - Create schema for AI to understand available tools
  - Store discovered tools in the database

- [x] **Integrate with existing AI infrastructure**
  - Connect agent with existing AI model selection
  - Enhance AI prompts with MCP tool information
  - Create examples of proper tool usage for AI training
  - Develop system prompts for AI to understand MCP concepts

### Phase 5: Intelligent Command Processing ✅
- [x] **Develop request analysis capability**
  - Create system to analyze user messages for MCP tool relevance
  - Use existing AI model to understand user intent
  - Identify when MCP tools would be helpful
  - Suggest relevant tools based on message content

- [x] **Build command generation system**
  - Use AI to generate appropriate MCP commands
  - Format commands according to MCP protocol requirements
  - Handle parameter validation
  - Include explanations of what commands will do

- [x] **Create command approval workflow**
  - Display generated commands to user before execution
  - Allow user to modify commands if needed
  - Provide clear explanations of command purpose
  - Include option to cancel command execution

### Phase 6: Results Display and Error Handling ✅
- [x] **Implement result parsing and display**
  - Parse different result types from MCP server
  - Format and display results in chat interface
  - Handle multi-format results (text, JSON, etc.)
  - Include contextual information with results

- [x] **Develop robust error handling**
  - Create user-friendly error messages
  - Suggest recovery actions for common errors
  - Implement automatic retry logic where appropriate
  - Provide detailed logs for troubleshooting

### Phase 7: Iterative "Cycle-of-Thought" Loop Implementation 🔄
- [x] **Fix SSE Connection & ClientId Reliability Issues**
  - [x] Add dedicated event listener for 'connected' event in SSE handler
  - [x] Improve timeout logic with proper cleanup and race condition handling
  - [x] Implement client-side reconnection with exponential backoff
  - [x] Add verification step to ensure clientId is valid before allowing command execution
  - [x] Create fallback methods to reacquire clientId when missing

- [x] **Implement Tool Result Streaming & Processing**
  - [x] Update SSE event handler to process 'tool_result' events
  - [x] Create dispatcher to push tool results into React state via MCPAgentContext
  - [x] Add debounce/throttling for high-frequency tool output
  - [x] Implement progress indicators for long-running commands
  - [x] Handle special markers like [DONE] to detect command completion

- [x] **Design and Implement Analysis Loop**
  - [x] Extend CommandResult type to include followUp flag and analysis status
  - [x] Create pending analysis queue in MCPAgentContext
  - [x] Implement worker effect to process analysis queue
  - [x] Connect to /api/ai/analyze-result endpoint with proper context
  - [x] Update UI to display analysis results and follow-up suggestions

- [x] **Create Task Orchestration Flow**
  - [x] Implement Stop/Resume controls for user to pause processing
  - [x] Add task state machine (IDLE, GENERATING, EXECUTING, ANALYZING, STOPPED, COMPLETED)
  - [x] Create breadcrumb/history view of executed steps
  - [x] Develop completion detection (when LLM indicates task is done)
  - [x] Add session persistence so reload doesn't lose context

- [x] **Improve User Experience**
  - [x] Update ChatInput to integrate with MCP agent workflow
  - [x] Create visual indicators for active processing and state transitions
  - [x] Implement better formatting for different result types
  - [x] Add collapse/expand for large command outputs
  - [x] Create error recovery suggestions based on specific failure types
  
- [x] **Integrate with Database Models**
  - [x] Fix hardcoded model references in AI routes
  - [x] Implement proper model selection hierarchy using database settings
  - [x] Add model validation in MCPAgentCommands component
  - [x] Create helpful user guidance for model configuration
  - [x] Update documentation to reflect model selection process

## Recent Testing Findings

Through comprehensive testing of the MCP server at 172.16.16.54:8080, we've determined:

1. **SSE Connection Process**:
   - The server provides a `/sse` endpoint for establishing Server-Sent Events connections
   - Upon connection, the server sends a `{"type":"connected","clientId":"XXXX"}` message
   - This clientId is required for all subsequent tool interactions

2. **Command Execution Requirements**:
   - All commands to `/messages` endpoint must include a valid clientId from an active SSE connection
   - Without a valid clientId, the server returns `{"error":"Missing clientId","message":"Please provide a clientId from your SSE connection"}`
   - Command format requires: `{"clientId":"XXX","tool":"toolName","parameters":{...}}`

3. **Key Testing Results**:
   - HTTP connections work properly for `/info` and `/tools` endpoints
   - SSE connection returns a valid clientId which must be stored
   - Commands without a valid clientId are rejected with 400 Bad Request
   - Command execution succeeds when proper clientId is provided

## Technical Requirements

### Connection Flow
1. Toggle activates agent mode
2. System retrieves user's MCP connections from database
3. User selects desired MCP server from existing connections
4. Agent establishes connections to MCP endpoints:
   - `/info` - For server metadata
   - `/tools` - For discovering available tools
   - `/messages` - For sending commands
   - `/sse` - For establishing a Server-Sent Events connection for real-time updates
5. Agent analyzes user messages and generates appropriate commands
6. User approves commands before execution
7. Agent displays results in chat interface

### Database Integration
- Use the user_mcp_server_configurations table for server information
- Key fields to utilize:
  - id (UUID): Primary identifier
  - user_id (UUID): Link to user account
  - mcp_nickname: Human-readable server name
  - mcp_host: Server hostname or IP
  - mcp_port: Server port number
  - mcp_connection_status: Current connection state
  - mcp_last_error_message: Latest error information
  - mcp_discovered_tools_schema: JSON schema of available tools
  - is_default: Boolean indicating preferred server
  - created_at/updated_at: Tracking timestamps

### MCP Protocol Requirements
- SSE Connection: Establish Server-Sent Events connection to /sse endpoint
  - Store the clientId returned in the "connected" event
  - Maintain the SSE connection while MCP Agent is active
- Tool Discovery: Fetch available tools from /tools endpoint
- Command Invocation: Send commands to /messages endpoint with:
  - Valid clientId from SSE connection
  - Proper tool name and parameters format
- Result Handling: Process tool_result and error messages from SSE stream

### AI Integration
- Leverage existing AI model infrastructure for:
  - Command generation based on user intent
  - Tool selection based on request analysis
  - Result interpretation and explanation
  - Learning from user interactions

## User Experience Considerations

### Activation Experience
- Toggle should be clearly visible in chat interface
- Status indicators should show current state (connected/disconnected)
- Provide feedback during connection process
- Show helpful messages when no connections are configured

### Server Selection Experience
- Default to previously selected server when available
- Clearly indicate connection status with visual cues
- Provide guidance if no servers are configured
- Remember user preferences across sessions

### Command Approval Experience
- Show clear preview of commands before execution
- Allow editing of commands if needed
- Explain purpose and potential impact of commands
- Provide cancel option for safety

### Results Display Experience
- Format command results for maximum readability
- Highlight important information in results
- Provide context for interpreting command output
- Include appropriate error explanations

## Critical Issues to Address

### Current Implementation Problems
- ✅ Hardcoded server details (172.16.16.54:8080) must be removed
- ✅ Missing dynamic server retrieval and selection
- ✅ Lack of proper connection to all required MCP endpoints
- ✅ No integration with existing AI infrastructure for intelligent processing
- ✅ Incomplete command approval workflow
- ✅ Insufficient error handling

### Iterative Loop Implementation Issues
- ✅ **SSE ClientId Acquisition Issues**: Green connection badge appears but clientId remains null causing command rejection
- ✅ **SSE Event Handling Gap**: Tool result events not integrated into React state
- ✅ **Missing Analysis Loop**: No processing of command results to generate follow-up actions
- ✅ **Task Orchestration Incomplete**: No proper task state management or stop/resume controls

## Implementation Completed

### Key Components Updated:

1. **MCPContext.tsx** ✅
   - Removed hardcoded MCP server details
   - Added SSE connection management
   - Implemented clientId storage and management
   - Added error handling and reconnection logic
   - Enhanced command execution with clientId validation

2. **MCPAgentContext.tsx** ✅
   - Connected with AI infrastructure
   - Implemented proper MCP agent state management
   - Added tool discovery and processing
   - Developed command generation and execution flow
   - Added robust error handling with user-friendly messages

3. **MCPServerSelector.tsx** ✅
   - Improved UI for server selection
   - Added SSE connection status indicators
   - Enhanced server testing functionality
   - Added helpful guidance messages
   - Improved direct connection interface

4. **MCPCommandApproval.tsx** ✅
   - Displays generated commands for approval
   - Enables command editing
   - Provides clear purpose explanations
   - Includes cancel option

5. **MCPCommandResult.tsx** ✅
   - Formats various result types properly
   - Handles errors gracefully
   - Provides context for interpreting results
   - Supports multi-format results

6. **SSE Connection Handler** ✅
   - Established and maintained SSE connections
   - Process incoming SSE events
   - Handles connection errors and timeouts
   - Implements reconnection logic

7. **Enhanced SSE Connection & ClientId Management** ✅
   - Implemented multiple fallback mechanisms for clientId retrieval
   - Added local storage caching of clientId data
   - Created server-side proxy endpoint for clientId recovery
   - Implemented manual XHR-based retrieval for difficult cases
   - Added visual indicators and controls for clientId issues
   - Enhanced command execution with clientId validation and auto-retry

## Testing Milestones

### Connection Testing
- [x] Verify retrieval of MCP servers from database
- [x] Test connection establishment to all MCP endpoints
- [x] Validate server selection interface functionality
- [x] Confirm connection status updates correctly
- [x] Test clientId acquisition from various MCP server implementations
- [x] Verify clientId recovery mechanisms function properly

### Tool Discovery Testing
- [x] Verify tool discovery from MCP server
- [x] Confirm tool schema storage in database
- [x] Test tool categorization functionality
- [x] Validate tool refresh on server change

### Command Generation Testing
- [x] Test AI-based command generation
- [x] Verify parameter validation
- [x] Confirm command approval workflow
- [x] Test command modification capabilities

### Results Display Testing
- [x] Verify proper formatting of different result types
- [x] Test error handling scenarios
- [x] Confirm reconnection functionality
- [x] Validate overall user experience

### Iterative Loop Testing
- [x] Verify SSE clientId is properly acquired and stored
- [x] Test tool_result events are captured and displayed
- [x] Validate result analysis and follow-up command generation
- [x] Test multi-step workflows with continuous task execution
- [x] Verify stop/resume functionality works properly
- [x] Confirm task completion detection works reliably
- [x] Test clientId recovery and command retry when issues occur
- [x] Verify proper visual indicators for connection and clientId status

## Progress Tracking

### Completed Tasks
- ✅ Fixed MCPContext to remove hardcoded server details
- ✅ Added proper error handling for missing server configurations
- ✅ Updated MCPServerSelector with better UI and guidance
- ✅ Created MCPAgentContext for managing AI agent state
- ✅ Implemented MCPCommandApproval component for displaying and approving commands
- ✅ Created MCPCommandResult component for displaying command results
- ✅ Added MCPAgentCommands component to integrate into MessageList
- ✅ Updated toggle behavior to properly enable/disable the agent
- ✅ Integrated with existing AI model selection
- ✅ Validated connections to all required MCP endpoints (/info, /tools, /messages, /sse)
- ✅ Confirmed the AI model can successfully generate and execute MCP commands
- ✅ Completed MCP server testing to understand connection requirements
- ✅ Implemented SSE connection handler with proper clientId management
- ✅ Updated command execution flow to use clientId from SSE connection
- ✅ Enhanced error handling for connection issues and command failures
- ✅ Improved reconnection logic for dropped SSE connections
- ✅ Fixed hardcoded AI model issue
- ✅ Implemented Iterative "Cycle-of-Thought" Loop functionality
- ✅ Created MCPAgentControls component with stop/resume capability
- ✅ Added tool result processing system with event handlers and dispatchers
- ✅ Implemented analysis loop with pending queue and worker effect
- ✅ Added session persistence for command history and state
- ✅ Enhanced MCPAgentCommands to validate models and display appropriate warnings
- ✅ Updated documentation to reflect model selection priority and integration
- ✅ Implemented robust clientId acquisition and recovery mechanisms
- ✅ Created server-side proxy endpoint for clientId retrieval
- ✅ Added visual indicators and controls for clientId issues
- ✅ Enhanced command execution with automatic retry on clientId failures
- ✅ Improved session persistence with localStorage and sessionStorage

### Pending Tasks For Iterative Loop Implementation
All iterative loop tasks have been completed. The MCP AI Agent now features:
- ✅ Robust SSE connection with dedicated event listeners and reconnection logic
- ✅ Tool result streaming with dispatcher system
- ✅ Analysis loop for intelligent follow-up actions
- ✅ Task orchestration with state management
- ✅ User controls for stop/resume functionality
- ✅ Improved UX with processing indicators and error handling
- ✅ Database model integration with validation and fallbacks

## Validation Testing
1. ✅ Verify successful connection to 172.16.16.54:8080 MCP server
2. ✅ Verify the agent can receive and parse the tools from the server (16 tools available)
3. ✅ Confirm the AI can generate appropriate commands based on user requests
4. ✅ Test the command approval workflow with command modification
5. ✅ Validate proper display of command results in the chat interface
6. ✅ Test error scenarios and recovery mechanisms
7. ✅ Perform end-to-end testing of the entire user flow
8. ✅ Test iterative workflow with multiple steps
   - ✅ Verify clientId is properly acquired and used
   - ✅ Verify clientId can be recovered when missing
   - ✅ Test recovery mechanisms across different MCP server types
   - ✅ Test tool_result events flow through to UI
   - ✅ Validate analysis and follow-up generation
   - ✅ Test stop/resume functionality
   - ✅ Confirm completion detection

## Current Implementation Status (2024-07-30)

The MCP AI Agent implementation is now complete with all planned features implemented and critical issues resolved:

1. **Connection Management** ✅ 
   - Dynamic server selection from database
   - SSE connection with reliable clientId acquisition
   - Multiple fallback mechanisms for clientId recovery
   - Automatic reconnection with exponential backoff
   - Visual indicators of connection and clientId status

2. **Command Processing** ✅
   - AI-based command generation
   - User-friendly command approval UI
   - Parameter validation and editing
   - Execution with proper clientId validation

3. **Result Handling** ✅
   - Formatted display of different result types
   - Error handling with recovery suggestions
   - Tool result event processing
   - Result analysis with follow-up suggestions

4. **Iterative Loop** ✅
   - Continuous command-result-analysis cycle
   - Intelligent follow-up command generation
   - Task state management with user controls
   - Session persistence for context preservation

5. **User Experience** ✅
   - Intuitive controls for stop/resume
   - Clear visual indicators of processing state
   - Helpful guidance and error messages
   - Seamless integration with chat interface

6. **AI Integration** ✅
   - Proper model selection from database
   - Fallback mechanisms for model selection
   - Validation of selected models
   - Clear warnings for model configuration issues

The agent now provides a true "Cycle-of-Thought" capability, analyzing results and suggesting follow-up actions until the task is complete or the user stops the process.

## Integration with Chat System
- The MCP AI Agent is toggled via a button in the chat interface
- When enabled, it analyzes user messages for potential MCP tool usage
- Commands are generated and displayed for user approval
- Results are shown within the chat interface
- The agent maintains context across multiple interactions

## Resources and References
- MCP Protocol Documentation: documentation/mcp-integration/mcp-protocol-guide.md
- Server Configuration API: /api/mcp/server/config
- Tool Discovery Endpoint: /tools
- Database Structure: DatabaseStructure.md
- Test Scripts: mcp-curl-tests.ps1, mcp-agent-tests.ps1, mcp-curl-commands.txt

## Current Issues (2024-07-15)

1. **Connection Flow Issues**:
   - ✓ Server selection is working but the connection doesn't automatically establish SSE
   - ✓ After successful connection, AI agent doesn't initiate conversation
   - ✓ Command approval and execution flow needs integration with ChatInput

2. **User Experience Gaps**:
   - ✓ Missing visual indicators for active MCP connections in chat interface
   - ✓ No clear guidance for users on how to interact with MCP agent
   - ✓ Command approval UI needs better integration in message flow

3. **Iterative Loop Issues (2024-07-22)**:
   - [ ] SSE connection sometimes shows as connected but clientId is null
   - [ ] Tool result events from SSE aren't pushed into React state
   - [ ] No analysis of command results to determine next steps
   - [ ] Missing task orchestration loop (generate command → execute → analyze → repeat)
   - [ ] No stop/resume capability for multi-step workflows
   - [ ] Incomplete session persistence (lost context on page reload)

## Next Implementation Steps

1. **Improve Connection Flow**:
   - ✓ Ensure SSE connection is properly established after server selection
   - ✓ Add automatic initialization of agent after successful connection
   - ✓ Fix transition from server selection to chat interface

2. **Enhance AI Agent Interaction**:
   - ✓ Implement welcome message with server/tool information
   - ✓ Create initial agent message listing available capabilities
   - ✓ Add visual indicators for MCP mode in chat interface

3. **Complete Command Workflow**:
   - ✓ Ensure commands are properly displayed for approval
   - ✓ Implement command execution with proper SSE clientId
   - ✓ Add result display in chat interface

4. **Integration with ChatInput**:
   - ✓ Update ChatInput to recognize MCP mode
   - ✓ Add special handling for MCP commands
   - ✓ Implement command parsing and validation

5. **Implement Iterative Loop**:
   - [ ] Fix SSE clientId acquisition issues
     - [ ] Add 'connected' event listener
     - [ ] Improve timeout handling
     - [ ] Add verification before command execution
   - [ ] Implement tool result integration
     - [ ] Add 'tool_result' event listener 
     - [ ] Create dispatcher to update React state
     - [ ] Add progress indicators
   - [ ] Build analysis loop
     - [ ] Create pendingAnalysis queue
     - [ ] Implement worker effect
     - [ ] Connect to analyze-result endpoint
   - [ ] Create task orchestration
     - [ ] Add stop/resume controls
     - [ ] Implement task state machine
     - [ ] Add completion detection

6. **Testing and Refinement**:
   - 🔄 Test full user flow from connection to command execution
   - 🔄 Ensure error handling works correctly
   - 🔄 Optimize performance and user experience
   - [ ] Test iterative workflow with multi-step tasks
   - [ ] Verify session persistence

## Detailed Implementation Plan for Iterative Loop

### 1. Fix SSE Connection & ClientId Reliability

#### a. Update `connectSSE()` in MCPContext.tsx
```typescript
// Add specific event listener for 'connected' event
eventSource.addEventListener('connected', (event) => {
  try {
    const data = JSON.parse(event.data);
    if (data.clientId) {
      console.log(`Received clientId from SSE 'connected' event: ${data.clientId}`);
      setSSEConnection(prev => ({
        ...prev,
        clientId: data.clientId,
        status: 'connected',
        error: null
      }));
    }
  } catch (error) {
    console.error('Error parsing connected event:', error);
  }
});
```

#### b. Improve Timeout Logic
```typescript
// Replace current timeout with ref-based approach
const timeoutRef = useRef<number | null>(null);

// In connectSSE function
if (timeoutRef.current) {
  clearTimeout(timeoutRef.current);
}
timeoutRef.current = window.setTimeout(() => {
  if (sseConnection.status !== 'connected') {
    console.error('Timeout waiting for clientId');
    setSSEConnection(prev => ({
      ...prev,
      status: 'error',
      error: 'Timeout waiting for clientId from MCP server'
    }));
    disconnectSSE();
  }
  timeoutRef.current = null;
}, 10000);

// Clear on unmount
return () => {
  if (timeoutRef.current) {
    clearTimeout(timeoutRef.current);
    timeoutRef.current = null;
  }
};
```

#### c. Add ClientId Verification
```typescript
// Add verification method
const verifyClientId = () => {
  if (!sseConnection.clientId) {
    console.error('No valid clientId available');
    reconnectSSE();
    return false;
  }
  return true;
};

// Update executeCommand
const executeCommand = async (toolName, parameters) => {
  if (!verifyClientId()) {
    throw new Error('No valid clientId. Please reconnect to the MCP server.');
  }
  // Existing code...
};
```

### 2. Implement Tool Result Streaming & Processing

#### a. Add Tool Result Listener to SSE
```typescript
// In connectSSE function
eventSource.addEventListener('tool_result', (event) => {
  try {
    console.log('Tool result received:', event.data);
    const data = JSON.parse(event.data);
    dispatchToolResult(data);
  } catch (error) {
    console.error('Error parsing tool_result event:', error);
  }
});
```

#### b. Create Dispatch Method in MCPContext
```typescript
// Add to MCPContext
const dispatchToolResult = (data) => {
  if (mcpAgentDispatcher && typeof mcpAgentDispatcher === 'function') {
    mcpAgentDispatcher(data);
  }
};

// Export via context
return (
  <MCPContext.Provider value={{
    // Existing values...
    dispatchToolResult
  }}>
    {children}
  </MCPContext.Provider>
);
```

#### c. Register Dispatcher in MCPAgentContext
```typescript
// In MCPAgentProvider
const { dispatchToolResult } = useMCP();

// Set up dispatcher ref
const dispatcherRef = useRef((data) => {
  // Convert tool result to CommandResult
  const commandResult = {
    id: `result-${Date.now()}-${Math.floor(Math.random() * 1000)}`,
    commandId: data.commandId || 'unknown',
    success: !data.error,
    result: data.result || data,
    error: data.error,
    timestamp: Date.now(),
    needsAnalysis: true
  };
  
  // Add to results
  addCommandResult(commandResult);
  
  // Queue for analysis if needed
  if (!data.error) {
    addToPendingAnalysis(commandResult);
  }
});

// Register dispatcher with MCPContext
useEffect(() => {
  mcpAgentDispatcher.current = dispatcherRef.current;
  return () => {
    mcpAgentDispatcher.current = null;
  };
}, [dispatchToolResult]);
```

### 3. Design and Implement Analysis Loop

#### a. Add State for Analysis Queue
```typescript
// In MCPAgentProvider
const [pendingAnalysis, setPendingAnalysis] = useState<CommandResult[]>([]);
const [isAnalyzing, setIsAnalyzing] = useState(false);
const [isStopped, setIsStopped] = useState(false);
```

#### b. Create Analysis Worker Effect
```typescript
// Analysis worker effect
useEffect(() => {
  if (
    isAgentEnabled && 
    !isProcessing && 
    !isAnalyzing && 
    pendingAnalysis.length > 0 && 
    !isStopped
  ) {
    const analyzeNextResult = async () => {
      setIsAnalyzing(true);
      
      try {
        const resultToAnalyze = pendingAnalysis[0];
        
        // Find the original command
        const originalCommand = pendingCommands.find(
          cmd => cmd.id === resultToAnalyze.commandId
        );
        
        if (!originalCommand) {
          console.warn(`Could not find original command for result: ${resultToAnalyze.id}`);
          setPendingAnalysis(prev => prev.slice(1));
          setIsAnalyzing(false);
          return;
        }
        
        // Call analysis endpoint
        const response = await axios.post('/api/ai/analyze-result', {
          result: resultToAnalyze.result,
          originalRequest: originalCommand.command,
          toolName: originalCommand.toolName,
          parameters: originalCommand.parameters,
          modelId: activeModel
        });
        
        if (response.data) {
          // Add analysis as a command result
          const analysisResult = {
            id: `analysis-${Date.now()}`,
            commandId: 'analysis',
            success: true,
            result: {
              type: 'text',
              text: response.data.analysis
            },
            timestamp: Date.now()
          };
          
          addCommandResult(analysisResult);
          
          // Add follow-up commands if any
          if (
            response.data.followUpCommands && 
            Array.isArray(response.data.followUpCommands) && 
            response.data.followUpCommands.length > 0
          ) {
            const newCommands = response.data.followUpCommands.map(cmd => ({
              id: `cmd-${Date.now()}-${Math.floor(Math.random() * 1000)}`,
              command: cmd.command,
              toolName: cmd.toolName,
              parameters: cmd.parameters,
              description: cmd.description,
              timestamp: Date.now()
            }));
            
            setPendingCommands(prev => [...prev, ...newCommands]);
          } else {
            // No follow-up commands, task is complete
            const completionMessage = {
              id: `completion-${Date.now()}`,
              commandId: 'completion',
              success: true,
              result: {
                type: 'text',
                text: 'Task completed. No further actions needed.'
              },
              timestamp: Date.now()
            };
            
            addCommandResult(completionMessage);
          }
        }
      } catch (error) {
        console.error('Error analyzing result:', error);
        
        // Add error message
        const errorResult = {
          id: `analysis-error-${Date.now()}`,
          commandId: 'error',
          success: false,
          result: null,
          error: `Error analyzing result: ${error.message}`,
          timestamp: Date.now()
        };
        
        addCommandResult(errorResult);
      } finally {
        // Remove the analyzed result from the queue
        setPendingAnalysis(prev => prev.slice(1));
        setIsAnalyzing(false);
      }
    };
    
    analyzeNextResult();
  }
}, [
  isAgentEnabled, 
  isProcessing, 
  isAnalyzing, 
  pendingAnalysis, 
  isStopped, 
  pendingCommands,
  activeModel
]);
```

### 4. Create Task Orchestration Flow

#### a. Add Stop/Resume Controls
```tsx
// In MCPAgentCommands.tsx
{isProcessing || isAnalyzing ? (
  <div className="flex items-center space-x-2 p-3 rounded-md mb-4" 
    style={{ backgroundColor: 'var(--color-surface-light)' }}>
    <div className="animate-pulse h-3 w-3 bg-blue-500 rounded-full"></div>
    <div className="animate-pulse h-3 w-3 bg-blue-500 rounded-full" style={{ animationDelay: '0.2s' }}></div>
    <div className="animate-pulse h-3 w-3 bg-blue-500 rounded-full" style={{ animationDelay: '0.4s' }}></div>
    <span className="text-sm" style={{ color: 'var(--color-text-muted)' }}>
      {isProcessing ? 'Processing your request...' : 'Analyzing results...'}
    </span>
    
    <button
      onClick={() => setIsStopped(true)}
      className="ml-2 px-2 py-1 text-xs rounded"
      style={{ backgroundColor: 'var(--color-error-bg)', color: 'var(--color-error)' }}
    >
      Stop
    </button>
  </div>
) : isStopped && (pendingAnalysis.length > 0 || pendingCommands.length > 0) ? (
  <div className="flex items-center space-x-2 p-3 rounded-md mb-4" 
    style={{ backgroundColor: 'var(--color-surface-light)' }}>
    <span className="text-sm" style={{ color: 'var(--color-text-muted)' }}>
      Processing paused. Would you like to continue?
    </span>
    
    <button
      onClick={() => setIsStopped(false)}
      className="ml-2 px-2 py-1 text-xs rounded"
      style={{ backgroundColor: 'var(--color-success-bg)', color: 'var(--color-success)' }}
    >
      Resume
    </button>
  </div>
) : null}
```

#### b. Add Session Persistence
```typescript
// In MCPAgentProvider
// Save state to sessionStorage
useEffect(() => {
  if (isAgentEnabled && commandResults.length > 0) {
    sessionStorage.setItem('mcp_command_results', JSON.stringify(commandResults));
  }
}, [isAgentEnabled, commandResults]);

useEffect(() => {
  if (isAgentEnabled && pendingCommands.length > 0) {
    sessionStorage.setItem('mcp_pending_commands', JSON.stringify(pendingCommands));
  }
}, [isAgentEnabled, pendingCommands]);

// Restore state from sessionStorage
useEffect(() => {
  if (isAgentEnabled && isConnected) {
    const savedResults = sessionStorage.getItem('mcp_command_results');
    const savedCommands = sessionStorage.getItem('mcp_pending_commands');
    
    if (savedResults && commandResults.length === 0) {
      try {
        setCommandResults(JSON.parse(savedResults));
      } catch (e) {
        console.error('Error restoring command results:', e);
      }
    }
    
    if (savedCommands && pendingCommands.length === 0) {
      try {
        setPendingCommands(JSON.parse(savedCommands));
      } catch (e) {
        console.error('Error restoring pending commands:', e);
      }
    }
  }
}, [isAgentEnabled, isConnected]);
```

## Implementation Completed (2024-07-15)

1. **MCPStatusIndicator Component**:
   - Created a new component to display MCP connection status in chat interface
   - Shows different status indicators based on connection state (connected, error, connecting)
   - Includes server information and reconnect button
   - Provides helpful guidance messages

2. **Enhanced MCPAgentCommands**:
   - Added welcome message with server and tools information
   - Incorporated status indicator for better visibility
   - Added examples of commands users can try
   - Added processing indicator for better feedback

3. **Fixed MCPServerSelector**:
   - Improved connection flow with better SSE handling
   - Added connecting indicator for clearer feedback
   - Added automatic close of selector upon successful connection
   - Added error handling for connection failures
   - Ensured agent is automatically enabled after connection

4. **Completed Integration**:
   - Connected all components for seamless user experience
   - Ensured proper transitions between connection and chat
   - Added visual feedback throughout the process
   - Improved error handling and recovery mechanisms

## Iterative Loop Implementation Tasks (2024-07-22)

1. **Robust SSE Connection**:
   - Fix clientId acquisition issues with better event handling
   - Implement verification before command execution
   - Add reconnection with exponential backoff

2. **Tool Result Integration**:
   - Add event listener for tool_result events
   - Implement dispatcher to update React state
   - Create progress indicators for long-running commands

3. **Analysis Loop Implementation**:
   - Build queueing system for results that need analysis
   - Implement worker effect to process queue
   - Connect to analyze-result endpoint with proper context

4. **Task Orchestration**:
   - Add stop/resume controls for better user experience
   - Implement task state machine to track progress
   - Create completion detection based on LLM response

5. **User Experience Improvements**:
   - Update ChatInput integration
   - Add visual indicators for state transitions
   - Improve result formatting for different output types
   - Add session persistence to maintain context

## Target User Experience

1. User activates MCP toggle
2. User selects a server and connects
3. AI agent automatically initializes and sends welcome message
4. User interacts with agent naturally in chat
5. Agent generates commands for user approval
6. User approves commands and sees results in chat
7. Agent analyzes results and suggests follow-up actions
8. Process continues until task is complete or user stops it
9. Agent maintains context even after page reload

## Implementation Milestones for Iterative Loop

1. **SSE ClientId Reliability**:
   - [ ] Fix connectSSE with dual-approach event listeners
   - [ ] Add clientId verification before command execution
   - [ ] Implement exponential backoff for reconnection

2. **Result Processing**:
   - [ ] Implement tool_result event handler
   - [ ] Create dispatcher in MCPContext
   - [ ] Register handler in MCPAgentContext
   - [ ] Add progress indicators for long-running commands

3. **Analysis Loop**:
   - [ ] Define CommandResult extension with analysis flags
   - [ ] Implement pendingAnalysis queue
   - [ ] Create analysis worker effect
   - [ ] Update AI endpoints to support follow-up commands

4. **Task Orchestration**:
   - [ ] Implement stop/resume controls
   - [ ] Create task state machine
   - [ ] Add session persistence
   - [ ] Implement completion detection

5. **Testing & Refinement**:
   - [ ] Test iterative workflows with various tools
   - [ ] Validate error handling and recovery
   - [ ] Optimize performance for large outputs
   - [ ] Perform end-to-end testing with complete tasks
