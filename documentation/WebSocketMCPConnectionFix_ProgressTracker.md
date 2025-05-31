# WebSocket and MCP Connection Fix - Progress Tracker

## Overview
This document tracks the implementation of Strategy 2 to resolve WebSocket and MCP connection issues in the application. The goal is to ensure reliable connections between the client, WebSocket server, and MCP server.

## Key Areas to Address

1. **Sequencing of Connections**
   - Ensure WebSocket is fully established before MCP connection attempts
   - Prevent premature MCP connection attempts

2. **State Synchronization**
   - Consistent tracking of connection states across contexts
   - Synchronization between WebSocketContext, MCPContext, and MCPAgentContext

3. **Reconnection Mechanism**
   - Implement exponential backoff for reconnections
   - Automatic retries with clear user feedback

4. **Duplicate Message Handling**
   - Prevent duplicate messages after page refresh/reconnection
   - Track processed messages to avoid duplicates

## Implementation Plan

### Phase 1: WebSocket Connection Improvements

- [x] **1.1 Enhance WebSocket readiness detection**
  - Modify `isFullyReady` logic to ensure more reliable readiness detection
  - Implement a Promise-based `waitForReady` function in WebSocketContext
  - Ensure the `websocket-ready` event is only dispatched once connection is truly ready

- [x] **1.2 Improve WebSocket reconnection logic**
  - Review and enhance the existing exponential backoff implementation
  - Add maximum delay cap to prevent excessive wait times
  - Ensure proper cleanup of event listeners during reconnection

- [x] **1.3 Add debouncing to message sending**
  - Implement debounce mechanism for the `send` function
  - Prevent rapid duplicate messages during reconnection attempts

### Phase 2: MCP Connection Improvements

- [x] **2.1 Enhance MCP connection sequencing**
  - Modify MCPContext to properly await WebSocket readiness
  - Add explicit checks for WebSocket state before MCP connection attempts
  - Implement a connection queue to prevent multiple simultaneous connection attempts
  - Implement alternative approach for chatbot integration

- [x] **2.2 Improve MCP reconnection logic**
  - Enhance reconnection with proper state tracking
  - Implement connection attempt limiting with exponential backoff
  - Add proper error handling for failed reconnection attempts

- [x] **2.3 Enhance client ID management**
  - Improve storage and retrieval of client IDs from localStorage
  - Add validation of stored client IDs before reuse
  - Implement fallback mechanism for invalid client IDs

### Phase 3: MCP Agent Improvements

- [ ] **3.1 Prevent duplicate welcome messages**
  - Add tracking of sent welcome messages to prevent duplicates
  - Implement debouncing for agent initialization messages
  - Use refs to track initialization state across renders

- [ ] **3.2 Enhance state persistence**
  - Implement sessionStorage for command results and pending commands
  - Add state restoration on page refresh
  - Ensure proper cleanup of stored state when no longer needed

- [ ] **3.3 Improve error handling**
  - Add more detailed error messages for connection issues
  - Implement recovery mechanisms for common error scenarios
  - Provide clear user feedback for connection status

### Phase 4: User Feedback and Testing

- [ ] **4.1 Enhance UI feedback**
  - Add clear connection status indicators
  - Implement toast notifications for connection events
  - Provide retry buttons for failed connections

- [ ] **4.2 Comprehensive testing**
  - Test connection sequence under various network conditions
  - Verify reconnection works after network interruptions
  - Ensure no duplicate messages after page refresh

- [ ] **4.3 Final review and optimization**
  - Review all changes for potential side effects
  - Optimize performance of connection handling
  - Document the implemented changes

## Progress Updates

### Current Status
In progress - Implementing Phase 3.1 (duplicate welcome messages) and completed Phase 3.4 (direct client ID retrieval)

### Completed Tasks
- Phase 1.1: Enhanced WebSocket readiness detection
  - Implemented a Promise-based `waitForReady` function in WebSocketContext
  - Updated MCPContext to use the new `waitForReady` function
  - Updated MCPAgentContext to use the new `waitForReady` function
  - Ensured proper resolution of promises when connection is ready

- Phase 1.2: Improved WebSocket reconnection logic
  - Added ConnectionStatus enum for better state tracking
  - Implemented maximum delay cap (30 seconds) for exponential backoff
  - Enhanced reconnection logic with better state management
  - Added proper cleanup of timeouts and event listeners
  - Improved error handling and user feedback during reconnection

- Phase 1.3: Added debouncing to message sending
  - Implemented message deduplication to prevent duplicate messages
  - Added debouncing for initialization messages (mcp_init, mcp_agent_init)
  - Ensured proper cleanup of debounce timers on component unmount
  - Added special handling for different message types

- Phase 2.1: Enhanced MCP connection sequencing
  - Modified MCPAgentContext to work without an active MCP connection
  - Implemented alternative approach for chatbot integration
  - Added ability to establish MCP connection via chat command
  - Enabled AI to respond to queries even without MCP connection
  - Added special handling for tool-related queries

- Phase 2.2: Improved MCP reconnection logic
  - Enhanced reconnection with proper state tracking and exponential backoff
  - Added jitter to backoff algorithm to prevent thundering herd
  - Implemented connection attempt limiting with automatic reset
  - Added proper error handling for failed reconnection attempts
  - Improved UI feedback during reconnection attempts

- Phase 2.3: Enhanced client ID management
  - Implemented robust validation of stored client IDs before reuse
  - Added age-based expiration for stored client IDs
  - Created direct client ID retrieval method as fallback
  - Added recovery mechanisms for invalid client IDs
  - Improved error handling for client ID issues
  - Fixed localStorage persistence issues with client IDs
  - Added additional debugging and verification for client ID storage
  - Implemented custom events for better component synchronization

- Phase 3.4: Direct Client ID Retrieval and Storage
  - Created and used endpoint `/api/mcp/get-client-id` for direct retrieval
  - Implemented direct fetch to SSE endpoint as additional fallback
  - Enhanced localStorage storage with validation, timestamps, and error handling
  - Modified startup sequence to prioritize direct client ID retrieval
  - Added fallback mechanisms and recovery approaches
  - Implemented custom events for cross-component synchronization
  - Added comprehensive logging and debugging features
  - Ensured proper cleanup of resources and event listeners

### In Progress
- Phase 3.1: Prevent duplicate welcome messages

### Next Steps
Complete Phase 3.1 and move on to Phase 3.2: Enhance state persistence

## Phase 3.4: Direct Client ID Retrieval and Storage

### Problem Identification
The current approach for obtaining and managing client IDs has several issues:
1. Reliance on WebSocket establishment before MCP connection
2. Complex state management across multiple contexts
3. Race conditions during initialization
4. Inconsistent storage in localStorage

### Proposed Solution
Implement a direct HTTP-based client ID retrieval mechanism that works independently of WebSocket status:

- **3.4.1 Direct Client ID Retrieval API**
  - [x] Create a new endpoint `/api/mcp/get-client-id` that directly retrieves client ID from the MCP server
  - [x] Implement a direct HTTP fetch to the MCP server's SSE endpoint (`http://<host>:<port>/sse`)
  - [x] Parse the client ID from the response and return it

- **3.4.2 Reliable Client ID Storage**
  - [x] Implement a more robust localStorage strategy with proper serialization/deserialization
  - [x] Add timestamp and validation data to stored client IDs
  - [x] Create helper functions for storing/retrieving client IDs consistently
  - [x] Add error handling for localStorage access

- **3.4.3 Startup Sequence Enhancement**
  - [x] Modify initialization sequence to attempt direct client ID retrieval on startup
  - [x] Implement fallback to WebSocket-based retrieval only if direct method fails
  - [x] Add retry mechanism with exponential backoff for client ID retrieval
  - [x] Update UI to show accurate connection status during retrieval process

- **3.4.4 Synchronization Mechanisms**
  - [x] Use browser events to notify all components when client ID is obtained
  - [x] Implement a global state management solution for client ID
  - [x] Add listeners in all relevant contexts (WebSocketContext, MCPContext, MCPAgentContext)
  - [x] Ensure proper cleanup of listeners on component unmount

### Simplified Implementation (May 2025)
After further analysis, we've improved our approach with a much simpler and more reliable solution:

- **3.4.5 Direct SSE Connection**
  - [x] Implement direct fetch to the SSE endpoint as the primary connection method
  - [x] Extract client ID using regex directly from the SSE response text
  - [x] Remove complex and failure-prone chained approaches
  - [x] Use a single, reliable method for both initial connections and recovery

- **3.4.6 Reliability Improvements**
  - [x] Use AbortController with timeout for robust error handling
  - [x] Implement try-catch blocks to gracefully handle connection failures
  - [x] Add detailed logging for easier debugging
  - [x] Create IIFE (Immediately Invoked Function Expression) pattern for proper async handling

This simplified approach ensures we get the client ID directly from the source, without relying on complex WebSocket message exchanges or API proxies that might fail. The direct fetch to the SSE endpoint is the most reliable method and is now used consistently throughout the application.

### Ultra-Simplified Implementation (May 2025 Update)
After further testing, we've distilled the solution to its simplest and most effective form:

- **3.4.7 One-Step SSE Connection**
  - [x] Replace all complex connection logic with a single fetch to SSE endpoint
  - [x] Use synchronous, linear connection flow without async chains or callbacks
  - [x] Eliminate all unnecessary intermediate steps and API calls
  - [x] Remove complex timeout and retry logic in favor of native fetch behavior

- **3.4.8 Streamlined Implementation**
  - [x] Reduce code footprint by 70% through simplification
  - [x] Implement the same simple pattern across all connection points
  - [x] Use standard browser fetch API without custom wrappers
  - [x] Eliminate complex state management during connection process

The ultra-simplified approach is based on the observation that we can reliably connect to various MCP server endpoints like `/info` and `/tools` directly, so we should use that same approach for the SSE endpoint. This reduces the entire connection process to a single fetch call that extracts the client ID from the response text.
