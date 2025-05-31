# WebSocket Integration Plan for Product Demo

## 1. Introduction

This document outlines the strategic plan for integrating WebSocket technology into the Product Demo application. The goal is to enhance real-time communication capabilities, improve user experience by providing instant updates, and reduce server load compared to traditional polling methods. This plan builds upon the initial analysis and proposals in `documentation/websocket-integration.md`.

WebSockets will provide a persistent, bidirectional communication channel between the client and server, enabling features such as live document processing updates, real-time RAG query feedback, interactive chat, and live MCP installation monitoring.

## 2. Goals

*   **Enhance User Experience**: Provide users with immediate visual feedback for ongoing processes and data changes.
*   **Reduce Server Load**: Replace HTTP polling mechanisms with efficient WebSocket push notifications.
*   **Enable New Real-time Features**: Lay the groundwork for future real-time collaborative and interactive functionalities.
*   **Standardize Real-time Communication**: Create a unified WebSocket layer for various real-time features.
*   **Maintain Security**: Ensure WebSocket communication is as secure as existing HTTP communication, incorporating robust authentication and authorization.

## 3. Security Considerations

Security is paramount. The WebSocket integration will adhere to the following principles:

*   **WSS (WebSocket Secure)**: All WebSocket connections in production environments must use WSS, which encrypts data in transit, analogous to HTTPS. The client will dynamically choose `ws:` or `wss:` based on `window.location.protocol`.
*   **Authentication**:
    *   WebSocket connections will be authenticated during the initial HTTP handshake.
    *   The existing session cookie (`connect.sid`) will be used to identify and authenticate the user. The `src/websocket/auth.js` module (as proposed in the initial plan) will handle parsing this cookie and verifying the session.
    *   Unauthenticated connection attempts will be rejected and closed immediately.
*   **Authorization**:
    *   Once authenticated, user roles and permissions (already managed by the application) will be respected for any actions or data subscriptions over WebSockets.
    *   Sensitive operations or data streams will require appropriate authorization checks on the server before allowing subscription or sending data.
*   **Input Validation**: All messages received from clients over WebSockets must be rigorously validated on the server to prevent malicious payloads.
*   **Origin Validation**: The WebSocket server should be configured to check the `Origin` header of incoming handshake requests to allow connections only from authorized client origins.
*   **Rate Limiting/Throttling**: Consider implementing rate limiting for messages sent from clients to prevent abuse, if applicable to specific use cases.
*   **Data Sanitization**: Ensure any data broadcast to clients is properly sanitized if it includes user-generated content that might be displayed.

## 4. Proposed Architecture

The WebSocket architecture will be modular on both the backend and frontend.

### 4.1. Backend Architecture

*   **WebSocket Server Instance**: A single WebSocket server instance (using the `ws` library) will be attached to the existing Express HTTP server.
*   **Connection Management**: The server will manage active client connections, associating them with authenticated user IDs. This allows for targeted broadcasting of messages.
*   **Message Handling**:
    *   A central message handler or router might be beneficial for processing messages received from clients (if clients send more than just subscription requests).
    *   For server-to-client communication, services will trigger events to be broadcast over WebSockets.
*   **Modularity**:
    *   `src/websocket/server.js`: Core WebSocket server setup, connection lifecycle management (connect, disconnect, error), and broadcast utilities.
    *   `src/websocket/auth.js`: Handles authentication for WebSocket connections.
    *   `src/websocket/messageHandlers.js` (Optional, for client-to-server messages): Logic to process different types of messages from clients.
    *   Existing services (e.g., `src/services/documentService.js`, `src/services/ragService.js`) will be modified to use the WebSocket server for broadcasting updates.

### 4.2. Frontend Architecture

*   **WebSocket Context**: A React Context (`WebSocketContext`) will manage the WebSocket connection lifecycle (connect, disconnect, message handling) and provide a clean interface (`useWebSocket` hook) for components.
*   **Component Subscription**: Components requiring real-time updates will use the `useWebSocket` hook to receive messages and update their state accordingly.
*   **Message Dispatching**: The `WebSocketContext` can either provide raw messages for components to filter or include a simple dispatcher/event emitter for specific message types.

## 5. Implementation Details

### 5.1. Backend Implementation

**Step 1: Setup WebSocket Server (`src/websocket/server.js`)**
*   (As per `documentation/websocket-integration.md`)
*   Initialize `ws.Server` and attach to the existing HTTP server.
*   Implement connection handling:
    *   On `connection`: Authenticate using `src/websocket/auth.js`. Store authenticated client `ws` instances, mapped by `userId` (e.g., `Map<userId, Set<ws>>`).
    *   On `message`: (If client-to-server messages are needed beyond simple pings) Route message to a handler.
    *   On `close`: Clean up client connection from the storage.
    *   On `error`: Log errors.
*   Implement broadcast functions:
    *   `broadcastToUser(userId, data)`: Sends data to all WebSocket connections for a specific user.
    *   `broadcastToAll(data)`: Sends data to all connected and authenticated clients (use with caution, typically for system-wide notifications).
    *   `broadcastToRoom(roomId, data)` (Future): If channel/room-based communication is needed.

**Step 2: WebSocket Authentication (`src/websocket/auth.js`)**
*   (As per `documentation/websocket-integration.md`)
*   Function `authenticateWebSocket(httpRequest)`:
    *   Parses `cookie` header from `httpRequest`.
    *   Extracts session ID (e.g., from `connect.sid`).
    *   Validates the session using existing application logic (e.g., querying the session store or using a session utility).
    *   Returns `userId` if valid, `null` otherwise.

**Step 3: Integrate WebSocket Server into Main App (`src/server.js` or `app.js`)**
*   Import `setupWebSocketServer` from `src/websocket/server.js`.
*   Create an `http.Server` instance from the Express `app`.
*   Pass the `http.Server` to `setupWebSocketServer`.
*   Store the returned WebSocket utility object (with broadcast functions) on the Express `app` instance (e.g., `app.set('wsServer', wsUtils)`) so it's accessible in route handlers and services.
*   Start the `http.Server` (instead of `app.listen`).

**Step 4: Modify Services to Emit WebSocket Events**
*   Example: `src/services/documentProcessor.js` (for `updateDocumentProgress`):
    ```javascript
    // ...
    const wsServer = req.app.get('wsServer'); // Assuming req is available or app passed directly
    if (wsServer && userId) {
      wsServer.broadcastToUser(userId, {
        type: 'document_status_update',
        payload: {
          documentId: documentId,
          status: progress.status,
          progress: progress.progress,
          message: progress.message
        }
      });
    }
    // ...
    ```
*   Apply similar logic to `ragService.js` for query progress, `sshService.js` for MCP installation output.

### 5.2. Frontend Implementation

**Step 1: Create WebSocket Context (`client/src/contexts/WebSocketContext.tsx`)**
*   (Largely as per `documentation/websocket-integration.md`)
*   Establish WebSocket connection (`ws://` or `wss://`).
*   Manage connection state (`connected`).
*   Handle `onopen`, `onmessage`, `onclose`, `onerror` events.
*   In `onmessage`, parse JSON data and update `lastMessage` state. Consider a more robust event system for different message types.
*   Provide `send` function (for client-to-server messages, if needed).
*   Ensure connection is only attempted/maintained if the user is authenticated (via `AuthContext`).

**Step 2: Integrate WebSocketProvider (`client/src/App.tsx`)**
*   Wrap the main application structure (likely inside `AuthProvider` and `ThemeProvider`) with `<WebSocketProvider>`.

**Step 3: Consume WebSocket Data in Components**
*   Use the `useWebSocket()` hook in components that need real-time updates.
*   Example: A document status component:
    ```typescript
    // client/src/components/DocumentStatusDisplay.tsx
    import React, { useEffect, useState } from 'react';
    import { useWebSocket } from '../contexts/WebSocketContext';

    const DocumentStatusDisplay = ({ documentId }) => {
      const { lastMessage } = useWebSocket();
      const [statusInfo, setStatusInfo] = useState(null);

      useEffect(() => {
        if (lastMessage && lastMessage.type === 'document_status_update' && lastMessage.payload.documentId === documentId) {
          setStatusInfo(lastMessage.payload);
        }
      }, [lastMessage, documentId]);

      if (!statusInfo) return <p>Loading status...</p>;

      return (
        <div>
          <p>Status: {statusInfo.status}</p>
          <p>Progress: {statusInfo.progress}%</p>
          <p>{statusInfo.message}</p>
        </div>
      );
    };

    export default DocumentStatusDisplay;
    ```

## 6. Phased Rollout Plan

Integrate WebSockets feature by feature to manage complexity and risk.

**Phase 1: Core Infrastructure (Backend & Frontend)**
*   Implement `src/websocket/server.js` and `src/websocket/auth.js`.
*   Integrate into `src/server.js`.
*   Implement `client/src/contexts/WebSocketContext.tsx`.
*   Integrate `WebSocketProvider` in `client/src/App.tsx`.
*   **Goal**: Establish a stable, authenticated WebSocket connection. Basic connect/disconnect logging.

**Phase 2: Document Processing Status Updates**
*   Modify `src/services/documentProcessor.js` to emit `document_status_update` events.
*   Create or update frontend components to consume these events and display live document status.
*   **Goal**: Replace polling for document status with WebSocket updates.

**Phase 3: MCP Installation Real-time Output**
*   Modify `src/services/sshService.js` (specifically `installMCPViaSSH`) to stream installation output (stdout/stderr) using `mcp_install_output` events and progress via `mcp_install` events.
    *   The `installCommand` in `installMCPViaSSH` will be sourced from `conf/config.ini` (`mcp_terminal_command_1`).
*   Develop frontend UI to display this live terminal-like output during MCP installation.
*   **Goal**: Provide real-time visibility into the MCP installation process.

**Phase 4: RAG Query Progress**
*   Modify `src/services/ragService.js` to emit `rag_progress` events during different stages of RAG processing.
*   Update chat UI or relevant components to display this progress.
*   **Goal**: Give users feedback on long-running RAG queries.

**Phase 5: Chat Message Streaming (Optional Enhancement)**
*   Evaluate if current HTTP streaming for chat can be migrated to use WebSockets for consistency. This might involve a unified `message_chunk` event.
*   **Goal**: Potentially unify all real-time chat communication over WebSockets.

**Phase 6: Additional Features (As Prioritized)**
*   Real-time notifications.
*   User presence indicators.
*   Live dashboard updates.

## 7. Testing Strategy

*   **Unit Tests**:
    *   Test WebSocket authentication logic (`src/websocket/auth.js`).
    *   Test message broadcasting logic in `src/websocket/server.js`.
    *   Test client-side context logic for connection management and message handling.
*   **Integration Tests**:
    *   Test the full flow: service triggers event -> WebSocket server broadcasts -> client context receives -> component updates.
    *   Test authentication: ensure unauthenticated clients cannot connect or receive restricted data.
    *   Test multiple client connections for the same user and for different users.
*   **End-to-End Tests**:
    *   Use UI automation tools (e.g., Cypress, Playwright) to simulate user actions and verify that real-time updates appear correctly in the UI.
    *   Test scenarios like connection loss and reconnection.
*   **Load Testing (Optional, for high-traffic features)**:
    *   Simulate many concurrent WebSocket connections to assess server performance and resource usage.
*   **Manual Testing**: Thoroughly test each feature across different browsers and network conditions.

## 8. Documentation Updates

*   Update this `websocketplan.md` as the implementation progresses.
*   Document the WebSocket message types and payload structures.
*   Update `DatabaseStructure.md` if any database changes are needed (unlikely for the core WebSocket setup, but possible for related features like notification storage).
*   Add troubleshooting steps for WebSocket connection issues to `FAQ.md` or a new `websocket-troubleshooting.md`.
*   Ensure `progress-tracker.md` reflects WebSocket integration tasks.

## 9. Future Enhancements

*   **Rooms/Channels**: For features requiring broadcasting to specific groups of users (e.g., collaborative sessions on a particular document).
*   **Advanced Reconnection Logic**: More sophisticated client-side reconnection strategies with exponential backoff.
*   **Message Queues**: For very high-throughput systems, integrate a message queue (e.g., Redis Pub/Sub) on the backend to decouple services from the WebSocket server and improve scalability across multiple server instances.
*   **Client-to-Server RPC**: If clients need to invoke server-side procedures frequently over WebSockets, establish a simple RPC-like mechanism.

## 10. Conclusion

Integrating WebSockets will be a significant enhancement to the Product Demo application. It will lead to a more modern, responsive, and efficient user experience. By following a phased approach and paying close attention to security and modularity, this integration can be achieved successfully, paving the way for even richer real-time features in the future. The primary benefits will be a vastly improved UX for monitoring long-running tasks and receiving instant updates, along with a reduction in server load from polling. 