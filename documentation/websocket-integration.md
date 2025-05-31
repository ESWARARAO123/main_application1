# WebSocket Integration for Real-time Updates in Product Demo

## Overview

This document outlines a comprehensive plan for integrating WebSockets into the application to enable real-time updates across various features. WebSockets provide a persistent, bidirectional communication channel between the client and server, making them ideal for features requiring immediate updates without polling.

## Current State Analysis

The application currently uses several approaches for real-time or near-real-time updates:

1. **Polling**: Used in document processing status checks
2. **Server-Sent Events (SSE)**: Used in MCP integration for tool execution updates
3. **Streaming HTTP Responses**: Used for AI chat completions

However, there is no unified WebSocket implementation, which could provide more efficient real-time communication for multiple features.

## Potential WebSocket Implementation Areas

### 1. Document Processing Status Updates

**Current Implementation**: Polling via interval-based API calls
```javascript
const statusInterval = setInterval(async () => {
  const statusResponse = await chatbotService.getDocumentStatus(documentId);
  // Update UI based on status
}, 2000);
```

**WebSocket Improvement**: Push updates when document processing status changes
```javascript
// Server sends:
{
  type: 'document_status',
  documentId: 'doc-123',
  status: 'PROCESSING',
  progress: 75,
  message: 'Generating embeddings'
}
```

### 2. RAG Query Processing

**Current Implementation**: Synchronous request/response
```javascript
const ragResult = await ragService.queryWithRAG(message, modelId, sessionId);
```

**WebSocket Improvement**: Stream intermediate results during RAG processing
```javascript
// Server sends:
{
  type: 'rag_progress',
  queryId: 'query-123',
  stage: 'RETRIEVAL',
  message: 'Found 5 relevant chunks',
  progress: 50
}
```

### 3. Chat Message Updates

**Current Implementation**: Streaming HTTP responses for AI messages
```javascript
abortFunctionRef.current = await aiChatService.streamChatCompletion(
  { modelId, messages, options: { stream: true } },
  (chunk) => {
    // Update UI with chunk
  }
);
```

**WebSocket Improvement**: Unified streaming for all message types
```javascript
// Server sends:
{
  type: 'message_chunk',
  messageId: 'msg-123',
  sessionId: 'session-456',
  content: 'New content chunk...',
  isComplete: false
}
```

### 4. MCP Installation Process

**Current Implementation**: Not implemented yet, planned to use polling or SSE
```javascript
// Planned implementation in mcpprogress-tracker.md
// Create WebSocket or Server-Sent Events endpoint for real-time output:
// POST /api/mcp/ssh/install/stream
```

**WebSocket Improvement**: Real-time installation progress and terminal output
```javascript
// Server sends:
{
  type: 'mcp_install_progress',
  installId: 'install-123',
  stage: 'EXECUTING',
  output: 'Installing dependencies...',
  progress: 65
}
```

### 5. User Authentication and Session Management

**Current Implementation**: HTTP-only cookies with session expiration
```javascript
app.use(session({
  secret: config.security.secret_key,
  resave: false,
  saveUninitialized: false,
  cookie: { /* ... */ }
}));
```

**WebSocket Improvement**: Real-time session management and notifications
```javascript
// Server sends:
{
  type: 'session_update',
  status: 'expiring_soon',
  expiresIn: 300 // seconds
}
```

### 6. Collaborative Features

**Current Implementation**: Not implemented

**WebSocket Improvement**: Enable real-time collaboration features
```javascript
// Server sends:
{
  type: 'user_activity',
  userId: 'user-123',
  username: 'john_doe',
  action: 'viewing_document',
  documentId: 'doc-456'
}
```

## Implementation Plan

### Phase 1: WebSocket Server Setup

1. **Add WebSocket Server**
   ```javascript
   // src/websocket/server.js
   const WebSocket = require('ws');
   const http = require('http');

   function setupWebSocketServer(server) {
     const wss = new WebSocket.Server({ server });

     // Store client connections
     const clients = new Map();

     wss.on('connection', (ws, req) => {
       // Extract user ID from session
       const userId = extractUserIdFromSession(req);

       if (!userId) {
         ws.close();
         return;
       }

       // Store connection
       if (!clients.has(userId)) {
         clients.set(userId, new Set());
       }
       clients.get(userId).add(ws);

       // Handle messages
       ws.on('message', handleMessage(ws, userId));

       // Handle disconnection
       ws.on('close', () => {
         if (clients.has(userId)) {
           clients.get(userId).delete(ws);
           if (clients.get(userId).size === 0) {
             clients.delete(userId);
           }
         }
       });
     });

     return {
       broadcast: (userId, data) => {
         if (clients.has(userId)) {
           clients.get(userId).forEach(client => {
             if (client.readyState === WebSocket.OPEN) {
               client.send(JSON.stringify(data));
             }
           });
         }
       },
       broadcastAll: (data) => {
         wss.clients.forEach(client => {
           if (client.readyState === WebSocket.OPEN) {
             client.send(JSON.stringify(data));
           }
         });
       }
     };
   }
   ```

2. **Integrate with Express Server**
   ```javascript
   // src/server.js
   const http = require('http');
   const { setupWebSocketServer } = require('./websocket/server');

   // Create HTTP server
   const server = http.createServer(app);

   // Setup WebSocket server
   const wsServer = setupWebSocketServer(server);

   // Make WebSocket server available to routes
   app.set('wsServer', wsServer);

   // Start server
   server.listen(port, host, () => {
     console.log(`Server running on ${config.server.protocol}://${host}:${port}`);
   });
   ```

### Phase 2: Client-Side WebSocket Integration

1. **Create WebSocket Context**
   ```typescript
   // client/src/contexts/WebSocketContext.tsx
   import React, { createContext, useContext, useEffect, useState } from 'react';
   import { useAuth } from './AuthContext';

   interface WebSocketContextType {
     connected: boolean;
     send: (data: any) => void;
     lastMessage: any;
   }

   const WebSocketContext = createContext<WebSocketContextType | null>(null);

   export const WebSocketProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
     const [socket, setSocket] = useState<WebSocket | null>(null);
     const [connected, setConnected] = useState<boolean>(false);
     const [lastMessage, setLastMessage] = useState<any>(null);
     const { isAuthenticated } = useAuth();

     useEffect(() => {
       if (!isAuthenticated) {
         if (socket) {
           socket.close();
           setSocket(null);
           setConnected(false);
         }
         return;
       }

       const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
       const newSocket = new WebSocket(`${protocol}//${window.location.host}/ws`);

       newSocket.onopen = () => {
         setConnected(true);
       };

       newSocket.onmessage = (event) => {
         try {
           const data = JSON.parse(event.data);
           setLastMessage(data);
           // Handle specific message types here or let consumers handle them
         } catch (error) {
           console.error('WebSocket message error:', error);
         }
       };

       newSocket.onclose = () => {
         setConnected(false);
       };

       setSocket(newSocket);

       return () => {
         newSocket.close();
       };
     }, [isAuthenticated]);

     const send = (data: any) => {
       if (socket && connected) {
         socket.send(JSON.stringify(data));
       }
     };

     return (
       <WebSocketContext.Provider value={{ connected, send, lastMessage }}>
         {children}
       </WebSocketContext.Provider>
     );
   };

   export const useWebSocket = () => {
     const context = useContext(WebSocketContext);
     if (!context) {
       throw new Error('useWebSocket must be used within a WebSocketProvider');
     }
     return context;
   };
   ```

2. **Add Provider to App**
   ```typescript
   // client/src/App.tsx
   import { WebSocketProvider } from './contexts/WebSocketContext';

   function App() {
     return (
       <AuthProvider>
         <ThemeProvider>
           <WebSocketProvider>
             {/* Rest of the app */}
           </WebSocketProvider>
         </ThemeProvider>
       </AuthProvider>
     );
   }
   ```

### Phase 3: Feature-Specific Implementations

1. **Document Processing Updates**
   ```javascript
   // src/services/documentProcessor.js
   async updateDocumentProgress(documentId, progress) {
     // Update database
     await db.query('UPDATE documents SET status = $1, progress = $2 WHERE id = $3',
       [progress.status, progress.progress, documentId]);

     // Get user ID for the document
     const result = await db.query('SELECT user_id FROM documents WHERE id = $1', [documentId]);
     const userId = result.rows[0]?.user_id;

     if (userId) {
       // Send WebSocket update
       const wsServer = app.get('wsServer');
       wsServer.broadcast(userId, {
         type: 'document_status',
         documentId,
         status: progress.status,
         progress: progress.progress,
         message: progress.message
       });
     }
   }
   ```

2. **RAG Query Processing**
   ```javascript
   // src/services/ragService.js
   async queryWithRAG(query, model, userId, options = {}) {
     const queryId = uuidv4();
     const wsServer = app.get('wsServer');

     // Send initial status
     wsServer.broadcast(userId, {
       type: 'rag_progress',
       queryId,
       stage: 'STARTED',
       message: 'Processing query',
       progress: 0
     });

     // Generate embedding
     wsServer.broadcast(userId, {
       type: 'rag_progress',
       queryId,
       stage: 'EMBEDDING',
       message: 'Generating query embedding',
       progress: 20
     });

     const queryEmbedding = await this.getQueryEmbedding(query);

     // Retrieve chunks
     wsServer.broadcast(userId, {
       type: 'rag_progress',
       queryId,
       stage: 'RETRIEVAL',
       message: 'Retrieving relevant chunks',
       progress: 40
     });

     const relevantChunks = await this.retrieveRelevantChunks(queryEmbedding);

     // Generate response
     wsServer.broadcast(userId, {
       type: 'rag_progress',
       queryId,
       stage: 'GENERATION',
       message: 'Generating response',
       progress: 70
     });

     const response = await this.generateResponse(query, relevantChunks);

     // Complete
     wsServer.broadcast(userId, {
       type: 'rag_progress',
       queryId,
       stage: 'COMPLETE',
       message: 'Query processing complete',
       progress: 100,
       result: response
     });

     return response;
   }
   ```

See continuation in websocket-integration-part2.md

