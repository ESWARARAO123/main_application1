/**
 * WebSocket Server Module
 *
 * Sets up a WebSocket server and provides utilities for broadcasting messages to clients.
 */
const WebSocket = require('ws');
const { authenticateWebSocket } = require('./auth');

/**
 * Sets up a WebSocket server attached to an HTTP server
 *
 * @param {Object} server - HTTP server instance to attach the WebSocket server to
 * @returns {Object} - WebSocket utilities for broadcasting messages
 */
function setupWebSocketServer(server) {
  // Create WebSocket server attached to the HTTP server
  const wss = new WebSocket.Server({
    server,
    // Path for WebSocket connections
    path: '/ws'
  });

  console.log('WebSocket server initialized');

  // Store client connections by user ID
  // Map<userId, Set<WebSocket>>
  const clients = new Map();

  // Store pending messages for users who are not connected
  // Map<userId, Array<{data: Object, timestamp: number}>>
  const pendingMessages = new Map();

  // Store message handlers by message type
  // Map<messageType, Function>
  const messageHandlers = new Map();

  // Client disconnect callback
  let onClientDisconnectCallback = null;

  // Register a message handler
  wss.registerMessageHandler = (messageType, handler) => {
    messageHandlers.set(messageType, handler);
    console.log(`Registered message handler for message type: ${messageType}`);
    return () => {
      messageHandlers.delete(messageType);
      console.log(`Unregistered message handler for message type: ${messageType}`);
    };
  };

  // Set client disconnect handler
  Object.defineProperty(wss, 'onClientDisconnect', {
    set: function(callback) {
      onClientDisconnectCallback = callback;
      console.log('Set client disconnect callback handler');
    }
  });

  // Handle new WebSocket connections
  wss.on('connection', async (ws, req) => {
    console.log('New WebSocket connection attempt');
    console.log('Connection headers:', req.headers);

    // Log client IP and other connection details
    const ip = req.socket.remoteAddress;
    console.log(`Connection from IP: ${ip}`);

    // Check if cookie header is present
    if (!req.headers.cookie) {
      console.log('No cookie header found in WebSocket connection request');
      ws.close(1008, 'No cookie header found');
      return;
    }

    // Authenticate the connection
    console.log('Attempting to authenticate WebSocket connection');
    const userId = await authenticateWebSocket(req);
    console.log('Authentication result:', userId ? `User ID: ${userId}` : 'Authentication failed');

    // If authentication failed, close the connection
    if (!userId) {
      console.log('WebSocket authentication failed, closing connection');
      ws.close(1008, 'Authentication failed');
      return;
    }

    console.log(`WebSocket connection authenticated for user: ${userId}`);

    // Store the user ID on the WebSocket instance for reference
    ws.userId = userId;

    // Mark the connection as alive initially
    ws.isAlive = true;
    
    // Add a timestamp for connection time
    ws.connectionTime = Date.now();
    
    // Add a flag to track ping reliability
    ws.pingsMissed = 0;
    
    // Set a heartbeat timer - more frequent pings for better reliability
    const pingInterval = 20000; // 20 seconds (reduced from 30)

    // Set up ping interval for this connection
    const connectionPingInterval = setInterval(() => {
      // If the connection has missed too many pings, terminate it
      if (ws.pingsMissed >= 2) {
        console.log(`Terminating WebSocket connection for user ${userId} after missing ${ws.pingsMissed} pings`);
        clearInterval(connectionPingInterval);
        return ws.terminate();
      }

      // Increment missed pings counter - will be reset when pong is received
      ws.pingsMissed++;

      // Send a ping (the client will automatically respond with a pong)
      try {
        ws.ping();
      } catch (error) {
        console.error(`Error sending ping to user ${userId}:`, error);
        clearInterval(connectionPingInterval);
        ws.terminate();
      }
    }, pingInterval);

    // Add the connection to the clients map
    if (!clients.has(userId)) {
      clients.set(userId, new Set());
    }

    // More frequent pinging for MCP connections to detect issues earlier
    ws.addEventListener('message', (message) => {
      try {
        const data = JSON.parse(message.data);
        if (data.type === 'mcp_connect' || data.type === 'mcp_execute_tool') {
          // This is an MCP-related connection, increase ping frequency for reliability
          clearInterval(connectionPingInterval);
          const mcpPingInterval = 15000; // 15 seconds for MCP connections
          
          const newPingInterval = setInterval(() => {
            if (ws.pingsMissed >= 2) {
              console.log(`Terminating MCP WebSocket connection for user ${userId} after missing ${ws.pingsMissed} pings`);
              clearInterval(newPingInterval);
              return ws.terminate();
            }
            
            ws.pingsMissed++;
            
            try {
              ws.ping();
            } catch (error) {
              console.error(`Error sending MCP ping to user ${userId}:`, error);
              clearInterval(newPingInterval);
              ws.terminate();
            }
          }, mcpPingInterval);
        }
      } catch (e) {
        // Non-JSON message, not related to MCP
      }
    });

    // Handle pong messages (automatic response to ping)
    ws.on('pong', () => {
      // Reset the missed pings counter when we receive a pong
      ws.pingsMissed = 0;
      
      // Mark the connection as alive when we receive a pong
      ws.isAlive = true;
    });

    // Check if the user already has too many connections
    const maxConnectionsPerUser = 3; // Limit to 3 connections per user
    const userConnections = clients.get(userId);

    if (userConnections.size >= maxConnectionsPerUser) {
      console.log(`User ${userId} has reached the maximum number of connections (${maxConnectionsPerUser}). Closing oldest connections.`);

      // Convert the Set to an Array so we can sort and limit
      const connectionsArray = Array.from(userConnections);

      // Sort connections by connection time (oldest first)
      connectionsArray.sort((a, b) => (a.connectionTime || 0) - (b.connectionTime || 0));

      // We'll close the oldest connection to make room for this new one
      // Only close one connection at a time to prevent closing too many
      const oldestConnection = connectionsArray[0];

      console.log(`Closing old connection for user ${userId}`);
      userConnections.delete(oldestConnection);

      try {
        // Send a message to the client before closing
        oldestConnection.send(JSON.stringify({
          type: 'connection_replaced',
          message: 'Your connection was closed because a newer connection was established'
        }));

        // Close the connection
        oldestConnection.close(1000, 'Connection replaced by newer connection');
      } catch (error) {
        console.error(`Error closing old connection for user ${userId}:`, error);
        // Force terminate if close fails
        oldestConnection.terminate();
      }
    }

    // Add the new connection
    clients.get(userId).add(ws);

    console.log(`WebSocket client connected for user ${userId}. Total connections for this user: ${clients.get(userId).size}`);

    // Now that we have authenticated the user, send a confirmation message immediately
    try {
      ws.send(JSON.stringify({
        type: 'connection_established',
        message: 'WebSocket connection established and authenticated',
        userId: userId, 
        timestamp: Date.now()
      }));
      console.log(`Sent connection_established confirmation to user ${userId}`);
    } catch (error) {
      console.error(`Error sending connection confirmation to user ${userId}:`, error);
    }

    // Check if there are any pending messages for this user
    if (pendingMessages.has(userId) && pendingMessages.get(userId).length > 0) {
      const userPendingMessages = pendingMessages.get(userId);
      console.log(`Found ${userPendingMessages.length} pending messages for user ${userId}`);

      // Only deliver messages that are less than 10 minutes old (increased from 5 minutes)
      const now = Date.now();
      const validMessages = userPendingMessages.filter(msg => (now - msg.timestamp) < 600000);

      // Group messages by document ID to avoid sending multiple updates for the same document
      const documentGroups = new Map();

      // Group messages by document ID and keep only the latest for each status
      validMessages.forEach(msg => {
        if (msg.data.type === 'document_status_update' && msg.data.payload && msg.data.payload.documentId) {
          const docId = msg.data.payload.documentId;
          const status = msg.data.payload.status;

          // Create a key that combines document ID and status
          const key = `${docId}-${status}`;

          if (!documentGroups.has(key) || documentGroups.get(key).timestamp < msg.timestamp) {
            documentGroups.set(key, msg);
          }
        } else {
          // For non-document messages, treat each as unique
          const uniqueKey = `other-${msg.timestamp}`;
          documentGroups.set(uniqueKey, msg);
        }
      });

      // Convert the map back to an array of messages
      const dedupedMessages = Array.from(documentGroups.values());

      // Further filter to only include the latest message for each document
      const latestDocMessages = new Map();
      dedupedMessages.forEach(msg => {
        if (msg.data.type === 'document_status_update' && msg.data.payload && msg.data.payload.documentId) {
          const docId = msg.data.payload.documentId;

          if (!latestDocMessages.has(docId) || latestDocMessages.get(docId).timestamp < msg.timestamp) {
            latestDocMessages.set(docId, msg);
          }
        } else {
          // For non-document messages, keep all
          const uniqueKey = `other-${msg.timestamp}`;
          latestDocMessages.set(uniqueKey, msg);
        }
      });

      // Convert back to array and sort by timestamp
      const finalMessages = Array.from(latestDocMessages.values())
        .sort((a, b) => a.timestamp - b.timestamp);

      if (finalMessages.length > 0) {
        console.log(`Delivering ${finalMessages.length} deduplicated messages to user ${userId} (from original ${validMessages.length})`);

        // Deliver each message with a small delay to prevent overwhelming the client
        finalMessages.forEach((msg, index) => {
          setTimeout(() => {
            try {
              if (ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify(msg.data));
                console.log(`Delivered pending message ${index + 1}/${finalMessages.length} to user ${userId}`);
              }
            } catch (error) {
              console.error(`Error delivering pending message to user ${userId}:`, error);
            }
          }, index * 200); // Increased delay between messages to 200ms
        });
      }

      // Clear the pending messages for this user
      pendingMessages.delete(userId);
    }

    // Handle incoming messages
    ws.on('message', (message) => {
      try {
        // Parse the message as JSON
        const data = JSON.parse(message);

        // Only log non-ping messages to reduce noise
        if (data.type !== 'ping') {
          console.log(`Received message from user ${userId}:`, data);
        }

        // Handle different message types
        if (data.type === 'ping') {
          // Client-initiated ping (different from server ping)
          ws.send(JSON.stringify({
            type: 'pong',
            timestamp: Date.now()
          }));

          // Also mark the connection as alive
          ws.isAlive = true;
        } else if (data.type === 'manual_connection') {
          // Handle manual connection message
          console.log(`Received manual connection message from user ${userId}:`, data);

          // Send a confirmation message back
          ws.send(JSON.stringify({
            type: 'manual_connection_confirmed',
            userId: userId,
            timestamp: Date.now(),
            message: 'Manual connection confirmed'
          }));

          // Send any pending messages for this user
          if (pendingMessages.has(userId) && pendingMessages.get(userId).length > 0) {
            const userPendingMessages = pendingMessages.get(userId);
            console.log(`Found ${userPendingMessages.length} pending messages for user ${userId} on manual connection`);

            // Deliver each message with a small delay
            userPendingMessages.forEach((msg, index) => {
              setTimeout(() => {
                try {
                  if (ws.readyState === WebSocket.OPEN) {
                    ws.send(JSON.stringify(msg.data));
                    console.log(`Delivered pending message ${index + 1}/${userPendingMessages.length} to user ${userId} on manual connection`);
                  }
                } catch (error) {
                  console.error(`Error delivering pending message to user ${userId} on manual connection:`, error);
                }
              }, index * 200);
            });

            // Clear the pending messages for this user
            pendingMessages.delete(userId);
          }
        } else if (messageHandlers.has(data.type)) {
          // Call the registered handler for this message type
          const handler = messageHandlers.get(data.type);
          handler(ws, data, userId);
        }
        // Add more message handlers as needed

      } catch (error) {
        console.error('Error handling WebSocket message:', error);
      }
    });

    // Handle connection close
    ws.on('close', () => {
      console.log(`WebSocket connection closed for user ${userId}`);

      // Clean up the ping interval
      clearInterval(connectionPingInterval);

      // Remove the connection from the clients map
      if (clients.has(userId)) {
        clients.get(userId).delete(ws);

        // If this was the last connection for this user, remove the user entry
        if (clients.get(userId).size === 0) {
          clients.delete(userId);
          console.log(`Removed last connection for user ${userId}`);
        } else {
          console.log(`Remaining connections for user ${userId}: ${clients.get(userId).size}`);
        }
      }

      // Call the client disconnect callback if available
      if (onClientDisconnectCallback) {
        try {
          onClientDisconnectCallback(userId, ws);
        } catch (error) {
          console.error(`Error in client disconnect callback for user ${userId}:`, error);
        }
      }
    });

    // Handle connection errors
    ws.on('error', (error) => {
      console.error(`WebSocket error for user ${userId}:`, error);

      // Clean up the ping interval on error
      clearInterval(connectionPingInterval);
    });
  });

  // Return utility functions for broadcasting messages
  return {
    /**
     * Broadcasts a message to all WebSocket connections for a specific user
     *
     * @param {string} userId - The user ID to broadcast to
     * @param {Object} data - The data to broadcast (will be JSON stringified)
     * @returns {boolean} - Whether the message was sent to any connections
     */
    broadcastToUser: (userId, data) => {
      if (!clients.has(userId) || clients.get(userId).size === 0) {
        console.log(`No active connections for user ${userId}`);

        // Store the message in a temporary queue for this user
        // This will be sent when the user reconnects
        if (!pendingMessages.has(userId)) {
          pendingMessages.set(userId, []);
        }

        // Add the message to the pending queue with a timestamp
        pendingMessages.get(userId).push({
          data,
          timestamp: Date.now()
        });

        console.log(`Message queued for user ${userId} (${pendingMessages.get(userId).length} pending messages)`);
        return false;
      }

      const userClients = clients.get(userId);
      console.log(`Broadcasting to user ${userId} (${userClients.size} connections)`);

      let sentCount = 0;
      userClients.forEach((client) => {
        if (client.readyState === WebSocket.OPEN) {
          try {
            client.send(JSON.stringify(data));
            sentCount++;
          } catch (error) {
            console.error(`Error sending message to client for user ${userId}:`, error);
          }
        }
      });

      console.log(`Sent message to ${sentCount}/${userClients.size} connections for user ${userId}`);
      return sentCount > 0;
    },

    /**
     * Broadcasts a message to all connected WebSocket clients
     *
     * @param {Object} data - The data to broadcast (will be JSON stringified)
     */
    broadcastToAll: (data) => {
      console.log(`Broadcasting to all users (${wss.clients.size} total connections)`);

      wss.clients.forEach((client) => {
        if (client.readyState === WebSocket.OPEN) {
          client.send(JSON.stringify(data));
        }
      });
    },

    /**
     * Returns the number of connected clients
     *
     * @returns {Object} - Connection statistics
     */
    getStats: () => {
      return {
        totalConnections: wss.clients.size,
        uniqueUsers: clients.size,
        userConnections: Array.from(clients.entries()).map(([userId, connections]) => ({
          userId,
          connectionCount: connections.size
        }))
      };
    },

    /**
     * Gets pending messages for a specific user
     *
     * @param {string} userId - The user ID to get pending messages for
     * @returns {Array} - Array of pending messages
     */
    getPendingMessages: (userId) => {
      if (!pendingMessages.has(userId)) {
        return [];
      }

      return pendingMessages.get(userId);
    },

    /**
     * Clears pending messages for a specific user
     *
     * @param {string} userId - The user ID to clear pending messages for
     * @returns {number} - Number of messages cleared
     */
    clearPendingMessages: (userId) => {
      if (!pendingMessages.has(userId)) {
        return 0;
      }

      const count = pendingMessages.get(userId).length;
      pendingMessages.delete(userId);
      return count;
    },

    /**
     * Register a message handler for a specific message type
     *
     * @param {string} messageType - The message type to handle
     * @param {Function} handler - The handler function for this message type
     * @returns {Function} - A function to unregister the handler
     */
    registerMessageHandler: wss.registerMessageHandler,

    /**
     * Set a callback for client disconnection
     * 
     * @param {Function} callback - Function to call when a client disconnects
     */
    set onClientDisconnect(callback) {
      wss.onClientDisconnect = callback;
    },

    /**
     * The underlying WebSocket server instance
     */
    wss
  };
}

module.exports = {
  setupWebSocketServer
};
