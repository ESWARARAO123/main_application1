/**
 * WebSocket Routes
 *
 * Provides endpoints for WebSocket functionality
 */
const express = require('express');
const router = express.Router();
const { requireAuth } = require('./auth');
const { logger } = require('../utils/logger');
const crypto = require('crypto');
const axios = require('axios');
const http = require('http');
const EventSource = require('eventsource');

// Store MCP clients with their WebSocket connection
const mcpConnections = new Map();

/**
 * Generate a unique ID for MCP connections
 */
function generateConnectionId() {
  return `mcp-${Date.now()}-${crypto.randomBytes(4).toString('hex')}`;
}

/**
 * Register WebSocket message handlers for MCP functionality
 * @param {Object} wsServer - WebSocket server instance
 */
function registerMCPHandlers(wsServer) {
  logger.info('Registering MCP WebSocket message handlers');

  // Handle MCP connection requests
  wsServer.registerMessageHandler('mcp_connect', async (ws, message, userId) => {
    try {
      const { host, port } = message;
      
      if (!host || !port) {
        return ws.send(JSON.stringify({
          type: 'mcp_connect_error',
          error: 'Host and port are required'
        }));
      }
      
      logger.info(`WebSocket: User ${userId} attempting to connect to MCP server ${host}:${port}`);
      
      // Check if we already have a connection for this host:port for this user
      const existingConn = Array.from(mcpConnections.values()).find(
        conn => conn.host === host && conn.port === port && conn.userId === userId
      );
      
      if (existingConn) {
        logger.info(`WebSocket: Using existing MCP connection for ${host}:${port}`);
        
        // Check if the connection is still active by making a health check request
        try {
          await axios.get(`http://${host}:${port}/info`, { timeout: 3000 });
          
          // Send the existing client ID to the client
          ws.send(JSON.stringify({
            type: 'mcp_connected',
            connectionId: existingConn.connectionId,
            clientId: existingConn.clientId,
            host,
            port
          }));
          
          return;
        } catch (healthCheckError) {
          logger.warn(`WebSocket: Health check failed for existing MCP connection to ${host}:${port}, creating new connection`);
          
          // Remove the existing connection as it's no longer valid
          mcpConnections.delete(existingConn.connectionId);
        }
      }
      
      // Generate a unique connection ID for this MCP connection
      const connectionId = generateConnectionId();
      
      // Set up a proxy connection to the MCP server
      connectToMCPServer(host, port, connectionId, ws, userId);
      
    } catch (error) {
      logger.error(`WebSocket: Error handling mcp_connect: ${error.message}`);
      ws.send(JSON.stringify({
        type: 'mcp_connect_error',
        error: error.message
      }));
    }
  });
  
  // Handle MCP disconnect requests
  wsServer.registerMessageHandler('mcp_disconnect', (ws, message, userId) => {
    try {
      const { connectionId } = message;
      
      if (!connectionId) {
        return ws.send(JSON.stringify({
          type: 'mcp_disconnect_error',
          error: 'Connection ID is required'
        }));
      }
      
      logger.info(`WebSocket: User ${userId} attempting to disconnect from MCP server ${connectionId}`);
      
      // Check if this connection exists and belongs to this user
      if (mcpConnections.has(connectionId)) {
        const conn = mcpConnections.get(connectionId);
        
        if (conn.userId !== userId) {
          return ws.send(JSON.stringify({
            type: 'mcp_disconnect_error',
            error: 'You do not have permission to disconnect this MCP connection'
          }));
        }
        
        // Close the EventSource connection
        if (conn.eventSource) {
          conn.eventSource.close();
        }
        
        // Remove the connection from the map
        mcpConnections.delete(connectionId);
        
        // Send success response
        ws.send(JSON.stringify({
          type: 'mcp_disconnected',
          connectionId
        }));
        
        logger.info(`WebSocket: User ${userId} disconnected from MCP server ${connectionId}`);
      } else {
        ws.send(JSON.stringify({
          type: 'mcp_disconnect_error',
          error: 'Connection not found'
        }));
      }
    } catch (error) {
      logger.error(`WebSocket: Error handling mcp_disconnect: ${error.message}`);
      ws.send(JSON.stringify({
        type: 'mcp_disconnect_error',
        error: error.message
      }));
    }
  });
  
  // Handle MCP execute tool requests
  wsServer.registerMessageHandler('mcp_execute_tool', async (ws, message, userId) => {
    try {
      const { connectionId, tool, parameters } = message;
      
      if (!connectionId || !tool) {
        return ws.send(JSON.stringify({
          type: 'mcp_execute_error',
          error: 'Connection ID and tool name are required'
        }));
      }
      
      // Check if this connection exists and belongs to this user
      if (!mcpConnections.has(connectionId)) {
        return ws.send(JSON.stringify({
          type: 'mcp_execute_error',
          error: 'Connection not found'
        }));
      }
      
      const conn = mcpConnections.get(connectionId);
      
      if (conn.userId !== userId) {
        return ws.send(JSON.stringify({
          type: 'mcp_execute_error',
          error: 'You do not have permission to use this MCP connection'
        }));
      }
      
      // Check if we have a valid client ID
      if (!conn.clientId) {
        return ws.send(JSON.stringify({
          type: 'mcp_execute_error',
          error: 'No valid client ID available'
        }));
      }
      
      logger.info(`WebSocket: User ${userId} executing MCP tool ${tool} on connection ${connectionId}`);
      
      // Execute the tool
      const executeId = `exec-${Date.now()}-${crypto.randomBytes(4).toString('hex')}`;
      
      // Prepare the command payload
      const payload = {
        clientId: conn.clientId,
        tool,
        parameters: parameters || {}
      };
      
      // Send execution start notification
      ws.send(JSON.stringify({
        type: 'mcp_execute_start',
        executeId,
        tool,
        parameters
      }));
      
      // Execute the command against the MCP server
      try {
        const response = await axios.post(
          `http://${conn.host}:${conn.port}/messages`,
          payload,
          { timeout: 10000 }
        );
        
        // Send the execution result
        ws.send(JSON.stringify({
          type: 'mcp_execute_result',
          executeId,
          result: response.data
        }));
        
        logger.info(`WebSocket: MCP tool ${tool} execution completed on connection ${connectionId}`);
      } catch (execError) {
        logger.error(`WebSocket: Error executing MCP tool ${tool}: ${execError.message}`);
        
        // Check if the error is due to an invalid client ID
        if (execError.response && execError.response.data && 
            execError.response.data.error === 'Missing clientId') {
          
          // Try to reconnect to get a new client ID
          ws.send(JSON.stringify({
            type: 'mcp_reconnecting',
            connectionId,
            message: 'Client ID is invalid or expired, attempting to reconnect'
          }));
          
          // Attempt to reconnect
          connectToMCPServer(conn.host, conn.port, connectionId, ws, userId);
          
          // Also send an error for the current execution
          ws.send(JSON.stringify({
            type: 'mcp_execute_error',
            executeId,
            error: 'Client ID expired, reconnecting to MCP server'
          }));
        } else {
          // Send back generic execution error
          ws.send(JSON.stringify({
            type: 'mcp_execute_error',
            executeId,
            error: execError.message,
            details: execError.response?.data || {}
          }));
        }
      }
    } catch (error) {
      logger.error(`WebSocket: Error handling mcp_execute_tool: ${error.message}`);
      ws.send(JSON.stringify({
        type: 'mcp_execute_error',
        error: error.message
      }));
    }
  });
  
  // Add cleanup handler for client connection close
  wsServer.onClientDisconnect = (userId, ws) => {
    // Check for MCP connections belonging to this user
    const userConnIds = Array.from(mcpConnections.entries())
      .filter(([_, conn]) => conn.userId === userId)
      .map(([id, _]) => id);
      
    // Close all connections for this user
    for (const connId of userConnIds) {
      const conn = mcpConnections.get(connId);
      
      if (conn.eventSource) {
        conn.eventSource.close();
      }
      
      mcpConnections.delete(connId);
      logger.info(`WebSocket: Cleaned up MCP connection ${connId} for user ${userId} on WebSocket close`);
    }
  };
}

/**
 * Connect to an MCP server and establish a proxy connection
 * This function focuses on obtaining the client ID as quickly as possible
 */
async function connectToMCPServer(host, port, connectionId, ws, userId) {
  logger.info(`WebSocket: Connecting to MCP server ${host}:${port} with connection ID ${connectionId}`);
  
  // Create a connection record
  const connection = {
    connectionId,
    userId,
    host,
    port,
    clientId: null,
    eventSource: null,
    status: 'connecting',
    connectedAt: null,
    connectAttempts: 0
  };
  
  // Store the connection
  mcpConnections.set(connectionId, connection);
  
  // First check if the server is reachable at all with a quick ping
  try {
    // Send connecting status first
    ws.send(JSON.stringify({
      type: 'mcp_connecting',
      connectionId,
      host,
      port
    }));
    
    // Basic connectivity check with short timeout
    await axios.get(`http://${host}:${port}/info`, { timeout: 3000 });
    
    // If basic connectivity check succeeded, immediately set up SSE for client ID
    setupSSEConnection(host, port, connectionId, ws, connection);
  } catch (error) {
    logger.error(`WebSocket: Error connecting to MCP server ${host}:${port}: ${error.message || 'Unknown error'}`);
    
    // Update connection status
    connection.status = 'error';
    
    // Remove the connection from the map
    mcpConnections.delete(connectionId);
    
    // Notify the client
    ws.send(JSON.stringify({
      type: 'mcp_connect_error',
      connectionId,
      error: error.message || 'Failed to connect to MCP server',
      host,
      port
    }));
  }
}

/**
 * Set up the SSE connection to get a client ID
 * Separated from the main connection function for better error handling
 */
function setupSSEConnection(host, port, connectionId, ws, connection) {
  // Set up SSE connection
  const sseUrl = `http://${host}:${port}/sse`;
  logger.info(`WebSocket: Establishing SSE connection to ${sseUrl}`);
  
  // Create the EventSource for SSE connection
  try {
    const es = new EventSource(sseUrl, {
      headers: {
        'Accept': 'text/event-stream',
        'Cache-Control': 'no-cache'
      }
    });
    
    // Store the EventSource in the connection
    connection.eventSource = es;
    connection.connectAttempts += 1;
    
    // Set up a timeout for the connection
    const connectionTimeout = setTimeout(() => {
      if (connection.status === 'connecting') {
        logger.warn(`WebSocket: MCP connection timeout for ${host}:${port}`);
        
        // Clean up the connection
        es.close();
        
        // Set status to error
        connection.status = 'error';
        
        // Notify the client
        ws.send(JSON.stringify({
          type: 'mcp_connect_error',
          connectionId,
          error: 'Connection timeout - failed to get client ID',
          host,
          port
        }));
        
        // Try to reconnect if this is the first attempt
        if (connection.connectAttempts <= 2) {
          logger.info(`WebSocket: Retrying SSE connection to ${host}:${port} (attempt ${connection.connectAttempts + 1})`);
          // Wait briefly before retry
          setTimeout(() => {
            setupSSEConnection(host, port, connectionId, ws, connection);
          }, 1000);
        }
      }
    }, 8000); // 8 second timeout (reduced from 10 seconds)
    
    // Handle connection open
    es.onopen = () => {
      logger.info(`WebSocket: SSE connection opened to ${host}:${port}`);
      connection.status = 'connected';
      connection.connectedAt = new Date();
      
      // Note: We don't send a connected message yet, we wait for the clientId
    };
    
    // Handle SSE errors
    es.onerror = (err) => {
      logger.error(`WebSocket: SSE connection error for ${host}:${port}: ${err.message || 'Unknown error'}`);
      
      // If we already have a clientId, just log the error and return without retrying
      if (connection.clientId) {
        logger.warn(`WebSocket: SSE error occurred after clientId was obtained for ${host}:${port}. Keeping existing connection.`);
        return; // Do not attempt to reconnect; tolerate transient errors
      }
      
      connection.status = 'error';
      
      // Clear the timeout
      clearTimeout(connectionTimeout);
      
      // Notify the client
      ws.send(JSON.stringify({
        type: 'mcp_connection_error',
        connectionId,
        error: 'SSE connection error',
        host,
        port
      }));
      
      // Try to reconnect if this is not the last attempt and no clientId yet
      if (connection.connectAttempts <= 2 && !connection.clientId) {
        logger.info(`WebSocket: Retrying SSE connection to ${host}:${port} (attempt ${connection.connectAttempts + 1})`);
        // Wait briefly before retry
        setTimeout(() => {
          setupSSEConnection(host, port, connectionId, ws, connection);
        }, 2000);
      }
    };
    
    // Handle general SSE messages
    es.onmessage = (event) => {
      try {
        // Parse the message data
        const data = JSON.parse(event.data);
        logger.debug(`WebSocket: SSE message received from ${host}:${port}: ${JSON.stringify(data)}`);
        
        // Check for clientId in the message
        if (data.type === 'connected' && data.clientId) {
          logger.info(`WebSocket: Received clientId ${data.clientId} from MCP server ${host}:${port}`);
          
          // Store the clientId
          connection.clientId = data.clientId;
          
          // Clear the connection timeout
          clearTimeout(connectionTimeout);
          
          // Notify the client
          ws.send(JSON.stringify({
            type: 'mcp_connected',
            connectionId,
            clientId: data.clientId,
            host,
            port
          }));
        } else if (data.type === 'tool_result') {
          // Forward tool results to the client
          ws.send(JSON.stringify({
            type: 'mcp_tool_result',
            connectionId,
            result: data
          }));
        }
        
        // Forward all SSE events to the client as well
        ws.send(JSON.stringify({
          type: 'mcp_sse_event',
          connectionId,
          event: data
        }));
      } catch (error) {
        logger.error(`WebSocket: Error parsing SSE message: ${error.message}`);
      }
    };
    
    // Listen specifically for the 'connected' event
    es.addEventListener('connected', (event) => {
      try {
        const data = JSON.parse(event.data);
        logger.info(`WebSocket: Received 'connected' event from ${host}:${port}: ${JSON.stringify(data)}`);
        
        if (data.clientId) {
          logger.info(`WebSocket: Received clientId ${data.clientId} from 'connected' event`);
          
          // Store the clientId
          connection.clientId = data.clientId;
          
          // Clear the connection timeout
          clearTimeout(connectionTimeout);
          
          // Notify the client
          ws.send(JSON.stringify({
            type: 'mcp_connected',
            connectionId,
            clientId: data.clientId,
            host,
            port
          }));
        }
      } catch (error) {
        logger.error(`WebSocket: Error parsing 'connected' event: ${error.message}`);
      }
    });
    
    // Listen for tool result events
    es.addEventListener('tool_result', (event) => {
      try {
        const data = JSON.parse(event.data);
        
        // Forward tool results to the client
        ws.send(JSON.stringify({
          type: 'mcp_tool_result',
          connectionId,
          result: data
        }));
      } catch (error) {
        logger.error(`WebSocket: Error parsing 'tool_result' event: ${error.message}`);
      }
    });
  } catch (error) {
    logger.error(`WebSocket: Error creating EventSource for ${host}:${port}: ${error.message}`);
    
    // Update connection status
    connection.status = 'error';
    
    // Notify the client
    ws.send(JSON.stringify({
      type: 'mcp_connect_error',
      connectionId,
      error: `Error creating EventSource: ${error.message}`,
      host,
      port
    }));
  }
}

/**
 * Get WebSocket connection statistics
 * GET /api/websocket/stats
 */
router.get('/stats', requireAuth, (req, res) => {
  try {
    // Get the WebSocket server instance
    const wsServer = req.app.get('wsServer');

    if (!wsServer) {
      return res.status(500).json({ error: 'WebSocket server not available' });
    }

    // Get connection statistics
    const stats = wsServer.getStats();

    res.json(stats);
  } catch (error) {
    console.error('Error getting WebSocket stats:', error);
    res.status(500).json({ error: 'Failed to get WebSocket statistics' });
  }
});

/**
 * Get pending messages for a user
 * GET /api/websocket/pending-messages/:userId
 */
router.get('/pending-messages/:userId', requireAuth, (req, res) => {
  try {
    const { userId } = req.params;

    // Only allow admins or the user themselves to access their pending messages
    if (req.session.userId !== userId && !req.session.isAdmin) {
      return res.status(403).json({ error: 'Unauthorized' });
    }

    // Get the WebSocket server instance
    const wsServer = req.app.get('wsServer');

    if (!wsServer) {
      return res.status(500).json({ error: 'WebSocket server not available' });
    }

    // Get pending messages for the user
    const pendingMessages = wsServer.getPendingMessages(userId);

    res.json({
      userId,
      pendingMessageCount: pendingMessages.length,
      pendingMessages
    });
  } catch (error) {
    console.error('Error getting pending messages:', error);
    res.status(500).json({ error: 'Failed to get pending messages' });
  }
});

/**
 * Clear pending messages for a user
 * DELETE /api/websocket/pending-messages/:userId
 */
router.delete('/pending-messages/:userId', requireAuth, (req, res) => {
  try {
    const { userId } = req.params;

    // Only allow admins or the user themselves to clear their pending messages
    if (req.session.userId !== userId && !req.session.isAdmin) {
      return res.status(403).json({ error: 'Unauthorized' });
    }

    // Get the WebSocket server instance
    const wsServer = req.app.get('wsServer');

    if (!wsServer) {
      return res.status(500).json({ error: 'WebSocket server not available' });
    }

    // Clear pending messages for the user
    const clearedCount = wsServer.clearPendingMessages(userId);

    res.json({
      userId,
      clearedCount,
      message: `Cleared ${clearedCount} pending messages for user ${userId}`
    });
  } catch (error) {
    console.error('Error clearing pending messages:', error);
    res.status(500).json({ error: 'Failed to clear pending messages' });
  }
});

// Register routes for WebSocket initialization
router.get('/status', requireAuth, (req, res) => {
  res.json({
    websocket: {
      enabled: true,
      mcpConnections: mcpConnections.size
    }
  });
});

module.exports = {
  router,
  registerMCPHandlers
};
