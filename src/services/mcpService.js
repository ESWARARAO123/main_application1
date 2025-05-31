/**
 * MCP Service
 * Manages connections to Model Context Protocol servers
 */

const axios = require('axios');
const EventSource = require('eventsource');
const { v4: uuidv4 } = require('uuid');
const { logger } = require('../utils/logger');
const config = require('../utils/config');

// In-memory store for active MCP connections
const connections = new Map();

/**
 * Test a connection to an MCP server
 * @param {Object} serverConfig - MCP server configuration
 * @param {string} serverConfig.mcp_host - The MCP server host
 * @param {number} serverConfig.mcp_port - The MCP server port
 * @returns {Promise<Object>} Connection test result
 */
async function testMCPConnection(serverConfig) {
  const { mcp_host, mcp_port } = serverConfig;
  
  try {
    // Set a short timeout for the test
    const timeout = 5000; // 5 seconds
    
    // Check if the server host/port are valid
    if (!mcp_host || !mcp_port) {
      throw new Error('Invalid MCP server configuration');
    }
    
    // Try to connect to the info endpoint
    const baseUrl = `http://${mcp_host}:${mcp_port}`;
    
    // Use direct httpAgent instead of proxy config to avoid ERR_BAD_RESPONSE issues
    const response = await axios.get(`${baseUrl}/info`, {
      timeout,
      httpAgent: new (require('http').Agent)({ keepAlive: true }),
      httpsAgent: new (require('https').Agent)({ keepAlive: true }),
      proxy: false // Explicitly disable proxy
    });
    
    // Check if we got a valid response
    if (response.status === 200) {
      const serverInfo = response.data || {};
      
      // Try to get available tools
      let tools = [];
      
      try {
        const toolsResponse = await axios.get(`${baseUrl}/tools`, {
          timeout,
          httpAgent: new (require('http').Agent)({ keepAlive: true }),
          httpsAgent: new (require('https').Agent)({ keepAlive: true }),
          proxy: false // Explicitly disable proxy
        });
        
        if (toolsResponse.data && toolsResponse.data.tools) {
          tools = toolsResponse.data.tools;
        }
      } catch (toolsError) {
        logger.warn(`Failed to get MCP tools: ${toolsError.message}`);
      }
      
      return {
        success: true,
        message: 'MCP server connection successful',
        server: {
          host: mcp_host,
          port: mcp_port,
          name: serverInfo.name,
          version: serverInfo.version,
          platform: serverInfo.platform
        },
        toolCount: tools.length
      };
    } else {
      throw new Error(`MCP server returned status ${response.status}`);
    }
  } catch (err) {
    logger.error(`MCP connection test failed for ${mcp_host}:${mcp_port}: ${err.message}`);
    
    return {
      success: false,
      error: err.message || 'Failed to connect to MCP server',
      server: {
        host: mcp_host,
        port: mcp_port
      }
    };
  }
}

/**
 * Connect to an MCP server and establish SSE connection
 * @param {Object} serverConfig - MCP server configuration
 * @param {string} serverConfig.id - The server ID
 * @param {string} serverConfig.mcp_host - The MCP server host
 * @param {number} serverConfig.mcp_port - The MCP server port
 * @returns {Promise<Object>} Connection status
 */
async function connectToMCP(serverConfig) {
  const { id, mcp_host, mcp_port } = serverConfig;
  const timeout = config.get('mcp-server.mcp_terminal_command_default_timeout_1') || 300;
  
  try {
    // Close existing connection if any
    if (connections.has(id)) {
      await disconnectFromMCP(id);
    }
    
    // Create connection object
    const connection = {
      id,
      host: mcp_host,
      port: mcp_port,
      status: 'connecting',
      lastError: null,
      sse: null,
      tools: [],
      connectedAt: null,
      connectionAttempts: 0,
      timeoutId: null
    };
    
    // Set up timeout
    connection.timeoutId = setTimeout(() => {
      if (connection.status === 'connecting') {
        connection.status = 'timeout';
        connection.lastError = 'Connection timeout';
        logger.warn(`MCP connection timeout for server ${id}`);
        
        // Clean up SSE if it exists
        if (connection.sse) {
          connection.sse.close();
        }
      }
    }, timeout * 1000);
    
    // Store the connection
    connections.set(id, connection);
    
    // Test connection with a basic HTTP request
    const baseUrl = `http://${mcp_host}:${mcp_port}`;
    await axios.get(`${baseUrl}/health`, { timeout: 5000 });
    
    // Set up SSE connection
    const sseUrl = `${baseUrl}/sse`;
    const sse = new EventSource(sseUrl);
    
    // Set up SSE event handlers
    sse.onopen = () => {
      connection.status = 'connected';
      connection.connectedAt = new Date();
      clearTimeout(connection.timeoutId);
      logger.info(`Connected to MCP server at ${mcp_host}:${mcp_port}`);
    };
    
    sse.onerror = (err) => {
      connection.status = 'error';
      connection.lastError = err.message || 'SSE connection error';
      clearTimeout(connection.timeoutId);
      logger.error(`MCP SSE connection error: ${connection.lastError}`);
      
      // Try to reconnect
      connection.connectionAttempts += 1;
      if (connection.connectionAttempts < 3) {
        logger.info(`Attempting to reconnect to MCP server (attempt ${connection.connectionAttempts})...`);
        setTimeout(() => {
          sse.close();
          connectToMCP(serverConfig);
        }, 5000);
      }
    };
    
    sse.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        logger.debug(`MCP message received: ${JSON.stringify(data)}`);
        
        // Handle different message types
        if (data.type === 'connected') {
          connection.clientId = data.clientId;
        } else if (data.type === 'tool_result') {
          // Store tool results
        }
      } catch (err) {
        logger.error(`Error parsing MCP SSE message: ${err.message}`);
      }
    };
    
    // Store SSE connection
    connection.sse = sse;
    
    // Discover tools
    try {
      await discoverTools(id);
    } catch (err) {
      logger.warn(`Failed to discover MCP tools: ${err.message}`);
    }
    
    return {
      success: true,
      status: connection.status,
      serverInfo: {
        id,
        host: mcp_host,
        port: mcp_port
      }
    };
  } catch (err) {
    // Update connection status if we have one
    if (connections.has(id)) {
      const connection = connections.get(id);
      connection.status = 'error';
      connection.lastError = err.message || 'Connection failed';
      clearTimeout(connection.timeoutId);
    }
    
    logger.error(`Failed to connect to MCP server at ${mcp_host}:${mcp_port}: ${err.message}`);
    
    return {
      success: false,
      error: err.message || 'Failed to connect to MCP server',
      status: 'error'
    };
  }
}

/**
 * Discover available tools from an MCP server
 * @param {string} serverId - The MCP server ID
 * @returns {Promise<Array>} List of available tools
 */
async function discoverTools(serverId) {
  if (!connections.has(serverId)) {
    throw new Error('MCP server not connected');
  }
  
  const connection = connections.get(serverId);
  
  if (connection.status !== 'connected') {
    throw new Error(`MCP server is not connected. Current status: ${connection.status}`);
  }
  
  try {
    const baseUrl = `http://${connection.host}:${connection.port}`;
    const response = await axios.get(`${baseUrl}/tools`, { timeout: 5000 });
    
    if (response.data && response.data.tools) {
      connection.tools = response.data.tools;
      logger.info(`Discovered ${connection.tools.length} tools from MCP server ${serverId}`);
      return connection.tools;
    } else {
      throw new Error('Invalid response from MCP server');
    }
  } catch (err) {
    logger.error(`Failed to discover MCP tools: ${err.message}`);
    throw err;
  }
}

/**
 * Execute a tool on an MCP server
 * @param {string} serverId - The MCP server ID
 * @param {string} toolName - The name of the tool to execute
 * @param {Object} params - The parameters for the tool
 * @returns {Promise<Object>} Tool execution result
 */
async function executeTool(serverId, toolName, params) {
  if (!connections.has(serverId)) {
    throw new Error('MCP server not connected');
  }
  
  const connection = connections.get(serverId);
  
  if (connection.status !== 'connected') {
    throw new Error(`MCP server is not connected. Current status: ${connection.status}`);
  }
  
  try {
    const baseUrl = `http://${connection.host}:${connection.port}`;
    const messageId = `msg-${Date.now()}-${uuidv4().substring(0, 8)}`;
    
    // Prepare the message
    const message = {
      id: messageId,
      type: 'invoke_tool',
      content: {
        name: toolName,
        parameters: params
      },
      clientId: connection.clientId
    };
    
    // Send the tool invocation request
    const response = await axios.post(`${baseUrl}/messages`, message, { timeout: 10000 });
    
    if (response.data && response.data.type === 'tool_result') {
      logger.info(`Successfully executed MCP tool ${toolName}`);
      return response.data.content;
    } else if (response.data && response.data.type === 'error') {
      throw new Error(response.data.error?.message || 'Tool execution failed');
    } else {
      throw new Error('Invalid response from MCP server');
    }
  } catch (err) {
    logger.error(`Failed to execute MCP tool ${toolName}: ${err.message}`);
    throw err;
  }
}

/**
 * Get MCP connection status
 * @param {string} serverId - The MCP server ID
 * @returns {string} Connection status
 */
function getMCPStatus(serverId) {
  if (!connections.has(serverId)) {
    return 'not_connected';
  }
  
  return connections.get(serverId).status;
}

/**
 * Disconnect from an MCP server
 * @param {string} serverId - The MCP server ID
 * @returns {Promise<boolean>} Success status
 */
async function disconnectFromMCP(serverId) {
  if (!connections.has(serverId)) {
    return true; // Already disconnected
  }
  
  try {
    const connection = connections.get(serverId);
    
    // Close SSE connection if it exists
    if (connection.sse) {
      connection.sse.close();
    }
    
    // Clear timeout if it exists
    if (connection.timeoutId) {
      clearTimeout(connection.timeoutId);
    }
    
    // Remove connection from map
    connections.delete(serverId);
    
    logger.info(`Disconnected from MCP server ${serverId}`);
    return true;
  } catch (err) {
    logger.error(`Error disconnecting from MCP server ${serverId}: ${err.message}`);
    return false;
  }
}

/**
 * Get all active MCP connections
 * @returns {Array} List of active connections
 */
function getActiveConnections() {
  return Array.from(connections.values()).map(connection => ({
    id: connection.id,
    host: connection.host,
    port: connection.port,
    status: connection.status,
    connectedAt: connection.connectedAt,
    toolsCount: connection.tools.length
  }));
}

module.exports = {
  connectToMCP,
  discoverTools,
  executeTool,
  getMCPStatus,
  disconnectFromMCP,
  getActiveConnections,
  testMCPConnection
}; 