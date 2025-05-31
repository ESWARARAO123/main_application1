/**
 * MCP API Routes
 * Handles API endpoints for MCP functionality
 */

const express = require('express');
const router = express.Router();
const { requireAuth, requireAdmin } = require('./auth');
const mcpService = require('../services/mcpService');
const sshService = require('../services/sshService');
const mcpDBService = require('../services/mcpDBService');
const config = require('../utils/config');
const { logger } = require('../utils/logger');
const db = require('../database');
const axios = require('axios');
const http = require('http');
const { v4: uuidv4 } = require('uuid');
const EventSource = require('eventsource');
const OllamaService = require('../services/ollamaService');
const ollamaService = new OllamaService();

// Initialize the service when the module is loaded
(async () => {
  try {
    await ollamaService.initialize();
    logger.info('OllamaService initialized for MCP routes');
  } catch (error) {
    logger.error('Failed to initialize OllamaService for MCP routes:', error);
  }
})();

// Get MCP configuration from config file
router.get('/config', requireAuth, (req, res) => {
  try {
    // Get MCP configuration from config
    const mcpConfig = {
      defaultTool: {
        name: config.get('mcp-server.mcp_terminal_name_1') || 'mcp-terminal_executor',
        installCommand: config.get('mcp-server.mcp_terminal_command_1') || '',
        defaultPort: parseInt(config.get('mcp-server.mcp_terminal_command_default_port_1')) || 8080,
        defaultHost: config.get('mcp-server.mcp_terminal_command_default_host_1') || 'localhost',
        timeout: parseInt(config.get('mcp-server.mcp_terminal_command_default_timeout_1')) || 300,
        timeoutUnit: config.get('mcp-server.mcp_terminal_command_default_timeout_unit_1') || 'seconds'
      }
    };

    res.json(mcpConfig);
  } catch (err) {
    logger.error(`Error fetching MCP config: ${err.message}`);
    res.status(500).json({ error: 'Failed to fetch MCP configuration' });
  }
});

// Test MCP connection
router.post('/test-connection', requireAuth, async (req, res) => {
  try {
    const { mcp_host, mcp_port } = req.body;

    if (!mcp_host || !mcp_port) {
      return res.status(400).json({ error: 'Host and port are required' });
    }

    // Test the MCP connection
    const result = await mcpService.testMCPConnection({
      mcp_host,
      mcp_port
    });

    res.json(result);
  } catch (err) {
    logger.error(`Error testing MCP connection: ${err.message}`);
    res.status(500).json({ error: 'Failed to test MCP connection' });
  }
});

// SSH Configuration Routes

// Get all SSH configurations
router.get('/ssh/config', requireAuth, async (req, res) => {
  try {
    const configurations = await mcpDBService.getUserSSHConfigurations(req.session.userId);
    res.json({ configurations });
  } catch (err) {
    logger.error(`Error fetching SSH configs: ${err.message}`);
    res.status(500).json({ error: 'Failed to fetch SSH configurations' });
  }
});

// Get a specific SSH configuration
router.get('/ssh/config/:id', requireAuth, async (req, res) => {
  try {
    const config = await mcpDBService.getSSHConfiguration(req.params.id, req.session.userId);

    if (!config) {
      return res.status(404).json({ error: 'SSH configuration not found' });
    }

    res.json(config);
  } catch (err) {
    logger.error(`Error fetching SSH config ${req.params.id}: ${err.message}`);
    res.status(500).json({ error: 'Failed to fetch SSH configuration' });
  }
});

// Create a new SSH configuration
router.post('/ssh/config', requireAuth, async (req, res) => {
  try {
    const config = req.body;

    // Validate required fields
    if (!config.machine_nickname || !config.ssh_host || !config.ssh_port ||
        !config.ssh_user || !config.ssh_auth_method) {
      return res.status(400).json({ error: 'Missing required fields' });
    }

    // Validate auth method
    if (config.ssh_auth_method === 'password' && !config.ssh_password) {
      return res.status(400).json({ error: 'Password is required for password authentication' });
    }

    if (config.ssh_auth_method === 'key' && !config.ssh_key_path) {
      return res.status(400).json({ error: 'Key path is required for key authentication' });
    }

    const created = await mcpDBService.createSSHConfiguration(config, req.session.userId);
    res.status(201).json(created);
  } catch (err) {
    logger.error(`Error creating SSH config: ${err.message}`);
    res.status(500).json({ error: 'Failed to create SSH configuration' });
  }
});

// Update an existing SSH configuration
router.put('/ssh/config/:id', requireAuth, async (req, res) => {
  try {
    const config = {
      ...req.body,
      id: req.params.id
    };

    // Validate required fields
    if (!config.machine_nickname || !config.ssh_host || !config.ssh_port ||
        !config.ssh_user || !config.ssh_auth_method) {
      return res.status(400).json({ error: 'Missing required fields' });
    }

    // Validate auth method
    if (config.ssh_auth_method === 'key' && !config.ssh_key_path) {
      return res.status(400).json({ error: 'Key path is required for key authentication' });
    }

    const updated = await mcpDBService.updateSSHConfiguration(config, req.session.userId);
    res.json(updated);
  } catch (err) {
    logger.error(`Error updating SSH config ${req.params.id}: ${err.message}`);
    res.status(500).json({ error: 'Failed to update SSH configuration' });
  }
});

// Delete an SSH configuration
router.delete('/ssh/config/:id', requireAuth, async (req, res) => {
  try {
    const deleted = await mcpDBService.deleteSSHConfiguration(req.params.id, req.session.userId);

    if (!deleted) {
      return res.status(404).json({ error: 'SSH configuration not found' });
    }

    res.json({ success: true });
  } catch (err) {
    logger.error(`Error deleting SSH config ${req.params.id}: ${err.message}`);
    res.status(500).json({ error: 'Failed to delete SSH configuration' });
  }
});

// Test SSH connection
router.post('/ssh/test', requireAuth, async (req, res) => {
  try {
    const config = req.body;

    // Validate required fields
    if (!config.ssh_host || !config.ssh_user || !config.ssh_auth_method) {
      return res.status(400).json({ error: 'Missing required fields' });
    }

    // Set default port if not provided
    if (!config.ssh_port) {
      config.ssh_port = 22;
    }

    // Validate auth method
    if (config.ssh_auth_method === 'password' && !config.ssh_password) {
      return res.status(400).json({ error: 'Password is required for password authentication' });
    }

    if (config.ssh_auth_method === 'key' && !config.ssh_key_path) {
      return res.status(400).json({ error: 'Key path is required for key authentication' });
    }

    // Test the connection
    const result = await sshService.testSSHConnection(config);

    // Update status in DB if config has an ID
    if (config.id) {
      await mcpDBService.updateSSHConnectionStatus(
        config.id,
        result.success ? 'successful' : 'failed',
        result.error || null
      );
    }

    res.json(result);
  } catch (err) {
    logger.error(`Error testing SSH connection: ${err.message}`);
    res.status(500).json({ error: 'Failed to test SSH connection' });
  }
});

// Install MCP via SSH
router.post('/ssh/install', requireAuth, async (req, res) => {
  try {
    const { sshConfigId, mcpToolId } = req.body;

    if (!sshConfigId) {
      return res.status(400).json({ error: 'SSH configuration ID is required' });
    }

    // Get SSH configuration with password
    const sshConfig = await mcpDBService.getSSHConfigurationWithPassword(sshConfigId, req.session.userId);

    if (!sshConfig) {
      return res.status(404).json({ error: 'SSH configuration not found' });
    }

    // Get MCP installation command from config if mcpToolId is 'default'
    if (mcpToolId === 'default') {
      // Use the default MCP installation command from config
      logger.info(`Using default MCP installation tool from config`);
    } else {
      // In the future, we could support custom MCP tools from the database
      logger.info(`Using MCP tool ID: ${mcpToolId}`);
    }

    // Install MCP via SSH
    const result = await sshService.installMCPViaSSH(sshConfig);

    // Update SSH connection status
    await mcpDBService.updateSSHConnectionStatus(
      sshConfigId,
      result.success ? 'successful' : 'failed',
      result.error || null
    );

    // If successful, create an MCP server configuration
    if (result.success) {
      const serverConfig = {
        server_name: `MCP on ${sshConfig.machine_nickname}`,
        mcp_host: result.host,
        mcp_port: result.port,
        is_default: true
      };

      await mcpDBService.createMCPServerConfiguration(serverConfig, req.session.userId);
    }

    res.json(result);
  } catch (err) {
    logger.error(`Error installing MCP via SSH: ${err.message}`);
    res.status(500).json({ error: 'Failed to install MCP' });
  }
});

// MCP Server Configuration Routes

// Get all MCP server configurations
router.get('/server/config', requireAuth, async (req, res) => {
  try {
    const configurations = await mcpDBService.getUserMCPServerConfigurations(req.session.userId);
    res.json({ configurations });
  } catch (err) {
    logger.error(`Error fetching MCP server configs: ${err.message}`);
    res.status(500).json({ error: 'Failed to fetch MCP server configurations' });
  }
});

// Get a specific MCP server configuration
router.get('/server/config/:id', requireAuth, async (req, res) => {
  try {
    const config = await mcpDBService.getMCPServerConfiguration(req.params.id, req.session.userId);

    if (!config) {
      return res.status(404).json({ error: 'MCP server configuration not found' });
    }

    res.json(config);
  } catch (err) {
    logger.error(`Error fetching MCP server config ${req.params.id}: ${err.message}`);
    res.status(500).json({ error: 'Failed to fetch MCP server configuration' });
  }
});

// Create a new MCP server configuration
router.post('/server/config', requireAuth, async (req, res) => {
  try {
    const config = req.body;

    // Validate required fields
    if (!config.server_name || !config.mcp_host || !config.mcp_port) {
      return res.status(400).json({ error: 'Missing required fields' });
    }

    const created = await mcpDBService.createMCPServerConfiguration(config, req.session.userId);
    res.status(201).json(created);
  } catch (err) {
    logger.error(`Error creating MCP server config: ${err.message}`);
    res.status(500).json({ error: 'Failed to create MCP server configuration' });
  }
});

// Update an existing MCP server configuration
router.put('/server/config/:id', requireAuth, async (req, res) => {
  try {
    const config = {
      ...req.body,
      id: req.params.id
    };

    // Validate required fields
    if (!config.server_name || !config.mcp_host || !config.mcp_port) {
      return res.status(400).json({ error: 'Missing required fields' });
    }

    const updated = await mcpDBService.updateMCPServerConfiguration(config, req.session.userId);
    res.json(updated);
  } catch (err) {
    logger.error(`Error updating MCP server config ${req.params.id}: ${err.message}`);
    res.status(500).json({ error: 'Failed to update MCP server configuration' });
  }
});

// Update MCP server connection status
router.post('/server/config/:id/status', requireAuth, async (req, res) => {
  try {
    const { id } = req.params;
    const { status, errorMessage } = req.body;

    // Validate required fields
    if (!status) {
      return res.status(400).json({ error: 'Status is required' });
    }

    // Get server configuration to verify ownership
    const serverConfig = await mcpDBService.getMCPServerConfiguration(id, req.session.userId);

    if (!serverConfig) {
      return res.status(404).json({ error: 'MCP server configuration not found' });
    }

    // Update connection status
    const updated = await mcpDBService.updateMCPConnectionStatus(id, status, errorMessage);

    res.json({ success: true, updated });
  } catch (err) {
    logger.error(`Error updating MCP server status ${req.params.id}: ${err.message}`);
    res.status(500).json({ error: 'Failed to update MCP server status' });
  }
});

// Delete an MCP server configuration
router.delete('/server/config/:id', requireAuth, async (req, res) => {
  try {
    const deleted = await mcpDBService.deleteMCPServerConfiguration(req.params.id, req.session.userId);

    if (!deleted) {
      return res.status(404).json({ error: 'MCP server configuration not found' });
    }

    res.json({ success: true });
  } catch (err) {
    logger.error(`Error deleting MCP server config ${req.params.id}: ${err.message}`);
    res.status(500).json({ error: 'Failed to delete MCP server configuration' });
  }
});

// MCP Connection and Tool Execution Routes

// Connect to an MCP server
router.post('/connect', requireAuth, async (req, res) => {
  try {
    const { serverId } = req.body;

    if (!serverId) {
      return res.status(400).json({ error: 'Server ID is required' });
    }

    // Get server configuration
    const serverConfig = await mcpDBService.getMCPServerConfiguration(serverId, req.session.userId);

    if (!serverConfig) {
      return res.status(404).json({ error: 'MCP server configuration not found' });
    }

    // Connect to the MCP server
    const result = await mcpService.connectToMCP({
      id: serverConfig.id,
      mcp_host: serverConfig.mcp_host,
      mcp_port: serverConfig.mcp_port
    });

    // Update connection status
    await mcpDBService.updateMCPConnectionStatus(
      serverId,
      result.success ? 'connected' : 'error',
      result.error || null
    );

    res.json(result);
  } catch (err) {
    logger.error(`Error connecting to MCP server: ${err.message}`);
    res.status(500).json({ error: 'Failed to connect to MCP server' });
  }
});

// Disconnect from an MCP server
router.post('/disconnect', requireAuth, async (req, res) => {
  try {
    const { serverId } = req.body;

    if (!serverId) {
      return res.status(400).json({ error: 'Server ID is required' });
    }

    // Disconnect from the MCP server
    const success = await mcpService.disconnectFromMCP(serverId);

    // Update connection status
    await mcpDBService.updateMCPConnectionStatus(serverId, 'disconnected');

    res.json({ success });
  } catch (err) {
    logger.error(`Error disconnecting from MCP server: ${err.message}`);
    res.status(500).json({ error: 'Failed to disconnect from MCP server' });
  }
});

// Get MCP connection status
router.get('/status/:serverId', requireAuth, async (req, res) => {
  try {
    const { serverId } = req.params;

    if (!serverId) {
      return res.status(400).json({ error: 'Server ID is required' });
    }

    // Get server configuration
    const serverConfig = await mcpDBService.getMCPServerConfiguration(serverId, req.session.userId);

    if (!serverConfig) {
      return res.status(404).json({ error: 'MCP server configuration not found' });
    }

    // Get connection status
    const status = mcpService.getMCPStatus(serverId);

    res.json({ status, config: serverConfig });
  } catch (err) {
    logger.error(`Error getting MCP status: ${err.message}`);
    res.status(500).json({ error: 'Failed to get MCP status' });
  }
});

// Get available MCP tools
router.get('/tools/:serverId', requireAuth, async (req, res) => {
  try {
    const { serverId } = req.params;

    if (!serverId) {
      return res.status(400).json({ error: 'Server ID is required' });
    }

    // Get server configuration
    const serverConfig = await mcpDBService.getMCPServerConfiguration(serverId, req.session.userId);

    if (!serverConfig) {
      return res.status(404).json({ error: 'MCP server configuration not found' });
    }

    // Get connection status
    const status = mcpService.getMCPStatus(serverId);

    if (status !== 'connected') {
      return res.status(400).json({ error: `MCP server is not connected. Current status: ${status}` });
    }

    // Discover tools
    const tools = await mcpService.discoverTools(serverId);

    res.json({ tools });
  } catch (err) {
    logger.error(`Error discovering MCP tools: ${err.message}`);
    res.status(500).json({ error: 'Failed to discover MCP tools' });
  }
});

// Execute an MCP tool
router.post('/execute', requireAuth, async (req, res) => {
  try {
    const { serverId, toolName, parameters } = req.body;

    if (!serverId || !toolName) {
      return res.status(400).json({ error: 'Server ID and tool name are required' });
    }

    // Get server configuration
    const serverConfig = await mcpDBService.getMCPServerConfiguration(serverId, req.session.userId);

    if (!serverConfig) {
      return res.status(404).json({ error: 'MCP server configuration not found' });
    }

    // Get connection status
    const status = mcpService.getMCPStatus(serverId);

    if (status !== 'connected') {
      return res.status(400).json({ error: `MCP server is not connected. Current status: ${status}` });
    }

    // Execute the tool
    const result = await mcpService.executeTool(serverId, toolName, parameters || {});

    res.json(result);
  } catch (err) {
    logger.error(`Error executing MCP tool: ${err.message}`);
    res.status(500).json({ error: 'Failed to execute MCP tool' });
  }
});

// Get active MCP connections
router.get('/connections', requireAuth, (req, res) => {
  try {
    const connections = mcpService.getActiveConnections();
    res.json({ connections });
  } catch (err) {
    logger.error(`Error getting active MCP connections: ${err.message}`);
    res.status(500).json({ error: 'Failed to get active MCP connections' });
  }
});

// Get default MCP server for user
router.get('/default-server', requireAuth, async (req, res) => {
  try {
    const defaultServer = await mcpDBService.getDefaultMCPServerConfiguration(req.session.userId);

    if (!defaultServer) {
      return res.json({ defaultServer: null });
    }

    // Get connection status
    const status = mcpService.getMCPStatus(defaultServer.id);

    res.json({
      defaultServer: {
        ...defaultServer,
        connectionStatus: status
      }
    });
  } catch (err) {
    logger.error(`Error getting default MCP server: ${err.message}`);
    res.status(500).json({ error: 'Failed to get default MCP server' });
  }
});

// SSH Filesystem Operations

// Authenticate SSH connection
router.post('/ssh/authenticate', requireAuth, async (req, res) => {
  try {
    const { sshConfigId, password } = req.body;

    if (!sshConfigId || !password) {
      return res.status(400).json({ error: 'SSH configuration ID and password are required' });
    }

    // Get SSH configuration
    const sshConfig = await mcpDBService.getSSHConfiguration(sshConfigId, req.session.userId);

    if (!sshConfig) {
      return res.status(404).json({ error: 'SSH configuration not found' });
    }

    // Authenticate SSH connection
    const result = await sshService.authenticateSSH(sshConfig, password);

    res.json(result);
  } catch (err) {
    logger.error(`Error authenticating SSH connection: ${err.message}`);
    res.status(500).json({ error: 'Failed to authenticate SSH connection' });
  }
});

// List directory contents
router.get('/ssh/fs/list', requireAuth, async (req, res) => {
  try {
    const { sshConfigId, path } = req.query;
    const password = req.headers['x-ssh-password']; // Get password from header

    if (!sshConfigId || !path || !password) {
      return res.status(400).json({ error: 'SSH configuration ID, path, and password are required' });
    }

    // Get SSH configuration
    const sshConfig = await mcpDBService.getSSHConfiguration(sshConfigId, req.session.userId);

    if (!sshConfig) {
      return res.status(404).json({ error: 'SSH configuration not found' });
    }

    // List directory contents
    const result = await sshService.listDirectory(sshConfig, path, password);

    if (!result.success) {
      return res.status(400).json({ error: result.error });
    }

    res.json({ items: result.items });
  } catch (err) {
    logger.error(`Error listing directory: ${err.message}`);
    res.status(500).json({ error: 'Failed to list directory' });
  }
});

// Get file/directory information
router.get('/ssh/fs/info', requireAuth, async (req, res) => {
  try {
    const { sshConfigId, path } = req.query;
    const password = req.headers['x-ssh-password']; // Get password from header

    if (!sshConfigId || !path || !password) {
      return res.status(400).json({ error: 'SSH configuration ID, path, and password are required' });
    }

    // Get SSH configuration
    const sshConfig = await mcpDBService.getSSHConfiguration(sshConfigId, req.session.userId);

    if (!sshConfig) {
      return res.status(404).json({ error: 'SSH configuration not found' });
    }

    // Get file/directory information
    const result = await sshService.getFileInfo(sshConfig, path, password);

    if (!result.success) {
      return res.status(400).json({ error: result.error });
    }

    res.json({ info: result.info });
  } catch (err) {
    logger.error(`Error getting file info: ${err.message}`);
    res.status(500).json({ error: 'Failed to get file info' });
  }
});

// Create directory
router.post('/ssh/fs/mkdir', requireAuth, async (req, res) => {
  try {
    const { sshConfigId, path } = req.body;
    const password = req.headers['x-ssh-password']; // Get password from header

    if (!sshConfigId || !path || !password) {
      return res.status(400).json({ error: 'SSH configuration ID, path, and password are required' });
    }

    // Get SSH configuration
    const sshConfig = await mcpDBService.getSSHConfiguration(sshConfigId, req.session.userId);

    if (!sshConfig) {
      return res.status(404).json({ error: 'SSH configuration not found' });
    }

    // Create directory
    const result = await sshService.createDirectory(sshConfig, path, password);

    if (!result.success) {
      return res.status(400).json({ error: result.error });
    }

    res.json({ success: true, message: result.message });
  } catch (err) {
    logger.error(`Error creating directory: ${err.message}`);
    res.status(500).json({ error: 'Failed to create directory' });
  }
});

// Install MCP in specific directory
router.post('/ssh/install-in-directory', requireAuth, async (req, res) => {
  try {
    const { sshConfigId, installDir, mcpToolId, password } = req.body;

    if (!sshConfigId || !installDir || !password) {
      return res.status(400).json({ error: 'SSH configuration ID, installation directory, and password are required' });
    }

    // Get SSH configuration
    const sshConfig = await mcpDBService.getSSHConfiguration(sshConfigId, req.session.userId);

    if (!sshConfig) {
      return res.status(404).json({ error: 'SSH configuration not found' });
    }

    // Get MCP installation command from config if mcpToolId is 'default'
    let installCommand = '';
    if (mcpToolId === 'default') {
      // Use the curl command from the config.ini file
      installCommand = config.get('mcp-server.mcp_terminal_command_1');
      if (!installCommand) {
        return res.status(400).json({ error: 'MCP installation command not found in configuration' });
      }
      logger.info(`Using default MCP installation tool from config: ${installCommand}`);
    } else {
      // In the future, we could support custom MCP tools from the database
      installCommand = config.get('mcp-server.mcp_terminal_command_1');
      if (!installCommand) {
        return res.status(400).json({ error: 'MCP installation command not found in configuration' });
      }
      logger.info(`Using MCP tool ID: ${mcpToolId} with command: ${installCommand}`);
    }

    // Log the available commands from config
    const startCmd = config.get('mcp-server.mcp_terminal_command_1_install_cmd');
    const stopCmd = config.get('mcp-server.mcp_terminal_command_1_stop_cmd');
    const restartCmd = config.get('mcp-server.mcp_terminal_command_1_restart_cmd');
    const statusCmd = config.get('mcp-server.mcp_terminal_command_1_status_cmd');

    logger.info(`Available MCP commands from config:
      - Start: ${startCmd}
      - Stop: ${stopCmd}
      - Restart: ${restartCmd}
      - Status: ${statusCmd}`);


    // Install MCP in the specified directory
    const result = await sshService.installMCPInDirectory(sshConfig, installDir, installCommand, password);

    // Update SSH connection status
    await mcpDBService.updateSSHConnectionStatus(
      sshConfigId,
      result.success ? 'successful' : 'failed',
      result.error || null
    );

    // If successful, create an MCP server configuration
    if (result.success) {
      try {
        const serverConfig = {
          server_name: `MCP on ${sshConfig.machine_nickname} (${installDir})`,
          mcp_host: result.host,
          mcp_port: result.port,
          is_default: true
        };

        await mcpDBService.createMCPServerConfiguration(serverConfig, req.session.userId);
      } catch (dbErr) {
        // If database operation fails, log the error but don't fail the installation
        logger.error(`Failed to create MCP server configuration in database: ${dbErr.message}`);

        // Add a warning to the result
        result.warning = `MCP server was installed successfully, but there was an error saving the configuration to the database: ${dbErr.message}`;
      }
    }

    res.json(result);
  } catch (err) {
    logger.error(`Error installing MCP in directory: ${err.message}`);
    res.status(500).json({ error: 'Failed to install MCP' });
  }
});

// Get a client ID from an MCP server directly
// This is a fallback method when WebSocket/SSE connections fail
// GET /api/mcp/get-client-id
router.get('/get-client-id', requireAuth, async (req, res) => {
  const { host, port } = req.query;
  
  if (!host || !port) {
    return res.status(400).json({ error: 'Host and port are required' });
  }
  
  try {
    logger.info(`Manual client ID request for MCP server ${host}:${port}`);
    
    // Create an EventSource to get the client ID
    const EventSource = require('eventsource');
    const sseUrl = `http://${host}:${port}/sse`;
    
    // Create a promise to handle the connection
    const clientIdPromise = new Promise((resolve, reject) => {
      // Create timeout to prevent hanging
      const timeout = setTimeout(() => {
        reject(new Error('Timeout waiting for client ID'));
      }, 8000);
      
      try {
        const es = new EventSource(sseUrl);
        
        // Listen for the connected event
        es.addEventListener('connected', (event) => {
          try {
            const data = JSON.parse(event.data);
            if (data.clientId) {
              clearTimeout(timeout);
              resolve(data.clientId);
              es.close();
            }
          } catch (e) {
            logger.error(`Error parsing connected event: ${e.message}`);
          }
        });
        
        // Listen for the open event
        es.onopen = () => {
          logger.info(`SSE connection opened to ${sseUrl}`);
        };
        
        // Listen for errors
        es.onerror = (err) => {
          logger.error(`SSE connection error: ${err.message || 'Unknown error'}`);
          clearTimeout(timeout);
          reject(new Error('SSE connection error'));
          es.close();
        };
        
        // Listen for general messages that might contain clientId
        es.onmessage = (event) => {
          try {
            const data = JSON.parse(event.data);
            if (data.type === 'connected' && data.clientId) {
              clearTimeout(timeout);
              resolve(data.clientId);
              es.close();
            }
          } catch (e) {
            logger.error(`Error parsing message: ${e.message}`);
          }
        };
      } catch (e) {
        clearTimeout(timeout);
        reject(e);
      }
    });
    
    // Wait for the client ID or timeout
    const clientId = await clientIdPromise;
    
    // Return the client ID
    return res.json({ clientId });
  } catch (error) {
    logger.error(`Error getting client ID: ${error.message}`);
    return res.status(500).json({ error: `Failed to get client ID: ${error.message}` });
  }
});

/**
 * New route for MCP chat
 * Handles sending chat messages through the MCP client
 * POST /api/mcp/chat
 */
router.post('/chat', requireAuth, async (req, res) => {
  try {
    const { messages, modelId, mcpClientId, mcpServer, options } = req.body;

    if (!messages) {
      return res.status(400).json({ error: 'Messages are required' });
    }

    if (!mcpClientId) {
      return res.status(400).json({ error: 'MCP Client ID is required' });
    }

    if (!mcpServer || !mcpServer.host || !mcpServer.port) {
      return res.status(400).json({ error: 'MCP server details are required' });
    }

    logger.info(`Processing MCP chat request with client ID: ${mcpClientId}, model: ${modelId || 'default'}`);
    
    // Check if the request is for streaming
    const isStreaming = options && options.stream === true;
    
    // Format messages for Ollama - extract just the required fields
    const formattedMessages = messages.map(msg => ({
      role: msg.role,
      content: msg.content
    }));

    try {
      // Get the modelId from request or use default
      const selectedModel = modelId || ollamaService.settings?.default_model || process.env.DEFAULT_OLLAMA_MODEL || 'llama3';
      
      // Log the model being used
      logger.info(`Using model for MCP chat: ${selectedModel}`);
      
      // Extract the current user message for simplified calls
      const userMessage = formattedMessages.length > 0 ? 
        formattedMessages[formattedMessages.length - 1].content : 
        'Hello';
      
      // Use the ollamaService to generate a response
      if (isStreaming) {
        // Set proper headers for SSE
        res.setHeader('Content-Type', 'text/event-stream');
        res.setHeader('Cache-Control', 'no-cache');
        res.setHeader('Connection', 'keep-alive');
        
        // Log the streaming request
        logger.info(`Starting stream for model ${selectedModel} with ${formattedMessages.length} messages`);
        
        try {
          // Use streaming mode with proper error handling
          const { success, response, error } = await ollamaService.chat(
            selectedModel,
            formattedMessages,
            null, // No system prompt
            true, // Enable streaming
            (chunk) => {
              // Send each chunk as an SSE event
              res.write(`data: ${JSON.stringify(chunk)}\n\n`);
            }
          );
          
          // If chat fails, handle error
          if (!success) {
            logger.error(`Streaming error for MCP chat: ${error}`);
            // For streaming errors, we need to write an error event and end the stream
            const errorChunk = {
              id: `error-${Date.now()}`,
              created: Date.now(),
              model: selectedModel,
              choices: [{
                index: 0,
                delta: { content: `Error: ${error}` },
                message: { role: 'assistant', content: `Error: ${error}` },
                finish_reason: 'error'
              }]
            };
            
            res.write(`data: ${JSON.stringify(errorChunk)}\n\n`);
            res.write('data: [DONE]\n\n');
            res.end();
            return;
          }
          
          // Send done signal and end response
          res.write('data: [DONE]\n\n');
          res.end();
        } catch (streamError) {
          logger.error(`Streaming error for MCP chat: ${streamError.message}`);
          // For streaming errors, we need to write an error event and end the stream
          const errorChunk = {
            id: `error-${Date.now()}`,
            created: Date.now(),
            model: selectedModel,
            choices: [{
              index: 0,
              delta: { content: `Error: ${streamError.message}` },
              message: { role: 'assistant', content: `Error: ${streamError.message}` },
              finish_reason: 'error'
            }]
          };
          
          res.write(`data: ${JSON.stringify(errorChunk)}\n\n`);
          res.write('data: [DONE]\n\n');
          res.end();
        }
      } else {
        // Regular non-streaming request
        logger.info(`Starting regular chat request for model ${selectedModel} with message: ${userMessage.substring(0, 50)}...`);
        
        // Use generateResponse for simplified requests
        const result = await ollamaService.generateResponse(userMessage, selectedModel);
        
        logger.info(`Completed request for model ${selectedModel}, success: ${result.success}`);
        
        if (!result.success) {
          return res.status(500).json({ error: result.error || 'Failed to generate response' });
        }
        
        return res.json({ response: result.response });
      }
    } catch (aiError) {
      logger.error(`Error with Ollama service in MCP chat: ${aiError.message}`, aiError.stack);
      
      return res.status(500).json({ 
        error: 'AI service unavailable. Please check Ollama configuration.' 
      });
    }
  } catch (error) {
    logger.error(`Error processing MCP chat request: ${error.message}`, error.stack);
    return res.status(500).json({ error: 'Failed to process MCP chat request' });
  }
});

module.exports = router;