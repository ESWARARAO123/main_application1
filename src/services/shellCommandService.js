const { spawn } = require('child_process');
const path = require('path');
const { logger } = require('../utils/logger');
const config = require('../utils/config');
const mcpDBService = require('./mcpDBService');

/**
 * Service for executing shell commands via Python MCP orchestrator
 * Integrates with the existing MCP server management system
 */
class ShellCommandService {
  constructor() {
    // Get Python interpreter path from config and normalize for current OS
    const configPythonPath = config.get('python.interpreter') || 'python';
    
    // Handle Windows path separators in config
    this.pythonInterpreter = configPythonPath.replace(/\\/g, path.sep);
    
    // Path to the Python orchestrator (relative to project root)
    this.orchestratorPath = path.join(__dirname, '../../python/terminal-mcp-orchestrator/orchestrator.py');
    
    logger.info(`Shell Command Service initialized with Python: ${this.pythonInterpreter}`);
    logger.info(`Orchestrator path: ${this.orchestratorPath}`);
  }

  /**
   * Execute a shell command via the Python MCP orchestrator
   * @param {string} command - The shell command to execute
   * @param {string} userId - User ID for server configuration lookup
   * @param {Object} options - Execution options
   * @param {string} options.serverId - Specific MCP server ID (optional, uses default if not provided)
   * @param {number} options.timeout - Execution timeout in seconds (default: 30)
   * @returns {Promise<Object>} - Execution result
   */
  async executeShellCommand(command, userId, options = {}) {
    const { serverId, timeout = 30 } = options;

    // Get MCP server configuration for the user
    let serverConfig;
    if (serverId) {
      serverConfig = await mcpDBService.getMCPServerConfiguration(serverId, userId);
      if (!serverConfig) {
        throw new Error(`MCP server configuration not found for server ID: ${serverId}`);
      }
    } else {
      // Use default server for the user
      serverConfig = await mcpDBService.getDefaultMCPServerConfiguration(userId);
      if (!serverConfig) {
        throw new Error('No default MCP server configured for user. Please configure an MCP server first.');
      }
    }

    return new Promise((resolve, reject) => {
      const serverUrl = `http://${serverConfig.mcp_host}:${serverConfig.mcp_port}`;
      const parameters = JSON.stringify({ command });
      
      logger.info(`Executing shell command via MCP orchestrator for user ${userId}`);
      logger.info(`Command: ${command}`);
      logger.info(`MCP Server: ${serverUrl} (${serverConfig.server_name})`);

      // Spawn Python orchestrator process using configured Python interpreter
      const pythonProcess = spawn(this.pythonInterpreter, [
        this.orchestratorPath,
        '--server', serverUrl,
        'runShellCommand',
        parameters
      ], {
        stdio: ['pipe', 'pipe', 'pipe'],
        timeout: timeout * 1000
      });

      let stdout = '';
      let stderr = '';

      // Collect stdout data
      pythonProcess.stdout.on('data', (data) => {
        stdout += data.toString();
      });

      // Collect stderr data
      pythonProcess.stderr.on('data', (data) => {
        stderr += data.toString();
      });

      // Handle process completion
      pythonProcess.on('close', (code) => {
        logger.info(`Python orchestrator process exited with code: ${code}`);
        
        if (code === 0) {
          // For successful execution, always preserve the complete raw output
          // The user wants to see the full orchestrator logs + JSON result
          resolve({
            success: true,
            command,
            result: null, // Don't try to parse - just show raw output
            output: stdout, // This contains the complete orchestrator output
            serverConfig: {
              id: serverConfig.id,
              name: serverConfig.server_name,
              host: serverConfig.mcp_host,
              port: serverConfig.mcp_port
            },
            timestamp: new Date().toISOString()
          });
        } else {
          resolve({
            success: false,
            command,
            error: `Process exited with code ${code}`,
            output: stdout,
            stderr,
            serverConfig: {
              id: serverConfig.id,
              name: serverConfig.server_name,
              host: serverConfig.mcp_host,
              port: serverConfig.mcp_port
            },
            timestamp: new Date().toISOString()
          });
        }
      });

      // Handle process errors
      pythonProcess.on('error', (error) => {
        logger.error('Error spawning Python orchestrator:', error);
        reject({
          success: false,
          command,
          error: error.message,
          serverConfig: {
            id: serverConfig.id,
            name: serverConfig.server_name,
            host: serverConfig.mcp_host,
            port: serverConfig.mcp_port
          },
          timestamp: new Date().toISOString()
        });
      });

      // Handle timeout
      setTimeout(() => {
        if (!pythonProcess.killed) {
          pythonProcess.kill();
          reject({
            success: false,
            command,
            error: `Command execution timed out after ${timeout} seconds`,
            serverConfig: {
              id: serverConfig.id,
              name: serverConfig.server_name,
              host: serverConfig.mcp_host,
              port: serverConfig.mcp_port
            },
            timestamp: new Date().toISOString()
          });
        }
      }, timeout * 1000);
    });
  }

  /**
   * Get available MCP tools from a user's server
   * @param {string} userId - User ID for server configuration lookup
   * @param {Object} options - Options
   * @param {string} options.serverId - Specific MCP server ID (optional, uses default if not provided)
   * @returns {Promise<Object>} - Available tools
   */
  async getAvailableTools(userId, options = {}) {
    const { serverId } = options;

    // Get MCP server configuration for the user
    let serverConfig;
    if (serverId) {
      serverConfig = await mcpDBService.getMCPServerConfiguration(serverId, userId);
      if (!serverConfig) {
        throw new Error(`MCP server configuration not found for server ID: ${serverId}`);
      }
    } else {
      // Use default server for the user
      serverConfig = await mcpDBService.getDefaultMCPServerConfiguration(userId);
      if (!serverConfig) {
        throw new Error('No default MCP server configured for user. Please configure an MCP server first.');
      }
    }

    return new Promise((resolve, reject) => {
      const serverUrl = `http://${serverConfig.mcp_host}:${serverConfig.mcp_port}`;
      
      logger.info(`Getting available tools from MCP server for user ${userId}`);
      logger.info(`MCP Server: ${serverUrl} (${serverConfig.server_name})`);

      // Spawn Python orchestrator process with --list flag
      const pythonProcess = spawn(this.pythonInterpreter, [
        this.orchestratorPath,
        '--server', serverUrl,
        '--list'
      ], {
        stdio: ['pipe', 'pipe', 'pipe']
      });

      let stdout = '';
      let stderr = '';

      pythonProcess.stdout.on('data', (data) => {
        stdout += data.toString();
      });

      pythonProcess.stderr.on('data', (data) => {
        stderr += data.toString();
      });

      pythonProcess.on('close', (code) => {
        if (code === 0) {
          resolve({
            success: true,
            tools: stdout,
            serverConfig: {
              id: serverConfig.id,
              name: serverConfig.server_name,
              host: serverConfig.mcp_host,
              port: serverConfig.mcp_port
            },
            timestamp: new Date().toISOString()
          });
        } else {
          reject({
            success: false,
            error: `Failed to get tools. Process exited with code ${code}`,
            output: stdout,
            stderr,
            serverConfig: {
              id: serverConfig.id,
              name: serverConfig.server_name,
              host: serverConfig.mcp_host,
              port: serverConfig.mcp_port
            },
            timestamp: new Date().toISOString()
          });
        }
      });

      pythonProcess.on('error', (error) => {
        logger.error('Error getting tools from MCP server:', error);
        reject({
          success: false,
          error: error.message,
          serverConfig: {
            id: serverConfig.id,
            name: serverConfig.server_name,
            host: serverConfig.mcp_host,
            port: serverConfig.mcp_port
          },
          timestamp: new Date().toISOString()
        });
      });
    });
  }

  /**
   * Test connection to a user's MCP server
   * @param {string} userId - User ID for server configuration lookup
   * @param {Object} options - Options
   * @param {string} options.serverId - Specific MCP server ID (optional, uses default if not provided)
   * @returns {Promise<Object>} - Connection test result
   */
  async testConnection(userId, options = {}) {
    try {
      const result = await this.getAvailableTools(userId, options);
      return {
        success: true,
        message: 'Successfully connected to MCP server',
        ...result
      };
    } catch (error) {
      return {
        success: false,
        message: 'Failed to connect to MCP server',
        error: error.error || error.message || error,
        serverConfig: error.serverConfig
      };
    }
  }

  /**
   * Get user's MCP server configurations
   * @param {string} userId - User ID
   * @returns {Promise<Array>} - User's MCP server configurations
   */
  async getUserMCPServers(userId) {
    try {
      return await mcpDBService.getUserMCPServerConfigurations(userId);
    } catch (error) {
      logger.error(`Error getting user MCP servers for user ${userId}:`, error);
      throw error;
    }
  }

  /**
   * Get user's default MCP server configuration
   * @param {string} userId - User ID
   * @returns {Promise<Object|null>} - Default MCP server configuration or null
   */
  async getUserDefaultMCPServer(userId) {
    try {
      return await mcpDBService.getDefaultMCPServerConfiguration(userId);
    } catch (error) {
      logger.error(`Error getting default MCP server for user ${userId}:`, error);
      throw error;
    }
  }
}

module.exports = ShellCommandService; 