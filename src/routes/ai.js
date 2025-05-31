const express = require('express');
const router = express.Router();
const { logger } = require('../utils/logger');
const OllamaService = require('../services/ollamaService');
const ShellCommandService = require('../services/shellCommandService');
const { requireAuth } = require('./auth');

// Initialize the Ollama service
const ollamaService = new OllamaService();

// Initialize the Shell Command service with error handling
let shellCommandService;
try {
  shellCommandService = new ShellCommandService();
  logger.info('Shell Command Service initialized successfully');
} catch (error) {
  logger.error('Failed to initialize Shell Command Service:', error);
  shellCommandService = null;
}

// Load settings at startup
let serviceInitialized = false;
const initService = async () => {
  if (!serviceInitialized) {
    await ollamaService.initialize();
    serviceInitialized = true;
  }
};
initService();

/**
 * Simple chat endpoint for AI requests
 * POST /api/ai/chat
 */
router.post('/chat', async (req, res) => {
  try {
    const { message, modelId } = req.body;

    if (!message) {
      return res.status(400).json({ error: 'Message is required' });
    }

    logger.info(`Processing chat request: ${message.substring(0, 100)}${message.length > 100 ? '...' : ''}`);

    // Make sure service is initialized to get settings
    if (!serviceInitialized) {
      await initService();
    }
    
    // Get default model from service settings
    let defaultModel = '';
    try {
      if (ollamaService.settings && ollamaService.settings.default_model) {
        defaultModel = ollamaService.settings.default_model;
      }
    } catch (err) {
      logger.warn('Could not get default model from settings:', err);
    }

    try {
      // Use provided modelId, or service default, or environment variable, or lastly a fallback
      const selectedModel = modelId || defaultModel || process.env.DEFAULT_OLLAMA_MODEL || 'llama3';
      const aiResponse = await ollamaService.generateResponse(message, selectedModel);

      return res.json({ response: aiResponse });
    } catch (aiError) {
      logger.error('Error with Ollama service:', aiError);
      
      // Fallback: Return error message
      return res.status(500).json({ 
        error: 'AI service unavailable. Please check Ollama configuration.'
      });
    }
  } catch (error) {
    logger.error('Error processing chat request:', error);
    return res.status(500).json({ error: 'Failed to process chat request' });
  }
});

/**
 * Execute shell command via MCP orchestrator
 * POST /api/ai/tools/runshellcommand
 */
router.post('/tools/runshellcommand', requireAuth, async (req, res) => {
  try {
    // Check if shell command service is available
    if (!shellCommandService) {
      logger.error('Shell Command Service not available');
      return res.status(503).json({ 
        success: false,
        error: 'Shell command service is not available. Please check server configuration.'
      });
    }

    const { command, serverId, timeout } = req.body;
    const userId = req.session.userId;

    if (!command) {
      return res.status(400).json({ 
        success: false,
        error: 'Command is required' 
      });
    }

    if (!userId) {
      return res.status(401).json({ 
        success: false,
        error: 'User authentication required' 
      });
    }

    logger.info(`Processing shell command execution for user ${userId}: ${command}`);

    // Execute the shell command
    const result = await shellCommandService.executeShellCommand(command, userId, {
      serverId,
      timeout: timeout || 30
    });

    logger.info(`Shell command execution completed for user ${userId}: ${result.success ? 'SUCCESS' : 'FAILED'}`);
    return res.json(result);
  } catch (error) {
    logger.error('Error executing shell command:', error);
    return res.status(500).json({ 
      success: false,
      error: error.message || 'Failed to execute shell command',
      timestamp: new Date().toISOString()
    });
  }
});

/**
 * Test MCP server connection
 * GET /api/ai/tools/runshellcommand/test
 */
router.get('/tools/runshellcommand/test', requireAuth, async (req, res) => {
  try {
    const { serverId } = req.query;
    const userId = req.session.userId;

    if (!userId) {
      return res.status(401).json({ error: 'User authentication required' });
    }

    logger.info(`Testing MCP connection for user ${userId}${serverId ? ` with server ${serverId}` : ' (default server)'}`);

    // Test the connection
    const result = await shellCommandService.testConnection(userId, { serverId });

    return res.json(result);
  } catch (error) {
    logger.error('Error testing MCP connection:', error);
    return res.status(500).json({ 
      success: false,
      error: error.message || 'Failed to test MCP connection'
    });
  }
});

/**
 * Get available tools from MCP server
 * GET /api/ai/tools/runshellcommand/tools
 */
router.get('/tools/runshellcommand/tools', requireAuth, async (req, res) => {
  try {
    const { serverId } = req.query;
    const userId = req.session.userId;

    if (!userId) {
      return res.status(401).json({ error: 'User authentication required' });
    }

    logger.info(`Getting available tools for user ${userId}${serverId ? ` with server ${serverId}` : ' (default server)'}`);

    // Get available tools
    const result = await shellCommandService.getAvailableTools(userId, { serverId });

    return res.json(result);
  } catch (error) {
    logger.error('Error getting available tools:', error);
    return res.status(500).json({ 
      success: false,
      error: error.message || 'Failed to get available tools'
    });
  }
});

/**
 * Get user's MCP server configurations
 * GET /api/ai/tools/runshellcommand/servers
 */
router.get('/tools/runshellcommand/servers', requireAuth, async (req, res) => {
  try {
    const userId = req.session.userId;

    if (!userId) {
      return res.status(401).json({ error: 'User authentication required' });
    }

    logger.info(`Getting MCP servers for user ${userId}`);

    // Get user's MCP servers
    const servers = await shellCommandService.getUserMCPServers(userId);
    const defaultServer = await shellCommandService.getUserDefaultMCPServer(userId);

    return res.json({ 
      success: true,
      servers,
      defaultServer
    });
  } catch (error) {
    logger.error('Error getting user MCP servers:', error);
    return res.status(500).json({ 
      success: false,
      error: error.message || 'Failed to get MCP servers'
    });
  }
});

module.exports = router;
