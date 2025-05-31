/**
 * Ollama API Routes
 * Handles API endpoints for Ollama settings, models, and chat interactions
 */

const express = require('express');
const router = express.Router();
const OllamaService = require('../services/ollamaService');
const { requireAuth, requireAdmin } = require('./auth');
const { logger } = require('../utils/logger');

module.exports = function(config) {
  // Initialize the Ollama service
  const ollamaService = new OllamaService(config);
  ollamaService.initialize().catch(err => {
    logger.error(`Failed to initialize OllamaService: ${err.message}`);
  });

  /**
   * Get current Ollama settings (Admin only)
   * GET /api/ollama/settings
   */
  router.get('/settings', requireAdmin, async (req, res) => {
    try {
      await ollamaService.loadSettings();
      res.json(ollamaService.settings);
    } catch (error) {
      logger.error(`Error getting Ollama settings: ${error.message}`);
      res.status(500).json({ error: 'Failed to retrieve Ollama settings' });
    }
  });

  /**
   * Update Ollama settings (Admin only)
   * PUT /api/ollama/settings
   */
  router.put('/settings', requireAdmin, async (req, res) => {
    try {
      const { host, port, default_model } = req.body;

      if (!host) {
        return res.status(400).json({
          success: false,
          message: 'Host is required'
        });
      }

      if (!port || isNaN(parseInt(port))) {
        return res.status(400).json({
          success: false,
          message: 'Valid port is required'
        });
      }

      const settings = await ollamaService.saveSettings({
        host,
        port: parseInt(port),
        default_model: default_model || ''
      });

      // Return a properly formatted success response
      res.json({
        success: true,
        settings,
        message: 'Settings saved successfully'
      });
    } catch (error) {
      logger.error(`Error saving Ollama settings: ${error.message}`);
      res.status(500).json({
        success: false,
        message: `Failed to save settings: ${error.message}`
      });
    }
  });

  /**
   * Test connection to Ollama server (Admin only)
   * POST /api/ollama/test-connection
   */
  router.post('/test-connection', requireAdmin, async (req, res) => {
    try {
      const { host, port } = req.body;
      const result = await ollamaService.testConnection(host, port);
      res.json(result);
    } catch (error) {
      logger.error(`Error testing Ollama connection: ${error.message}`);
      res.status(500).json({
        success: false,
        message: 'Failed to test connection',
        error: error.message
      });
    }
  });

  /**
   * Get available models directly from Ollama server (Admin only)
   * GET /api/ollama/models/available
   */
  router.get('/models/available', requireAdmin, async (req, res) => {
    try {
      const result = await ollamaService.getAvailableModels();
      res.json(result);
    } catch (error) {
      logger.error(`Error fetching available models: ${error.message}`);
      res.status(500).json({
        success: false,
        models: [],
        error: `Failed to fetch available models: ${error.message}`
      });
    }
  });

  /**
   * Sync models from Ollama server to database (Admin only)
   * POST /api/ollama/models/sync
   */
  router.post('/models/sync', requireAdmin, async (req, res) => {
    try {
      const { selectedModelIds } = req.body;
      const result = await ollamaService.syncModels(selectedModelIds);

      if (result.success) {
        res.json({
          success: true,
          syncedCount: result.added + result.updated,
          added: result.added,
          updated: result.updated,
          unchanged: result.unchanged,
          inactivated: result.inactivated || 0,
          total: result.total
        });
      } else {
        res.status(500).json({
          success: false,
          message: result.error || 'Failed to sync models'
        });
      }
    } catch (error) {
      logger.error(`Error syncing models: ${error.message}`);
      res.status(500).json({
        success: false,
        message: `Failed to sync models: ${error.message}`
      });
    }
  });

  /**
   * Get all models from database (Admin only)
   * GET /api/ollama/models/db
   */
  router.get('/models/db', requireAdmin, async (req, res) => {
    try {
      const result = await ollamaService.getModelsFromDB();
      res.json(result);
    } catch (error) {
      logger.error(`Error fetching models from DB: ${error.message}`);
      res.status(500).json({
        success: false,
        models: [],
        error: `Failed to fetch models from database: ${error.message}`
      });
    }
  });

  /**
   * Toggle model active status (Admin only)
   * PUT /api/ollama/models/:id/status
   */
  router.put('/models/:id/status', requireAdmin, async (req, res) => {
    try {
      const { id } = req.params;
      const { is_active } = req.body;

      if (is_active === undefined) {
        return res.status(400).json({ error: 'is_active status is required' });
      }

      const result = await ollamaService.updateModelStatus(id, is_active);

      if (!result.success) {
        return res.status(404).json({ error: result.error || 'Model not found' });
      }

      res.json(result.model);
    } catch (error) {
      logger.error(`Error updating model status: ${error.message}`);
      res.status(500).json({ error: `Failed to update model status: ${error.message}` });
    }
  });

  /**
   * Get active models from database (Authenticated users)
   * GET /api/ollama/models/active
   */
  router.get('/models/active', requireAuth, async (req, res) => {
    try {
      const result = await ollamaService.getActiveModels();
      res.json(result);
    } catch (error) {
      logger.error(`Error fetching active models: ${error.message}`);
      res.status(500).json({
        success: false,
        models: [],
        error: `Failed to fetch active models: ${error.message}`
      });
    }
  });

  /**
   * Check Ollama API status
   * GET /api/ollama/status
   */
  router.get('/status', requireAuth, async (req, res) => {
    try {
      const result = await ollamaService.testConnection();
      res.json(result);
    } catch (error) {
      logger.error(`Error checking Ollama status: ${error.message}`);
      res.status(500).json({
        success: false,
        message: 'Failed to check Ollama status',
        error: error.message
      });
    }
  });

  /**
   * Send a chat message to Ollama (Authenticated users)
   * POST /api/ollama/chat
   *
   * Request body:
   * {
   *   "modelId": "uuid-of-model-from-db",
   *   "messages": [
   *     { "role": "user", "content": "Hello" },
   *     { "role": "assistant", "content": "Hi there!" },
   *     { "role": "user", "content": "How are you?" }
   *   ],
   *   "options": {
   *     "temperature": 0.7,
   *     "max_tokens": 2048
   *   }
   * }
   */
  router.post('/chat', requireAuth, async (req, res) => {
    try {
      const { model, messages, system_prompt, options } = req.body;

      if (!model) {
        return res.status(400).json({ error: 'Model ID is required' });
      }

      if (!messages || !Array.isArray(messages) || messages.length === 0) {
        return res.status(400).json({ error: 'Valid messages array is required' });
      }

      // Check if streaming is requested
      const stream = options?.stream === true;

      if (stream) {
        // Set appropriate headers for streaming
        res.setHeader('Content-Type', 'application/json');
        res.setHeader('Transfer-Encoding', 'chunked');
        res.setHeader('Cache-Control', 'no-cache');
        res.setHeader('Connection', 'keep-alive');

        // Create a callback function to send chunks to the client
        const sendChunk = (chunk) => {
          res.write(JSON.stringify(chunk) + '\n');
        };

        try {
          // Call the streaming version of chat
          const result = await ollamaService.chat(model, messages, system_prompt, true, sendChunk);

          if (!result.success) {
            // If there was an error during streaming, the client might have already received some data
            // So we send a special error chunk and end the response
            res.write(JSON.stringify({ error: result.error || 'Chat request failed' }) + '\n');
            res.end();
            return;
          }

          // End the response when streaming is complete
          res.end();
        } catch (streamError) {
          logger.error(`Error in streaming chat with Ollama: ${streamError.message}`);
          // Try to send an error message if we can
          try {
            res.write(JSON.stringify({ error: `Streaming error: ${streamError.message}` }) + '\n');
          } catch (writeError) {
            logger.error(`Failed to write error to stream: ${writeError.message}`);
          }
          res.end();
        }
      } else {
        // Non-streaming request
        const result = await ollamaService.chat(model, messages, system_prompt);

        if (!result.success) {
          return res.status(500).json({ error: result.error || 'Chat request failed' });
        }

        res.json(result.response);
      }
    } catch (error) {
      logger.error(`Error in chat with Ollama: ${error.message}`, error);
      res.status(500).json({
        error: `Chat request failed: ${error.message}`,
        stack: process.env.NODE_ENV === 'development' ? error.stack : undefined,
        details: error.response?.data || error.cause || 'No additional details available'
      });
    }
  });

  /**
   * Delete RAG data for a session
   * DELETE /api/ollama/rag-data/:sessionId
   */
  router.delete('/rag-data/:sessionId', requireAuth, async (req, res) => {
    try {
      const { sessionId } = req.params;
      const userId = req.session.userId;

      if (!sessionId) {
        return res.status(400).json({
          success: false,
          message: 'Session ID is required'
        });
      }

      logger.info(`Deleting RAG data for session ${sessionId} requested by user ${userId}`);

      // Import the vector store service
      const vectorStoreService = require('../services/vectorStoreService');

      // Delete the collection for this session
      const result = await vectorStoreService.deleteSessionData(sessionId, userId);

      logger.info(`RAG data deletion result for session ${sessionId}: ${result.success ? 'Success' : 'Failed'}`);
      if (result.success) {
        logger.info(`Successfully deleted RAG data for session ${sessionId}`);
      } else {
        logger.warn(`Failed to delete RAG data: ${result.error}`);
      }

      res.json({
        success: result.success,
        message: result.success
          ? `Successfully deleted RAG data for session ${sessionId}`
          : `Failed to delete RAG data: ${result.error}`
      });
    } catch (error) {
      logger.error(`Error deleting RAG data for session: ${error.message}`, error);
      res.status(500).json({
        success: false,
        error: `Failed to delete RAG data: ${error.message}`
      });
    }
  });

  /**
   * Check if RAG is available
   * GET /api/ollama/rag-availability
   */
  router.get('/rag-availability', requireAuth, async (req, res) => {
    try {
      // Import the RAG service
      const ragService = require('../services/ragService');

      // Check if RAG is available
      const available = await ragService.isRagAvailable();

      res.json({
        available,
        message: available
          ? 'RAG is available'
          : 'RAG is not available. No documents have been processed yet.'
      });
    } catch (error) {
      logger.error(`Error checking RAG availability: ${error.message}`, error);
      res.status(500).json({
        available: false,
        error: `Failed to check RAG availability: ${error.message}`
      });
    }
  });

  /**
   * Send a RAG-enhanced chat message to Ollama (Authenticated users)
   * POST /api/ollama/rag-chat
   *
   * Request body:
   * {
   *   "model": "model-name",
   *   "message": "User question about documents",
   *   "sessionId": "optional-session-id"
   * }
   */
  router.post('/rag-chat', requireAuth, async (req, res) => {
    try {
      // Import the RAG service
      const ragService = require('../services/ragService');

      const { model, message, sessionId } = req.body;
      const userId = req.session.userId;

      if (!model) {
        return res.status(400).json({ error: 'Model is required' });
      }

      if (!message) {
        return res.status(400).json({ error: 'Message is required' });
      }

      // Check if RAG is available
      const ragAvailable = await ragService.isRagAvailable();
      if (!ragAvailable) {
        logger.warn(`RAG chat requested but RAG is not available. Returning error.`);
        return res.status(400).json({
          error: 'RAG is not available. No documents have been processed yet.',
          ragAvailable: false
        });
      }

      logger.info(`RAG is available, processing chat message: "${message.substring(0, 50)}..."`);
      if (sessionId) {
        logger.info(`Using session context: ${sessionId}`);
      } else {
        logger.info(`No session ID provided, using global context`);
      }

      // Process the message with RAG
      const result = await ragService.processRagChat(message, model, {
        userId,
        sessionId // Pass sessionId to ensure context filtering
      });

      if (!result.success) {
        return res.status(500).json({
          error: result.error || 'RAG chat request failed',
          ragAvailable: true
        });
      }

      // Extract the content from the response
      let content = '';
      if (result.response && result.response.choices && result.response.choices.length > 0) {
        content = result.response.choices[0].message.content;
      } else if (typeof result.response === 'string') {
        content = result.response;
      } else {
        logger.warn(`Unexpected response format from RAG service:`, result.response);
        content = 'I processed your request but encountered an issue with the response format.';
      }

      // Return the response with sources
      res.json({
        content: content,
        sources: result.sources || [],
        ragAvailable: true
      });
    } catch (error) {
      logger.error(`Error in RAG chat: ${error.message}`, error);
      res.status(500).json({
        error: `RAG chat request failed: ${error.message}`,
        ragAvailable: true,
        details: error.stack
      });
    }
  });

  return router;
};