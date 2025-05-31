/**
 * Context Agent Routes
 * 
 * These routes handle context agent functionality, allowing the AI to retrieve
 * and utilize user context when MCP mode is enabled.
 */

const express = require('express');
const router = express.Router();
const { db } = require('../database');
const logger = require('../utils/logger');
const contextAgentService = require('../services/contextAgentService');

// Middleware to check if user is authenticated
const authenticateToken = (req, res, next) => {
  if (req.session && req.session.userId) {
    next();
  } else {
    res.status(401).json({ error: 'Unauthorized' });
  }
};

/**
 * Get context for the current user
 * GET /api/context-agent/context
 * 
 * This endpoint is used by the AI to retrieve the user's context
 * when MCP mode is enabled.
 */
router.get('/context', authenticateToken, async (req, res) => {
  try {
    const userId = req.session.userId;
    
    // Get user context rules
    const context = await contextAgentService.getUserContextRules(userId);
    
    return res.json({
      success: true,
      context
    });
  } catch (error) {
    logger.error(`Error retrieving context: ${error.message}`, error);
    return res.status(500).json({
      success: false,
      error: 'Failed to retrieve context'
    });
  }
});

/**
 * Generate context-aware prompt
 * POST /api/context-agent/prompt
 * 
 * This endpoint generates a context-aware system prompt based on
 * the user's context rules and the provided base prompt.
 */
router.post('/prompt', authenticateToken, async (req, res) => {
  try {
    const userId = req.session.userId;
    const { basePrompt } = req.body;
    
    if (!basePrompt) {
      return res.status(400).json({
        success: false,
        error: 'Base prompt is required'
      });
    }
    
    // Get context and generate enhanced prompt
    const result = await contextAgentService.getContextForUser(userId, basePrompt);
    
    return res.json({
      success: true,
      ...result
    });
  } catch (error) {
    logger.error(`Error generating context-aware prompt: ${error.message}`, error);
    return res.status(500).json({
      success: false,
      error: 'Failed to generate context-aware prompt'
    });
  }
});

/**
 * Check if context agent is available
 * GET /api/context-agent/status
 * 
 * This endpoint checks if the context agent is available and
 * if the user has any context rules.
 */
router.get('/status', authenticateToken, async (req, res) => {
  try {
    const userId = req.session.userId;
    
    // Check if user has context rules
    const context = await contextAgentService.getUserContextRules(userId);
    
    return res.json({
      success: true,
      available: true,
      has_rules: context.has_rules
    });
  } catch (error) {
    logger.error(`Error checking context agent status: ${error.message}`, error);
    return res.json({
      success: false,
      available: false,
      error: 'Context agent is not available'
    });
  }
});

module.exports = router;
