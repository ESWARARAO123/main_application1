const express = require('express');
const router = express.Router();
const { db } = require('../database');
const logger = require('../utils/logger');

// Middleware to check if user is authenticated
const authenticateToken = (req, res, next) => {
  if (req.session && req.session.userId) {
    next();
  } else {
    res.status(401).json({ error: 'Unauthorized' });
  }
};

/**
 * Read AI context for a user
 * GET /api/ai-context/read
 *
 * This endpoint allows the AI to read the user's context rules
 * It can be called by the AI when it needs to access the user's preferences
 */
router.get('/read', authenticateToken, async (req, res) => {
  try {
    const userId = req.session.userId;

    // Get the user's AI rules
    const query = `
      SELECT rule_content
      FROM user_ai_rules
      WHERE user_id = $1
      ORDER BY updated_at DESC
      LIMIT 1
    `;

    const result = await db.query(query, [userId]);

    // If the user has no rules, return an empty context
    if (result.rows.length === 0) {
      return res.json({
        success: true,
        context: {
          user_rules: null,
          has_rules: false
        }
      });
    }

    // Return the user's rules as context
    return res.json({
      success: true,
      context: {
        user_rules: result.rows[0].rule_content,
        has_rules: true
      }
    });
  } catch (error) {
    logger.error(`Error reading AI context: ${error.message}`, error);
    return res.status(500).json({
      success: false,
      error: 'Failed to read AI context'
    });
  }
});

module.exports = router;
