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
 * Get AI rules for a user
 * GET /api/ai-rules
 */
router.get('/', authenticateToken, async (req, res) => {
  try {
    const userId = req.session.userId;

    const query = `
      SELECT id, rule_content, created_at, updated_at
      FROM user_ai_rules
      WHERE user_id = $1
      ORDER BY updated_at DESC
    `;

    const result = await db.query(query, [userId]);

    return res.json({
      success: true,
      rules: result.rows
    });
  } catch (error) {
    logger.error(`Error fetching AI rules: ${error.message}`, error);
    return res.status(500).json({
      success: false,
      error: 'Failed to fetch AI rules'
    });
  }
});

/**
 * Create or update AI rule for a user
 * POST /api/ai-rules
 */
router.post('/', authenticateToken, async (req, res) => {
  try {
    const userId = req.session.userId;
    const { rule_content } = req.body;

    if (!rule_content) {
      return res.status(400).json({
        success: false,
        error: 'Rule content is required'
      });
    }

    // Check if user already has a rule
    const checkQuery = `
      SELECT id FROM user_ai_rules
      WHERE user_id = $1
    `;

    const checkResult = await db.query(checkQuery, [userId]);

    let result;

    if (checkResult.rows.length > 0) {
      // Update existing rule
      const updateQuery = `
        UPDATE user_ai_rules
        SET rule_content = $1, updated_at = CURRENT_TIMESTAMP
        WHERE user_id = $2
        RETURNING id, rule_content, created_at, updated_at
      `;

      result = await db.query(updateQuery, [rule_content, userId]);
    } else {
      // Create new rule
      const insertQuery = `
        INSERT INTO user_ai_rules (user_id, rule_content)
        VALUES ($1, $2)
        RETURNING id, rule_content, created_at, updated_at
      `;

      result = await db.query(insertQuery, [userId, rule_content]);
    }

    return res.json({
      success: true,
      rule: result.rows[0]
    });
  } catch (error) {
    logger.error(`Error saving AI rule: ${error.message}`, error);
    return res.status(500).json({
      success: false,
      error: 'Failed to save AI rule'
    });
  }
});

/**
 * Delete AI rule
 * DELETE /api/ai-rules/:id
 */
router.delete('/:id', authenticateToken, async (req, res) => {
  try {
    const userId = req.session.userId;
    const ruleId = req.params.id;

    const query = `
      DELETE FROM user_ai_rules
      WHERE id = $1 AND user_id = $2
      RETURNING id
    `;

    const result = await db.query(query, [ruleId, userId]);

    if (result.rows.length === 0) {
      return res.status(404).json({
        success: false,
        error: 'Rule not found or not authorized to delete'
      });
    }

    return res.json({
      success: true,
      message: 'Rule deleted successfully'
    });
  } catch (error) {
    logger.error(`Error deleting AI rule: ${error.message}`, error);
    return res.status(500).json({
      success: false,
      error: 'Failed to delete AI rule'
    });
  }
});

module.exports = router;
