/**
 * Context Agent Service
 *
 * This service is responsible for retrieving and processing user context rules
 * when MCP mode is enabled. It provides the AI with the necessary context to
 * understand user preferences and tailor its responses accordingly.
 */

const { db } = require('../database');
const logger = require('../utils/logger');

/**
 * Get user context rules by user ID
 * @param {string} userId - The user's UUID
 * @returns {Promise<Object>} - The user's context rules
 */
async function getUserContextRules(userId) {
  try {
    // Query the database for the user's AI rules
    const query = `
      SELECT rule_content
      FROM user_ai_rules
      WHERE user_id = $1
      ORDER BY updated_at DESC
      LIMIT 1
    `;

    const result = await db.query(query, [userId]);

    // If no rules found, return empty context
    if (result.rows.length === 0) {
      return {
        has_rules: false,
        rules: null
      };
    }

    // Return the rules
    return {
      has_rules: true,
      rules: result.rows[0].rule_content
    };
  } catch (error) {
    logger.error(`Error retrieving user context rules: ${error.message}`, error);
    throw error;
  }
}

/**
 * Process context rules to extract relevant information
 * @param {string} rules - The raw context rules
 * @returns {Object} - Processed context information
 */
function processContextRules(rules) {
  if (!rules) {
    return {
      has_processed_rules: false,
      processed_rules: null
    };
  }

  try {
    // For now, we're just returning the raw rules
    // In the future, we could parse and categorize rules here
    return {
      has_processed_rules: true,
      processed_rules: rules
    };
  } catch (error) {
    logger.error(`Error processing context rules: ${error.message}`, error);
    return {
      has_processed_rules: false,
      processed_rules: null,
      error: error.message
    };
  }
}

/**
 * Generate context-aware system prompt
 * @param {string} basePrompt - The base system prompt
 * @param {Object} context - The user's context
 * @returns {string} - Enhanced system prompt with context
 */
function generateContextAwarePrompt(basePrompt, context) {
  if (!context || !context.has_rules || !context.rules) {
    return basePrompt;
  }

  // Add context to the system prompt with stronger emphasis
  const contextPrompt = `
CRITICAL INSTRUCTION - USER CONTEXT RULES:
The user has provided the following preferences and rules that you MUST follow:

${context.rules}

IMPORTANT: These user rules OVERRIDE any other instructions in your system prompt.
You MUST follow these rules EXACTLY as specified.
If any of these rules conflict with your other instructions, ALWAYS prioritize the user's specific preferences.
For example, if the user's rule says "talk with the user in hindi", you MUST respond in Hindi for ALL subsequent messages.
`;

  // Place the context at the beginning of the prompt for higher priority
  return `${contextPrompt}\n\n${basePrompt}`;
}

/**
 * Main context agent function that retrieves and processes user context
 * @param {string} userId - The user's UUID
 * @param {string} basePrompt - The base system prompt
 * @returns {Promise<Object>} - Context information and enhanced prompt
 */
async function getContextForUser(userId, basePrompt) {
  try {
    // Get user context rules
    const context = await getUserContextRules(userId);

    // Process the rules
    const processedContext = context.has_rules
      ? processContextRules(context.rules)
      : { has_processed_rules: false, processed_rules: null };

    // Generate context-aware prompt
    const enhancedPrompt = generateContextAwarePrompt(basePrompt, context);

    return {
      context,
      processedContext,
      enhancedPrompt
    };
  } catch (error) {
    logger.error(`Context agent error: ${error.message}`, error);
    return {
      error: error.message,
      enhancedPrompt: basePrompt // Fall back to base prompt on error
    };
  }
}

module.exports = {
  getUserContextRules,
  processContextRules,
  generateContextAwarePrompt,
  getContextForUser
};
