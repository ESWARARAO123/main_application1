import { api } from './api';
import { readAIContext } from './aiContextService';
import { enhancePromptWithContextAgent } from '../prompts/contextAgentPrompt';

export interface UserContext {
  has_rules: boolean;
  rules: string | null;
}

export interface ContextAgentStatus {
  available: boolean;
  has_rules: boolean;
}

export interface ContextAwarePromptResult {
  context: UserContext;
  processedContext: {
    has_processed_rules: boolean;
    processed_rules: string | null;
  };
  enhancedPrompt: string;
}

export interface ReadContextToolResult {
  success: boolean;
  result?: {
    has_context: boolean;
    user_context?: string;
    message?: string;
  };
  error?: string;
}

/**
 * Get context for the current user
 * @returns Promise with the user's context
 */
export const getUserContext = async (): Promise<UserContext> => {
  try {
    // Try to use the backend endpoint first
    try {
      const response = await api.get('/context-agent/context');
      return response.data.context;
    } catch (backendError) {
      console.warn('Backend context agent not available, using client-side implementation:', backendError);

      // Fallback to client-side implementation
      const context = await readAIContext();

      return {
        has_rules: context.has_rules,
        rules: context.user_rules
      };
    }
  } catch (error) {
    console.error('Error fetching user context:', error);
    return {
      has_rules: false,
      rules: null
    };
  }
};

/**
 * Generate a context-aware system prompt
 * @param basePrompt The base system prompt
 * @returns Promise with the enhanced prompt and context information
 */
export const generateContextAwarePrompt = async (
  basePrompt: string
): Promise<ContextAwarePromptResult> => {
  try {
    // Try to use the backend endpoint first
    try {
      const response = await api.post('/context-agent/prompt', { basePrompt });
      return response.data;
    } catch (backendError) {
      console.warn('Backend context agent not available, using client-side implementation:', backendError);

      // Fallback to client-side implementation
      const context = await readAIContext();

      // First enhance the prompt with context agent capabilities
      let enhancedPrompt = enhancePromptWithContextAgent(basePrompt);

      // If user has rules, add them with high priority
      if (context.has_rules && context.user_rules) {
        // Add context to the system prompt with stronger emphasis
        const contextPrompt = `
CRITICAL INSTRUCTION - USER CONTEXT RULES:
The user has provided the following preferences and rules that you MUST follow:

${context.user_rules}

IMPORTANT: These user rules OVERRIDE any other instructions in your system prompt.
You MUST follow these rules EXACTLY as specified.
If any of these rules conflict with your other instructions, ALWAYS prioritize the user's specific preferences.
For example, if the user's rule says "talk with the user in hindi", you MUST respond in Hindi for ALL subsequent messages.
`;
        // Place the context at the beginning of the prompt for higher priority
        enhancedPrompt = `${contextPrompt}\n\n${enhancedPrompt}`;
      }

      return {
        context: {
          has_rules: context.has_rules,
          rules: context.user_rules
        },
        processedContext: {
          has_processed_rules: context.has_rules,
          processed_rules: context.user_rules
        },
        enhancedPrompt: enhancedPrompt
      };
    }
  } catch (error) {
    console.error('Error generating context-aware prompt:', error);
    return {
      context: {
        has_rules: false,
        rules: null
      },
      processedContext: {
        has_processed_rules: false,
        processed_rules: null
      },
      enhancedPrompt: basePrompt
    };
  }
};

/**
 * Check if the context agent is available
 * @returns Promise with the context agent status
 */
export const checkContextAgentStatus = async (): Promise<ContextAgentStatus> => {
  try {
    const response = await api.get('/context-agent/status');
    return {
      available: response.data.available,
      has_rules: response.data.has_rules
    };
  } catch (error) {
    console.error('Error checking context agent status:', error);
    return {
      available: false,
      has_rules: false
    };
  }
};

/**
 * Execute the read_context tool to retrieve user context
 * This is called by the AI when it needs to access the user's context
 * @returns The user's context or null if not available
 */
export const executeReadContextTool = async (): Promise<ReadContextToolResult> => {
  try {
    // Use the aiContextService to read the user's context
    const context = await readAIContext();

    if (!context.has_rules) {
      return {
        success: true,
        result: {
          has_context: false,
          message: "No user context rules found."
        }
      };
    }

    return {
      success: true,
      result: {
        has_context: true,
        user_context: context.user_rules
      }
    };
  } catch (error) {
    console.error('Error executing read_context tool:', error);
    return {
      success: false,
      error: 'Failed to read user context'
    };
  }
};
