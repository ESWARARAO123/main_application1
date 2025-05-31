import { api } from './api';

export interface AIContext {
  user_rules: string | null;
  has_rules: boolean;
}

/**
 * Read AI context for the current user
 * This can be called by the AI when it needs to access the user's preferences
 * @returns Promise with the user's AI context
 */
export const readAIContext = async (): Promise<AIContext> => {
  try {
    const response = await api.get('/ai-context/read');
    return response.data.context;
  } catch (error) {
    console.error('Error reading AI context:', error);
    // Return default context if there's an error
    return {
      user_rules: null,
      has_rules: false
    };
  }
};
