import { api } from './api';

export interface AIRule {
  id: number;
  rule_content: string;
  created_at: string;
  updated_at: string;
}

/**
 * Get AI rules for the current user
 * @returns Promise with the user's AI rules
 */
export const getUserAIRules = async (): Promise<AIRule[]> => {
  try {
    const response = await api.get('/ai-rules');
    return response.data.rules || [];
  } catch (error) {
    console.error('Error fetching AI rules:', error);
    throw error;
  }
};

/**
 * Save AI rule for the current user
 * @param ruleContent The rule content to save
 * @returns Promise with the saved rule
 */
export const saveAIRule = async (ruleContent: string): Promise<AIRule> => {
  try {
    const response = await api.post('/ai-rules', { rule_content: ruleContent });
    return response.data.rule;
  } catch (error) {
    console.error('Error saving AI rule:', error);
    throw error;
  }
};

/**
 * Delete an AI rule
 * @param ruleId The ID of the rule to delete
 * @returns Promise with success status
 */
export const deleteAIRule = async (ruleId: number): Promise<boolean> => {
  try {
    const response = await api.delete(`/ai-rules/${ruleId}`);
    return response.data.success;
  } catch (error) {
    console.error('Error deleting AI rule:', error);
    throw error;
  }
};
