import { readAIContext } from './aiContextService';
import { api } from './api';

/**
 * Interface for AI tools that can be used in the chat
 */
export interface AITool {
  name: string;
  description: string;
  execute: (params: any) => Promise<any>;
}

/**
 * Read context tool for AI
 * Allows the AI to access the user's context (rules, preferences, etc.)
 */
export const readContextTool: AITool = {
  name: 'read_context',
  description: 'Read the user\'s context including rules and preferences',
  execute: async () => {
    try {
      const context = await readAIContext();
      return {
        success: true,
        context
      };
    } catch (error) {
      console.error('Error executing read_context tool:', error);
      return {
        success: false,
        error: 'Failed to read context'
      };
    }
  }
};

/**
 * Run shell command tool
 * Allows the AI to execute shell commands via MCP orchestrator
 */
export const runShellCommandTool: AITool = {
  name: 'runshellcommand',
  description: 'Execute shell commands on the system via MCP orchestrator. Use Linux commands for better compatibility.',
  execute: async (params: { command: string; serverId?: string; timeout?: number }) => {
    try {
      const { command, serverId, timeout } = params;
      
      if (!command) {
        return {
          success: false,
          error: 'Command parameter is required'
        };
      }

      const response = await api.post('/ai/tools/runshellcommand', {
        command,
        serverId,
        timeout: timeout || 30
      });

      return response.data;
    } catch (error: any) {
      console.error('Error executing runshellcommand tool:', error);
      
      // Extract error message from response if available
      const errorMessage = error.response?.data?.error || error.message || 'Failed to execute shell command';
      
      return {
        success: false,
        error: errorMessage,
        command: params.command
      };
    }
  }
};

/**
 * List of available AI tools
 */
export const aiTools: AITool[] = [
  readContextTool,
  runShellCommandTool
];

/**
 * Get an AI tool by name
 * @param name The name of the tool to get
 * @returns The tool or undefined if not found
 */
export const getToolByName = (name: string): AITool | undefined => {
  return aiTools.find(tool => tool.name === name);
};
