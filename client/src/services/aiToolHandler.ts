import { executeReadContextTool } from './contextAgentService';
import { runShellCommandTool } from './aiToolsService';

/**
 * Interface for AI tool parameters
 */
export interface AIToolParams {
  [key: string]: any;
}

/**
 * Interface for AI tool result
 */
export interface AIToolResult {
  success: boolean;
  result?: any;
  error?: string;
  toolName?: string;
}

/**
 * Type for tool execution callbacks
 */
export type ToolExecutionCallback = (status: 'start' | 'end', toolName: string) => void;

/**
 * Handles AI tool execution based on tool name
 * @param toolName The name of the tool to execute
 * @param params Parameters for the tool
 * @param callback Optional callback for tool execution status updates
 * @returns Promise with the tool execution result
 */
export const handleToolExecution = async (
  toolName: string,
  params: AIToolParams = {},
  callback?: ToolExecutionCallback
): Promise<AIToolResult> => {
  console.log(`Executing tool: ${toolName} with params:`, params);

  // Notify start of tool execution
  if (callback) {
    callback('start', toolName);
  }

  let result: AIToolResult;

  try {
    switch (toolName) {
      case 'read_context':
        result = await executeReadContextTool();
        break;

      case 'runshellcommand':
        const shellResult = await runShellCommandTool.execute(params);
        result = {
          success: shellResult.success,
          result: shellResult,
          error: shellResult.success ? undefined : shellResult.error
        };
        break;

      // Add more tools here as needed

      default:
        result = {
          success: false,
          error: `Unknown tool: ${toolName}`
        };
    }

    // Add the tool name to the result
    result.toolName = toolName;

    return result;
  } finally {
    // Notify end of tool execution (even if there was an error)
    if (callback) {
      callback('end', toolName);
    }
  }
};
