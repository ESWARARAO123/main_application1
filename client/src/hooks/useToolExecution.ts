import { useState, useCallback } from 'react';
import { handleToolExecution, AIToolResult } from '../services/aiToolHandler';
import { extractToolCall, ParsedToolCall, replaceToolCall } from '../utils/toolParser';

interface ToolExecutionState {
  isExecutingTool: boolean;
  currentTool: string | null;
  executeTool: (text: string) => Promise<string>;
}

/**
 * Hook for handling tool execution in chat
 * @returns Tool execution state and functions
 */
export const useToolExecution = (): ToolExecutionState => {
  const [isExecutingTool, setIsExecutingTool] = useState<boolean>(false);
  const [currentTool, setCurrentTool] = useState<string | null>(null);

  /**
   * Executes a tool if the text contains a tool call
   * @param text The text to check for tool calls
   * @returns The text with tool calls replaced with results
   */
  const executeTool = useCallback(async (text: string): Promise<string> => {
    // Extract tool call from text
    const toolCall = extractToolCall(text);
    if (!toolCall) {
      return text;
    }

    // Special handling for read_context tool - don't execute automatically
    // and don't modify the text so the UI component can handle it in-place
    if (toolCall.isContextReadingTool) {
      console.log('Found read_context tool call - will be handled by UI component');
      // Set the current tool to read_context so the UI knows what's happening
      setCurrentTool('read_context');
      return text;
    }

    try {
      // Set executing state
      setIsExecutingTool(true);
      setCurrentTool(toolCall.toolName);

      // Execute the tool
      const result = await handleToolExecution(
        toolCall.toolName,
        toolCall.parameters,
        (status, toolName) => {
          if (status === 'start') {
            console.log(`Started executing tool: ${toolName}`);
          } else {
            console.log(`Finished executing tool: ${toolName}`);
          }
        }
      );

      // Format the result
      const formattedResult = formatToolResult(toolCall, result);

      // Replace the tool call with the result
      const updatedText = replaceToolCall(text, toolCall, formattedResult);

      return updatedText;
    } catch (error) {
      console.error('Error executing tool:', error);

      // Replace the tool call with an error message
      const errorMessage = `[Error executing ${toolCall.toolName}: ${error.message}]`;
      return replaceToolCall(text, toolCall, errorMessage);
    } finally {
      // Reset state
      setIsExecutingTool(false);
      setCurrentTool(null);
    }
  }, []);

  /**
   * Formats a tool result for display
   * @param toolCall The original tool call
   * @param result The result of the tool execution
   * @returns Formatted result text
   */
  const formatToolResult = (toolCall: ParsedToolCall, result: AIToolResult): string => {
    if (!result.success) {
      return `[Error executing ${toolCall.toolName}: ${result.error}]`;
    }

    // Special handling for read_context tool - this is now handled by the UI component
    // We should never reach this code for read_context tools, but keep it for safety
    if (toolCall.toolName === 'read_context') {
      console.log('formatToolResult called for read_context tool - this should be handled by the UI component');
      // Return the original text so the UI component can handle it
      return '';
    }

    // Generic formatting for other tools
    return `[Tool result: ${JSON.stringify(result.result)}]`;
  };

  return {
    isExecutingTool,
    currentTool,
    executeTool
  };
};
