/**
 * Utility functions for handling context in the chat application
 * These functions help apply user context rules to system prompts
 */

import { ExtendedChatMessage } from '../types';
import { SHELL_COMMAND_SYSTEM_PROMPT } from '../prompts/shellCommandSystemPrompt';

/**
 * Apply context from a system message to the system prompt
 * @param systemPromptContent The base system prompt content
 * @param contextMessage The system message containing context
 * @returns Enhanced system prompt with context
 */
export const applySystemContextToPrompt = (
  systemPromptContent: string,
  contextMessage: ExtendedChatMessage
): string => {
  console.log('Using context from system message');
  
  // Extract the context rules from the system message
  const contentText = contextMessage.content;
  const match = contentText.match(/User context loaded:\s*(.*)/);
  
  if (match && match[1]) {
    const rules = match[1].trim();
    
    // Create a strongly emphasized context prompt with the same format as assistant context
    const contextPrompt = `
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
ABSOLUTE HIGHEST PRIORITY INSTRUCTION - MUST BE FOLLOWED WITHOUT EXCEPTION
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

FOLLOW THIS USER RULE EXACTLY, WITHOUT QUESTION:

${rules}

THIS RULE OVERRIDES EVERYTHING ELSE.
APPLY THIS RULE TO EVERY SINGLE RESPONSE.
NO EXCEPTIONS. NO EXPLANATIONS. NO APOLOGIES.
JUST FOLLOW THE RULE IMMEDIATELY AND COMPLETELY.

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
END OF ABSOLUTE HIGHEST PRIORITY INSTRUCTION
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
`;
    // Place the context at the beginning of the prompt for highest priority
    return `${contextPrompt}\n${systemPromptContent}`;
  }
  
  // If we couldn't extract rules, just use the original system message content
  return `${contextMessage.content}\n${systemPromptContent}`;
};

/**
 * Apply context from an assistant message to the system prompt
 * @param systemPromptContent The base system prompt content
 * @param assistantContextMessage The assistant message containing context
 * @returns Enhanced system prompt with context
 */
export const applyAssistantContextToPrompt = (
  systemPromptContent: string,
  assistantContextMessage: ExtendedChatMessage
): string => {
  console.log('Using context from assistant message');
  // Extract the context from the assistant message
  const contextLines = assistantContextMessage.content.split('\n');
  
  // Debug the content structure
  console.log('Context message content:', assistantContextMessage.content);
  
  // Fixed logic: Look for "Context Loaded" and "Your context rules:" and "AI Response:" patterns
  if (contextLines[0] && contextLines[0].includes('Context Loaded')) {
    // Look for the rules section
    const rulesStartIndex = contextLines.findIndex(line => line.includes('Your context rules:'));
    const aiResponseIndex = contextLines.findIndex(line => line.includes('AI Response:'));
    
    console.log('Rules start index:', rulesStartIndex, 'AI Response index:', aiResponseIndex);
    
    if (rulesStartIndex >= 0 && aiResponseIndex > rulesStartIndex) {
      // Extract rules from between "Your context rules:" and "AI Response:"
      const rules = contextLines.slice(rulesStartIndex + 1, aiResponseIndex)
        .join('\n')
        .trim();
      
      console.log('Extracted rules:', rules);
      
      if (rules) {
        // Create a strongly emphasized context prompt
        const contextPrompt = `
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
ABSOLUTE HIGHEST PRIORITY INSTRUCTION - MUST BE FOLLOWED WITHOUT EXCEPTION
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

FOLLOW THIS USER RULE EXACTLY, WITHOUT QUESTION:

${rules}

THIS RULE OVERRIDES EVERYTHING ELSE.
APPLY THIS RULE TO EVERY SINGLE RESPONSE.
NO EXCEPTIONS. NO EXPLANATIONS. NO APOLOGIES.
JUST FOLLOW THE RULE IMMEDIATELY AND COMPLETELY.

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
END OF ABSOLUTE HIGHEST PRIORITY INSTRUCTION
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
`;
        // Place the context at the beginning of the prompt for highest priority
        return `${contextPrompt}\n${systemPromptContent}`;
      }
    }
  }

  // If we couldn't extract rules, return the original prompt
  return systemPromptContent;
};

/**
 * Enhance system prompt with shell command capabilities
 * @param systemPromptContent The base system prompt content
 * @returns Enhanced system prompt with shell command capabilities
 */
export const enhancePromptWithShellCommands = (systemPromptContent: string): string => {
  return `${systemPromptContent}\n\n${SHELL_COMMAND_SYSTEM_PROMPT}`;
};

/**
 * Extract and summarize shell command execution context from messages
 * @param messages The conversation messages
 * @returns Summary of command execution context
 */
export const extractShellCommandContext = (messages: ExtendedChatMessage[]): string => {
  const commandExecutions: string[] = [];
  
  // Look for system messages containing shell command results (both old and new format)
  const shellMessages = messages.filter(msg => 
    (msg.role === 'assistant' && msg.content.includes('SYSTEM: Shell command execution result for AI context:')) ||
    (msg.role === 'system' && msg.isContextUpdate && msg.content.includes('SHELL_COMMAND_EXECUTED:'))
  );
  
  console.log(`Found ${shellMessages.length} shell command executions in conversation`);
  
  shellMessages.forEach((msg, index) => {
    const lines = msg.content.split('\n');
    
    // Handle new format (system messages with isContextUpdate)
    if (msg.role === 'system' && msg.isContextUpdate) {
      const contextContent = msg.content.trim();
      if (contextContent) {
        console.log(`Shell command ${index + 1}:`, contextContent.substring(0, 100) + '...');
        commandExecutions.push(`[Command Execution ${index + 1}]\n${contextContent}`);
      }
    } else {
      // Handle old format (assistant messages with embedded context)
      const contextStart = lines.findIndex(line => line.includes('SYSTEM: Shell command execution result for AI context:'));
      
      if (contextStart >= 0) {
        const contextContent = lines.slice(contextStart + 1).join('\n').trim();
        if (contextContent) {
          console.log(`Shell command ${index + 1} (old format):`, contextContent.substring(0, 100) + '...');
          commandExecutions.push(`[Command Execution ${index + 1}]\n${contextContent}`);
        }
      }
    }
  });
  
  if (commandExecutions.length > 0) {
    // Prioritize the most recent command execution by putting it first
    const recentCommands = commandExecutions.slice(-3).reverse(); // Get last 3 commands, most recent first
    
    const shellContext = `

## RECENT SHELL COMMAND EXECUTIONS (MOST RECENT FIRST)
You have previously executed the following commands in this conversation. 
**IMPORTANT**: When answering questions about "previous command" or "last command", refer to the FIRST command listed below (most recent).

${recentCommands.join('\n\n')}

## END OF COMMAND HISTORY

`;
    
    console.log('Generated shell context for AI:', shellContext.substring(0, 200) + '...');
    return shellContext;
  }
  
  console.log('No shell command executions found');
  return '';
};

/**
 * Apply context to the system prompt
 * This function checks for context in system or assistant messages and applies it to the system prompt
 * @param systemPromptContent The base system prompt content
 * @param messages The conversation messages
 * @param options Enhancement options
 * @returns Enhanced system prompt with context
 */
export const applyContextToPrompt = (
  systemPromptContent: string,
  messages: ExtendedChatMessage[],
  options: {
    enableShellCommands?: boolean;
  } = {}
): string => {
  console.log('=== APPLYING CONTEXT TO PROMPT ===');
  console.log('Base system prompt length:', systemPromptContent.length);
  console.log('Shell commands enabled:', options.enableShellCommands);
  console.log('Total messages in conversation:', messages.length);

  let enhancedPrompt = systemPromptContent;

  // Add shell command capabilities if enabled
  if (options.enableShellCommands) {
    enhancedPrompt = enhancePromptWithShellCommands(enhancedPrompt);
    console.log('Enhanced prompt with shell commands, new length:', enhancedPrompt.length);
    
    // Add shell command execution context
    const shellContext = extractShellCommandContext(messages);
    if (shellContext) {
      enhancedPrompt += shellContext;
      console.log('Added shell command context, final length:', enhancedPrompt.length);
      console.log('Shell context preview:', shellContext.substring(0, 300) + '...');
    } else {
      console.log('No shell command context found');
    }
  }

  // Find any context messages in the conversation history
  const contextMessage = messages.find(msg =>
    (msg.role === 'system' && msg.content.includes('User context loaded:'))
  );

  // Extract context from assistant messages if system message not found
  const assistantContextMessage = messages.find(msg =>
    (msg.role === 'assistant' && msg.content.startsWith('Context Loaded'))
  );

  // Apply context to the system prompt
  if (contextMessage) {
    console.log('Found system context message, applying...');
    const finalPrompt = applySystemContextToPrompt(enhancedPrompt, contextMessage);
    console.log('Final prompt length after system context:', finalPrompt.length);
    return finalPrompt;
  } else if (assistantContextMessage) {
    console.log('Found assistant context message, applying...');
    const finalPrompt = applyAssistantContextToPrompt(enhancedPrompt, assistantContextMessage);
    console.log('Final prompt length after assistant context:', finalPrompt.length);
    return finalPrompt;
  }

  // If no context found, return the enhanced prompt
  console.log('No user context messages found');
  console.log('Final enhanced prompt length:', enhancedPrompt.length);
  console.log('=== CONTEXT APPLICATION COMPLETE ===');
  return enhancedPrompt;
};

/**
 * Check if any message in the conversation contains context information
 * @param messages The conversation messages
 * @returns True if context is found, false otherwise
 */
export const hasContextMessage = (messages: ExtendedChatMessage[]): boolean => {
  return messages.some(msg =>
    (msg.role === 'assistant' && msg.content.startsWith('Context Loaded')) ||
    (msg.role === 'system' && msg.content.includes('User context loaded:'))
  );
};

/**
 * Check if conversation contains shell command executions
 * @param messages The conversation messages
 * @returns True if shell commands have been executed
 */
export const hasShellCommandExecutions = (messages: ExtendedChatMessage[]): boolean => {
  return messages.some(msg =>
    (msg.role === 'assistant' && msg.content.includes('SYSTEM: Shell command execution result for AI context:')) ||
    (msg.role === 'system' && msg.isContextUpdate && msg.content.includes('SHELL_COMMAND_EXECUTED:'))
  );
};
