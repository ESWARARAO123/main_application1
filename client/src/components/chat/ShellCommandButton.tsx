import React, { useState, useEffect, useRef } from 'react';
import { PlayIcon, XMarkIcon } from '@heroicons/react/24/solid';
import { CommandLineIcon, CheckCircleIcon, ExclamationTriangleIcon } from '@heroicons/react/24/outline';
import { shellCommandService } from '../../services/shellCommandService';
import ShellCommandResult from './ShellCommandResult';
import { chatbotService } from '../../services/chatbotService';

interface ShellCommandButtonProps {
  onComplete: (result: any, aiResponse?: string) => void;
  toolText?: string;
  messageId?: string;
  conversationId?: string;
  command: string;
}

/**
 * A button component that triggers shell command execution
 * Displays Run/Decline buttons that, when Run is clicked, executes the shell command
 * via the MCP orchestrator and updates the current message with the result
 */
const ShellCommandButton: React.FC<ShellCommandButtonProps> = ({
  onComplete,
  toolText,
  messageId,
  conversationId,
  command
}) => {
  const [isLoading, setIsLoading] = useState(false);
  const [isExpanded, setIsExpanded] = useState(false);
  const [isComplete, setIsComplete] = useState(false);
  const [isDeclined, setIsDeclined] = useState(false);
  const [loadingStage, setLoadingStage] = useState<'initial' | 'executing' | 'processing'>('initial');
  const [executionResult, setExecutionResult] = useState<any>(null);
  const [executionLogs, setExecutionLogs] = useState<string[]>([]);
  
  // Use a ref to track if the component has already been executed
  const hasExecuted = useRef<boolean>(false);

  // Check localStorage on mount to see if this button has already been executed
  useEffect(() => {
    if (conversationId && messageId) {
      const commandKey = `shell_command_${conversationId}_${messageId}`;
      const savedState = localStorage.getItem(commandKey);
      
      if (savedState) {
        try {
          const parsed = JSON.parse(savedState);
          if (parsed.executed && !hasExecuted.current) {
            setIsComplete(true);
            setIsLoading(false);
            setExecutionResult(parsed.result);
            hasExecuted.current = true;
            
            // Restore the result to the parent
            if (parsed.aiResponse) {
              onComplete(parsed.result, parsed.aiResponse);
            }
          } else if (parsed.declined) {
            setIsDeclined(true);
          }
        } catch (error) {
          console.error('Error parsing saved shell command state:', error);
        }
      }
    }
  }, [conversationId, messageId, onComplete]);

  // Prevent any state changes if already executed
  const isAlreadyExecuted = hasExecuted.current || isComplete || isDeclined;

  // Capture console logs during execution
  const captureLog = (message: string) => {
    setExecutionLogs(prev => [...prev, `[${new Date().toISOString()}] ${message}`]);
  };

  const handleRunCommand = async () => {
    if (isLoading || isComplete || isDeclined || hasExecuted.current) return;

    setIsLoading(true);
    setLoadingStage('executing');
    setExecutionLogs([]);

    try {
      captureLog(`Starting execution of command: ${command}`);
      console.log('Executing shell command:', command);

      // Execute the shell command via the service
      const result = await shellCommandService.executeCommand(command);
      setExecutionResult({ ...result, executionLogs });

      captureLog(`Command execution completed. Success: ${result.success}`);

      // Update loading stage
      setLoadingStage('processing');

      // Create a clean AI-readable result for context
      let contextResult: string;
      
      if (result.success) {
        contextResult = `SHELL_COMMAND_EXECUTED: ${command}\n`;
        contextResult += `STATUS: SUCCESS\n`;
        contextResult += `SERVER: ${result.serverConfig?.name} (${result.serverConfig?.host}:${result.serverConfig?.port})\n`;
        contextResult += `TIMESTAMP: ${new Date(result.timestamp).toISOString()}\n`;
        
        if (result.output && result.output.trim()) {
          // Extract just the meaningful text from the JSON output
          const outputText = typeof result.output === 'string' ? result.output : JSON.stringify(result.output);
          
          // Try to parse if it's JSON with a text field
          try {
            const parsed = JSON.parse(outputText);
            if (parsed.text) {
              contextResult += `OUTPUT:\n${parsed.text.trim()}`;
            } else {
              contextResult += `OUTPUT:\n${outputText.trim()}`;
            }
          } catch {
            contextResult += `OUTPUT:\n${outputText.trim()}`;
          }
        } else {
          contextResult += `OUTPUT: (Command completed successfully with no output)`;
        }
      } else {
        contextResult = `SHELL_COMMAND_EXECUTED: ${command}\n`;
        contextResult += `STATUS: FAILED\n`;
        if (result.serverConfig) {
          contextResult += `SERVER: ${result.serverConfig.name} (${result.serverConfig.host}:${result.serverConfig.port})\n`;
        }
        contextResult += `TIMESTAMP: ${new Date(result.timestamp).toISOString()}\n`;
        
        if (result.error) {
          contextResult += `ERROR: ${result.error}\n`;
        }
        
        if (result.stderr && result.stderr.trim()) {
          contextResult += `STDERR: ${result.stderr.trim()}\n`;
        }
        
        if (result.output && result.output.trim()) {
          contextResult += `OUTPUT: ${result.output.trim()}`;
        }
      }

      // Save this command execution to the conversation history as a separate context message for AI
      // This ensures the AI can see the command execution context in future messages
      // The UI display is handled by the ShellCommandResult component
      if (conversationId) {
        try {
          captureLog('Saving command result as context message for AI');
          
          // Save as a separate context message for AI consumption
          // This won't be displayed in the UI but will be available to the AI
          await chatbotService.sendMessage(
            '', // No user message
            conversationId,
            contextResult, // Clean context result for AI
            true // isContextUpdate = true
          );
          
          captureLog('Command result saved as context message for AI');
        } catch (saveError: any) {
          console.error('Failed to save command result as context message:', saveError);
          captureLog(`Failed to save context: ${saveError.message}`);
        }
      } else {
        captureLog(`Cannot save context: conversationId=${conversationId}`);
      }

      // Format user-friendly response for display
      let userResponse: string;
      
      if (result.success) {
        userResponse = `✅ **Command executed successfully**\n\n`;
        userResponse += `**Command:** \`${result.command}\`\n`;
        userResponse += `**Server:** ${result.serverConfig?.name} (${result.serverConfig?.host}:${result.serverConfig?.port})\n`;
        userResponse += `**Time:** ${new Date(result.timestamp).toLocaleString()}\n\n`;
        
        if (result.output && result.output.trim()) {
          try {
            const parsed = JSON.parse(result.output);
            if (parsed.text) {
              userResponse += `**Output:**\n\`\`\`\n${parsed.text.trim()}\n\`\`\``;
            } else {
              userResponse += `**Output:**\n\`\`\`\n${result.output.trim()}\n\`\`\``;
            }
          } catch {
            userResponse += `**Output:**\n\`\`\`\n${result.output.trim()}\n\`\`\``;
          }
        } else {
          userResponse += `The command completed successfully.`;
        }
      } else {
        userResponse = `❌ **Command execution failed**\n\n`;
        userResponse += `**Command:** \`${result.command}\`\n`;
        if (result.serverConfig) {
          userResponse += `**Server:** ${result.serverConfig.name} (${result.serverConfig.host}:${result.serverConfig.port})\n`;
        }
        userResponse += `**Time:** ${new Date(result.timestamp).toLocaleString()}\n\n`;
        
        if (result.error) {
          userResponse += `**Error:** ${result.error}\n\n`;
        }
        
        if (result.stderr && result.stderr.trim()) {
          userResponse += `**Error Details:**\n\`\`\`\n${result.stderr.trim()}\n\`\`\`\n`;
        }
        
        if (result.output && result.output.trim()) {
          userResponse += `**Raw Output:**\n\`\`\`\n${result.output.trim()}\n\`\`\`\n`;
        }
      }

      // Mark as complete
      setIsComplete(true);
      setIsLoading(false);
      hasExecuted.current = true;

      // Save the state to localStorage
      if (conversationId && messageId) {
        const commandKey = `shell_command_${conversationId}_${messageId}`;
        const stateToSave = {
          executed: true,
          isComplete: true,
          isLoading: false,
          result: { ...result, executionLogs },
          aiResponse: userResponse,
          contextResult: contextResult,
          timestamp: new Date().toISOString(),
          messageId: messageId,
          conversationId: conversationId,
          command: command
        };

        localStorage.setItem(commandKey, JSON.stringify(stateToSave));
        console.log('Saved shell command state to storage:', commandKey);
      }

      // Call the completion handler
      onComplete({ ...result, executionLogs }, userResponse);

      // Dispatch an event to refresh the messages
      if (conversationId) {
        window.dispatchEvent(new CustomEvent('refreshMessages', {
          detail: {
            conversationId,
            source: 'shell_command_tool'
          }
        }));
      }

    } catch (error: any) {
      console.error('Error executing shell command:', error);
      captureLog(`Error during execution: ${error.message}`);
      
      const errorResult = {
        success: false,
        command: command,
        error: error.message || 'Unknown error occurred',
        timestamp: new Date().toISOString(),
        executionLogs
      };

      setExecutionResult(errorResult);

      const errorResponse = `❌ **Command execution failed**\n\n**Command:** \`${command}\`\n\n**Error:** ${error.message || 'Unknown error occurred'}`;
      
      setIsComplete(true);
      setIsLoading(false);
      hasExecuted.current = true;

      // Save error state to localStorage
      if (conversationId && messageId) {
        const commandKey = `shell_command_${conversationId}_${messageId}`;
        const stateToSave = {
          executed: true,
          isComplete: true,
          isLoading: false,
          result: errorResult,
          aiResponse: errorResponse,
          timestamp: new Date().toISOString(),
          messageId: messageId,
          conversationId: conversationId,
          command: command,
          error: true
        };

        localStorage.setItem(commandKey, JSON.stringify(stateToSave));
      }

      onComplete(errorResult, errorResponse);
    }
  };

  const handleDeclineCommand = () => {
    if (isLoading || isComplete || isDeclined) return;

    setIsDeclined(true);
    const declineResponse = `Command execution declined: \`${command}\`\n\nIs there something else I can help you with?`;
    
    // Save declined state to localStorage
    if (conversationId && messageId) {
      const commandKey = `shell_command_${conversationId}_${messageId}`;
      const stateToSave = {
        executed: false,
        declined: true,
        isComplete: false,
        isLoading: false,
        timestamp: new Date().toISOString(),
        messageId: messageId,
        conversationId: conversationId,
        command: command
      };

      localStorage.setItem(commandKey, JSON.stringify(stateToSave));
    }

    onComplete(null, declineResponse);
  };

  const getStatusIcon = () => {
    if (isComplete && executionResult?.success) {
      return <CheckCircleIcon className="h-4 w-4 mr-2" style={{ color: 'var(--color-success)' }} />;
    } else if (isComplete && !executionResult?.success) {
      return <ExclamationTriangleIcon className="h-4 w-4 mr-2" style={{ color: 'var(--color-error)' }} />;
    } else if (isDeclined) {
      return <XMarkIcon className="h-4 w-4 mr-2" style={{ color: 'var(--color-text-muted)' }} />;
    } else {
      return <CommandLineIcon className="h-4 w-4 mr-2" style={{ color: 'var(--color-primary)' }} />;
    }
  };

  const getStatusText = () => {
    if (isComplete && executionResult?.success) {
      return 'Command Executed Successfully';
    } else if (isComplete && !executionResult?.success) {
      return 'Command Execution Failed';
    } else if (isDeclined) {
      return 'Command Declined';
    } else if (isLoading) {
      return loadingStage === 'executing' ? 'Executing Command...' : 'Processing Result...';
    } else {
      return 'Shell Command';
    }
  };

  const getStatusColor = () => {
    if (isComplete && executionResult?.success) {
      return 'var(--color-success)';
    } else if (isComplete && !executionResult?.success) {
      return 'var(--color-error)';
    } else if (isDeclined) {
      return 'var(--color-text-muted)';
    } else {
      return 'var(--color-primary)';
    }
  };

  return (
    <div className="shell-command-button-container my-3">
      <div className="flex flex-col rounded-md overflow-hidden" style={{
        backgroundColor: 'var(--color-surface-accent)',
        border: '1px solid var(--color-border-accent)'
      }}>
        {/* Tool header with command icon */}
        <div className="flex items-center p-2 border-b border-opacity-20" style={{
          borderColor: 'var(--color-border-accent)',
          backgroundColor: isComplete
            ? (executionResult?.success ? 'rgba(var(--color-success-rgb), 0.1)' : 'rgba(var(--color-error-rgb), 0.1)')
            : isDeclined
            ? 'rgba(var(--color-text-muted-rgb), 0.1)'
            : 'rgba(var(--color-primary-rgb), 0.1)'
        }}>
          {getStatusIcon()}
          <span className="text-xs font-medium" style={{ color: getStatusColor() }}>
            {getStatusText()}
          </span>
          {toolText && !isComplete && !isDeclined && (
            <button
              onClick={() => setIsExpanded(!isExpanded)}
              className="ml-auto text-xs underline"
              style={{ color: 'var(--color-text-muted)' }}
            >
              {isExpanded ? 'Hide code' : 'Show code'}
            </button>
          )}
        </div>

        {/* Tool code (optional) */}
        {isExpanded && toolText && !isComplete && !isDeclined && (
          <div className="p-2 text-xs font-mono overflow-x-auto" style={{
            backgroundColor: 'rgba(0,0,0,0.1)',
            color: 'var(--color-text-muted)',
            maxHeight: '100px'
          }}>
            <pre>{toolText}</pre>
          </div>
        )}

        {/* Command display */}
        <div className="p-3">
          <div className="text-sm mb-2" style={{ color: 'var(--color-text)' }}>
            <strong>Command:</strong> <code className="px-1 py-0.5 rounded text-xs" style={{
              backgroundColor: 'rgba(0,0,0,0.1)',
              color: 'var(--color-text)'
            }}>{command}</code>
          </div>

          {/* Action buttons - only show if not completed, declined, or loading */}
          {!isAlreadyExecuted && !isLoading && (
            <div className="flex gap-2 mt-3">
              <button
                onClick={handleRunCommand}
                className="flex items-center px-3 py-1 text-xs font-medium rounded transition-colors"
                style={{
                  backgroundColor: 'var(--color-success)',
                  color: 'white'
                }}
                disabled={isLoading || isAlreadyExecuted}
              >
                <PlayIcon className="h-3 w-3 mr-1" />
                Run
              </button>
              <button
                onClick={handleDeclineCommand}
                className="flex items-center px-3 py-1 text-xs font-medium rounded transition-colors"
                style={{
                  backgroundColor: 'var(--color-text-muted)',
                  color: 'white'
                }}
                disabled={isLoading || isAlreadyExecuted}
              >
                <XMarkIcon className="h-3 w-3 mr-1" />
                Decline
              </button>
            </div>
          )}

          {/* Loading indicator */}
          {isLoading && (
            <div className="flex items-center mt-3 text-xs" style={{ color: 'var(--color-text-muted)' }}>
              <div className="animate-spin rounded-full h-3 w-3 border-b-2 mr-2" style={{
                borderColor: 'var(--color-primary)'
              }}></div>
              {loadingStage === 'executing' ? 'Executing command via MCP orchestrator...' : 'Processing result...'}
            </div>
          )}

          {/* Completion status */}
          {(isComplete || isDeclined) && (
            <div className="mt-3 text-xs" style={{
              color: isComplete
                ? (executionResult?.success ? 'var(--color-success)' : 'var(--color-error)')
                : 'var(--color-text-muted)'
            }}>
              {isComplete
                ? (executionResult?.success
                  ? '✓ Command executed successfully'
                  : '✗ Command execution failed')
                : '⚬ Command execution declined'}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default ShellCommandButton; 