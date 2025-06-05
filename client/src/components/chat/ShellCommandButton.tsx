import React, { useState, useEffect, useRef, useCallback } from 'react';
import { PlayIcon, XMarkIcon, ClipboardDocumentIcon, CheckIcon } from '@heroicons/react/24/solid';
import { CommandLineIcon, CheckCircleIcon, ExclamationTriangleIcon, ChevronDownIcon, ChevronUpIcon } from '@heroicons/react/24/outline';
import { shellCommandService } from '../../services/shellCommandService';
import ShellCommandResult from './ShellCommandResult';
import { chatbotService } from '../../services/chatbotService';
import { useTheme } from '../../contexts/ThemeContext';

interface ShellCommandButtonProps {
  onComplete: (result: any, aiResponse?: string) => void;
  toolText?: string;
  messageId?: string;
  conversationId?: string;
  command: string;
}

type ExecutionState = 'pending' | 'executing' | 'completed' | 'declined' | 'error';
type LoadingStage = 'preparing' | 'executing' | 'processing' | 'saving';

/**
 * Professional shell command execution tool
 * Features enterprise-grade UI with comprehensive execution tracking
 */
const ShellCommandButton: React.FC<ShellCommandButtonProps> = ({
  onComplete,
  toolText,
  messageId,
  conversationId,
  command
}) => {
  const [executionState, setExecutionState] = useState<ExecutionState>('pending');
  const [loadingStage, setLoadingStage] = useState<LoadingStage>('preparing');
  const [isExpanded, setIsExpanded] = useState(false);
  const [executionResult, setExecutionResult] = useState<any>(null);
  const [executionLogs, setExecutionLogs] = useState<string[]>([]);
  const [progress, setProgress] = useState(0);
  const [copiedCommand, setCopiedCommand] = useState(false);
  const [copiedResult, setCopiedResult] = useState(false);
  
  // Use refs to prevent race conditions
  const isProcessingRef = useRef<boolean>(false);
  const lastClickTimeRef = useRef<number>(0);
  
  // Theme context
  const { currentTheme } = useTheme();
  const isDarkTheme = currentTheme !== 'light';
  const isMidnightTheme = currentTheme === 'midnight';
  
  // Generate unique storage key
  const storageKey = `shell_command_${conversationId}_${messageId}`;

  // Debounce function to prevent rapid clicks
  const debounceDelay = 1000; // 1 second

  // Copy to clipboard functionality
  const copyToClipboard = useCallback(async (text: string, type: 'command' | 'result') => {
    try {
      await navigator.clipboard.writeText(text);
      if (type === 'command') {
        setCopiedCommand(true);
        setTimeout(() => setCopiedCommand(false), 2000);
      } else {
        setCopiedResult(true);
        setTimeout(() => setCopiedResult(false), 2000);
      }
    } catch (error) {
      console.error('Failed to copy to clipboard:', error);
    }
  }, []);

  // Check localStorage on mount to see if this button has already been executed
  useEffect(() => {
    if (conversationId && messageId) {
      try {
        // Try sessionStorage first (faster) then fall back to localStorage
        let storedState = sessionStorage.getItem(storageKey);
        if (!storedState) {
          storedState = localStorage.getItem(storageKey);
        }
        
        if (storedState) {
          const parsed = JSON.parse(storedState);
          
          if (parsed.executed && parsed.result) {
            setExecutionState('completed');
            setExecutionResult(parsed.result);
            setProgress(100);
            
            // Restore the result to the parent without triggering another save
            if (parsed.aiResponse && onComplete) {
              setTimeout(() => onComplete(parsed.result, parsed.aiResponse), 100);
            }
          } else if (parsed.declined) {
            setExecutionState('declined');
          }
        }
      } catch (error) {
        console.error('Error parsing saved shell command state:', error);
        // Clear corrupted data
        localStorage.removeItem(storageKey);
        sessionStorage.removeItem(storageKey);
      }
    }
  }, [conversationId, messageId, storageKey, onComplete]);

  // Capture console logs during execution
  const captureLog = useCallback((message: string) => {
    const logEntry = `[${new Date().toISOString()}] ${message}`;
    setExecutionLogs(prev => [...prev, logEntry]);
    console.log(`ShellCommand: ${logEntry}`);
  }, []);

  // Save state to storage (both localStorage and sessionStorage)
  const saveStateToStorage = useCallback((state: any) => {
    if (!conversationId || !messageId) return;
    
    try {
      const stateToSave = {
        ...state,
        timestamp: new Date().toISOString(),
        messageId: messageId,
        conversationId: conversationId,
        command: command,
        storageVersion: '2.0' // Version for future compatibility
      };

      const serializedState = JSON.stringify(stateToSave);
      localStorage.setItem(storageKey, serializedState);
      sessionStorage.setItem(storageKey, serializedState);
      
      captureLog(`State saved to storage: ${storageKey}`);
    } catch (error) {
      console.error('Error saving shell command state to storage:', error);
    }
  }, [conversationId, messageId, storageKey, command, captureLog]);

  const handleRunCommand = useCallback(async () => {
    // Prevent rapid clicks and multiple executions
    const now = Date.now();
    if (now - lastClickTimeRef.current < debounceDelay) {
      captureLog('Click ignored due to debounce protection');
      return;
    }
    lastClickTimeRef.current = now;

    // Prevent multiple simultaneous executions
    if (isProcessingRef.current || executionState !== 'pending') {
      captureLog('Execution already in progress or completed');
      return;
    }

    isProcessingRef.current = true;
    setExecutionState('executing');
    setLoadingStage('preparing');
    setProgress(10);
    setExecutionLogs([]);

    try {
      captureLog(`Starting execution of command: ${command}`);
      
      setLoadingStage('executing');
      setProgress(30);

      // Execute the shell command via the service
      const result = await shellCommandService.executeCommand(command);
      
      setProgress(60);
      setLoadingStage('processing');
      setExecutionResult(result);

      captureLog(`Command execution completed. Success: ${result.success}`);

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

      setProgress(80);
      setLoadingStage('saving');

      // Save this command execution to the conversation history as a separate context message for AI
      if (conversationId) {
        try {
          captureLog('Saving command result as context message for AI');
          
          // Save as a separate context message for AI consumption
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
      }

      // Format user-friendly response for display
      let userResponse: string;
      
      if (result.success) {
        userResponse = `**Command executed successfully**\n\n`;
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
        userResponse = `**Command execution failed**\n\n`;
        userResponse += `**Command:** \`${result.command}\`\n`;
        if (result.serverConfig) {
          userResponse += `**Server:** ${result.serverConfig.name} (${result.serverConfig.host}:${result.serverConfig.port})\n`;
        }
        userResponse += `**Time:** ${new Date(result.timestamp).toLocaleString()}\n\n`;
        
        if (result.error) {
          userResponse += `**Error:** ${result.error}\n\n`;
        }
        
        if (result.stderr && result.stderr.trim()) {
          userResponse += `**Error Output:**\n\`\`\`\n${result.stderr.trim()}\n\`\`\`\n\n`;
        }
        
        if (result.output && result.output.trim()) {
          userResponse += `**Output:**\n\`\`\`\n${result.output.trim()}\n\`\`\``;
        }
      }

      setProgress(95);

      // Save execution state to storage
      saveStateToStorage({
          executed: true,
          isComplete: true,
          isLoading: false,
        result: result,
        aiResponse: userResponse
      });

      setProgress(100);
      setExecutionState('completed');
      
      captureLog('Command execution completed successfully');
      
      // Notify parent component
      onComplete(result, userResponse);

    } catch (error: any) {
      console.error('Shell command execution failed:', error);
      captureLog(`Command execution failed: ${error.message}`);
      
      const errorResult = {
        success: false,
        command: command,
        error: error.message || 'Unknown error occurred',
        timestamp: Date.now(),
        executionLogs: executionLogs
      };

      const errorResponse = `**Command execution failed**\n\n**Command:** \`${command}\`\n**Error:** ${error.message || 'Unknown error occurred'}`;

      setExecutionResult(errorResult);
      setExecutionState('error');

      // Save error state to localStorage
      saveStateToStorage({
          executed: true,
          isComplete: true,
          isLoading: false,
          result: errorResult,
          aiResponse: errorResponse,
          error: true
      });

      onComplete(errorResult, errorResponse);
    } finally {
      isProcessingRef.current = false;
    }
  }, [command, conversationId, executionLogs, onComplete, saveStateToStorage, captureLog, executionState]);

  const handleDeclineCommand = useCallback(() => {
    if (isProcessingRef.current || executionState !== 'pending') return;

    setExecutionState('declined');
    const declineResponse = `Command execution declined: \`${command}\`\n\nIs there something else I can help you with?`;
    
    // Save declined state to localStorage
    saveStateToStorage({
        executed: false,
        declined: true,
        isComplete: false,
      isLoading: false
    });

    onComplete(null, declineResponse);
  }, [command, executionState, saveStateToStorage, onComplete]);

  const getStatusIcon = () => {
    switch (executionState) {
      case 'completed':
        return executionResult?.success 
          ? <CheckCircleIcon className="h-5 w-5" style={{ color: 'var(--color-success)' }} />
          : <ExclamationTriangleIcon className="h-5 w-5" style={{ color: 'var(--color-error)' }} />;
      case 'error':
        return <ExclamationTriangleIcon className="h-5 w-5" style={{ color: 'var(--color-error)' }} />;
      case 'declined':
        return <XMarkIcon className="h-5 w-5" style={{ color: 'var(--color-text-muted)' }} />;
      case 'executing':
        return (
          <div className="animate-spin rounded-full h-5 w-5 border-2 border-transparent border-t-current" style={{
            color: 'var(--color-primary)'
          }}></div>
        );
      default:
        return null; // No icon for pending state
    }
  };

  const getStatusText = () => {
    switch (executionState) {
      case 'completed':
        return executionResult?.success ? 'Execution Completed' : 'Execution Failed';
      case 'error':
        return 'Execution Failed';
      case 'declined':
      return 'Command Declined';
      case 'executing':
        switch (loadingStage) {
          case 'preparing': return 'Initializing...';
          case 'executing': return 'Processing Command...';
          case 'processing': return 'Analyzing Results...';
          case 'saving': return 'Finalizing...';
          default: return 'In Progress...';
        }
      default:
        return 'Command Execution Tool';
    }
  };

  const getStatusColor = () => {
    switch (executionState) {
      case 'completed':
        return executionResult?.success ? 'var(--color-success)' : 'var(--color-error)';
      case 'error':
      return 'var(--color-error)';
      case 'declined':
      return 'var(--color-text-muted)';
      case 'executing':
        return 'var(--color-primary)';
      default:
      return 'var(--color-primary)';
    }
  };

  const getContainerBackground = () => {
    if (isMidnightTheme) {
      return 'linear-gradient(135deg, rgba(15, 23, 42, 0.98) 0%, rgba(30, 41, 59, 0.95) 100%)';
    } else if (isDarkTheme) {
      return 'linear-gradient(135deg, rgba(30, 41, 59, 0.98) 0%, rgba(51, 65, 85, 0.95) 100%)';
    } else {
      return 'linear-gradient(135deg, rgba(255, 255, 255, 0.98) 0%, rgba(248, 250, 252, 0.95) 100%)';
    }
  };

  const getHeaderBackground = () => {
    switch (executionState) {
      case 'completed':
        return executionResult?.success 
          ? 'linear-gradient(90deg, rgba(34, 197, 94, 0.08), rgba(34, 197, 94, 0.03))'
          : 'linear-gradient(90deg, rgba(239, 68, 68, 0.08), rgba(239, 68, 68, 0.03))';
      case 'error':
        return 'linear-gradient(90deg, rgba(239, 68, 68, 0.08), rgba(239, 68, 68, 0.03))';
      case 'declined':
        return 'linear-gradient(90deg, rgba(107, 114, 128, 0.08), rgba(107, 114, 128, 0.03))';
      case 'executing':
        return 'linear-gradient(90deg, rgba(59, 130, 246, 0.08), rgba(59, 130, 246, 0.03))';
      default:
        return 'linear-gradient(90deg, rgba(99, 102, 241, 0.08), rgba(99, 102, 241, 0.03))';
    }
  };

  const isButtonDisabled = executionState !== 'pending' || isProcessingRef.current;

  return (
    <div className="command-execution-tool my-4">
      <div 
        className="rounded-xl overflow-hidden shadow-xl backdrop-blur-sm border transition-all duration-300 hover:shadow-2xl"
        style={{
          background: getContainerBackground(),
          border: `1px solid ${isDarkTheme ? 'rgba(71, 85, 105, 0.4)' : 'rgba(203, 213, 225, 0.6)'}`,
          boxShadow: isDarkTheme 
            ? '0 10px 25px -5px rgba(0, 0, 0, 0.4), 0 4px 6px -2px rgba(0, 0, 0, 0.2)' 
            : '0 10px 25px -5px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05)'
        }}
      >
        {/* Professional Header */}
        <div 
          className="flex items-center justify-between p-4 border-b"
          style={{
            background: getHeaderBackground(),
            borderColor: isDarkTheme ? 'rgba(71, 85, 105, 0.4)' : 'rgba(203, 213, 225, 0.4)'
          }}
        >
          <div className="flex items-center space-x-3">
            {getStatusIcon()}
            <div>
              <div className="font-semibold text-sm" style={{ color: getStatusColor() }}>
                {getStatusText()}
              </div>
              <div className="text-xs font-medium" style={{ color: 'var(--color-text-muted)' }}>
                Shell Command Interface
              </div>
            </div>
          </div>
          
          <div className="flex items-center space-x-3">
            {executionState === 'executing' && (
              <div className="flex items-center space-x-3">
                <div className="text-sm font-semibold" style={{ color: 'var(--color-text)' }}>
                  {progress}%
                </div>
                <div className="w-20 h-2 bg-gray-200 rounded-full overflow-hidden">
                  <div 
                    className="h-full bg-gradient-to-r from-indigo-500 to-purple-600 rounded-full transition-all duration-300 ease-out"
                    style={{ width: `${progress}%` }}
                  ></div>
                </div>
              </div>
            )}
            
            {toolText && executionState === 'pending' && (
            <button
              onClick={() => setIsExpanded(!isExpanded)}
                className="flex items-center px-3 py-1.5 text-xs font-medium rounded-lg transition-all duration-200 hover:bg-black hover:bg-opacity-10 border"
                style={{ 
                  color: 'var(--color-text-muted)',
                  borderColor: 'var(--color-border)'
                }}
              >
                {isExpanded ? <ChevronUpIcon className="h-4 w-4 mr-1" /> : <ChevronDownIcon className="h-4 w-4 mr-1" />}
                {isExpanded ? 'Hide' : 'View'} Source
            </button>
          )}
          </div>
        </div>

        {/* Enhanced Progress Bar */}
        {executionState === 'executing' && (
          <div className="w-full h-1 bg-gray-200 overflow-hidden">
            <div 
              className="h-full bg-gradient-to-r from-indigo-500 via-purple-500 to-blue-500 transition-all duration-500 ease-out"
              style={{ width: `${progress}%` }}
            ></div>
          </div>
        )}

        {/* Source Code Section */}
        {isExpanded && toolText && executionState === 'pending' && (
          <div 
            className="p-4 border-b text-sm font-mono"
            style={{
              backgroundColor: isDarkTheme ? 'rgba(0, 0, 0, 0.3)' : 'rgba(0, 0, 0, 0.03)',
              borderColor: isDarkTheme ? 'rgba(71, 85, 105, 0.4)' : 'rgba(203, 213, 225, 0.4)',
            color: 'var(--color-text-muted)',
              maxHeight: '150px',
              overflowY: 'auto'
            }}
          >
            <div className="mb-2 text-xs font-semibold uppercase tracking-wider" style={{ color: 'var(--color-text)' }}>
              Tool Implementation
            </div>
            <pre className="whitespace-pre-wrap">{toolText}</pre>
          </div>
        )}

        {/* Main Content */}
        <div className="p-5">
          {/* Command Display */}
          <div className="mb-5">
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center space-x-2">
                <CommandLineIcon className="h-4 w-4" style={{ color: 'var(--color-text-muted)' }} />
                <span className="text-sm font-semibold" style={{ color: 'var(--color-text)' }}>
                  Command
                </span>
              </div>
              <button
                onClick={() => copyToClipboard(command, 'command')}
                className="flex items-center px-3 py-1.5 text-xs font-medium rounded-lg transition-all duration-200 hover:bg-black hover:bg-opacity-10 border"
                style={{ 
                  color: 'var(--color-primary)',
                  borderColor: 'var(--color-border)'
                }}
              >
                {copiedCommand ? (
                  <>
                    <CheckIcon className="h-3 w-3 mr-1.5" />
                    Copied
                  </>
                ) : (
                  <>
                    <ClipboardDocumentIcon className="h-3 w-3 mr-1.5" />
                    Copy
                  </>
                )}
              </button>
            </div>
            <div 
              className="p-4 rounded-lg font-mono text-sm border"
              style={{
                backgroundColor: isDarkTheme ? 'rgba(0, 0, 0, 0.4)' : 'rgba(0, 0, 0, 0.03)',
                borderColor: isDarkTheme ? 'rgba(71, 85, 105, 0.4)' : 'rgba(203, 213, 225, 0.6)',
              color: 'var(--color-text)'
              }}
            >
              {command}
            </div>
          </div>

          {/* Professional Action Buttons */}
          {executionState === 'pending' && (
            <div className="flex space-x-3">
              <button
                onClick={handleRunCommand}
                disabled={isButtonDisabled}
                className="flex-1 flex items-center justify-center px-6 py-3 text-sm font-semibold rounded-lg transition-all duration-300 transform hover:scale-105 hover:shadow-lg disabled:transform-none disabled:hover:scale-100 disabled:cursor-not-allowed"
                style={{
                  background: isButtonDisabled 
                    ? 'linear-gradient(135deg, #9ca3af, #6b7280)' 
                    : 'linear-gradient(135deg, #059669, #047857)',
                  color: 'white',
                  opacity: isButtonDisabled ? 0.6 : 1,
                  boxShadow: isButtonDisabled 
                    ? 'none' 
                    : '0 4px 14px 0 rgba(5, 150, 105, 0.4)'
                }}
              >
                <PlayIcon className="h-4 w-4 mr-2" />
                Execute Command
              </button>
              
              <button
                onClick={handleDeclineCommand}
                disabled={isButtonDisabled}
                className="flex-1 flex items-center justify-center px-6 py-3 text-sm font-semibold rounded-lg transition-all duration-300 transform hover:scale-105 hover:shadow-lg disabled:transform-none disabled:hover:scale-100 disabled:cursor-not-allowed"
                style={{
                  background: isButtonDisabled 
                    ? 'linear-gradient(135deg, #9ca3af, #6b7280)' 
                    : 'linear-gradient(135deg, #dc2626, #b91c1c)',
                  color: 'white',
                  opacity: isButtonDisabled ? 0.6 : 1,
                  boxShadow: isButtonDisabled 
                    ? 'none' 
                    : '0 4px 14px 0 rgba(220, 38, 38, 0.4)'
                }}
              >
                <XMarkIcon className="h-4 w-4 mr-2" />
                Cancel
              </button>
            </div>
          )}

          {/* Professional Loading Indicator */}
          {executionState === 'executing' && (
            <div className="mt-5">
              <div className="flex items-center mb-4">
                <div className="animate-spin rounded-full h-6 w-6 border-2 border-transparent border-t-current mr-3" style={{
                  color: 'var(--color-primary)'
              }}></div>
                <div className="flex-1">
                  <div className="text-sm font-medium" style={{ color: 'var(--color-text)' }}>
                    {getStatusText()}
                  </div>
                  <div className="text-xs" style={{ color: 'var(--color-text-muted)' }}>
                    Processing on remote server...
                  </div>
                </div>
                <div className="text-sm font-bold" style={{ color: 'var(--color-primary)' }}>
                  {progress}%
                </div>
              </div>
              
              {executionLogs.length > 0 && (
                <div 
                  className="rounded-lg p-3 text-xs font-mono max-h-24 overflow-y-auto border"
                  style={{
                    backgroundColor: isDarkTheme ? 'rgba(0, 0, 0, 0.5)' : 'rgba(243, 244, 246, 0.8)',
                    borderColor: isDarkTheme ? 'rgba(71, 85, 105, 0.4)' : 'rgba(203, 213, 225, 0.6)'
                  }}
                >
                  <div className="mb-2 text-xs font-semibold uppercase tracking-wider" style={{ color: 'var(--color-text)' }}>
                    Execution Log
                  </div>
                  {executionLogs.slice(-3).map((log, index) => (
                    <div key={index} className="mb-1" style={{ color: 'var(--color-text-muted)' }}>
                      {log}
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}

          {/* Professional Status Display */}
          {(executionState === 'completed' || executionState === 'error' || executionState === 'declined') && (
            <div className="mt-5">
              <div 
                className="p-4 rounded-lg border"
                style={{
                  backgroundColor: (() => {
                    switch (executionState) {
                      case 'completed':
                        return executionResult?.success 
                          ? isDarkTheme ? 'rgba(34, 197, 94, 0.1)' : 'rgba(34, 197, 94, 0.05)'
                          : isDarkTheme ? 'rgba(239, 68, 68, 0.1)' : 'rgba(239, 68, 68, 0.05)';
                      case 'error':
                        return isDarkTheme ? 'rgba(239, 68, 68, 0.1)' : 'rgba(239, 68, 68, 0.05)';
                      case 'declined':
                        return isDarkTheme ? 'rgba(107, 114, 128, 0.1)' : 'rgba(107, 114, 128, 0.05)';
                      default:
                        return 'transparent';
                    }
                  })(),
                  borderColor: (() => {
                    switch (executionState) {
                      case 'completed':
                        return executionResult?.success ? 'var(--color-success)' : 'var(--color-error)';
                      case 'error':
                        return 'var(--color-error)';
                      case 'declined':
                        return 'var(--color-text-muted)';
                      default:
                        return 'transparent';
                    }
                  })()
                }}
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-3">
                    {getStatusIcon()}
                    <div>
                      <div className="text-sm font-semibold" style={{ color: getStatusColor() }}>
                        {(() => {
                          switch (executionState) {
                            case 'completed':
                              return executionResult?.success
                                ? 'Command executed successfully'
                                : 'Command execution failed';
                            case 'error':
                              return 'Command execution failed';
                            case 'declined':
                              return 'Command execution cancelled';
                            default:
                              return '';
                          }
                        })()}
                      </div>
                      <div className="text-xs" style={{ color: 'var(--color-text-muted)' }}>
                        {new Date().toLocaleString()}
                      </div>
                    </div>
                  </div>
                  
                  {executionResult && (
                    <button
                      onClick={() => copyToClipboard(JSON.stringify(executionResult, null, 2), 'result')}
                      className="flex items-center px-3 py-1.5 text-xs font-medium rounded-lg transition-all duration-200 hover:bg-black hover:bg-opacity-10 border"
                      style={{ 
                        color: 'var(--color-primary)',
                        borderColor: 'var(--color-border)'
                      }}
                    >
                      {copiedResult ? (
                        <>
                          <CheckIcon className="h-3 w-3 mr-1.5" />
                          Copied
                        </>
                      ) : (
                        <>
                          <ClipboardDocumentIcon className="h-3 w-3 mr-1.5" />
                          Copy Result
                        </>
                      )}
                    </button>
                  )}
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default ShellCommandButton;