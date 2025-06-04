import React, { useState, useEffect, useRef } from 'react';
import { ChatMessage as ChatMessageType, ExtendedChatMessage, PredictionResult } from '../../types';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism';
import { oneLight } from 'react-syntax-highlighter/dist/esm/styles/prism';
import {
  ClipboardDocumentIcon,
  CheckIcon,
  DocumentTextIcon,
  DocumentIcon,
  ArrowDownTrayIcon,
  InformationCircleIcon,
  ChevronDownIcon,
  ChevronUpIcon,
  TableCellsIcon
} from '@heroicons/react/24/outline';
import { messageBubbleStyles, markdownStyles } from './chatStyles';
import { useTheme } from '../../contexts/ThemeContext';
import { RagSource } from '../../services/ragChatService';
import { containsReadContextToolCall, extractToolCall, containsShellCommandToolCall, extractShellCommand } from '../../utils/toolParser';
import ContextReadingButton from './ContextReadingButton';
import ShellCommandResult from './ShellCommandResult';
import UserIcon from '../UserIcon';

interface ChatMessageProps {
  message: ExtendedChatMessage;
  isAI?: boolean;
  conversationId?: string;
}

const ChatMessage: React.FC<ChatMessageProps> = ({ message, isAI = false, conversationId }) => {
  const [copiedCode, setCopiedCode] = useState<string | null>(null);
  const [showSources, setShowSources] = useState<boolean>(false);
  const [contextResult, setContextResult] = useState<any>(null);
  // Add a state variable to force re-renders when needed
  const [lastUpdated, setLastUpdated] = useState<number>(Date.now());
  const { currentTheme } = useTheme();
  const isDarkTheme = currentTheme !== 'light';

  // Use a ref to track if the context has already been processed
  const contextProcessedRef = useRef<boolean>(false);

  // Extract tool text if present
  const extractToolText = (): string | undefined => {
    if (!isAI || !message.content) return undefined;

    // Look for TOOL: marker and extract the tool call text
    const toolMatch = message.content.match(/TOOL:\s*(\{[\s\S]*?\})/);
    if (toolMatch && toolMatch[0]) {
      return toolMatch[0];
    }

    // Alternative pattern matching for read_context
    if (message.content.includes('read_context')) {
      const lines = message.content.split('\n');
      const toolLines = lines.filter(line =>
        line.includes('TOOL:') ||
        line.includes('read_context') ||
        line.includes('{') ||
        line.includes('}')
      );

      if (toolLines.length > 0) {
        return toolLines.join('\n');
      }
    }

    return undefined;
  };



  // Check if the message contains a read_context tool call or has isContextTool flag
  const hasReadContextTool = isAI && (containsReadContextToolCall(message.content) || message.isContextTool);
  const toolText = hasReadContextTool ? extractToolText() : undefined;

  // Check if the message contains a shell command tool call
  const hasShellCommandTool = isAI && containsShellCommandToolCall(message.content);
  const shellCommand = hasShellCommandTool ? extractShellCommand(message.content) : undefined;

  // Determine if this is a Chat2SQL message
  const isChat2SqlMessage = message.chat2sql || message.isSqlResult || message.isSqlQuery;
  
  // Debug logging for Chat2SQL messages
  if (isChat2SqlMessage) {
    console.log('Chat2SQL message detected:', {
      messageId: message.id,
      role: message.role,
      isAI,
      chat2sql: message.chat2sql,
      isSqlResult: message.isSqlResult,
      isSqlQuery: message.isSqlQuery,
      content: message.content?.substring(0, 50) + '...'
    });
  }



  // Check if the message contains phrases that should trigger the context tool
  // Be more selective to avoid false positives
  const shouldTriggerContextTool = isAI && !hasReadContextTool && (
    message.content.includes('I can read your context rules') || 
    message.content.includes('I can check your context preferences') || 
    message.content.match(/use the context tool/i) !== null ||
    message.content.match(/I should (use|run) the context tool/i) !== null ||
    message.content.match(/let me (use|run) the context tool/i) !== null
  );

  // State for AI response to context
  const [aiContextResponse, setAiContextResponse] = useState<string | null>(null);
  const [showContextTool, setShowContextTool] = useState<boolean>(shouldTriggerContextTool);

  // State for shell command results
  const [shellCommandResult, setShellCommandResult] = useState<any>(null);



  // If we detect a phrase that should trigger the context tool, show the button
  useEffect(() => {
    if (shouldTriggerContextTool && !contextProcessedRef.current) {
      setShowContextTool(true);
    }
  }, [shouldTriggerContextTool]);

  // Check if this message is a context result message
  const isContextResultMessage = isAI && message.content.startsWith('Context Loaded');

  // Check storage for context button state
  useEffect(() => {
    if (hasReadContextTool && conversationId) {
      try {
        // Check if we have already executed this context tool
        const contextKey = `context_button_${conversationId}_${message.id}`;

        // Try sessionStorage first (faster) then fall back to localStorage
        let storedState = sessionStorage.getItem(contextKey);
        if (!storedState) {
          storedState = localStorage.getItem(contextKey);
        }

        if (storedState) {
          const parsedState = JSON.parse(storedState);
          if (parsedState.executed) {
            console.log('Context tool was previously executed, restoring state');

            // Restore the context result from storage
            setContextResult(parsedState.result);
            if (parsedState.aiResponse) {
              setAiContextResponse(parsedState.aiResponse);
            }

            // Re-save to both storage types to ensure consistency
            try {
              localStorage.setItem(contextKey, storedState);
              sessionStorage.setItem(contextKey, storedState);
            } catch (storageError) {
              console.error('Error re-saving context button state to storage:', storageError);
            }

            // Force a re-render to ensure the button shows as completed
            setLastUpdated(Date.now());

            // Add a small delay and force another re-render to ensure the button state is updated
            setTimeout(() => {
              setLastUpdated(Date.now());

              // Find and update any context buttons in the DOM
              const contextButtons = document.querySelectorAll(`[data-message-id="${message.id}"][data-context-button-state]`);
              contextButtons.forEach(button => {
                button.setAttribute('data-context-button-state', 'complete');
                if (button instanceof HTMLElement) {
                  button.style.backgroundColor = 'var(--color-success)';
                  button.style.cursor = 'default';
                }
              });
            }, 200);
          }
        }
      } catch (error) {
        console.error('Error checking storage for context button state:', error);
      }
    }
  }, [hasReadContextTool, conversationId, message.id]);

  // Check storage for shell command results
  useEffect(() => {
    if (hasShellCommandTool && conversationId) {
      try {
        const commandKey = `shell_command_${conversationId}_${message.id}`;
        
        // Try sessionStorage first (faster) then fall back to localStorage
        let storedState = sessionStorage.getItem(commandKey);
        if (!storedState) {
          storedState = localStorage.getItem(commandKey);
        }

        if (storedState) {
          const parsedState = JSON.parse(storedState);
          if (parsedState.executed && parsedState.result) {
            console.log('Shell command was previously executed, restoring state');
            setShellCommandResult(parsedState.result);
            
            // Re-save to both storage types to ensure consistency
            try {
              localStorage.setItem(commandKey, storedState);
              sessionStorage.setItem(commandKey, storedState);
            } catch (storageError) {
              console.error('Error re-saving shell command state to storage:', storageError);
            }
          }
        }
      } catch (error) {
        console.error('Error checking storage for shell command state:', error);
      }
    }
  }, [hasShellCommandTool, conversationId, message.id]);

  // Handle context reading completion
  const handleContextReadComplete = async (result: any, aiResponse?: string) => {
    // Mark context as processed to prevent multiple executions
    contextProcessedRef.current = true;
    
    setContextResult(result);
    if (aiResponse) {
      setAiContextResponse(aiResponse);

      // Save the button state to both localStorage and sessionStorage
      if (conversationId && message.id) {
        try {
          const contextKey = `context_button_${conversationId}_${message.id}`;
          const contextRulesKey = `context_rules_${conversationId}`;
          
          // Extract the context rules if available
          let contextRules = '';
          if (result && result.result && result.result.user_context) {
            contextRules = result.result.user_context;
            
            // Save the actual context rules separately for easier access
            localStorage.setItem(contextRulesKey, JSON.stringify({
              rules: contextRules,
              timestamp: new Date().toISOString(),
              hasContext: true
            }));
            
            // Also save to session storage for quicker access
            sessionStorage.setItem(contextRulesKey, JSON.stringify({
              rules: contextRules,
              timestamp: new Date().toISOString(),
              hasContext: true
            }));
            
            console.log('Saved context rules to storage:', contextRules);
          }
          
          const stateToSave = {
            executed: true,
            isComplete: true,
            isLoading: false,
            result: result,
            aiResponse: aiResponse,
            timestamp: new Date().toISOString(),
            messageId: message.id,
            conversationId: conversationId,
            contextRules: contextRules
          };

          // Save to both storage types for redundancy - only once
          localStorage.setItem(contextKey, JSON.stringify(stateToSave));
          sessionStorage.setItem(contextKey, JSON.stringify(stateToSave));

          // Force a re-render to ensure the button shows as completed
          setLastUpdated(Date.now());

          console.log('Saved context button state to storage:', contextKey);
          
          // Add a system message with the context rules to ensure it's included in future prompts
          const systemContextMessage: ExtendedChatMessage = {
            id: `system-context-${Date.now()}`,
            role: 'system',
            content: `User context loaded: ${contextRules}`,
            timestamp: new Date(),
            isContextMessage: true
          };
          
          // Dispatch a custom event to add this system message to the conversation
          const event = new CustomEvent('addSystemMessage', { 
            detail: { message: systemContextMessage }
          });
          window.dispatchEvent(event);
        } catch (storageError) {
          console.error('Error saving context button state to storage:', storageError);
        }
      }

      // We don't need to trigger a refresh of messages anymore
      // The context is updated in-place in the current message
      // This prevents empty messages and duplicated context tools
      console.log('Context tool execution completed for message:', message.id);
    }
  };

  // Handle shell command completion
  const handleShellCommandComplete = async (result: any) => {
    setShellCommandResult(result);

    // Save the shell command result to storage
    if (conversationId && message.id) {
      try {
        const commandKey = `shell_command_${conversationId}_${message.id}`;
        
        const stateToSave = {
          executed: true,
          result: result,
          timestamp: new Date().toISOString(),
          messageId: message.id,
          conversationId: conversationId
        };

        // Save to both storage types for redundancy
        localStorage.setItem(commandKey, JSON.stringify(stateToSave));
        sessionStorage.setItem(commandKey, JSON.stringify(stateToSave));

        console.log('Saved shell command state to storage:', commandKey);
      } catch (storageError) {
        console.error('Error saving shell command state to storage:', storageError);
      }
    }

    console.log('Shell command execution completed for message:', message.id);
  };

  // Function to copy code to clipboard
  const copyToClipboard = (code: string) => {
    navigator.clipboard.writeText(code);
    setCopiedCode(code);
    setTimeout(() => setCopiedCode(null), 2000); // Reset after 2 seconds
  };

  // Format file size to human-readable format
  const formatFileSize = (bytes: number): string => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  // Render code blocks with syntax highlighting and copy button
  const components = {
    code({ node, inline, className, children, ...props }: any) {
      const match = /language-(\w+)/.exec(className || '');
      const code = String(children).replace(/\n$/, '');

      return !inline && match ? (
        <div style={messageBubbleStyles.ai.codeBlock}>
          <div style={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            padding: '0.5rem 1rem',
            backgroundColor: isDarkTheme ? '#252526' : '#f0f4f8',
            borderBottom: `1px solid ${isDarkTheme ? '#3E3E42' : '#e2e8f0'}`,
            fontSize: '0.8rem',
            color: isDarkTheme ? '#e6e6e6' : '#334155',
            borderRadius: '0.5rem 0.5rem 0 0'
          }}>
            <span style={{
              fontWeight: 600,
              color: isDarkTheme ? 'var(--color-primary-light)' : 'var(--color-primary)'
            }}>
              {match[1]}
            </span>
            <button
              onClick={() => copyToClipboard(code)}
              style={{
                background: 'none',
                border: 'none',
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                color: copiedCode === code ? 'var(--color-success)' : isDarkTheme ? '#e6e6e6' : '#64748b',
                padding: '0.25rem',
                borderRadius: '0.25rem',
                transition: 'all 0.2s ease'
              }}
              title="Copy code"
            >
              {copiedCode === code ? (
                <>
                  <CheckIcon className="w-4 h-4 mr-1" /> Copied
                </>
              ) : (
                <>
                  <ClipboardDocumentIcon className="w-4 h-4 mr-1" /> Copy
                </>
              )}
            </button>
          </div>
          <SyntaxHighlighter
            style={isDarkTheme ? vscDarkPlus : oneLight}
            language={match[1]}
            PreTag="div"
            {...props}
            customStyle={{
              margin: 0,
              borderRadius: '0 0 0.5rem 0.5rem',
              fontSize: '0.9rem',
              lineHeight: 1.5,
              padding: '1rem',
              backgroundColor: isDarkTheme ? '#1E1E1E' : '#f8fafc'
            }}
            codeTagProps={{
              style: {
                fontFamily: 'Menlo, Monaco, Consolas, "Courier New", monospace',
              }
            }}
          >
            {code}
          </SyntaxHighlighter>
        </div>
      ) : (
        <code style={{
          ...messageBubbleStyles.ai.inlineCode,
          backgroundColor: isDarkTheme ? 'var(--color-surface-dark)' : '#f1f5f9',
          color: isDarkTheme ? 'var(--color-primary-light)' : 'var(--color-primary)',
          padding: '0.1rem 0.3rem',
          borderRadius: '0.25rem',
          fontFamily: 'Menlo, Monaco, Consolas, "Courier New", monospace',
          fontSize: '0.9em',
        }} {...props}>
          {children}
        </code>
      );
    }
  };

  // Add or update these helper functions for the status display
  const getStatusColor = (status: string): string => {
    switch (status) {
      case 'UPLOADED':
        return 'var(--color-info)';
      case 'PROCESSING':
      case 'EMBEDDING':
        return 'var(--color-warning)';
      case 'PROCESSED':
        return 'var(--color-success)';
      case 'ERROR':
        return 'var(--color-error)';
      default:
        return 'var(--color-text-muted)';
    }
  };

  const getStatusText = (status: string): string => {
    switch (status) {
      case 'UPLOADED':
        return 'Uploaded';
      case 'PROCESSING':
        return 'Processing';
      case 'EMBEDDING':
        return 'Generating embeddings';
      case 'PROCESSED':
        return 'Ready';
      case 'ERROR':
        return 'Error';
      default:
        return status;
    }
  };

  // Determine message type
  const isPredictorMessage = message.predictor || message.predictions || message.isUserCommand;
  
  // Get appropriate styles based on message type and theme
  const getContainerStyle = () => {
    if (isPredictorMessage && isAI) {
      return messageBubbleStyles.predictor.container;
    }
    return isAI ? messageBubbleStyles.ai.container : messageBubbleStyles.user.container;
  };

  const getHeaderStyle = () => {
    if (isPredictorMessage && isAI) {
      return messageBubbleStyles.predictor.header;
    }
    return isAI ? messageBubbleStyles.ai.header : messageBubbleStyles.user.header;
  };

  const getContentStyle = () => {
    if (isPredictorMessage && isAI) {
      // Use theme-specific styles for better visibility
      if (isDarkTheme) {
        return messageBubbleStyles.predictor.contentDark;
      } else {
        return messageBubbleStyles.predictor.contentLight;
      }
    }
    return isAI ? messageBubbleStyles.ai.content : messageBubbleStyles.user.content;
  };

  const getAvatarStyle = () => {
    if (isPredictorMessage && isAI) {
      return messageBubbleStyles.predictor.avatar;
    }
    return isAI ? messageBubbleStyles.ai.avatar : messageBubbleStyles.user.avatar;
  };

  const getTimestampStyle = () => {
    if (isPredictorMessage && isAI) {
      return messageBubbleStyles.predictor.timestamp;
    }
    return isAI ? messageBubbleStyles.ai.timestamp : messageBubbleStyles.user.timestamp;
  };

  return (
    <div
      style={getContainerStyle()}
      data-context-tool={hasReadContextTool ? "true" : "false"}
      data-message-id={message.id}
      data-predictor={isPredictorMessage ? "true" : "false"}
    >
      {/* User messages: Show header above the message content */}
      {!isAI && (
        <div style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'flex-end',
          marginBottom: '0.5rem',
          width: '100%'
        }}>
          <div style={{
            display: 'flex',
            alignItems: 'center',
            gap: '0.5rem'
          }}>
            <div style={{
              fontSize: '0.875rem',
              fontWeight: 700,
              color: 'var(--color-primary)',
              letterSpacing: '0.025em'
            }}>
              {message.isUserCommand ? 'USER COMMAND' : 'USER'}
            </div>
            <UserIcon size={24} variant="small" />
            <div style={{
              fontSize: '0.75rem',
              color: 'var(--color-text-muted)'
            }}>
              {new Date(message.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
            </div>
          </div>
        </div>
      )}

      {/* AI messages: Keep original header layout */}
      {isAI && (
        <div style={getHeaderStyle()}>
          <div style={getAvatarStyle()}>
            {isPredictorMessage ? 'P' : 'AI'}
          </div>
          <div style={{
            fontSize: '0.875rem',
            fontWeight: 700,
            color: isPredictorMessage ? '#4f8bff' : 'var(--color-primary)',
            marginRight: '0.5rem',
            letterSpacing: '0.025em'
          }}>
            {isPredictorMessage ? 'PREDICTOR' : 'AI'}
          </div>
          {isPredictorMessage && (
            <div style={messageBubbleStyles.predictor.badge}>
              Predictor Result
            </div>
          )}
          <div style={getTimestampStyle()}>
            {new Date(message.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
          </div>
        </div>
      )}

      <div style={message.error ? messageBubbleStyles.predictor.errorContent : getContentStyle()}>
        {/* File attachment for user messages */}
        {!isAI && message.fileAttachment && (
          <div style={{
            display: 'flex',
            alignItems: 'center',
            padding: '0.5rem',
            backgroundColor: 'var(--color-surface-light)',
            borderRadius: '0.5rem',
            marginBottom: '0.5rem',
            maxWidth: '100%',
            overflow: 'hidden'
          }}>
            <div style={{ marginRight: '0.5rem' }}>
              {message.fileAttachment.type === 'application/pdf' ? (
                <DocumentTextIcon className="h-6 w-6 text-red-500" />
              ) : message.fileAttachment.type === 'application/vnd.openxmlformats-officedocument.wordprocessingml.document' ? (
                <DocumentTextIcon className="h-6 w-6 text-blue-500" />
              ) : message.fileAttachment.type === 'text/plain' ? (
                <DocumentTextIcon className="h-6 w-6 text-gray-500" />
              ) : (
                <DocumentIcon className="h-6 w-6 text-gray-500" />
              )}
            </div>
            <div style={{ flex: 1, minWidth: 0 }}>
              <div style={{
                fontWeight: 500,
                whiteSpace: 'nowrap',
                overflow: 'hidden',
                textOverflow: 'ellipsis'
              }}>
                {message.fileAttachment.name}
              </div>
              <div style={{
                fontSize: '0.75rem',
                display: 'flex',
                alignItems: 'center',
                color: message.fileAttachment.status === 'ERROR' ? 'var(--color-error)' : 'var(--color-text-muted)'
              }}>
                {formatFileSize(message.fileAttachment.size)}
                {message.fileAttachment.status && (
                  <>
                    <span style={{ margin: '0 0.25rem' }}>?</span>
                    <span style={{
                      display: 'flex',
                      alignItems: 'center',
                      color: getStatusColor(message.fileAttachment.status)
                    }}>
                      {message.fileAttachment.status === 'PROCESSING' && (
                        <span className="loading-dot-animation" style={{ marginRight: '0.25rem' }}></span>
                      )}
                      {message.fileAttachment.status === 'EMBEDDING' && (
                        <span className="loading-dot-animation" style={{ marginRight: '0.25rem' }}></span>
                      )}
                      {getStatusText(message.fileAttachment.status)}
                    </span>
                  </>
                )}
                {message.fileAttachment.documentId && message.fileAttachment.status === 'PROCESSED' && (
                  <a
                    href={`/api/documents/download/${message.fileAttachment.documentId}`}
                    target="_blank"
                    rel="noopener noreferrer"
                    style={{
                      marginLeft: '0.5rem',
                      color: 'var(--color-primary)',
                      display: 'flex',
                      alignItems: 'center'
                    }}
                  >
                    <ArrowDownTrayIcon className="h-3 w-3 mr-1" />
                    Download
                  </a>
                )}
              </div>
              {message.fileAttachment.processingError && (
                <div style={{
                  fontSize: '0.75rem',
                  color: 'var(--color-error)',
                  marginTop: '0.25rem'
                }}>
                  Error: {message.fileAttachment.processingError}
                </div>
              )}
            </div>
            {message.fileAttachment.url && (
              <a
                href={message.fileAttachment.url}
                download={message.fileAttachment.name}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  padding: '0.25rem',
                  borderRadius: '0.25rem',
                  color: 'var(--color-primary)',
                  transition: 'all 0.2s ease',
                  textDecoration: 'none'
                }}
                title="Download file"
              >
                <ArrowDownTrayIcon className="h-5 w-5" />
              </a>
            )}
          </div>
        )}

        {isAI ? (
          <div style={{
            ...markdownStyles.container,
            fontSize: '0.95rem',
            lineHeight: '1.6',
          }}>
            {(message.isStreaming || (message as any).isProcessingOnly) && (message.content === '' || (message as any).isLoadingOnly) ? (
              // Show an animated loading indicator when streaming just started, processing documents, or for loading-only messages
              <div style={{ color: 'var(--color-text-muted)' }}>
                <div className="typing-animation">
                  <span className="dot"></span>
                  <span className="dot"></span>
                  <span className="dot"></span>
                </div>
              </div>
            ) : isContextResultMessage ? (
              // This is a context result message, display it directly
              <>
                <div className="flex flex-col rounded-md overflow-hidden my-3" style={{
                  border: '1px solid var(--color-border-accent)',
                  backgroundColor: 'var(--color-surface-accent)'
                }}>
                  {/* Result header */}
                  <div className="flex items-center p-2 border-b border-opacity-20" style={{
                    borderColor: 'var(--color-border-accent)',
                    backgroundColor: 'rgba(var(--color-success-rgb), 0.1)'
                  }}>
                    <span className="text-xs font-medium" style={{
                      color: 'var(--color-success)'
                    }}>
                      Context Loaded
                    </span>
                  </div>

                  {/* Result content */}
                  <div className="p-3">
                    {/* Extract and display the context rules */}
                    {(() => {
                      const lines = message.content.split('\n');
                      const rulesStartIndex = lines.findIndex(line => line.includes('Your context rules:')) + 1;
                      const aiResponseIndex = lines.findIndex(line => line.includes('AI Response:'));

                      if (rulesStartIndex > 0 && aiResponseIndex > rulesStartIndex) {
                        const rules = lines.slice(rulesStartIndex, aiResponseIndex).join('\n').trim();
                        const aiResponse = lines.slice(aiResponseIndex + 1).join('\n').trim();

                        return (
                          <>
                            <div className="text-sm mb-2" style={{ color: 'var(--color-text)' }}>
                              Your context rules:
                            </div>
                            <div style={{
                              backgroundColor: isDarkTheme ? 'rgba(0,0,0,0.2)' : 'rgba(255,255,255,0.5)',
                              padding: '0.75rem',
                              borderRadius: '0.25rem',
                              color: 'var(--color-text)',
                              fontSize: '0.9rem',
                              fontStyle: 'italic',
                              border: '1px solid var(--color-border)'
                            }}>
                              {rules}
                            </div>

                            {/* AI Response section */}
                            <div className="mt-3 pt-3 border-t" style={{ borderColor: 'var(--color-border-accent)' }}>
                              <div className="font-medium mb-1 text-xs" style={{ color: 'var(--color-primary)' }}>
                                AI Response:
                              </div>
                              <div className="text-sm" style={{ color: 'var(--color-text)' }}>
                                {aiResponse}
                              </div>
                            </div>
                          </>
                        );
                      }

                      return (
                        <div className="text-sm" style={{ color: 'var(--color-text-muted)' }}>
                          {message.content}
                        </div>
                      );
                    })()}
                  </div>
                </div>
              </>
            ) : (showContextTool || (hasReadContextTool && !contextResult)) ? (
              // Show the context reading button if the message contains a read_context tool call
              // or if we detected a phrase that should trigger the context tool
              <>
                <div>
                  {/* Display the message content up to the tool call */}
                  {message.content.includes('TOOL:')
                    ? message.content.split('TOOL:')[0]
                    : message.content.split(/read_context|context rules/i)[0]}
                </div>
                <ContextReadingButton
                  onComplete={handleContextReadComplete}
                  toolText={toolText}
                  messageId={message.id}
                  conversationId={conversationId || message.conversationId}
                  key={`context-button-${message.id}-${lastUpdated}`} // Add key with lastUpdated to force re-render
                />
              </>
            ) : hasReadContextTool && contextResult ? (
              // Show the context result if the context has been read
              <>
                <div>
                  {/* Display the message content up to the tool call */}
                  {message.content.includes('TOOL:')
                    ? message.content.split('TOOL:')[0]
                    : message.content.split(/read_context|context rules/i)[0]}
                </div>

                <div className="flex flex-col rounded-md overflow-hidden my-3" style={{
                  border: '1px solid var(--color-border-accent)',
                  backgroundColor: 'var(--color-surface-accent)'
                }}>
                  {/* Result header */}
                  <div className="flex items-center p-2 border-b border-opacity-20" style={{
                    borderColor: 'var(--color-border-accent)',
                    backgroundColor: contextResult.success
                      ? 'rgba(var(--color-success-rgb), 0.1)'
                      : 'rgba(var(--color-error-rgb), 0.1)'
                  }}>
                    <span className="text-xs font-medium" style={{
                      color: contextResult.success ? 'var(--color-success)' : 'var(--color-error)'
                    }}>
                      {contextResult.success ? 'Context Loaded' : 'Error Loading Context'}
                    </span>
                  </div>

                  {/* Result content */}
                  <div className="p-3">
                    {contextResult.success ? (
                      contextResult.result?.has_context ? (
                        <>
                          <div className="text-sm mb-2" style={{ color: 'var(--color-text)' }}>
                            Your context rules:
                          </div>
                          <div style={{
                            backgroundColor: isDarkTheme ? 'rgba(0,0,0,0.2)' : 'rgba(255,255,255,0.5)',
                            padding: '0.75rem',
                            borderRadius: '0.25rem',
                            color: 'var(--color-text)',
                            fontSize: '0.9rem',
                            fontStyle: 'italic',
                            border: '1px solid var(--color-border)'
                          }}>
                            {contextResult.result.user_context}
                          </div>

                          {/* AI Response section - always shown as part of the same box */}
                          <div className="mt-3 pt-3 border-t" style={{ borderColor: 'var(--color-border-accent)' }}>
                            <div className="font-medium mb-1 text-xs" style={{ color: 'var(--color-primary)' }}>
                              AI Response:
                            </div>
                            <div className="text-sm" style={{ color: 'var(--color-text)' }}>
                              {aiContextResponse || "I've read your context preferences and will adjust my responses accordingly."}
                            </div>
                          </div>
                        </>
                      ) : (
                        <>
                          <div className="text-sm" style={{ color: 'var(--color-text-muted)' }}>
                            No user context rules found. You can add rules in the AI Rules settings.
                          </div>

                          {/* AI Response section - always shown as part of the same box */}
                          <div className="mt-3 pt-3 border-t" style={{ borderColor: 'var(--color-border-accent)' }}>
                            <div className="font-medium mb-1 text-xs" style={{ color: 'var(--color-primary)' }}>
                              AI Response:
                            </div>
                            <div className="text-sm" style={{ color: 'var(--color-text)' }}>
                              {aiContextResponse || "I didn't find any saved context preferences. I'll continue with default behavior."}
                            </div>
                          </div>
                        </>
                      )
                    ) : (
                      <div className="text-sm" style={{ color: 'var(--color-error)' }}>
                        Error: {contextResult.error}
                      </div>
                    )}
                  </div>
                </div>
              </>
            ) : hasShellCommandTool && shellCommandResult ? (
              // Show the shell command result if the command has been executed
              <>
                <div>
                  {/* Display the message content up to the tool call */}
                  {message.content.split(/\{[^}]*"tool"[^}]*"runshellcommand"/i)[0]}
                </div>

                {/* Use the new ShellCommandResult component for better presentation */}
                {shellCommandResult && (
                  <ShellCommandResult result={shellCommandResult} />
                )}
              </>
            ) : message.isSqlResult ? (
              // Show SQL query results with special formatting
              <>
                <div className="sql-result-header flex items-center mb-2 p-2 bg-blue-50 dark:bg-blue-900 rounded">
                  <span className="sql-result-icon mr-2">
                    <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 text-blue-600 dark:text-blue-300" viewBox="0 0 20 20" fill="currentColor">
                      <path d="M3 12v3c0 1.657 3.134 3 7 3s7-1.343 7-3v-3c0 1.657-3.134 3-7 3s-7-1.343-7-3z" />
                      <path d="M3 7v3c0 1.657 3.134 3 7 3s7-1.343 7-3V7c0 1.657-3.134 3-7 3S3 8.657 3 7z" />
                      <path d="M17 5c0 1.657-3.134 3-7 3S3 6.657 3 5s3.134-3 7-3 7 1.343 7 3z" />
                    </svg>
                  </span>
                  <span className="sql-result-title font-medium text-blue-700 dark:text-blue-300">Database Query Result</span>
                </div>
                <div style={{
                  overflowX: 'auto',
                  border: isDarkTheme ? '1px solid #2f374f' : '1px solid #e5e7eb',
                  borderRadius: '0.5rem',
                  padding: '1rem',
                  backgroundColor: isDarkTheme ? '#1e2333' : '#ffffff',
                  color: isDarkTheme ? '#ffffff' : '#1f2937'
                }}>
                  <ReactMarkdown
                    remarkPlugins={[remarkGfm]}
                    components={components}
                  >
                    {message.content}
                  </ReactMarkdown>
                </div>
              </>
            ) : message.predictor ? (
              // Show Predictor results with special formatting
              <>
                <div className="predictor-result-header flex items-center mb-2 p-2 bg-green-50 dark:bg-green-900 rounded">
                  <span className="predictor-result-icon mr-2">
                    <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 text-green-600 dark:text-green-300" viewBox="0 0 20 20" fill="currentColor">
                      <path fillRule="evenodd" d="M3 3a1 1 0 011-1h12a1 1 0 011 1v3a1 1 0 01-.293.707l-5.414 5.414a1 1 0 01-.707.293H8a1 1 0 01-.707-.293L2.293 6.707A1 1 0 012 6V3zm14 12a1 1 0 01-1 1H4a1 1 0 01-1-1v-3a1 1 0 01.293-.707l5.414-5.414a1 1 0 01.707-.293h2.586a1 1 0 01.707.293l5.414 5.414A1 1 0 0117 12v3z" clipRule="evenodd" />
                    </svg>
                  </span>
                  <span className="predictor-result-title font-medium text-green-700 dark:text-green-300">
                    Predictor Result
                  </span>
                </div>
                <div style={{
                  overflowX: 'auto',
                  border: isDarkTheme ? '1px solid #2f374f' : '1px solid #e5e7eb',
                  borderRadius: '0.5rem',
                  padding: '1rem',
                  backgroundColor: isDarkTheme ? '#1e2333' : '#ffffff',
                  color: isDarkTheme ? '#ffffff' : '#1f2937'
                }}>
                  <ReactMarkdown
                    remarkPlugins={[remarkGfm]}
                    components={components}
                  >
                    {message.content}
                  </ReactMarkdown>
                </div>
                {/* Display Prediction Table */}
                {(() => {
                  console.log('Checking prediction display conditions:', {
                    isPredictor: message.predictor,
                    hasPredictions: !!message.predictions,
                    predictionsLength: message.predictions?.length,
                    messageId: message.id,
                    predictions: message.predictions
                  });
                  return message.predictor && message.predictions && message.predictions.length > 0;
                })() && (
                  <div style={{ marginTop: '1rem' }}>
                    <div style={{
                      backgroundColor: 'var(--color-surface-light)',
                      borderRadius: '0.5rem',
                      padding: '1rem',
                      border: '2px solid var(--color-border)',
                      marginBottom: '1rem',
                      boxShadow: '0 4px 6px rgba(0, 0, 0, 0.1)',
                      backdropFilter: 'blur(10px)'
                    }}>
                      <h4 style={{
                        margin: '0 0 1rem 0',
                        fontSize: '1rem',
                        fontWeight: 600,
                        color: 'var(--color-text)',
                        textShadow: '0 1px 2px rgba(0,0,0,0.1)'
                      }}>
                        🎯 Predicted Route Table ({message.predictions.length} routes)
                      </h4>
                      
                      <div style={{
                        overflowX: 'auto',
                        maxHeight: '400px',
                        overflowY: 'auto'
                      }}>
                        <table style={{
                          width: '100%',
                          borderCollapse: 'collapse',
                          fontSize: '0.85rem'
                        }}>
                          <thead>
                            <tr style={{ 
                              backgroundColor: 'var(--color-surface-dark)',
                              borderBottom: '2px solid var(--color-border)'
                            }}>
                              <th style={{ 
                                padding: '0.75rem 0.5rem', 
                                border: '1px solid var(--color-border)', 
                                textAlign: 'left',
                                fontWeight: 600,
                                color: 'var(--color-text)',
                                backgroundColor: 'rgba(59, 130, 246, 0.1)',
                                fontSize: '0.875rem'
                              }}>Startpoint</th>
                              <th style={{ 
                                padding: '0.75rem 0.5rem', 
                                border: '1px solid var(--color-border)', 
                                textAlign: 'left',
                                fontWeight: 600,
                                color: 'var(--color-text)',
                                backgroundColor: 'rgba(59, 130, 246, 0.1)',
                                fontSize: '0.875rem'
                              }}>Endpoint</th>
                              <th style={{ 
                                padding: '0.75rem 0.5rem', 
                                border: '1px solid var(--color-border)', 
                                textAlign: 'left',
                                fontWeight: 600,
                                color: 'var(--color-text)',
                                backgroundColor: 'rgba(59, 130, 246, 0.1)',
                                fontSize: '0.875rem'
                              }}>Place Slack</th>
                              <th style={{ 
                                padding: '0.75rem 0.5rem', 
                                border: '1px solid var(--color-border)', 
                                textAlign: 'left',
                                fontWeight: 600,
                                color: 'var(--color-text)',
                                backgroundColor: 'rgba(59, 130, 246, 0.1)',
                                fontSize: '0.875rem'
                              }}>CTS Slack</th>
                              <th style={{ 
                                padding: '0.75rem 0.5rem', 
                                border: '1px solid var(--color-border)', 
                                textAlign: 'left',
                                fontWeight: 600,
                                color: 'var(--color-text)',
                                backgroundColor: 'rgba(59, 130, 246, 0.1)',
                                fontSize: '0.875rem'
                              }}>Predicted Route Slack</th>
                            </tr>
                          </thead>
                          <tbody>
                            {message.predictions.slice(0, 10).map((prediction: PredictionResult, index: number) => (
                              <tr key={index} style={{ 
                                backgroundColor: index % 2 === 0 ? 'rgba(255, 255, 255, 0.02)' : 'rgba(59, 130, 246, 0.05)',
                                transition: 'background-color 0.2s ease'
                              }}>
                                <td style={{ 
                                  padding: '0.75rem 0.5rem', 
                                  border: '1px solid var(--color-border)',
                                  color: 'var(--color-text)',
                                  fontSize: '0.875rem',
                                  fontWeight: 500
                                }}>
                                  {prediction.startpoint || prediction.beginpoint}
                                </td>
                                <td style={{ 
                                  padding: '0.75rem 0.5rem', 
                                  border: '1px solid var(--color-border)',
                                  color: 'var(--color-text)',
                                  fontSize: '0.875rem',
                                  fontWeight: 500
                                }}>
                                  {prediction.endpoint}
                                </td>
                                <td style={{ 
                                  padding: '0.75rem 0.5rem', 
                                  border: '1px solid var(--color-border)',
                                  color: 'var(--color-text)',
                                  fontSize: '0.875rem',
                                  fontFamily: 'monospace',
                                  textAlign: 'right'
                                }}>
                                  {typeof prediction.place_slack === 'number' ? prediction.place_slack.toFixed(4) : prediction.place_slack}
                                </td>
                                <td style={{ 
                                  padding: '0.75rem 0.5rem', 
                                  border: '1px solid var(--color-border)',
                                  color: 'var(--color-text)',
                                  fontSize: '0.875rem',
                                  fontFamily: 'monospace',
                                  textAlign: 'right'
                                }}>
                                  {typeof prediction.cts_slack === 'number' ? prediction.cts_slack.toFixed(4) : prediction.cts_slack}
                                </td>
                                <td style={{ 
                                  padding: '0.75rem 0.5rem', 
                                  border: '1px solid var(--color-border)', 
                                  fontWeight: 700,
                                  color: 'var(--color-success)',
                                  fontSize: '0.875rem',
                                  fontFamily: 'monospace',
                                  textAlign: 'right',
                                  backgroundColor: 'rgba(34, 197, 94, 0.1)'
                                }}>
                                  {typeof prediction.predicted_route_slack === 'number' ? prediction.predicted_route_slack.toFixed(4) : prediction.predicted_route_slack}
                                </td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                      
                      {message.predictions.length > 10 && (
                        <div style={{
                          marginTop: '0.75rem',
                          padding: '0.5rem',
                          fontSize: '0.875rem',
                          color: 'var(--color-text)',
                          textAlign: 'center',
                          backgroundColor: 'rgba(59, 130, 246, 0.1)',
                          borderRadius: '0.25rem',
                          border: '1px solid rgba(59, 130, 246, 0.2)',
                          fontWeight: 500
                        }}>
                          📊 Showing first 10 of {message.predictions.length} predictions (Download CSV for complete data)
                        </div>
                      )}
                    </div>
                    
                    {/* Download Button */}
                    <div style={{ marginTop: '0.75rem' }}>
                      <button
                        style={{
                          display: 'inline-flex',
                          alignItems: 'center',
                          backgroundColor: 'var(--color-success)',
                          color: 'white',
                          padding: '0.75rem 1.5rem',
                          borderRadius: '0.5rem',
                          border: 'none',
                          fontSize: '0.875rem',
                          fontWeight: 600,
                          cursor: 'pointer',
                          transition: 'all 0.2s ease',
                          boxShadow: '0 2px 4px rgba(34, 197, 94, 0.3)',
                          textShadow: '0 1px 2px rgba(0,0,0,0.2)'
                        }}
                        onClick={(e) => {
                          e.preventDefault();
                          
                          // Create CSV content from predictions
                          const headers = ['Startpoint', 'Endpoint', 'Place Slack', 'CTS Slack', 'Predicted Route Slack'];
                          const csvContent = [
                            headers.join(','),
                            ...message.predictions.map((prediction: any) => [
                              `"${prediction.startpoint || prediction.beginpoint || ''}"`,
                              `"${prediction.endpoint || ''}"`,
                              prediction.place_slack || '',
                              prediction.cts_slack || '',
                              prediction.predicted_route_slack || ''
                            ].join(','))
                          ].join('\n');
                          
                          // Create and download CSV file
                          const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
                          const link = document.createElement('a');
                          const url = URL.createObjectURL(blob);
                          link.setAttribute('href', url);
                          link.setAttribute('download', `predicted_route_table_${new Date().toISOString().slice(0, 19).replace(/:/g, '-')}.csv`);
                          link.style.visibility = 'hidden';
                          document.body.appendChild(link);
                          link.click();
                          document.body.removeChild(link);
                          URL.revokeObjectURL(url);
                          
                          console.log('CSV download completed');
                        }}
                        onMouseEnter={(e) => {
                          e.currentTarget.style.backgroundColor = 'var(--color-success-dark)';
                          e.currentTarget.style.transform = 'translateY(-1px)';
                          e.currentTarget.style.boxShadow = '0 4px 8px rgba(34, 197, 94, 0.4)';
                        }}
                        onMouseLeave={(e) => {
                          e.currentTarget.style.backgroundColor = 'var(--color-success)';
                          e.currentTarget.style.transform = 'translateY(0)';
                          e.currentTarget.style.boxShadow = '0 2px 4px rgba(34, 197, 94, 0.3)';
                        }}
                      >
                        <ArrowDownTrayIcon className="h-4 w-4 mr-1" />
                        Download Complete Route Table ({message.predictions.length} routes) as CSV
                      </button>
                    </div>
                  </div>
                )}
                
                {/* Display error information for predictor messages */}
                {message.predictor && message.error && (
                  <div style={{
                    marginTop: '1rem',
                    backgroundColor: 'var(--color-error-light)',
                    borderRadius: '0.5rem',
                    padding: '1rem',
                    border: '1px solid var(--color-error)',
                  }}>
                    <h4 style={{
                      margin: '0 0 0.5rem 0',
                      fontSize: '1rem',
                      fontWeight: 600,
                      color: 'var(--color-error)',
                      display: 'flex',
                      alignItems: 'center'
                    }}>
                      <InformationCircleIcon className="h-5 w-5 mr-2" />
                      Prediction Error Details
                    </h4>
                    <div style={{
                      fontSize: '0.9rem',
                      color: 'var(--color-text)',
                      fontFamily: 'monospace',
                      backgroundColor: 'var(--color-surface)',
                      padding: '0.5rem',
                      borderRadius: '0.25rem',
                      border: '1px solid var(--color-border)'
                    }}>
                      {message.error}
                    </div>
                  </div>
                )}
                
                {/* Handle Download Initiation Message */}
                {message.downloadUrl && message.fileName && (
                  <div style={{ marginTop: '0.75rem' }}>
                    <a
                      href={message.downloadUrl}
                      download={message.fileName}
                      style={{
                        display: 'inline-flex',
                        alignItems: 'center',
                        backgroundColor: 'var(--color-success)',
                        color: 'white',
                        padding: '0.5rem 1rem',
                        borderRadius: '0.25rem',
                        textDecoration: 'none',
                        fontSize: '0.85rem',
                        fontWeight: 500,
                        transition: 'all 0.2s ease',
                      }}
                    >
                      <ArrowDownTrayIcon className="h-4 w-4 mr-1" />
                      Download {message.fileName}
                    </a>
                  </div>
                )}
              </>
            ) : (
              <div style={isPredictorMessage ? (isDarkTheme ? markdownStyles.predictorContainerDark : markdownStyles.predictorContainerLight) : markdownStyles.container}>
                <ReactMarkdown
                  remarkPlugins={[remarkGfm]}
                  components={{
                  ...components,
                  // Add styling for other markdown elements
                  p: ({node, children, ...props}) => (
                    <p style={{
                      marginTop: '0.75rem', 
                      marginBottom: '0.75rem',
                      color: isDarkTheme ? '#ffffff' : '#1f2937'
                    }} {...props}>
                      {children}
                    </p>
                  ),
                  h1: ({node, children, ...props}) => (
                    <h1 style={{
                      fontSize: '1.5rem',
                      fontWeight: 600,
                      marginTop: '1.5rem',
                      marginBottom: '0.75rem',
                      color: isDarkTheme ? 'var(--color-primary-light)' : 'var(--color-primary)'
                    }} {...props}>
                      {children}
                    </h1>
                  ),
                  h2: ({node, children, ...props}) => (
                    <h2 style={{
                      fontSize: '1.3rem',
                      fontWeight: 600,
                      marginTop: '1.25rem',
                      marginBottom: '0.75rem',
                      color: isDarkTheme ? 'var(--color-primary-light)' : 'var(--color-primary)'
                    }} {...props}>
                      {children}
                    </h2>
                  ),
                  h3: ({node, children, ...props}) => (
                    <h3 style={{
                      fontSize: '1.1rem',
                      fontWeight: 600,
                      marginTop: '1rem',
                      marginBottom: '0.5rem',
                      color: isDarkTheme ? 'var(--color-primary-light)' : 'var(--color-primary)'
                    }} {...props}>
                      {children}
                    </h3>
                  ),
                  ul: ({node, children, ...props}) => (
                    <ul style={{
                      paddingLeft: '1.5rem',
                      marginTop: '0.5rem',
                      marginBottom: '0.5rem',
                      listStyleType: 'disc',
                      color: isDarkTheme ? '#ffffff' : '#1f2937'
                    }} {...props}>
                      {children}
                    </ul>
                  ),
                  ol: ({node, children, ...props}) => (
                    <ol style={{
                      paddingLeft: '1.5rem',
                      marginTop: '0.5rem',
                      marginBottom: '0.5rem',
                      listStyleType: 'decimal',
                      color: isDarkTheme ? '#ffffff' : '#1f2937'
                    }} {...props}>
                      {children}
                    </ol>
                  ),
                  li: ({node, children, ...props}) => (
                    <li style={{
                      marginTop: '0.25rem',
                      marginBottom: '0.25rem',
                      color: isDarkTheme ? '#ffffff' : '#1f2937'
                    }} {...props}>
                      {children}
                    </li>
                  ),
                  a: ({node, children, ...props}) => (
                    <a style={{
                      color: 'var(--color-primary)',
                      textDecoration: 'underline',
                      fontWeight: 500
                    }} {...props} target="_blank" rel="noopener noreferrer">
                      {children}
                    </a>
                  ),
                  blockquote: ({node, children, ...props}) => (
                    <blockquote style={{
                      borderLeft: `4px solid ${isDarkTheme ? 'var(--color-primary)' : 'var(--color-primary-light)'}`,
                      paddingLeft: '1rem',
                      margin: '1rem 0',
                      color: 'var(--color-text-muted)',
                      fontStyle: 'italic'
                    }} {...props}>
                      {children}
                    </blockquote>
                  ),
                  table: ({node, children, ...props}) => (
                    <div style={{ overflowX: 'auto', marginTop: '1rem', marginBottom: '1rem' }}>
                      <table style={{
                        borderCollapse: 'collapse',
                        width: '100%',
                        fontSize: '0.9rem',
                      }} {...props}>
                        {children}
                      </table>
                    </div>
                  ),
                  thead: ({node, children, ...props}) => (
                    <thead style={{
                      backgroundColor: isDarkTheme ? 'var(--color-surface-dark)' : 'var(--color-surface-light)',
                      borderBottom: `2px solid ${isDarkTheme ? '#444' : '#e2e8f0'}`,
                    }} {...props}>
                      {children}
                    </thead>
                  ),
                  tbody: ({node, children, ...props}) => (
                    <tbody {...props}>
                      {children}
                    </tbody>
                  ),
                  tr: ({node, children, ...props}) => (
                    <tr style={{
                      borderBottom: `1px solid ${isDarkTheme ? '#333' : '#e2e8f0'}`,
                    }} {...props}>
                      {children}
                    </tr>
                  ),
                  th: ({node, children, ...props}) => (
                    <th style={{
                      padding: '0.75rem',
                      textAlign: 'left',
                      fontWeight: 600,
                      color: isDarkTheme ? 'var(--color-primary-light)' : 'var(--color-primary)',
                    }} {...props}>
                      {children}
                    </th>
                  ),
                  td: ({node, children, ...props}) => (
                    <td style={{
                      padding: '0.75rem',
                      borderRight: `1px solid ${isDarkTheme ? '#333' : '#e2e8f0'}`,
                      color: isDarkTheme ? '#ffffff' : '#1f2937',
                    }} {...props}>
                      {children}
                    </td>
                  ),
                }}
              >
                {message.content}
              </ReactMarkdown>
              </div>
            )}
          </div>
        ) : (
          <div style={{ whiteSpace: 'pre-wrap', wordBreak: 'break-word' }}>
            {message.content}
          </div>
        )}

        {/* Sources section for RAG responses */}
        {isAI && message.sources && message.sources.length > 0 && (
          <div style={{
            marginTop: '0.75rem',
            borderTop: `1px solid ${isDarkTheme ? '#444' : '#e2e8f0'}`,
            paddingTop: '0.5rem'
          }}>
            <button
              onClick={() => setShowSources(!showSources)}
              style={{
                display: 'flex',
                alignItems: 'center',
                background: 'none',
                border: 'none',
                color: 'var(--color-primary)',
                padding: '0.25rem 0',
                fontSize: '0.85rem',
                fontWeight: 500,
                cursor: 'pointer'
              }}
            >
              <InformationCircleIcon className="w-4 h-4 mr-1" />
              {showSources ? 'Hide sources' : `Show sources (${message.sources.length})`}
              {showSources ? <ChevronUpIcon className="w-3 h-3 ml-1" /> : <ChevronDownIcon className="w-3 h-3 ml-1" />}
            </button>

            {showSources && (
              <div style={{
                marginTop: '0.5rem',
                fontSize: '0.85rem',
                color: 'var(--color-text-muted)'
              }}>
                <div style={{ fontWeight: 500, marginBottom: '0.25rem' }}>Sources:</div>
                <div style={{ maxHeight: '200px', overflowY: 'auto' }}>
                  {message.sources.map((source, index) => (
                    <div key={index} style={{
                      padding: '0.5rem',
                      marginBottom: '0.5rem',
                      backgroundColor: isDarkTheme ? 'var(--color-surface-dark)' : 'var(--color-surface-light)',
                      borderRadius: '0.25rem',
                      fontSize: '0.8rem'
                    }}>
                      <div style={{ fontWeight: 500, marginBottom: '0.25rem' }}>
                        {source.metadata.fileName || 'Document'}
                        {source.score && (
                          <span style={{
                            marginLeft: '0.5rem',
                            color: 'var(--color-success)',
                            fontSize: '0.75rem'
                          }}>
                            {(source.score * 100).toFixed(1)}% match
                          </span>
                        )}
                      </div>
                      <div style={{
                        whiteSpace: 'pre-wrap',
                        wordBreak: 'break-word',
                        color: isDarkTheme ? '#ccc' : '#555'
                      }}>
                        {source.text}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Add CSS for animations */}
      <style>
        {`
          .animate-pulse {
            animation: pulse 1.5s infinite;
          }

          @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.3; }
            100% { opacity: 1; }
          }

          .animate-pulse span {
            animation: pulse 1.5s infinite;
            display: inline-block;
          }

          .animate-pulse span:nth-child(1) {
            animation-delay: 0s;
          }

          .animate-pulse span:nth-child(2) {
            animation-delay: 0.3s;
          }

          .animate-pulse span:nth-child(3) {
            animation-delay: 0.6s;
          }

          /* Code block styling */
          pre {
            position: relative;
            overflow-x: auto;
            border-radius: 0 0 0.5rem 0.5rem !important;
          }

          code {
            font-family: Menlo, Monaco, Consolas, "Courier New", monospace !important;
          }

          /* Improve table styling */
          table {
            width: 100%;
            border-collapse: collapse;
            margin: 1rem 0;
          }

          /* Improve link styling */
          a {
            transition: color 0.2s ease;
          }

          a:hover {
            text-decoration: underline;
            opacity: 0.9;
          }

          /* Typing animation */
          .typing-animation {
            display: inline-flex;
            align-items: center;
            height: 24px;
          }

          .typing-animation .dot {
            display: inline-block;
            width: 8px;
            height: 8px;
            border-radius: 50%;
            margin-right: 6px;
            background-color: var(--color-primary);
            animation: typing-dot 1.4s infinite ease-in-out both;
          }

          .typing-animation .dot:nth-child(1) {
            animation-delay: -0.32s;
          }

          .typing-animation .dot:nth-child(2) {
            animation-delay: -0.16s;
          }

          @keyframes typing-dot {
            0%, 80%, 100% {
              transform: scale(0.6);
              opacity: 0.6;
            }
            40% {
              transform: scale(1);
              opacity: 1;
            }
          }
        `}
      </style>
    </div>
  );
};

export default ChatMessage;