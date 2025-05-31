import React, { useState, useEffect, useRef } from 'react';
import { ChatMessage as ChatMessageType, ExtendedChatMessage } from '../../types';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism';
import { oneLight } from 'react-syntax-highlighter/dist/esm/styles/prism';
import {
  UserIcon,
  ClipboardDocumentIcon,
  CheckIcon,
  DocumentTextIcon,
  DocumentIcon,
  ArrowDownTrayIcon,
  InformationCircleIcon,
  ChevronDownIcon,
  ChevronUpIcon
} from '@heroicons/react/24/outline';
import { messageBubbleStyles, markdownStyles } from './chatStyles';
import { useTheme } from '../../contexts/ThemeContext';
import { RagSource } from '../../services/ragChatService';
import { containsReadContextToolCall, extractToolCall, containsShellCommandToolCall, extractShellCommand } from '../../utils/toolParser';
import ContextReadingButton from './ContextReadingButton';
import ShellCommandButton from './ShellCommandButton';
import ShellCommandResult from './ShellCommandResult';

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

  // Function to extract shell command tool text for display
  const extractShellToolText = (): string | undefined => {
    if (!isAI || !message.content) return undefined;

    // Look for JSON pattern in the content
    const jsonPattern = /\{\s*"tool":\s*"runshellcommand"[\s\S]*?\}/i;
    const match = message.content.match(jsonPattern);
    if (match) {
      return match[0];
    }

    // Look for code block with JSON
    const codeBlockPattern = /```(?:json)?\s*(\{[\s\S]*?"tool":\s*"runshellcommand"[\s\S]*?\})\s*```/i;
    const codeBlockMatch = message.content.match(codeBlockPattern);
    if (codeBlockMatch) {
      return codeBlockMatch[0];
    }

    return undefined;
  };

  // Check if the message contains a read_context tool call or has isContextTool flag
  const hasReadContextTool = isAI && (containsReadContextToolCall(message.content) || message.isContextTool);
  const toolText = hasReadContextTool ? extractToolText() : undefined;

  // Check if the message contains a runshellcommand tool call
  const hasShellCommandTool = isAI && containsShellCommandToolCall(message.content);
  const shellCommand = hasShellCommandTool ? extractShellCommand(message.content) : null;
  const shellToolText = hasShellCommandTool ? extractShellToolText() : undefined;

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

  // State for shell command tool
  const [shellCommandResult, setShellCommandResult] = useState<any>(null);
  const [aiShellCommandResponse, setAiShellCommandResponse] = useState<string | null>(null);

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

  // Check localStorage for shell command button state
  useEffect(() => {
    if (hasShellCommandTool && conversationId) {
      try {
        const commandKey = `shell_command_${conversationId}_${message.id}`;
        let storedState = sessionStorage.getItem(commandKey);
        if (!storedState) {
          storedState = localStorage.getItem(commandKey);
        }

        if (storedState) {
          const parsedState = JSON.parse(storedState);
          if (parsedState.executed) {
            console.log('Shell command was previously executed, restoring state');
            setShellCommandResult(parsedState.result);
            if (parsedState.aiResponse) {
              setAiShellCommandResponse(parsedState.aiResponse);
            }
            setLastUpdated(Date.now());
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
  const handleShellCommandComplete = async (result: any, aiResponse?: string) => {
    setShellCommandResult(result);
    if (aiResponse) {
      setAiShellCommandResponse(aiResponse);
    }
    setLastUpdated(Date.now());
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

  return (
    <div
      style={isAI ? messageBubbleStyles.ai.container : messageBubbleStyles.user.container}
      data-context-tool={hasReadContextTool ? "true" : "false"}
      data-message-id={message.id}
    >
      <div style={isAI ? messageBubbleStyles.ai.header : messageBubbleStyles.user.header}>
        {isAI ? (
          <div style={messageBubbleStyles.ai.avatar}>
            AI
          </div>
        ) : (
          <div style={messageBubbleStyles.user.avatar}>
            <UserIcon className="w-5 h-5" />
          </div>
        )}
        <div style={isAI ? messageBubbleStyles.ai.timestamp : messageBubbleStyles.user.timestamp}>
          {new Date(message.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
        </div>
      </div>

      <div style={isAI ? messageBubbleStyles.ai.content : messageBubbleStyles.user.content}>
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
                    <span style={{ margin: '0 0.25rem' }}>•</span>
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
            ) : (hasShellCommandTool && !shellCommandResult) ? (
              // Show the shell command button if the message contains a runshellcommand tool call
              <>
                <div>
                  {/* Display the message content up to the tool call */}
                  {message.content.split(/\{[^}]*"tool"[^}]*"runshellcommand"/i)[0]}
                </div>
                {shellCommand && (
                  <ShellCommandButton
                    onComplete={handleShellCommandComplete}
                    toolText={shellToolText}
                    messageId={message.id}
                    conversationId={conversationId || message.conversationId}
                    command={shellCommand}
                    key={`shell-command-button-${message.id}-${lastUpdated}`}
                  />
                )}
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
                <div className="sql-result-content overflow-x-auto border border-gray-200 dark:border-gray-700 rounded p-2 bg-white dark:bg-gray-800">
                  <ReactMarkdown
                    remarkPlugins={[remarkGfm]}
                    components={components}
                  >
                    {message.content}
                  </ReactMarkdown>
                </div>
              </>
            ) : (
              <ReactMarkdown
                remarkPlugins={[remarkGfm]}
                components={{
                  ...components,
                  // Add styling for other markdown elements
                  p: ({node, children, ...props}) => (
                    <p style={{marginTop: '0.75rem', marginBottom: '0.75rem'}} {...props}>
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
                      listStyleType: 'disc'
                    }} {...props}>
                      {children}
                    </ul>
                  ),
                  ol: ({node, children, ...props}) => (
                    <ol style={{
                      paddingLeft: '1.5rem',
                      marginTop: '0.5rem',
                      marginBottom: '0.5rem',
                      listStyleType: 'decimal'
                    }} {...props}>
                      {children}
                    </ol>
                  ),
                  li: ({node, children, ...props}) => (
                    <li style={{
                      marginTop: '0.25rem',
                      marginBottom: '0.25rem'
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
                    }} {...props}>
                      {children}
                    </td>
                  ),
                }}
              >
                {message.content}
              </ReactMarkdown>
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