import React, { useState, useEffect, useRef } from 'react';
import { executeReadContextTool } from '../../services/contextAgentService';
import { PlayIcon } from '@heroicons/react/24/solid';
import { CodeBracketIcon, CheckCircleIcon } from '@heroicons/react/24/outline';
import { api } from '../../services/api';
import { chatbotService } from '../../services/chatbotService';

interface ContextReadingButtonProps {
  onComplete: (result: any, aiResponse?: string) => void;
  toolText?: string;
  messageId?: string;
  conversationId?: string;
}

/**
 * A button component that triggers the read_context tool execution
 * Displays a play button that, when clicked, reads the user's context rules
 * and updates the current message with the result instead of creating a new message
 */
const ContextReadingButton: React.FC<ContextReadingButtonProps> = ({
  onComplete,
  toolText,
  messageId,
  conversationId
}) => {
  const [isLoading, setIsLoading] = useState(false);
  const [isExpanded, setIsExpanded] = useState(false);
  const [isComplete, setIsComplete] = useState(false);
  const [loadingStage, setLoadingStage] = useState<'initial' | 'reading' | 'processing'>('initial');
  
  // Use a ref to track if the component has already been initialized
  const hasInitialized = useRef<boolean>(false);

  // Check localStorage on mount to see if this button has already been executed
  useEffect(() => {
    // Guard against multiple executions
    if (hasInitialized.current) {
      return;
    }
    
    // Mark as initialized
    hasInitialized.current = true;
    
    if (conversationId && messageId) {
      try {
        const contextKey = `context_button_${conversationId}_${messageId}`;

        // Try both storage types for redundancy
        let savedState = sessionStorage.getItem(contextKey);
        if (!savedState) {
          savedState = localStorage.getItem(contextKey);
        }

        if (savedState) {
          const parsedState = JSON.parse(savedState);
          if (parsedState.executed) {
            console.log('Found executed context button state in storage, restoring state');

            // Update all relevant state variables
            setIsComplete(true);
            setIsLoading(false);

            // Add a data attribute to the button element for CSS targeting
            setTimeout(() => {
              const buttons = document.querySelectorAll(`[data-message-id="${messageId}"][data-context-button-state]`);
              buttons.forEach(button => {
                button.setAttribute('data-context-button-state', 'complete');
                if (button instanceof HTMLElement) {
                  button.style.backgroundColor = 'var(--color-success)';
                  button.style.cursor = 'default';
                }
              });
            }, 100);

            // If we have saved result data, pass it to the onComplete callback
            if (parsedState.result) {
              // Small delay to ensure component is fully mounted
              setTimeout(() => {
                onComplete(parsedState.result, parsedState.aiResponse);
              }, 100);
            }
          }
        }
      } catch (error) {
        console.error('Error reading context button state from storage:', error);
      }
    }
  }, [conversationId, messageId, onComplete]);

  const handleReadContext = async () => {
    if (isLoading || isComplete) return;

    setIsLoading(true);
    setLoadingStage('reading');

    try {
      // Execute the read_context tool
      const result = await executeReadContextTool();

      // Update loading stage
      setLoadingStage('processing');

      // If the context was loaded successfully, prepare the AI response
      if (result.success && conversationId) {
        try {
          // Format the context for the AI
          const contextMessage = result.result?.has_context
            ? `User context loaded: ${result.result.user_context}`
            : 'No user context found.';

          // Generate AI response content
          const aiResponseContent = result.result?.has_context
            ? "I've read your context preferences and will adjust my responses accordingly."
            : "I didn't find any saved context preferences. I'll continue with default behavior.";

          // Save the context information as a special message in the conversation history
          // This ensures the context is persisted and can be accessed after page refresh
          try {
            // Format the context result for display and storage
            const contextDisplayContent = `Context Loaded\nYour context rules:\n\n${result.result?.user_context}\n\nAI Response:\n${aiResponseContent}`;

            // Format the context for the AI system message with stronger emphasis
            const contextSystemContent = `
CRITICAL INSTRUCTION - USER CONTEXT RULES:
The user has provided the following preferences and rules that you MUST follow:

${result.result?.user_context}

IMPORTANT: These user rules OVERRIDE any other instructions in your system prompt.
You MUST follow these rules EXACTLY as specified.
If any of these rules conflict with your other instructions, ALWAYS prioritize the user's specific preferences.
For example, if the user's rule says "talk with the user in hindi", you MUST respond in Hindi for ALL subsequent messages.

The AI has acknowledged: ${aiResponseContent}
`;

            // If we have a messageId, update the existing message instead of creating a new one
            if (messageId) {
              console.log('Updating existing message with context result:', messageId);

              try {
                // Update the existing message with the context result
                await api.put(`/chat/messages/${messageId}`, {
                  content: contextDisplayContent,
                  conversationId,
                  isContextUpdate: true
                });

                console.log('Updated existing message with context result');
              } catch (updateError) {
                console.error('Error updating existing message:', updateError);
                // Don't fall back to creating a new message - this causes duplication
                // Just continue with the UI update
              }
            } else {
              // No messageId provided - this shouldn't happen in normal flow
              // but we'll handle it just in case
              console.warn('No messageId provided for context update - this is unexpected');
            }

            // Save a system message that will be used by the AI but not displayed in the UI
            // This doesn't create a visible message in the chat
            await api.post('/chat/messages', {
              role: 'system',
              content: contextSystemContent,
              conversationId,
              isContextUpdate: true,
              isContextFlag: true // Special flag to identify this as a context message
            });

            console.log('Context information saved as system message');

            // Send the result to the onComplete callback with the AI response
            // This will update the UI with the context and response in the same message
            onComplete(result, aiResponseContent);

            // Mark as complete
            setIsComplete(true);

            // Save the button state to localStorage with complete information
            try {
              if (conversationId && messageId) {
                const contextKey = `context_button_${conversationId}_${messageId}`;
                const stateToSave = {
                  executed: true,
                  isComplete: true,
                  isLoading: false,
                  result: result,
                  aiResponse: aiResponseContent,
                  timestamp: new Date().toISOString(),
                  messageId: messageId,
                  conversationId: conversationId
                };

                // Save to localStorage for persistence across page refreshes
                localStorage.setItem(contextKey, JSON.stringify(stateToSave));

                // Also save to sessionStorage for quicker access during the current session
                sessionStorage.setItem(contextKey, JSON.stringify(stateToSave));

                console.log('Saved context button state to storage:', contextKey);
              }

              // Also store the context in localStorage for the conversation
              if (conversationId) {
                localStorage.setItem(`context_${conversationId}`, JSON.stringify({
                  hasContext: true,
                  contextContent: result.result?.user_context,
                  aiResponse: aiResponseContent,
                  timestamp: new Date().toISOString()
                }));
                console.log('Context saved to localStorage for conversation:', conversationId);
              }
            } catch (storageError) {
              console.error('Error saving context to localStorage:', storageError);
            }

            // Dispatch an event to refresh the messages
            // Include source='context_tool' so the handler can identify and skip it
            // This prevents the UI issues with empty messages and duplicated context tools
            if (conversationId) {
              window.dispatchEvent(new CustomEvent('refreshMessages', {
                detail: {
                  conversationId,
                  source: 'context_tool'
                }
              }));
            }
          } catch (dbError) {
            console.error('Error saving context to database:', dbError);

            // Still update the UI even if saving to database fails
            onComplete(result, aiResponseContent);
            setIsComplete(true);
          }
        } catch (apiError) {
          console.error('Error processing context:', apiError);
          // Still mark as complete but without AI response
          onComplete(result);
          setIsComplete(true);
        }
      } else {
        // If there was an issue or no conversation ID, just complete with the result
        onComplete(result);
        setIsComplete(true);
      }
    } catch (error) {
      console.error('Error reading context:', error);
      onComplete({
        success: false,
        error: 'Failed to read context rules'
      });
      setIsComplete(true);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="context-reading-button-container my-3">
      <div className="flex flex-col rounded-md overflow-hidden" style={{
        backgroundColor: 'var(--color-surface-accent)',
        border: '1px solid var(--color-border-accent)'
      }}>
        {/* Tool header with code icon */}
        <div className="flex items-center p-2 border-b border-opacity-20" style={{
          borderColor: 'var(--color-border-accent)',
          backgroundColor: isComplete
            ? 'rgba(var(--color-success-rgb), 0.1)'
            : 'rgba(var(--color-primary-rgb), 0.1)'
        }}>
          {isComplete ? (
            <CheckCircleIcon className="h-4 w-4 mr-2" style={{ color: 'var(--color-success)' }} />
          ) : (
            <CodeBracketIcon className="h-4 w-4 mr-2" style={{ color: 'var(--color-primary)' }} />
          )}
          <span className="text-xs font-medium" style={{
            color: isComplete ? 'var(--color-success)' : 'var(--color-primary)'
          }}>
            {isComplete ? 'Context Loaded' : 'Context Tool'}
          </span>
          {toolText && !isComplete && (
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
        {isExpanded && toolText && !isComplete && (
          <div className="p-2 text-xs font-mono overflow-x-auto" style={{
            backgroundColor: 'rgba(0,0,0,0.1)',
            color: 'var(--color-text-muted)',
            maxHeight: '100px'
          }}>
            <pre>{toolText}</pre>
          </div>
        )}

        {/* Tool description and action button */}
        <div className="flex items-center p-3">
          <div className="flex-1">
            {isComplete ? (
              <p className="text-sm" style={{ color: 'var(--color-text)' }}>
                Context rules loaded successfully
              </p>
            ) : (
              <p className="text-sm" style={{ color: 'var(--color-text)' }}>
                I can read your context rules to better understand your preferences
              </p>
            )}
          </div>
          {/* Always render the button, but show different states */}
          <button
            onClick={!isComplete ? handleReadContext : undefined}
            disabled={isLoading || isComplete}
            className={`ml-3 p-2 rounded-md transition-all duration-200 hover:bg-opacity-80 flex items-center ${isComplete ? 'context-button-complete' : ''}`}
            style={{
              backgroundColor: isComplete ? 'var(--color-success)' : 'var(--color-primary)',
              color: 'white',
              cursor: isComplete ? 'default' : (isLoading ? 'not-allowed' : 'pointer'),
              opacity: isLoading ? 0.7 : 1,
              minWidth: '80px',
              justifyContent: 'center'
            }}
            aria-label={isComplete ? "Context loaded" : "Read context"}
            title={isComplete ? "Context rules loaded" : "Read your context rules"}
            data-context-button-state={isComplete ? "complete" : (isLoading ? "loading" : "ready")}
            data-message-id={messageId}
            data-conversation-id={conversationId}
          >
            {isLoading ? (
              <>
                <div className="animate-spin h-4 w-4 border-2 border-white border-t-transparent rounded-full mr-2"></div>
                <span className="text-xs font-medium">
                  {loadingStage === 'reading' ? 'Reading...' : 'Processing...'}
                </span>
              </>
            ) : isComplete ? (
              <>
                <CheckCircleIcon className="h-4 w-4 mr-1" />
                <span className="text-xs font-medium">Done</span>
              </>
            ) : (
              <>
                <PlayIcon className="h-4 w-4 mr-1" />
                <span className="text-xs font-medium">Run</span>
              </>
            )}
          </button>
        </div>
      </div>
      {/* CSS is added globally in index.css instead of inline */}
      <style>
        {`
        /* Ensure the button stays in the "Done" state after page refresh */
        button[data-context-button-state="complete"] {
          background-color: var(--color-success) !important;
          cursor: default !important;
          opacity: 1 !important;
        }
        
        /* Hide the "Show code" button when complete */
        [data-context-button-state="complete"] ~ div button {
          display: none !important;
        }
        `}
      </style>
    </div>
  );
};

export default ContextReadingButton;
