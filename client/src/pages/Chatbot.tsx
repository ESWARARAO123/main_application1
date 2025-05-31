import React, { useRef, useEffect, useState } from 'react'; // Added useState
import { animations } from '../components/chat/chatStyles';
import {
  ArrowPathIcon,
  PencilIcon,
  CheckIcon,
  PlusIcon
} from '@heroicons/react/24/outline';
import { containsReadContextToolCall } from '../utils/toolParser';
import { useSidebar } from '../contexts/SidebarContext';
import { useWebSocket } from '../contexts/WebSocketContext';
import { useToolExecution } from '../hooks/useToolExecution';
import ChatInput from '../components/chat/ChatInput';
import ChatSidebar from '../components/chat/ChatSidebar';
import MessageList from '../components/chat/MessageList';
import ModelSelector from '../components/chat/ModelSelector';
import MCPServerSelector from '../components/chat/MCPServerSelector';
import ContextReadingIndicator from '../components/chat/ContextReadingIndicator';
import MCPNotifications from '../components/mcp/MCPNotifications';
import { useMCP } from '../contexts/MCPContext';
import { useMCPAgent } from '../contexts/MCPAgentContext';
import { useChatSessions } from '../hooks/useChatSessions';
import { useChatMessaging } from '../hooks/useChatMessaging';
import { useContextHandling } from '../hooks/useContextHandling';
import { ExtendedChatMessage } from '../types';
import { chatbotService } from '../services/chatbotService';

const Chatbot: React.FC = () => {
  const { isExpanded: isMainSidebarExpanded } = useSidebar();

  // Use a ref to track if context rules have been loaded
  const contextRulesLoadedRef = useRef<{[key: string]: boolean}>({});
  
  // Get chat sessions functionality
  const {
    sessions,
    activeSessionId,
    sessionTitle,
    editingTitle,
    messages,
    loadingMessages,
    loadingSessions,
    hasMoreMessages,
    expandedGroups,
    showSidebar,
    setActiveSessionId,
    setSessionTitle,
    setEditingTitle,
    setMessages,
    fetchSessions,
    fetchSessionMessages,
    loadMoreMessages,
    createNewSession,
    deleteSession,
    updateSessionTitle,
    toggleSidebar,
    toggleGroup,
    resetChat
  } = useChatSessions();

  // Get chat messaging functionality
  const {
    isLoading,
    isStreaming,
    isUploading,
    uploadProgress,
    setIsLoading,
    setIsStreaming,
    streamedContentRef,
    abortFunctionRef,
    sendChatMessage,
    stopGeneration
  } = useChatMessaging();

  // Get context handling functionality
  const {
    isRagAvailable,
    isRagEnabled,
    ragNotificationShown,
    setRagNotificationShown,
    checkForStoredContext,
    checkRagAvailability,
    forceCheckDocumentStatus,
    toggleRagMode,
    showRagAvailableNotification
  } = useContextHandling(activeSessionId);

  // Model selection state
  const [selectedModelId, setSelectedModelId] = React.useState<string | undefined>(() => {
    return localStorage.getItem('selectedModelId') || undefined;
  });

  // Get MCP functionality from the actual contexts
  const { isConnected: isMCPConnected, defaultServer, connectToServer } = useMCP();
  const { isAgentEnabled: isMCPEnabled, toggleAgent: toggleMCPEnabled } = useMCPAgent();
  
  // MCP Server selector state
  const [showServerSelector, setShowServerSelector] = useState(false);

  // Enhanced MCP helper functions
  const selectServer = async (serverId: string) => {
    console.log('Selected MCP server:', serverId);
    
    try {
      // The MCPServerSelector component handles the actual connection
      // We just need to wait a moment for the connection to establish
      // and then enable the agent and close the dialog
      
      // Wait for connection establishment (3 seconds should be enough)
      setTimeout(() => {
        // Enable the MCP agent if not already enabled
        if (!isMCPEnabled) {
          toggleMCPEnabled();
        }
        
        // Close the server selector
        setShowServerSelector(false);
      }, 3000);
      
    } catch (error) {
      console.error('Error in server selection:', error);
      // Keep dialog open if there's an error
    }
  };

  const createContextToolMessage = () => {
    const contextMessage: ExtendedChatMessage = {
      id: `context-tool-${Date.now()}`,
      role: 'assistant',
      content: '🔍 Reading context from uploaded documents...',
      timestamp: new Date(),
      isContextMessage: true
    };
    return contextMessage;
  };

  const handleMCPChatMessage = async (
    content: string,
    messages: ExtendedChatMessage[],
    activeSessionId: string | null,
    selectedModel: { id?: string },
    streamedContentRef: React.MutableRefObject<{ [key: string]: string }>,
    abortFunctionRef: React.MutableRefObject<(() => void) | null>,
    setMessages: React.Dispatch<React.SetStateAction<ExtendedChatMessage[]>>,
    setIsStreaming: React.Dispatch<React.SetStateAction<boolean>>,
    setIsLoading: React.Dispatch<React.SetStateAction<boolean>>,
    executeTool: any,
    chatbotService: any,
    fetchSessions: () => void
  ) => {
    console.log('MCP Chat message handling:', content);
    // For now, fall back to regular chat handling
    // This can be enhanced when MCP Agent context provides chat functionality
    throw new Error('MCP chat functionality not yet implemented');
  };

  // Tool execution state
  const { isExecutingTool, currentTool, executeTool } = useToolExecution();

  // Chat2SQL state - Added
  const [isChat2SqlEnabled, setIsChat2SqlEnabled] = useState(false);

  // Toggle Chat2SQL mode - Added
  const handleToggleChat2Sql = () => {
    setIsChat2SqlEnabled(prev => !prev);
  };

  const titleInputRef = useRef<HTMLInputElement>(null);

  // Fetch sessions on component mount and ensure WebSocket connection
  useEffect(() => {
    fetchSessions();

    // Initial RAG availability check
    checkRagAvailability();

    // Force check document status on mount
    forceCheckDocumentStatus(messages, setMessages, setIsLoading, setIsStreaming);
    
    // Check for stored context rules and add them to the messages array
    if (activeSessionId) {
      // Check if we've already loaded context rules for this session
      if (contextRulesLoadedRef.current[activeSessionId]) {
        console.log('Context rules already loaded for session:', activeSessionId);
        return;
      }
      
      const contextRulesKey = `context_rules_${activeSessionId}`;
      try {
        // Try session storage first, then local storage
        let storedContextRules = sessionStorage.getItem(contextRulesKey) || localStorage.getItem(contextRulesKey);
        
        if (storedContextRules) {
          const parsedRules = JSON.parse(storedContextRules);
          
          if (parsedRules.hasContext && parsedRules.rules) {
            console.log('Found stored context rules for conversation:', activeSessionId);
            
            // Add the system message with context rules to the messages array
            const systemContextMessage: ExtendedChatMessage = {
              id: `system-context-${Date.now()}`,
              role: 'system',
              content: `User context loaded: ${parsedRules.rules}`,
              timestamp: new Date(),
              isContextMessage: true
            };
            
            // Add the system message to the messages array
              setMessages(prev => {
              // Check if we already have a similar system message to avoid duplicates
              const hasSimilarMessage = prev.some(msg => 
                msg.role === 'system' && 
                msg.content.includes('User context loaded:')
              );
              
              if (hasSimilarMessage) {
                console.log('Similar system message already exists, not adding another one');
                  return prev;
                }

              return [...prev, systemContextMessage];
            });
            
            // Mark this session as processed
            contextRulesLoadedRef.current[activeSessionId] = true;
              }
            }
          } catch (error) {
        console.error('Error checking for stored context rules:', error);
      }
    }
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // Fetch messages when active session changes
  useEffect(() => {
    if (activeSessionId) {
      fetchSessionMessages(activeSessionId);

      // Check for context in localStorage
      const storedContext = checkForStoredContext(activeSessionId);
      if (storedContext) {
        console.log('Using stored context from localStorage:', storedContext);
      }
      
      // Check for stored context rules
      const contextRulesKey = `context_rules_${activeSessionId}`;
      try {
        // Check if we've already loaded context rules for this session
        if (contextRulesLoadedRef.current[activeSessionId]) {
          console.log('Context rules already loaded for session:', activeSessionId);
          return;
        }
        
        // Try session storage first, then local storage
        let storedContextRules = sessionStorage.getItem(contextRulesKey) || localStorage.getItem(contextRulesKey);
        
        if (storedContextRules) {
          const parsedRules = JSON.parse(storedContextRules);
          
          if (parsedRules.hasContext && parsedRules.rules) {
            console.log('Found stored context rules for conversation:', activeSessionId);
            
            // Add the system message with context rules to the messages array
            const systemContextMessage: ExtendedChatMessage = {
              id: `system-context-${Date.now()}`,
              role: 'system',
              content: `User context loaded: ${parsedRules.rules}`,
              timestamp: new Date(),
              isContextMessage: true
            };
            
            // Add the system message to the messages array after a slight delay
            // to ensure the messages from fetchSessionMessages are loaded first
            setTimeout(() => {
              setMessages(prev => {
                // Check if we already have a similar system message to avoid duplicates
                const hasSimilarMessage = prev.some(msg => 
                  msg.role === 'system' && 
                  msg.content.includes('User context loaded:')
                );
                
                if (hasSimilarMessage) {
                  console.log('Similar system message already exists, not adding another one');
                  return prev;
                }
                
                return [...prev, systemContextMessage];
              });
              
              // Mark this session as processed
              contextRulesLoadedRef.current[activeSessionId] = true;
            }, 500);
        }
      }
    } catch (error) {
        console.error('Error checking for stored context rules:', error);
      }
    } else {
      setMessages([]);
    }
  }, [activeSessionId]); // eslint-disable-line react-hooks/exhaustive-deps

  // Listen for refreshMessages events
  useEffect(() => {
    const handleRefreshMessages = (event: CustomEvent<{ conversationId: string; source?: string }>) => {
      const { conversationId, source } = event.detail;

      // Skip refreshing if the source is the context tool
      if (source === 'context_tool') {
        console.log('Skipping refresh from context tool to prevent UI issues');
        return;
      }

      if (conversationId && conversationId === activeSessionId) {
        console.log('Refreshing messages for conversation:', conversationId);
        fetchSessionMessages(conversationId);
      }
    };

    // Add event listener for refreshMessages
    window.addEventListener('refreshMessages', handleRefreshMessages as EventListener);

    // Listen for system message additions (like context rules)
    const handleAddSystemMessage = (event: CustomEvent<{ message: ExtendedChatMessage }>) => {
      const { message } = event.detail;
      console.log('Adding system message to conversation:', message);
      
      // Add the system message to the messages array
        setMessages(prev => {
        // Check if we already have a similar system message to avoid duplicates
        const hasSimilarMessage = prev.some(msg => 
          msg.role === 'system' && 
          msg.content.includes('User context loaded:')
        );
        
        if (hasSimilarMessage) {
          console.log('Similar system message already exists, replacing it');
          // Return the same array if we're already replacing a message to prevent re-renders
          const updatedMessages = prev.map(msg => 
            (msg.role === 'system' && msg.content.includes('User context loaded:'))
              ? message
              : msg
          );
          
          // Check if anything actually changed
          const hasChanges = updatedMessages.some((msg, idx) => msg !== prev[idx]);
          if (!hasChanges) {
            console.log('No changes needed to system messages');
            return prev;
          }

          return updatedMessages;
        }
        
        return [...prev, message];
      });
    };
    
    // Add event listener for system messages
    window.addEventListener('addSystemMessage', handleAddSystemMessage as EventListener);

    // Clean up
    return () => {
      window.removeEventListener('refreshMessages', handleRefreshMessages as EventListener);
      window.removeEventListener('addSystemMessage', handleAddSystemMessage as EventListener);
    };
  }, [activeSessionId, fetchSessionMessages]);

  // WebSocket reconnection
  const { connected: wsConnected, reconnect: wsReconnect } = useWebSocket();
  useEffect(() => {
    // Set up periodic checks with a reasonable interval (30 seconds)
    const periodicCheckInterval = setInterval(() => {
      // Only check RAG if we haven't already shown the notification
      if (!ragNotificationShown) {
        console.log('Performing periodic document status check');
        forceCheckDocumentStatus(messages, setMessages, setIsLoading, setIsStreaming);
        checkRagAvailability();
      }

      // Always check WebSocket connection
      if (!wsConnected) {
        console.log('WebSocket not connected during periodic check, attempting to reconnect...');
        wsReconnect();
      }
    }, 30000);

    // Clean up interval on unmount
    return () => {
      clearInterval(periodicCheckInterval);
    };
  }, [wsConnected, wsReconnect, ragNotificationShown]); // eslint-disable-line react-hooks/exhaustive-deps

  // Focus title input when editing
  useEffect(() => {
    if (editingTitle) {
      titleInputRef.current?.focus();
    }
  }, [editingTitle]);

  const handleSendMessage = async (content: string, file?: File, meta?: any) => {
    // Allow sending if there's text or a file
    if ((content.trim() === '' && !file) || isLoading || isUploading) return;

    // Handle Chat2SQL messages - Added
    if (meta?.chat2sql) {
      console.log('Handling Chat2SQL response in Chatbot.tsx...', meta);
      
      // If this is the SQL result from the server
      if (meta.isServerResponse) {
        const aiMessage: ExtendedChatMessage = {
          id: meta.id,
          role: 'assistant',
          content: meta.error ? `Error: ${meta.error}` : meta.content,
          timestamp: new Date(meta.timestamp),
          isSqlResult: true
        };
        
        console.log('SQL result message:', aiMessage);
        setMessages(prev => [...prev, aiMessage]);
        return;
      }
      
      // If this is the user query
      if (meta.isUserMessage) {
        const userMessage: ExtendedChatMessage = {
          id: `user-${Date.now()}`,
          role: 'user',
          content: content.trim(),
          timestamp: new Date(),
          isSqlQuery: true // Mark this as a SQL query
        };

        console.log('User SQL query message:', userMessage);
        // Add the user message to the chat
        setMessages(prev => [...prev, userMessage]);

        // Update session ID in database entries if needed
        if (activeSessionId) {
          try {
            await chatbotService.updateMessageSessionId('temp-session-id', activeSessionId);
          } catch (error) {
            console.error('Error updating session ID in database:', error);
          }
        }

        // Important: Skip AI response generation by returning early
        return;
      }

      return;
    }

    // Special handling for read_context command - only trigger for exact match
    if (content.trim().toLowerCase() === 'read_context') {
      console.log('Detected exact read_context command, triggering context tool directly');
      const aiMessage = createContextToolMessage();
      setMessages(prev => [...prev, aiMessage]);
      return;
    }

    // Don't trigger for phrases that contain read_context but aren't exactly the command
    // For example "what does read_context do?" should not trigger the tool
    
    // If MCP is enabled, use MCP chat service
    if (isMCPEnabled && !file && content.trim() !== '') {
      try {
        // First add the user message to the UI immediately
        const userMessage: ExtendedChatMessage = {
          id: `user-${Date.now()}`,
          role: 'user',
          content: content.trim(),
          timestamp: new Date()
        };
        setMessages(prev => [...prev, userMessage]);

        // Then handle the MCP chat message
        await handleMCPChatMessage(
          content,
          messages,
          activeSessionId,
          { id: selectedModelId }, // Just pass the ID, the handler will get details
          streamedContentRef,
          abortFunctionRef,
          setMessages,
          setIsStreaming,
          setIsLoading,
          executeTool,
          chatbotService, // Pass chatbotService to ensure messages are saved
          fetchSessions
        );
        return;
      } catch (error: any) {
        console.error('Error using MCP chat mode:', error);
        setMessages(prev => [...prev, {
          id: `error-${Date.now()}`,
          role: 'assistant',
          content: `Error: ${error.message}. Falling back to normal chat.`,
          timestamp: new Date()
        }]);
        // Fall through to regular chat handling below
      }
    }

    // Use our extracted chat message sending function
    const result = await sendChatMessage(
      content, 
      file, 
      messages, 
      activeSessionId, 
      selectedModelId, 
      isRagAvailable, 
      isRagEnabled, 
      setMessages, 
      fetchSessions
    );
    
    // Update active session ID if needed
    if (result?.newSessionId && (!activeSessionId || activeSessionId !== result.newSessionId)) {
      setActiveSessionId(result.newSessionId);
    }
  };

  // Toggle MCP mode
  const handleToggleMCP = () => {
    if (!isMCPEnabled && !isMCPConnected) {
      // If MCP is not enabled and not connected, show server selector first
      setShowServerSelector(true);
    } else {
      // If already enabled/connected, just toggle the agent
      toggleMCPEnabled();
    }
  };

  const isEmpty = messages.length === 0;

  useEffect(() => {
    const styleElement = document.createElement('style');
    styleElement.textContent = `
      ${animations.bounce}
      ${animations.fadeIn}
      ${animations.slideIn}

      .input-area-blur {
        background-color: transparent !important;
        -webkit-backdrop-filter: blur(5px) !important;
        backdrop-filter: blur(5px) !important;
        border: none !important;
        box-shadow: none !important;
        isolation: isolate !important;
        opacity: 1 !important;
      }

      .input-area-blur > * {
        isolation: isolate !important;
      }
    `;
    document.head.appendChild(styleElement);

    return () => {
      document.head.removeChild(styleElement);
    };
  }, []);

  return (
    <div
      className="fixed inset-0 flex flex-col"
      style={{
        backgroundColor: 'var(--color-bg)',
        left: isMainSidebarExpanded ? '64px' : '63px',
        width: isMainSidebarExpanded ? 'calc(100% - 64px)' : 'calc(100% - 50px)'
      }}
    >
      {/* MCP Server Selector */}
      <MCPServerSelector
        isOpen={showServerSelector}
        onClose={() => setShowServerSelector(false)}
        onServerSelect={selectServer}
      />
      <div
        className="px-4 py-3 flex items-center justify-between z-10 relative"
        style={{
          backgroundColor: 'transparent',
          borderColor: 'transparent',
          borderRadius: '0 0 12px 12px'
        }}
      >
        <div className="flex items-center space-x-4">
          {editingTitle ? (
            <div className="flex items-center">
              <input
                ref={titleInputRef}
                type="text"
                value={sessionTitle}
                onChange={(e) => setSessionTitle(e.target.value)}
                onBlur={updateSessionTitle}
                onKeyDown={(e) => e.key === 'Enter' && updateSessionTitle()}
                className="px-3 py-1 rounded-full"
                style={{
                  backgroundColor: 'transparent',
                  color: 'var(--color-text)',
                  border: '1px solid rgba(255, 255, 255, 0.15)'
                }}
              />
              <button
                onClick={updateSessionTitle}
                className="ml-2 p-2 rounded-full hover:bg-opacity-20 hover:bg-gray-500 transition-all hover:scale-105"
                style={{
                  color: 'var(--color-primary)',
                  backgroundColor: 'transparent',
                  border: '1px solid rgba(255, 255, 255, 0.15)'
                }}
              >
                <CheckIcon className="w-3 h-3" />
              </button>
            </div>
          ) : (
            <div className="flex items-center">
              <h2
                className="text-base md:text-lg font-semibold truncate max-w-[200px] md:max-w-none"
                style={{ color: 'var(--color-text)' }}
              >
                {activeSessionId ? sessionTitle : 'New Chat'}
              </h2>
              {activeSessionId && (
                <button
                  onClick={() => setEditingTitle(true)}
                  className="ml-2 p-1 rounded-full hover:bg-opacity-20 hover:bg-gray-500 transition-all hover:scale-105"
                  style={{
                    color: 'var(--color-text-muted)',
                    backgroundColor: 'transparent',
                    border: '1px solid rgba(255, 255, 255, 0.15)'
                  }}
                >
                  <PencilIcon className="w-3 h-3" />
                </button>
              )}
            </div>
          )}
        </div>

        <div className="flex items-center space-x-4">
          {isMCPEnabled && (
            <MCPNotifications />
          )}
          
          <ModelSelector
            onSelectModel={setSelectedModelId}
            selectedModelId={selectedModelId}
          />
          {!isEmpty && (
            <button
              onClick={() => resetChat()}
              className="p-2 rounded-full hover:bg-opacity-20 hover:bg-gray-500 transition-all hover:scale-105"
              style={{
                color: 'var(--color-text-muted)',
                backgroundColor: 'transparent',
                border: '1px solid rgba(255, 255, 255, 0.15)'
              }}
              title="Clear current chat"
            >
              <ArrowPathIcon className="w-4 h-4" />
            </button>
          )}
        </div>
      </div>

      <div className="flex-1 relative overflow-hidden">
        {showSidebar && (
          <div
            className="absolute md:relative h-full transition-all duration-300 ease-in-out z-20 md:z-0"
            style={{
              left: '0',
              width: window.innerWidth < 768 ? '100%' : '260px'
            }}
          >
            <ChatSidebar
              sessions={sessions}
              activeSessionId={activeSessionId}
              expandedGroups={expandedGroups}
              loadingSessions={loadingSessions}
              isCollapsed={false}
              onCreateSession={createNewSession}
              onSelectSession={setActiveSessionId}
              onDeleteSession={deleteSession}
              onToggleGroup={toggleGroup}
              onToggleCollapse={toggleSidebar}
            />
          </div>
        )}

        {!showSidebar && (
          <ChatSidebar
            sessions={sessions}
            activeSessionId={activeSessionId}
            expandedGroups={expandedGroups}
            loadingSessions={loadingSessions}
            isCollapsed={true}
            onCreateSession={createNewSession}
            onSelectSession={setActiveSessionId}
            onDeleteSession={deleteSession}
            onToggleGroup={toggleGroup}
            onToggleCollapse={toggleSidebar}
          />
        )}

        <div
          className={`absolute inset-0 transition-all duration-300 ease-in-out flex flex-col`}
          style={{
            backgroundColor: 'var(--color-bg)',
            marginLeft: showSidebar ? (window.innerWidth < 768 ? '0' : '260px') : '0'
          }}
        >
          {/* Context reading indicator - only show when not already displayed in a message */}
          {isExecutingTool && currentTool === 'read_context' && !messages.some(msg =>
            msg.role === 'assistant' && containsReadContextToolCall(msg.content)
          ) && (
            <div className="px-4 pt-2">
              <ContextReadingIndicator isReading={true} />
            </div>
          )}

          <MessageList
            messages={messages.filter(msg => msg.role !== 'system')}
            isLoading={isLoading}
            hasMoreMessages={hasMoreMessages}
            loadMoreMessages={loadMoreMessages}
            loadingMessages={loadingMessages}
            isEmpty={isEmpty}
            conversationId={activeSessionId || undefined}
          />

          <div
            className={`${isEmpty ? "absolute left-1/2 bottom-[10%] transform -translate-x-1/2" : "absolute bottom-0 left-0 right-0"}
            ${!isEmpty && ""} py-4 px-4 md:px-8 lg:px-16 xl:px-24 input-area-blur`}
            style={{
              maxWidth: '100%',
              margin: '0 auto',
              zIndex: 10,
              boxShadow: '0 -4px 12px rgba(0, 0, 0, 0.05)',
              backgroundColor: isEmpty ? 'transparent' : 'var(--color-bg-translucent)'
            }}
          >
            <ChatInput
              onSendMessage={handleSendMessage}
              isLoading={isLoading}
              isEmpty={isEmpty}
              isStreaming={isStreaming}
              isUploading={isUploading}
              uploadProgress={uploadProgress}
              onStopGeneration={stopGeneration}
              isRagAvailable={isRagAvailable}
              isRagEnabled={isRagEnabled}
              onToggleRag={toggleRagMode}
              isMCPAvailable={isMCPConnected}
              isMCPEnabled={isMCPEnabled}
              onToggleMCP={handleToggleMCP}
              isChat2SqlEnabled={isChat2SqlEnabled} // Added for Chat2SQL
              onToggleChat2Sql={handleToggleChat2Sql} // Added for Chat2SQL
            />

            {isEmpty && (
              <div className="flex justify-center mt-12">
                <div className="flex flex-wrap justify-center gap-2">
                  <button
                    onClick={createNewSession}
                    className="px-4 py-2 rounded-md text-sm flex items-center hover:bg-opacity-10 hover:bg-gray-500"
                    style={{
                      backgroundColor: 'var(--color-surface-dark)',
                      color: 'var(--color-text)',
                      boxShadow: '0 2px 8px rgba(0, 0, 0, 0.1)'
                    }}
                  >
                    <PlusIcon className="h-4 w-4 mr-1.5" />
                    <span>New Chat</span>
                  </button>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default Chatbot;