import React, { useRef, useEffect, useState } from 'react';
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
import TrainingForm from '../components/TrainingForm'; // Added for predictor training form

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
    editSession,
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

  // Chat2SQL state - Added with persistence
  const [isChat2SqlEnabled, setIsChat2SqlEnabled] = useState(() => {
    try {
      const saved = localStorage.getItem('chat2sql_mode_enabled');
      return saved ? JSON.parse(saved) : false;
    } catch {
      return false;
    }
  });

  // Predictor state - Added with persistence
  const [isPredictorEnabled, setIsPredictorEnabled] = useState(() => {
    try {
      const saved = localStorage.getItem('predictor_mode_enabled');
      return saved ? JSON.parse(saved) : false;
    } catch {
      return false;
    }
  });
  const [showTrainingForm, setShowTrainingForm] = useState(false);

  // Helper functions for predictor message persistence
  const savePredictorMessage = (sessionId: string, message: ExtendedChatMessage) => {
    try {
      const key = `predictor_messages_${sessionId}`;
      const existing = localStorage.getItem(key);
      let messages = existing ? JSON.parse(existing) : [];
      
      // Add the new message
      messages.push({
        ...message,
        timestamp: message.timestamp.toISOString() // Convert Date to string for storage
      });
      
      // Keep only the last 50 messages per session to prevent localStorage bloat
      if (messages.length > 50) {
        messages = messages.slice(-50);
      }
      
      localStorage.setItem(key, JSON.stringify(messages));
      console.log('Predictor message saved to localStorage:', message.id);
    } catch (error) {
      console.error('Error saving predictor message to localStorage:', error);
    }
  };

  const loadPredictorMessages = (sessionId: string): ExtendedChatMessage[] => {
    try {
      const key = `predictor_messages_${sessionId}`;
      const stored = localStorage.getItem(key);
      if (stored) {
        const messages = JSON.parse(stored);
        return messages.map((msg: any) => ({
          ...msg,
          timestamp: new Date(msg.timestamp) // Convert string back to Date
        }));
      }
    } catch (error) {
      console.error('Error loading predictor messages from localStorage:', error);
    }
    return [];
  };

  const clearPredictorMessages = (sessionId: string) => {
    try {
      const key = `predictor_messages_${sessionId}`;
      localStorage.removeItem(key);
      console.log('Predictor messages cleared for session:', sessionId);
    } catch (error) {
      console.error('Error clearing predictor messages:', error);
    }
  };

  // Enhanced MCP helper functions
  const selectServer = async (serverId: string) => {
    console.log('Selected MCP server:', serverId);
    
    try {
      setTimeout(() => {
        if (!isMCPEnabled) {
          toggleMCPEnabled();
        }
        setShowServerSelector(false);
      }, 3000);
    } catch (error) {
      console.error('Error in server selection:', error);
    }
  };

  const createContextToolMessage = () => {
    const contextMessage: ExtendedChatMessage = {
      id: `context-tool-${Date.now()}`,
      role: 'assistant',
      content: '? Reading context from uploaded documents...',
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
    throw new Error('MCP chat functionality not yet implemented');
  };

  // Tool execution state
  const { isExecutingTool, currentTool, executeTool } = useToolExecution();

  // Toggle Chat2SQL mode
  const handleToggleChat2Sql = () => {
    setIsChat2SqlEnabled(prev => {
      const newValue = !prev;
      try {
        localStorage.setItem('chat2sql_mode_enabled', JSON.stringify(newValue));
      } catch (error) {
        console.error('Error saving Chat2SQL mode to localStorage:', error);
      }
      return newValue;
    });
  };

  // Listen for predictor messages from TrainingForm - Added
  useEffect(() => {
    const handlePredictorMessage = (event: CustomEvent) => {
      setMessages((prev) => [...prev, event.detail.message]);
    };

    window.addEventListener('addPredictorMessage', handlePredictorMessage as EventListener);
    return () => {
      window.removeEventListener('addPredictorMessage', handlePredictorMessage as EventListener);
    };
  }, [setMessages]);

  const titleInputRef = useRef<HTMLInputElement>(null);

  // Fetch sessions on component mount and ensure WebSocket connection
  useEffect(() => {
    fetchSessions();
    checkRagAvailability();
    forceCheckDocumentStatus(messages, setMessages, setIsLoading, setIsStreaming);
    
    if (activeSessionId) {
      if (contextRulesLoadedRef.current[activeSessionId]) {
        console.log('Context rules already loaded for session:', activeSessionId);
        return;
      }
      
      const contextRulesKey = `context_rules_${activeSessionId}`;
      try {
        let storedContextRules = sessionStorage.getItem(contextRulesKey) || localStorage.getItem(contextRulesKey);
        
        if (storedContextRules) {
          const parsedRules = JSON.parse(storedContextRules);
          
          if (parsedRules.hasContext && parsedRules.rules) {
            console.log('Found stored context rules for conversation:', activeSessionId);
            
            const systemContextMessage: ExtendedChatMessage = {
              id: `system-context-${Date.now()}`,
              role: 'system',
              content: `User context loaded: ${parsedRules.rules}`,
              timestamp: new Date(),
              isContextMessage: true
            };
            
            setMessages(prev => {
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
      const storedContext = checkForStoredContext(activeSessionId);
      if (storedContext) {
        console.log('Using stored context from localStorage:', storedContext);
      }
      
      const contextRulesKey = `context_rules_${activeSessionId}`;
      try {
        if (contextRulesLoadedRef.current[activeSessionId]) {
          console.log('Context rules already loaded for session:', activeSessionId);
          return;
        }
        
        let storedContextRules = sessionStorage.getItem(contextRulesKey) || localStorage.getItem(contextRulesKey);
        
        if (storedContextRules) {
          const parsedRules = JSON.parse(storedContextRules);
          
          if (parsedRules.hasContext && parsedRules.rules) {
            console.log('Found stored context rules for conversation:', activeSessionId);
            
            const systemContextMessage: ExtendedChatMessage = {
              id: `system-context-${Date.now()}`,
              role: 'system',
              content: `User context loaded: ${parsedRules.rules}`,
              timestamp: new Date(),
              isContextMessage: true
            };
            
            setTimeout(() => {
              setMessages(prev => {
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

      if (source === 'context_tool') {
        console.log('Skipping refresh from context tool to prevent UI issues');
        return;
      }

      if (conversationId && conversationId === activeSessionId) {
        console.log('Refreshing messages for conversation:', conversationId);
        fetchSessionMessages(conversationId);
      }
    };

    window.addEventListener('refreshMessages', handleRefreshMessages as EventListener);

    const handleAddSystemMessage = (event: CustomEvent<{ message: ExtendedChatMessage }>) => {
      const { message } = event.detail;
      console.log('Adding system message to conversation:', message);
      
      setMessages(prev => {
        const hasSimilarMessage = prev.some(msg => 
          msg.role === 'system' && 
          msg.content.includes('User context loaded:')
        );
        
        if (hasSimilarMessage) {
          console.log('Similar system message already exists, replacing it');
          const updatedMessages = prev.map(msg => 
            (msg.role === 'system' && msg.content.includes('User context loaded:'))
              ? message
              : msg
          );
          
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
    
    window.addEventListener('addSystemMessage', handleAddSystemMessage as EventListener);

    return () => {
      window.removeEventListener('refreshMessages', handleRefreshMessages as EventListener);
      window.removeEventListener('addSystemMessage', handleAddSystemMessage as EventListener);
    };
  }, [activeSessionId, fetchSessionMessages]);

  // Load predictor messages when session changes
  useEffect(() => {
    if (activeSessionId && isPredictorEnabled) {
      console.log('Loading predictor messages for session:', activeSessionId);
      const predictorMessages = loadPredictorMessages(activeSessionId);
      if (predictorMessages.length > 0) {
        console.log(`Found ${predictorMessages.length} predictor messages in localStorage`);
        // Merge predictor messages with existing messages, avoiding duplicates
        setMessages(prev => {
          const existingIds = new Set(prev.map(msg => msg.id));
          const newMessages = predictorMessages.filter(msg => !existingIds.has(msg.id));
          return [...prev, ...newMessages].sort((a, b) => a.timestamp.getTime() - b.timestamp.getTime());
        });
      }
    }
  }, [activeSessionId, isPredictorEnabled]);

  // WebSocket reconnection
  const { connected: wsConnected, reconnect: wsReconnect } = useWebSocket();
  useEffect(() => {
    const periodicCheckInterval = setInterval(() => {
      if (!ragNotificationShown) {
        console.log('Performing periodic document status check');
        forceCheckDocumentStatus(messages, setMessages, setIsLoading, setIsStreaming);
        checkRagAvailability();
      }

      if (!wsConnected) {
        console.log('WebSocket not connected during periodic check, attempting to reconnect...');
        wsReconnect();
      }
    }, 30000);

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
    if ((content.trim() === '' && !file) || isLoading || isUploading) return;

    // Handle Predictor messages - Added
    if (isPredictorEnabled) {
      if (content.toLowerCase().trim() === 'train') {
        setShowTrainingForm(true);
        return;
      }
    }

    // Handle Predictor messages
    if (meta?.predictor) {
      console.log('Handling Predictor response in Chatbot.tsx...', meta);
      
      if (meta.isServerResponse) {
        const aiMessage: ExtendedChatMessage = {
          id: meta.id,
          role: 'assistant',
          content: meta.error ? `Error: ${meta.error}` : meta.content,
          timestamp: new Date(meta.timestamp),
          predictor: true,
          predictions: meta.predictions,
          error: meta.error,
          showDownloadButton: meta.showDownloadButton
        };
        
        console.log('Predictor result message:', aiMessage);
        setMessages(prev => [...prev, aiMessage]);
        
        // Save predictor AI response to database and localStorage
        let sessionId = activeSessionId;
        if (!sessionId) {
          // Create a new session if none exists
          try {
            const newSession = await createNewSession('Predictor Session');
            sessionId = newSession.id;
            setActiveSessionId(sessionId);
            await fetchSessions(); // Refresh the sessions list
            console.log('Created new session for predictor:', sessionId);
          } catch (error) {
            console.error('Error creating new session for predictor:', error);
          }
        }
        
        if (sessionId) {
          try {
            await chatbotService.sendPredictorMessage(
              '', // Empty user message since this is AI response
              sessionId,
              meta.error ? `Error: ${meta.error}` : meta.content,
              {
                predictor: aiMessage.predictor,
                predictions: aiMessage.predictions,
                error: aiMessage.error,
                showDownloadButton: aiMessage.showDownloadButton,
                isServerResponse: aiMessage.isServerResponse
              }
            );
            console.log('Predictor AI message saved to database');
          } catch (error) {
            console.error('Error saving predictor AI message to database:', error);
          }
          savePredictorMessage(sessionId, aiMessage);
        }
        return;
      }
      
      if (meta.isUserCommand) {
        const userMessage: ExtendedChatMessage = {
          id: meta.id,
          role: 'user',
          content: content.trim(),
          timestamp: new Date(meta.timestamp),
          predictor: true,
          isUserCommand: true
        };

        console.log('User predictor command message:', userMessage);
        setMessages(prev => [...prev, userMessage]);
        
        // Save predictor user command to database and localStorage
        let sessionId = activeSessionId;
        if (!sessionId) {
          // Create a new session if none exists
          try {
            const newSession = await createNewSession('Predictor Session');
            sessionId = newSession.id;
            setActiveSessionId(sessionId);
            await fetchSessions(); // Refresh the sessions list
            console.log('Created new session for predictor:', sessionId);
          } catch (error) {
            console.error('Error creating new session for predictor:', error);
          }
        }
        
        if (sessionId) {
          try {
            await chatbotService.sendPredictorMessage(
              content.trim(),
              sessionId,
              '', // Empty response since this is user message
              {
                isUserCommand: userMessage.isUserCommand
              }
            );
            console.log('Predictor user command saved to database');
          } catch (error) {
            console.error('Error saving predictor user command to database:', error);
          }
          savePredictorMessage(sessionId, userMessage);
        }
        return;
      }

      return;
    }

    // Handle Chat2SQL messages
    if (meta?.chat2sql) {
      console.log('Handling Chat2SQL response in Chatbot.tsx...', meta);
      
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
        
        // Save Chat2SQL AI response to database
        let sessionId = activeSessionId;
        if (!sessionId) {
          // Create a new session if none exists
          try {
            const newSession = await createNewSession('Chat2SQL Session');
            sessionId = newSession.id;
            setActiveSessionId(sessionId);
            await fetchSessions(); // Refresh the sessions list
            console.log('Created new session for Chat2SQL:', sessionId);
          } catch (error) {
            console.error('Error creating new session for Chat2SQL:', error);
          }
        }
        
        if (sessionId) {
          try {
            await chatbotService.sendMessage(
              '', // Empty user message since this is AI response
              sessionId,
              meta.error ? `Error: ${meta.error}` : meta.content,
              false
            );
            console.log('Chat2SQL AI message saved to database');
          } catch (error) {
            console.error('Error saving Chat2SQL AI message to database:', error);
          }
        }
        return;
      }
      
      if (meta.isUserMessage) {
        const userMessage: ExtendedChatMessage = {
          id: `user-${Date.now()}`,
          role: 'user',
          content: content.trim(),
          timestamp: new Date(),
          isSqlQuery: true
        };

        console.log('User SQL query message:', userMessage);
        setMessages(prev => [...prev, userMessage]);

        // Save Chat2SQL user message to database
        let sessionId = activeSessionId;
        if (!sessionId) {
          // Create a new session if none exists
          try {
            const newSession = await createNewSession('Chat2SQL Session');
            sessionId = newSession.id;
            setActiveSessionId(sessionId);
            await fetchSessions(); // Refresh the sessions list
            console.log('Created new session for Chat2SQL:', sessionId);
          } catch (error) {
            console.error('Error creating new session for Chat2SQL:', error);
          }
        }
        
        if (sessionId) {
          try {
            await chatbotService.sendMessage(
              content.trim(),
              sessionId,
              '', // Empty response since this is user message
              false
            );
            console.log('Chat2SQL user message saved to database');
          } catch (error) {
            console.error('Error saving Chat2SQL user message to database:', error);
          }
        }

        return;
      }

      return;
    }

    // Special handling for read_context command
    if (content.trim().toLowerCase() === 'read_context') {
      console.log('Detected exact read_context command, triggering context tool directly');
      
      // Create user message for the command
      const userMessage: ExtendedChatMessage = {
        id: `user-${Date.now()}`,
        role: 'user',
        content: content.trim(),
        timestamp: new Date()
      };
      setMessages(prev => [...prev, userMessage]);
      
      // Create AI response message
      const aiMessage = createContextToolMessage();
      setMessages(prev => [...prev, aiMessage]);
      
      // Save both messages to database
      let sessionId = activeSessionId;
      if (!sessionId) {
        try {
          const newSession = await createNewSession('Context Session');
          sessionId = newSession.id;
          setActiveSessionId(sessionId);
          await fetchSessions();
          console.log('Created new session for context:', sessionId);
        } catch (error) {
          console.error('Error creating new session for context:', error);
        }
      }
      
      if (sessionId) {
        try {
          await chatbotService.sendMessage(
            content.trim(),
            sessionId,
            aiMessage.content,
            false
          );
          console.log('Context command and response saved to database');
        } catch (error) {
          console.error('Error saving context messages to database:', error);
        }
      }
      
      return;
    }
    
    // Handle MCP messages
    if (isMCPEnabled && !file && content.trim() !== '') {
      try {
        const userMessage: ExtendedChatMessage = {
          id: `user-${Date.now()}`,
          role: 'user',
          content: content.trim(),
          timestamp: new Date()
        };
        setMessages(prev => [...prev, userMessage]);

        // Save MCP user message to database
        let sessionId = activeSessionId;
        if (!sessionId) {
          // Create a new session if none exists
          try {
            const newSession = await createNewSession('MCP Session');
            sessionId = newSession.id;
            setActiveSessionId(sessionId);
            await fetchSessions(); // Refresh the sessions list
            console.log('Created new session for MCP:', sessionId);
          } catch (error) {
            console.error('Error creating new session for MCP:', error);
          }
        }
        
        if (sessionId) {
          try {
            await chatbotService.sendMessage(
              content.trim(),
              sessionId,
              '', // Empty response since this is user message
              false
            );
            console.log('MCP user message saved to database');
          } catch (error) {
            console.error('Error saving MCP user message to database:', error);
          }
        }

        await handleMCPChatMessage(
          content,
          messages,
          activeSessionId,
          { id: selectedModelId },
          streamedContentRef,
          abortFunctionRef,
          setMessages,
          setIsStreaming,
          setIsLoading,
          executeTool,
          chatbotService,
          fetchSessions
        );
        return;
      } catch (error: any) {
        console.error('Error using MCP chat mode:', error);
        const errorMessage: ExtendedChatMessage = {
          id: `error-${Date.now()}`,
          role: 'assistant',
          content: `Error: ${error.message}. Falling back to normal chat.`,
          timestamp: new Date()
        };
        setMessages(prev => [...prev, errorMessage]);
        
        // Save MCP error message to database
        if (activeSessionId) {
          try {
            await chatbotService.sendMessage(
              '',
              activeSessionId,
              errorMessage.content,
              false
            );
            console.log('MCP error message saved to database');
          } catch (dbError) {
            console.error('Error saving MCP error message to database:', dbError);
          }
        }
      }
    }

    // Skip regular chat if predictor mode is enabled (predictor handles its own messages)
    if (isPredictorEnabled && !meta) {
      console.log('Predictor mode enabled but no meta data - skipping regular chat');
      return;
    }

    // Regular chat message handling
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
    
    if (result?.newSessionId && (!activeSessionId || activeSessionId !== result.newSessionId)) {
      setActiveSessionId(result.newSessionId);
    }
  };

  // Toggle MCP mode
  const handleToggleMCP = () => {
    if (!isMCPEnabled && !isMCPConnected) {
      setShowServerSelector(true);
    } else {
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
          <div className="flex items-center">
            <h2
              className="text-base md:text-lg font-semibold truncate max-w-[200px] md:max-w-none"
              style={{ color: 'var(--color-text)' }}
            >
              {activeSessionId ? sessionTitle : 'New Chat'}
            </h2>
          </div>
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
              width: window.innerWidth < 768 ? '100%' : '320px'
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
              onEditSession={editSession}
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
            onEditSession={editSession}
            onToggleGroup={toggleGroup}
            onToggleCollapse={toggleSidebar}
          />
        )}

        <div
          className={`absolute inset-0 transition-all duration-300 ease-in-out flex flex-col`}
          style={{
            backgroundColor: 'var(--color-bg)',
            marginLeft: showSidebar ? (window.innerWidth < 768 ? '0' : '320px') : '0'
          }}
        >
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

          {/* Training Form (shown when triggered) - Added */}
          {showTrainingForm && (
            <div className="px-4 py-2">
              <TrainingForm
                onTrainingComplete={() => setShowTrainingForm(false)}
              />
            </div>
          )}

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
              isChat2SqlEnabled={isChat2SqlEnabled}
              onToggleChat2Sql={handleToggleChat2Sql}
              // Added predictor props
              isPredictorEnabled={isPredictorEnabled}
              onTogglePredictor={() => {
                setIsPredictorEnabled((prev) => {
                  const newValue = !prev;
                  try {
                    localStorage.setItem('predictor_mode_enabled', JSON.stringify(newValue));
                  } catch (error) {
                    console.error('Error saving predictor mode to localStorage:', error);
                  }
                  
                  if (newValue) {
                    // Send welcome message when predictor is activated
                    const welcomeMessage = {
                      id: `predictor-welcome-${Date.now()}`,
                      role: 'assistant' as const,
                      content: `🤖 **Predictor Mode Activated**

I'm ready to help you train models and make predictions! 

To get started, you can:
- Type: "To obtain the results, I need to train and then make a prediction using the 3 csvs"
- Or simply type: "train" to start training
- Or type: "predict" to make predictions (after training)

What would you like to do?`,
                      timestamp: new Date(),
                      predictor: true,
                      isServerResponse: true,
                    };
                    
                    setMessages(prev => [...prev, welcomeMessage]);
                  }
                  setShowTrainingForm(false);
                  return newValue;
                });
              }}
            />

            {isEmpty && (
              <div className="flex justify-center mt-12">
                <div className="flex flex-wrap justify-center gap-2">
                  <button
                    onClick={() => createNewSession()}
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