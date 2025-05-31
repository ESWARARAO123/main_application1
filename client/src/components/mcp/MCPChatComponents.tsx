import React from 'react';
import { ChatMessage as ChatMessageType } from '../../types';
import { RagSource } from '../../services/ragChatService';
import { containsReadContextToolCall } from '../../utils/toolParser';
import { useMCP } from '../../contexts/MCPContext';
import { useMCPAgent } from '../../contexts/MCPAgentContext';
import { useWebSocket } from '../../contexts/WebSocketContext';
import { enhancePromptWithMCPCapabilities } from '../../prompts/mcpSystemPrompt';
import mcpChatService from '../../services/mcpChatService';
import { applyContextToPrompt } from '../../utils/contextUtils';

// Define a custom message type that includes all needed properties
export interface MCPExtendedChatMessageType {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: Date;
  isStreaming?: boolean;
  fileAttachment?: {
    name: string;
    type: string;
    size: number;
    url?: string;
    documentId?: string;
    status?: string;
    processingError?: string;
  };
  isProcessingFile?: boolean;
  isProcessingOnly?: boolean;
  isLoadingOnly?: boolean;
  documentId?: string;
  documentStatus?: string;
  sources?: RagSource[];
  useRag?: boolean;
  isContextTool?: boolean;
  conversationId?: string;
}

/**
 * Handles MCP-specific chat functionality
 * This component encapsulates MCP-related logic that was previously in the Chatbot component
 */
export const useMCPChat = () => {
  const {
    isConnected: isMCPConnected,
    isMCPEnabled,
    toggleMCPEnabled,
    showServerSelector,
    setShowServerSelector,
    selectServer,
    getClientId,
    defaultServer
  } = useMCP();

  // MCP Agent state
  const { isAgentEnabled, toggleAgent } = useMCPAgent();

  // Get WebSocket context
  const { connected: wsConnected, reconnect: wsReconnect } = useWebSocket();

  /**
   * Creates a context tool message
   * @returns A message object with the context tool
   */
  const createContextToolMessage = (): MCPExtendedChatMessageType => {
    // Check if there's already a context tool message in the DOM
    // This helps prevent duplicate context tools
    const existingContextTools = document.querySelectorAll('[data-context-tool="true"]');
    if (existingContextTools.length > 0) {
      console.log('Context tool already exists in the DOM, not creating a new one');
      // Return the existing message ID if possible
      const existingId = existingContextTools[0].getAttribute('data-message-id');
      if (existingId) {
        return {
          id: existingId,
          role: 'assistant',
          content: 'I can read your context rules to better understand your preferences\n\n```\nread_context\n```',
          timestamp: new Date(),
          isContextTool: true
        };
      }
    }

    const aiMessageId = `ai-${Date.now()}`;
    return {
      id: aiMessageId,
      role: 'assistant',
      content: 'I can read your context rules to better understand your preferences\n\n```\nread_context\n```',
      timestamp: new Date(),
      isContextTool: true // Mark this as a context tool message
    };
  };

  /**
   * Handles MCP chat message sending
   * @param content Message content
   * @param messages Current messages array
   * @param activeSessionId Current session ID
   * @param selectedModel Selected model details
   * @param streamedContentRef Reference to store streamed content
   * @param abortFunctionRef Reference to store abort function
   * @param setMessages Function to update messages state
   * @param setIsStreaming Function to update streaming state
   * @param setIsLoading Function to update loading state
   * @param executeTool Function to execute tools
   * @param chatbotService Chatbot service for database operations
   * @param fetchSessions Function to refresh sessions
   * @returns Promise that resolves when message is sent
   */
  const handleMCPChatMessage = async (
    content: string,
    messages: MCPExtendedChatMessageType[],
    activeSessionId: string | null,
    selectedModel: any,
    streamedContentRef: React.MutableRefObject<{[key: string]: string}>,
    abortFunctionRef: React.MutableRefObject<(() => void) | null>,
    setMessages: React.Dispatch<React.SetStateAction<MCPExtendedChatMessageType[]>>,
    setIsStreaming: React.Dispatch<React.SetStateAction<boolean>>,
    setIsLoading: React.Dispatch<React.SetStateAction<boolean>>,
    executeTool: (content: string) => Promise<string>,
    chatbotService: any,
    fetchSessions: () => Promise<void>
  ) => {
    try {
      // Get client ID from MCP context
      const mcpClientId = getClientId();

      if (!mcpClientId) {
        console.error('MCP is enabled but no client ID available');
        // Show an error message in the chat
        const errorMessage: MCPExtendedChatMessageType = {
          id: `error-${Date.now()}`,
          role: 'assistant',
          content: "Error: MCP is enabled but not properly connected. Please try reconnecting to the MCP server.",
          timestamp: new Date()
        };
        setMessages(prev => [...prev, errorMessage]);
        return;
      }

      // Get default server info
      if (!defaultServer) {
        console.error('MCP is enabled but no default server configured');
        // Show an error message in the chat
        const errorMessage: MCPExtendedChatMessageType = {
          id: `error-${Date.now()}`,
          role: 'assistant',
          content: "Error: No MCP server configured. Please select a server from the settings.",
          timestamp: new Date()
        };
        setMessages(prev => [...prev, errorMessage]);
        return;
      }

      console.log('Using regular chat flow but with MCP client ID:', mcpClientId);

      // Create a temporary AI message for streaming
      const aiMessageId = `ai-${Date.now()}`;
      const aiMessage: MCPExtendedChatMessageType = {
        id: aiMessageId,
        role: 'assistant',
        content: '',
        timestamp: new Date(),
        isStreaming: true
      };

      // Add the message to the UI immediately to show streaming
      setMessages(prev => [...prev, aiMessage]);
      setIsStreaming(true);
      setIsLoading(true);

      // If we're using MCP, just use it as a connection - but use the regular chat flow
      const conversationHistory = messages
        .filter(msg => msg.role !== 'system') // Filter out system messages
        .map(msg => ({
          role: msg.role as 'user' | 'assistant',
          content: msg.content
        }));

      // Add the current message
      conversationHistory.push({ role: 'user', content: content.trim() });

      try {
        // Base system prompt
        let systemPromptContent = 'You are a helpful AI assistant. Answer the user\'s questions accurately and concisely.';

        // Apply context to the system prompt using our utility function with shell command capabilities
        systemPromptContent = applyContextToPrompt(systemPromptContent, messages, {
          enableShellCommands: true
        });

        // Log the system prompt with context for debugging
        console.log('System prompt with context applied:', systemPromptContent);

        // Add MCP capabilities to the system prompt
        const systemMessage = {
          role: 'system',
          content: enhancePromptWithMCPCapabilities(systemPromptContent)
        };

        // Add system message to the beginning of the conversation
        const enhancedConversation = [
          systemMessage as { role: 'user' | 'assistant' | 'system'; content: string },
          ...conversationHistory as { role: 'user' | 'assistant' | 'system'; content: string }[]
        ];

        // Send to Ollama through mcpChatService but treat it like regular chat
        const mcpRequest = {
          modelId: selectedModel.ollama_model_id,
          messages: enhancedConversation,
          mcpClientId: mcpClientId,
          mcpServer: {
            host: defaultServer.mcp_host,
            port: defaultServer.mcp_port
          },
          options: { stream: true }
        };

        console.log('Sending MCP chat request with model:', selectedModel.ollama_model_id);

        // Use the abort controller to allow stopping generation
        abortFunctionRef.current = await mcpChatService.streamChatCompletion(
          mcpRequest,
          (chunk) => {
            // Debug logging to track chunks
            console.log('Raw chunk:', JSON.stringify(chunk));

            const newContent = chunk.choices?.[0]?.delta?.content || chunk.choices?.[0]?.message?.content || '';
            if (newContent) {
              // Update the ref with the accumulated content
              streamedContentRef.current[aiMessageId] = (streamedContentRef.current[aiMessageId] || '') + newContent;

              // Check for tool calls in the accumulated content
              const accumulatedContent = streamedContentRef.current[aiMessageId];
              if (containsReadContextToolCall(accumulatedContent)) {
                // Process the tool call
                executeTool(accumulatedContent).then(processedContent => {
                  // Update the message with the processed content
                  streamedContentRef.current[aiMessageId] = processedContent;

                  // Update the UI
                  setMessages(prev => prev.map(msg =>
                    msg.id === aiMessageId ? { ...msg, content: processedContent } : msg
                  ));
                });
              } else {
                // Update the UI with the regular content
                setMessages(prev => prev.map(msg =>
                  msg.id === aiMessageId ? { ...msg, content: accumulatedContent } : msg
                ));
              }
            }
          },
          async () => {
            // Add a small delay to ensure all content is accumulated
            await new Promise(resolve => setTimeout(resolve, 500));

            try {
              // Get final content after the delay
              const finalContentAfterDelay = streamedContentRef.current[aiMessageId] || '';

              console.log('Sending message to database:', {
                messageLength: content.trim().length,
                responseLength: finalContentAfterDelay.length,
                sessionId: activeSessionId || undefined
              });

              // Save the message to the database
              const dbResponse = await chatbotService.sendMessage(
                content.trim(),
                activeSessionId || undefined,
                finalContentAfterDelay
              );

              console.log('Message saved successfully:', dbResponse);

              if (!activeSessionId || activeSessionId !== dbResponse.sessionId) {
                await fetchSessions();
              }

              // Update message with database ID
              setMessages(prev => prev.map(msg =>
                msg.id === aiMessageId ? {
                  ...msg,
                  id: dbResponse.id,
                  isStreaming: false,
                  content: finalContentAfterDelay
                } : msg
              ));

              // Clean up the ref
              delete streamedContentRef.current[aiMessageId];
            } catch (dbError) {
              console.error('Error saving message to database:', dbError);
              // Still mark the message as not streaming even if saving fails
              setMessages(prev => prev.map(msg =>
                msg.id === aiMessageId ? { ...msg, isStreaming: false } : msg
              ));

              // Clean up the ref even on error
              delete streamedContentRef.current[aiMessageId];
            }

            setIsLoading(false);
            setIsStreaming(false);
            abortFunctionRef.current = null;
          },
          (error) => {
            console.error('MCP streaming error:', error);

            // Show the error in chat
            setMessages(prev => {
              const withoutLoadingMessage = prev.filter(msg => msg.id !== aiMessageId);
              return [
                ...withoutLoadingMessage,
                {
                  id: `error-${Date.now()}`,
                  role: 'assistant',
                  content: `Error: ${error.message}. Falling back to normal chat.`,
                  timestamp: new Date()
                }
              ];
            });

            // Clean up the ref on error
            delete streamedContentRef.current[aiMessageId];
            setIsLoading(false);
            setIsStreaming(false);
            abortFunctionRef.current = null;
          }
        );
      } catch (modelError: any) {
        console.error('Error getting model details:', modelError);

        // Show error and cancel streaming
        setMessages(prev => {
          const withoutLoadingMessage = prev.filter(msg => msg.id !== aiMessageId);
          return [
            ...withoutLoadingMessage,
            {
              id: `error-${Date.now()}`,
              role: 'assistant',
              content: `Error: ${modelError.message}. Please check your model configuration.`,
              timestamp: new Date()
            }
          ];
        });

        setIsLoading(false);
        setIsStreaming(false);
      }
    } catch (error: any) {
      console.error('Error using MCP chat mode:', error);

      // Add an error message to the chat
      const errorMessage: MCPExtendedChatMessageType = {
        id: `error-${Date.now()}`,
        role: 'assistant',
        content: `Error: ${error.message}. Falling back to normal chat.`,
        timestamp: new Date()
      };

      setMessages(prev => [...prev, errorMessage]);
    }
  };

  return {
    isMCPConnected,
    isMCPEnabled,
    toggleMCPEnabled,
    showServerSelector,
    setShowServerSelector,
    selectServer,
    getClientId,
    defaultServer,
    isAgentEnabled,
    toggleAgent,
    wsConnected,
    wsReconnect,
    createContextToolMessage,
    handleMCPChatMessage
  };
};
