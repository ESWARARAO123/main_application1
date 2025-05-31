import { useState, useRef } from 'react';
import { ExtendedChatMessage } from '../types';
import { chatbotService } from '../services/chatbotService';
import { aiChatService, StreamChunk } from '../services/aiChatService';
import { ragChatService } from '../services/ragChatService';
import { getActiveOllamaModels } from '../services/ollamaService';
import { applyContextToPrompt } from '../utils/contextUtils';

/**
 * Hook for managing chat message sending and streaming
 */
export const useChatMessaging = () => {
  // Message state
  const [isLoading, setIsLoading] = useState(false);
  const [isStreaming, setIsStreaming] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);

  // Refs for streaming
  const streamedContentRef = useRef<{[key: string]: string}>({});
  const abortFunctionRef = useRef<(() => void) | null>(null);

  // Helper function to get the selected model details
  const getSelectedModelDetails = async (selectedModelId: string | undefined) => {
    try {
      if (!selectedModelId) {
        console.warn('No model selected, using default');

        // Try to get active models to find a default
        const modelsResponse = await getActiveOllamaModels();
        if (modelsResponse && modelsResponse.length > 0) {
          // Find the first active model
          const firstActiveModel = modelsResponse.find(model => model.is_active);
          if (firstActiveModel) {
            console.log('Using first active model as default:', firstActiveModel.name);
            return firstActiveModel;
          }
        }

        // If we couldn't find a model, create a placeholder with a default ID
        return {
          id: 'default',
          name: 'Default Model',
          model_id: 'default',
          ollama_model_id: process.env.DEFAULT_OLLAMA_MODEL || 'llama3',
          is_active: true,
          description: 'Default Model'
        };
      }

      const modelsResponse = await getActiveOllamaModels();
      const selectedModel = modelsResponse.find(model => model.id === selectedModelId);

      if (!selectedModel) {
        console.warn('Selected model not found, using default');

        // If the selected model is not found, use the first active model
        const firstActiveModel = modelsResponse.find(model => model.is_active);
        if (firstActiveModel) {
          return firstActiveModel;
        }

        // If no active models, create a placeholder
        return {
          id: 'default',
          name: 'Default Model',
          model_id: 'default',
          ollama_model_id: process.env.DEFAULT_OLLAMA_MODEL || 'llama3',
          is_active: true,
          description: 'Default Model'
        };
      }

      return selectedModel;
    } catch (error) {
      console.error('Error getting model details:', error);

      // Return a fallback model in case of error
      return {
        id: 'fallback',
        name: 'Fallback Model',
        model_id: 'fallback',
        ollama_model_id: process.env.DEFAULT_OLLAMA_MODEL || 'llama3',
        is_active: true,
        description: 'Fallback Model'
      };
    }
  };

  // Send a chat message
  const sendChatMessage = async (
    content: string,
    file: File | undefined,
    messages: ExtendedChatMessage[],
    activeSessionId: string | null,
    selectedModelId: string | undefined,
    isRagAvailable: boolean,
    isRagEnabled: boolean,
    setMessages: React.Dispatch<React.SetStateAction<ExtendedChatMessage[]>>,
    fetchSessions: () => Promise<void>
  ) => {
    // Allow sending if there's text or a file
    if ((content.trim() === '' && !file) || isLoading || isUploading) return;

    const tempId = `temp-${Date.now()}`;
    
    // For file uploads, create a descriptive message if content is empty
    const displayContent = (file && content.trim() === '')
      ? `I'm uploading ${file.name} for analysis.`
      : content.trim();

    const userMessage: ExtendedChatMessage = {
      id: tempId,
      role: 'user',
      content: displayContent,
      timestamp: new Date(),
      // Add file metadata if a file is provided
      fileAttachment: file ? {
        name: file.name,
        type: file.type,
        size: file.size
      } : undefined
    };

    setMessages(prev => [...prev, userMessage]);
    
    setIsLoading(true);

    try {
      if (selectedModelId) {
        const selectedModel = await getSelectedModelDetails(selectedModelId);

        // Check if we should use RAG for this message
        const shouldUseRag = isRagAvailable && isRagEnabled;

        // Create a temporary AI message for streaming
        const aiMessageId = `ai-${Date.now()}`;
        const aiMessage: ExtendedChatMessage = {
          id: aiMessageId,
          role: 'assistant',
          content: '',
          timestamp: new Date(),
          isStreaming: true, // Mark as streaming to show the loading indicator
          useRag: shouldUseRag // Mark if we're using RAG
        };

        // Add the message to the UI immediately to show streaming
        setMessages(prev => [...prev, aiMessage]);
        setIsStreaming(true);

        // If RAG is available and enabled, use it
        if (shouldUseRag) {
          try {
            console.log('Using RAG for this message');

            // Call the RAG service
            const ragResponse = await ragChatService.sendRagChatMessage({
              model: selectedModel.ollama_model_id,
              message: content.trim(),
              sessionId: activeSessionId || undefined
            });

            // Update the message with the RAG response
            setMessages(prev => prev.map(msg =>
              msg.id === aiMessageId ? {
                ...msg,
                content: ragResponse.content,
                sources: ragResponse.sources,
                isStreaming: false
              } : msg
            ));

            // Save the message to the database
            const dbResponse = await chatbotService.sendMessage(
              content.trim(),
              activeSessionId || undefined,
              ragResponse.content
            );

            // Update the activeSessionId and fetch sessions if needed
            if (!activeSessionId || activeSessionId !== dbResponse.sessionId) {
              // We will return the new session ID to be set in the calling component
              const newSessionId = dbResponse.sessionId;
              await fetchSessions();
              
              // Return the new session ID
              return { success: true, newSessionId };
            }

            setIsLoading(false);
            setIsStreaming(false);
            return { success: true };
          } catch (ragError) {
            console.error('Error using RAG:', ragError);
            // Fall back to regular chat if RAG fails
            console.log('Falling back to regular chat');

            // Update the message to indicate RAG failed
            setMessages(prev => prev.map(msg =>
              msg.id === aiMessageId ? {
                ...msg,
                content: 'RAG processing failed, falling back to regular chat...',
                useRag: false
              } : msg
            ));
          }
        }

        // If we get here, either RAG is not available/enabled or it failed
        // Use regular chat with conversation history

        // Base system prompt
        let systemPromptContent = 'You are a helpful AI assistant. Answer the user\'s questions accurately and concisely.';

        // Apply context to the system prompt using our utility function with shell command capabilities
        systemPromptContent = applyContextToPrompt(systemPromptContent, messages, {
          enableShellCommands: true
        });

        // Log the system prompt with context for debugging
        console.log('System prompt with context applied:', systemPromptContent);

        // Create conversation history with user and assistant messages
        const conversationHistory = messages
          .filter(msg => msg.role !== 'system' && !msg.content.startsWith('Context Loaded')) // Filter out system messages and context messages
          .map(msg => ({
            role: msg.role as 'user' | 'assistant',
            content: msg.content
          }));

        // Add the current message to conversation history
        conversationHistory.push({ role: 'user', content: content.trim() });

        // Create the full conversation with system message
        const fullConversation = [
          { role: 'system' as 'user' | 'assistant' | 'system', content: systemPromptContent },
          ...conversationHistory
        ];

        // Set isLoading and isStreaming to true to indicate we're waiting for a response
        setIsStreaming(true);

        // Store the abort function so we can call it if the user clicks the stop button
        abortFunctionRef.current = await aiChatService.streamChatCompletion(
          {
            modelId: selectedModel.ollama_model_id,
            messages: fullConversation,
            options: { stream: true }
          },
          (chunk: StreamChunk) => {
            const newContent = chunk.choices?.[0]?.delta?.content || chunk.choices?.[0]?.message?.content || '';
            if (newContent) {
              // Update the ref with the accumulated content
              streamedContentRef.current[aiMessageId] = (streamedContentRef.current[aiMessageId] || '') + newContent;

              // Update the UI
              setMessages(prev => prev.map(msg =>
                msg.id === aiMessageId ? { ...msg, content: streamedContentRef.current[aiMessageId] } : msg
              ));
            }
          },
          async () => {
            console.log('Preparing to get final AI message content');
            console.log('Checking content length');

            // Add a small delay to ensure all content is accumulated
            await new Promise(resolve => setTimeout(resolve, 500));

            try {
              // Double-check the final content after the delay
              const finalContentAfterDelay = streamedContentRef.current[aiMessageId] || '';
              console.log('Final content after delay:', finalContentAfterDelay.length);

              // Save the message to the database
              const dbResponse = await chatbotService.sendMessage(
                content.trim(),
                activeSessionId || undefined,
                finalContentAfterDelay
              );
              console.log('Database response:', dbResponse);

              // Update the activeSessionId and fetch sessions if needed
              if (!activeSessionId || activeSessionId !== dbResponse.sessionId) {
                // We will return the new session ID to be set in the calling component
                const newSessionId = dbResponse.sessionId;
                await fetchSessions();
                
                // Update the existing message with the database ID
                setMessages(prev => {
                  return prev.map(msg =>
                    msg.id === aiMessageId ? {
                      ...msg,
                      id: dbResponse.id,
                      isStreaming: false,
                      content: finalContentAfterDelay
                    } : msg
                  );
                });
                
                // Clean up the ref
                delete streamedContentRef.current[aiMessageId];
                
                // Return the new session ID
                setIsLoading(false);
                setIsStreaming(false);
                abortFunctionRef.current = null;
                return { success: true, newSessionId };
              }

              // Update the existing message with the database ID
              setMessages(prev => {
                console.log('Updating message with DB ID:', dbResponse.id);
                return prev.map(msg =>
                  msg.id === aiMessageId ? {
                    ...msg,
                    id: dbResponse.id,
                    isStreaming: false,
                    content: finalContentAfterDelay
                  } : msg
                );
              });

              // Clean up the ref
              delete streamedContentRef.current[aiMessageId];
            } catch (error) {
              console.error('Error saving message to database:', error);
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
            console.error('Streaming error:', error);
            setMessages(prev => prev.filter(msg => msg.id !== aiMessageId));
            // Clean up the ref on error
            delete streamedContentRef.current[aiMessageId];
            setIsLoading(false);
            setIsStreaming(false);
            abortFunctionRef.current = null;
          }
        );
      } else {
        // Fallback for when no model is selected
        const response = await chatbotService.sendMessage(userMessage.content, activeSessionId || undefined);
        
        // Return the new session ID if needed
        const newSessionId = response.sessionId;
        
        setMessages(prev => {
          const filteredMessages = prev.filter(m => m.id !== tempId);
          return [
            ...filteredMessages,
            { ...userMessage, id: `user-${Date.now()}` },
            { id: response.id, role: 'assistant', content: response.content, timestamp: new Date() }
          ];
        });
        
        await fetchSessions();
        setIsLoading(false);
        
        return { success: true, newSessionId };
      }
      
      return { success: true };
    } catch (error) {
      console.error('Error sending message:', error);
      setMessages(prev => prev.filter(m => m.id !== tempId));
      setIsLoading(false);
      setIsStreaming(false);
      return { success: false, error };
    }
  };

  // Stop message generation
  const stopGeneration = () => {
    if (abortFunctionRef.current) {
      abortFunctionRef.current();
      // The abort function will call onComplete which will reset isStreaming and abortFunctionRef
    }
  };

  return {
    // State
    isLoading,
    isStreaming,
    isUploading,
    uploadProgress,
    
    // Setters
    setIsLoading,
    setIsStreaming,
    setIsUploading,
    setUploadProgress,
    
    // References
    streamedContentRef,
    abortFunctionRef,
    
    // Actions
    sendChatMessage,
    stopGeneration,
    getSelectedModelDetails
  };
}; 