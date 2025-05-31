import { useState, useRef, useEffect } from 'react';
import { ExtendedChatMessage } from '../types';
import { chatbotService } from '../services/chatbotService';
import { ragChatService } from '../services/ragChatService';

/**
 * Hook for managing document processing and context functionality
 */
export const useContextHandling = (activeSessionId: string | null) => {
  // RAG state
  const [isRagAvailable, setIsRagAvailable] = useState<boolean>(false);
  const [isRagEnabled, setIsRagEnabled] = useState<boolean>(() => {
    // Get from localStorage or default to true
    const savedPreference = localStorage.getItem('ragEnabled');
    return savedPreference !== null ? savedPreference === 'true' : true;
  });
  // Track if we've already shown a RAG notification for the current document
  const [ragNotificationShown, setRagNotificationShown] = useState<boolean>(false);
  
  // Store the last time we checked RAG availability
  const lastRagCheckRef = useRef<number>(0);

  // Check if there's stored context for a session
  const checkForStoredContext = (sessionId: string) => {
    try {
      const storedContext = localStorage.getItem(`context_${sessionId}`);
      if (storedContext) {
        const parsedContext = JSON.parse(storedContext);
        if (parsedContext.hasContext) {
          console.log('Found stored context for conversation:', sessionId);
          return parsedContext;
        }
      }
    } catch (error) {
      console.error('Error checking for stored context:', error);
    }
    return null;
  };

  // Check if RAG is available with debounce to prevent excessive checks
  const checkRagAvailability = async () => {
    // Implement debounce - only check if it's been at least 5 seconds since the last check
    const now = Date.now();
    if (now - lastRagCheckRef.current < 5000) {
      console.log(`Skipping RAG availability check - last check was ${(now - lastRagCheckRef.current) / 1000}s ago`);
      return isRagAvailable; // Return current state without checking
    }

    // Update the last check timestamp
    lastRagCheckRef.current = now;

    try {
      const available = await ragChatService.isRagAvailable();
      console.log(`RAG availability checked: ${available ? 'Available' : 'Not available'} at ${new Date().toISOString()}`);

      // Update state after checking
      if (available !== isRagAvailable) {
        setIsRagAvailable(available);
      }

      return available;
    } catch (error) {
      console.error('Error checking RAG availability:', error);
      setIsRagAvailable(false);
      return false;
    }
  };

  // Function to force check document status and update UI
  const forceCheckDocumentStatus = async (
    messages: ExtendedChatMessage[],
    setMessages: React.Dispatch<React.SetStateAction<ExtendedChatMessage[]>>,
    setIsLoading: React.Dispatch<React.SetStateAction<boolean>>,
    setIsStreaming: React.Dispatch<React.SetStateAction<boolean>>
  ) => {
    // Find any processing messages
    const processingMessages = messages.filter(msg => msg.isProcessingFile && msg.documentId);

    if (processingMessages.length > 0) {
      console.log('Found processing messages, checking status:', processingMessages.length);

      // Check each document status
      for (const msg of processingMessages) {
        if (msg.documentId) {
          try {
            const statusResponse = await chatbotService.getDocumentStatus(msg.documentId);
            console.log(`Force checking document ${msg.documentId} status:`, statusResponse);

            // If document is processed, update UI
            if (statusResponse.status === 'PROCESSED') {
              console.log('Document is processed, updating UI');

              // Remove processing message
              setMessages(prev => prev.filter(m => m.id !== msg.id));

              // Use a more comprehensive check for success messages
              const checkForSuccessMessages = (msgs: ExtendedChatMessage[]) => {
                return msgs.some(msg =>
                  msg.role === 'assistant' &&
                  (msg.content.includes("Your document has been fully processed") ||
                   msg.content.includes("Your document has been processed"))
                );
              };

              // Check if we already have a success message before adding a new one
              setMessages(prev => {
                // Check if we already have a success message to avoid duplicates
                if (checkForSuccessMessages(prev)) {
                  console.log('Success message already exists, not adding another one');
                  return prev;
                }

                // Add success message
                const successMessage: ExtendedChatMessage = {
                  id: `system-success-${Date.now()}`,
                  role: 'assistant',
                  content: "Your document has been fully processed and is ready for questions! You can now ask me anything about the content, and I'll use the document to provide accurate answers.",
                  timestamp: new Date()
                };

                return [...prev, successMessage];
              });

              // Mark notification as shown
              setRagNotificationShown(true);

              // Check RAG availability
              await checkRagAvailability();

              // Reset loading states
              setIsLoading(false);
              setIsStreaming(false);

              // Enable RAG mode automatically
              setIsRagEnabled(true);
              localStorage.setItem('ragEnabled', 'true');
            } else if (statusResponse.status === 'ERROR') {
              // Handle error
              console.log('Document processing error, updating UI');

              // Remove processing message
              setMessages(prev => prev.filter(m => m.id !== msg.id));

              // Add error message
              const errorMessage: ExtendedChatMessage = {
                id: `system-error-${Date.now()}`,
                role: 'assistant',
                content: "I encountered an error processing your document. " +
                         (statusResponse.error || "Please try uploading it again."),
                timestamp: new Date()
              };
              setMessages(prev => [...prev, errorMessage]);

              // Reset loading states
              setIsLoading(false);
              setIsStreaming(false);
            } else {
              // Check how long the message has been processing
              const messageTime = new Date(msg.timestamp).getTime();
              const currentTime = new Date().getTime();
              const processingTime = currentTime - messageTime;

              // If processing for more than 2 minutes, force reset UI
              if (processingTime > 120000) { // 2 minutes
                console.log('Document processing timeout, force resetting UI');

                // Remove processing message
                setMessages(prev => prev.filter(m => m.id !== msg.id));

                // Add timeout message
                const timeoutMessage: ExtendedChatMessage = {
                  id: `system-timeout-${Date.now()}`,
                  role: 'assistant',
                  content: "I've received your file, but the processing is taking longer than expected. " +
                           "I'll continue processing it in the background, and you can ask questions about it later.",
                  timestamp: new Date()
                };
                setMessages(prev => [...prev, timeoutMessage]);

                // Reset loading states
                setIsLoading(false);
                setIsStreaming(false);
              }
            }
          } catch (error) {
            console.error('Error force checking document status:', error);
          }
        }
      }
    } else if (messages.some(msg => msg.isStreaming)) {
      // If no processing messages but still loading, check if we should reset
      const loadingStartTime = messages.find(msg => msg.isStreaming)?.timestamp;

      if (loadingStartTime) {
        const messageTime = new Date(loadingStartTime).getTime();
        const currentTime = new Date().getTime();
        const loadingTime = currentTime - messageTime;

        // If loading for more than 1 minute, force reset UI
        if (loadingTime > 60000) { // 1 minute
          console.log('Loading timeout, force resetting UI');

          // Reset loading states
          setIsLoading(false);
          setIsStreaming(false);
        }
      }
    }
  };

  // Toggle RAG mode
  const toggleRagMode = () => {
    setIsRagEnabled(prev => {
      const newValue = !prev;
      localStorage.setItem('ragEnabled', String(newValue));
      return newValue;
    });
  };
  
  // Handle RAG availability notification
  const showRagAvailableNotification = (
    setMessages: React.Dispatch<React.SetStateAction<ExtendedChatMessage[]>>,
    setIsLoading: React.Dispatch<React.SetStateAction<boolean>>,
    setIsStreaming: React.Dispatch<React.SetStateAction<boolean>>
  ) => {
    // Use a function to check for success messages to ensure we have the latest state
    const checkForSuccessMessages = (msgs: ExtendedChatMessage[]) => {
      return msgs.some(msg =>
        msg.role === 'assistant' &&
        (msg.content.includes("Your document has been fully processed") ||
         msg.content.includes("Your document has been processed"))
      );
    };

    // Get the current messages directly from state to ensure we have the latest
    setMessages(prev => {
      // First, completely remove ALL error messages and loading indicators
      const filteredMessages = prev.filter(msg =>
        // Remove error messages
        !(msg.role === 'assistant' && msg.content.includes("Sorry, there was an error")) &&
        // Remove loading indicators
        !msg.isProcessingFile
      );

      // Check if we already have a success message
      if (checkForSuccessMessages(filteredMessages)) {
        console.log('Success message already exists, not adding another one');
        return filteredMessages;
      }

      // Add a system message to notify the user
      const ragAvailableMessage: ExtendedChatMessage = {
        id: `system-rag-available-${Date.now()}`,
        role: 'assistant',
        content: "Your document has been fully processed and is ready for questions! You can now ask me anything about the content, and I'll use the document to provide accurate answers.",
        timestamp: new Date()
      };

      // Add the success message to the filtered messages
      return [...filteredMessages, ragAvailableMessage];
    });

    // Mark that we've shown the notification
    setRagNotificationShown(true);

    // Reset loading and streaming states
    setIsLoading(false);
    setIsStreaming(false);

    // Enable RAG mode automatically
    setIsRagEnabled(true);
    localStorage.setItem('ragEnabled', 'true');
  };

  return {
    isRagAvailable,
    isRagEnabled,
    ragNotificationShown,
    setRagNotificationShown,
    checkForStoredContext,
    checkRagAvailability,
    forceCheckDocumentStatus,
    toggleRagMode,
    showRagAvailableNotification
  };
}; 