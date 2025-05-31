import { useState, useEffect, useCallback } from 'react';
import { useWebSocket } from '../contexts/WebSocketContext';
import chatbotService from '../services/chatbotService';

/**
 * Document status type
 */
export interface DocumentStatus {
  documentId: string;
  status: string;
  progress: number;
  message: string;
  timestamp: string;
  error?: string;
}

/**
 * Hook options
 */
interface UseDocumentStatusOptions {
  documentId: string;
  initialStatus?: Partial<DocumentStatus>;
  pollingInterval?: number;
  enablePolling?: boolean;
}

/**
 * Custom hook for tracking document status
 * Uses WebSockets for real-time updates with polling fallback
 */
export function useDocumentStatus({
  documentId,
  initialStatus,
  pollingInterval = 5000,
  enablePolling = true
}: UseDocumentStatusOptions) {
  // State for document status
  const [status, setStatus] = useState<DocumentStatus | null>(
    initialStatus ? {
      documentId,
      status: initialStatus.status || 'pending',
      progress: initialStatus.progress || 0,
      message: initialStatus.message || 'Waiting to process',
      timestamp: initialStatus.timestamp || new Date().toISOString(),
      error: initialStatus.error
    } : null
  );

  // WebSocket context for real-time updates
  const { connected, addMessageListener } = useWebSocket();

  // State for tracking if we're using WebSocket or polling
  const [usingWebSocket, setUsingWebSocket] = useState<boolean>(false);

  // Fetch status from API
  const fetchStatus = useCallback(async () => {
    try {
      const response = await chatbotService.getDocumentStatus(documentId);

      // Map status to progress percentage
      let progressValue = 0;
      let messageText = 'Processing your document...'; // Keep it simple

      // Estimate progress based on status
      switch (response.status) {
        case 'pending':
        case 'PENDING':
          progressValue = 0;
          break;
        case 'UPLOADED':
        case 'uploaded':
          progressValue = 10;
          break;
        case 'processing':
        case 'PROCESSING':
          progressValue = 30;
          break;
        case 'EMBEDDING':
        case 'embedding':
          progressValue = 70;
          break;
        case 'PROCESSED':
        case 'processed':
        case 'completed':
        case 'COMPLETED':
          progressValue = 100;
          messageText = 'Processing complete';
          break;
        case 'ERROR':
        case 'error':
          progressValue = 0;
          messageText = response.error || 'Error processing document';
          break;
        default:
          // If we get an unknown status, assume it's still processing
          console.log(`Unknown document status: ${response.status}, assuming processing`);
          progressValue = 50;
      }

      // Convert API response to our status format
      setStatus(prev => ({
        documentId,
        status: response.status,
        progress: progressValue,
        message: messageText,
        timestamp: new Date().toISOString(),
        error: response.error
      }));

      return response;
    } catch (error) {
      console.error('Error fetching document status:', error);

      // If we get an error, don't immediately assume failure
      // The document might still be processing but the status endpoint is not ready yet

      // Check if we already have a status
      if (status && ['PROCESSING', 'processing', 'EMBEDDING', 'embedding', 'UPLOADED', 'uploaded', 'pending', 'PENDING'].includes(status.status)) {
        console.log('Status fetch failed but document might still be processing, keeping current status');
        // Keep the current status, don't update it
        return {
          status: status.status,
          error: null
        };
      }

      return null;
    }
  }, [documentId, status]);

  // Set up WebSocket listener for document status updates
  useEffect(() => {
    if (!documentId) return;

    // Set a flag in localStorage to indicate document processing is in progress
    // This will be used by the WebSocket heartbeat to increase frequency
    localStorage.setItem('documentProcessing', 'true');

    // Initial fetch to get current status
    fetchStatus().then(response => {
      // If document is processed or has an error, clear the processing flag
      if (response && (response.status === 'PROCESSED' || response.status === 'processed' ||
          response.status === 'ERROR' || response.status === 'error')) {
        localStorage.removeItem('documentProcessing');
      }
    });

    // Always set up polling as a fallback, with more aggressive polling if WebSocket is not connected
    const pollingIntervalToUse = connected ? pollingInterval * 1.5 : pollingInterval; // Slightly longer interval if WebSocket is connected

    console.log(`Setting up polling for document ${documentId} (${pollingIntervalToUse}ms), WebSocket ${connected ? 'connected' : 'disconnected'}`);

    // Create a more robust polling mechanism that adapts to WebSocket connection status
    let lastPollTime = Date.now();
    const interval = setInterval(() => {
      const now = Date.now();
      const timeSinceLastPoll = now - lastPollTime;

      // If it's been too long since our last poll (more than 2x the interval),
      // do an immediate poll regardless of WebSocket status
      const shouldForcePoll = timeSinceLastPoll > pollingIntervalToUse * 2;

      if (shouldForcePoll || !connected) {
        console.log(`Polling document status for ${documentId} (${shouldForcePoll ? 'forced poll' : 'WebSocket disconnected'})`);
        lastPollTime = now;

        fetchStatus().then(response => {
          // If document is processed or has an error, clear the processing flag
          if (response && (response.status === 'PROCESSED' || response.status === 'processed' ||
              response.status === 'ERROR' || response.status === 'error')) {
            localStorage.removeItem('documentProcessing');
          }
        });
      }
    }, pollingIntervalToUse);

    // Set up WebSocket listener if connected
    if (connected) {
      console.log(`Setting up WebSocket listener for document ${documentId}`);
      setUsingWebSocket(true);

      // Add listener for document status updates
      const removeListener = addMessageListener('document_status_update', (message) => {
        // Check if this update is for our document
        if (message.payload && message.payload.documentId === documentId) {
          console.log(`Received WebSocket update for document ${documentId}:`, message.payload);

          // Map status to progress percentage (similar to fetchStatus)
          let progressValue = message.payload.progress || 0;
          let messageText = 'Processing your document...'; // Keep it simple

          // If no progress is provided, estimate based on status
          if (!message.payload.progress) {
            switch (message.payload.status) {
              case 'pending':
              case 'PENDING':
                progressValue = 0;
                break;
              case 'UPLOADED':
              case 'uploaded':
                progressValue = 10;
                break;
              case 'processing':
              case 'PROCESSING':
                progressValue = 30;
                break;
              case 'EMBEDDING':
              case 'embedding':
                progressValue = 70;
                break;
              case 'PROCESSED':
              case 'processed':
              case 'completed':
              case 'COMPLETED':
                progressValue = 100;
                messageText = 'Processing complete';
                break;
              case 'ERROR':
              case 'error':
                progressValue = 0;
                messageText = message.payload.error || 'Error processing document';
                break;
              default:
                // If we get an unknown status, assume it's still processing
                console.log(`Unknown document status in WebSocket: ${message.payload.status}, assuming processing`);
                progressValue = 50;
            }
          }

          // Update status from WebSocket message with enhanced progress info
          const newStatus = {
            documentId,
            status: message.payload.status,
            progress: progressValue,
            message: messageText,
            timestamp: message.payload.timestamp || new Date().toISOString(),
            error: message.payload.error
          };

          setStatus(newStatus);

          // If document is processed or has an error, clear the processing flag
          if (newStatus.status === 'PROCESSED' || newStatus.status === 'processed' ||
              newStatus.status === 'completed' || newStatus.status === 'COMPLETED' ||
              newStatus.status === 'ERROR' || newStatus.status === 'error') {
            localStorage.removeItem('documentProcessing');
          }
        }
      });

      // Clean up function will remove both the WebSocket listener and the polling interval
      return () => {
        removeListener();
        clearInterval(interval);
        // Clear the document processing flag when unmounting
        localStorage.removeItem('documentProcessing');
      };
    } else {
      // If not connected to WebSocket, we're already polling
      setUsingWebSocket(false);

      // Clean up just the interval when component unmounts
      return () => {
        clearInterval(interval);
        // Clear the document processing flag when unmounting
        localStorage.removeItem('documentProcessing');
      };
    }
  }, [documentId, connected, addMessageListener, fetchStatus, pollingInterval, enablePolling]);

  // Return status and metadata
  return {
    status,
    usingWebSocket,
    refresh: fetchStatus
  };
}
