import { api } from './api';

export interface RagChatRequest {
  model: string;
  message: string;
  sessionId?: string;
}

export interface RagChatResponse {
  content: string;
  sources: RagSource[];
  ragAvailable: boolean;
}

export interface RagSource {
  text: string;
  metadata: {
    documentId: string;
    fileName: string;
    userId: string;
    fileType: string;
    [key: string]: any;
  };
  score: number;
}

export const ragChatService = {
  /**
   * Send a message to the RAG-enhanced chat API
   * @param request The RAG chat request
   * @returns The RAG chat response with sources
   */
  sendRagChatMessage: async (request: RagChatRequest): Promise<RagChatResponse> => {
    try {
      console.log('Sending RAG chat request:', {
        model: request.model,
        messageLength: request.message.length,
        sessionId: request.sessionId
      });

      const response = await api.post('/ollama/rag-chat', request);
      return response.data;
    } catch (error) {
      console.error('Error in ragChatService.sendRagChatMessage:', error);
      throw error;
    }
  },

  /**
   * Check if RAG is available
   * @returns Whether RAG is available
   */
  isRagAvailable: async (): Promise<boolean> => {
    try {
      // Use a dedicated endpoint to check RAG availability
      const response = await api.get('/ollama/rag-availability');
      console.log('RAG availability checked:', response.data);

      // Add a timestamp to the console log to track when checks happen
      console.log('RAG check timestamp:', new Date().toISOString());

      // Make sure we have a valid response with the available property
      if (response.data && typeof response.data.available === 'boolean') {
        return response.data.available === true;
      } else {
        console.warn('Invalid RAG availability response format:', response.data);
        return false;
      }
    } catch (error: any) {
      // If we get a specific response indicating RAG is not available
      if (error.response?.data?.available === false) {
        console.log('RAG availability checked: Not Available');
        return false;
      }
      // For other errors, log and return false
      console.error('Error checking RAG availability:', error);
      return false;
    }
  },

  /**
   * Clear RAG data for a session
   * @param sessionId The session ID to clear RAG data for
   * @returns Whether the operation was successful
   */
  clearRagData: async (sessionId: string): Promise<boolean> => {
    try {
      const response = await api.delete(`/ollama/rag-data/${sessionId}`);
      console.log('RAG data cleared for session:', sessionId);
      return response.data.success === true;
    } catch (error: any) {
      console.error('Error clearing RAG data:', error);
      return false;
    }
  }
};
