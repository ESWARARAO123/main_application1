import { api } from './api';

export interface DocumentUploadResponse {
  success: boolean;
  document: {
    id: string;
    name: string;
    type: string;
    size: number;
    status: string;
  };
  message: string;
}

export interface DocumentStatusResponse {
  id: string;
  status: string;
  error?: string;
  createdAt: string;
  updatedAt: string;
}

export const documentService = {
  // Upload document to the correct endpoint
  uploadDocument: async (
    file: File,
    sessionId?: string,
    collectionId?: string,
    onUploadProgress?: (progress: number) => void
  ): Promise<DocumentUploadResponse> => {
    try {
      const formData = new FormData();
      formData.append('file', file);

      if (sessionId) {
        formData.append('sessionId', sessionId);
      }

      if (collectionId) {
        formData.append('collectionId', collectionId);
      }

      console.log(`Uploading document: ${file.name} (${file.size} bytes) to session: ${sessionId || 'none'}`);

      const response = await api.post('/documents/upload', formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        },
        onUploadProgress: (progressEvent) => {
          if (progressEvent.total && onUploadProgress) {
            const progress = Math.round((progressEvent.loaded * 100) / progressEvent.total);
            onUploadProgress(progress);
          }
        }
      });

      console.log('Document uploaded successfully:', response.data);
      return response.data;
    } catch (error) {
      console.error('Error uploading document:', error);
      throw error;
    }
  },

  // Get document status
  getDocumentStatus: async (documentId: string): Promise<DocumentStatusResponse> => {
    try {
      const response = await api.get(`/documents/${documentId}/status`);
      return response.data;
    } catch (error) {
      console.error('Error getting document status:', error);
      throw error;
    }
  },

  // Get all documents for user
  getUserDocuments: async (collectionId?: string): Promise<any[]> => {
    try {
      const params = collectionId ? { collectionId } : {};
      const response = await api.get('/documents', { params });
      return response.data;
    } catch (error) {
      console.error('Error getting user documents:', error);
      throw error;
    }
  },

  // Delete document
  deleteDocument: async (documentId: string): Promise<{ message: string }> => {
    try {
      const response = await api.delete(`/documents/${documentId}`);
      return response.data;
    } catch (error) {
      console.error('Error deleting document:', error);
      throw error;
    }
  }
};

export default documentService; 