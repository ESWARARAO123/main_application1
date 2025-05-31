const fs = require('fs');
const path = require('path');
const { pool } = require('../database');
const { loadPathsFromConfig, ensureDirectoriesExist } = require('../utils/pathConfig');
// Don't require documentProcessor here to avoid circular dependency
// We'll require it only when needed in specific methods

class DocumentService {
  constructor() {
    // Load paths from config and ensure directories exist
    const paths = loadPathsFromConfig();
    ensureDirectoriesExist(paths);

    // Set document directory from config
    this.documentsDir = paths.documentsDir;
  }

  /**
   * Get user's document directory, create if it doesn't exist
   * @param {string} userId - User ID
   * @returns {string} - Path to user's document directory
   */
  getUserDocumentDir(userId) {
    const userDir = path.join(this.documentsDir, userId);
    if (!fs.existsSync(userDir)) {
      fs.mkdirSync(userDir, { recursive: true });
    }
    return userDir;
  }

  /**
   * Save document metadata to database
   * @param {Object} document - Document metadata
   * @param {boolean} processDocument - Whether to trigger document processing
   * @returns {Promise<Object>} - Saved document with ID
   */
  async saveDocument(document, processDocument = false) {
    const { userId, filename, originalName, filePath, fileType, fileSize, collectionId } = document;

    try {
      const result = await pool.query(
        'INSERT INTO documents (user_id, filename, original_name, file_path, file_type, file_size, collection_id, status) VALUES ($1, $2, $3, $4, $5, $6, $7, $8) RETURNING id, status, created_at',
        [userId, filename, originalName, filePath, fileType, fileSize, collectionId, 'UPLOADED']
      );

      const savedDocument = {
        id: result.rows[0].id,
        userId,
        filename,
        originalName,
        filePath,
        fileType,
        fileSize,
        collectionId,
        status: result.rows[0].status,
        createdAt: result.rows[0].created_at
      };

      // Trigger document processing if requested
      if (processDocument) {
        // We'll do this asynchronously so it doesn't block the response
        this.triggerDocumentProcessing(savedDocument.id).catch(error => {
          console.error(`Error triggering processing for document ${savedDocument.id}:`, error);
        });
      }

      return savedDocument;
    } catch (error) {
      console.error('Error saving document to database:', error);
      throw new Error('Failed to save document metadata');
    }
  }

  /**
   * Trigger document processing
   * @param {number} documentId - Document ID
   * @returns {Promise<void>}
   */
  async triggerDocumentProcessing(documentId) {
    try {
      // Get the document from the database
      const document = await this.getDocument(documentId);

      if (!document) {
        throw new Error(`Document not found: ${documentId}`);
      }

      // Update status to processing
      await this.updateDocumentStatus(documentId, 'processing');

      // Process the document using the document processor
      // Require documentProcessor only when needed to avoid circular dependency
      const documentProcessor = require('./documentProcessor');
      await documentProcessor.processDocument(document, {
        userId: document.user_id,
        sessionId: document.session_id || null
      });
    } catch (error) {
      console.error(`Error processing document ${documentId}:`, error);
      // Update document status to ERROR
      await this.updateDocumentStatus(
        documentId,
        'ERROR',
        error.message || 'Unknown error during processing'
      );
      throw error;
    }
  }

  /**
   * Get document by ID
   * @param {number} id - Document ID
   * @param {string} [userId] - Optional User ID (for authorization)
   * @returns {Promise<Object>} - Document metadata
   */
  async getDocument(id, userId = null) {
    try {
      let query, params;

      if (userId) {
        // If userId is provided, use it for authorization
        query = 'SELECT * FROM documents WHERE id = $1 AND user_id = $2';
        params = [id, userId];
      } else {
        // If no userId, just get the document by ID (for internal use)
        query = 'SELECT * FROM documents WHERE id = $1';
        params = [id];
      }

      const result = await pool.query(query, params);

      if (result.rows.length === 0) {
        throw new Error('Document not found');
      }

      return result.rows[0];
    } catch (error) {
      console.error('Error getting document:', error);
      throw error;
    }
  }

  /**
   * Get all documents for a user
   * @param {string} userId - User ID
   * @param {string} collectionId - Optional collection ID to filter by
   * @returns {Promise<Array>} - Array of document metadata
   */
  async getUserDocuments(userId, collectionId = null) {
    try {
      let query = 'SELECT * FROM documents WHERE user_id = $1';
      let params = [userId];

      if (collectionId) {
        query += ' AND collection_id = $2';
        params.push(collectionId);
      }

      query += ' ORDER BY created_at DESC';

      const result = await pool.query(query, params);
      return result.rows;
    } catch (error) {
      console.error('Error getting user documents:', error);
      throw error;
    }
  }

  /**
   * Update document status
   * @param {number} id - Document ID
   * @param {string} status - New status
   * @param {string} errorMessage - Optional error message
   * @param {boolean} isLongRunning - Whether this is a long-running process that needs to keep the session alive
   * @returns {Promise<Object>} - Updated document
   */
  async updateDocumentStatus(id, status, errorMessage = null, isLongRunning = false) {
    try {
      let query = 'UPDATE documents SET status = $1, updated_at = NOW()';
      let params = [status];

      if (errorMessage) {
        query += ', processing_error = $2';
        params.push(errorMessage);
      }

      query += ' WHERE id = $' + (params.length + 1) + ' RETURNING *';
      params.push(id);

      const result = await pool.query(query, params);

      if (result.rows.length === 0) {
        throw new Error('Document not found');
      }

      // If this is a long-running process, update a timestamp to help keep the session alive
      if (isLongRunning) {
        // Update a last_active timestamp in the document record
        // This database activity helps prevent session timeouts during long-running processes
        try {
          await pool.query(
            'UPDATE documents SET last_activity_timestamp = NOW() WHERE id = $1',
            [id]
          );
        } catch (error) {
          // Don't fail the main operation if this fails
          console.warn(`Failed to update last_activity_timestamp for document ${id}:`, error.message);
        }
      }

      return result.rows[0];
    } catch (error) {
      console.error('Error updating document status:', error);
      throw error;
    }
  }

  /**
   * Delete document
   * @param {number} id - Document ID
   * @param {string} userId - User ID (for authorization)
   * @returns {Promise<boolean>} - Success status
   */
  async deleteDocument(id, userId) {
    try {
      // Get document info first
      const document = await this.getDocument(id, userId);

      // Delete from database
      await pool.query('DELETE FROM documents WHERE id = $1', [id]);

      // Delete file if it exists
      if (fs.existsSync(document.file_path)) {
        fs.unlinkSync(document.file_path);
      }

      return true;
    } catch (error) {
      console.error('Error deleting document:', error);
      throw error;
    }
  }

  /**
   * Process document for vector storage and RAG
   * @param {string} documentId - Document ID
   * @param {Object} options - Processing options
   * @returns {Promise<Object>} - Processing result
   */
  async processDocument(documentId, options = {}) {
    try {
      const { userId, sessionId } = options;

      // Get document from database
      const document = await this.getDocument(documentId);
      if (!document) {
        throw new Error(`Document not found: ${documentId}`);
      }

      // Update status to processing
      await this.updateDocumentStatus(documentId, 'processing');

      console.log(`Processing document ${documentId} with options:`, {
        userId: userId || document.user_id,
        sessionId: sessionId || document.session_id || null
      });

      // Require documentProcessor only when needed to avoid circular dependency
      const documentProcessor = require('./documentProcessor');
      // Process using document processor
      const result = await documentProcessor.processDocument(document, {
        userId: userId || document.user_id,
        sessionId: sessionId || document.session_id || null
      });

      // Update status based on result
      if (result.success) {
        await this.updateDocumentStatus(documentId, 'processed');
      } else {
        await this.updateDocumentStatus(documentId, 'error', result.error);
      }

      return result;
    } catch (error) {
      console.error(`Error processing document ${documentId}:`, error);
      // Update document status to error
      await this.updateDocumentStatus(documentId, 'error', error.message);
      return {
        success: false,
        error: error.message
      };
    }
  }

  /**
   * Create a new document in the database
   * @param {Object} documentData - Document data
   * @returns {Promise<Object>} - The created document
   */
  async createDocument(documentData) {
    const {
      user_id,
      original_name,
      file_path,
      file_type,
      file_size,
      mime_type,
      collection_id = null,
      status = 'pending',
      session_id = null,
      filename = null
    } = documentData;

    console.log(`Creating document record: ${original_name} for user ${user_id}${session_id ? `, session ${session_id}` : ''}`);

    // Validate required fields
    if (!user_id || !original_name || !file_path) {
      throw new Error('Missing required document data: user_id, original_name, and file_path are required');
    }

    // Generate a filename if not provided
    const generatedFilename = filename || path.basename(file_path) || original_name;

    try {
      const query = `
        INSERT INTO documents
        (user_id, original_name, file_path, file_type, file_size, mime_type, collection_id, status, session_id, filename)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
        RETURNING *
      `;

      const params = [
        user_id,
        original_name,
        file_path,
        file_type,
        file_size,
        mime_type,
        collection_id,
        status,
        session_id,
        generatedFilename
      ];

      const result = await pool.query(query, params);

      if (result.rows.length === 0) {
        throw new Error('Failed to create document record');
      }

      console.log(`Document created successfully with ID: ${result.rows[0].id}`);

      return result.rows[0];
    } catch (error) {
      console.error('Error creating document in database:', error);
      throw new Error(`Failed to create document: ${error.message}`);
    }
  }
}

module.exports = new DocumentService();
