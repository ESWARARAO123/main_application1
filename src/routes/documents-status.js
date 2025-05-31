/**
 * Document Status API
 * Provides endpoints for checking document processing status
 */

const express = require('express');
const router = express.Router();
const fs = require('fs');
const path = require('path');
const documentService = require('../services/documentService');

// Middleware to check if user is authenticated
const authenticateToken = (req, res, next) => {
  if (req.session && req.session.userId) {
    next();
  } else {
    res.status(401).json({ error: 'Unauthorized' });
  }
};

/**
 * Get document processing status
 * GET /api/documents-status/:id
 */
router.get('/:id', authenticateToken, async (req, res) => {
  try {
    const documentId = req.params.id;
    const userId = req.session.userId;

    // Get document info
    const document = await documentService.getDocument(documentId, userId);

    // Check if document exists
    if (!document) {
      return res.status(404).json({ error: 'Document not found' });
    }

    // Check embedding progress if document is in EMBEDDING state
    let embeddingProgress = null;
    if (document.status === 'EMBEDDING') {
      const embeddingsDir = path.join(__dirname, '../../embeddings', userId);
      const progressFilePath = path.join(embeddingsDir, `${documentId}_embedding_progress.json`);
      
      if (fs.existsSync(progressFilePath)) {
        try {
          const progressData = JSON.parse(fs.readFileSync(progressFilePath, 'utf8'));
          embeddingProgress = progressData;
        } catch (error) {
          console.warn(`Error reading embedding progress file: ${error.message}`);
        }
      }
    }

    // Return document status with embedding progress if available
    res.json({
      id: document.id,
      status: document.status,
      error: document.processing_error,
      createdAt: document.created_at,
      updatedAt: document.updated_at,
      embeddingProgress
    });
  } catch (error) {
    console.error('Error getting document status:', error);
    if (error.message === 'Document not found') {
      return res.status(404).json({ error: 'Document not found' });
    }
    res.status(500).json({ error: 'Failed to get document status' });
  }
});

module.exports = router;
