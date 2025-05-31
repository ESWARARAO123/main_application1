const express = require('express');
const router = express.Router();
const path = require('path');
const fs = require('fs');
const { loadPathsFromConfig, ensureDirectoriesExist } = require('../utils/pathConfig');
let multer;
let uuidv4;

// Try to require multer and uuid, but don't fail if they're not available
try {
  multer = require('multer');
} catch (error) {
  console.error('Multer package not found. File upload functionality will not be available.');
  console.error('Please install multer: npm install multer');
}

try {
  const { v4 } = require('uuid');
  uuidv4 = v4;
} catch (error) {
  console.error('UUID package not found. Using timestamp-based IDs instead.');
  console.error('Please install uuid: npm install uuid');
  // Fallback implementation of uuidv4
  uuidv4 = () => `fallback-${Date.now()}-${Math.random().toString(36).substring(2, 15)}`;
}

const { pool } = require('../database');
const documentService = require('../services/documentService');
const vectorStoreService = require('../services/vectorStoreService');
const logger = require('../utils/logger');

// Middleware to check if user is authenticated (similar to chatbot.js)
const authenticateToken = (req, res, next) => {
  if (req.session && req.session.userId) {
    next();
  } else {
    res.status(401).json({ error: 'Unauthorized' });
  }
};

// Load paths from config and ensure directories exist
const paths = loadPathsFromConfig();
ensureDirectoriesExist(paths);

// Get directory paths
const { documentsDir, embeddingsDir } = paths;

// Configure multer for file uploads if available
let upload = null;

if (multer) {
  const storage = multer.diskStorage({
    destination: function (req, file, cb) {
      // Create user directory if it doesn't exist
      const userDir = path.join(documentsDir, req.session.userId);
      if (!fs.existsSync(userDir)) {
        fs.mkdirSync(userDir, { recursive: true });
      }
      cb(null, userDir);
    },
    filename: function (req, file, cb) {
      // Generate a unique filename with original extension
      const fileExt = path.extname(file.originalname);
      const fileName = `${uuidv4()}${fileExt}`;
      cb(null, fileName);
    }
  });

  // File filter to only allow certain file types
  const fileFilter = (req, file, cb) => {
    const allowedTypes = [
      'application/pdf',
      'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
      'text/plain'
    ];

    if (allowedTypes.includes(file.mimetype)) {
      cb(null, true);
    } else {
      cb(new Error('Invalid file type. Only PDF, DOCX, and TXT files are allowed.'), false);
    }
  };

  upload = multer({
    storage: storage,
    fileFilter: fileFilter,
    limits: {
      fileSize: 10 * 1024 * 1024 // 10MB limit
    }
  });
} else {
  console.warn('Multer not available, file upload routes will return 503 Service Unavailable');
  // Create a dummy middleware that returns 503 for file upload routes
  upload = {
    single: () => (req, res, next) => {
      res.status(503).json({
        error: 'File upload service unavailable. Please install multer: npm install multer'
      });
    }
  };
}

/**
 * Handle document upload and processing
 * POST /api/documents/upload
 */
router.post('/upload', authenticateToken, upload.single('file'), async (req, res) => {
  try {
    const userId = req.session.userId;
    const { sessionId } = req.body;
    const file = req.file;

    if (!file) {
      return res.status(400).json({ error: 'No file provided' });
    }

    logger.info(`Document upload initiated for file: ${file.originalname} by user ${userId}`);

    // If a sessionId is provided, we need to clear any existing documents for this session
    // to prevent cross-contamination of RAG sources
    if (sessionId) {
      logger.info(`Clearing previous document data for session ${sessionId}`);

      try {
        // Delete any previous documents associated with this session
        const clearResult = await vectorStoreService.clearSessionDocuments(sessionId);

        if (clearResult.success) {
          logger.info(`Successfully cleared previous documents for session ${sessionId}: ${clearResult.deletedCount} chunks removed`);
        } else {
          logger.warn(`Failed to clear previous documents for session ${sessionId}: ${clearResult.error}`);
        }
      } catch (clearError) {
        logger.error(`Error clearing previous session documents: ${clearError.message}`);
        // Continue with upload even if clearing fails
      }
    }

    const { originalname, mimetype, size, filename, path: filePath } = file;
    const collectionId = req.body.collectionId || null;

    // Process the document (rest of the document upload code)
    // Create database entry
    const document = await documentService.createDocument({
      user_id: userId,
      original_name: file.originalname,
      file_path: file.path, // Use the actual file path
      file_type: path.extname(file.originalname).substring(1), // Get extension without dot
      file_size: file.size,
      mime_type: file.mimetype,
      collection_id: collectionId,
      status: 'pending',
      session_id: sessionId || null, // Associate document with a session if provided
      filename: file.filename || file.originalname // Add filename field
    });

    // Return success response immediately
    res.status(201).json({
      success: true,
      document: {
        id: document.id,
        name: document.original_name,
        type: document.file_type,
        size: document.file_size,
        status: document.status
      },
      message: 'Document uploaded successfully. Processing has started.'
    });

    // Start document processing in the background
    documentService.processDocument(document.id, {
      userId,
      sessionId: sessionId || null
    }).catch(error => {
      logger.error(`Background document processing failed for ${document.id}: ${error.message}`);
      // The error is handled inside processDocument, so we just log it here
    });
  } catch (error) {
    console.error('Error uploading document:', error);
    res.status(500).json({ error: 'Failed to upload document' });
  }
});

// Route to download a document
router.get('/download/:id', authenticateToken, async (req, res) => {
  try {
    const documentId = req.params.id;
    const userId = req.session.userId;

    // Get document info using the document service
    const document = await documentService.getDocument(documentId, userId);
    const filePath = document.file_path;

    // Check if file exists
    if (!fs.existsSync(filePath)) {
      return res.status(404).json({ error: 'File not found on server' });
    }

    // Set content disposition and type
    res.setHeader('Content-Disposition', `attachment; filename="${document.original_name}"`);
    res.setHeader('Content-Type', document.file_type);

    // Stream the file
    const fileStream = fs.createReadStream(filePath);
    fileStream.pipe(res);
  } catch (error) {
    console.error('Error downloading document:', error);
    if (error.message === 'Document not found') {
      return res.status(404).json({ error: 'Document not found' });
    }
    res.status(500).json({ error: 'Failed to download document' });
  }
});

// Route to get all documents for a user
router.get('/', authenticateToken, async (req, res) => {
  try {
    const userId = req.session.userId;
    const collectionId = req.query.collectionId;

    // Get documents using the document service
    const documents = await documentService.getUserDocuments(userId, collectionId);
    res.json(documents);
  } catch (error) {
    console.error('Error getting documents:', error);
    res.status(500).json({ error: 'Failed to get documents' });
  }
});

// Route to get document status
router.get('/:id/status', authenticateToken, async (req, res) => {
  try {
    const documentId = req.params.id;
    const userId = req.session.userId;

    // Get document using the document service
    const document = await documentService.getDocument(documentId, userId);

    // Return document status
    res.json({
      id: document.id,
      status: document.status,
      error: document.processing_error,
      createdAt: document.created_at,
      updatedAt: document.updated_at
    });
  } catch (error) {
    console.error('Error getting document status:', error);
    if (error.message === 'Document not found') {
      return res.status(404).json({ error: 'Document not found' });
    }
    res.status(500).json({ error: 'Failed to get document status' });
  }
});

// Route to delete a document
router.delete('/:id', authenticateToken, async (req, res) => {
  try {
    const documentId = req.params.id;
    const userId = req.session.userId;

    // Delete document using the document service
    await documentService.deleteDocument(documentId, userId);
    res.json({ message: 'Document deleted successfully' });
  } catch (error) {
    console.error('Error deleting document:', error);
    if (error.message === 'Document not found') {
      return res.status(404).json({ error: 'Document not found' });
    }
    res.status(500).json({ error: 'Failed to delete document' });
  }
});

module.exports = router;
