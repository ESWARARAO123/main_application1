/**
 * Vector Store Service
 * Handles interactions with ChromaDB for document embeddings storage and retrieval
 */

const path = require('path');
const fs = require('fs');
const { ChromaClient } = require('chromadb');

class VectorStoreService {
  constructor(config = {}) {
    this.config = {
      chromaUrl: 'http://172.16.16.54:8000', // URL for ChromaDB server
      collectionName: 'rag_docs',
      ...config
    };
    this.client = null;
    this.collection = null;
    this.isInitialized = false;
    this.useChromaDB = true;
    this.vectorStoreDir = path.join(process.cwd(), 'DATA/vector_store'); // Fallback directory

    // Create the fallback vector store directory if it doesn't exist
    if (!fs.existsSync(this.vectorStoreDir)) {
      fs.mkdirSync(this.vectorStoreDir, { recursive: true });
    }
  }

  /**
   * Initialize the vector store
   * @returns {Promise<boolean>} Success status
   */
  async initialize() {
    if (this.isInitialized) return true;

    try {
      // Try to initialize ChromaDB
      if (this.useChromaDB) {
        try {
          console.log(`Connecting to ChromaDB server at: ${this.config.chromaUrl}`);

          // Create ChromaClient with the server URL
          this.client = new ChromaClient({
            path: this.config.chromaUrl
          });

          // Get or create collection
          try {
            // Try to get the existing collection first
            try {
              console.log(`Attempting to get existing collection: ${this.config.collectionName}`);
              this.collection = await this.client.getCollection({
                name: this.config.collectionName
              });
              console.log(`Retrieved existing ChromaDB collection: ${this.config.collectionName}`);
            } catch (getError) {
              // If the collection doesn't exist, create a new one
              console.log(`Collection doesn't exist, creating new one: ${getError.message}`);

              // Create a new collection
              console.log(`Creating new ChromaDB collection: ${this.config.collectionName}`);
              this.collection = await this.client.createCollection({
                name: this.config.collectionName,
                metadata: { description: 'Document collection for RAG' }
              });
              console.log(`Created new ChromaDB collection: ${this.config.collectionName}`);
            }
          } catch (collectionError) {
            console.error('Error during collection operations:', collectionError);
            throw collectionError;
          }

          console.log('ChromaDB connected successfully');
          this.isInitialized = true;
          return true;
        } catch (chromaError) {
          console.error('Failed to connect to ChromaDB server:', chromaError);
          console.log('Falling back to file-based vector store');
          this.useChromaDB = false;
        }
      }

      // Fallback to file-based storage if ChromaDB initialization fails
      console.log(`Initialized file-based vector store at: ${this.vectorStoreDir}`);
      this.isInitialized = true;
      return true;
    } catch (error) {
      console.error('Failed to initialize vector store:', error);
      return false;
    }
  }

  /**
   * Add document chunks to the vector store
   * @param {Array} chunks - Array of chunks with text and embedding
   * @param {string} documentId - Document ID
   * @param {Object} metadata - Additional metadata
   * @returns {Promise<Object>} Result object
   */
  async addDocumentChunks(chunks, documentId, metadata = {}) {
    if (!await this.initialize()) {
      return {
        success: false,
        error: 'Vector store not initialized'
      };
    }

    try {
      // Ensure documentId is a string
      const documentIdStr = String(documentId);

      // Log if we have a sessionId in metadata
      if (metadata.sessionId) {
        console.log(`Adding chunks for document ${documentId} with sessionId: ${metadata.sessionId}`);
      }

      // If ChromaDB is available, use it
      if (this.useChromaDB && this.client) {
        try {
          // Make sure we have a valid collection reference
          if (!this.collection) {
            console.log('Getting ChromaDB collection before adding chunks');
            try {
              this.collection = await this.client.getCollection({
                name: this.config.collectionName
              });
              console.log(`Retrieved ChromaDB collection: ${this.config.collectionName}`);
            } catch (getError) {
              // If collection doesn't exist, create it
              console.log(`Creating new ChromaDB collection: ${this.config.collectionName}`);
              this.collection = await this.client.createCollection({
                name: this.config.collectionName,
                metadata: { description: 'Document collection for RAG' }
              });
            }
          }

          const ids = chunks.map((_, index) => `${documentIdStr}_chunk_${index}`);
          const embeddings = chunks.map(chunk => chunk.embedding);
          const documents = chunks.map(chunk => chunk.text);
          const metadatas = chunks.map(() => ({
            documentId: documentIdStr,
            source: metadata.fileName || 'unknown',
            // Include sessionId only if it exists in metadata
            ...(metadata.sessionId ? { sessionId: String(metadata.sessionId) } : {}),
            ...metadata
          }));

          await this.collection.add({
            ids,
            embeddings,
            documents,
            metadatas
          });

          console.log(`Stored ${chunks.length} vectors for document ${documentId} in ChromaDB`);
          return { success: true, count: chunks.length };
        } catch (chromaError) {
          console.error('Failed to store in ChromaDB, falling back to file-based store:', chromaError);
          // Don't disable ChromaDB permanently, just for this operation
        }
      }

      // Fallback to file-based storage
      const documentDir = path.join(this.vectorStoreDir, documentIdStr);
      if (!fs.existsSync(documentDir)) {
        fs.mkdirSync(documentDir, { recursive: true });
      }

      const vectorData = {
        documentId,
        metadata,
        chunks: chunks.map((chunk, index) => ({
          id: `${documentIdStr}_chunk_${index}`,
          text: chunk.text,
          embedding: chunk.embedding,
          metadata: {
            documentId: documentIdStr,
            source: metadata.fileName || 'unknown',
            // Include sessionId only if it exists in metadata
            ...(metadata.sessionId ? { sessionId: String(metadata.sessionId) } : {}),
            ...metadata
          }
        }))
      };

      const vectorFilePath = path.join(documentDir, 'vectors.json');
      fs.writeFileSync(vectorFilePath, JSON.stringify(vectorData, null, 2));

      console.log(`Stored ${chunks.length} vectors for document ${documentId} in file-based store`);
      return { success: true, count: chunks.length };
    } catch (error) {
      console.error('Error adding document chunks to vector store:', error);
      return { success: false, error: error.message };
    }
  }

  /**
   * Search for similar documents using a query embedding
   * @param {Array} queryEmbedding - Query embedding vector
   * @param {Object} options - Additional search options
   * @returns {Promise<Object>} Search results
   */
  async search(queryEmbedding, options = {}) {
    if (!await this.initialize()) {
      return {
        success: false,
        error: 'Vector store not initialized'
      };
    }

    try {
      // Extract options with defaults
      const limit = options.limit || 10; // Increased from 5 to 10
      const { sessionId } = options;
      
      console.log(`Searching with options:`, {
        limit,
        sessionId: sessionId || 'not specified',
        useChromaDB: this.useChromaDB && this.client
      });

      if (this.useChromaDB && this.client) {
        // Make sure we have a valid collection reference
        if (!this.collection) {
          console.log('Getting ChromaDB collection before searching');
          try {
            this.collection = await this.client.getCollection({
              name: this.config.collectionName
            });
            console.log(`Retrieved ChromaDB collection: ${this.config.collectionName}`);
          } catch (getError) {
            console.error(`Error getting collection for search: ${getError.message}`);
            // If we can't get the collection, fall back to file-based search
            throw getError;
          }
        }

        console.log(`Searching ChromaDB collection with limit: ${limit}`);

        // Build the query parameters
        const queryParams = {
          queryEmbeddings: [queryEmbedding],
          nResults: limit, // Use the configured limit
          include: ['documents', 'metadatas', 'distances']
        };

        // Note: We'll filter by sessionId after getting results
        // ChromaDB where clause is causing issues, so we'll do client-side filtering
        console.log(`Will filter search results by sessionId: ${sessionId} (post-query)`);
        // Don't add where clause to queryParams

        const results = await this.collection.query(queryParams);

        // Create the initial results array
        let searchResults = results.documents[0].map((text, index) => ({
          text,
          metadata: results.metadatas[0][index],
          score: 1 - results.distances[0][index] // Convert distance to similarity score
        }));

        // Filter by sessionId if provided (post-query filtering)
        if (sessionId) {
          const sessionIdStr = String(sessionId);
          console.log(`Post-query filtering for sessionId: ${sessionIdStr}`);

          // Keep only results that match the session ID or don't have a sessionId
          searchResults = searchResults.filter(result =>
            !result.metadata.sessionId ||
            result.metadata.sessionId === sessionIdStr
          );

          console.log(`After filtering: ${searchResults.length} results remain`);
        }

        return {
          success: true,
          results: searchResults
        };
      }

      // Fallback to file-based search
      const documentDirs = fs.readdirSync(this.vectorStoreDir)
        .filter(dir => fs.statSync(path.join(this.vectorStoreDir, dir)).isDirectory());

      if (documentDirs.length === 0) {
        return {
          success: true,
          results: []
        };
      }

      let allChunks = [];
      for (const documentId of documentDirs) {
        const vectorFilePath = path.join(this.vectorStoreDir, documentId, 'vectors.json');
        if (fs.existsSync(vectorFilePath)) {
          const vectorData = JSON.parse(fs.readFileSync(vectorFilePath, 'utf8'));

          // Filter chunks by sessionId if provided (for file-based store)
          let chunksToAdd = vectorData.chunks;
          if (sessionId) {
            chunksToAdd = chunksToAdd.filter(chunk =>
              !chunk.metadata.sessionId || chunk.metadata.sessionId === String(sessionId)
            );
          }

          allChunks = allChunks.concat(chunksToAdd);
        }
      }

      if (allChunks.length === 0) {
        return {
          success: true,
          results: []
        };
      }

      const similarities = allChunks.map(chunk => {
        const similarity = this.cosineSimilarity(queryEmbedding, chunk.embedding);
        return {
          ...chunk,
          score: similarity
        };
      });

      const topResults = similarities
        .sort((a, b) => b.score - a.score)
        .slice(0, limit); // Use the configured limit

      return {
        success: true,
        results: topResults.map(result => ({
          text: result.text,
          metadata: result.metadata,
          score: result.score
        }))
      };
    } catch (error) {
      console.error('Error searching vector store:', error);
      return { success: false, error: error.message };
    }
  }

  /**
   * Calculate cosine similarity between two vectors
   * @param {Array} vecA - First vector
   * @param {Array} vecB - Second vector
   * @returns {number} Cosine similarity (-1 to 1)
   */
  cosineSimilarity(vecA, vecB) {
    if (!vecA || !vecB || vecA.length !== vecB.length) {
      return 0;
    }

    let dotProduct = 0;
    let normA = 0;
    let normB = 0;

    for (let i = 0; i < vecA.length; i++) {
      dotProduct += vecA[i] * vecB[i];
      normA += vecA[i] * vecA[i];
      normB += vecB[i] * vecB[i];
    }

    if (normA === 0 || normB === 0) {
      return 0;
    }

    return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
  }

  /**
   * Delete document chunks from the vector store
   * @param {string} documentId - Document ID
   * @returns {Promise<Object>} Result object
   */
  async deleteDocumentChunks(documentId) {
    if (!await this.initialize()) {
      return {
        success: false,
        error: 'Vector store not initialized'
      };
    }

    try {
      const documentIdStr = String(documentId);

      if (this.useChromaDB && this.collection) {
        await this.collection.delete({
          where: { documentId: documentIdStr }
        });
        console.log(`Deleted document ${documentId} from ChromaDB`);
      }

      const documentDir = path.join(this.vectorStoreDir, documentIdStr);
      if (fs.existsSync(documentDir)) {
        fs.rmdirSync(documentDir, { recursive: true });
        console.log(`Deleted document ${documentId} from file-based vector store`);
      }

      return { success: true };
    } catch (error) {
      console.error(`Error deleting document ${documentId} from vector store:`, error);
      return { success: false, error: error.message };
    }
  }

  /**
   * Delete all RAG data associated with a session
   * @param {string} sessionId - Session ID
   * @param {string} userId - User ID
   * @returns {Promise<Object>} Result object
   */
  async deleteSessionData(sessionId, userId) {
    if (!await this.initialize()) {
      return {
        success: false,
        error: 'Vector store not initialized'
      };
    }

    try {
      // Ensure sessionId is a string
      const sessionIdStr = String(sessionId);
      console.log(`Deleting RAG data for session ${sessionIdStr} and user ${userId}`);

      let deletedCount = 0;

      // Using ChromaDB if available
      if (this.useChromaDB && this.collection) {
        try {
          // Delete all entries where sessionId matches
          const deleteResult = await this.collection.delete({
            where: { sessionId: sessionIdStr }
          });

          // Check if delete was successful (ChromaDB may not return a count)
          console.log(`Attempted to delete data for session ${sessionIdStr} from ChromaDB`, deleteResult);
          deletedCount += 1; // Since ChromaDB doesn't always return count
        } catch (chromaError) {
          console.error(`Error deleting from ChromaDB for session ${sessionIdStr}:`, chromaError);
          // Continue with file-based cleanup
        }
      }

      // File-based cleanup (fallback or additional cleanup)
      // This requires scanning all document directories for chunks with this sessionId
      const documentDirs = fs.readdirSync(this.vectorStoreDir)
        .filter(dir => fs.statSync(path.join(this.vectorStoreDir, dir)).isDirectory());

      for (const documentId of documentDirs) {
        const documentDir = path.join(this.vectorStoreDir, documentId);
        const vectorFilePath = path.join(documentDir, 'vectors.json');

        if (fs.existsSync(vectorFilePath)) {
          try {
            // Read the vector data
            const vectorData = JSON.parse(fs.readFileSync(vectorFilePath, 'utf8'));

            // Filter out chunks associated with this session
            const originalChunksCount = vectorData.chunks.length;
            vectorData.chunks = vectorData.chunks.filter(chunk =>
              !chunk.metadata.sessionId || chunk.metadata.sessionId !== sessionIdStr
            );

            // If any chunks were removed, update the file
            if (vectorData.chunks.length < originalChunksCount) {
              // If all chunks were removed, delete the file
              if (vectorData.chunks.length === 0) {
                console.log(`Deleting empty vector file for document ${documentId}`);
                fs.unlinkSync(vectorFilePath);

                // If this was the only file in the directory, delete the directory
                const dirFiles = fs.readdirSync(documentDir);
                if (dirFiles.length === 0) {
                  console.log(`Deleting empty document directory ${documentId}`);
                  fs.rmdirSync(documentDir);
                }
              } else {
                // Otherwise update the file with the remaining chunks
                console.log(`Updating vector file for document ${documentId}, removed ${originalChunksCount - vectorData.chunks.length} chunks`);
                fs.writeFileSync(vectorFilePath, JSON.stringify(vectorData, null, 2));
              }

              deletedCount += (originalChunksCount - vectorData.chunks.length);
            }
          } catch (fileError) {
            console.error(`Error processing vector file for document ${documentId}:`, fileError);
          }
        }
      }

      return {
        success: true,
        message: `Successfully deleted RAG data for session ${sessionIdStr}`,
        deletedCount
      };
    } catch (error) {
      console.error(`Error deleting session data for ${sessionId}:`, error);
      return {
        success: false,
        error: error.message
      };
    }
  }

  /**
   * Get vector store statistics
   * @returns {Promise<Object>} Vector store stats
   */
  async getStats() {
    if (!await this.initialize()) {
      return {
        success: false,
        error: 'Vector store not initialized'
      };
    }

    try {
      if (this.useChromaDB && this.collection) {
        try {
          // Check if we need to reconnect to the collection
          if (!this.client) {
            console.log('ChromaDB client not initialized, connecting...');
            this.client = new ChromaClient({
              path: this.config.chromaUrl
            });
          }

          // Get the collection without reinitializing
          if (!this.collection) {
            console.log('Getting ChromaDB collection for stats');
            try {
              this.collection = await this.client.getCollection({
                name: this.config.collectionName
              });
              console.log(`Retrieved ChromaDB collection: ${this.config.collectionName}`);
            } catch (getError) {
              console.error(`Error getting collection: ${getError.message}`);
              // If we can't get the collection, fall back to file-based stats
              throw getError;
            }
          }

          const count = await this.collection.count();
          console.log(`ChromaDB collection has ${count} items`);
          return {
            success: true,
            count,
            documents: count > 0 ? Math.ceil(count / 100) : 0, // Approximate number of documents (assuming ~100 chunks per doc)
            storageType: 'chromadb'
          };
        } catch (countError) {
          console.error('Error getting count from ChromaDB:', countError);
          // Fall back to file-based stats
        }
      }

      const documentDirs = fs.existsSync(this.vectorStoreDir)
        ? fs.readdirSync(this.vectorStoreDir).filter(dir =>
            fs.statSync(path.join(this.vectorStoreDir, dir)).isDirectory())
        : [];

      let totalChunks = 0;
      for (const documentId of documentDirs) {
        const vectorFilePath = path.join(this.vectorStoreDir, documentId, 'vectors.json');
        if (fs.existsSync(vectorFilePath)) {
          const vectorData = JSON.parse(fs.readFileSync(vectorFilePath, 'utf8'));
          totalChunks += vectorData.chunks.length;
        }
      }

      return {
        success: true,
        count: totalChunks,
        documents: documentDirs.length,
        storageType: 'file-based'
      };
    } catch (error) {
      console.error('Error getting vector store stats:', error);
      return { success: false, error: error.message };
    }
  }

  /**
   * Clear all documents associated with a specific session
   * @param {string} sessionId - Session ID
   * @returns {Promise<Object>} Result object
   */
  async clearSessionDocuments(sessionId) {
    if (!await this.initialize()) {
      return {
        success: false,
        error: 'Vector store not initialized'
      };
    }

    try {
      // Ensure sessionId is a string
      const sessionIdStr = String(sessionId);
      console.log(`Clearing all document data for session ${sessionIdStr}`);

      let deletedCount = 0;
      let documentsAffected = 0;

      // First, identify documents associated with this session
      // We'll use the metadata to find them
      if (this.useChromaDB && this.collection) {
        try {
          // Get entries where sessionId matches
          const queryResult = await this.collection.get({
            where: { sessionId: sessionIdStr }
          });

          if (queryResult && queryResult.metadatas && queryResult.metadatas.length > 0) {
            console.log(`Found ${queryResult.metadatas.length} entries for session ${sessionIdStr}`);

            // Extract unique document IDs
            const documentIds = new Set();
            queryResult.metadatas.forEach(metadata => {
              if (metadata && metadata.documentId) {
                documentIds.add(String(metadata.documentId));
              }
            });

            console.log(`Found ${documentIds.size} documents associated with session ${sessionIdStr}`);

            if (documentIds.size > 0) {
              // Delete all entries for these documents
              for (const docId of documentIds) {
                try {
                  await this.collection.delete({
                    where: { documentId: docId }
                  });
                  documentsAffected++;
                  console.log(`Deleted document ${docId} data from ChromaDB`);
                } catch (docDeleteError) {
                  console.error(`Error deleting document ${docId} from ChromaDB:`, docDeleteError);
                }
              }

              deletedCount += queryResult.metadatas.length;
            }
          } else {
            console.log(`No entries found for session ${sessionIdStr}`);
          }
        } catch (chromaError) {
          console.error(`Error clearing documents for session ${sessionIdStr} from ChromaDB:`, chromaError);
          // Continue with file-based cleanup
        }
      }

      // File-based cleanup
      const documentDirs = fs.readdirSync(this.vectorStoreDir)
        .filter(dir => fs.statSync(path.join(this.vectorStoreDir, dir)).isDirectory());

      for (const documentId of documentDirs) {
        const documentDir = path.join(this.vectorStoreDir, documentId);
        const vectorFilePath = path.join(documentDir, 'vectors.json');

        if (fs.existsSync(vectorFilePath)) {
          try {
            const vectorData = JSON.parse(fs.readFileSync(vectorFilePath, 'utf8'));

            // Check if any chunks are associated with this session
            const hasSessionChunks = vectorData.chunks.some(chunk =>
              chunk.metadata && chunk.metadata.sessionId === sessionIdStr
            );

            if (hasSessionChunks) {
              console.log(`Document ${documentId} has chunks for session ${sessionIdStr}, removing it entirely`);

              // Count the chunks we're removing
              const sessionChunksCount = vectorData.chunks.filter(chunk =>
                chunk.metadata && chunk.metadata.sessionId === sessionIdStr
              ).length;

              deletedCount += sessionChunksCount;

              // Delete the entire document directory
              fs.rmdirSync(documentDir, { recursive: true });
              documentsAffected++;
              console.log(`Deleted document directory ${documentId}`);
            }
          } catch (fileError) {
            console.error(`Error processing vector file for document ${documentId}:`, fileError);
          }
        }
      }

      return {
        success: true,
        message: `Successfully cleared document data for session ${sessionIdStr}`,
        deletedCount,
        documentsAffected
      };
    } catch (error) {
      console.error(`Error clearing document data for session ${sessionId}:`, error);
      return {
        success: false,
        error: error.message
      };
    }
  }
}

module.exports = new VectorStoreService();