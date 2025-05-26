/**
 * RAG (Retrieval-Augmented Generation) Service
 * Handles retrieval of relevant document chunks and augmentation of prompts
 */

const vectorStoreService = require('./vectorStoreService');
const OllamaService = require('./ollamaService');
const { formatRagResponse } = require('./ragResponseFormatter');
const { logger } = require('../utils/logger');

class RAGService {
  constructor() {
    this.vectorStoreService = vectorStoreService;
    // Create a new instance of OllamaService
    this.ollamaService = new OllamaService();

    // Initialize immediately
    this.initialized = false;
    this.initializeServices();

    this.embeddingModel = 'nomic-embed-text';
  }

  /**
   * Initialize services
   */
  async initializeServices() {
    try {
      // Initialize OllamaService
      await this.ollamaService.initialize();
      logger.info('RAG: OllamaService initialized successfully');
      console.log('RAG: OllamaService initialized successfully');
      this.initialized = true;
    } catch (err) {
      logger.error(`RAG: Failed to initialize OllamaService: ${err.message}`);
      console.error(`RAG: Failed to initialize OllamaService: ${err.message}`);
      this.initialized = false;
    }
  }

  /**
   * Generate a query embedding and retrieve relevant context
   * @param {string} query - User query
   * @param {Object} options - Options for retrieval
   * @returns {Promise<Object>} - Result with context and sources
   */
  async retrieveContext(query, options = {}) {
    const {
      topK = 5,
      model = this.embeddingModel,
      sessionId = null,
      userId = null
    } = options;

    try {
      // Make sure OllamaService is initialized
      if (!this.initialized) {
        console.log('RAG: OllamaService not initialized, initializing now...');
        await this.initializeServices();

        if (!this.initialized) {
          console.error('RAG: Failed to initialize OllamaService');
          return {
            success: false,
            error: 'Failed to initialize OllamaService'
          };
        }
      }

      console.log(`RAG: Generating embedding for query: "${query.substring(0, 50)}..."`);
      if (sessionId) {
        console.log(`RAG: Using session context: ${sessionId}`);
      }

      // Generate embedding for the query - using the existing generateEmbedding method
      const embedResult = await this.ollamaService.generateEmbedding(query, model);
      if (!embedResult.success) {
        console.error(`RAG: Failed to generate query embedding: ${embedResult.error}`);
        return {
          success: false,
          error: 'Failed to generate query embedding'
        };
      }

      console.log(`RAG: Successfully generated query embedding, searching for relevant chunks...`);
      
      // Table-specific query expansion
      let isTableQuery = false;
      let tableReference = null;
      
      // Check if this is a query about tables
      if (query.toLowerCase().includes('table')) {
        isTableQuery = true;
        // Extract table references like "Table 2-2" or "table 2.2"
        const tableRefMatch = query.match(/table\s+(\d+[-\.]\d+)/i);
        if (tableRefMatch) {
          tableReference = tableRefMatch[0];
          console.log(`RAG: Detected table reference: ${tableReference}`);
        }
      }

      // Search options with increased limit for table queries
      const searchOptions = {
        sessionId,
        limit: isTableQuery ? 10 : topK, // Increase limit for table queries
        isTableQuery,
        tableReference
      };

      // Search for relevant documents - pass sessionId as an option
      const searchResult = await this.vectorStoreService.search(
        embedResult.embedding,
        searchOptions
      );

      if (!searchResult.success) {
        console.error(`RAG: Failed to retrieve relevant documents: ${searchResult.error}`);
        return {
          success: false,
          error: 'Failed to retrieve relevant documents'
        };
      }

      // If no results found, return empty context
      if (searchResult.results.length === 0) {
        console.log(`RAG: No relevant documents found for query`);
        return {
          success: true,
          context: '',
          sources: []
        };
      }

      console.log(`RAG: Found ${searchResult.results.length} relevant chunks`);

      // For table queries, prioritize chunks that contain table markers
      let results = searchResult.results;
      if (isTableQuery) {
        // Boost scores for chunks that contain table markers
        results = results.map(result => {
          // Check if the chunk contains table markers
          const hasTableMarker = result.text.match(/###\s+(?:Table|Extracted Table)/i);
          
          // If it has a table reference and matches the one in query, boost the score significantly
          if (tableReference && result.text.toLowerCase().includes(tableReference.toLowerCase())) {
            return {
              ...result,
              score: result.score * 1.5, // 50% boost
              containsRequestedTable: true
            };
          } 
          // If it has any table, boost the score slightly
          else if (hasTableMarker) {
            return {
              ...result,
              score: result.score * 1.2, // 20% boost
              containsTable: true
            };
          }
          
          return result;
        });
        
        // Re-sort by the adjusted scores
        results.sort((a, b) => b.score - a.score);
      }

      // Prepare context from retrieved documents
      const context = results
        .map(result => result.text)
        .join('\n\n---\n\n');

      // Format sources for citation
      const sources = results.map(result => ({
        text: result.text.substring(0, 150) + (result.text.length > 150 ? '...' : ''),
        metadata: result.metadata,
        score: result.score,
        containsTable: result.containsTable || false,
        containsRequestedTable: result.containsRequestedTable || false
      }));

      return {
        success: true,
        context,
        sources,
        isTableQuery,
        tableReference
      };
    } catch (error) {
      console.error(`RAG: Error retrieving context:`, error);
      return {
        success: false,
        error: error.message || 'Unknown error retrieving context'
      };
    }
  }

  /**
   * Process a chat message with RAG
   * @param {string} message - User message
   * @param {string} model - LLM model to use
   * @param {Object} options - Additional options
   * @returns {Promise<Object>} - Chat response with sources
   */
  async processRagChat(message, modelId, options = {}) {
    try {
      const { sessionId, userId } = options;
      
      console.log(`RAG: Processing chat message with RAG: "${message.substring(0, 50)}..." using model ${modelId}`);
      if (sessionId) {
        console.log(`RAG: Using session ID for context filtering: ${sessionId}`);
      }

      // Make sure OllamaService is initialized
      if (!this.initialized) {
        console.log('RAG: OllamaService not initialized, initializing now...');
        await this.initializeServices();

        if (!this.initialized) {
          console.error('RAG: Failed to initialize OllamaService');
          return {
            success: false,
            error: 'Failed to initialize OllamaService'
          };
        }
      }

      // Get context for the query
      const ragResult = await this.retrieveContext(message, {
        ...options,
        model: this.embeddingModel, // Always use embedding model for retrieval
        sessionId  // Explicitly pass sessionId for context filtering
      });

      // If retrieval failed, use regular chat
      if (!ragResult.success) {
        console.log(`RAG: Retrieval failed, falling back to regular chat`);
        // Use a fallback response instead of chat
        return {
          success: true,
          response: {
            choices: [{
              message: {
                role: 'assistant',
                content: `I don't have enough context to answer your question about "${message}". Please try a different question or provide more details.`
              }
            }]
          }
        };
      }

      // If no context found, use regular chat
      if (!ragResult.context) {
        console.log(`RAG: No relevant context found, using regular chat`);
        // Use a fallback response instead of chat
        return {
          success: true,
          response: {
            choices: [{
              message: {
                role: 'assistant',
                content: `I couldn't find relevant information about "${message}" in the available documents. Please try a different question or provide more details.`
              }
            }]
          }
        };
      }

      console.log(`RAG: Using context from ${ragResult.sources.length} sources`);

      // Create the prompt with context
      const ragMessages = [
        {
          role: 'system',
          content: 'You are a helpful assistant that answers questions based on the provided context. If the information is not in the context, acknowledge that and provide your best response based on your general knowledge, but make it clear which parts are from the documents and which are not.'
        },
        {
          role: 'user',
          content: `Use the following context from relevant documents to answer the question:

Context:
${ragResult.context}

Question: ${message}`
        }
      ];

      // Use the ollamaService.chat method to generate a response based on the context
      try {
        console.log(`RAG: Calling ollamaService.chat with model: ${modelId}`);
        const chatResult = await this.ollamaService.chat(modelId, ragMessages);

        if (chatResult.success) {
          console.log(`RAG: Successfully generated response using ollamaService.chat`);
          // Add sources to the response
          return {
            success: true,
            response: chatResult.response,
            sources: ragResult.sources,
            context: ragResult.context
          };
        } else {
          console.warn(`RAG: Failed to generate chat response: ${chatResult.error}`);
          // Fall back to the formatter
        }
      } catch (chatError) {
        console.error(`RAG: Error calling ollamaService.chat: ${chatError.message}`);
        // Fall back to the formatter
      }

      // Fallback: Use the formatter to create a response if chat fails
      console.log(`RAG: Using formatter fallback for response`);
      const formattedResponse = formatRagResponse(message, ragResult.sources);

      // Create the response object with the formatted answer
      const chatResponse = {
        success: true,
        response: {
          choices: [{
            message: {
              role: 'assistant',
              content: formattedResponse
            }
          }]
        },
        sources: ragResult.sources,
        context: ragResult.context
      };

      return chatResponse;
    } catch (error) {
      console.error(`RAG: Error in RAG chat processing:`, error);
      return {
        success: false,
        error: error.message || 'Unknown error in RAG processing'
      };
    }
  }

  /**
   * Check if RAG is available
   * @returns {Promise<boolean>} - Whether RAG is available
   */
  async isRagAvailable() {
    try {
      // First check if the vector store is initialized
      if (!this.vectorStoreService) {
        console.log('RAG: Vector store service not available');
        return false;
      }

      // Then check if there are any documents in the vector store
      const stats = await this.vectorStoreService.getStats();
      const hasDocuments = stats.success && stats.count > 0;

      console.log(`RAG: Availability check - Vector store has ${stats.count} chunks in ${stats.documents || 'unknown'} documents`);

      if (!hasDocuments) {
        console.log('RAG: No documents found in vector store');
        return false;
      }

      // Also check if OllamaService is initialized
      if (!this.initialized) {
        console.log('RAG: OllamaService not initialized, initializing now...');
        await this.initializeServices();

        if (!this.initialized) {
          console.error('RAG: Failed to initialize OllamaService');
          return false;
        }
      }

      // Verify we can generate embeddings
      try {
        const testEmbed = await this.ollamaService.generateEmbedding("test", this.embeddingModel);
        if (!testEmbed.success) {
          console.error('RAG: Embedding generation test failed');
          return false;
        }
      } catch (embedError) {
        console.error('RAG: Error testing embedding generation:', embedError);
        return false;
      }

      console.log('RAG: All checks passed, RAG is available');
      return true;
    } catch (error) {
      console.error(`RAG: Error checking RAG availability:`, error);
      return false;
    }
  }
}

module.exports = new RAGService();
