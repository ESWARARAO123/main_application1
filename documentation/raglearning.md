# RAG (Retrieval-Augmented Generation) Learning

## What is RAG?

Retrieval-Augmented Generation (RAG) is a technique that enhances LLM outputs by incorporating relevant information from external knowledge sources. RAG consists of three main components:

1. **Retrieval**: Finding relevant information from a knowledge base for a given query
2. **Augmentation**: Enhancing the prompt with the retrieved context
3. **Generation**: Using an LLM to create a response based on the augmented prompt

## How RAG Works in `rag.py`

The provided `rag.py` file demonstrates a complete RAG implementation with these key components:

### 1. Document Processing

```python
def process_pdf_chunks(file_path):
    all_text = ""
    with pdfplumber.open(file_path) as pdf:
        all_text = "".join([page.extract_text() + "\n" for page in pdf.pages if page.extract_text()])
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(all_text)
    return chunks, len(chunks), None
```

This extracts text from PDFs and splits it into manageable chunks with overlap to maintain context across chunks.

### 2. Embedding Generation

```python
embedding = ollama_client.embeddings(model="nomic-embed-text", prompt=chunk)['embedding']
```

The system uses Ollama's `nomic-embed-text` model to convert text chunks into vector embeddings.

### 3. Vector Database Storage

```python
collection.add(
    ids=batch_ids,
    documents=batch_chunks,
    embeddings=batch_embeddings,
    metadatas=batch_metadatas
)
```

ChromaDB is used as the vector database to store embeddings along with their source documents.

### 4. Query Processing

```python
def rag_query(user_query):
    query_embedding = ollama_client.embeddings(model="nomic-embed-text", prompt=user_query)['embedding']
    results = collection.query(query_embeddings=[query_embedding], n_results=5, include=['documents', 'metadatas'])
    retrieved_docs = results['documents'][0] if results and results['documents'] else []
    context = "\n\n---\n\n".join(retrieved_docs)
    
    response = ollama_client.chat(model="qwen2", messages=[
        {"role": "system", "content": "You are a helpful assistant..."},
        {"role": "user", "content": f"Use the following context to answer the question:\n\nContext:\n{context}\n\nQuestion: {user_query}"}
    ])
    return response['message']['content'], None
```

When a user asks a question:
1. The query is converted to an embedding
2. Similar document chunks are retrieved from ChromaDB
3. Retrieved documents are combined into a context
4. The context and original query are sent to the LLM (qwen2) for generating the final response

## Integrating RAG with Your Current System

Based on the provided code, your system has partial RAG implementation. To complete it:

### 1. Install and Configure ChromaDB

```bash
pip install chromadb
```

### 2. Create a VectorStore Service

Create a service similar to OllamaService that manages ChromaDB interactions:

```javascript
// src/services/vectorStoreService.js
const chromadb = require('chromadb');

class VectorStoreService {
  constructor(config) {
    this.config = config;
    this.client = null;
    this.collection = null;
  }

  async initialize() {
    try {
      this.client = new chromadb.PersistentClient({ path: './chroma_db' });
      
      // Create or get the collection
      try {
        this.collection = await this.client.getCollection({ name: 'rag_docs' });
        console.log('Retrieved existing ChromaDB collection');
      } catch (error) {
        this.collection = await this.client.createCollection({
          name: 'rag_docs',
          metadata: { 'description': 'Document collection for RAG' }
        });
        console.log('Created new ChromaDB collection');
      }
      
      return true;
    } catch (error) {
      console.error('Failed to initialize ChromaDB:', error);
      return false;
    }
  }

  async addDocumentChunks(chunks, documentId, metadata = {}) {
    if (!this.collection) await this.initialize();
    
    const ids = chunks.map((_, i) => `${documentId}_chunk_${i}`);
    const embeddings = chunks.map(chunk => chunk.embedding);
    const documents = chunks.map(chunk => chunk.text);
    const metadatas = chunks.map(() => ({ 
      documentId, 
      source: metadata.fileName || 'unknown',
      ...metadata 
    }));
    
    try {
      await this.collection.add({
        ids,
        embeddings,
        documents,
        metadatas
      });
      return { success: true, count: chunks.length };
    } catch (error) {
      console.error('Error adding document chunks to ChromaDB:', error);
      return { success: false, error: error.message };
    }
  }

  async search(queryEmbedding, limit = 5) {
    if (!this.collection) await this.initialize();
    
    try {
      const results = await this.collection.query({
        queryEmbeddings: [queryEmbedding],
        nResults: limit,
        include: ['metadatas', 'documents', 'distances']
      });
      
      return {
        success: true,
        results: results.documents[0].map((doc, i) => ({
          text: doc,
          metadata: results.metadatas[0][i],
          score: results.distances[0][i]
        }))
      };
    } catch (error) {
      console.error('Error searching ChromaDB:', error);
      return { success: false, error: error.message };
    }
  }
}

module.exports = VectorStoreService;
```

### 3. Update DocumentProcessor to Use VectorStore

Modify `DocumentProcessor.js` to store embeddings in ChromaDB:

```javascript
// In src/services/documentProcessor.js
async generateEmbeddings(chunks, documentId, userEmbeddingsDir) {
  // Existing code to generate embeddings using OllamaService
  
  // After embeddings are generated, add to vector database
  if (this.vectorStoreService) {
    try {
      await this.vectorStoreService.addDocumentChunks(
        chunks.map((chunk, index) => ({
          text: chunk.text,
          embedding: result.embeddings[index]
        })),
        documentId,
        { fileName: path.basename(documentId) }
      );
      console.log(`Added ${chunks.length} chunks to vector database for document ${documentId}`);
    } catch (vectorError) {
      console.error(`Error adding embeddings to vector database: ${vectorError.message}`);
      // Continue with file-based storage as fallback
    }
  }
  
  // Existing code for file-based storage
}
```

### 4. Create RAG Query Service

Create a service to handle RAG queries:

```javascript
// src/services/ragService.js
class RAGService {
  constructor(ollamaService, vectorStoreService) {
    this.ollamaService = ollamaService;
    this.vectorStoreService = vectorStoreService;
  }
  
  async generateQueryWithContext(query, options = {}) {
    const {
      topK = 5,
      model = null
    } = options;
    
    // Generate embedding for the query
    const embedResult = await this.ollamaService.generateEmbedding(query);
    if (!embedResult.success) {
      return { 
        success: false, 
        error: 'Failed to generate query embedding' 
      };
    }
    
    // Search for relevant documents
    const searchResult = await this.vectorStoreService.search(
      embedResult.embedding,
      topK
    );
    
    if (!searchResult.success) {
      return { 
        success: false, 
        error: 'Failed to retrieve relevant documents' 
      };
    }
    
    // Prepare context from retrieved documents
    const context = searchResult.results
      .map(result => result.text)
      .join('\n\n---\n\n');
    
    // Create the prompt with context
    const messages = [
      {
        role: 'system',
        content: 'You are a helpful assistant that answers questions based on the provided context. If the information is not in the context, say you don\'t know.'
      },
      {
        role: 'user',
        content: `Use the following context to answer the question:

Context:
${context}

Question: ${query}`
      }
    ];
    
    // Return both the augmented messages and source references
    return {
      success: true,
      messages,
      sources: searchResult.results.map(result => ({
        text: result.text.substring(0, 100) + '...',
        metadata: result.metadata,
        score: result.score
      }))
    };
  }
}

module.exports = RAGService;
```

### 5. Update Chat API to Use RAG

Modify your chat endpoints to use RAG when appropriate:

```javascript
// In your chat route
router.post('/chat', async (req, res) => {
  const { message, sessionId, useRag = true } = req.body;
  
  if (useRag) {
    // Generate contextually-enhanced prompt using RAG
    const ragResult = await ragService.generateQueryWithContext(message);
    
    if (!ragResult.success) {
      // Fall back to regular chat if RAG fails
      // ...existing code...
    } else {
      // Use the RAG-enhanced messages for chat
      const chatResponse = await ollamaService.generateChatCompletion({
        messages: ragResult.messages,
        model: selectedModel
      });
      
      // Save message with source references
      const savedMessage = await chatbotService.saveMessage({
        // ...existing fields...
        retrieved_chunks: ragResult.sources
      });
      
      res.json({
        // ...existing response...
        sources: ragResult.sources
      });
    }
  } else {
    // Existing non-RAG chat code
  }
});
```

### 6. Update Frontend to Display Sources

Enhance `ChatMessage.tsx` to display source references:

```tsx
// Add to ChatMessage.tsx
{message.sources && (
  <div style={messageBubbleStyles.ai.sources}>
    <div style={messageBubbleStyles.ai.sourcesHeader}>
      Sources:
    </div>
    <div style={messageBubbleStyles.ai.sourcesList}>
      {message.sources.map((source, idx) => (
        <div key={idx} style={messageBubbleStyles.ai.sourceItem}>
          <span style={messageBubbleStyles.ai.sourceTitle}>
            {source.metadata.source}
          </span>
          <span style={messageBubbleStyles.ai.sourcePreview}>
            {source.text}
          </span>
        </div>
      ))}
    </div>
  </div>
)}
```

## Next Steps

1. Implement the ChromaDB vectorStoreService
2. Update the documentProcessor to store embeddings in ChromaDB
3. Create the RAG query service
4. Modify the chat API to use RAG
5. Enhance the UI to display source references
6. Add a toggle in the chat UI to enable/disable RAG

With these steps, you'll have a fully functional RAG system integrated with your current application. 