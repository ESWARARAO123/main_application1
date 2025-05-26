# Retrieval-Augmented Generation (RAG) System Explanation

## Introduction

The Retrieval-Augmented Generation (RAG) system in this application enhances AI responses by incorporating relevant information from user-uploaded documents. This document explains how the system works, from document upload to generating AI responses with contextual document references.

## How RAG Works in Our Application

### The User Experience

When using our application with RAG:

1. Users upload documents through the chat interface or documents page
2. Documents are processed in the background (text extraction, chunking, embedding)
3. Users can ask questions about their documents
4. The system retrieves relevant information from the uploaded documents
5. The AI generates responses based on both its general knowledge and the specific document content
6. Users see responses with citations to the source documents

### Document Processing Workflow

#### 1. Document Upload and Initial Processing

When a user uploads a document:

- The upload is received by either `/api/documents/upload` or `/message-with-file` endpoints in `documents.js` or `chatbot.js`
- A document record is created in the database via `documentService.createDocument()`
- The system begins asynchronous processing with `documentService.processDocument()`

#### 2. Text Extraction

The `documentProcessor.js` service handles extracting text from various file formats:

- **PDF Files**: Uses LangChain's PDFLoader or falls back to pdf-parse
- **DOCX Files**: Uses LangChain's DocxLoader or falls back to mammoth
- **Text Files**: Reads directly using Node.js file system

This extraction turns documents of different formats into plain text that can be processed further.

#### 3. Text Chunking

After extraction, the text is split into manageable chunks:

- The system uses `documentProcessor.chunkText()` method
- Default chunk size is 1000 characters with 200 character overlap
- The system tries to break at natural boundaries (paragraphs, sentences)
- LangChain's RecursiveCharacterTextSplitter is used when available

Chunking is critical for RAG because:
- It breaks documents into focused semantic units
- It creates pieces small enough for effective vector search
- Overlapping ensures concepts spanning chunk boundaries aren't lost

#### 4. Embedding Generation

Each text chunk is converted into a numerical vector representation (embedding):

- `documentProcessor.generateEmbeddings()` processes all chunks in a document
- The system connects to Ollama API via `ollamaService.generateEmbedding()`
- Default embedding model is "nomic-embed-text" (768 dimensions)
- If Ollama is unavailable, the system falls back to random placeholder embeddings

These embeddings represent the semantic meaning of text in a format that enables similarity search.

#### 5. Vector Storage

Embeddings are stored in a vector database for efficient similarity search:

- `vectorStoreService.addDocumentChunks()` handles storage
- Primary storage is ChromaDB running in a Docker container
- Each chunk is stored with metadata (document ID, filename, user ID, session ID)
- A file-based fallback system exists if ChromaDB is unavailable

## The Semantic Search Process

When a user asks a question, the system performs semantic search to find relevant information:

### 1. Query Embedding

The user's question is converted to an embedding vector:

- `ragService.retrieveContext()` initiates the process
- The question text is sent to `ollamaService.generateEmbedding()`
- Same embedding model is used for both documents and queries (consistency is important)

### 2. Vector Similarity Search

The system searches for document chunks similar to the question:

- `vectorStoreService.search()` finds chunks with similar meaning
- Uses cosine similarity between the question embedding and stored chunk embeddings
- Returns the top K most similar chunks (default is 5)
- Filters results by session ID when appropriate

This is "semantic search" because it matches based on meaning rather than keywords. For example:
- A query about "climate impacts" might match document chunks discussing "environmental effects" even if the exact words aren't used
- The vector representation captures the semantic relationships between concepts

### 3. Context Formation

The retrieved chunks are assembled into a context for the AI:

- Multiple relevant chunks are combined with separators
- Source information is preserved for citation
- This context becomes part of the prompt to the AI model

## The RAG Response Generation

### 1. Enhanced Prompt Creation

The system creates a specialized prompt combining:

- A system message defining the assistant's role
- The retrieved document context
- The user's original question

```
You are a helpful assistant that answers questions based on the provided context...

Context:
[Retrieved document chunks]

Question: [User's question]
```

### 2. AI Response Generation

The enhanced prompt is sent to the AI model:

- `ragService.processRagChat()` orchestrates this process
- The prompt is passed to `ollamaService.chat()` for response generation
- The AI generates a response based on both its training and the specific context

### 3. Response Formatting and Delivery

The system returns the AI response along with source information:

- The response is sent back to the client with metadata about the source documents
- The client displays the response with citation information
- Users can see which documents and passages informed the answer

## Component Roles in the RAG System

### Server-Side Components

1. **documentService.js**:
   - Manages document metadata in the database
   - Coordinates the document processing workflow
   - Provides status updates during processing

2. **documentProcessor.js**:
   - Extracts text from different file formats
   - Splits text into semantic chunks
   - Generates embeddings for chunks

3. **vectorStoreService.js**:
   - Manages connections to ChromaDB
   - Stores document chunks and embeddings
   - Performs similarity search for retrieval
   - Handles session-specific document management

4. **ollamaService.js**:
   - Connects to the Ollama API
   - Generates embeddings for documents and queries
   - Sends enhanced prompts to the AI model
   - Processes streaming responses

5. **ragService.js**:
   - Orchestrates the RAG workflow
   - Retrieves relevant context for questions
   - Formats prompts with contextual information
   - Manages the response generation process

6. **API Routes**:
   - **documents.js**: Handles document upload and status checking
   - **documents-status.js**: Provides document processing status updates
   - **ollama.js**: Exposes RAG-specific endpoints
   - **chatbot.js**: Integrates RAG with the chat interface

### Client-Side Components

1. **ragChatService.ts**:
   - Provides client-side access to RAG functionality
   - Checks RAG availability
   - Sends RAG-enhanced chat requests

2. **ollamaService.ts**:
   - Manages Ollama settings and models
   - Tests connection to Ollama server

3. **Chatbot.tsx**:
   - Implements the chat interface
   - Toggles between regular chat and RAG mode
   - Displays RAG responses with source citations
   - Handles document uploads in chat context

## Session-Based Context Management

Our RAG implementation has a unique feature: session-specific document context.

When a user uploads a document within a chat session:

1. The document is associated with that specific session
2. Previous documents for that session are cleared to prevent context confusion
3. Vector searches are filtered to only include documents from the current session

This enables:
- Topic-specific chat sessions with relevant documents
- Multiple users with their own document contexts
- Clean contextual boundaries between different conversations

## Technical Implementation Details

### Vector Database Configuration

ChromaDB runs as a Docker container with these settings:

- Host: 0.0.0.0
- Port: 8000
- Data persistence via Docker volume
- CORS enabled for web client access

### Embedding Model

The default embedding model has these characteristics:

- Model: "nomic-embed-text"
- Dimensions: 768
- Processing: Batch size of 5-10 chunks to avoid overloading

### Fallback Mechanisms

The system includes several fallbacks for robustness:

- File-based vector storage if ChromaDB is unavailable
- Alternative text extraction methods if primary methods fail
- Placeholder embeddings if Ollama embedding service is down
- Regular chat if no relevant document context is found

## Conclusion

The RAG system in our application creates a seamless bridge between document knowledge and AI capabilities. Through a carefully designed pipeline of document processing, embedding generation, semantic search, and prompt enhancement, users can interact with their documents in a conversational way.

The modularity of the system allows for future enhancements, such as:

- Support for additional document formats
- Integration of different embedding models
- More sophisticated retrieval mechanisms
- Advanced citation and source tracking

This approach to information retrieval and generation represents a powerful paradigm for knowledge-based AI interactions, going beyond simple chatbots to create truly context-aware assistants. 