# Document Chunking Process in the RAG Pipeline

## Overview

The RAG (Retrieval-Augmented Generation) pipeline in this project involves several steps:
1. Document upload and text extraction
2. Text chunking (splitting text into smaller segments)
3. Embedding generation for chunks
4. Storage in vector database (ChromaDB)
5. Retrieval of relevant chunks for user queries

This document focuses specifically on the **chunking process**, which is a critical component of the RAG system.

## Chunking Functions

### Primary Chunking Implementation

The main chunking functionality is implemented in the `DocumentProcessor` class:

```javascript
// File: src/services/documentProcessor.js
async chunkText(text, fileType, chunkSize = 1000, chunkOverlap = 200) {
  if (!text || text.length === 0) {
    return [];
  }

  console.log(`Chunking text (${text.length} chars) with size=${chunkSize}, overlap=${chunkOverlap}`);

  // Try using LangChain's RecursiveCharacterTextSplitter if available
  if (RecursiveCharacterTextSplitter) {
    try {
      console.log(`Using LangChain RecursiveCharacterTextSplitter`);

      const splitter = new RecursiveCharacterTextSplitter({
        chunkSize: chunkSize,
        chunkOverlap: chunkOverlap,
        // These separators help ensure chunks break at natural boundaries
        separators: ["\n\n", "\n", ". ", "! ", "? ", ".", "!", "?", ";", ":", " ", ""]
      });

      // Split the text
      const langchainChunks = await splitter.createDocuments([text]);

      // Convert LangChain documents to our chunk format
      const chunks = langchainChunks.map((doc, index) => {
        return {
          text: doc.pageContent,
          index: index,
          length: doc.pageContent.length
        };
      });

      console.log(`LangChain text splitting created ${chunks.length} chunks`);
      return chunks;
    } catch (error) {
      console.warn(`LangChain text splitting failed: ${error.message}. Using fallback chunking method.`);
      // Fall through to custom implementation
    }
  }

  // Fallback to custom chunking implementation
  console.log(`Using custom text chunking method`);
  const chunks = [];
  let startIndex = 0;

  while (startIndex < text.length) {
    // Calculate end index for this chunk
    let endIndex = Math.min(startIndex + chunkSize, text.length);

    // If we're not at the end of the text, try to find a good breaking point
    if (endIndex < text.length) {
      // Look for paragraph breaks first
      const paragraphBreak = text.lastIndexOf('\n\n', endIndex);
      if (paragraphBreak > startIndex && paragraphBreak > endIndex - 200) {
        endIndex = paragraphBreak + 2; // +2 to include the newlines
      } else {
        // Look for sentence endings (.!?)
        const sentenceMatch = text.substring(endIndex - 100, endIndex + 100).match(/[.!?]\s/);
        if (sentenceMatch) {
          const matchIndex = sentenceMatch.index;
          endIndex = endIndex - 100 + matchIndex + 2; // +2 to include punctuation and space
        } else {
          // Look for a space as last resort
          const lastSpace = text.lastIndexOf(' ', endIndex);
          if (lastSpace > startIndex) {
            endIndex = lastSpace + 1;
          }
        }
      }
    }

    // Extract the chunk
    const chunk = text.substring(startIndex, endIndex).trim();

    // Add to chunks array if not empty
    if (chunk.length > 0) {
      chunks.push({
        text: chunk,
        index: chunks.length,
        length: chunk.length
      });
    }

    // Move to next chunk, accounting for overlap
    startIndex = Math.max(startIndex + 1, endIndex - chunkOverlap);
  }

  console.log(`Custom text chunking created ${chunks.length} chunks`);
  return chunks;
}
```

## Files Containing Chunking Logic

The chunking functionality is implemented in:

1. **Primary Implementation**: `src/services/documentProcessor.js`
   - Contains `chunkText()` method which performs the actual chunking
   - Has both LangChain-based and custom fallback implementations

2. **Alternative Implementation**: `documentation/RAGINTEGRATION.md`
   - Contains a simpler `chunkText()` method in the `DocumentProcessingService` class
   - This appears to be documentation/example code rather than active code

## How Chunking is Triggered

Chunking is triggered during document processing in the RAG pipeline:

```javascript
// File: src/services/documentProcessor.js
async processDocument(document, options = {}) {
  try {
    // ... initialization and logging ...

    // Update progress to indicate start
    await this.updateDocumentProgress(document.id, {
      status: 'processing',
      progress: 10,
      message: 'Started document processing'
    });

    // Extract text from the document
    const textResult = await this.extractText(document);
    if (!textResult.success) {
      console.error(`Failed to extract text: ${textResult.error}`);
      return textResult;
    }

    // Update progress after text extraction
    await this.updateDocumentProgress(document.id, {
      progress: 30,
      message: 'Text extracted, chunking document'
    });

    // Split text into chunks
    const chunks = await this.chunkText(textResult.text, document.file_type);
    console.log(`Document ${document.id} chunked into ${chunks.length} segments`);

    // Update progress after chunking
    await this.updateDocumentProgress(document.id, {
      progress: 50,
      message: 'Document chunked, generating embeddings'
    });

    // Generate embeddings for the chunks
    const embeddingsResult = await this.generateEmbeddings(chunks, document.id, userId, sessionId);
    
    // ... rest of the method ...
  } catch (error) {
    // ... error handling ...
  }
}
```

The chunking process is part of a larger workflow:
1. User uploads a document (via web UI or API)
2. Text is extracted from the document (PDF, DOCX, etc.)
3. Text is chunked into smaller segments
4. Embeddings are generated for each chunk
5. Chunks and embeddings are stored in ChromaDB

## Chunking Strategy

The implementation uses two different chunking strategies:

### 1. Primary Strategy: LangChain's RecursiveCharacterTextSplitter

If available, the system uses LangChain's `RecursiveCharacterTextSplitter` with the following configuration:
- **chunkSize**: 1000 characters (default)
- **chunkOverlap**: 200 characters (default)
- **Separator Tokens**: 
  ```javascript
  ["\n\n", "\n", ". ", "! ", "? ", ".", "!", "?", ";", ":", " ", ""]
  ```

This splitter tries to break text at natural boundaries in the order specified in the separators array. It first tries to break at paragraph boundaries ("\n\n"), then at line breaks ("\n"), then at sentence endings, and so on.

### 2. Fallback Strategy: Custom Character-Based Chunking

If LangChain is not available or fails, the system uses a custom chunking implementation:
- **Character-based**: Uses a target size of 1000 characters with 200 character overlap
- **Intelligent Boundaries**: Tries to find natural break points in this order:
  1. Paragraph breaks (`\n\n`)
  2. Sentence endings (`.`, `!`, `?` followed by a space)
  3. Word boundaries (spaces)
- **Overlap Handling**: Ensures chunks overlap by approximately 200 characters to maintain context across chunks

## Output Format

Each chunk produced by the chunking process has the following structure:

```javascript
{
  text: "The actual text content of the chunk...",
  index: 0,  // Sequential index of the chunk in the document
  length: 985  // Character length of the chunk
}
```

These chunks are then passed to the embedding generation phase, where each chunk is embedded separately.

## Downstream Usage

After chunks are created:

1. **Embedding Generation**: 
   - Each chunk's text is passed to Ollama for embedding generation
   - The `generateEmbeddings()` method in `documentProcessor.js` handles this

2. **Vector Store Storage**:
   - Chunks and their embeddings are stored in ChromaDB via `vectorStoreService.js`
   - Metadata including document ID, file name, and session ID is attached

3. **Retrieval During Query**:
   - When a user asks a question, their query is embedded
   - Vector similarity search finds the most relevant chunks
   - The `search()` method in `vectorStoreService.js` performs this search
   - Retrieved chunks become context for the LLM response

## Considerations for Improvement

The current chunking strategy has several areas for potential enhancement:

1. **Semantic Chunking**: The current approach is primarily character-based with some awareness of text structure. True semantic chunking would consider meaning and could produce more coherent chunks.

2. **Section-Based Chunking**: For structured documents, chunking based on sections, headers, or document structure could be more effective.

3. **Adaptive Chunk Size**: Currently uses fixed chunk sizes. An adaptive approach could use larger chunks for simple text and smaller chunks for dense, information-rich text.

4. **Document Type Awareness**: While the `fileType` parameter is passed to `chunkText()`, it doesn't currently influence the chunking strategy.

5. **Token-Based Chunking**: The current implementation is character-based. A token-based approach might be more aligned with how LLMs process text. 