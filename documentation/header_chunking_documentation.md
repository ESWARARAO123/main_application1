# Header-Based Document Chunking for RAG

## Overview

The header-based chunking feature enhances the Retrieval-Augmented Generation (RAG) pipeline by intelligently splitting documents based on their section structure. This approach produces more semantically coherent chunks that align with the document's natural organization, resulting in improved retrieval accuracy and context preservation.

## Problem Solved

Traditional character-based chunking methods often break documents at arbitrary points, which can:
- Split related content across different chunks
- Separate headers from their content
- Result in contextually disconnected chunks
- Reduce retrieval accuracy for structured documents

Header-based chunking solves these issues by respecting the document's inherent structure, keeping related content together, and preserving the hierarchical organization of information.

## Implementation Details

### Core Components

1. **Header Detection**: 
   - Uses regular expressions to identify section headers (e.g., "1.2 Section Title")
   - Default pattern: `/^\s*(\d+(?:\.\d+)*)\s+([A-Z][^\n]{3,})$/gm`
   - Captures hierarchical numbering (1, 1.1, 1.1.2) and section titles

2. **Section-Based Chunking**:
   - Groups content between headers into coherent sections
   - Preserves section hierarchy and relationships
   - Retains section titles as metadata for improved context

3. **Adaptive Size Management**:
   - Handles large sections by further splitting them using paragraph boundaries
   - Ensures chunks don't exceed maximum size limits
   - Maintains intelligent overlap between chunks

4. **Metadata Enrichment**:
   - Attaches section titles to chunks as metadata
   - Records character positions for traceability
   - Preserves section hierarchy information

### Integration with Existing Pipeline

The header-based chunking is integrated into the document processing pipeline:

1. Document text is extracted as before
2. The chunker analyzes the text for section headers
3. Content is split according to the section structure
4. Each chunk includes its section title as metadata
5. Embeddings are generated for the chunks as before
6. Chunks with metadata are stored in the vector database

## Benefits

### Improved Retrieval Accuracy

- More semantically coherent chunks result in better vector representations
- Section titles provide valuable context to the embedding model
- Related information stays together, improving context availability

### Enhanced User Experience

- Responses include more relevant and complete information from the documents
- Citations can reference specific sections rather than arbitrary chunks
- Users receive answers that better align with document structure

### Technical Advantages

- Better handling of structured technical documents like whitepapers and documentation
- More efficient use of token limits in context windows
- Improved performance for documents with complex hierarchical structures

## Usage Example

When processing a technical whitepaper with sections like:

```
1 Introduction
1.1 Purpose
1.2 Scope
2 System Architecture
2.1 High-Level Components
...
```

The system will create chunks that preserve this structure:

```javascript
[
  {
    text: "1 Introduction\n\nThis document describes...",
    sectionTitle: "1 Introduction",
    index: 0,
    // other metadata
  },
  {
    text: "1.1 Purpose\n\nThe purpose of this document...",
    sectionTitle: "1.1 Purpose",
    index: 1,
    // other metadata
  },
  // ...
]
```

This results in more natural and meaningful chunks that better preserve the document's intent and organization.

## Configuration Options

The header chunking system supports several configuration options:

| Option | Description | Default |
|--------|-------------|---------|
| `chunkSize` | Maximum size of a chunk in characters | 1000 |
| `overlap` | Character overlap between adjacent chunks | 200 |
| `headerRegex` | Custom regex pattern for header detection | `/^\s*(\d+(?:\.\d+)*)\s+([A-Z][^\n]{3,})$/gm` |

## Limitations and Future Improvements

- Currently optimized for documents with numbered section headers
- Could be extended to detect non-numbered headers and alternative formats
- Future versions may implement more intelligent split points for very large sections
- Could be enhanced with machine learning for header detection in unstructured documents 