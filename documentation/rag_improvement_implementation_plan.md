# RAG Improvement Implementation Plan

Based on our tracker file analysis, this document outlines specific implementation steps to enhance our RAG system with a focus on improving query search effectiveness.

## 1. Immediate Search Improvements (Days 1-7)

### Increase Search Result Limit
```javascript
// In vectorStoreService.js
async function search(embeddingVector, options = {}) {
  const limit = options.limit || 10; // Increase from 5 to 10
  
  // Rest of the search function
}
```

### Add Query Expansion for Table Searches
```javascript
// In ragService.js
async function retrieveContext(query, options) {
  // Original query embedding
  const queryEmbedding = await ollamaService.generateEmbedding(query);
  
  // Table-specific query expansion
  let tableQuery = query;
  if (query.toLowerCase().includes('table')) {
    // Extract table references like "Table 2-2"
    const tableRefMatch = query.match(/table\s+([0-9]+[-\.][0-9]+)/i);
    if (tableRefMatch) {
      tableQuery = `${tableRefMatch[0]} ${query}`;
    }
  }
  
  // Search with both queries and merge results
  const results = await vectorStoreService.search(queryEmbedding, {
    ...options,
    limit: 10,
    tableQuery: tableQuery
  });
  
  return results;
}
```

## 2. Table Extraction Enhancements (Days 8-14)

### Improve Table Caption Detection
```python
# In extract_text_with_tables.py
def detect_table_caption(text_before_table, page_number):
    """Extract table caption from text preceding the table"""
    # Look for patterns like "Table 2-2: Description" or "Table 2.2 Description"
    caption_pattern = r"Table\s+(\d+[-\.]\d+)[\s\:]+(.*?)(?:\n|$)"
    match = re.search(caption_pattern, text_before_table, re.IGNORECASE)
    
    if match:
        table_number = match.group(1)
        description = match.group(2).strip()
        return f"Table {table_number}: {description}"
    else:
        return f"Extracted Table from Page {page_number}"
```

### Preserve Table Structure
```python
def convert_table_to_markdown(table, caption=""):
    """Enhanced table conversion with better structure preservation"""
    # Add caption as a header
    markdown = f"### {caption}\n\n"
    
    # Check if first row looks like a header
    has_header = _is_header_row(table[0])
    
    # Generate column alignments based on data types
    alignments = _generate_column_alignments(table)
    
    # Convert table with proper alignment and header
    # ...existing conversion code...
    
    # Add table metadata as HTML comment for post-processing
    markdown += f"\n<!-- TABLE_METADATA: {json.dumps({'caption': caption})} -->\n"
    
    return markdown
```

## 3. Chunking Optimizations (Days 15-21)

### Table-Aware Chunking
```javascript
// In headerChunker.js
function chunkBySection(text, options = {}) {
  // Existing section detection code...
  
  // Add table detection
  const tablePattern = /###\s+Table\s+\d+[-\.]\d+.*?\n\n\|.*?\|(?:\n\|.*?\|)+/gs;
  const tables = [];
  let match;
  
  // Extract all tables
  while ((match = tablePattern.exec(text)) !== null) {
    tables.push({
      content: match[0],
      start: match.index,
      end: match.index + match[0].length
    });
  }
  
  // Modify chunking to keep tables intact
  for (const section of sections) {
    // Check if section contains tables
    const sectionTables = tables.filter(t => 
      t.start >= section.start && t.end <= section.end);
    
    if (sectionTables.length > 0) {
      // Ensure chunks don't break tables
      // Implementation logic that preserves tables as atomic units
    }
  }
  
  return chunks;
}
```

## 4. Hybrid Search Implementation (Days 22-30)

### Keyword + Semantic Search
```javascript
// In vectorStoreService.js
async function hybridSearch(query, embeddingVector, options = {}) {
  // Semantic search using embeddings
  const semanticResults = await chromaDB.query({
    queryEmbeddings: [embeddingVector],
    nResults: options.limit || 10,
    // other params
  });
  
  // Keyword search (using metadata or text search capabilities)
  const keywordResults = await performKeywordSearch(query, options);
  
  // Combine and re-rank results
  const combinedResults = mergeAndRankResults(semanticResults, keywordResults);
  
  return combinedResults.slice(0, options.limit || 10);
}

function mergeAndRankResults(semanticResults, keywordResults) {
  // Combine results with a scoring function that considers:
  // 1. Semantic similarity score
  // 2. Keyword match count/relevance
  // 3. Presence of table markers for table queries
  // 4. Section hierarchy relevance
  
  // Return sorted by combined score
}
```

## 5. Table-Specific Metadata (Days 31-40)

### Enhanced Metadata Schema
```javascript
// In documentProcessor.js
function processChunk(chunk, document, options = {}) {
  // Extract table metadata if present
  const tableMetadata = extractTableMetadata(chunk);
  
  // Create enhanced metadata object
  const metadata = {
    documentId: document.id,
    fileName: document.filename,
    userId: options.userId,
    sessionId: options.sessionId,
    // Add table-specific metadata
    containsTable: !!tableMetadata,
    tableNumber: tableMetadata?.tableNumber || null,
    tableCaption: tableMetadata?.caption || null,
    tableColumns: tableMetadata?.columns || null,
  };
  
  return {
    text: chunk,
    metadata: metadata
  };
}
```

## 6. Testing & Validation (Throughout)

### Query Test Suite
Create a suite of specific test queries focused on tables with expected results:

```javascript
const tableQueries = [
  {
    query: "What information is in Table 2-2?",
    expectedTableNumbers: ["2-2"],
    minExpectedResults: 1
  },
  {
    query: "List the column headers in the LPDDR4 Single Channel table",
    expectedTableNumbers: ["2-2"],
    expectedContent: ["DFIW", "Signal", "Channel"]
  },
  // Add more test cases
];

async function runQueryTests() {
  const results = [];
  for (const test of tableQueries) {
    const searchResults = await ragService.retrieveContext(test.query);
    
    results.push({
      query: test.query,
      passed: validateResults(searchResults, test),
      actualResults: searchResults
    });
  }
  
  return results;
}
```

## Implementation Timeline

| Week | Focus Area | Key Deliverables |
|------|------------|------------------|
| 1    | Search Limits & Query Expansion | Increased result limit, basic query expansion for tables |
| 2    | Table Extraction | Improved caption detection, better structure preservation |
| 3    | Chunking | Table-aware chunking implementation |
| 4-5  | Hybrid Search | Combined semantic + keyword search with re-ranking |
| 6    | Metadata & Retrieval | Enhanced metadata schema, filtered retrieval |

## Prioritized Approach

1. Start with search enhancements first (limit increase, basic query expansion) as these provide the quickest improvements with minimal code changes
2. Add table caption detection and improved structure next to make tables more identifiable
3. Implement table-aware chunking to ensure tables stay together
4. Develop hybrid search for more robust retrieval capabilities
5. Finally, enhance metadata to enable more sophisticated filtering and retrieval

By following this implementation plan, we'll systematically address the key issues identified in our tracker while focusing on improving search effectiveness for table content. 