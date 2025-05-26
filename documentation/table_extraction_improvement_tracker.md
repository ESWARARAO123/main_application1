# Table Extraction Improvement Tracker

## Current Implementation Assessment

### Strengths
- Successfully using `pdfplumber` to extract both text and tables from PDFs
- Converting tables to Markdown format for better preservation of structure
- Tables are integrated into the extracted text with clear markers
- Header-based chunking preserves document structure better than character-based chunking
- Chunking effectively preserves section titles as metadata

### Observed Issues
- Table retrieval appears inconsistent based on user's example query
- Only 2 results returned for a specific table query after filtering
- Possible loss of table context during chunking or embedding
- Tables may be split across chunks, reducing retrieval effectiveness

## Root Cause Analysis

1. **Table Extraction Quality**
   - Tables may not be properly identified by pdfplumber in complex documents
   - Some tables might be detected as images rather than structured data
   - Complex tables with merged cells might not convert cleanly to Markdown

2. **Chunking Issues**
   - Header-based chunking might separate tables from their contextual headers
   - Large tables might be split across multiple chunks
   - Table titles might not be properly preserved as metadata

3. **Retrieval Challenges**
   - Embedding models may not effectively capture tabular relationships
   - Cosine similarity search might not be optimal for finding structured tabular data
   - Session-based filtering might be too restrictive

## Improvement Plan

### 1. Enhance Table Extraction
- [x] Modify `extract_text_with_tables.py` to improve table boundary detection
- [x] Add special handling for complex tables with merged cells
- [x] Include table titles and captions in the extracted content
- [x] Implement table numbering that matches the original document

### 2. Optimize Chunking for Tables
- [x] Modify header chunker to keep tables intact with their headers
- [x] Add special chunking logic for table content to prevent table splitting
- [x] Ensure table captions/identifiers are included in chunk metadata
- [x] Implement "table-aware" chunking that treats tables as atomic units

### 3. Improve Retrieval
- [x] Increase default search result limit from 5 to improve recall
- [x] Experiment with alternative similarity metrics beyond cosine
- [x] Add table-specific metadata fields for improved filtering
- [ ] Implement hybrid search combining keyword and semantic search

### 4. Validation and Testing
- [ ] Create test suite with various table formats to validate extraction
- [ ] Implement metrics to measure table extraction accuracy
- [ ] Add automated tests to verify table integrity after chunking
- [ ] Create benchmark queries specifically for table content

### 5. Project Organization Improvements
- [x] Move Python scripts to dedicated RAG-MODULE folder for better organization
- [x] Update script paths in Node.js code
- [x] Update documentation to reflect new file locations
- [x] Ensure consistent script references across the system
- [x] Update installvenv.sh and test_extraction.sh scripts for new folder structure

## Specific Technical Tasks

### Short-term (1-2 weeks)
1. Add table detection validation in `extract_text_with_tables.py`
   ```python
   def validate_table_extraction(table):
       # Check if table has expected structure
       # Verify column headers are present
       # Return validation metrics
   ```

2. Modify table markers to include table numbers matching document
   ```python
   # Instead of: "### Extracted Table 1 from Page 3"
   # Use: "### Table 2-2: LPDDR4 Single Channel DFIW-to-DRAM Command/Address Pin Map (Page 3)"
   ```

3. Implement table-preserving chunking logic
   ```javascript
   function preserveTablesInChunks(chunks) {
     // Identify chunks containing table markers
     // Ensure tables aren't split across chunks
     // Return optimized chunks
   }
   ```

4. Increase search result limit and add post-processing
   ```javascript
   // Increase from current 5 results
   const searchOptions = {
     limit: 10,
     sessionId: sessionId
   };
   ```

### Medium-term (1-2 months)
1. Implement hybrid search that combines semantic and keyword matching
2. Add table-specific embeddings that better capture tabular relationships
3. Develop more sophisticated table parsing for complex formats
4. Create a table validation framework with accuracy metrics

## Progress Tracking

| Date       | Issue Addressed                  | Solution Implemented               | Results                           |
|------------|----------------------------------|-----------------------------------|-----------------------------------|
| 2023-06-15 | Limited search results           | Increased default limit to 10     | Improved recall for table queries |
| 2023-06-15 | Poor table retrieval             | Added table-specific query expansion | Better prioritization of table content |
| 2023-06-15 | Missing table captions           | Enhanced caption detection in extract_text_with_tables.py | Tables now retain original numbering |
| 2023-06-15 | Tables split across chunks       | Implemented table-aware chunking  | Tables now preserved as atomic units |
| 2023-06-16 | Disorganized Python scripts      | Moved scripts to RAG-MODULE folder| Better code organization and maintainability |
| 2023-06-16 | Inconsistent script paths        | Updated install and test scripts  | Improved virtual env setup and testing |
| YYYY-MM-DD | Poor table caption recognition   | Enhanced caption extraction       | Pending                           |

## Success Metrics
- At least 90% of tables correctly identified and extracted
- Tables consistently retrievable via semantic search
- Table references (e.g., "Table 2-2") correctly matched to their content
- User queries about specific tables consistently return relevant results

## Implemented Improvements

### 1. Increased Search Result Limit
Modified `vectorStoreService.js` to increase the default search result limit from 5 to 10:
```javascript
// Previous implementation
async search(queryEmbedding, limit = 5, options = {}) {
  // ...
}

// New implementation
async search(queryEmbedding, options = {}) {
  const limit = options.limit || 10; // Increased from 5 to 10
  // ...
}
```

### 2. Table-Specific Query Expansion
Enhanced `ragService.js` to detect and boost table-related queries:
```javascript
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

// Boost scores for chunks that contain table markers
results = results.map(result => {
  // If it has a table reference and matches the one in query, boost the score significantly
  if (tableReference && result.text.toLowerCase().includes(tableReference.toLowerCase())) {
    return {
      ...result,
      score: result.score * 1.5, // 50% boost
      containsRequestedTable: true
    };
  }
  // ...
});
```

### 3. Enhanced Table Caption Detection
Improved `extract_text_with_tables.py` to detect and preserve original table numbering:
```python
def detect_table_caption(text_before_table: str, page_num: int, max_chars_to_scan: int = 500) -> Optional[str]:
    """Detect and extract table caption from text preceding the table."""
    # Look for patterns like "Table 2-2: Description" or "Table 2.2 Description"
    caption_patterns = [
        r"Table\s+(\d+[-\.]\d+)[\s\:]+([^\n]+)",  # Table 2-2: Description
        r"Table\s+(\d+)[\s\:]+([^\n]+)",          # Table 2: Description
        # ...
    ]
    
    # Return detected caption or fallback
    # ...
```

### 4. Table-Aware Chunking
Modified `headerChunker.js` to implement table preservation:
```javascript
// Function to detect tables in the text
function detectTables(text) {
  const foundTables = [];
  // Match both new format ("### Table 2-2: Description") and old format
  const tablePattern = /###\s+(Table\s+\d+(?:[-\.]\d+)?(?:\s*:\s*[^\n]+)?|Extracted Table \d+ from Page \d+)\n\n(\|[^\n]*\|\n\|[-:\s|]*\|\n(?:\|[^\n]*\|\n)+)/g;
  // ...
}

// Process chunks that exceed the size limit
const finalChunks = [];
for (const chunk of chunks) {
  if (chunk.text.length > chunkSize && !chunk.hasTables) {
    // Only split non-table chunks
    // ...
  }
}
```

### 5. Python Script Reorganization
Moved Python scripts to a dedicated RAG-MODULE folder and updated paths in Node.js files:

```javascript
// Updated in pdfProcessor.js
const pythonScriptPath = path.resolve(process.cwd(), 'python/RAG-MODULE/extract_text_with_tables.py');

// Updated in documentProcessor.js
const pythonScriptPath = path.resolve(process.cwd(), 'python/RAG-MODULE/extract_text.py');
```

### 6. Updated Installation and Testing Scripts
Modified installation and testing scripts to support the new directory structure:

```bash
# Updated directory structure in installvenv.sh
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"
VENV_PATH="${PARENT_DIR}/venv"

# Added support for both extraction scripts in test_extraction.sh
EXTRACT_SCRIPT="${SCRIPT_DIR}/extract_text.py"
EXTRACT_TABLES_SCRIPT="${SCRIPT_DIR}/extract_text_with_tables.py"
```

These improvements organize the codebase better and ensure the Python virtual environment is properly set up and shared between scripts. 