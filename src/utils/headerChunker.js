/**
 * Header-Based Text Chunker
 * Splits documents into chunks based on section headers for improved RAG retrieval
 */

/**
 * Splits text into chunks based on section headers
 * @param {string} text - The document text to chunk
 * @param {Object} options - Chunking options
 * @param {number} options.chunkSize - Maximum chunk size in characters (default: 1000)
 * @param {number} options.overlap - Overlap between chunks in characters (default: 200)
 * @param {string} options.headerRegex - Custom regex for header detection
 * @returns {Array<Object>} Array of chunks with metadata
 */
function chunkBySection(text, options = {}) {
  if (!text || typeof text !== 'string' || text.length === 0) {
    return [];
  }

  const {
    chunkSize = 1000,
    overlap = 200,
    headerRegex = /^\s*(\d+(?:\.\d+)*)\s+([A-Z][^\n]{3,})$/gm
  } = options;

  // First, detect tables in the document so we can keep them intact
  const tables = detectTables(text);
  console.log(`Detected ${tables.length} tables in the document`);

  // Split text into lines for processing
  const lines = text.split('\n');
  const chunks = [];
  let currentChunk = {
    text: '',
    sectionTitle: 'Introduction', // Default title for content before first header
    index: 0,
    startChar: 0,
    endChar: 0,
    containsTables: []
  };
  
  let currentSectionTitle = 'Introduction';
  let currentPosition = 0;
  let inSection = false;
  
  // Helper function to check if a line matches header pattern
  function isHeader(line) {
    // Reset regex lastIndex to ensure it works correctly
    headerRegex.lastIndex = 0;
    return headerRegex.test(line);
  }

  // Helper function to finalize and add a chunk
  function finalizeChunk(chunk, endPosition) {
    if (chunk.text.trim().length > 0) {
      // Update the end character position
      chunk.endChar = endPosition - 1;
      
      // Check if this chunk contains any tables
      chunk.containsTables = tables.filter(table => 
        // Table starts within this chunk
        (table.startPos >= chunk.startChar && table.startPos <= chunk.endChar) ||
        // Table ends within this chunk
        (table.endPos >= chunk.startChar && table.endPos <= chunk.endChar) ||
        // Table spans across this chunk
        (table.startPos < chunk.startChar && table.endPos > chunk.endChar)
      );
      
      // Add metadata about tables
      if (chunk.containsTables.length > 0) {
        chunk.hasTables = true;
        const tableNumbers = chunk.containsTables
          .map(t => t.caption.match(/Table\s+(\d+[-\.]\d+)/i))
          .filter(Boolean)
          .map(m => m[1]);
        
        if (tableNumbers.length > 0) {
          chunk.tableNumbers = tableNumbers;
        }
      }
      
      chunks.push({...chunk});
      console.log(`Created chunk for section: ${chunk.sectionTitle} (${chunk.text.length} chars)${chunk.containsTables.length > 0 ? ', contains tables' : ''}`);
    }
  }

  // Helper function to split a large section into smaller chunks
  function splitLargeSection(sectionText, sectionTitle, startPosition) {
    // Check if this section contains tables
    const sectionTables = tables.filter(table => 
      (table.startPos >= startPosition && table.startPos < startPosition + sectionText.length) || 
      (table.endPos >= startPosition && table.endPos < startPosition + sectionText.length)
    );
    
    // If it has tables, use table-aware splitting
    if (sectionTables.length > 0) {
      return splitSectionWithTables(sectionText, sectionTitle, startPosition, sectionTables);
    }
    
    // If the section is small enough, return it as a single chunk
    if (sectionText.length <= chunkSize) {
      return [{
        text: sectionText,
        sectionTitle,
        index: chunks.length,
        startChar: startPosition,
        endChar: startPosition + sectionText.length - 1,
        containsTables: []
      }];
    }
    
    // Otherwise, split it further by paragraphs
    const paragraphs = sectionText.split('\n\n');
    const subChunks = [];
    let currentSubChunk = {
      text: '',
      sectionTitle,
      index: chunks.length,
      startChar: startPosition,
      endChar: 0,
      containsTables: []
    };
    let currentLength = 0;
    let currentStartChar = startPosition;
    
    for (const paragraph of paragraphs) {
      // If adding this paragraph would exceed chunk size, finalize current chunk
      if (currentLength + paragraph.length > chunkSize && currentLength > 0) {
        currentSubChunk.endChar = currentStartChar + currentLength - 1;
        subChunks.push({...currentSubChunk});
        
        // Start a new subchunk with overlap
        const overlapText = currentSubChunk.text.substring(
          Math.max(0, currentSubChunk.text.length - overlap)
        );
        
        currentStartChar = currentSubChunk.endChar - overlapText.length + 1;
        currentSubChunk = {
          text: overlapText,
          sectionTitle: `${sectionTitle} (continued)`,
          index: chunks.length + subChunks.length,
          startChar: currentStartChar,
          endChar: 0,
          containsTables: []
        };
        currentLength = overlapText.length;
      }
      
      // Add paragraph to current subchunk
      currentSubChunk.text += (currentLength > 0 ? '\n\n' : '') + paragraph;
      currentLength += (currentLength > 0 ? 2 : 0) + paragraph.length;
    }
    
    // Add the last subchunk if it has content
    if (currentSubChunk.text.trim().length > 0) {
      currentSubChunk.endChar = currentStartChar + currentLength - 1;
      subChunks.push(currentSubChunk);
    }
    
    return subChunks;
  }
  
  // Helper function to split a section containing tables
  function splitSectionWithTables(sectionText, sectionTitle, startPosition, sectionTables) {
    // If this section's length (including tables) is under the chunk size, keep it whole
    if (sectionText.length <= chunkSize * 1.5) { // Allow up to 50% larger chunks for tables
      return [{
        text: sectionText,
        sectionTitle,
        index: chunks.length,
        startChar: startPosition,
        endChar: startPosition + sectionText.length - 1,
        containsTables: sectionTables,
        hasTables: true
      }];
    }
    
    // Otherwise, we need to split around tables
    const subChunks = [];
    let currentPos = 0;
    let lastEndPos = 0;
    
    // Sort tables by start position
    sectionTables.sort((a, b) => a.startPos - b.startPos);
    
    // Process each table and text before it
    for (let i = 0; i < sectionTables.length; i++) {
      const table = sectionTables[i];
      const tableStartInSection = table.startPos - startPosition;
      const tableEndInSection = table.endPos - startPosition;
      
      // Handle text before this table
      if (tableStartInSection > currentPos) {
        const textBeforeTable = sectionText.substring(currentPos, tableStartInSection);
        
        // If text before table is large enough to warrant splitting
        if (textBeforeTable.length > chunkSize) {
          // Recursively split text before table (without tables)
          const beforeChunks = splitLargeSection(
            textBeforeTable, 
            `${sectionTitle} (before table)`, 
            startPosition + currentPos
          );
          subChunks.push(...beforeChunks);
        } else if (textBeforeTable.trim().length > 0) {
          // Add as a small chunk
          subChunks.push({
            text: textBeforeTable,
            sectionTitle: `${sectionTitle} (before table)`,
            index: chunks.length + subChunks.length,
            startChar: startPosition + currentPos,
            endChar: startPosition + tableStartInSection - 1,
            containsTables: []
          });
        }
      }
      
      // Handle the table as its own chunk to prevent splitting
      const tableText = sectionText.substring(tableStartInSection, tableEndInSection + 1);
      
      subChunks.push({
        text: tableText,
        sectionTitle: table.caption || sectionTitle,
        index: chunks.length + subChunks.length,
        startChar: startPosition + tableStartInSection,
        endChar: startPosition + tableEndInSection,
        containsTables: [table],
        hasTables: true,
        isTableChunk: true
      });
      
      currentPos = tableEndInSection + 1;
      lastEndPos = tableEndInSection;
    }
    
    // Handle text after the last table
    if (currentPos < sectionText.length) {
      const textAfterTables = sectionText.substring(currentPos);
      
      // If text after table is large enough to warrant splitting
      if (textAfterTables.length > chunkSize) {
        // Recursively split text after table (without tables)
        const afterChunks = splitLargeSection(
          textAfterTables, 
          `${sectionTitle} (after table)`, 
          startPosition + currentPos
        );
        subChunks.push(...afterChunks);
      } else if (textAfterTables.trim().length > 0) {
        // Add as a small chunk
        subChunks.push({
          text: textAfterTables,
          sectionTitle: `${sectionTitle} (after table)`,
          index: chunks.length + subChunks.length,
          startChar: startPosition + currentPos,
          endChar: startPosition + sectionText.length - 1,
          containsTables: []
        });
      }
    }
    
    return subChunks;
  }
  
  // Function to detect tables in the text
  function detectTables(text) {
    const foundTables = [];
    // Match both new format ("### Table 2-2: Description") and old format ("### Extracted Table 1 from Page X")
    const tablePattern = /###\s+(Table\s+\d+(?:[-\.]\d+)?(?:\s*:\s*[^\n]+)?|Extracted Table \d+ from Page \d+)\n\n(\|[^\n]*\|\n\|[-:\s|]*\|\n(?:\|[^\n]*\|\n)+)/g;
    
    let match;
    while ((match = tablePattern.exec(text)) !== null) {
      const caption = match[1].trim();
      const tableContent = match[2];
      const startPos = match.index;
      const endPos = startPos + match[0].length - 1;
      
      foundTables.push({
        caption,
        startPos,
        endPos,
        length: match[0].length
      });
    }
    
    return foundTables;
  }

  // Process the document line by line
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    const isHeaderLine = isHeader(line);
    
    // If we find a header, finalize the current chunk
    if (isHeaderLine) {
      // Extract the section title from the header line
      const headerMatch = line.match(/^\s*(\d+(?:\.\d+)*)\s+(.+)$/);
      const newSectionTitle = headerMatch ? 
        `${headerMatch[1]} ${headerMatch[2].trim()}` : 
        line.trim();
      
      // Finalize the current chunk if it has content
      if (currentChunk.text.trim().length > 0) {
        finalizeChunk(currentChunk, currentPosition);
      }
      
      // Start a new chunk with this header
      currentSectionTitle = newSectionTitle;
      inSection = true;
      
      // Create a new chunk starting with the header line
      currentChunk = {
        text: line,
        sectionTitle: currentSectionTitle,
        index: chunks.length,
        startChar: currentPosition,
        endChar: 0,
        containsTables: []
      };
    } else if (inSection) {
      // Add this line to the current chunk
      currentChunk.text += (currentChunk.text.length > 0 ? '\n' : '') + line;
    } else {
      // Content before first header, add to initial chunk
      currentChunk.text += (currentChunk.text.length > 0 ? '\n' : '') + line;
    }
    
    // Update position counter (add 1 for the newline character)
    currentPosition += line.length + 1;
  }
  
  // Finalize the last chunk if it has content
  if (currentChunk.text.trim().length > 0) {
    finalizeChunk(currentChunk, currentPosition);
  }
  
  // Process chunks that exceed the size limit
  const finalChunks = [];
  for (const chunk of chunks) {
    if (chunk.text.length > chunkSize && !chunk.hasTables) {
      const subChunks = splitLargeSection(chunk.text, chunk.sectionTitle, chunk.startChar);
      finalChunks.push(...subChunks);
    } else {
      finalChunks.push(chunk);
    }
  }
  
  // Re-index the chunks to ensure sequential indices
  finalChunks.forEach((chunk, index) => {
    chunk.index = index;
  });
  
  console.log(`Header-based chunking created ${finalChunks.length} chunks from sections`);
  return finalChunks;
}

module.exports = {
  chunkBySection
}; 