# Table Extraction for RAG Documentation

## Overview

The Table Extraction feature enhances the Retrieval-Augmented Generation (RAG) pipeline by extracting not only the text content from PDF documents but also the tables embedded within them. These tables are converted to Markdown format and integrated seamlessly into the extracted text, preserving the document's structure and ensuring that table data is properly represented in the RAG system.

## Problem Solved

Traditional text extraction from PDFs often ignores tables or extracts them as disjointed text, leading to:
- Loss of tabular structure and relationships between data points
- Confused context when processing data that was originally in table format
- Reduced ability to answer questions that rely on tabular information

By extracting tables as properly formatted Markdown, we maintain the structure and relationships in the data, significantly improving the RAG system's ability to understand and retrieve information from tables.

## Implementation Details

### Components

1. **Python Extraction Script** (`python/extract_text_with_tables.py`):
   - Uses `pdfplumber` to extract both text and tables from PDFs
   - Processes each page to identify embedded tables
   - Converts tables to Markdown format with column alignment
   - Integrates tables directly into the extracted text
   - Adds clear table markers with page references

2. **Node.js Processing Service** (`src/services/pdfProcessor.js`):
   - Provides a `extractTextAndTablesSmart()` function to call the Python script
   - Handles errors and timeouts gracefully
   - Parses JSON results and provides a clean interface

3. **Integration with Document Processor** (`src/services/documentProcessor.js`):
   - Prioritizes table-aware extraction over basic text extraction
   - Falls back to simpler methods if table extraction fails
   - Preserves compatibility with existing chunking systems

### Technical Approach

The implementation follows these key steps:

1. **Table Detection**:
   The Python script uses `pdfplumber`'s `page.extract_tables()` method to identify tables on each page.

2. **Markdown Conversion**:
   ```python
   def convert_table_to_markdown(table):
       # Clean cell content
       # Create header row with separator
       # Ensure proper column alignment
       # Convert all rows to Markdown format
   ```

3. **Document Integration**:
   Tables are inserted directly after the text of the page they appear on, with clear markers:
   ```
   ### Extracted Table 1 from Page 3
   
   | Header 1 | Header 2 | Header 3 |
   |----------|----------|----------|
   | Data 1   | Data 2   | Data 3   |
   ```

4. **Chunking Compatibility**:
   The output is compatible with both basic chunking and header-based chunking systems.

## Usage

The table extraction is automatically used when processing PDF documents through the RAG pipeline:

```javascript
// Extract text with tables from a PDF
const result = await documentProcessor.extractText(document);
```

For testing or direct usage:

```javascript
const { extractTextAndTablesSmart } = require('./src/services/pdfProcessor');
const result = await extractTextAndTablesSmart('/path/to/document.pdf');
```

## Benefits

### Enhanced Information Retrieval

- **Structural Preservation**: Maintains the tabular format of data, preserving relationships between data points
- **Context Integration**: Tables are kept close to their surrounding text, maintaining the original context
- **Improved Answers**: Enables the RAG system to provide more accurate answers to queries about tabular data

### Developer Experience

- **Graceful Fallbacks**: System automatically falls back to simpler extraction methods if table extraction fails
- **Clear Debugging**: Table markers clearly indicate the source page of each table
- **Markdown Compatibility**: Output can be rendered in Markdown viewers for easy inspection

## Example

Original PDF with a table:
```
Technical Specifications

The following table shows the supported pin configurations:

[Table with pin configurations]

All pins support 3.3V operation unless otherwise specified.
```

Extracted text with table:
```
Technical Specifications

The following table shows the supported pin configurations:

### Extracted Table 1 from Page 2

| Pin | Function | Voltage |
|-----|----------|---------|
| A1  | CLK      | 3.3V    |
| A2  | RESET    | 3.3V    |
| B1  | DATA     | 1.8V    |

All pins support 3.3V operation unless otherwise specified.
```

## Testing

A test script (`test_table_extraction.js`) is provided to validate the table extraction functionality:

```bash
node test_table_extraction.js sample.pdf
```

The script outputs statistics about the extraction and saves the result to a Markdown file for inspection.

## Limitations and Future Enhancements

- Current implementation works best with simple, well-structured tables
- Complex merged cells or multi-level headers may not be perfectly represented
- Future enhancements could include:
  - Support for more complex table structures
  - Improved layout preservation for better visual representation
  - Table metadata extraction for enhanced context understanding 