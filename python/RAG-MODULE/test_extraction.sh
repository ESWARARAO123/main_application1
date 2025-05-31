#!/bin/bash

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"
VENV_PATH="${PARENT_DIR}/venv"
EXTRACT_SCRIPT="${SCRIPT_DIR}/extract_text.py"
EXTRACT_TABLES_SCRIPT="${SCRIPT_DIR}/extract_text_with_tables.py"

# Check if virtual environment exists
if [ ! -d "$VENV_PATH" ]; then
    echo "Error: Virtual environment not found at $VENV_PATH"
    echo "Please run installvenv.sh first"
    exit 1
fi

# Check if extract_text.py exists
if [ ! -f "$EXTRACT_SCRIPT" ]; then
    echo "Error: extract_text.py not found at $EXTRACT_SCRIPT"
    echo "Please run installvenv.sh first"
    exit 1
fi

# Check if extract_text_with_tables.py exists
if [ ! -f "$EXTRACT_TABLES_SCRIPT" ]; then
    echo "Error: extract_text_with_tables.py not found at $EXTRACT_TABLES_SCRIPT"
    echo "Please run installvenv.sh first"
    exit 1
fi

# Activate virtual environment
source "$VENV_PATH/bin/activate"

# Test if pdfplumber is installed correctly
echo "Testing pdfplumber installation..."
python -c "import pdfplumber; print(f'pdfplumber version: {pdfplumber.__version__}')"
if [ $? -ne 0 ]; then
    echo "Error: pdfplumber not installed correctly"
    deactivate
    exit 1
fi
echo "pdfplumber installation verified successfully!"
echo ""

# Check if a PDF file was provided as an argument
if [ $# -eq 0 ]; then
    echo "No PDF file provided for testing"
    echo "Usage: $0 <path-to-pdf-file>"
    echo ""
    echo "Installation test completed without PDF extraction test"
    deactivate
    exit 0
fi

PDF_FILE="$1"

# Check if the provided file exists and is a PDF
if [ ! -f "$PDF_FILE" ]; then
    echo "Error: File not found: $PDF_FILE"
    deactivate
    exit 1
fi

if [[ "$PDF_FILE" != *.pdf ]]; then
    echo "Error: $PDF_FILE does not appear to be a PDF file"
    deactivate
    exit 1
fi

# Test basic PDF extraction
echo "===== Testing Basic PDF Extraction ====="
echo "Running command: python $EXTRACT_SCRIPT \"$PDF_FILE\""
echo ""

# Run the extraction and save output to a temporary file
TEMP_OUTPUT=$(mktemp)
python "$EXTRACT_SCRIPT" "$PDF_FILE" > "$TEMP_OUTPUT"

# Check if the extraction was successful
if [ $? -ne 0 ]; then
    echo "Error: Basic PDF extraction failed"
    rm "$TEMP_OUTPUT"
    deactivate
    exit 1
fi

# Check if the output is valid JSON
if ! python -c "import json; json.load(open('$TEMP_OUTPUT'))"; then
    echo "Error: Output is not valid JSON"
    rm "$TEMP_OUTPUT"
    deactivate
    exit 1
fi

# Display a summary of the extraction results
echo ""
echo "Basic PDF extraction successful!"
echo "-----------------------------"
PAGES=$(python -c "import json; data = json.load(open('$TEMP_OUTPUT')); print(data.get('page_count', 'unknown'))")
CHARS=$(python -c "import json; data = json.load(open('$TEMP_OUTPUT')); print(len(data.get('text', '')))")
echo "Pages extracted: $PAGES"
echo "Characters extracted: $CHARS"
echo ""
echo "First 300 characters of extracted text:"
python -c "import json; data = json.load(open('$TEMP_OUTPUT')); print(data.get('text', '')[:300] + '...')"
echo ""

# Clean up
rm "$TEMP_OUTPUT"

# Test table extraction
echo "===== Testing Table-aware PDF Extraction ====="
echo "Running command: python $EXTRACT_TABLES_SCRIPT \"$PDF_FILE\""
echo ""

# Run the table extraction and save output to a temporary file
TEMP_TABLE_OUTPUT=$(mktemp)
python "$EXTRACT_TABLES_SCRIPT" "$PDF_FILE" > "$TEMP_TABLE_OUTPUT"

# Check if the extraction was successful
if [ $? -ne 0 ]; then
    echo "Error: Table-aware PDF extraction failed"
    rm "$TEMP_TABLE_OUTPUT"
    deactivate
    exit 1
fi

# Check if the output is valid JSON
if ! python -c "import json; json.load(open('$TEMP_TABLE_OUTPUT'))"; then
    echo "Error: Table output is not valid JSON"
    rm "$TEMP_TABLE_OUTPUT"
    deactivate
    exit 1
fi

# Display a summary of the table extraction results
echo ""
echo "Table-aware PDF extraction successful!"
echo "-----------------------------"
TABLE_PAGES=$(python -c "import json; data = json.load(open('$TEMP_TABLE_OUTPUT')); print(data.get('page_count', 'unknown'))")
TABLE_CHARS=$(python -c "import json; data = json.load(open('$TEMP_TABLE_OUTPUT')); print(len(data.get('text', '')))")
HAS_TABLES=$(python -c "import json; data = json.load(open('$TEMP_TABLE_OUTPUT')); print('Yes' if data.get('has_tables', False) else 'No')")
echo "Pages extracted: $TABLE_PAGES"
echo "Characters extracted: $TABLE_CHARS"
echo "Contains tables: $HAS_TABLES"
echo ""
echo "First 300 characters of table-aware extracted text:"
python -c "import json; data = json.load(open('$TEMP_TABLE_OUTPUT')); print(data.get('text', '')[:300] + '...')"
echo ""

# Clean up
rm "$TEMP_TABLE_OUTPUT"

# Deactivate venv
deactivate
echo "Test completed successfully for both extraction methods!" 