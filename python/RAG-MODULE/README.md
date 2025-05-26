# RAG-MODULE: Python PDF Text Extraction

This directory contains the Python components for enhanced PDF text extraction with RAG (Retrieval Augmented Generation) capabilities.

## Contents

- `extract_text.py` - Python script for basic text extraction from PDFs with layout preservation
- `extract_text_with_tables.py` - Advanced script for extracting both text and tables in Markdown format
- `requirements.txt` - List of Python dependencies
- `installvenv.sh` - Script to set up the Python 3.9 virtual environment
- `test_extraction.sh` - Script to verify installation and test extraction functionality
- `venv/` - Python virtual environment directory (created by installvenv.sh)

## Features

- **Text Extraction**: Basic PDF text extraction with page markers
- **Table Extraction**: Advanced extraction that converts tables to Markdown format
- **Layout Preservation**: Maintains document structure with page markers
- **Table Caption Detection**: Smart detection of table captions and numbering

## Setup Instructions

1. Make sure Python 3.9 is installed on your system:
   ```bash
   python3.9 --version
   ```

2. Run the installation script to create the virtual environment:
   ```bash
   # You can run it from any directory
   ./python/RAG-MODULE/installvenv.sh
   ```
   
   This script will:
   - Create the Python virtual environment
   - Install all required dependencies
   - Create script files if they don't exist
   - Create `requirements.txt` if it doesn't exist

3. After installation, update the config.ini with the relative path:
   ```ini
   [python]
   interpreter = ./python/venv/bin/python
   ```
   
   Using a relative path allows the configuration to work regardless of where the project is installed.

## Testing the Installation

After setting up the environment, you can verify it works correctly using the test script:

```bash
# Basic verification (tests if pdfplumber is installed correctly)
./python/RAG-MODULE/test_extraction.sh

# Full test with PDF extraction (provide a PDF file path)
./python/RAG-MODULE/test_extraction.sh /path/to/sample.pdf
```

## Manual Setup (if installvenv.sh fails)

If the automated script fails, you can set up the environment manually:

```bash
# Create virtual environment
cd python
python3.9 -m venv venv

# Activate the environment
source venv/bin/activate

# Update pip
pip install --upgrade pip

# Install dependencies
pip install -r RAG-MODULE/requirements.txt
```

## Testing the Scripts Directly

You can test the PDF extraction scripts directly:

```bash
# Activate the virtual environment
source python/venv/bin/activate

# Run the basic extraction script
python python/RAG-MODULE/extract_text.py /path/to/your/document.pdf

# Run the advanced table extraction script
python python/RAG-MODULE/extract_text_with_tables.py /path/to/your/document.pdf
```

Both scripts will output JSON with the extracted text and metadata.

## Integration with Node.js

These scripts are called by the following Node.js modules:
- `src/services/documentProcessor.js` - Calls extract_text.py for basic processing
- `src/services/pdfProcessor.js` - Calls extract_text_with_tables.py for table-aware processing

## Troubleshooting

If you encounter any issues:

1. Make sure Python 3.9 is installed and accessible
2. Check that all dependencies were installed correctly
3. Verify the paths in config.ini are correct
4. Run the test script to verify the installation
5. Check script permissions (should be executable) 