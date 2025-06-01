#!/bin/bash

# Initialize variables
PYTHON_VERSION="python3.9"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"
VENV_PATH="${PARENT_DIR}/venv"
REQUIREMENTS="${SCRIPT_DIR}/requirements.txt"
EXTRACT_SCRIPT="${SCRIPT_DIR}/extract_text.py"
EXTRACT_TABLES_SCRIPT="${SCRIPT_DIR}/extract_text_with_tables.py"

# Print script header
echo "==============================================="
echo "PDF Text Extraction - Virtual Environment Setup"
echo "==============================================="
echo ""

# Check if we're in the right directory
if [[ $(basename "${SCRIPT_DIR}") != "RAG-MODULE" ]]; then
    echo "Warning: This script should be run from the 'RAG-MODULE' directory"
    echo "Current directory: ${SCRIPT_DIR}"
    echo ""
fi

# Check if Python 3.9 is installed
if ! command -v $PYTHON_VERSION &> /dev/null
then
    echo "Error: $PYTHON_VERSION is not installed or not in PATH"
    echo "Please install Python 3.9 and try again"
    exit 1
fi

# Check if venv module is available
$PYTHON_VERSION -c "import venv" &> /dev/null
if [ $? -ne 0 ]; then
    echo "Error: Python venv module is not available"
    echo "Please install python3.9-venv package if needed"
    echo "Example: apt-get install python3.9-venv (Debian/Ubuntu)"
    echo "Example: yum install python3.9-venv (RHEL/CentOS)"
    exit 1
fi

# Create requirements.txt if it doesn't exist
if [ ! -f "$REQUIREMENTS" ]; then
    echo "requirements.txt not found, creating it..."
    cat > "$REQUIREMENTS" << EOL
pdfplumber==0.9.0
Pillow>=9.0.0
Wand>=0.6.7
cryptography>=38.0.0
EOL
    echo "Created requirements.txt with necessary dependencies"
fi

# Create extract_text.py if it doesn't exist
if [ ! -f "$EXTRACT_SCRIPT" ]; then
    echo "extract_text.py not found, creating it..."
    cat > "$EXTRACT_SCRIPT" << 'EOL'
#!/usr/bin/env python3.9

import sys
import json
import argparse
import traceback

def extract_text_with_pdfplumber(pdf_path):
    """
    Extract text from a PDF file using pdfplumber with page markers.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Dictionary with text content and metadata
    """
    try:
        try:
            import pdfplumber
        except ImportError:
            return {
                "success": False,
                "error": "pdfplumber module not installed. Install with: pip install pdfplumber",
                "instructions": "Run: pip install --user pdfplumber"
            }
        
        full_text = ""
        page_texts = []
        
        with pdfplumber.open(pdf_path) as pdf:
            total_pages = len(pdf.pages)
            
            for i, page in enumerate(pdf.pages):
                page_num = i + 1
                page_text = page.extract_text()
                
                if page_text:
                    # Add page marker at the beginning of each page
                    marked_text = f"\n\n[Page {page_num} of {total_pages}]\n{page_text}"
                    full_text += marked_text
                    page_texts.append({
                        "page_num": page_num,
                        "text": page_text
                    })
        
        return {
            "success": True,
            "text": full_text.strip(),
            "page_count": total_pages,
            "pages": page_texts
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }

def process_pdf(pdf_path):
    """
    Process a PDF file and extract text.
    
    This function is a wrapper that handles errors and ensures valid JSON output.
    """
    if not pdf_path or not pdf_path.lower().endswith('.pdf'):
        return {
            "success": False,
            "error": f"Invalid PDF file: {pdf_path}"
        }
    
    try:
        # Try to access the file
        with open(pdf_path, 'rb') as f:
            # Just check if we can open it
            pass
    except Exception as e:
        return {
            "success": False,
            "error": f"Could not access PDF file: {str(e)}"
        }
    
    # Extract text using pdfplumber
    return extract_text_with_pdfplumber(pdf_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract text from PDF using pdfplumber")
    parser.add_argument("pdf_path", help="Path to the PDF file")
    
    if len(sys.argv) < 2:
        print(json.dumps({
            "success": False,
            "error": "PDF path argument is required"
        }))
        sys.exit(1)
    
    args = parser.parse_args()
    
    # Process the PDF and get the result
    result = process_pdf(args.pdf_path)
    
    # Print JSON output to stdout for Node.js to capture
    print(json.dumps(result))
EOL
    chmod +x "$EXTRACT_SCRIPT"
    echo "Created extract_text.py script"
fi

# Create extract_text_with_tables.py if it doesn't exist
if [ ! -f "$EXTRACT_TABLES_SCRIPT" ]; then
    echo "extract_text_with_tables.py not found, creating it..."
    cat > "$EXTRACT_TABLES_SCRIPT" << 'EOL'
#!/usr/bin/env python3

import sys
import json
import argparse
import traceback
import re
from typing import List, Dict, Any, Tuple, Optional

def detect_table_caption(text_before_table: str, page_num: int, max_chars_to_scan: int = 500) -> Optional[str]:
    """
    Detect and extract table caption from text preceding the table.
    
    Args:
        text_before_table: Text content before the table
        page_num: Current page number for fallback caption
        max_chars_to_scan: Maximum characters to scan backwards for captions
        
    Returns:
        Extracted table caption or None if not found
    """
    # Limit the text to scan for efficiency
    text_to_scan = text_before_table[-max_chars_to_scan:] if len(text_before_table) > max_chars_to_scan else text_before_table
    
    # Look for patterns like "Table 2-2: Description" or "Table 2.2 Description"
    caption_patterns = [
        r"Table\s+(\d+[-\.]\d+)[\s\:]+([^\n]+)",  # Table 2-2: Description
        r"Table\s+(\d+)[\s\:]+([^\n]+)",          # Table 2: Description
        r"TABLE\s+(\d+[-\.]\d+)[\s\:]+([^\n]+)",  # TABLE 2-2: Description
        r"Table\s+(\d+\.\d+)[\s\:]+([^\n]+)"      # Table 2.1: Description
    ]
    
    for pattern in caption_patterns:
        match = re.search(pattern, text_to_scan, re.MULTILINE)
        if match:
            table_number = match.group(1)
            description = match.group(2).strip()
            return f"Table {table_number}: {description}"
    
    # If no caption found, return a generic one with the page number
    return f"Extracted Table from Page {page_num}"

def _is_header_row(row: List[str]) -> bool:
    """
    Check if a row is likely a header row based on its content.
    
    Args:
        row: Table row to check
        
    Returns:
        True if row is likely a header, False otherwise
    """
    if not row:
        return False
    
    # Check if all cells are non-empty and relatively short
    non_empty = [cell for cell in row if cell and str(cell).strip()]
    if len(non_empty) < len(row) * 0.7:  # At least 70% of cells should be non-empty
        return False
    
    # Check if cells are relatively short (headers tend to be concise)
    avg_length = sum(len(str(cell)) for cell in non_empty) / len(non_empty) if non_empty else 0
    return avg_length < 25  # Headers are usually shorter than 25 chars

def _generate_column_alignments(table: List[List[str]]) -> List[str]:
    """
    Generate column alignments based on data types.
    
    Args:
        table: The table data
        
    Returns:
        List of alignment strings for each column
    """
    if not table or len(table) < 2:
        return []
    
    num_cols = len(table[0])
    alignments = []
    
    # Check each column
    for col_idx in range(num_cols):
        # Get column values (skip header)
        col_values = [row[col_idx] if col_idx < len(row) else "" for row in table[1:]]
        
        # Check if column appears to be numeric
        numeric_count = 0
        for val in col_values:
            if val and re.match(r'^-?\d+(?:\.\d+)?$', str(val).strip()):
                numeric_count += 1
        
        # If more than 70% of values are numeric, right-align
        if numeric_count > len(col_values) * 0.7:
            alignments.append("---:")
        else:
            alignments.append("---")
    
    return alignments

def convert_table_to_markdown(table: List[List[str]], caption: str = "") -> str:
    """
    Convert a table extracted by pdfplumber into a Markdown table string.
    
    Args:
        table: A list of lists representing the table rows and columns
        caption: Table caption to include
        
    Returns:
        Markdown formatted table as a string
    """
    if not table or not table[0]:
        return ""
    
    # Clean up cell content - remove extra whitespace, newlines
    cleaned_table = []
    for row in table:
        cleaned_row = []
        for cell in row:
            if cell is None:
                cell = ""
            else:
                # Replace multiple spaces with a single space
                cell = re.sub(r'\s+', ' ', str(cell).strip())
            cleaned_row.append(cell)
        cleaned_table.append(cleaned_row)
    
    # Get column widths for formatting
    num_cols = len(cleaned_table[0])
    
    # Determine if first row is likely a header
    has_header = _is_header_row(cleaned_table[0])
    
    # Get column alignments
    alignments = _generate_column_alignments(cleaned_table)
    
    # Start markdown with caption if provided
    markdown_table = ""
    if caption:
        markdown_table = f"### {caption}\n\n"
    
    # Create table header row
    markdown_table += "| " + " | ".join(cleaned_table[0]) + " |\n"
    
    # Create separator row with proper alignment
    if alignments and len(alignments) == num_cols:
        markdown_table += "| " + " | ".join(alignments) + " |\n"
    else:
        markdown_table += "| " + " | ".join(["---"] * num_cols) + " |\n"
    
    # Create data rows
    for row in cleaned_table[1:]:
        # Ensure row has the correct number of columns
        while len(row) < num_cols:
            row.append("")
        markdown_table += "| " + " | ".join(row[:num_cols]) + " |\n"
    
    # Add table metadata as HTML comment for post-processing
    table_metadata = {
        'caption': caption,
        'has_header': has_header,
        'num_rows': len(cleaned_table),
        'num_cols': num_cols
    }
    markdown_table += f"\n<!-- TABLE_METADATA: {json.dumps(table_metadata)} -->\n"
    
    return markdown_table

def extract_text_and_tables(pdf_path: str) -> Dict[str, Any]:
    """
    Extract text and tables from a PDF file using pdfplumber.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Dictionary with extracted text and tables
    """
    try:
        import pdfplumber
    except ImportError:
        return {
            "success": False,
            "error": "pdfplumber module not installed. Install with: pip install pdfplumber",
            "instructions": "Run: pip install --user pdfplumber"
        }
    
    try:
        full_text = ""
        
        with pdfplumber.open(pdf_path) as pdf:
            total_pages = len(pdf.pages)
            
            for i, page in enumerate(pdf.pages):
                page_num = i + 1
                
                # Extract page text
                page_text = page.extract_text() or ""
                
                # Add page marker
                page_marker = f"\n\n[Page {page_num} of {total_pages}]\n"
                full_text += page_marker + page_text
                
                # Extract tables from the page
                tables = page.extract_tables()
                
                # If tables were found, convert them to markdown and add to the text
                if tables:
                    for table_idx, table in enumerate(tables):
                        if table and len(table) > 0:
                            # Look for table caption in the text before the table
                            table_caption = detect_table_caption(page_text, page_num)
                            
                            # Convert to markdown with caption
                            markdown_table = convert_table_to_markdown(table, table_caption)
                            
                            # Add the table to the text
                            if markdown_table:
                                full_text += f"\n\n{markdown_table}\n"
        
        return {
            "success": True,
            "text": full_text.strip(),
            "page_count": total_pages,
            "has_tables": True
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }

def process_pdf(pdf_path: str) -> Dict[str, Any]:
    """
    Process a PDF file and extract text with tables.
    
    This function is a wrapper that handles errors and ensures valid JSON output.
    """
    if not pdf_path or not pdf_path.lower().endswith('.pdf'):
        return {
            "success": False,
            "error": f"Invalid PDF file: {pdf_path}"
        }
    
    try:
        # Try to access the file
        with open(pdf_path, 'rb') as f:
            # Just check if we can open it
            pass
    except Exception as e:
        return {
            "success": False,
            "error": f"Could not access PDF file: {str(e)}"
        }
    
    # Extract text and tables
    return extract_text_and_tables(pdf_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract text and tables from PDF using pdfplumber")
    parser.add_argument("pdf_path", help="Path to the PDF file")
    
    if len(sys.argv) < 2:
        print(json.dumps({
            "success": False,
            "error": "PDF path argument is required"
        }))
        sys.exit(1)
    
    args = parser.parse_args()
    
    # Process the PDF and get the result
    result = process_pdf(args.pdf_path)
    
    # Print JSON output to stdout for Node.js to capture
    print(json.dumps(result))
EOL
    chmod +x "$EXTRACT_TABLES_SCRIPT"
    echo "Created extract_text_with_tables.py script"
fi

# Remove existing venv if it exists
if [ -d "$VENV_PATH" ]; then
    echo "Removing existing virtual environment..."
    rm -rf "$VENV_PATH"
    echo "Done!"
fi

# Create new virtual environment
echo "Creating new virtual environment with $PYTHON_VERSION..."
$PYTHON_VERSION -m venv "$VENV_PATH"
if [ $? -ne 0 ]; then
    echo "Error: Failed to create virtual environment"
    exit 1
fi
echo "Virtual environment created successfully!"

# Activate virtual environment and install requirements
echo "Installing required packages..."
source "$VENV_PATH/bin/activate"

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
if [ -f "$REQUIREMENTS" ]; then
    echo "Installing packages from requirements.txt..."
    pip install -r "$REQUIREMENTS"
    if [ $? -ne 0 ]; then
        echo "Error: Failed to install required packages"
        exit 1
    fi
else
    echo "Error: requirements.txt not found at $REQUIREMENTS"
    exit 1
fi

# Print success message
echo ""
echo "Installation complete!"
echo "Virtual environment location: $VENV_PATH"
echo ""
echo "To activate the virtual environment, run:"
echo "  source $VENV_PATH/bin/activate"
echo ""
echo "Update your config.ini with this interpreter path:"
echo "  interpreter = ./python/venv/bin/python"

# Deactivate virtual environment
deactivate 