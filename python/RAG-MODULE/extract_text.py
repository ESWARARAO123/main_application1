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