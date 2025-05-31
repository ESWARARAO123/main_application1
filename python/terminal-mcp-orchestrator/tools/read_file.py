#!/usr/bin/env python3
"""
readFile - Reads the content of a file
"""

import os
import json
import sys
from typing import Dict, Any, Optional

# Tool information
TOOL_NAME = "readFile"
DESCRIPTION = "Read the content of a file"
PARAMETERS = {
    "filePath": {
        "type": "string",
        "description": "Path to the file to read",
        "required": True
    },
    "encoding": {
        "type": "string",
        "description": "File encoding (default: utf-8)",
        "required": False
    },
    "startLine": {
        "type": "integer",
        "description": "Start line (0-based, inclusive)",
        "required": False
    },
    "endLine": {
        "type": "integer",
        "description": "End line (0-based, inclusive)",
        "required": False
    }
}

def read_file_content(file_path: str, encoding: str = "utf-8") -> str:
    """Read the entire content of a file"""
    with open(file_path, 'r', encoding=encoding) as f:
        return f.read()

def read_file_lines(
    file_path: str, 
    encoding: str = "utf-8", 
    start_line: Optional[int] = None, 
    end_line: Optional[int] = None
) -> Dict[str, Any]:
    """
    Read specific lines from a file
    
    Args:
        file_path: Path to the file
        encoding: File encoding
        start_line: Starting line (0-based, inclusive)
        end_line: Ending line (0-based, inclusive)
        
    Returns:
        Dictionary with content and line information
    """
    try:
        with open(file_path, 'r', encoding=encoding) as f:
            lines = f.readlines()
        
        total_lines = len(lines)
        
        # Adjust line ranges
        if start_line is None:
            start_line = 0
        if end_line is None:
            end_line = total_lines - 1
        
        # Ensure valid ranges
        start_line = max(0, min(start_line, total_lines - 1))
        end_line = max(start_line, min(end_line, total_lines - 1))
        
        # Extract the requested lines
        selected_lines = lines[start_line:end_line + 1]
        content = ''.join(selected_lines)
        
        return {
            "content": content,
            "startLine": start_line,
            "endLine": end_line,
            "totalLines": total_lines
        }
    except Exception as e:
        return {"error": f"Error reading file lines: {str(e)}"}

def execute(parameters: Dict[str, Any]) -> Dict[str, Any]:
    """
    Read content from a file
    
    Args:
        parameters: Dictionary containing:
            - filePath: Path to the file to read
            - encoding: File encoding (optional)
            - startLine: Start line (optional)
            - endLine: End line (optional)
            
    Returns:
        Dictionary containing:
            - content: File content
            - startLine: Starting line read
            - endLine: Ending line read
            - totalLines: Total lines in file
            - error: Error message, if any
    """
    try:
        # Extract parameters
        file_path = parameters.get("filePath")
        if not file_path:
            return {"error": "filePath parameter is required"}
        
        encoding = parameters.get("encoding", "utf-8")
        start_line = parameters.get("startLine")
        end_line = parameters.get("endLine")
        
        # Check if file exists
        if not os.path.exists(file_path):
            return {"error": f"File does not exist: {file_path}"}
        
        if not os.path.isfile(file_path):
            return {"error": f"Path is not a file: {file_path}"}
        
        # Read file based on parameters
        if start_line is not None or end_line is not None:
            # Read specific lines
            return read_file_lines(file_path, encoding, start_line, end_line)
        else:
            # Read entire file
            content = read_file_content(file_path, encoding)
            total_lines = content.count('\n') + (1 if content else 0)
            
            return {
                "content": content,
                "startLine": 0,
                "endLine": total_lines - 1,
                "totalLines": total_lines
            }
    
    except Exception as e:
        return {"error": f"Error reading file: {str(e)}"}

# For testing directly
if __name__ == "__main__":
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        start_line = int(sys.argv[2]) if len(sys.argv) > 2 else None
        end_line = int(sys.argv[3]) if len(sys.argv) > 3 else None
        
        result = execute({
            "filePath": file_path,
            "startLine": start_line,
            "endLine": end_line
        })
    else:
        # Show usage
        print("Usage: python read_file.py <file_path> [start_line] [end_line]")
        sys.exit(1)
    
    print(json.dumps(result, indent=2)) 