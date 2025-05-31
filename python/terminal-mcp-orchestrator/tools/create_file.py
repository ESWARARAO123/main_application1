#!/usr/bin/env python3
"""
createFile - Creates a new file with specified content
"""

import os
import json
import sys
from typing import Dict, Any

# Tool information
TOOL_NAME = "createFile"
DESCRIPTION = "Create a new file with specified content"
PARAMETERS = {
    "filePath": {
        "type": "string",
        "description": "Path to the file to create",
        "required": True
    },
    "content": {
        "type": "string",
        "description": "Content to write to the file",
        "required": True
    },
    "overwrite": {
        "type": "boolean",
        "description": "Whether to overwrite the file if it already exists (default: false)",
        "required": False
    },
    "encoding": {
        "type": "string",
        "description": "File encoding (default: utf-8)",
        "required": False
    }
}

def execute(parameters: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a file with the specified content
    
    Args:
        parameters: Dictionary containing:
            - filePath: Path to the file to create
            - content: Content to write to the file
            - overwrite: Whether to overwrite existing file
            - encoding: File encoding
            
    Returns:
        Dictionary containing:
            - success: Whether the operation was successful
            - filePath: Path to the created file
            - error: Error message, if any
    """
    try:
        # Extract parameters
        file_path = parameters.get("filePath")
        if not file_path:
            return {"error": "filePath parameter is required"}
        
        content = parameters.get("content", "")
        overwrite = parameters.get("overwrite", False)
        encoding = parameters.get("encoding", "utf-8")
        
        # Check if file already exists
        if os.path.exists(file_path) and not overwrite:
            return {
                "error": f"File already exists: {file_path}. Set overwrite=true to replace it."
            }
        
        # Create parent directories if needed
        parent_dir = os.path.dirname(file_path)
        if parent_dir and not os.path.exists(parent_dir):
            os.makedirs(parent_dir)
        
        # Write content to file
        with open(file_path, 'w', encoding=encoding) as f:
            f.write(content)
        
        return {
            "success": True,
            "filePath": os.path.abspath(file_path)
        }
    
    except Exception as e:
        return {
            "error": f"Error creating file: {str(e)}"
        }

# For testing directly
if __name__ == "__main__":
    if len(sys.argv) > 2:
        file_path = sys.argv[1]
        content = sys.argv[2]
        result = execute({
            "filePath": file_path,
            "content": content
        })
    else:
        # Show usage
        print("Usage: python create_file.py <file_path> <content>")
        sys.exit(1)
    
    print(json.dumps(result, indent=2)) 