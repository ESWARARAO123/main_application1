#!/usr/bin/env python3
"""
readDirectory - Lists files and folders in a directory
"""

import os
import json
from typing import Dict, Any, List

# Tool information
TOOL_NAME = "readDirectory"
DESCRIPTION = "List files and folders in a directory"
PARAMETERS = {
    "dirPath": {
        "type": "string",
        "description": "Path to the directory to read (default: workspace root)",
        "required": True
    }
}

def execute(parameters: Dict[str, Any]) -> Dict[str, Any]:
    """
    List files and directories in the specified path
    
    Args:
        parameters: Dictionary containing:
            - dirPath: Path to the directory to read
            
    Returns:
        Dictionary containing:
            - files: List of file names
            - directories: List of directory names
            - error: Error message, if any
    """
    try:
        # Get directory path from parameters, default to current directory
        dir_path = parameters.get("dirPath", ".")
        
        # Check if directory exists
        if not os.path.exists(dir_path):
            return {
                "error": f"Directory does not exist: {dir_path}"
            }
        
        if not os.path.isdir(dir_path):
            return {
                "error": f"Path is not a directory: {dir_path}"
            }
        
        # List all entries in the directory
        entries = os.listdir(dir_path)
        
        # Separate files and directories
        files = []
        directories = []
        
        for entry in entries:
            full_path = os.path.join(dir_path, entry)
            if os.path.isdir(full_path):
                directories.append(entry)
            else:
                files.append(entry)
        
        # Sort alphabetically
        files.sort()
        directories.sort()
        
        return {
            "files": files,
            "directories": directories,
            "path": os.path.abspath(dir_path)
        }
    
    except Exception as e:
        return {
            "error": f"Error reading directory: {str(e)}"
        }

# For testing directly
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        dir_path = sys.argv[1]
        result = execute({"dirPath": dir_path})
    else:
        result = execute({"dirPath": "."})
    
    print(json.dumps(result, indent=2)) 