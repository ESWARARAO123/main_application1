#!/usr/bin/env python3
"""
MCP Tool Orchestrator - Local Execution Version

This version of the orchestrator is designed to work locally without requiring
a remote MCP server. It implements the core MCP tools directly in Python.

Usage:
  python orchestrator.py <tool_name> <parameters_json>
  
Example:
  python orchestrator.py readDirectory {"dirPath": "."}
"""

import sys
import json
import os
import logging
import argparse
from typing import Dict, Any, List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('mcp_local_orchestrator')

# Available MCP tools
AVAILABLE_TOOLS = [
    "runShellCommand",
    "readDirectory",
    "createFile",
    "readFile",
    "deleteFile"
]

# Tool implementations

def run_shell_command(parameters: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute a shell command
    
    Args:
        parameters: Dictionary containing:
            - command: The command to execute
            
    Returns:
        Dictionary containing:
            - stdout: Standard output
            - stderr: Standard error
            - exitCode: Exit code
    """
    import subprocess
    
    try:
        # Get command from parameters
        command = parameters.get("command", "")
        if not command:
            return {"error": "No command specified"}
        
        # Execute command
        process = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True
        )
        
        return {
            "stdout": process.stdout,
            "stderr": process.stderr,
            "exitCode": process.returncode
        }
    except Exception as e:
        return {"error": f"Error executing command: {str(e)}"}

def read_directory(parameters: Dict[str, Any]) -> Dict[str, Any]:
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

def create_file(parameters: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a new file with the specified content
    
    Args:
        parameters: Dictionary containing:
            - filePath: Path to the file to create
            - content: Content to write to the file
            
    Returns:
        Dictionary containing:
            - success: True if file was created successfully
            - filePath: Absolute path to the created file
            - error: Error message, if any
    """
    try:
        # Get parameters
        file_path = parameters.get("filePath", "")
        content = parameters.get("content", "")
        
        if not file_path:
            return {"error": "No file path specified"}
        
        # Create directory if it doesn't exist
        dir_path = os.path.dirname(file_path)
        if dir_path and not os.path.exists(dir_path):
            os.makedirs(dir_path)
        
        # Write content to file
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return {
            "success": True,
            "filePath": os.path.abspath(file_path)
        }
    
    except Exception as e:
        return {"error": f"Error creating file: {str(e)}"}

def read_file(parameters: Dict[str, Any]) -> Dict[str, Any]:
    """
    Read the content of a file
    
    Args:
        parameters: Dictionary containing:
            - filePath: Path to the file to read
            - encoding: File encoding (default: utf-8)
            - startLine: Starting line (0-based, inclusive)
            - endLine: Ending line (0-based, inclusive)
            
    Returns:
        Dictionary containing:
            - content: File content
            - startLine: Starting line
            - endLine: Ending line
            - totalLines: Total number of lines in the file
            - error: Error message, if any
    """
    try:
        # Get parameters
        file_path = parameters.get("filePath", "")
        encoding = parameters.get("encoding", "utf-8")
        start_line = parameters.get("startLine")
        end_line = parameters.get("endLine")
        
        if not file_path:
            return {"error": "No file path specified"}
        
        # Check if file exists
        if not os.path.exists(file_path):
            return {"error": f"File does not exist: {file_path}"}
        
        if not os.path.isfile(file_path):
            return {"error": f"Path is not a file: {file_path}"}
        
        # Read file content
        with open(file_path, 'r', encoding=encoding) as f:
            if start_line is not None or end_line is not None:
                # Read specific lines
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
                content = ''.join(lines[start_line:end_line + 1])
            else:
                # Read entire file
                content = f.read()
                total_lines = content.count('\n') + (1 if content else 0)
                start_line = 0
                end_line = total_lines - 1
        
        return {
            "content": content,
            "startLine": start_line,
            "endLine": end_line,
            "totalLines": total_lines
        }
    
    except Exception as e:
        return {"error": f"Error reading file: {str(e)}"}

def delete_file(parameters: Dict[str, Any]) -> Dict[str, Any]:
    """
    Delete a file
    
    Args:
        parameters: Dictionary containing:
            - filePath: Path to the file to delete
            
    Returns:
        Dictionary containing:
            - success: True if file was deleted successfully
            - filePath: Absolute path to the deleted file
            - error: Error message, if any
    """
    try:
        # Get parameters
        file_path = parameters.get("filePath", "")
        
        if not file_path:
            return {"error": "No file path specified"}
        
        # Check if file exists
        if not os.path.exists(file_path):
            return {"error": f"File does not exist: {file_path}"}
        
        if not os.path.isfile(file_path):
            return {"error": f"Path is not a file: {file_path}"}
        
        # Delete file
        os.remove(file_path)
        
        return {
            "success": True,
            "filePath": os.path.abspath(file_path)
        }
    
    except Exception as e:
        return {"error": f"Error deleting file: {str(e)}"}

# Tool mapping
TOOL_FUNCTIONS = {
    "runShellCommand": run_shell_command,
    "readDirectory": read_directory,
    "createFile": create_file,
    "readFile": read_file,
    "deleteFile": delete_file
}

def execute_tool(tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute the specified tool with the given parameters
    
    Args:
        tool_name: Name of the tool to execute
        parameters: Parameters to pass to the tool
        
    Returns:
        Tool execution result
    """
    if tool_name not in AVAILABLE_TOOLS:
        return {"error": f"Unknown tool: {tool_name}. Available tools: {', '.join(AVAILABLE_TOOLS)}"}
    
    try:
        # Execute the tool
        if tool_name in TOOL_FUNCTIONS:
            return TOOL_FUNCTIONS[tool_name](parameters)
        else:
            return {"error": f"Tool implementation not found for: {tool_name}"}
    except Exception as e:
        logger.error(f"Error executing tool {tool_name}: {str(e)}")
        return {"error": f"Error executing tool {tool_name}: {str(e)}"}

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tool_name", nargs="?", help="Name of the tool to execute")
    parser.add_argument("parameters", nargs="?", help="Tool parameters as JSON string")
    parser.add_argument("--list", action="store_true", help="List available tools")
    
    args = parser.parse_args()
    
    # Handle list command
    if args.list:
        print("\nAvailable MCP Tools:")
        for tool_name in AVAILABLE_TOOLS:
            print(f"- {tool_name}")
        sys.exit(0)
    
    # Check for required arguments
    if not args.tool_name:
        parser.print_help()
        sys.exit(1)
    
    # Parse parameters
    parameters = {}
    if args.parameters:
        try:
            parameters = json.loads(args.parameters)
        except json.JSONDecodeError:
            # If not valid JSON, treat as a single string parameter
            parameters = {"input": args.parameters}
    
    return args.tool_name, parameters

def main():
    """Main entry point"""
    tool_name, parameters = parse_arguments()
    
    # Execute the tool
    print("Executing tool locally")
    result = execute_tool(tool_name, parameters)
    
    # Output the result as JSON
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
